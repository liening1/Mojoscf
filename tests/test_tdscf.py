"""Linear response (mojoscf.tdscf, the fxc kernels of mojoscf.dft) against pyscf.tdscf."""
import numpy as np
import pytest
from pyscf import dft, gto, lib, scf
from pyscf import tdscf as ptd
from pyscf.dft import numint

import mojoscf
from mojoscf import dft as mdft
from mojoscf import kernels
from mojoscf import tdscf as mtd

WATER = "O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587"


@pytest.fixture(scope="module")
def water():
    return gto.M(atom=WATER, basis="def2-svp", verbose=0)


@pytest.fixture(scope="module")
def water_cation():
    return gto.M(atom=WATER, basis="def2-svp", charge=1, spin=1, verbose=0)


def _scf(mol, xc, accelerate=True, df=True):
    unrestricted = mol.spin > 0
    if xc is None:
        mf = (scf.UHF if unrestricted else scf.RHF)(mol)
    else:
        mf = (dft.UKS if unrestricted else dft.RKS)(mol, xc=xc)
    if df:
        mf = mf.density_fit()
    if xc is not None and accelerate:
        mojoscf.dft.accelerate(mf)
    mf.conv_tol = 1e-10
    return mf.run()


def _pyscf_copy(mf):
    """A plain pyscf object (pyscf's NumInt, J/K and TD classes) with the reference of ``mf``."""
    mol = mf.mol
    unrestricted = mf.mo_coeff.ndim == 3
    if isinstance(mf, scf.hf.KohnShamDFT):
        ref = (dft.UKS if unrestricted else dft.RKS)(mol, xc=mf.xc)
    else:
        ref = (scf.UHF if unrestricted else scf.RHF)(mol)
    if hasattr(mf, "with_df"):
        ref = ref.density_fit()
    for key in ("mo_coeff", "mo_occ", "mo_energy", "e_tot", "converged"):
        setattr(ref, key, getattr(mf, key))
    return ref


def _check(td_m, td_p, nstates=3, op_tol=1e-11, e_tol=1e-8, mo_path=True):
    if mo_path:     # the occupied-virtual operator runs (not pyscf's, which gives the same results)
        assert mtd._operator(td_m) is not None
    vm, hm = td_m.gen_vind()
    vp, hp = td_p.gen_vind()
    assert abs(hm - hp).max() < 1e-14
    x = np.random.default_rng(7).normal(size=(4, hp.size))
    a, b = vm(x), vp(x)
    assert abs(a - b).max() < op_tol * abs(b).max()
    td_m.nstates = td_p.nstates = nstates
    em = td_m.kernel()[0]
    ep = td_p.kernel()[0]
    assert abs(em - ep).max() < e_tol
    return em


def test_kernels_df_mo_and_sandwich(water, ext):
    from pyscf import df

    cderi = df.incore.cholesky_eri(water, auxbasis="def2-universal-jkfit")
    rng = np.random.default_rng(0)
    cl = rng.normal(size=(water.nao, 5))
    cr = rng.normal(size=(water.nao, 7))
    ref = np.einsum("pi,Qpq,qa->Qia", cl, lib.unpack_tril(cderi), cr)
    assert abs(kernels.df_mo(cderi, cl, cr) - ref).max() < 1e-12
    a = rng.normal(size=(30, 4, 6))
    b = rng.normal(size=(30, 9, 8))
    x = rng.normal(size=(6, 3 * 9))
    out = np.ones((12, 8))
    kernels.df_sandwich(a, x, b, 3, -0.5, out=out)
    ref = 1.0 - 0.5 * np.einsum("qmk,knj,qjp->mnp", a, x.reshape(6, 3, 9), b).reshape(12, 8)
    assert abs(out - ref).max() < 1e-12


@pytest.mark.parametrize("xc", ["lda,vwn", "pbe", "b3lyp", "tpss", "r2scan"])
def test_fxc_kernels_match_pyscf(water, xc):
    """cache_xc_kernel, nr_rks_fxc, nr_rks_fxc_st and nr_uks_fxc of mojoscf.dft.NumInt against pyscf's."""
    grids = dft.gen_grid.Grids(water).build()
    mf = dft.RKS(water, xc="pbe").run()
    c, occ = mf.mo_coeff, mf.mo_occ
    p, m = numint.NumInt(), mdft.NumInt()
    for spin in (0, 1):
        r0, v0, f0 = p.cache_xc_kernel(water, grids, xc, c, occ, spin)
        r1, v1, f1 = m.cache_xc_kernel(water, grids, xc, c, occ, spin)
        assert abs(np.asarray(r0) - np.asarray(r1)).max() < 1e-10
        # libxc's kernels are ill-conditioned where the density vanishes: compare where it does not
        r0 = np.asarray(r0)
        dens = r0[..., 0, :] if r0.ndim > 1 and r0.shape[-2] in (4, 5) else r0
        keep = dens.reshape(-1, r0.shape[-1]).min(axis=0) > 1e-8
        f0, f1 = np.asarray(f0)[..., keep], np.asarray(f1)[..., keep]
        assert abs(f0 - f1).max() < 1e-8 * max(1.0, abs(f0).max())
    rng = np.random.default_rng(0)
    dms = rng.normal(size=(3, water.nao, water.nao)) * 0.1             # not symmetric
    fx = p.cache_xc_kernel(water, grids, xc, c, occ, 0)[2]
    a = p.nr_rks_fxc(water, grids, xc, None, dms, 0, 0, None, None, fx)
    b = m.nr_rks_fxc(water, grids, xc, None, dms, 0, 0, None, None, fx)
    assert abs(a - b).max() < 1e-12
    assert abs(m.nr_rks_fxc(water, grids, xc, None, dms[0], 0, 0, None, None, fx) - a[0]).max() < 1e-12
    fs = p.cache_xc_kernel(water, grids, xc, c, occ, 1)[2] * 0.5
    for singlet in (True, False):
        a = p.nr_rks_fxc_st(water, grids, xc, None, dms, 0, singlet, None, None, fs)
        b = m.nr_rks_fxc_st(water, grids, xc, None, dms, 0, singlet, None, None, fs)
        assert abs(a - b).max() < 1e-12
    fu = p.cache_xc_kernel(water, grids, xc, np.array([c, c]), np.array([occ / 2, occ / 2]), 1)[2]
    du = rng.normal(size=(2, 3, water.nao, water.nao)) * 0.1
    a = p.nr_uks_fxc(water, grids, xc, None, du, 0, 0, None, None, fu)
    b = m.nr_uks_fxc(water, grids, xc, None, du, 0, 0, None, None, fu)
    assert abs(a - b).max() < 1e-12


@pytest.mark.parametrize("kind", [0, 1, 2])
def test_fxc_matrices_factors_and_projection(water, kind):
    """The low-rank densities and the projected output (V R) agree with the dense route."""
    grids = dft.gen_grid.Grids(water).build()
    nvar = (1, 4, 5)[kind]
    rng = np.random.default_rng(1)
    fxc = rng.normal(size=(nvar, nvar, grids.weights.size))
    fxc = fxc + fxc.transpose(1, 0, 2)
    nao = water.nao
    for rank in (3, 20):                       # 20: the blocks keep the dense route
        lf = rng.normal(size=(4, nao, rank))
        rf = rng.normal(size=(nao, rank))
        dense = mdft.fxc_matrices(water, grids, kind, fxc, dms=(lf @ rf.T)[None])
        low = mdft.fxc_matrices(water, grids, kind, fxc, factors=[(lf, rf)])
        assert abs(dense - low).max() < 1e-10 * abs(dense).max()
        proj = mdft.fxc_matrices(water, grids, kind, fxc, factors=[(lf, rf)], project=True)
        assert abs(proj - dense @ rf).max() < 1e-10 * abs(proj).max()


CASES = [
    ("pbe0", ["TDA", "TDDFT"]),
    ("pbe", ["TDA", "TDDFT", "CasidaTDDFT"]),
    ("camb3lyp", ["TDA", "TDDFT"]),
    ("tpss", ["TDA", "CasidaTDDFT"]),
    (None, ["TDA", "TDDFT"]),
]


@pytest.mark.parametrize("xc, names", CASES)
def test_rks_operators_match_pyscf(water, xc, names):
    mf = _scf(water, xc)
    ref = _pyscf_copy(mf)
    for name in names:
        for singlet in (True, False):
            td_m = getattr(mtd, name)(mf)
            td_p = ptd.rhf.TDHF(ref) if (xc is None and name == "TDDFT") else getattr(ref, name)()
            assert type(td_m).__mro__[2] is type(td_p)
            td_m.singlet = td_p.singlet = singlet
            _check(td_m, td_p)


@pytest.mark.parametrize("xc, names", [("b3lyp", ["TDA", "TDDFT"]), ("pbe", ["TDA", "CasidaTDDFT"]),
                                       ("wb97x", ["TDA", "TDDFT"]), (None, ["TDA", "TDDFT"])])
def test_uks_operators_match_pyscf(water_cation, xc, names):
    mf = _scf(water_cation, xc)
    ref = _pyscf_copy(mf)
    for name in names:
        td_m = getattr(mtd, name)(mf)
        td_p = ptd.uhf.TDHF(ref) if (xc is None and name == "TDDFT") else getattr(ref, name)()
        _check(td_m, td_p)


def test_accelerated_objects_create_mojo_td(water, water_cation):
    mf = _scf(water, "b3lyp")
    assert isinstance(mf.TDA(), mtd._MojoTD) and isinstance(mf.TDA(), ptd.rks.TDA)
    assert isinstance(mf.TDDFT(), ptd.rks.TDDFT) and isinstance(mf.TDDFT(), mtd._MojoTD)
    pure = _scf(water, "pbe")
    assert isinstance(pure.TDDFT(), ptd.rks.CasidaTDDFT) and isinstance(pure.TDDFT(), mtd._MojoTD)
    umf = _scf(water_cation, "pbe0")
    td = umf.TDDFT()
    assert isinstance(td, ptd.uks.TDDFT) and isinstance(td, mtd._MojoTD)
    td.nstates = 3
    e = td.kernel()[0]
    ref = _pyscf_copy(umf).TDDFT()
    ref.nstates = 3
    assert abs(e - ref.kernel()[0]).max() < 1e-8
    # results and analysis stay pyscf's
    assert td.oscillator_strength().shape == (3,)


def test_frozen_orbitals(water):
    mf = _scf(water, "pbe0")
    ref = _pyscf_copy(mf)
    _check(mtd.TDA(mf, frozen=1), ref.TDA(frozen=1))
    _check(mtd.TDDFT(mf, frozen=[0, 1]), ref.TDDFT(frozen=[0, 1]))


def test_fallbacks_give_pyscf_results(water):
    """Cases the MO-space operator does not take run pyscf's operator (same results)."""
    # exact integrals: no DF tensor
    mf = _scf(water, "pbe0", df=False)
    td = mf.TDA()
    assert mtd._operator(td) is None
    _check(td, _pyscf_copy(mf).TDA(), mo_path=False)
    # short-range-only exact exchange (HSE)
    mf = _scf(water, "hse06")
    td = mf.TDA()
    assert mtd._operator(td) is None
    # point-group restricted states
    sym = gto.M(atom=WATER, basis="def2-svp", symmetry=True, verbose=0)
    mf = _scf(sym, "pbe0")
    td = mf.TDA()
    td.wfnsym = "B2"
    assert mtd._operator(td) is None
    td_p = _pyscf_copy(mf).TDA()
    td_p.wfnsym = "B2"
    _check(td, td_p, mo_path=False)


def test_metal_complex_uks():
    """An open-shell transition-metal complex with an ECP-free def2 basis (UKS, hybrid)."""
    mol = gto.M(atom="Cu 0 0 0; Cl 2.2 0 0; Cl -2.2 0 0", basis="def2-svp", charge=0, spin=1, verbose=0)
    mf = _scf(mol, "b3lyp")
    ref = _pyscf_copy(mf)
    _check(mtd.TDA(mf), ref.TDA(), nstates=4)
