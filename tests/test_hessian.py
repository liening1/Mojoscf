"""Analytical Hessians (mojoscf.hessian) against pyscf.hessian."""
import numpy as np
import pytest
from pyscf import dft, gto
from pyscf.hessian import rks as rks_hess
from pyscf.hessian import uks as uks_hess

import mojoscf
from mojoscf import dft as mdft
from mojoscf import hessian as mhess

WATER = "O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587"


def _pyscf_xc_partial(mf):
    """The XC part of pyscf's partial_hess_elec: _get_vxc_diag and _get_vxc_deriv2 contracted with D."""
    mol = mf.mol
    h = mf.Hessian()
    mo, occ = mf.mo_coeff, mf.mo_occ
    unrestricted = np.asarray(mo).ndim == 3
    if unrestricted:
        dms = mf.make_rdm1()
        diag = uks_hess._get_vxc_diag(h, mo, occ, 4000)
        vxc = uks_hess._get_vxc_deriv2(h, mo, occ, 4000)
    else:
        dms = [mf.make_rdm1()]
        diag = [rks_hess._get_vxc_diag(h, mo, occ, 4000)]
        vxc = [rks_hess._get_vxc_deriv2(h, mo, occ, 4000)]
    de2 = np.zeros((mol.natm, mol.natm, 3, 3))
    sl = mol.aoslice_by_atom()
    for dm, vd, vx in zip(dms, diag, vxc):
        for i in range(mol.natm):
            p0, p1 = sl[i][2:]
            de2[i, i] += np.einsum("xypq,pq->xy", vd[:, :, p0:p1], dm[p0:p1]) * 2
            for j in range(i + 1):
                q0, q1 = sl[j][2:]
                de2[i, j] += np.einsum("xypq,pq->xy", vx[i][:, :, q0:q1], dm[q0:q1]) * 2
    for i in range(mol.natm):
        for j in range(i):
            de2[j, i] = de2[i, j].T
    return de2


@pytest.mark.parametrize("xc, spin", [("lda,vwn", 0), ("pbe", 0), ("b3lyp", 0), ("pbe", 1), ("b3lyp", 1)])
def test_xc_partial_hessian_matches_pyscf(xc, spin):
    mol = gto.M(atom=WATER, basis="def2-svp", charge=spin, spin=spin, verbose=0)
    mf = (dft.UKS if spin else dft.RKS)(mol, xc=xc).run()
    ref = _pyscf_xc_partial(mf)
    de2 = mhess.xc_partial_hess(mdft.NumInt(), mol, mf.grids, xc, mf.mo_coeff, mf.mo_occ)
    assert abs(de2 - ref).max() < 1e-10


def test_xc_partial_hessian_rejects_meta_gga(water=None):
    mol = gto.M(atom=WATER, basis="sto-3g", verbose=0)
    mf = dft.RKS(mol, xc="tpss").run()
    assert mhess.xc_partial_hess(mdft.NumInt(), mol, mf.grids, "tpss", mf.mo_coeff, mf.mo_occ) is None


@pytest.mark.parametrize("xc, spin, df", [("b3lyp", 0, True), ("pbe", 0, False), ("camb3lyp", 0, True),
                                          ("pbe", 1, True), ("b3lyp", 1, False), ("tpss", 0, True),
                                          ("b3lyp", 0, False), ("camb3lyp", 0, False), ("wb97x", 1, False)])
def test_accelerated_hessian_matches_pyscf(xc, spin, df):
    mol = gto.M(atom=WATER, basis="def2-svp", charge=spin, spin=spin, verbose=0)

    def make():
        mf = (dft.UKS if spin else dft.RKS)(mol, xc=xc)
        return mf.density_fit() if df else mf

    mf = mojoscf.dft.accelerate(make())
    mf.conv_tol = 1e-11
    mf.kernel()
    ref = make()
    for key in ("mo_coeff", "mo_occ", "mo_energy", "e_tot", "converged"):
        setattr(ref, key, getattr(mf, key))
    h = mf.Hessian()
    assert isinstance(h, mhess._MojoHessMixin)
    # the J/K terms run natively (range-separated functionals included), DF or exact integrals
    native = mhess._df_jk_reason(h, mf.mo_coeff, mf.mo_occ) is None
    assert native == df
    assert (mhess._exact_jk_reason(h, mf.mo_coeff, mf.mo_occ) is None) == (not df)
    # the long-range metric of erf(omega r) / r is numerically singular (condition ~1e16 here): both
    # Cholesky solves carry noise at the 1e-9 level
    tol = 5e-9 if xc == "camb3lyp" else 1e-9
    assert abs(h.kernel() - ref.Hessian().kernel()).max() < tol


@pytest.mark.parametrize("df", [True, False])
@pytest.mark.parametrize("xc, spin", [("pbe", 0), ("b3lyp", 0), ("camb3lyp", 0), ("tpss", 0), (None, 0),
                                      ("b3lyp", 1), ("wb97x", 1), (None, 1)])
def test_cphf_operator_matches_pyscf(xc, spin, df):
    """The MO-basis coupled-perturbed operator against pyscf's Hessian gen_vind on random first-order orbitals
    (density fitting: MO-basis DF tensors; exact integrals: J/K of the first-order densities)."""
    from pyscf import scf
    from pyscf.hessian import rhf as rhf_hess
    from pyscf.hessian import uhf as uhf_hess

    mol = gto.M(atom=WATER, basis="def2-svp", charge=spin, spin=spin, verbose=0)
    if xc is None:
        mf = (scf.UHF if spin else scf.RHF)(mol)
    else:
        mf = mojoscf.dft.accelerate((dft.UKS if spin else dft.RKS)(mol, xc=xc))
    mf = (mf.density_fit() if df else mf).run()
    fx = mhess.cphf_operator(mf)
    assert fx is not None
    ref = (uhf_hess if spin else rhf_hess).gen_vind(mf, mf.mo_coeff, mf.mo_occ)
    cs = mf.mo_coeff if spin else [mf.mo_coeff]
    os = mf.mo_occ if spin else [mf.mo_occ]
    n = sum(c.shape[1] * int((o > 0).sum()) for c, o in zip(cs, os))
    x = np.random.default_rng(3).normal(size=(5, n))
    a, b = fx(x), ref(x)
    assert a.shape == np.asarray(b).shape
    assert abs(a - b).max() < 1e-12 * abs(b).max()


def test_cphf_operator_falls_back_with_own_get_jk():
    mol = gto.M(atom=WATER, basis="sto-3g", verbose=0)
    mf = mojoscf.dft.accelerate(dft.RKS(mol, xc="pbe")).run()
    assert mhess.cphf_operator(mf) is not None
    mf.get_jk = lambda *args, **kwargs: dft.rks.RKS.get_jk(mf, *args, **kwargs)
    assert mhess.cphf_operator(mf) is None


@pytest.mark.parametrize("xc, spin", [("lda,vwn", 0), ("pbe", 0), ("b3lyp", 1)])
def test_xc_h1mo_matches_pyscf(xc, spin):
    """The projected XC first-derivative Fock matrices against pyscf's _get_vxc_deriv1."""
    mol = gto.M(atom=WATER, basis="def2-svp", charge=spin, spin=spin, verbose=0)
    mf = (dft.UKS if spin else dft.RKS)(mol, xc=xc).run()
    h = mf.Hessian()
    if spin:
        ref = uks_hess._get_vxc_deriv1(h, mf.mo_coeff, mf.mo_occ, 4000)
        cs, os = mf.mo_coeff, mf.mo_occ
    else:
        ref = [rks_hess._get_vxc_deriv1(h, mf.mo_coeff, mf.mo_occ, 4000)]
        cs, os = [mf.mo_coeff], [mf.mo_occ]
    out = mhess.xc_h1mo(mdft.NumInt(), mol, mf.grids, xc, mf.mo_coeff, mf.mo_occ)
    for r, o, c, occ in zip(ref, out, cs, os):
        proj = np.einsum("pm,axpq,qi->axmi", c, r, c[:, occ > 0])
        assert abs(o - proj).max() < 1e-10


def _check_df_jk(mf, tol=1e-9):
    """df_jk_terms against pyscf's DF _partial_hess_ejk (ej - hyb ek) and _gen_jk (vj1 - hyb vk1, projected)."""
    from pyscf.df.hessian import rhf as df_rhf_hess
    from pyscf.df.hessian import uhf as df_uhf_hess

    unrestricted = np.asarray(mf.mo_coeff).ndim == 3
    mod = df_uhf_hess if unrestricted else df_rhf_hess
    h = mf.Hessian()
    omega = alpha = 0.0
    if hasattr(mf, "xc"):
        ni = mf._numint
        hybrid = ni.libxc.is_hybrid_xc(mf.xc)
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mf.mol.spin)
        hyb = hyb if hybrid else 0.0
    else:
        hybrid, hyb = True, 1.0
    e1, ej, ek = mod._partial_hess_ejk(h, mf.mo_energy, mf.mo_coeff, mf.mo_occ, None, 4000, None, hybrid)
    ref = ej - hyb * ek if hybrid else ej
    if hybrid and omega:
        with mf.with_df.range_coulomb(omega):
            ek_lr = mod._partial_hess_ejk(h, mf.mo_energy, mf.mo_coeff, mf.mo_occ, None, 4000, None, True)[2]
        ref = ref - (alpha - hyb) * ek_lr
    de2, h1 = mhess.df_jk_terms(h)
    assert abs(de2 - ref).max() < tol * abs(ref).max()
    assert abs(mhess._hess_e1(h, mf.mo_energy, mf.mo_coeff, mf.mo_occ) - e1).max() < 1e-10
    cs = list(mf.mo_coeff) if unrestricted else [mf.mo_coeff]
    os = list(mf.mo_occ) if unrestricted else [mf.mo_occ]
    fs = {}
    for ia, _, vj1, vk1 in mod._gen_jk(h, mf.mo_coeff, mf.mo_occ, None, None, None, hybrid):
        fs[ia] = [vj1 - ((hyb * vk1[s] if unrestricted else 0.5 * hyb * vk1) if hybrid else 0.0)
                  for s in range(len(cs))]
    if hybrid and omega:
        with mf.with_df.range_coulomb(omega):
            for ia, _, vj1, vk1 in mod._gen_jk(h, mf.mo_coeff, mf.mo_occ, None, None, None, True):
                for s in range(len(cs)):
                    fs[ia][s] = fs[ia][s] - (alpha - hyb) * (vk1[s] if unrestricted else 0.5 * vk1)
    for ia, f in fs.items():
        for s, (c, o) in enumerate(zip(cs, os)):
            proj = np.einsum("pm,xpq,qi->xmi", c, f[s], c[:, o > 0])
            assert abs(h1[s][ia] - proj).max() < tol * max(1.0, abs(proj).max())


@pytest.mark.parametrize("xc, spin", [("b3lyp", 0), ("pbe", 0), ("pbe0", 1), (None, 0), (None, 1),
                                      ("camb3lyp", 0), ("wb97x", 1), ("hse06", 0)])
def test_df_jk_terms_match_pyscf(xc, spin):
    from pyscf import scf

    mol = gto.M(atom=WATER, basis="def2-svp", charge=spin, spin=spin, verbose=0)
    if xc is None:
        mf = (scf.UHF if spin else scf.RHF)(mol).density_fit()
    else:
        mf = mojoscf.dft.accelerate((dft.UKS if spin else dft.RKS)(mol, xc=xc).density_fit())
    mf.conv_tol = 1e-10
    # the long-range metric is less well conditioned: agreement at that level
    _check_df_jk(mf.run(), tol=1e-8 if xc in ("camb3lyp", "wb97x", "hse06") else 1e-9)


def test_df_jk_terms_open_shell_metal():
    """Cu(II) (d functions, f/g auxiliary functions), UKS hybrid."""
    mol = gto.M(atom="Cu 0 0 0; F 1.75 0 0; F -1.75 0 0", basis="def2-svp", spin=1, verbose=0)
    mf = mojoscf.dft.accelerate(dft.UKS(mol, xc="pbe0").density_fit())
    mf.conv_tol = 1e-9
    # the metric of the auxiliary basis is less well conditioned: agreement at that level
    _check_df_jk(mf.run(), tol=1e-8)


def test_int3c2e_ip1_matches_libcint():
    from pyscf import df

    from mojoscf import integrals

    mol = gto.M(atom="Fe 0 0 0; O 1.6 0 0; H 2.2 0.7 0; H -1 1 0", basis="def2-tzvp", verbose=0)
    auxmol = df.addons.make_auxmol(mol, "def2-universal-jkfit")
    nao, naux = mol.nao, auxmol.nao
    ref = df.incore.aux_e2(mol, auxmol, "int3c2e_ip1", aosym="s1", comp=3).reshape(3, nao, nao, naux)
    ps0, ps1 = 4, auxmol.nbas - 3
    p0, p1 = auxmol.ao_loc[ps0], auxmol.ao_loc[ps1]
    ext = mojoscf._backend.get_extension()
    args = (integrals.basis_tables(mol), integrals.basis_tables(auxmol), integrals._boys_table(), ps0, ps1, 1e-16)
    out = np.zeros((3, p1 - p0, nao, nao))
    ext.int3c2e_ip1(*args, out, 0.0)
    assert abs(out - ref.transpose(0, 3, 1, 2)[:, p0:p1]).max() < 1e-11
    # long-range operator erf(omega r) / r
    with mol.with_range_coulomb(0.3), auxmol.with_range_coulomb(0.3):
        ref = df.incore.aux_e2(mol, auxmol, "int3c2e_ip1", aosym="s1", comp=3).reshape(3, nao, nao, naux)
    out[:] = 0.0
    ext.int3c2e_ip1(*args, out, 0.3)
    assert abs(out - ref.transpose(0, 3, 1, 2)[:, p0:p1]).max() < 1e-11


def test_df_jk_terms_fallbacks():
    mol = gto.M(atom=WATER, basis="sto-3g", verbose=0)
    # auxbasis_response < 2, exact integrals: pyscf's terms
    mf = mojoscf.dft.accelerate(dft.RKS(mol, xc="b3lyp").density_fit()).run()
    h = mf.Hessian()
    assert mhess.df_jk_terms(h) is not None
    h.auxbasis_response = 1
    assert mhess.df_jk_terms(h) is None
    mf = mojoscf.dft.accelerate(dft.RKS(mol, xc="b3lyp")).run()
    assert mhess.df_jk_terms(mf.Hessian()) is None


def test_solvent_hessian_keeps_pyscf_solvent_terms():
    """PCM: pyscf's solvent Hessian wraps the accelerated one; its solvent terms stay in (J/K via pyscf)."""
    mol = gto.M(atom=WATER, basis="sto-3g", verbose=0)

    def make():
        return dft.RKS(mol, xc="b3lyp").density_fit().PCM()

    mf = mojoscf.dft.accelerate(make())
    mf.conv_tol = 1e-11
    mf.kernel()
    ref = make()
    ref.conv_tol = 1e-11
    ref.kernel()
    h = mf.Hessian()
    assert isinstance(h, mhess._MojoHessMixin)
    assert mhess._df_jk_reason(h, mf.mo_coeff, mf.mo_occ) == "solvent models"
    assert abs(h.kernel() - ref.Hessian().kernel()).max() < 1e-7


def test_ecp_metal_hessian_matches_pyscf():
    """A Pd complex with def2 effective core potentials: the core-potential second derivatives are pyscf's
    one-electron terms, the two-electron and XC terms run natively."""
    mol = gto.M(atom="Pd 0 0 0; Cl 2.31 0 0; Cl -2.31 0 0", basis="def2-svp", ecp={"Pd": "def2-svp"}, verbose=0)
    assert mol.has_ecp()

    def make():
        return dft.RKS(mol, xc="pbe").density_fit()

    mf = mojoscf.dft.accelerate(make())
    mf.conv_tol = 1e-10
    mf.kernel()
    ref = make()
    for key in ("mo_coeff", "mo_occ", "mo_energy", "e_tot", "converged"):
        setattr(ref, key, getattr(mf, key))
    h = mf.Hessian()
    assert mhess._df_jk_reason(h, mf.mo_coeff, mf.mo_occ) is None
    assert abs(h.kernel() - ref.Hessian().kernel()).max() < 1e-8


@pytest.mark.parametrize("spin, omega", [(0, 0.0), (1, 0.0), (0, 0.35)])
def test_exact_two_electron_hessian_term(spin, omega):
    """hess2e: the J - K part of pyscf's _partial_hess_ejk (exact integrals), also long-range."""
    from pyscf import scf
    from pyscf.hessian import rhf as rhf_hess
    from pyscf.hessian import uhf as uhf_hess

    from mojoscf import integrals as mi

    mol = gto.M(atom=WATER, basis="def2-svp", charge=spin, spin=spin, verbose=0)
    mf = (scf.UHF if spin else scf.RHF)(mol).run(conv_tol=1e-10)
    h = mf.Hessian()
    if omega:
        with mol.with_range_coulomb(omega):
            _, ej, ek = rhf_hess._partial_hess_ejk(h, mf.mo_energy, mf.mo_coeff, mf.mo_occ)
    else:
        mod = uhf_hess if spin else rhf_hess
        _, ej, ek = mod._partial_hess_ejk(h, mf.mo_energy, mf.mo_coeff, mf.mo_occ)
    dm = mf.make_rdm1()
    if spin:
        de2 = mi.hess2e(mol, dm[0] + dm[1], dm, 1.0, 1.0)
    else:
        de2 = mi.hess2e(mol, dm, dm, 1.0, 0.5, omega=omega)
    assert abs(de2 - (ej - ek)).max() < 1e-9
    # translational invariance: every row of atom blocks sums to zero
    assert abs(de2.sum(axis=1)).max() < 1e-9


@pytest.mark.parametrize("xc, spin", [("b3lyp", 0), ("pbe", 0), ("camb3lyp", 0), ("pbe0", 1)])
def test_exact_partial_hessian_matches_pyscf(xc, spin):
    mol = gto.M(atom=WATER, basis="def2-svp", charge=spin, spin=spin, verbose=0)
    ref = (dft.UKS if spin else dft.RKS)(mol, xc=xc).run(conv_tol=1e-10)
    mf = mdft.accelerate((dft.UKS if spin else dft.RKS)(mol, xc=xc))
    for k in ("mo_coeff", "mo_occ", "mo_energy", "e_tot", "converged"):
        setattr(mf, k, getattr(ref, k))
    h = mf.Hessian()
    assert mhess.exact_jk_partial(h, mf.mo_coeff, mf.mo_occ) is not None
    de2 = h.partial_hess_elec()
    ref2 = ref.Hessian().partial_hess_elec()
    assert abs(de2 - ref2).max() < 1e-8


@pytest.mark.parametrize("spin, omega", [(0, 0.0), (1, 0.0), (0, 0.35)])
def test_exact_first_order_fock_jk(spin, omega):
    """h1_jk: the J - K part of pyscf's make_h1 (exact integrals) for every atom, also long-range."""
    from pyscf import scf
    from pyscf.hessian import rhf as rhf_hess
    from pyscf.hessian import uhf as uhf_hess

    from mojoscf import integrals as mi

    mol = gto.M(atom=WATER, basis="def2-svp", charge=spin, spin=spin, verbose=0)
    mf = (scf.UHF if spin else scf.RHF)(mol).run(conv_tol=1e-10)
    h = mf.Hessian()
    hcore = mf.nuc_grad_method().hcore_generator(mol)
    if omega:
        with mol.with_range_coulomb(omega):
            ref = rhf_hess.make_h1(h, mf.mo_coeff, mf.mo_occ)
    else:
        ref = (uhf_hess if spin else rhf_hess).make_h1(h, mf.mo_coeff, mf.mo_occ)
    dm = mf.make_rdm1()
    if spin:
        vj, vk = mi.h1_jk(mol, [dm[0] + dm[1]], dm)
        for s in range(2):
            for ia in range(mol.natm):
                assert abs(vj[ia, :, 0] - vk[ia, :, s] - (ref[s][ia] - hcore(ia))).max() < 1e-10
    else:
        vj, vk = mi.h1_jk(mol, [dm], [dm], omega=omega)
        for ia in range(mol.natm):
            assert abs(vj[ia, :, 0] - 0.5 * vk[ia, :, 0] - (ref[ia] - hcore(ia))).max() < 1e-10
