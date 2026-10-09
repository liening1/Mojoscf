"""UHF: kernels, open-shell and broken-symmetry SCF runs compared with pyscf."""
import numpy as np
import pytest
from pyscf import gto, lib, scf
from pyscf.scf import diis as pyscf_diis
from pyscf.scf import hf as pyscf_hf

import mojoscf
from mojoscf import kernels
from mojoscf.guess import afm_guess_by_atom, flip_spin_on_atoms, mix_homo_lumo_guess


def _ref(mol, dm0=None, **opts):
    mf = scf.UHF(mol)
    for k, v in opts.items():
        setattr(mf, k, v)
    mf.verbose = 0
    mf.kernel(dm0=dm0)
    return mf


def _mojo(mol, dm0=None, **opts):
    mf = mojoscf.UHF(mol)
    for k, v in opts.items():
        setattr(mf, k, v)
    mf.verbose = 0
    mf.kernel(dm0=dm0)
    return mf


def _compare(mf, ref, etol=1e-9, dm=True):
    """Same converged state and same SCF trajectory.

    With a partially filled *degenerate* shell (OH radical, stretched N2, ...) the
    unpaired electron may sit in either orbital of the pair depending on rounding,
    so the density matrices are only compared with ``dm=False`` there; the total
    energy, cycle count, <S^2>, orbital energies and per-spin electron numbers
    must agree regardless.
    """
    assert mf.converged == ref.converged
    assert abs(mf.e_tot - ref.e_tot) < etol
    assert mf.cycles == ref.cycles
    assert np.allclose(mf.mo_energy, ref.mo_energy, atol=1e-6)
    assert np.allclose(mf.mo_occ, ref.mo_occ)
    assert abs(mf.spin_square()[0] - ref.spin_square()[0]) < 1e-6
    s = mf.get_ovlp()
    for sp in (0, 1):
        assert np.einsum("ij,ji->", mf.make_rdm1()[sp], s) == pytest.approx(mf.mol.nelec[sp], abs=1e-8)
    if dm:
        assert np.allclose(mf.make_rdm1(), ref.make_rdm1(), atol=1e-5)


@pytest.fixture(scope="module")
def oh():
    return gto.M(atom="O 0 0 0; H 0 0 0.97", basis="cc-pvdz", spin=1, verbose=0)


@pytest.fixture(scope="module")
def nh2():
    """Doublet without degenerate orbitals: densities are unique and comparable."""
    return gto.M(atom="N 0 0 0; H 0 0.8 0.58; H 0 -0.8 0.58", basis="6-31g*", spin=1, verbose=0)


@pytest.fixture(scope="module")
def o2():
    return gto.M(atom="O 0 0 0; O 0 0 1.208", basis="cc-pvdz", spin=2, verbose=0)


# ----------------------------------------------------------------- kernels


def test_get_occ_unit_occupation():
    e = np.array([-3.0, -1.0, 0.5, -2.0, 0.2])
    occ, gap = kernels.get_occ(e, 2, occ=1.0)
    assert np.array_equal(occ, [1, 0, 0, 1, 0])
    assert gap == (-2.0, -1.0)
    occ0, gap0 = kernels.get_occ(e, 0, occ=1.0)
    assert np.all(occ0 == 0) and gap0 is None


def test_get_grad_prefactor(use_blas, rng):
    nao = 8
    c = rng.standard_normal((nao, nao))
    occ = np.array([1, 1, 0, 1, 0, 0, 0, 0.0])
    f = rng.standard_normal((nao, nao))
    f = f + f.T
    g1 = kernels.get_grad(c, occ, f, 1.0)
    g2 = kernels.get_grad(c, occ, f)  # RHF default
    assert np.allclose(g2, 2 * g1)
    # pyscf's UHF gradient of one spin channel
    ref = scf.uhf.get_grad((c, c), (occ, occ), (f, f))
    n = g1.size
    assert np.allclose(g1, ref[:n], atol=1e-12)


def test_level_shift_dm_scale(use_blas, rng):
    n = 7
    a = rng.standard_normal((n, n))
    s = a @ a.T + n * np.eye(n)
    d = rng.standard_normal((n, n))
    d = d + d.T
    f = rng.standard_normal((n, n))
    f = f + f.T
    assert np.allclose(kernels.level_shift(s, d, f, 0.4, dm_scale=1.0), pyscf_hf.level_shift(s, d, f, 0.4), atol=1e-11)
    assert np.allclose(kernels.level_shift(s, d, f, 0.4), pyscf_hf.level_shift(s, d * 0.5, f, 0.4), atol=1e-11)


@pytest.mark.parametrize("use_corth", [False, True])
def test_cdiis_stacked_uhf_fock(use_blas, rng, use_corth):
    n = 6
    a = rng.standard_normal((n, n))
    s = a @ a.T + n * np.eye(n)
    x = pyscf_hf.check_linear_dependency(s) if use_corth else None
    ref = pyscf_diis.CDIIS(Corth=x)
    mine = mojoscf.CDIIS(Corth=x)
    for step in range(10):
        f = rng.standard_normal((2, n, n))
        f = f + f.transpose(0, 2, 1)
        d = rng.standard_normal((2, n, n))
        d = d + d.transpose(0, 2, 1)
        assert np.allclose(mine.update(s, d, f), ref.update(s, d, f), atol=1e-9), step


# ------------------------------------------------------------- SCF, open shell


def test_doublet_and_triplet(oh, nh2, o2):
    _compare(_mojo(nh2), _ref(nh2))
    _compare(_mojo(o2), _ref(o2))
    _compare(_mojo(oh), _ref(oh), dm=False)  # degenerate pi shell


def test_closed_shell_with_uhf(h2o):
    # nalpha == nbeta from a symmetric start stays restricted
    mf, ref = _mojo(h2o), _ref(h2o)
    _compare(mf, ref)
    assert abs(mf.spin_square()[0]) < 1e-8


def test_empty_beta_channel():
    mol = gto.M(atom="H 0 0 0", basis="cc-pvtz", spin=1, verbose=0)
    mf = mojoscf.UHF(mol)
    mf.verbose = 0
    mf.kernel()
    ref = _ref(mol)
    assert mf.converged and abs(mf.e_tot - ref.e_tot) < 1e-10
    assert mf.mo_occ[1].sum() == 0


@pytest.mark.parametrize("charge, spin", [(1, 1), (-1, 1), (2, 0), (-2, 0)])
def test_ions(charge, spin):
    mol = gto.M(atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587", basis="6-31g", charge=charge, spin=spin, verbose=0)
    _compare(_mojo(mol), _ref(mol))


@pytest.mark.parametrize(
    "opts",
    [
        dict(level_shift=0.3),
        dict(level_shift=(0.2, 0.4)),
        dict(damp=0.3, diis_start_cycle=4),
        dict(level_shift=0.3, damp=0.4, diis_start_cycle=3, diis_space=5, conv_tol=1e-11),
        dict(diis_damp=0.25),
        dict(diis=False, max_cycle=300),
        dict(conv_check=False),
    ],
)
def test_options(nh2, opts):
    _compare(_mojo(nh2, **opts), _ref(nh2, **opts), etol=1e-9)


def test_max_cycle_zero_and_not_converged(oh):
    mf, ref = _mojo(oh, max_cycle=0), _ref(oh, max_cycle=0)
    assert abs(mf.e_tot - ref.e_tot) < 1e-10
    mf, ref = _mojo(oh, max_cycle=3), _ref(oh, max_cycle=3)
    assert not mf.converged and not ref.converged
    assert mf.cycles == ref.cycles == 3
    assert abs(mf.e_tot - ref.e_tot) < 1e-9


def test_callback_and_restart(oh):
    seen = []
    mf = mojoscf.UHF(oh)
    mf.verbose = 0
    mf.callback = lambda env: seen.append((env["cycle"], env["dm"].shape, env["mf"] is mf))
    mf.kernel()
    assert len(seen) == mf.cycles and seen[0][1] == (2, oh.nao_nr(), oh.nao_nr()) and all(x[2] for x in seen)
    e0 = mf.e_tot
    mf.callback = None
    mf.kernel()  # restart from the converged orbitals
    assert mf.converged and abs(mf.e_tot - e0) < 1e-10 and mf.cycles <= 2


def test_two_dimensional_dm0_is_split(nh2):
    dm0 = scf.hf.init_guess_by_minao(nh2)
    _compare(_mojo(nh2, dm0=dm0), _ref(nh2, dm0=dm0))


def test_density_fitting_and_accelerate(o2):
    ref = scf.UHF(o2).density_fit()
    ref.verbose = 0
    ref.kernel()
    mf = scf.UHF(o2).density_fit()
    mf.verbose = 0
    mojoscf.accelerate(mf)
    assert isinstance(mf, mojoscf.scf._MojoUHFMixin) and type(mf).__name__ == "MojoDFUHF"
    mf.kernel()
    _compare(mf, ref)


def test_fallback_to_pyscf_loop(oh):
    mf = mojoscf.UHF(oh)
    mf.verbose = 0
    mf.DIIS = pyscf_diis.EDIIS
    mf.kernel()
    ref = scf.UHF(oh)
    ref.verbose = 0
    ref.DIIS = pyscf_diis.EDIIS
    ref.kernel()
    assert mf.converged and abs(mf.e_tot - ref.e_tot) < 1e-8


def test_glue_methods_match_pyscf(oh):
    mf = _mojo(oh)
    ref = scf.UHF(oh)
    ref.__dict__.update({k: v for k, v in mf.__dict__.items() if k in ("mo_coeff", "mo_occ", "mo_energy")})
    dm = mf.make_rdm1()
    assert hasattr(dm, "mo_coeff") and dm.shape == (2, oh.nao_nr(), oh.nao_nr())
    assert np.allclose(dm, ref.make_rdm1(), atol=1e-12)
    h1e = mf.get_hcore()
    vhf = mf.get_veff(oh, dm)
    assert np.allclose(mf.energy_elec(dm, h1e, vhf), ref.energy_elec(dm, h1e, vhf), atol=1e-10)
    # a closed-shell (2D) density is split like in pyscf
    d2 = dm[0] + dm[1]
    assert np.allclose(mf.energy_elec(d2, h1e), ref.energy_elec(d2, h1e), atol=1e-10)
    fock = np.array((h1e + vhf[0], h1e + vhf[1]))
    assert np.allclose(mf.get_grad(mf.mo_coeff, mf.mo_occ, fock), ref.get_grad(ref.mo_coeff, ref.mo_occ, fock), atol=1e-10)
    e, c = mf.eig(fock, mf.get_ovlp())
    e_ref, c_ref = ref.eig(fock, ref.get_ovlp())
    assert np.allclose(e, e_ref, atol=1e-9)
    sign = np.sign(np.einsum("sij,sij->sj", c, c_ref))[:, None, :]
    assert np.allclose(c * sign, c_ref, atol=1e-6)


# --------------------------------------------------------- broken symmetry


@pytest.mark.parametrize("r, ss", [(1.0, 0.0), (2.0, 0.9), (3.0, 1.0)])
def test_h2_dissociation_mix_guess(r, ss):
    mol = gto.M(atom=f"H 0 0 0; H 0 0 {r}", basis="cc-pvdz", verbose=0)
    dm0 = mix_homo_lumo_guess(mol)
    mf, ref = _mojo(mol, dm0, conv_tol=1e-10), _ref(mol, dm0, conv_tol=1e-10)
    _compare(mf, ref, etol=1e-10)
    assert mf.spin_square()[0] == pytest.approx(ss, abs=0.1)
    rhf = scf.RHF(mol)
    rhf.verbose = 0
    rhf.kernel()
    if r > 1.5:  # a genuine broken-symmetry solution lies below the restricted one
        assert mf.e_tot < rhf.e_tot - 0.05


@pytest.mark.parametrize("n", [6, 10, 20])
def test_hydrogen_chain_antiferromagnet(n):
    mol = gto.M(atom="; ".join(f"H 0 0 {1.8 * i}" for i in range(n)), basis="6-31g", verbose=0)
    dm0 = afm_guess_by_atom(mol, set(range(0, n, 2)))
    mf, ref = _mojo(mol, dm0, conv_tol=1e-10), _ref(mol, dm0, conv_tol=1e-10)
    _compare(mf, ref, etol=1e-10)
    # alternating spin density along the chain
    dm = mf.make_rdm1()
    s = mol.intor("int1e_ovlp")
    spin = [np.einsum("ij,ji->", (dm[0] - dm[1])[p0:p1], s[:, p0:p1]) for _, p0, p1 in mojoscf.guess.atom_ao_slices(mol)]
    assert all(np.sign(spin[i]) == (-1) ** i for i in range(n))
    assert mf.spin_square()[0] > 1.0


@pytest.mark.parametrize("atom", ["N 0 0 0; N 0 0 2.2", "F 0 0 0; F 0 0 2.6"])
def test_stretched_diatomic_singlet_diradical(atom):
    mol = gto.M(atom=atom, basis="cc-pvdz", verbose=0)
    dm0 = mix_homo_lumo_guess(mol)
    mf, ref = _mojo(mol, dm0, conv_tol=1e-10), _ref(mol, dm0, conv_tol=1e-10)
    _compare(mf, ref, etol=1e-9, dm=False)  # pi shell of the dissociating bond is degenerate
    # Which of the degenerate HOMO orbitals the mix guess rotates can change with
    # rounding noise (for pyscf as well), so only require a clearly broken-symmetry,
    # spin-contaminated solution below the restricted one.
    assert mf.spin_square()[0] > 0.5


def test_flip_spin_guess_two_centre_antiferromagnet():
    # Two weakly coupled hydrogen-like centres built from a high-spin density.
    hs = gto.M(atom="Li 0 0 0; Li 0 0 4.5", basis="6-31g", spin=2, verbose=0)
    bs = gto.M(atom="Li 0 0 0; Li 0 0 4.5", basis="6-31g", spin=0, verbose=0)
    hs_mf = _ref(hs)
    dm0 = flip_spin_on_atoms(hs_mf.make_rdm1(), bs, {1})
    mf, ref = _mojo(bs, dm0, conv_tol=1e-10), _ref(bs, dm0, conv_tol=1e-10)
    _compare(mf, ref, etol=1e-10)
    assert 0.5 < mf.spin_square()[0] < 1.05  # singlet diradical-like, spin-contaminated


# ------------------------------------------------- guard and bookkeeping


def test_accelerate_rejections(h2o):
    from pyscf import dft

    import mojoscf.dft as mdft

    for obj in (scf.ROHF(h2o), scf.UHF(h2o).newton()):          # pyscf's loop with the Mojo kernels
        assert not mojoscf.is_supported(obj)
        assert isinstance(mojoscf.accelerate(obj), mojoscf.scf._MojoHFHook)
    for obj in (dft.RKS(h2o), dft.UKS(h2o)):
        assert isinstance(mojoscf.accelerate(obj), mdft._MojoKSHook)
    with pytest.raises(TypeError):
        mojoscf.accelerate(scf.GHF(h2o))
    assert mojoscf.is_supported(scf.UHF(h2o)) and mojoscf.is_supported(scf.RHF(h2o))


def test_overridden_glue_is_detected(oh):
    # Fermi smearing replaces get_occ & friends; the native loop would ignore it, so pyscf's loop runs with
    # the Mojo kernels
    smeared = scf.addons.smearing_(scf.UHF(oh), sigma=0.01, method="fermi")
    assert "get_occ" in mojoscf.scf.unsupported_reason(smeared)
    smeared = mojoscf.accelerate(smeared)
    assert isinstance(smeared, mojoscf.scf._MojoHFHook)
    smeared.verbose = 0
    ref = scf.addons.smearing_(scf.UHF(oh), sigma=0.01, method="fermi")
    ref.verbose = 0
    assert abs(smeared.kernel() - ref.kernel()) < 1e-9
    # ... and one installed *after* accelerate() makes kernel() fall back to pyscf's loop
    mf = mojoscf.accelerate(scf.UHF(oh))
    mf.verbose = 0
    calls = []
    orig = mf.get_occ
    mf.get_occ = lambda *a, **k: (calls.append(1), orig(*a, **k))[1]
    mf.kernel()
    assert calls and mf.converged


def test_veff_receives_orbital_tagged_density(h2o, oh):
    """pyscf's cheap density-fitting K build needs mo_coeff/mo_occ tags on the density."""
    for mol, cls in ((h2o, mojoscf.RHF), (oh, mojoscf.UHF)):
        mf = cls(mol).density_fit()
        mf.verbose = 0
        tags = []
        orig = mf.get_veff

        def spy(mol_, dm=None, *a, **k):
            tags.append(getattr(dm, "mo_coeff", None) is not None and getattr(dm, "mo_occ", None) is not None)
            return orig(mol_, dm, *a, **k)

        mf.get_veff = spy
        mf.kernel()
        # every call but the very first (minao guess carries no tags for UHF) is tagged
        assert all(tags[1:]) and len(tags) >= 3


def test_scf_summary_and_gap_match_pyscf(oh):
    mf, ref = _mojo(oh), _ref(oh)
    for key in ("e1", "e2", "nuc", "gap"):
        assert abs(mf.scf_summary[key] - ref.scf_summary[key]) < 1e-6, key
