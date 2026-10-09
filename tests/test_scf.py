"""Full SCF runs compared with pyscf."""
import numpy as np
import pytest
from pyscf import gto, scf

import mojoscf


def _ref(mol, **opts):
    mf = scf.RHF(mol)
    for k, v in opts.items():
        setattr(mf, k, v)
    mf.kernel()
    return mf


def _mojo(mol, **opts):
    mf = mojoscf.RHF(mol)
    for k, v in opts.items():
        setattr(mf, k, v)
    mf.kernel()
    return mf


def _compare(mf, ref, etol=1e-9):
    assert mf.converged == ref.converged
    assert abs(mf.e_tot - ref.e_tot) < etol
    assert mf.cycles == ref.cycles
    assert np.allclose(mf.mo_energy, ref.mo_energy, atol=1e-7)
    assert np.allclose(mf.mo_occ, ref.mo_occ)
    assert np.allclose(mf.make_rdm1(), ref.make_rdm1(), atol=1e-6)


def test_h2o_sto3g(h2o):
    _compare(_mojo(h2o), _ref(h2o))


def test_h2o_ccpvdz(h2o_dz):
    mf = _mojo(h2o_dz)
    ref = _ref(h2o_dz)
    _compare(mf, ref)
    # The pyscf bookkeeping is reproduced too.
    for key in ("e1", "e2", "nuc"):
        assert abs(mf.scf_summary[key] - ref.scf_summary[key]) < 1e-8
    assert abs(mf.scf_summary["gap"] - ref.scf_summary["gap"]) < 1e-6


def test_options_level_shift_damp_diis_start(h2o_dz):
    opts = dict(level_shift=0.3, damp=0.4, diis_start_cycle=3, diis_space=5, conv_tol=1e-11)
    _compare(_mojo(h2o_dz, **opts), _ref(h2o_dz, **opts), etol=1e-10)


def test_diis_damp(h2o_dz):
    opts = dict(diis_damp=0.3)
    _compare(_mojo(h2o_dz, **opts), _ref(h2o_dz, **opts))


def test_no_diis(h2o):
    opts = dict(diis=False, max_cycle=200)
    _compare(_mojo(h2o, **opts), _ref(h2o, **opts))


def test_no_conv_check(h2o_dz):
    opts = dict(conv_check=False)
    _compare(_mojo(h2o_dz, **opts), _ref(h2o_dz, **opts))


def test_init_guess_1e_and_dm0(h2o):
    _compare(_mojo(h2o, init_guess="1e"), _ref(h2o, init_guess="1e"))
    dm0 = np.eye(h2o.nao_nr()) * 0.5
    mf = mojoscf.RHF(h2o)
    mf.kernel(dm0=dm0)
    ref = scf.RHF(h2o)
    ref.kernel(dm0=dm0)
    _compare(mf, ref)


def test_max_cycle_zero(h2o):
    mf = mojoscf.RHF(h2o)
    mf.max_cycle = 0
    mf.kernel()
    ref = scf.RHF(h2o)
    ref.max_cycle = 0
    ref.kernel()
    assert abs(mf.e_tot - ref.e_tot) < 1e-10
    assert np.allclose(mf.mo_energy, ref.mo_energy, atol=1e-8)


def test_not_converged(h2o_dz):
    opts = dict(max_cycle=3)
    mf = _mojo(h2o_dz, **opts)
    ref = _ref(h2o_dz, **opts)
    assert not mf.converged and not ref.converged
    assert abs(mf.e_tot - ref.e_tot) < 1e-9
    assert mf.cycles == ref.cycles == 3


def test_callback(h2o):
    seen = []

    def cb(env):
        seen.append((env["cycle"], env["e_tot"], env["mf"]))
        assert env["dm"].shape == (h2o.nao_nr(),) * 2

    mf = mojoscf.RHF(h2o)
    mf.callback = cb
    mf.kernel()
    assert len(seen) == mf.cycles
    assert seen[0][0] == 0 and seen[-1][2] is mf


def test_restart_from_orbitals(h2o_dz):
    mf = _mojo(h2o_dz)
    e0 = mf.e_tot
    mf.kernel()  # restarts from converged orbitals
    assert mf.converged
    assert abs(mf.e_tot - e0) < 1e-10
    assert mf.cycles <= 2


def test_accelerate_density_fitting(h2o_dz):
    ref = scf.RHF(h2o_dz).density_fit()
    ref.kernel()
    mf = scf.RHF(h2o_dz).density_fit()
    mojoscf.accelerate(mf)
    assert isinstance(mf, mojoscf.scf._MojoRHFMixin)
    assert type(mf).__name__ == "MojoDFRHF"
    mf.kernel()
    _compare(mf, ref)


def test_accelerate_dispatch(h2o):
    """Native loop where it applies; pyscf's loop with hooks for other Hartree-Fock objects; Kohn-Sham objects
    through mojoscf.dft.accelerate; other classes rejected."""
    from pyscf import dft

    import mojoscf.dft as mdft

    assert isinstance(mojoscf.accelerate(scf.RHF(h2o)), mojoscf.scf._MojoRHFMixin)
    for mf in (scf.ROHF(h2o), scf.hf.RHF(h2o).newton(), scf.addons.smearing_(scf.hf.RHF(h2o), sigma=0.01)):
        assert not mojoscf.is_supported(mf)
        mf = mojoscf.accelerate(mf)
        assert isinstance(mf, mojoscf.scf._MojoHFHook) and mojoscf.accelerate(mf) is mf
    assert isinstance(mojoscf.accelerate(dft.RKS(h2o)), mdft._MojoKSHook)
    with pytest.raises(TypeError):
        mojoscf.accelerate(scf.GHF(h2o))
    assert not mojoscf.is_supported(scf.GHF(h2o))
    assert mojoscf.is_supported(scf.RHF(h2o)) and mojoscf.is_supported(scf.UHF(h2o))


def test_fallback_to_pyscf_loop(h2o):
    # A non-CDIIS scheme is not implemented natively: pyscf's loop is used
    # (with the Mojo glue) and must still give the right answer.
    from pyscf.scf import diis as pyscf_diis

    mf = mojoscf.RHF(h2o)
    mf.DIIS = pyscf_diis.EDIIS
    mf.kernel()
    ref = scf.RHF(h2o)
    ref.DIIS = pyscf_diis.EDIIS
    ref.kernel()
    assert abs(mf.e_tot - ref.e_tot) < 1e-8


def test_glue_methods_match_pyscf(h2o_dz):
    mf = _mojo(h2o_dz)
    ref = scf.RHF(h2o_dz)
    ref.__dict__.update({k: v for k, v in mf.__dict__.items() if k in ("mo_coeff", "mo_occ", "mo_energy")})
    dm = mf.make_rdm1()
    assert hasattr(dm, "mo_coeff")
    assert np.allclose(dm, ref.make_rdm1(), atol=1e-12)
    h1e = mf.get_hcore()
    vhf = mf.get_veff(h2o_dz, dm)
    assert np.allclose(mf.energy_elec(dm, h1e, vhf), ref.energy_elec(dm, h1e, vhf), atol=1e-10)
    fock = h1e + vhf
    assert np.allclose(mf.get_grad(mf.mo_coeff, mf.mo_occ, fock), ref.get_grad(ref.mo_coeff, ref.mo_occ, fock), atol=1e-10)
    assert np.array_equal(mf.get_occ(mf.mo_energy), ref.get_occ(ref.mo_energy))
    e, c = mf.eig(fock, mf.get_ovlp())
    e_ref, c_ref = ref.eig(fock, ref.get_ovlp())
    assert np.allclose(e, e_ref, atol=1e-9)
    # In a symmetric molecule two coefficients of an orbital can have equal
    # magnitude to 1e-16, so the "largest component positive" phase convention may
    # legitimately flip with rounding noise: compare up to a sign per orbital.
    sign = np.sign(np.einsum("ij,ij->j", c, c_ref))
    assert np.allclose(c * sign, c_ref, atol=1e-6)


def test_native_fallback_full_scf(h2o):
    saved = mojoscf.blas_config()
    mojoscf.use_native()
    try:
        mf = _mojo(h2o)
    finally:
        mojoscf._backend._blas = saved
    _compare(mf, _ref(h2o))


def test_backend_info():
    info = mojoscf.backend_info()
    assert info["kernels_version"] == mojoscf.__version__
    assert info["parallelism_level"] >= 1
    assert info["simd_width_f64"] >= 1


WATER = "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692"


@pytest.mark.parametrize("mode", ["incore", "direct", "df"])
@pytest.mark.parametrize("unrestricted", [False, True])
def test_get_jk_outside_the_loop(mode, unrestricted, monkeypatch):
    """get_jk/get_j/get_k of the Mojo classes and accelerated objects (get_fock, response equations, CASSCF
    call them after the SCF) come from the Mojo kernels and equal pyscf's."""
    import mojoscf.dft as mdft

    mol = gto.M(atom=WATER, basis="def2-svp", charge=int(unrestricted), spin=int(unrestricted), verbose=0)
    pyscf_cls = scf.UHF if unrestricted else scf.RHF

    def make(mf):
        if mode == "direct":
            mf.max_memory = 0
        return mf.density_fit() if mode == "df" else mf

    ref = make(pyscf_cls(mol))
    dm = ref.get_init_guess()
    rng = np.random.default_rng(7)
    x = rng.normal(size=(3,) + dm.shape) * 0.1
    x = x + x.swapaxes(-1, -2)
    routed = []
    orig = mdft.hooked_jk
    monkeypatch.setattr(mdft, "hooked_jk", lambda *a: routed.append(orig(*a)) or routed[-1])
    for mf in (make((mojoscf.UHF if unrestricted else mojoscf.RHF)(mol)), mojoscf.accelerate(make(pyscf_cls(mol)))):
        for d in (dm, x):
            del routed[:]
            j1, k1 = mf.get_jk(mol, d)
            assert routed and routed[-1] is not None
            j0, k0 = ref.get_jk(mol, d)
            assert abs(j1 - j0).max() < 1e-10 and abs(k1 - k0).max() < 1e-10
            assert abs(mf.get_j(mol, d) - j0).max() < 1e-10 and abs(mf.get_k(mol, d) - k0).max() < 1e-10
        omega = 0.4
        assert abs(mf.get_k(mol, dm, omega=omega) - ref.get_k(mol, dm, omega=omega)).max() < 1e-10
        if mode != "df":     # non-symmetric densities (pyscf's DF falls back for those)
            y = rng.normal(size=dm.shape)
            j0, k0 = ref.get_jk(mol, y, hermi=0)
            j1, k1 = mf.get_jk(mol, y, hermi=0)
            assert abs(j1 - j0).max() < 1e-10 and abs(k1 - k0).max() < 1e-10
        f0 = ref.get_fock(dm=dm)
        assert abs(mf.get_fock(dm=dm) - f0).max() < 1e-10


def test_own_get_jk_is_kept(h2o_dz):
    """A subclass with its own get_jk keeps it, inside the SCF (pyscf's get_veff route) and outside."""

    class CountingRHF(scf.hf.RHF):
        calls = 0

        def get_jk(self, *args, **kwargs):
            CountingRHF.calls += 1
            return super().get_jk(*args, **kwargs)

    mf = mojoscf.accelerate(CountingRHF(h2o_dz))
    mf.kernel()
    assert mf.scf_summary["mojoscf_veff_mode"] == 0 and CountingRHF.calls > 0
    assert abs(mf.e_tot - scf.RHF(h2o_dz).run().e_tot) < 1e-9
    n = CountingRHF.calls
    mf.get_jk(h2o_dz, mf.make_rdm1())
    assert CountingRHF.calls == n + 1
    mf = mojoscf.RHF(h2o_dz)
    mf.kernel()
    assert mf.scf_summary["mojoscf_veff_mode"] == 2


@pytest.mark.parametrize("mode", ["incore", "direct", "df"])
def test_accelerate_rohf_hook_mode(mode, monkeypatch):
    """ROHF keeps pyscf's loop with the Mojo J/K, eigensolver and DIIS: same iterations and results."""
    import mojoscf.dft as mdft

    mol = gto.M(atom=WATER, basis="def2-svp", charge=1, spin=1, verbose=0)

    def make():
        mf = scf.rohf.ROHF(mol)
        if mode == "direct":
            mf.max_memory = 0
        return mf.density_fit() if mode == "df" else mf

    ref = make().run(conv_tol=1e-10)
    routed = []
    orig = mdft.hooked_jk
    monkeypatch.setattr(mdft, "hooked_jk", lambda *a: routed.append(orig(*a)) or routed[-1])
    mf = mojoscf.accelerate(make())
    assert isinstance(mf, mojoscf.scf._MojoHFHook) and mf.DIIS is mojoscf.CDIIS
    mf.run(conv_tol=1e-10)
    assert routed and all(r is not None for r in routed)
    assert abs(mf.e_tot - ref.e_tot) < 1e-9 and mf.cycles == ref.cycles
    assert abs(mf.mo_energy - ref.mo_energy).max() < 1e-7
    assert abs(mf.nuc_grad_method().kernel() - ref.nuc_grad_method().kernel()).max() < 1e-7


@pytest.mark.parametrize("spin", [0, 2])
def test_accelerate_symmetry_hook_mode(spin):
    """Point-group symmetry: pyscf's symmetry-adapted loop (irreps, occupations, its CDIIS) with the Mojo J/K
    and eigensolver; the Mojo gradients."""
    mol = gto.M(atom=WATER, basis="def2-svp", spin=spin, symmetry=True, verbose=0)
    make = scf.RHF if spin == 0 else scf.UHF
    ref = make(mol).run(conv_tol=1e-10)
    mf = mojoscf.accelerate(make(mol))
    assert isinstance(mf, mojoscf.scf._MojoHFHook) and mf.DIIS is not mojoscf.CDIIS
    mf.run(conv_tol=1e-10)
    assert abs(mf.e_tot - ref.e_tot) < 1e-9 and mf.cycles == ref.cycles
    assert mf.get_irrep_nelec() == ref.get_irrep_nelec()
    g = mf.nuc_grad_method()
    assert isinstance(g, (mojoscf.grad.Gradients, mojoscf.grad.UGradients))
    assert abs(g.kernel() - ref.nuc_grad_method().kernel()).max() < 1e-7


def test_density_fit_after_accelerate():
    """density_fit() of a hooked object keeps the Mojo J/K in front of pyscf's density fitting."""
    from pyscf import dft

    import mojoscf.dft as mdft

    mol = gto.M(atom=WATER, basis="def2-svp", verbose=0)
    for mf, ref in ((mojoscf.accelerate(dft.RKS(mol, xc="pbe")), dft.RKS(mol, xc="pbe").density_fit()),
                    (mojoscf.accelerate(scf.rohf.ROHF(mol)), scf.rohf.ROHF(mol).density_fit())):
        dfmf = mf.density_fit()
        assert isinstance(dfmf, mojoscf.scf._MojoDFHook)
        dm = ref.get_init_guess()
        assert mdft.hooked_jk(dfmf, mojoscf.scf._MojoDFHook, None, dm, 1, True, True, None) is not None
        j0, k0 = ref.get_jk(mol, dm)
        j1, k1 = dfmf.get_jk(mol, dm)
        assert abs(j1 - j0).max() < 1e-10 and abs(k1 - k0).max() < 1e-10
        assert abs(dfmf.run(conv_tol=1e-10).e_tot - ref.run(conv_tol=1e-10).e_tot) < 1e-9


def test_solvent_response_methods_are_kept():
    """A solvent model's own TDA (the TD wrapper with the solvent response) is not bypassed by the hooks."""
    from pyscf import dft
    from pyscf.solvent import _attach_solvent

    mol = gto.M(atom=WATER, basis="6-31g", verbose=0)
    for make in (lambda: scf.hf.RHF(mol).PCM(), lambda: dft.RKS(mol, xc="b3lyp").PCM()):
        ref = make().run(conv_tol=1e-10)
        mf = mojoscf.accelerate(make()).run(conv_tol=1e-10)
        assert abs(mf.e_tot - ref.e_tot) < 1e-9
        for eq in (False, True):
            td = mf.TDA(equilibrium_solvation=eq)
            assert isinstance(td, _attach_solvent.TDSCFWithSolvent)
            td0 = ref.TDA(equilibrium_solvation=eq)
            assert abs(td.kernel(nstates=3)[0] - td0.kernel(nstates=3)[0]).max() < 1e-6
