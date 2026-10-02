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


def test_accelerate_rejects_unsupported(h2o):
    from pyscf import dft

    with pytest.raises(TypeError):
        mojoscf.accelerate(scf.ROHF(h2o))
    with pytest.raises(TypeError):
        mojoscf.accelerate(dft.RKS(h2o))
    with pytest.raises(TypeError):
        mojoscf.accelerate(scf.UHF(h2o))
    assert not mojoscf.is_supported(scf.UHF(h2o))
    assert mojoscf.is_supported(scf.RHF(h2o))


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
    assert np.allclose(c, c_ref, atol=1e-6)


def test_native_fallback_full_scf(h2o):
    saved = mojoscf.blas_args()
    mojoscf._backend._blas = ("", "")
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
