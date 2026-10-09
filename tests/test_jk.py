"""Two-electron builders: density-fitted and in-core J/K against pyscf."""
import numpy as np
import pytest
from pyscf import gto, lib, scf
from pyscf.df import df_jk as pyscf_df_jk
from pyscf.scf import hf as pyscf_hf

import mojoscf
from mojoscf import kernels


@pytest.fixture(scope="module")
def dfmf(h2o_dz):
    mf = scf.RHF(h2o_dz).density_fit()
    mf.verbose = 0
    mf.with_df.build()
    return mf


@pytest.fixture(scope="module")
def eri_s8(h2o_dz):
    return h2o_dz.intor("int2e", aosym="s8")


def _densities(mol, rng):
    """A tagged converged-like density, a positive-definite guess and an indefinite matrix."""
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    dm_conv = mf.make_rdm1()
    dm_guess = scf.hf.init_guess_by_minao(mol)
    a = rng.standard_normal((mol.nao_nr(), mol.nao_nr()))
    dm_rand = a + a.T
    return mf, dm_conv, dm_guess, dm_rand


def test_factorize_density(h2o_dz, rng):
    _, dm_conv, dm_guess, dm_rand = _densities(h2o_dz, rng)
    for dm in (dm_conv, dm_guess, dm_rand):
        orb, sign = kernels.factorize_density(dm)
        assert np.allclose((orb * sign) @ orb.T, dm, atol=1e-11)
    orb, sign = kernels.factorize_density(dm_conv)
    assert orb.shape[1] == h2o_dz.nelectron // 2 and np.all(sign > 0)


@pytest.mark.parametrize("which", ["tagged", "guess", "indefinite"])
def test_df_jk_single(use_blas, dfmf, h2o_dz, rng, which):
    mf, dm_conv, dm_guess, dm_rand = _densities(h2o_dz, rng)
    dm = {"tagged": dm_conv, "guess": dm_guess, "indefinite": dm_rand}[which]
    vj_ref, vk_ref = pyscf_df_jk.get_jk(dfmf.with_df, dm, hermi=1)
    if which == "tagged":
        vj, vk = kernels.df_jk(dfmf.with_df._cderi, dm, dm.mo_coeff, dm.mo_occ)
    else:
        vj, vk = kernels.df_jk(dfmf.with_df._cderi, dm)
    assert np.allclose(vj, vj_ref, atol=1e-10)
    assert np.allclose(vk, vk_ref, atol=1e-10)
    # small blocks exercise the block loop
    vj2, vk2 = kernels.df_jk(dfmf.with_df._cderi, dm, block_mb=1)
    assert np.allclose(vj2, vj_ref, atol=1e-10) and np.allclose(vk2, vk_ref, atol=1e-10)


def test_df_jk_stacked_and_partial(use_blas, dfmf, h2o_dz, rng):
    mf, dm_conv, dm_guess, _ = _densities(h2o_dz, rng)
    dms = np.array((dm_conv, 0.4 * dm_guess))
    vj_ref, vk_ref = pyscf_df_jk.get_jk(dfmf.with_df, dms, hermi=1)
    vj, vk = kernels.df_jk(dfmf.with_df._cderi, dms)
    assert vj.shape == dms.shape and np.allclose(vj, vj_ref, atol=1e-10) and np.allclose(vk, vk_ref, atol=1e-10)
    vj_only, none = kernels.df_jk(dfmf.with_df._cderi, dms, with_k=False)
    assert none is None and np.allclose(vj_only, vj_ref, atol=1e-10)
    none, vk_only = kernels.df_jk(dfmf.with_df._cderi, dms, with_j=False)
    assert none is None and np.allclose(vk_only, vk_ref, atol=1e-10)
    # UHF-style stacked orbitals
    mo_c = np.array((mf.mo_coeff, mf.mo_coeff))
    mo_o = np.array((mf.mo_occ / 2, mf.mo_occ / 2))
    dms_u = np.array((dm_conv / 2, dm_conv / 2))
    vj_u, vk_u = kernels.df_jk(dfmf.with_df._cderi, dms_u, mo_c, mo_o)
    vj_u_ref, vk_u_ref = pyscf_df_jk.get_jk(dfmf.with_df, lib.tag_array(dms_u, mo_coeff=mo_c, mo_occ=mo_o), hermi=1)
    assert np.allclose(vj_u, vj_u_ref, atol=1e-10) and np.allclose(vk_u, vk_u_ref, atol=1e-10)


def test_jk_s8_matches_pyscf(eri_s8, h2o_dz, rng):
    _, dm_conv, dm_guess, dm_rand = _densities(h2o_dz, rng)
    for dm in (dm_conv, dm_guess, dm_rand, np.eye(h2o_dz.nao_nr())):
        vj_ref, vk_ref = pyscf_hf.dot_eri_dm(eri_s8, dm, hermi=1)
        vj, vk = kernels.jk_s8(eri_s8, dm)
        assert np.allclose(vj, vj_ref, atol=1e-11)
        assert np.allclose(vk, vk_ref, atol=1e-11)
    dms = np.array((dm_conv, dm_rand))
    vj_ref, vk_ref = pyscf_hf.dot_eri_dm(eri_s8, dms, hermi=1)
    vj, vk = kernels.jk_s8(eri_s8, dms)
    assert np.allclose(vj, vj_ref, atol=1e-11) and np.allclose(vk, vk_ref, atol=1e-11)
    # against the dense tensor as an independent reference
    eri = h2o_dz.intor("int2e")
    assert np.allclose(vj[1], np.einsum("ijkl,lk->ij", eri, dm_rand), atol=1e-10)
    assert np.allclose(vk[1], np.einsum("ijkl,jk->il", eri, dm_rand), atol=1e-10)


def test_jk_s8_tiny_basis():
    mol = gto.M(atom="He 0 0 0; H 0 0 1.5", basis="sto-3g", charge=1, verbose=0)  # nao = 2
    eri = mol.intor("int2e", aosym="s8")
    dm = np.array([[1.3, 0.2], [0.2, 0.7]])
    vj_ref, vk_ref = pyscf_hf.dot_eri_dm(eri, dm, hermi=1)
    vj, vk = kernels.jk_s8(eri, dm)
    assert np.allclose(vj, vj_ref, atol=1e-13) and np.allclose(vk, vk_ref, atol=1e-13)


def test_jk_s8_nonsymmetric(eri_s8, h2o_dz, rng):
    """hermi=0: general, antisymmetric and symmetric densities in one stack, as pyscf's dot_eri_dm."""
    n = h2o_dz.nao_nr()
    a = rng.standard_normal((3, n, n))
    a[1] = a[1] - a[1].T
    a[2] = a[2] + a[2].T
    vj_ref, vk_ref = pyscf_hf.dot_eri_dm(eri_s8, a, hermi=0)
    vj, vk = kernels.jk_s8(eri_s8, a, hermi=0)
    assert abs(vj - vj_ref).max() < 1e-11 and abs(vk - vk_ref).max() < 1e-11
    assert abs(vj[1]).max() < 1e-12                       # J of an antisymmetric density
    eri = h2o_dz.intor("int2e")
    assert np.allclose(vk[0], np.einsum("ijkl,jk->il", eri, a[0]), atol=1e-10)
    none, vk0 = kernels.jk_s8(eri_s8, a[0], with_j=False, hermi=0)
    assert none is None and abs(vk0 - vk_ref[0]).max() < 1e-11
    vj0, none = kernels.jk_s8(eri_s8, a[:2], with_k=False, hermi=0)
    assert none is None and abs(vj0 - vj_ref[:2]).max() < 1e-11


# ------------------------------------------------------------- full SCF


def _ref_and_mojo(ref, mojo, etol=1e-9):
    assert ref.converged and mojo.converged
    assert abs(ref.e_tot - mojo.e_tot) < etol
    assert ref.cycles == mojo.cycles
    assert np.allclose(ref.mo_energy, mojo.mo_energy, atol=1e-6)


def test_scf_native_df_rhf(h2o_dz):
    ref = scf.RHF(h2o_dz).density_fit()
    ref.verbose = 0
    ref.kernel()
    mf = mojoscf.RHF(h2o_dz).density_fit()
    mf.verbose = 0
    mf.kernel()
    assert mf.scf_summary["mojoscf_veff_mode"] == 1
    _ref_and_mojo(ref, mf)


def test_scf_native_df_uhf():
    mol = gto.M(atom="O 0 0 0; H 0 0 0.97", basis="cc-pvdz", spin=1, verbose=0)
    ref = scf.UHF(mol).density_fit()
    ref.verbose = 0
    ref.kernel()
    mf = mojoscf.UHF(mol).density_fit()
    mf.verbose = 0
    mf.kernel()
    assert mf.scf_summary["mojoscf_veff_mode"] == 1
    _ref_and_mojo(ref, mf)
    assert abs(ref.spin_square()[0] - mf.spin_square()[0]) < 1e-6


def test_scf_native_incore_rhf(h2o_dz):
    ref = scf.RHF(h2o_dz)
    ref.verbose = 0
    ref.kernel()
    assert ref._eri is not None  # pyscf used the in-core path too
    mf = mojoscf.RHF(h2o_dz)
    mf.verbose = 0
    mf.kernel()
    assert mf.scf_summary["mojoscf_veff_mode"] == 2
    assert mf._eri is not None and mf._eri.ndim == 1
    _ref_and_mojo(ref, mf, etol=1e-10)


def test_scf_native_incore_uhf():
    # pyscf's UHF always recomputes integrals (direct SCF); mojoscf keeps them in
    # core, so the two agree only to the direct-SCF screening threshold.
    mol = gto.M(atom="O 0 0 0; O 0 0 1.208", basis="cc-pvdz", spin=2, verbose=0)
    ref = scf.UHF(mol)
    ref.verbose = 0
    ref.kernel()
    mf = mojoscf.UHF(mol)
    mf.verbose = 0
    mf.kernel()
    assert mf.scf_summary["mojoscf_veff_mode"] == 2
    assert mf.converged and abs(mf.e_tot - ref.e_tot) < 1e-8
    assert abs(mf.spin_square()[0] - ref.spin_square()[0]) < 1e-6


@pytest.fixture
def libcint_engine():
    """Run a test with the Mojo integral engine disabled."""
    saved = mojoscf.integrals.engine()
    mojoscf.integrals.set_engine("libcint")
    try:
        yield
    finally:
        mojoscf.integrals.set_engine(saved)


def test_scf_direct_falls_back_to_callback(h2o_dz, libcint_engine):
    mf = mojoscf.RHF(h2o_dz)
    mf.verbose = 0
    mf.max_memory = 1  # pyscf would not keep the ERIs in core either
    mf.kernel()
    assert mf.scf_summary["mojoscf_veff_mode"] == 0
    ref = scf.RHF(h2o_dz)
    ref.verbose = 0
    ref.max_memory = 1
    ref.kernel()
    _ref_and_mojo(ref, mf)


def test_scf_direct_native_rhf(h2o_dz):
    """Direct SCF (ERIs do not fit in memory) runs the integral-direct Mojo J/K."""
    ref = scf.RHF(h2o_dz)
    ref.verbose = 0
    ref.max_memory = 1
    ref.conv_tol = 1e-11
    ref.kernel()
    mf = mojoscf.RHF(h2o_dz)
    mf.verbose = 0
    mf.max_memory = 1
    mf.conv_tol = 1e-11
    mf.kernel()
    assert mf.scf_summary["mojoscf_veff_mode"] == 3
    assert mf._eri is None
    assert mf.converged and abs(mf.e_tot - ref.e_tot) < 1e-10
    assert mf.cycles == ref.cycles


def test_scf_direct_native_uhf_and_non_incremental():
    mol = gto.M(atom="O 0 0 0; H 0 0 0.97", basis="cc-pvdz", spin=1, verbose=0)
    ref = scf.UHF(mol)
    ref.max_memory = 1
    ref.conv_tol = 1e-11
    ref.kernel()
    mf = mojoscf.UHF(mol)
    mf.max_memory = 1
    mf.conv_tol = 1e-11
    mf.kernel()
    assert mf.scf_summary["mojoscf_veff_mode"] == 3
    assert mf.converged and abs(mf.e_tot - ref.e_tot) < 1e-10
    # full rebuild every cycle instead of the incremental update
    mf2 = mojoscf.UHF(mol)
    mf2.max_memory = 1
    mf2.direct_scf = False
    mf2.conv_tol = 1e-11
    mf2.kernel()
    assert mf2.converged and abs(mf2.e_tot - ref.e_tot) < 1e-10


def test_direct_get_jk_matches_pyscf(h2o_dz, rng):
    n = h2o_dz.nao_nr()
    a = rng.standard_normal((3, n, n))
    dms = a + a.transpose(0, 2, 1)
    vj, vk = mojoscf.integrals.get_jk(h2o_dz, dms, direct_scf_tol=0.0)
    rj, rk = scf.hf.get_jk(h2o_dz, dms)
    assert abs(vj - rj).max() < 1e-11
    assert abs(vk - rk).max() < 1e-11
    vj1, vk1 = mojoscf.integrals.get_jk(h2o_dz, dms[0], with_k=False)
    assert vk1 is None and abs(vj1 - rj[0]).max() < 1e-10


def test_direct_get_jk_nonsymmetric(h2o_dz, rng):
    """hermi=0 through the integral-direct kernel: one pass for all densities, antisymmetric parts K only."""
    n = h2o_dz.nao_nr()
    a = rng.standard_normal((3, n, n))
    a[1] = a[1] - a[1].T
    a[2] = a[2] + a[2].T
    rj, rk = scf.hf.get_jk(h2o_dz, a, hermi=0)
    vj, vk = mojoscf.integrals.get_jk(h2o_dz, a, direct_scf_tol=0.0, hermi=0)
    assert abs(vj - rj).max() < 1e-11 and abs(vk - rk).max() < 1e-11
    none, vk1 = mojoscf.integrals.get_jk(h2o_dz, a[1], with_j=False, direct_scf_tol=0.0, hermi=0)
    assert none is None and abs(vk1 - rk[1]).max() < 1e-11
    vj1, none = mojoscf.integrals.get_jk(h2o_dz, a[0], with_k=False, direct_scf_tol=0.0, hermi=0)
    assert none is None and abs(vj1 - rj[0]).max() < 1e-11


@pytest.mark.parametrize("atom, basis", [
    ("O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59", "def2-tzvp"),
    ("Cu 0 0 0; H 0 0 1.5", "def2-svp"),
    ("C 0 0 0; O 0 0 1.13", "cc-pvtz"),
])
def test_long_range_eris_and_direct_jk(atom, basis, rng):
    """erf(omega r) / r four-centre integrals: in-core ERIs and integral-direct J/K against libcint."""
    mol = gto.M(atom=atom, basis=basis, spin=None, verbose=0)
    with mol.with_range_coulomb(0.4):
        ref = mol.intor("int2e", aosym="s8")
    eri = mojoscf.integrals.int2e_s8(mol, schwarz_tol=0.0, omega=0.4)
    assert abs(eri - ref).max() < 1e-11
    n = mol.nao_nr()
    a = rng.standard_normal((2, n, n))
    a[0] = a[0] + a[0].T
    rj, rk = scf.hf.get_jk(mol, a, hermi=0, omega=0.4)
    vj, vk = mojoscf.integrals.get_jk(mol, a, direct_scf_tol=0.0, hermi=0, omega=0.4)
    assert abs(vj - rj).max() < 1e-10 * max(1.0, abs(rj).max())
    assert abs(vk - rk).max() < 1e-10 * max(1.0, abs(rk).max())
    with pytest.raises(NotImplementedError):
        mojoscf.integrals.get_jk(mol, a[0], omega=-0.4)


@pytest.mark.parametrize("atom, basis", [
    ("Cu 0 0 0; H 0 0 1.5", "def2-svp"),                       # segmented, f shell on Cu
    ("O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59", "def2-tzvp"),    # (fd), (ff) pairs: unbatched kets
    ("N 0 0 0; N 0 0 1.1", "6-31g*"),
    ("C 0 0 0; O 0 0 1.13", "cc-pvtz"),                          # general contraction
])
def test_direct_get_jk_segmented_and_general(atom, basis, rng):
    """Batched kets (one SIMD lane per primitive pair) and the single-quartet fallback against pyscf."""
    mol = gto.M(atom=atom, basis=basis, spin=None, verbose=0)
    n = mol.nao_nr()
    a = rng.standard_normal((2, n, n))
    dms = a + a.transpose(0, 2, 1)
    vj, vk = mojoscf.integrals.get_jk(mol, dms, direct_scf_tol=0.0)
    rj, rk = scf.hf.get_jk(mol, dms)
    assert abs(vj - rj).max() < 1e-10 * max(1.0, abs(rj).max())
    assert abs(vk - rk).max() < 1e-10 * max(1.0, abs(rk).max())


def test_incore_eris_and_df_tensor_from_mojo_engine(h2o_dz):
    mf = mojoscf.RHF(h2o_dz)
    mode, eri, _ = mojoscf.native_veff(mf)
    assert mode == 2
    assert abs(eri - h2o_dz.intor("int2e", aosym="s8")).max() < 1e-12
    mfd = mojoscf.RHF(h2o_dz).density_fit()
    mode, cderi, _ = mojoscf.native_veff(mfd)
    assert mode == 1 and mfd.with_df.auxmol is not None
    from pyscf.df import incore

    assert abs(cderi - incore.cholesky_eri(h2o_dz, auxmol=mfd.with_df.auxmol)).max() < 1e-9


def test_engine_switch_validation():
    with pytest.raises(ValueError):
        mojoscf.integrals.set_engine("fortran")
    assert mojoscf.integrals.engine() in ("mojo", "libcint")


def test_native_veff_rejects_custom_jk(h2o_dz):
    mf = mojoscf.RHF(h2o_dz).density_fit()
    mf.get_jk = lambda *a, **k: scf.RHF.get_jk(mf, *a, **k)
    mode, data, reason = mojoscf.native_veff(mf)
    assert mode == 0 and "get_jk" in reason
    dfj = mojoscf.RHF(h2o_dz).density_fit(only_dfj=True)
    assert mojoscf.native_veff(dfj)[0] == 0


def test_restart_uses_initial_orbitals(h2o_dz):
    mf = mojoscf.RHF(h2o_dz).density_fit()
    mf.verbose = 0
    mf.kernel()
    e0 = mf.e_tot
    mf.kernel()  # dm0 carries mo_coeff/mo_occ: first K build from orbitals
    assert mf.converged and abs(mf.e_tot - e0) < 1e-10 and mf.cycles <= 2


@pytest.mark.parametrize("path", ["df", "incore", "direct"])
@pytest.mark.parametrize("kind", ["rhf", "uhf"])
def test_scf_native_with_ecp(path, kind):
    """Molecules with ECPs use the native J/K with Mojo two-electron integrals in every mode."""
    if kind == "rhf":
        mol = gto.M(atom="Ag 0 0 0; Cl 0 0 2.28", basis="def2-svp", ecp={"Ag": "def2-svp"}, verbose=0)
        ref_cls, mojo_cls = scf.RHF, mojoscf.RHF
    else:
        mol = gto.M(atom="Ag 0 0 0", basis="def2-svp", ecp={"Ag": "def2-svp"}, spin=1, verbose=0)
        ref_cls, mojo_cls = scf.UHF, mojoscf.UHF

    def make(cls):
        mf = cls(mol)
        if path == "df":
            mf = mf.density_fit()
        elif path == "direct":
            mf.max_memory = 1
        mf.conv_tol = 1e-10
        return mf

    ref = make(ref_cls).run()
    mf = make(mojo_cls).run()
    assert ref.converged and mf.converged and mf.cycles == ref.cycles
    assert mf.scf_summary["mojoscf_veff_mode"] == {"df": 1, "incore": 2, "direct": 3}[path]
    assert abs(mf.e_tot - ref.e_tot) < 1e-9
