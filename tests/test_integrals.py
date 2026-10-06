"""The Mojo integral engine against libcint (``mol.intor``)."""
import numpy as np
import pytest
from pyscf import ao2mo, df, gto, scf

import mojoscf
from mojoscf import integrals as mi

H2O = "O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587"


def _mol(basis, **kw):
    return gto.M(atom=H2O, basis=basis, verbose=0, **kw)


@pytest.fixture(scope="module", params=["sto-3g", "cc-pvdz", "aug-cc-pvdz", "def2-svp"])
def mol(request):
    return _mol(request.param)


def test_one_electron(mol):
    s, t, v = mi.int1e(mol)
    assert np.allclose(s, mol.intor("int1e_ovlp"), atol=1e-13, rtol=0)
    assert np.allclose(t, mol.intor("int1e_kin"), atol=1e-12, rtol=0)
    assert np.allclose(v, mol.intor("int1e_nuc"), atol=1e-11, rtol=0)
    assert np.allclose(mi.get_hcore(mol), scf.hf.get_hcore(mol), atol=1e-11, rtol=0)
    assert np.allclose(mi.get_ovlp(mol), s)


def test_two_electron_s8(mol):
    eri = mi.int2e_s8(mol, schwarz_tol=0.0)
    ref = mol.intor("int2e", aosym="s8")
    assert eri.shape == ref.shape
    assert abs(eri - ref).max() < 1e-12


def test_two_electron_screened_and_full(mol):
    eri = mi.int2e(mol)  # default Schwarz screening
    ref = mol.intor("int2e")
    assert eri.shape == ref.shape
    assert abs(eri - ref).max() < 1e-12


def test_higher_angular_momentum():
    """f and g shells, general contractions."""
    mol = gto.M(atom="Ne 0 0 0; H 0 0 1.9", basis={"Ne": "cc-pvqz", "H": "cc-pvdz"}, spin=1, verbose=0)
    assert max(mol.bas_angular(i) for i in range(mol.nbas)) == 4
    s, t, v = mi.int1e(mol)
    assert abs(s - mol.intor("int1e_ovlp")).max() < 1e-13
    assert abs(t - mol.intor("int1e_kin")).max() < 1e-12
    assert abs(v - mol.intor("int1e_nuc")).max() < 1e-11
    assert abs(mi.int2e_s8(mol, 0.0) - mol.intor("int2e", aosym="s8")).max() < 1e-12


def test_cartesian_basis():
    mol = _mol("cc-pvdz", cart=True)
    s, t, v = mi.int1e(mol)
    assert abs(s - mol.intor("int1e_ovlp")).max() < 1e-13
    assert abs(t - mol.intor("int1e_kin")).max() < 1e-12
    assert abs(v - mol.intor("int1e_nuc")).max() < 1e-11
    assert abs(mi.int2e_s8(mol, 0.0) - mol.intor("int2e", aosym="s8")).max() < 1e-12


def test_density_fitting_tensors():
    mol = _mol("cc-pvdz")
    auxmol = df.addons.make_auxmol(mol, "cc-pvdz-jkfit")
    j3c = mi.int3c2e(mol, auxmol)
    ref3 = df.incore.aux_e2(mol, auxmol, "int3c2e", aosym="s2ij").T
    assert j3c.shape == (auxmol.nao_nr(), mol.nao_nr() * (mol.nao_nr() + 1) // 2)
    assert abs(j3c - ref3).max() < 1e-12
    j2c = mi.int2c2e(auxmol)
    assert abs(j2c - auxmol.intor("int2c2e")).max() < 1e-11
    cderi = mi.cholesky_eri(mol, auxmol=auxmol)
    ref = df.incore.cholesky_eri(mol, auxmol=auxmol)
    assert abs(cderi - ref).max() < 1e-10
    # sanity: the fitted integrals approximate the exact ones (jkfit bases are
    # tuned for J/K, individual integrals carry errors of order 1e-2)
    eri_df = cderi.T @ cderi
    eri = ao2mo.restore(4, mol.intor("int2e"), mol.nao_nr())
    assert abs(eri_df - eri).max() < 0.05


@pytest.mark.parametrize("omega", [0.11, 0.33, 1.5])
def test_long_range_density_fitting_tensors(omega):
    """erf(omega r) / r three- and two-centre integrals and the DF tensor of DF.range_coulomb, against libcint."""
    mol = gto.M(atom="Fe 0 0 0; C 1.9 0 0; O 3.05 0 0", basis="def2-svp", verbose=0)
    auxmol = df.addons.make_auxmol(mol, "def2-universal-jkfit")
    with mol.with_range_coulomb(omega), auxmol.with_range_coulomb(omega):
        ref3 = df.incore.aux_e2(mol, auxmol, "int3c2e", aosym="s2ij").T
        ref2 = auxmol.intor("int2c2e")
        refc = df.incore.cholesky_eri(mol, auxmol=auxmol)
        assert abs(mi.int3c2e(mol, auxmol) - ref3).max() < 1e-12            # omega taken from mol
        assert mi.unsupported_reason(mol, two_electron=True, allow_omega=True) is None
    assert abs(mi.int3c2e(mol, auxmol, omega) - ref3).max() < 1e-12
    assert abs(mi.int2c2e(auxmol, omega) - ref2).max() < 1e-11
    # the long-range metric is nearly singular: Cholesky or the eigen-decomposition fallback may be
    # taken by either code (noise-level differences), so compare the fitted integrals
    cderi = mi.cholesky_eri(mol, auxmol=auxmol, omega=omega)
    assert abs(cderi.T @ cderi - refc.T @ refc).max() < 1e-6
    with pytest.raises(NotImplementedError):
        mi.int2c2e(auxmol, -omega)                  # short-range operator: not supported


def test_cholesky_eri_default_auxbasis():
    mol = _mol("sto-3g")
    cderi = mi.cholesky_eri(mol)
    ref = df.incore.cholesky_eri(mol)
    assert cderi.shape == ref.shape
    assert abs(cderi - ref).max() < 1e-10


def test_attach_rhf_matches_pyscf():
    mol = _mol("cc-pvdz")
    e_ref = scf.RHF(mol).run(conv_tol=1e-11).e_tot
    mf = mi.attach(mojoscf.RHF(mol))
    assert mf._eri is not None
    assert mf.get_ovlp() is mf.get_ovlp(mol)
    mf.conv_tol = 1e-11
    mf.kernel()
    assert mf.converged
    assert abs(mf.e_tot - e_ref) < 1e-9
    assert mf.scf_summary["mojoscf_veff_mode"] == 2


def test_attach_uhf_matches_pyscf():
    mol = gto.M(atom="O 0 0 0; H 0 0 0.97", basis="cc-pvdz", spin=1, verbose=0)
    e_ref = scf.UHF(mol).run(conv_tol=1e-11).e_tot
    mf = mi.attach(mojoscf.UHF(mol))
    mf.conv_tol = 1e-11
    mf.kernel()
    assert mf.converged
    assert abs(mf.e_tot - e_ref) < 1e-9


def test_attach_density_fitting():
    mol = _mol("cc-pvdz")
    e_ref = scf.RHF(mol).density_fit().run(conv_tol=1e-11).e_tot
    mf = mi.attach(mojoscf.RHF(mol).density_fit())
    assert mf.with_df._cderi is not None
    mf.conv_tol = 1e-11
    mf.kernel()
    assert mf.converged
    assert abs(mf.e_tot - e_ref) < 1e-9
    assert mf.scf_summary["mojoscf_veff_mode"] == 1


def test_unsupported_molecules():
    mol = gto.M(atom="Cu 0 0 0", basis="lanl2dz", ecp="lanl2dz", spin=1, verbose=0)
    assert "core potential" in mi.unsupported_reason(mol)
    assert mi.unsupported_reason(mol, two_electron=True) is None
    assert mi.unsupported_reason(mol, allow_ecp=True) is None
    with pytest.raises(NotImplementedError):
        mi.get_hcore(mol)  # would miss the ECP
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", nucmod="G", verbose=0)
    assert "point nuclei" in mi.unsupported_reason(mol)
    assert mi.unsupported_reason(mol, two_electron=True) is None
    assert mi.unsupported_reason(_mol("sto-3g")) is None
    mol = _mol("sto-3g")
    mol.omega = 0.3
    assert "omega" in mi.unsupported_reason(mol)
    assert "omega" in mi.unsupported_reason(mol, two_electron=True)


def test_two_electron_integrals_with_ecp():
    """ECPs only enter the one-electron Hamiltonian: the ERIs and DF integrals come from the engine."""
    mol = gto.M(
        atom="Pt 0 0 0; Cl 0 0 2.32; N 2.05 0 0", basis="def2-svp", ecp={"Pt": "def2-svp"}, charge=0, spin=0,
        verbose=0,
    )
    assert abs(mi.int2e_s8(mol, 0.0) - mol.intor("int2e", aosym="s8")).max() < 1e-12
    auxmol = df.addons.make_auxmol(mol, "def2-universal-jkfit")
    assert abs(mi.int3c2e(mol, auxmol) - df.incore.aux_e2(mol, auxmol, "int3c2e", aosym="s2ij").T).max() < 1e-12
    # the Gaussian one-electron integrals (ECP atoms are point charges Z - core) match libcint;
    # the core Hamiltonian would miss the ECP and is refused
    s, t, v = mi.int1e(mol)
    assert abs(s - mol.intor("int1e_ovlp")).max() < 1e-13
    assert abs(t - mol.intor("int1e_kin")).max() < 1e-12
    assert abs(v - mol.intor("int1e_nuc")).max() < 1e-11
    s1, t1, v1 = mi.int1e_ip(mol)
    assert abs(v1 - mol.intor("int1e_ipnuc")).max() < 1e-10
    with pytest.raises(NotImplementedError):
        mi.get_hcore(mol)
    # attach keeps pyscf's hcore (with the ECP) and takes the ERIs from the engine
    e_ref = scf.RHF(mol).run(conv_tol=1e-11).e_tot
    mf = mi.attach(mojoscf.RHF(mol))
    assert "get_hcore" not in mf.__dict__ and mf._eri is not None
    assert abs(mf.run(conv_tol=1e-11).e_tot - e_ref) < 1e-9


def test_basis_tables_layout():
    mol = _mol("cc-pvdz")
    atm, bas, env, nf, c2s = mi.basis_tables(mol)
    assert atm.dtype == np.int64 and bas.dtype == np.int64 and env.dtype == np.float64
    assert list(nf) == [1, 3, 5]
    assert c2s.size == 1 * 1 + 3 * 3 + 6 * 5


def test_generated_kernel_is_current():
    """The generated kernels in integrals.mojo match tools/gen_eri_kernel.py."""
    import importlib.util
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("gen_eri_kernel", root / "tools" / "gen_eri_kernel.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    text = gen.TARGET.read_text()
    region = text[text.index(gen.BEGIN) + len(gen.BEGIN):text.index(gen.END)]
    assert region == gen.generated(), "run `python tools/gen_eri_kernel.py` after editing the generator"


def test_large_l_against_small_l_pairs():
    """f and g shells paired with s and p shells (the extended specialised kernels) and with each other."""
    mol = gto.M(
        atom="Cl 0 0 0; F 0 0 1.7; H 1.0 0.3 -0.6", basis={"Cl": "cc-pvtz", "F": "cc-pvdz", "H": "cc-pvdz"},
        spin=0, charge=1, verbose=0,
    )
    eri = mi.int2e_s8(mol, schwarz_tol=0.0)
    assert abs(eri - mol.intor("int2e", aosym="s8")).max() < 1e-12
