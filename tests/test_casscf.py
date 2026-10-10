"""DF-CASSCF with the Mojo DF kernels (mojoscf.casscf) against pyscf."""
import numpy as np
import pytest
from pyscf import gto, mcscf, scf
from pyscf.mcscf import df as mcscf_df
from pyscf.mcscf import mc1step

import mojoscf
from mojoscf import casscf as mcas


def _orig_eris():
    eris = mcscf_df._ERIS
    return getattr(eris, "_mojoscf_orig", eris)


@pytest.fixture(scope="module")
def o2():
    """O2 triplet / cc-pVDZ, accelerated density-fitted ROHF, CAS(8,6)."""
    mol = gto.M(atom="O 0 0 0; O 0 0 1.21", basis="cc-pvdz", spin=2, verbose=0)
    mf = mojoscf.accelerate(scf.ROHF(mol).density_fit()).run()
    return mf, mcscf.CASSCF(mf, 6, 8)


@pytest.mark.parametrize("keep", [True, False])
def test_eris_match_pyscf(o2, keep):
    """j_pc, k_pc, ppaa, papa and vhf_c, with the MO-basis tensor kept or in blocks, equal pyscf's _ERIS."""
    mf, mc = o2
    mo = mf.mo_coeff
    ref = _orig_eris()(mc, mo, mf.with_df)
    eris = mcas.ERIS(mc, mo, mf.with_df._cderi, 2000, keep=keep)
    assert (eris.bmo is not None) == keep
    for name in ("j_pc", "k_pc", "ppaa", "papa", "vhf_c"):
        a, b = np.asarray(getattr(eris, name)), np.asarray(getattr(ref, name))
        assert a.shape == b.shape and abs(a - b).max() < 1e-10, name


def test_update_jk_in_ah_matches_pyscf(o2):
    """The orbital-Hessian J/K in the MO basis equals pyscf's two AO J/K builds."""
    mf, mc = o2
    mo = mf.mo_coeff
    nmo = mo.shape[1]
    eris = mcas.ERIS(mc, mo, mf.with_df._cderi, 2000, keep=True)
    rng = np.random.default_rng(3)
    r = rng.standard_normal((nmo, nmo)) * 0.1
    r = r - r.T
    casdm1 = rng.standard_normal((mc.ncas, mc.ncas))
    casdm1 = casdm1 + casdm1.T
    va, vc = eris.update_jk_in_ah(r, casdm1)
    va0, vc0 = mc1step.CASSCF.update_jk_in_ah(mc, mo, r, casdm1, eris)
    assert va.shape == va0.shape and vc.shape == vc0.shape
    assert abs(va - va0).max() < 1e-10 and abs(vc - vc0).max() < 1e-10


@pytest.mark.parametrize("case", ["n2", "o2", "n2-sa"])
def test_dfcasscf_energy(case, monkeypatch):
    """pyscf's DF-CASSCF of an accelerated reference builds its integrals with mojoscf.casscf and its
    orbital-Hessian J/K in the MO basis, and reaches pyscf's energy (state-averaged too)."""
    if case == "o2":       # O2 triplet, ROHF reference, CAS(8,6)
        mol = gto.M(atom="O 0 0 0; O 0 0 1.21", basis="cc-pvdz", spin=2, verbose=0)
        make, ncas, nelecas = scf.ROHF, 6, 8
    else:
        mol = gto.M(atom="N 0 0 0; N 0 0 1.12", basis="cc-pvdz", verbose=0)
        make, ncas, nelecas = scf.RHF, 6, 6

    def casscf(mf):
        mc = mcscf.CASSCF(mf, ncas, nelecas)
        if case == "n2-sa":
            mc = mc.state_average_([0.5, 0.5])
        return mc.run(conv_tol=1e-10)

    ref_mf = make(mol).density_fit().run(conv_tol=1e-11)
    e0 = casscf(ref_mf).e_tot
    mf = mojoscf.accelerate(make(mol).density_fit()).run(conv_tol=1e-11)
    built, ah = [], []
    orig = mcas.make_eris
    monkeypatch.setattr(mcas, "make_eris", lambda *a: built.append(orig(*a)) or built[-1])
    orig_ah = mcas.ERIS.update_jk_in_ah
    monkeypatch.setattr(mcas.ERIS, "update_jk_in_ah", lambda self, *a: ah.append(1) or orig_ah(self, *a))
    mc = casscf(mf)
    assert built and all(isinstance(e, mcas.ERIS) and e.bmo is not None for e in built)
    assert ah
    assert abs(mc.e_tot - e0) < 1e-8


def test_blocked_without_memory_for_the_tensor(monkeypatch):
    """When the MO-basis tensor does not fit, the integrals are built in blocks and the orbital-Hessian
    J/K runs on the DF object; the energy is unchanged."""
    mol = gto.M(atom="N 0 0 0; N 0 0 1.12", basis="cc-pvdz", verbose=0)
    mf = mojoscf.accelerate(scf.RHF(mol).density_fit()).run(conv_tol=1e-11)
    e0 = mcscf.CASSCF(mf, 6, 6).run(conv_tol=1e-10).e_tot
    orig = mcas._memory_mb
    monkeypatch.setattr(mcas, "_memory_mb", lambda *a: 1e9 if a[-1] else orig(*a))
    built = []
    orig_make = mcas.make_eris
    monkeypatch.setattr(mcas, "make_eris", lambda *a: built.append(orig_make(*a)) or built[-1])
    mc = mcscf.CASSCF(mf, 6, 6).run(conv_tol=1e-10)
    assert built and all(isinstance(e, mcas.ERIS) and e.bmo is None for e in built)
    assert abs(mc.e_tot - e0) < 1e-9


def test_pyscf_eris_kept_without_mojo_df():
    """A plain pyscf reference keeps pyscf's integrals (and its AO orbital-Hessian J/K)."""
    mol = gto.M(atom="N 0 0 0; N 0 0 1.12", basis="cc-pvdz", verbose=0)
    mf = scf.RHF(mol).density_fit().run()
    mc = mcscf.CASSCF(mf, 4, 4)
    mcas.install()
    assert mcas.make_eris(mc, mf.mo_coeff, mf.with_df) is None
    assert not isinstance(mc.ao2mo(mf.mo_coeff), mcas.ERIS)


def _orig_nevpt2_eris():
    from pyscf.mrpt import dfnevpt2

    return getattr(dfnevpt2._ERIS, "_mojoscf_orig", dfnevpt2._ERIS)


def test_nevpt2_eris_match_pyscf(o2):
    """vhf_c, ppaa, papa, pacv and h1eff of DF-NEVPT2 equal pyscf's dfnevpt2._ERIS, and the DF factors
    kept instead of cvcv reproduce it."""
    mf, _ = o2
    mc = mcscf.CASCI(mf, 6, 8)
    mc.kernel()
    ref = _orig_nevpt2_eris()(mc, mc.mo_coeff, mf.with_df)
    eris = mcas.nevpt2_eris(mc, mc.mo_coeff, mf.with_df)
    assert isinstance(eris, mcas.NEVPT2ERIS) and eris["cvcv"] is None
    for name in ("vhf_c", "ppaa", "papa", "pacv", "h1eff"):
        a, b = np.asarray(eris[name]), np.asarray(ref[name])
        assert a.shape == b.shape and abs(a - b).max() < 1e-10, name
    cv = eris.cv.reshape(eris.cv.shape[0], -1)
    assert abs(cv.T @ cv - ref["cvcv"]).max() < 1e-10


def test_nevpt2_energy(o2, monkeypatch):
    """pyscf's NEVPT2 of a DF-CASCI on an accelerated reference uses mojoscf's integrals (and the Sijrs
    subspace from the DF factors) and gives the energy of pyscf's integral code."""
    from pyscf import mrpt
    from pyscf.mrpt import dfnevpt2, nevpt2

    mf, _ = o2
    mc = mcscf.CASCI(mf, 6, 8).run()
    built = []
    orig = mcas.nevpt2_eris
    monkeypatch.setattr(mcas, "nevpt2_eris", lambda *a: built.append(orig(*a)) or built[-1])
    e1 = mrpt.NEVPT(mc).kernel()
    assert built and isinstance(built[0], mcas.NEVPT2ERIS)
    monkeypatch.setattr(dfnevpt2, "_ERIS", getattr(dfnevpt2._ERIS, "_mojoscf_orig", dfnevpt2._ERIS))
    monkeypatch.setattr(nevpt2, "Sijrs", getattr(nevpt2.Sijrs, "_mojoscf_orig", nevpt2.Sijrs))
    e0 = mrpt.NEVPT(mc).kernel()
    assert abs(e1 - e0) < 1e-10
