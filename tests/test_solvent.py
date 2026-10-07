"""PCM-family and SMD solvation with the Mojo kernels (mojoscf.solvent) against pyscf."""
import numpy as np
import pytest
from pyscf import dft, gto, scf
from pyscf.solvent import pcm
from pyscf.solvent.grad import pcm as pcm_grad
from pyscf.solvent.grad import smd as smd_grad

import mojoscf
from mojoscf import integrals, solvent

WATER = "O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587"
FECO2 = "Fe 0 0 0; C 1.9 0 0; O 3.05 0 0; C -1.9 0 0; O -3.05 0 0"


def test_potential_at_points_matches_pyscf():
    from pyscf import df

    mol = gto.M(atom=FECO2, basis="def2-svp", verbose=0)
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(700, 3)) * 4
    zeta = rng.uniform(0.5, 20, 700)
    dm = scf.hf.init_guess_by_minao(mol)
    dms = np.array([dm, 0.3 * dm + 0.1 * np.eye(len(dm))])
    v = df.incore.aux_e2(mol, gto.fakemol_for_charges(pts, expnt=zeta), intor="int3c2e", aosym="s1")
    assert abs(integrals.int1e_grids_dm(mol, dms, pts, zeta) - np.einsum("ijL,sij->sL", v, dms)).max() < 1e-11
    ref = np.einsum("kij,ij->k", mol.intor("int1e_grids", grids=pts), dm)
    assert abs(integrals.int1e_grids_dm(mol, dm, pts) - ref).max() < 1e-11


def _make(mol, model, xc="pbe", spin=0):
    mf = (dft.UKS if spin else dft.RKS)(mol, xc=xc).density_fit()
    if model == "SMD":
        mf = mf.SMD()
        mf.with_solvent.solvent = "water"
    else:
        mf = mf.PCM()
        mf.with_solvent.method = model
    mf.conv_tol = 1e-11
    return mf


@pytest.mark.parametrize("model", ["IEF-PCM", "C-PCM", "COSMO", "SS(V)PE", "SMD"])
def test_models_match_pyscf(model):
    mol = gto.M(atom=WATER, basis="def2-svp", verbose=0)
    ref = _make(mol, model).run()
    mf = mojoscf.dft.accelerate(_make(mol, model))
    assert isinstance(mf.with_solvent, solvent._MojoPCMMixin)
    mf.run()
    assert abs(mf.e_tot - ref.e_tot) < 1e-9
    g1 = mf.nuc_grad_method().kernel()
    for k in ("mo_coeff", "mo_occ", "mo_energy", "e_tot", "converged"):
        setattr(ref, k, getattr(mf, k))
    assert abs(g1 - ref.nuc_grad_method().kernel()).max() < 1e-10


@pytest.mark.parametrize("model", ["IEF-PCM", "C-PCM", "SS(V)PE", "SMD"])
def test_terms_match_pyscf(model):
    """S/D matrices, solver and integral gradient terms against pyscf's functions on the same state."""
    mol = gto.M(atom=FECO2, basis="def2-svp", verbose=0)
    mf = mojoscf.dft.accelerate(_make(mol, model)).run(conv_tol=1e-9)
    s = mf.with_solvent
    D0, S0 = pcm.get_D_S(s.surface, with_S=True, with_D=True)
    D1, S1 = solvent.get_D_S(s.surface, with_S=True, with_D=True)
    assert abs(S0 - S1).max() < 1e-13 and abs(D0 - D1).max() < 1e-12
    dm = mf.make_rdm1()
    mod = smd_grad if model == "SMD" else pcm_grad
    g0 = mod.grad_solver._mojoscf_orig(s, dm)
    assert abs(mod.grad_solver(s, dm) - g0).max() < 1e-11 * max(1.0, abs(g0).max())
    q0 = pcm_grad.grad_qv._mojoscf_orig(s, dm)
    assert abs(pcm_grad.grad_qv(s, dm) - q0).max() < 1e-10 * max(1.0, abs(q0).max())
    v = mf.make_rdm1()[None]
    assert abs(pcm.PCM._get_v(s, v) - s._get_v(v)).max() < 1e-11
    q = np.random.default_rng(1).normal(size=(1, S0.shape[0]))
    assert abs(pcm.PCM._get_vmat(s, q) - s._get_vmat(q)).max() < 1e-11


def test_uks_and_hf_attach():
    mol = gto.M(atom=WATER, basis="def2-svp", charge=1, spin=1, verbose=0)
    ref = _make(mol, "IEF-PCM", spin=1).run()
    mf = mojoscf.dft.accelerate(_make(mol, "IEF-PCM", spin=1)).run()
    assert abs(mf.e_tot - ref.e_tot) < 1e-9
    assert abs(mf.nuc_grad_method().kernel() - ref.nuc_grad_method().kernel()).max() < 1e-7
    # Hartree-Fock: the native loop rejects solvent objects, attach gives the solvent the kernels
    mol = gto.M(atom=WATER, basis="def2-svp", verbose=0)
    with pytest.raises(TypeError, match="solvent"):
        mojoscf.accelerate(scf.RHF(mol).PCM())
    ref = scf.RHF(mol).PCM().run(conv_tol=1e-11)
    mf = solvent.attach(scf.RHF(mol).PCM()).run(conv_tol=1e-11)
    assert abs(mf.e_tot - ref.e_tot) < 1e-9
    assert abs(mf.nuc_grad_method().kernel() - ref.nuc_grad_method().kernel()).max() < 1e-8


def test_unattached_solvent_keeps_pyscf():
    mol = gto.M(atom=WATER, basis="sto-3g", verbose=0)
    mojoscf.dft.accelerate(_make(mol, "C-PCM")).run()          # installs the dispatching hooks
    mf = _make(mol, "C-PCM").run()
    assert not isinstance(mf.with_solvent, solvent._MojoPCMMixin)
    g = mf.nuc_grad_method().kernel()
    assert np.isfinite(g).all()
    assert solvent.attach(scf.RHF(mol)) is not None             # no solvent: unchanged
