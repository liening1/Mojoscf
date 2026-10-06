"""Kohn-Sham DFT with the Mojo XC integration (mojoscf.dft) against pyscf."""
import numpy as np
import pytest
from pyscf import dft, gto, qmmm, scf
from pyscf.dft import numint

import mojoscf
from mojoscf import dft as mdft

WATER = "O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587"


@pytest.fixture(scope="module")
def water_grid():
    mol = gto.M(atom=WATER, basis="cc-pvtz", verbose=0)
    grids = dft.gen_grid.Grids(mol)
    grids.build()
    return mol, grids


@pytest.mark.parametrize("basis, cart", [("cc-pvtz", False), ("6-31g*", True), ("cc-pvqz", False)])
def test_eval_ao_matches_pyscf(basis, cart):
    mol = gto.M(atom=WATER, basis=basis, cart=cart, verbose=0)
    coords = np.random.default_rng(0).uniform(-3, 3, (700, 3))
    for deriv in (0, 1):
        name = ("GTOval_cart" if cart else "GTOval_sph") + ("_deriv1" if deriv else "")
        ref = np.asarray(mol.eval_gto(name, coords))
        assert abs(mdft.eval_ao(mol, coords, deriv) - ref).max() < 1e-12


@pytest.mark.parametrize("xc", ["lda,vwn", "pbe", "b3lyp", "blyp"])
def test_nr_rks_matches_pyscf(water_grid, xc):
    mol, grids = water_grid
    dm = scf.hf.init_guess_by_minao(mol)
    n0, e0, v0 = numint.NumInt().nr_rks(mol, grids, xc, dm)
    n1, e1, v1 = mdft.NumInt().nr_rks(mol, grids, xc, dm)
    assert abs(n0 - n1) < 1e-10 and abs(e0 - e1) < 1e-10
    assert abs(v0 - v1).max() < 1e-10
    # a stack of densities
    n2, e2, v2 = mdft.NumInt().nr_rks(mol, grids, xc, np.array([dm, 0.5 * dm]))
    assert abs(n2[0] - n0) < 1e-10 and abs(v2[0] - v0).max() < 1e-10
    n3, e3, v3 = numint.NumInt().nr_rks(mol, grids, xc, 0.5 * dm)
    assert abs(e2[1] - e3) < 1e-10 and abs(v2[1] - v3).max() < 1e-10


@pytest.mark.parametrize("xc", ["lda,vwn", "pbe"])
def test_nr_uks_matches_pyscf(water_grid, xc):
    mol, grids = water_grid
    dm = scf.hf.init_guess_by_minao(mol)
    dms = np.array([0.55 * dm, 0.45 * dm])
    n0, e0, v0 = numint.NumInt().nr_uks(mol, grids, xc, dms)
    n1, e1, v1 = mdft.NumInt().nr_uks(mol, grids, xc, dms)
    assert abs(n0 - n1).max() < 1e-10 and abs(e0 - e1) < 1e-10
    assert abs(v0 - v1).max() < 1e-10


def test_cartesian_and_fallbacks():
    mol = gto.M(atom=WATER, basis="6-31g*", cart=True, verbose=0)
    grids = dft.gen_grid.Grids(mol)
    grids.build()
    dm = scf.hf.init_guess_by_minao(mol)
    for xc in ("pbe", "tpss"):          # meta-GGA keeps pyscf's code
        n0, e0, v0 = numint.NumInt().nr_rks(mol, grids, xc, dm)
        n1, e1, v1 = mdft.NumInt().nr_rks(mol, grids, xc, dm)
        assert abs(e0 - e1) < 1e-10 and abs(v0 - v1).max() < 1e-10
    a = np.random.default_rng(1).normal(size=dm.shape)        # non-symmetric density: pyscf's path
    n0, e0, v0 = numint.NumInt().nr_rks(mol, grids, "lda,vwn", dm + 1e-3 * a, hermi=0)
    n1, e1, v1 = mdft.NumInt().nr_rks(mol, grids, "lda,vwn", dm + 1e-3 * a, hermi=0)
    assert abs(e0 - e1) < 1e-12 and abs(v0 - v1).max() < 1e-12


@pytest.mark.parametrize("kind", ["rks", "uks", "dfrks", "dfuks"])
def test_scf_with_accelerate(kind):
    spin = 1 if kind.endswith("uks") else 0
    mol = gto.M(atom=WATER, basis="def2-svp", charge=spin, spin=spin, verbose=0)

    def make():
        mf = (dft.UKS if spin else dft.RKS)(mol, xc="b3lyp")
        if kind.startswith("df"):
            mf = mf.density_fit()
        mf.conv_tol = 1e-11
        return mf

    ref = make()
    ref.kernel()
    mf = mdft.accelerate(make())
    assert isinstance(mf._numint, mdft.NumInt) and isinstance(mf, mdft._MojoKSHook)
    mf.kernel()
    assert abs(mf.e_tot - ref.e_tot) < 1e-9
    g = mf.nuc_grad_method().kernel()
    assert abs(g - ref.nuc_grad_method().kernel()).max() < 1e-7


def test_accelerate_qmmm_dft():
    rng = np.random.default_rng(4)
    coords = rng.normal(size=(100, 3))
    coords = coords / np.linalg.norm(coords, axis=1)[:, None] * rng.uniform(3.5, 7.0, (100, 1))
    q = rng.uniform(-0.6, 0.6, 100)
    mol = gto.M(atom=WATER, basis="def2-svp", verbose=0)
    ref = qmmm.mm_charge(dft.RKS(mol, xc="pbe").density_fit(), coords, q).run(conv_tol=1e-11)
    mf = mdft.accelerate(qmmm.mm_charge(dft.RKS(mol, xc="pbe").density_fit(), coords, q))
    assert isinstance(mf, mojoscf.qmmm._MojoQMMMHook)
    mf.run(conv_tol=1e-11)
    assert abs(mf.e_tot - ref.e_tot) < 1e-9
    g = mf.nuc_grad_method()
    assert abs(g.kernel() - ref.nuc_grad_method().kernel()).max() < 1e-7


def test_accelerate_rejects_hf():
    with pytest.raises(TypeError):
        mdft.accelerate(scf.RHF(gto.M(atom=WATER, basis="sto-3g", verbose=0)))
