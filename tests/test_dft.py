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
    for deriv in (0, 1, 2):
        name = ("GTOval_cart" if cart else "GTOval_sph") + (f"_deriv{deriv}" if deriv else "")
        ref = np.asarray(mol.eval_gto(name, coords))
        assert abs(mdft.eval_ao(mol, coords, deriv) - ref).max() < 1e-12 * max(1.0, abs(ref).max())


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


@pytest.mark.parametrize("xc", ["lda,vwn", "pbe", "b3lyp", "tpss"])
def test_xc_gradient_matches_pyscf(water_grid, xc):
    from pyscf.grad import rks as rks_grad
    from pyscf.grad import uks as uks_grad

    mol, grids = water_grid
    dm = scf.hf.init_guess_by_minao(mol)
    dms = np.array([0.55 * dm, 0.45 * dm])
    # the hooked get_vxc (Mojo for mojoscf's NumInt) against pyscf's own
    for mod, d in ((rks_grad, dm), (uks_grad, dms)):
        orig = mod.get_vxc._mojoscf_orig
        e0, v0 = orig(numint.NumInt(), mol, grids, xc, d)
        e1, v1 = mod.get_vxc(mdft.NumInt(), mol, grids, xc, d)
        assert v0.shape == v1.shape and abs(v0 - v1).max() < 1e-12
        e2, v2 = mod.get_vxc(numint.NumInt(), mol, grids, xc, d)
        assert abs(v0 - v2).max() < 1e-14
    # the density-contracted gradient term
    for d, spin in ((dm, 0), (dms, 1)):
        ref = mdft.grad_xc(numint.NumInt(), mol, grids, xc, d, spin=spin)
        assert abs(mdft.grad_xc(mdft.NumInt(), mol, grids, xc, d, spin=spin) - ref).max() < 1e-12


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


@pytest.mark.parametrize("kind", ["rks", "uks", "dfrks", "dfuks", "directrks", "directuks"])
def test_scf_with_accelerate(kind):
    spin = 1 if kind.endswith("uks") else 0
    mol = gto.M(atom=WATER, basis="def2-svp", charge=spin, spin=spin, verbose=0)

    def make():
        mf = (dft.UKS if spin else dft.RKS)(mol, xc="b3lyp")
        if kind.startswith("df"):
            mf = mf.density_fit()
        elif kind.startswith("direct"):
            mf.max_memory = 0                   # no in-core ERIs: direct SCF
        mf.conv_tol = 1e-11
        return mf

    ref = make()
    ref.kernel()
    mf = mdft.accelerate(make())
    assert isinstance(mf._numint, mdft.NumInt) and isinstance(mf, mdft._MojoKSHook)
    mf.kernel()
    assert abs(mf.e_tot - ref.e_tot) < 1e-9
    assert (mf._eri is not None) == (kind in ("rks", "uks"))
    g = mf.nuc_grad_method()
    cls = {"rks": mojoscf.grad.KSGradients, "uks": mojoscf.grad.UKSGradients,
           "dfrks": mojoscf.grad.DFKSGradients, "dfuks": mojoscf.grad.DFUKSGradients}[kind.replace("direct", "")]
    assert type(g) is cls and g._direct_2e()
    assert abs(g.kernel() - ref.nuc_grad_method().kernel()).max() < 1e-7


def test_exact_jk_matches_pyscf():
    mol = gto.M(atom=WATER, basis="def2-svp", verbose=0)
    dm = scf.hf.init_guess_by_minao(mol)
    dms = np.array([dm, 0.3 * dm])
    for max_memory in (4000, 0):
        ref = dft.RKS(mol)
        ref.max_memory = max_memory
        mf = mdft.accelerate(dft.RKS(mol))
        mf.max_memory = max_memory
        for d in (dm, dms):
            j0, k0 = ref.get_jk(mol, d)
            j1, k1 = mf.get_jk(mol, d)
            assert abs(j0 - j1).max() < 1e-10 and abs(k0 - k1).max() < 1e-10
            assert abs(mf.get_j(mol, d) - j0).max() < 1e-10
        assert abs(mf.get_k(mol, dm, omega=0.3) - ref.get_k(mol, dm, omega=0.3)).max() < 1e-10   # pyscf's


@pytest.mark.parametrize("df", [False, True])
@pytest.mark.parametrize("xc", ["lda,vwn", "pbe", "tpss", "camb3lyp"])
def test_ks_gradients(df, xc):
    """Pure, meta-GGA and range-separated functionals; grid response and pyscf's NumInt."""
    mol = gto.M(atom=WATER, basis="def2-svp", verbose=0)

    def make():
        mf = dft.RKS(mol, xc=xc)
        return mf.density_fit() if df else mf

    ref = make().run(conv_tol=1e-11)
    g0 = ref.nuc_grad_method().kernel()
    mf = mdft.accelerate(make()).run(conv_tol=1e-11)
    g = mf.nuc_grad_method()
    assert g._direct_2e() == (xc != "camb3lyp")         # range separation: pyscf's get_veff
    assert abs(g.kernel() - g0).max() < 1e-7
    if xc == "pbe":
        g = mf.nuc_grad_method()
        g.grid_response = True
        assert not g._direct_2e()
        g1 = ref.nuc_grad_method()
        g1.grid_response = True
        assert abs(g.kernel() - g1.kernel()).max() < 1e-7
        mf._numint = numint.NumInt()                     # pyscf's XC code inside the Mojo gradient
        g = mf.nuc_grad_method()
        assert g._direct_2e() and abs(g.kernel() - g0).max() < 1e-7


def test_ks_gradient_scanner():
    mol = gto.M(atom=WATER, basis="def2-svp", verbose=0)
    scan = mdft.accelerate(dft.RKS(mol, xc="pbe").density_fit()).nuc_grad_method().as_scanner()
    ref = dft.RKS(mol, xc="pbe").density_fit().nuc_grad_method().as_scanner()
    for geom in (WATER, "O 0 0 0.05; H 0 0.78 0.6; H 0 -0.75 0.57"):
        e1, g1 = scan(geom)
        e0, g0 = ref(geom)
        assert abs(e1 - e0) < 1e-8 and abs(g1 - g0).max() < 1e-6


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
    assert isinstance(g, mojoscf.grad.DFKSGradients) and g._direct_2e()
    assert abs(g.kernel() - ref.nuc_grad_method().kernel()).max() < 1e-7


def test_diis_choice():
    from pyscf.scf import diis as scf_diis

    mol = gto.M(atom=WATER, basis="sto-3g", verbose=0)
    mf = mdft.accelerate(dft.RKS(mol))
    assert mf.DIIS is mojoscf.CDIIS
    mf.diis_space_rollback = 2
    assert mf.DIIS is scf_diis.CDIIS
    mf = mdft.accelerate(dft.RKS(mol))
    mf.DIIS = scf_diis.ADIIS                    # a scheme set on the object wins
    assert mf.DIIS is scf_diis.ADIIS
    mf.kernel()
    assert abs(mf.e_tot - dft.RKS(mol).run(DIIS=scf_diis.ADIIS).e_tot) < 1e-8


def test_accelerate_rejects_hf():
    with pytest.raises(TypeError):
        mdft.accelerate(scf.RHF(gto.M(atom=WATER, basis="sto-3g", verbose=0)))
