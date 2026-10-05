"""QM/MM (pyscf.qmmm) with the MM-charge integrals from the Mojo engine, against pyscf."""
import numpy as np
import pytest
from pyscf import df, gto, lib, qmmm, scf

import mojoscf
from mojoscf import integrals
from mojoscf import qmmm as mojo_qmmm

WATER = "O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587"


def _charges(n, seed=3, rmin=3.0, rmax=9.0):
    """``n`` charges (Angstrom) in a shell around the origin, and their values."""
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    coords = d * rng.uniform(rmin, rmax, n)[:, None]
    return coords, rng.uniform(-0.8, 0.8, n)


def _blocks(n, f, blk=200):
    return sum(f(i0, i1) for i0, i1 in lib.prange(0, n, blk))


@pytest.mark.parametrize("basis, cart", [("cc-pvdz", False), ("cc-pvqz", False), ("6-31g*", True)])
def test_mm_kernels_point_charges(basis, cart):
    """Potential, its derivative matrix and the charge forces; cc-pVQZ reaches g shells (degree 9)."""
    mol = gto.M(atom=WATER, basis=basis, cart=cart, verbose=0)
    coords, q = _charges(1100)          # more than one group of 1024 charges, a partial last chunk
    c = coords / lib.param.BOHR
    n = len(q)
    v_ref = _blocks(n, lambda i0, i1: np.einsum("kpq,k->pq", mol.intor("int1e_grids", grids=c[i0:i1]), q[i0:i1]))
    assert abs(integrals.int1e_grids_sum(mol, c, q) - v_ref).max() < 1e-12
    ip_ref = _blocks(n, lambda i0, i1: np.einsum("xkpq,k->xpq", mol.intor("int1e_grids_ip", grids=c[i0:i1]), q[i0:i1]))
    assert abs(integrals.int1e_grids_ip_sum(mol, c, q) - ip_ref).max() < 1e-12
    a = np.random.default_rng(0).normal(size=(mol.nao, mol.nao))
    dm = a + a.T
    def fake(i0, i1):
        m = gto.fakemol_for_charges(c[i0:i1], np.full(i1 - i0, 1e16))
        m.cart = cart
        return m

    f_ref = np.vstack([
        np.einsum("xpqk,qp->kx", df.incore.aux_e2(mol, fake(i0, i1), "int3c2e_ip2", aosym="s1", comp=3) * q[i0:i1], dm)
        for i0, i1 in lib.prange(0, n, 200)
    ])
    assert abs(integrals.mm_charge_forces(mol, dm, c, q) - f_ref).max() < 1e-11


def test_mm_kernels_gaussian_charges():
    mol = gto.M(atom=WATER, basis="cc-pvtz", verbose=0)
    coords, q = _charges(150, rmin=1.5)       # some charges overlap the density
    c = coords / lib.param.BOHR
    zeta = np.random.default_rng(1).uniform(0.5, 3.0, len(q))
    fake = gto.fakemol_for_charges(c, zeta)
    j3c = df.incore.aux_e2(mol, fake, "int3c2e", aosym="s1").reshape(mol.nao, mol.nao, -1)
    assert abs(integrals.int1e_grids_sum(mol, c, q, zeta) - np.einsum("pqk,k->pq", j3c, q)).max() < 1e-12
    ip = df.incore.aux_e2(mol, fake, "int3c2e_ip1", aosym="s1", comp=3)
    assert abs(integrals.int1e_grids_ip_sum(mol, c, q, zeta) - np.einsum("xpqk,k->xpq", ip, q)).max() < 1e-12
    dm = scf.hf.init_guess_by_minao(mol)
    ip2 = df.incore.aux_e2(mol, fake, "int3c2e_ip2", aosym="s1", comp=3)
    f_ref = np.einsum("xpqk,qp->kx", ip2 * q, dm)
    assert abs(integrals.mm_charge_forces(mol, dm, c, q, zeta) - f_ref).max() < 1e-12


def _make(kind, uhf, mojo_first, radii, coords, q):
    mol = gto.M(atom=WATER, basis="cc-pvdz", charge=int(uhf), spin=int(uhf), verbose=0)
    if mojo_first:
        mf = mojoscf.UHF(mol) if uhf else mojoscf.RHF(mol)
    else:
        mf = scf.UHF(mol) if uhf else scf.RHF(mol)
    if kind == "df":
        mf = mf.density_fit()
    mf = qmmm.mm_charge(mf, coords, q, radii=radii)
    mf.conv_tol = 1e-11
    return mf


@pytest.mark.parametrize("kind", ["incore", "df"])
@pytest.mark.parametrize("uhf", [False, True])
@pytest.mark.parametrize("gaussian", [False, True])
def test_qmmm_scf_gradient_and_mm_forces(kind, uhf, gaussian):
    coords, q = _charges(400)
    radii = np.full(len(q), 0.9) if gaussian else None
    ref = _make(kind, uhf, False, radii, coords, q)
    ref.kernel()
    gref = ref.nuc_grad_method()
    de_ref = gref.kernel()
    dm = ref.make_rdm1()
    dm = dm[0] + dm[1] if uhf else dm
    f_ref = gref.grad_hcore_mm(dm) + gref.grad_nuc_mm()
    for mojo_first in (False, True):
        mf = _make(kind, uhf, mojo_first, radii, coords, q)
        if not mojo_first:
            mojoscf.accelerate(mf)
        mf.kernel()
        assert isinstance(mf, mojo_qmmm._MojoQMMMHook)
        assert abs(mf.e_tot - ref.e_tot) < 1e-10
        g = mf.nuc_grad_method()
        assert isinstance(g, mojo_qmmm._MojoQMMMGrad)
        assert isinstance(g, (mojoscf.grad.Gradients, mojoscf.grad.UGradients,
                              mojoscf.grad.DFGradients, mojoscf.grad.DFUGradients))
        assert abs(g.kernel() - de_ref).max() < 1e-9
        f = g.grad_hcore_mm(mf.make_rdm1()) + g.grad_nuc_mm()      # (alpha, beta) pair accepted for UHF
        assert abs(f - f_ref).max() < 1e-9


def test_qmmm_libcint_engine_and_undo():
    coords, q = _charges(50)
    mol = gto.M(atom=WATER, basis="cc-pvdz", verbose=0)
    ref = qmmm.mm_charge(scf.RHF(mol), coords, q).run(conv_tol=1e-11)
    saved = integrals.engine()
    integrals.set_engine("libcint")
    try:
        mf = mojoscf.accelerate(qmmm.mm_charge(scf.RHF(mol), coords, q))
        assert not mojo_qmmm.mojo_ok(mol, mf.mm_mol)
        mf.run(conv_tol=1e-11)
        assert abs(mf.e_tot - ref.e_tot) < 1e-10
    finally:
        integrals.set_engine(saved)
    plain = mf.undo_qmmm()          # the Mojo hook stays in the class but steps aside
    assert abs(plain.get_hcore() - scf.hf.get_hcore(mol)).max() < 1e-12
    plain.kernel()
    assert abs(plain.e_tot - scf.RHF(mol).run(conv_tol=1e-11).e_tot) < 1e-9


def test_mm_supported_limits():
    mol = gto.M(atom="He 0 0 0", basis={"He": [[5, [1.0, 1.0]]]}, verbose=0)    # an h shell
    assert not integrals.mm_supported(mol)
    with pytest.raises(NotImplementedError):
        integrals.int1e_grids_sum(mol, [[0.0, 0.0, 3.0]], [1.0])


@pytest.mark.parametrize("gaussian", [False, True])
def test_mm_grad_terms_one_pass(gaussian):
    """Density-contracted Hermite matrices: atom term and charge forces equal the matrix route."""
    mol = gto.M(atom=WATER, basis="cc-pvtz", verbose=0)
    coords, q = _charges(300, rmin=2.0)
    c = coords / lib.param.BOHR
    zeta = np.full(len(q), 2.0) if gaussian else None
    a = np.random.default_rng(2).normal(size=(mol.nao, mol.nao))
    dm = a + a.T
    g_atoms, g_charges = integrals.mm_grad_terms(mol, dm, c, q, zeta)
    m = integrals.int1e_grids_ip_sum(mol, c, q, zeta)
    ref = np.array([2 * np.einsum("xij,ij->x", m[:, p0:p1], dm[p0:p1]) for p0, p1 in mol.aoslice_by_atom()[:, 2:]])
    assert abs(g_atoms - ref).max() < 1e-12
    assert abs(g_charges - integrals.mm_charge_forces(mol, dm, c, q, zeta)).max() < 1e-12
    # translational invariance: the charge term of the atoms and the charge forces cancel
    assert abs(g_atoms.sum(0) + g_charges.sum(0)).max() < 1e-10


def test_mm_force_cache(monkeypatch):
    coords, q = _charges(200)
    mol = gto.M(atom=WATER, basis="cc-pvdz", verbose=0)
    mf = mojoscf.accelerate(qmmm.mm_charge(scf.RHF(mol), coords, q)).run(conv_tol=1e-10)
    g = mf.nuc_grad_method()
    g.kernel()
    calls = []
    real = integrals.mm_charge_forces
    monkeypatch.setattr(integrals, "mm_charge_forces", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    f = g.grad_hcore_mm(mf.make_rdm1())
    assert not calls                                   # from the gradient's pass
    dm2 = mf.make_rdm1() * 1.01
    f2 = g.grad_hcore_mm(dm2)
    assert calls                                       # another density: recomputed
    assert abs(f2 - 1.01 * f).max() < 1e-12
