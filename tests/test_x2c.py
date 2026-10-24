"""Spin-free X2C one-electron gradient terms (mojoscf.x2c) against pyscf's sfx2c1e_grad."""
import numpy as np
import pytest
from pyscf import dft, gto, scf
from pyscf.x2c import sfx2c1e, sfx2c1e_grad

import mojoscf
from mojoscf import x2c

CUCLH = "Cu 0 0 0; Cl 0 0 2.1; H 0.3 1.6 -0.4"


def _mol():
    return gto.M(atom=CUCLH, basis="def2-svp", spin=1, verbose=0)


def _reference(x2cobj, mol, dms):
    """pyscf's per-atom derivative matrices contracted with each density."""
    gen = sfx2c1e_grad.hcore_grad_generator(x2cobj, mol)
    h1 = [gen(ia) for ia in range(mol.natm)]
    return np.array([[np.einsum("xij,ij->x", h, dm) for h in h1] for dm in dms])


@pytest.mark.parametrize(
    "opts", [{}, {"approx": "atom1e"}, {"xuncontract": False}, {"basis": "cc-pvtz-dk"}],
    ids=["1e", "atom1e", "contracted", "projected"],
)
def test_hcore_grad_dm_matches_pyscf(opts):
    """Every X2C variant of pyscf's generator (full 1e or atomic X, uncontracted or not, a projection basis)."""
    mol = _mol()
    mf = scf.UHF(mol).x2c()
    for key, val in opts.items():
        setattr(mf.with_x2c, key, val)
    assert x2c.grad_ok(mf.with_x2c)
    dms = np.random.default_rng(7).standard_normal((2, mol.nao, mol.nao))
    ref = _reference(mf.with_x2c, mol, dms)
    got = x2c.hcore_grad_dm(mf.with_x2c, mol, dms)
    assert got.shape == (2, mol.natm, 3)
    assert abs(got - ref).max() < 1e-9
    assert abs(x2c.hcore_grad_dm(mf.with_x2c, mol, dms[0]) - ref[0]).max() < 1e-9


def test_x2c_gradients_match_pyscf():
    """UHF, UKS and DF-UKS with X2C: the accelerated gradient classes on the same converged SCF."""
    mol = _mol()
    for make, cls in (
        (lambda: scf.UHF(mol).x2c(), mojoscf.grad.UGradients),
        (lambda: dft.UKS(mol, xc="b3lyp").x2c(), mojoscf.grad.UKSGradients),
        (lambda: dft.UKS(mol, xc="pbe").x2c().density_fit(), mojoscf.grad.DFUKSGradients),
    ):
        mf = make().run(conv_tol=1e-11)
        ref = mf.nuc_grad_method().kernel()
        acc = mojoscf.dft.accelerate(mf) if isinstance(mf, dft.rks.KohnShamDFT) else mojoscf.accelerate(mf)
        g = acc.nuc_grad_method()
        assert type(g) is cls
        assert abs(g.kernel() - ref).max() < 1e-9
        assert abs(g.kernel(atmlst=[0, 2]) - ref[[0, 2]]).max() < 1e-9


def test_hcore_deriv_dm_falls_back_to_the_generator():
    """A helper whose generator is overridden, and a gradient class with its own hcore_generator, keep the
    matrices of the generator."""

    class Helper(sfx2c1e.SpinFreeX2CHelper):
        def hcore_deriv_generator(self, mol=None, deriv=1):
            gen = super().hcore_deriv_generator(mol, deriv)
            return lambda ia: 2 * gen(ia)

    mol = _mol()
    mf = scf.UHF(mol).x2c()
    mf.with_x2c = Helper(mol)
    assert not x2c.grad_ok(mf.with_x2c)
    dm = np.random.default_rng(3).standard_normal((mol.nao, mol.nao))
    g = mojoscf.grad.UGradients(mf)
    ref = 2 * _reference(sfx2c1e.SpinFreeX2CHelper(mol), mol, [dm])[0]
    assert abs(mojoscf.grad.hcore_deriv_dm(g, mol, dm) - ref).max() < 1e-9
