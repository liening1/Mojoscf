"""Nuclear gradients and derivative integrals from the Mojo engine against pyscf."""
import numpy as np
import pytest
from pyscf import gto, scf
from pyscf.grad import rhf as rhf_grad

import mojoscf
from mojoscf import integrals as mi

H2O = "O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587"
H2O_DISTORTED = "O 0 0 0.05; H 0 0.76 0.58; H 0 -0.75 0.59"


def _mol(basis="cc-pvdz", atom=H2O, **kw):
    return gto.M(atom=atom, basis=basis, verbose=0, **kw)


@pytest.mark.parametrize(
    "mol",
    [
        _mol("sto-3g"),
        _mol("aug-cc-pvdz"),
        _mol("def2-svp", cart=True),
        _mol({"Ne": "cc-pvqz", "H": "cc-pvdz"}, atom="Ne 0 0 0; H 0 0 1.9", spin=1),
    ],
    ids=["sto-3g", "aug-cc-pvdz", "def2-svp-cart", "Ne-qz"],
)
def test_one_electron_derivatives(mol):
    s1, t1, v1 = mi.int1e_ip(mol)
    assert abs(s1 - mol.intor("int1e_ipovlp")).max() < 1e-13
    assert abs(t1 - mol.intor("int1e_ipkin")).max() < 1e-12
    assert abs(v1 - mol.intor("int1e_ipnuc")).max() < 1e-11
    for ia in range(mol.natm):
        with mol.with_rinv_at_nucleus(ia):
            ref = mol.intor("int1e_iprinv", comp=3)
        assert abs(mi.int1e_iprinv(mol, ia) - ref).max() < 1e-11


def test_derivative_jk_matches_pyscf():
    mol = _mol("cc-pvtz")
    rng = np.random.default_rng(7)
    dms = rng.standard_normal((2, mol.nao, mol.nao))
    dms = dms + dms.transpose(0, 2, 1)
    vj, vk = mi.get_jk_ip1(mol, dms)
    rj, rk = rhf_grad.get_jk(mol, dms)
    assert vj.shape == rj.shape == (2, 3, mol.nao, mol.nao)
    assert abs(vj - rj).max() < 1e-11
    assert abs(vk - rk).max() < 1e-11
    vj1, vk1 = mi.get_jk_ip1(mol, dms[0], with_k=False)
    assert vk1 is None and abs(vj1 - rj[0]).max() < 1e-11


def _pyscf_grad2e(mol, dm, unrestricted):
    vj, vk = rhf_grad.get_jk(mol, dm)
    vhf = vj[0] + vj[1] - vk if unrestricted else vj - 0.5 * vk
    de = np.zeros((mol.natm, 3))
    for ia, (p0, p1) in enumerate(mol.aoslice_by_atom()[:, 2:]):
        if unrestricted:
            de[ia] = 2 * np.einsum("sxij,sij->x", vhf[:, :, p0:p1], dm[:, p0:p1])
        else:
            de[ia] = 2 * np.einsum("xij,ij->x", vhf[:, p0:p1], dm[p0:p1])
    return de


def test_two_electron_gradient_term():
    mol = _mol("def2-tzvp", atom="C 0 0 0; O 0 0 1.13; H 0.9 0.3 -0.5", spin=1)
    dm = scf.UHF(mol).run().make_rdm1()
    de = mi.grad2e(mol, dm[0] + dm[1], dm)
    ref = _pyscf_grad2e(mol, dm, True)
    assert abs(de - ref).max() < 1e-11
    assert abs(de.sum(axis=0)).max() < 1e-11  # translational invariance
    mol = _mol("cc-pvtz")
    dm = scf.RHF(mol).run().make_rdm1()
    assert abs(mi.grad2e(mol, dm, dm, 1.0, 0.5) - _pyscf_grad2e(mol, dm, False)).max() < 1e-11


@pytest.mark.parametrize("basis", ["sto-3g", "cc-pvdz", "aug-cc-pvdz", "def2-tzvp"])
def test_rhf_gradient_matches_pyscf(basis):
    mol = _mol(basis, atom=H2O_DISTORTED)
    mf = mojoscf.RHF(mol).run(conv_tol=1e-12)
    g = mf.nuc_grad_method()
    assert type(g) is mojoscf.grad.Gradients
    ref = scf.RHF(mol).run(conv_tol=1e-12).nuc_grad_method().kernel()
    assert abs(g.kernel() - ref).max() < 1e-8
    assert abs(mf.Gradients().kernel() - ref).max() < 1e-8


def test_uhf_gradient_matches_pyscf():
    mol = _mol({"Ne": "cc-pvqz", "H": "cc-pvdz"}, atom="Ne 0 0 0; H 0 0 1.9", spin=1)
    mf = mojoscf.UHF(mol).run(conv_tol=1e-12)
    g = mf.nuc_grad_method()
    assert type(g) is mojoscf.grad.UGradients
    ref = scf.UHF(mol).run(conv_tol=1e-12).nuc_grad_method().kernel()
    assert abs(g.kernel() - ref).max() < 1e-8


def test_gradient_classes_on_pyscf_objects():
    """Same converged SCF: the gradients agree to integral precision."""
    mol = _mol("cc-pvdz", atom="C 0 0 0; O 0 0 1.13", cart=True)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    assert abs(mojoscf.grad.Gradients(mf).kernel() - mf.nuc_grad_method().kernel()).max() < 1e-11
    mol = _mol("cc-pvdz", atom="O 0 0 0; H 0 0 0.97", spin=1)
    mf = scf.UHF(mol).run(conv_tol=1e-12)
    assert abs(mojoscf.grad.UGradients(mf).kernel() - mf.nuc_grad_method().kernel()).max() < 1e-11


def test_partial_atom_list():
    mol = _mol("cc-pvdz", atom=H2O_DISTORTED)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    ref = mf.nuc_grad_method().kernel()
    assert abs(mojoscf.grad.Gradients(mf).kernel(atmlst=[0, 2]) - ref[[0, 2]]).max() < 1e-11


def test_accelerated_objects_and_fallbacks():
    mol = _mol("cc-pvdz")
    mf = mojoscf.accelerate(scf.RHF(mol))
    assert type(mf.nuc_grad_method()) is mojoscf.grad.Gradients
    # density fitting keeps pyscf's DF gradients, whichever way the object was made
    for mf in (mojoscf.RHF(mol).density_fit(), mojoscf.accelerate(scf.RHF(mol).density_fit())):
        mf.run(conv_tol=1e-12)
        g = mf.nuc_grad_method()
        assert not isinstance(g, mojoscf.grad.Gradients)
        ref = scf.RHF(mol).density_fit().run(conv_tol=1e-12).nuc_grad_method().kernel()
        assert abs(g.kernel() - ref).max() < 1e-8
    # an engine-unsupported molecule (ECP) runs pyscf's code inside the Mojo classes
    mol = gto.M(atom="Cu 0 0 0; H 0 0 1.5", basis={"Cu": "lanl2dz", "H": "sto-3g"}, ecp={"Cu": "lanl2dz"}, verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    g = mojoscf.grad.Gradients(mf)
    assert not g._mojo_ok()
    assert abs(g.kernel() - mf.nuc_grad_method().kernel()).max() < 1e-11


def test_gradient_get_jk_general_densities():
    """pyscf's TDHF gradients pass antisymmetric densities to get_jk."""
    mol = _mol("cc-pvdz")
    g = mojoscf.grad.Gradients(scf.RHF(mol))
    dm = np.random.default_rng(3).standard_normal((mol.nao, mol.nao))
    for d in (dm - dm.T, dm + dm.T):
        vj, vk = g.get_jk(mol, d)
        rj, rk = rhf_grad.get_jk(mol, d)
        assert abs(vj - rj).max() < 1e-11 and abs(vk - rk).max() < 1e-11
        assert abs(g.get_j(mol, d) - rj).max() < 1e-11
        assert abs(g.get_k(mol, d) - rk).max() < 1e-11


def test_tdhf_gradient_through_mojo_scf():
    from pyscf import tdscf

    mol = _mol("cc-pvdz")
    td = tdscf.TDHF(mojoscf.RHF(mol).run(conv_tol=1e-12)).run(nstates=2)
    ref = tdscf.TDHF(scf.RHF(mol).run(conv_tol=1e-12)).run(nstates=2)
    assert abs(td.nuc_grad_method().kernel(state=1) - ref.nuc_grad_method().kernel(state=1)).max() < 1e-7


def test_gradient_scanner_follows_geometry():
    mol = _mol("cc-pvdz")
    mf = mojoscf.RHF(mol)
    mf.conv_tol = 1e-12
    scanner = mf.nuc_grad_method().as_scanner()
    ref = scf.RHF(mol)
    ref.conv_tol = 1e-12
    ref_scanner = ref.nuc_grad_method().as_scanner()
    for geom in (mol, _mol("cc-pvdz", atom=H2O_DISTORTED), mol):
        e, g = scanner(geom)
        e0, g0 = ref_scanner(geom)
        assert abs(e - e0) < 1e-9
        assert abs(g - g0).max() < 1e-8
