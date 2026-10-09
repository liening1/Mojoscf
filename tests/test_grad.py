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


@pytest.mark.parametrize("basis", ["cc-pvdz", "def2-tzvp"])
def test_long_range_derivative_jk_and_gradient_term(basis):
    """erf(omega r) / r: derivative J/K and the two-electron gradient term against pyscf's range_coulomb."""
    mol = _mol(basis, atom="C 0 0 0; O 0 0 1.13; H 0.9 0.3 -0.5", spin=1)
    rng = np.random.default_rng(3)
    dms = rng.standard_normal((2, mol.nao, mol.nao))
    dms[0] = dms[0] + dms[0].T
    with mol.with_range_coulomb(0.33):
        rj, rk = rhf_grad.get_jk(mol, dms)
    vj, vk = mi.get_jk_ip1(mol, np.ascontiguousarray(dms[:1]), omega=0.33)
    assert abs(vj - rj[:1]).max() < 1e-11 and abs(vk - rk[:1]).max() < 1e-11
    assert abs(mi.get_jk_ip1(mol, dms, with_j=False, omega=0.33)[1] - rk).max() < 1e-11
    dm = scf.UHF(mol).run().make_rdm1()
    with mol.with_range_coulomb(0.33):
        ref = _pyscf_grad2e(mol, dm, True)
    de = mi.grad2e(mol, dm[0] + dm[1], dm, omega=0.33)
    assert abs(de - ref).max() < 1e-11
    assert abs(de.sum(axis=0)).max() < 1e-11


def _first_index_terms(mol, x, y, kind):
    """sum over atoms A of sum_(i on A) sum_j V'[x]_ij y_ij with V' pyscf's derivative J (or K) matrix."""
    vj, vk = rhf_grad.get_jk(mol, np.array([x]))
    v = (vj if kind == "j" else vk)[0]
    out = np.zeros((mol.natm, 3))
    for ia, (p0, p1) in enumerate(mol.aoslice_by_atom()[:, 2:]):
        out[ia] = np.einsum("xij,ij->x", v[:, p0:p1], y[p0:p1])
    return out


def test_pair_gradient_kernel():
    """d/dR of sum (ij|kl) L_ij R_kl and sum (ij|kl) A_jk B_il from the four first-index contractions."""
    mol = _mol("def2-svp", atom="C 0 0 0; O 0 0 1.13; H 0.9 0.3 -0.5", spin=1)
    rng = np.random.default_rng(5)
    n = mol.nao
    sym = [m + m.T for m in rng.standard_normal((2, n, n))]
    a = rng.standard_normal((n, n))
    anti = a - a.T
    l, r = sym
    ref_j = 2 * (_first_index_terms(mol, r, l, "j") + _first_index_terms(mol, l, r, "j"))
    ref_k = sum(_first_index_terms(mol, x, y, "k") for x, y in ((l, r), (r, l), (r.T, l.T), (l.T, r.T)))
    ref_m = 4 * _first_index_terms(mol, anti, anti, "k")
    de = mi.grad2e_pairs(mol, [(0.7, l, r)], [(-0.3, l, r), (0.2, anti, anti)])
    assert abs(de - (0.7 * ref_j - 0.3 * ref_k + 0.2 * ref_m)).max() < 1e-10
    assert abs(de.sum(axis=0)).max() < 1e-10
    # the single-pair case is grad2e
    dm = scf.RHF(_mol("cc-pvdz")).run().make_rdm1()
    mol2 = _mol("cc-pvdz")
    assert abs(mi.grad2e_pairs(mol2, [(0.5, dm, dm)], [(-0.25, dm, dm)]) - mi.grad2e(mol2, dm, dm, 1.0, 0.5)).max() < 1e-12
    # long-range operator
    with mol.with_range_coulomb(0.4):
        ref_lr = sum(_first_index_terms(mol, x, y, "k") for x, y in ((l, r), (r, l), (r.T, l.T), (l.T, r.T)))
    assert abs(mi.grad2e_pairs(mol, (), [(1.0, l, r)], omega=0.4) - ref_lr).max() < 1e-10


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
    # density fitting: the DF gradient classes, whichever way the object was made
    ref = scf.RHF(mol).density_fit().run(conv_tol=1e-12).nuc_grad_method().kernel()
    for mf in (
        mojoscf.RHF(mol).density_fit(),
        mojoscf.accelerate(scf.RHF(mol).density_fit()),
        mojoscf.accelerate(scf.RHF(mol)).density_fit(),
    ):
        mf.run(conv_tol=1e-12)
        g = mf.nuc_grad_method()
        assert type(g) is mojoscf.grad.DFGradients and g._direct_2e()
        assert abs(g.kernel() - ref).max() < 1e-8
    assert type(mojoscf.RHF(mol).density_fit().undo_df().nuc_grad_method()) is mojoscf.grad.Gradients
    # with an ECP only the ECP derivative integrals come from pyscf
    mol = gto.M(atom="Cu 0 0 0; H 0 0 1.5", basis={"Cu": "lanl2dz", "H": "sto-3g"}, ecp={"Cu": "lanl2dz"}, verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    g = mojoscf.grad.Gradients(mf)
    assert g._mojo_ok() and g._mojo_2e_ok() and g._direct_2e()
    ref = mf.nuc_grad_method()
    assert abs(g.get_hcore() - ref.get_hcore()).max() < 1e-11
    assert abs(g.hcore_generator()(0) - ref.hcore_generator()(0)).max() < 1e-11
    assert abs(g.kernel() - mf.nuc_grad_method().kernel()).max() < 1e-11
    # range separation is not supported at all: pyscf's code throughout
    mol = _mol("sto-3g")
    mol.omega = 0.3
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    g = mojoscf.grad.Gradients(mf)
    assert not g._mojo_ok() and not g._mojo_2e_ok()
    assert abs(g.kernel() - mf.nuc_grad_method().kernel()).max() < 1e-11


def test_ecp_gradients_match_pyscf():
    """Pt and Ag with def2 ECPs: exact and DF, RHF and UHF, on the same SCF objects."""
    mol = gto.M(
        atom="Pt 0 0 0; N 2.05 0 0; Cl 0 2.32 0; H 2.4 0.95 0; H 2.4 -0.48 0.83; H 2.4 -0.48 -0.83",
        basis="def2-svp", ecp={"Pt": "def2-svp"}, charge=1, verbose=0,
    )
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    assert abs(mojoscf.grad.Gradients(mf).kernel() - mf.nuc_grad_method().kernel()).max() < 1e-10
    mf = scf.RHF(mol).density_fit().run(conv_tol=1e-12)
    g = mojoscf.grad.DFGradients(mf)
    assert g._direct_2e()
    assert abs(g.kernel() - mf.nuc_grad_method().kernel()).max() < 1e-10
    mol = gto.M(atom="Ag 0 0 0; H 0 0 1.62", basis="def2-svp", ecp={"Ag": "def2-svp"}, charge=1, spin=1, verbose=0)
    mf = scf.UHF(mol).run(conv_tol=1e-12)
    assert mf.converged
    assert abs(mojoscf.grad.UGradients(mf).kernel() - mf.nuc_grad_method().kernel()).max() < 1e-10
    mf = scf.UHF(mol).density_fit().run(conv_tol=1e-12)
    assert abs(mojoscf.grad.DFUGradients(mf).kernel() - mf.nuc_grad_method().kernel()).max() < 1e-10


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


def _pyscf_df_grad2e(g, mol, dm, unrestricted):
    vhf = g.get_veff(mol, dm)
    de = np.zeros((mol.natm, 3))
    for ia, (p0, p1) in enumerate(mol.aoslice_by_atom()[:, 2:]):
        if unrestricted:
            de[ia] = 2 * np.einsum("sxij,sij->x", vhf[:, :, p0:p1], dm[:, p0:p1])
        else:
            de[ia] = 2 * np.einsum("xij,ij->x", vhf[:, p0:p1], dm[p0:p1])
    return de + vhf.aux


def test_density_fitted_two_electron_term():
    """grad2e_df against pyscf's DF J/K gradient plus its auxiliary-basis response."""
    mol = _mol("def2-svp", atom="C 0 0 0; O 0 0 1.13; H 0.9 0.3 -0.5", spin=1)
    mf = scf.UHF(mol).density_fit().run(conv_tol=1e-11)
    dm = mf.make_rdm1()
    orbs = [mf.mo_coeff[s][:, mf.mo_occ[s] > 0] for s in range(2)]
    occs = [mf.mo_occ[s][mf.mo_occ[s] > 0] for s in range(2)]
    ref = _pyscf_df_grad2e(mf.nuc_grad_method(), mol, dm, True)
    de = mi.grad2e_df(mol, mf.with_df.auxmol, dm[0] + dm[1], orbs, occs, 1.0, 1.0)
    assert abs(de - ref).max() < 1e-11
    assert abs(de.sum(axis=0)).max() < 1e-11  # translational invariance
    # many small auxiliary blocks give the same result
    small = mi.grad2e_df(mol, mf.with_df.auxmol, dm[0] + dm[1], orbs, occs, 1.0, 1.0, max_memory=0.02)
    assert abs(small - de).max() < 1e-12
    mol = _mol("aug-cc-pvtz")
    mf = scf.RHF(mol).density_fit().run(conv_tol=1e-11)
    dm = mf.make_rdm1()
    occ = mf.mo_occ > 0
    ref = _pyscf_df_grad2e(mf.nuc_grad_method(), mol, dm, False)
    de = mi.grad2e_df(mol, mf.with_df.auxmol, dm, [mf.mo_coeff[:, occ]], [mf.mo_occ[occ]], 1.0, 0.5)
    assert abs(de - ref).max() < 1e-11


@pytest.mark.parametrize("basis", ["cc-pvdz", "def2-tzvp"])
def test_df_rhf_gradient_matches_pyscf(basis):
    mol = _mol(basis, atom=H2O_DISTORTED)
    mf = scf.RHF(mol).density_fit().run(conv_tol=1e-12)
    ref = mf.nuc_grad_method().kernel()
    g = mojoscf.grad.DFGradients(mf)
    assert g._direct_2e()
    assert abs(g.kernel() - ref).max() < 1e-11
    assert abs(mojoscf.RHF(mol).density_fit().run(conv_tol=1e-12).nuc_grad_method().kernel() - ref).max() < 1e-8


def test_df_uhf_gradient_matches_pyscf():
    mol = _mol({"Ne": "cc-pvqz", "H": "cc-pvdz"}, atom="Ne 0 0 0; H 0 0 1.9", spin=1)
    mf = scf.UHF(mol).density_fit().run(conv_tol=1e-12)
    ref = mf.nuc_grad_method().kernel()
    assert abs(mojoscf.grad.DFUGradients(mf).kernel() - ref).max() < 1e-11
    g = mojoscf.UHF(mol).density_fit().run(conv_tol=1e-12).nuc_grad_method()
    assert type(g) is mojoscf.grad.DFUGradients
    assert abs(g.kernel() - ref).max() < 1e-8


def test_df_gradient_options_use_pyscf():
    """auxbasis_response = False and only_dfj keep pyscf's two-electron code (with Mojo 1e integrals)."""
    mol = _mol("cc-pvdz", atom=H2O_DISTORTED)
    mf = scf.RHF(mol).density_fit().run(conv_tol=1e-12)
    g, g0 = mojoscf.grad.DFGradients(mf), mf.nuc_grad_method()
    g.auxbasis_response = g0.auxbasis_response = False
    assert not g._direct_2e()
    assert abs(g.kernel() - g0.kernel()).max() < 1e-11
    mf = mojoscf.RHF(mol).density_fit(only_dfj=True).run(conv_tol=1e-12)
    ref = scf.RHF(mol).density_fit(only_dfj=True).run(conv_tol=1e-12).nuc_grad_method().kernel()
    g = mf.nuc_grad_method()
    assert not g._direct_2e()
    assert abs(g.kernel() - ref).max() < 1e-8


def test_df_gradient_scanner_follows_geometry():
    mol = _mol("cc-pvdz")
    mf = mojoscf.RHF(mol).density_fit()
    mf.conv_tol = 1e-12
    scanner = mf.nuc_grad_method().as_scanner()
    ref = scf.RHF(mol).density_fit()
    ref.conv_tol = 1e-12
    ref_scanner = ref.nuc_grad_method().as_scanner()
    for geom in (mol, _mol("cc-pvdz", atom=H2O_DISTORTED), mol):
        e, g = scanner(geom)
        e0, g0 = ref_scanner(geom)
        assert abs(e - e0) < 1e-9
        assert abs(g - g0).max() < 1e-8


def test_gradients_hooks_of_all_classes():
    """mf.Gradients() and mf.nuc_grad_method() give the Mojo gradients (pyscf binds Gradients to its own
    classes, the DF one included)."""
    from pyscf import dft

    from mojoscf import grad as mgrad

    mol = gto.M(atom="H 0 0 0; F 0 0 1", basis="sto-3g", verbose=0)
    objs = [mojoscf.RHF(mol), mojoscf.UHF(mol), mojoscf.RHF(mol).density_fit(), mojoscf.UHF(mol).density_fit(),
            mojoscf.dft.accelerate(dft.RKS(mol)), mojoscf.dft.accelerate(dft.UKS(mol).density_fit())]
    for mf in objs:
        for g in (mf.Gradients(), mf.nuc_grad_method()):
            assert type(g).__module__ == mgrad.__name__, type(mf).__name__
