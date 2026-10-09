"""Excited-state gradients (mojoscf.tdgrad) against pyscf.grad.tdrks/tduks/tdrhf on the same TD solution."""
import numpy as np
import pytest
from pyscf import dft, gto
from pyscf.grad import rhf as rhf_grad
from pyscf.grad import tdrhf, tdrks, tduks

import mojoscf
import mojoscf.tdscf
from mojoscf import grad as mgrad
from mojoscf import tdgrad

WATER = "O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587"


def _td(mf, method, singlet=True, nstates=3):
    td = getattr(mf, method)()
    td.singlet = singlet
    td.nstates = nstates
    td.conv_tol = 1e-10
    td.kernel()
    return td


def test_derivative_jk_of_nonsymmetric_densities():
    mol = gto.M(atom=WATER, basis="def2-svp", verbose=0)
    rng = np.random.default_rng(0)
    d = rng.normal(size=(3, mol.nao, mol.nao))
    d[0] = d[0] + d[0].T            # symmetric
    d[1] = d[1] - d[1].T            # antisymmetric
    vj, vk = mgrad._jk_ip1(mol, d)
    rj, rk = rhf_grad.get_jk(mol, d)
    assert abs(vj - rj).max() < 1e-11
    assert abs(vk - rk).max() < 1e-11
    assert abs(mgrad._jk_ip1(mol, d, with_k=False)[0] - rj).max() < 1e-11
    assert abs(mgrad._jk_ip1(mol, d[2], with_j=False)[1] - rk[2]).max() < 1e-11


@pytest.mark.parametrize("xc, method, singlet", [("b3lyp", "TDDFT", True), ("pbe", "TDA", True),
                                                 ("lda,vwn", "TDA", True), ("b3lyp", "TDDFT", False),
                                                 ("pbe", "TDA", False), ("camb3lyp", "TDDFT", True)])
def test_rks_td_gradients_match_pyscf(xc, method, singlet):
    mol = gto.M(atom=WATER, basis="def2-svp", verbose=0)
    mf = mojoscf.dft.accelerate(dft.RKS(mol, xc=xc))
    mf.conv_tol = 1e-11
    mf.kernel()
    td = _td(mf, method, singlet)
    g = td.Gradients()
    assert isinstance(g, tdgrad._MojoTDGradMixin)
    de = g.kernel(state=2)
    ref = tdrks.Gradients(td).kernel(state=2)
    assert abs(de - ref).max() < 1e-11
    # the native XC contraction ran
    res = tdgrad.tdrks_xc_kernel(g, mf.xc, np.eye(mol.nao), np.eye(mol.nao), singlet=singlet)
    assert res is not None and res[0].shape == (4, mol.nao, mol.nao)


@pytest.mark.parametrize("xc, method", [("b3lyp", "TDDFT"), ("pbe", "TDA")])
def test_uks_td_gradients_match_pyscf(xc, method):
    mol = gto.M(atom=WATER, basis="def2-svp", charge=1, spin=1, verbose=0)
    mf = mojoscf.dft.accelerate(dft.UKS(mol, xc=xc))
    mf.conv_tol = 1e-11
    mf.kernel()
    td = _td(mf, method)
    de = td.Gradients().kernel(state=2)
    ref = tduks.Gradients(td).kernel(state=2)
    assert abs(de - ref).max() < 1e-11


def test_tdhf_gradients_match_pyscf():
    mol = gto.M(atom=WATER, basis="def2-svp", verbose=0)
    mf = mojoscf.RHF(mol)
    mf.conv_tol = 1e-11
    mf.kernel()
    td = mojoscf.tdscf.TDDFT(mf)
    td.nstates = 3
    td.conv_tol = 1e-10
    td.kernel()
    g = td.Gradients()
    assert isinstance(g, tdgrad._MojoTDGradMixin)
    assert abs(g.kernel(state=1) - tdrhf.Gradients(td).kernel(state=1)).max() < 1e-11


def test_meta_gga_keeps_pyscf_xc_contraction():
    mol = gto.M(atom=WATER, basis="sto-3g", verbose=0)
    mf = mojoscf.dft.accelerate(dft.RKS(mol, xc="tpss")).run()
    td = _td(mf, "TDA", nstates=2)
    g = td.Gradients()
    assert tdgrad.tdrks_xc_kernel(g, mf.xc, np.eye(mol.nao)) is None
    assert abs(g.kernel(state=1) - tdrks.Gradients(td).kernel(state=1)).max() < 1e-11
