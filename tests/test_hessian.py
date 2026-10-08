"""Analytical Hessians (mojoscf.hessian) against pyscf.hessian."""
import numpy as np
import pytest
from pyscf import dft, gto
from pyscf.hessian import rks as rks_hess
from pyscf.hessian import uks as uks_hess

import mojoscf
from mojoscf import dft as mdft
from mojoscf import hessian as mhess

WATER = "O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587"


def _pyscf_xc_partial(mf):
    """The XC part of pyscf's partial_hess_elec: _get_vxc_diag and _get_vxc_deriv2 contracted with D."""
    mol = mf.mol
    h = mf.Hessian()
    mo, occ = mf.mo_coeff, mf.mo_occ
    unrestricted = np.asarray(mo).ndim == 3
    if unrestricted:
        dms = mf.make_rdm1()
        diag = uks_hess._get_vxc_diag(h, mo, occ, 4000)
        vxc = uks_hess._get_vxc_deriv2(h, mo, occ, 4000)
    else:
        dms = [mf.make_rdm1()]
        diag = [rks_hess._get_vxc_diag(h, mo, occ, 4000)]
        vxc = [rks_hess._get_vxc_deriv2(h, mo, occ, 4000)]
    de2 = np.zeros((mol.natm, mol.natm, 3, 3))
    sl = mol.aoslice_by_atom()
    for dm, vd, vx in zip(dms, diag, vxc):
        for i in range(mol.natm):
            p0, p1 = sl[i][2:]
            de2[i, i] += np.einsum("xypq,pq->xy", vd[:, :, p0:p1], dm[p0:p1]) * 2
            for j in range(i + 1):
                q0, q1 = sl[j][2:]
                de2[i, j] += np.einsum("xypq,pq->xy", vx[i][:, :, q0:q1], dm[q0:q1]) * 2
    for i in range(mol.natm):
        for j in range(i):
            de2[j, i] = de2[i, j].T
    return de2


@pytest.mark.parametrize("xc, spin", [("lda,vwn", 0), ("pbe", 0), ("b3lyp", 0), ("pbe", 1), ("b3lyp", 1)])
def test_xc_partial_hessian_matches_pyscf(xc, spin):
    mol = gto.M(atom=WATER, basis="def2-svp", charge=spin, spin=spin, verbose=0)
    mf = (dft.UKS if spin else dft.RKS)(mol, xc=xc).run()
    ref = _pyscf_xc_partial(mf)
    de2 = mhess.xc_partial_hess(mdft.NumInt(), mol, mf.grids, xc, mf.mo_coeff, mf.mo_occ)
    assert abs(de2 - ref).max() < 1e-10


def test_xc_partial_hessian_rejects_meta_gga(water=None):
    mol = gto.M(atom=WATER, basis="sto-3g", verbose=0)
    mf = dft.RKS(mol, xc="tpss").run()
    assert mhess.xc_partial_hess(mdft.NumInt(), mol, mf.grids, "tpss", mf.mo_coeff, mf.mo_occ) is None


@pytest.mark.parametrize("xc, spin, df", [("b3lyp", 0, True), ("pbe", 0, False), ("camb3lyp", 0, True),
                                          ("pbe", 1, True), ("b3lyp", 1, False), ("tpss", 0, True)])
def test_accelerated_hessian_matches_pyscf(xc, spin, df):
    mol = gto.M(atom=WATER, basis="def2-svp", charge=spin, spin=spin, verbose=0)

    def make():
        mf = (dft.UKS if spin else dft.RKS)(mol, xc=xc)
        return mf.density_fit() if df else mf

    mf = mojoscf.dft.accelerate(make())
    mf.conv_tol = 1e-11
    mf.kernel()
    ref = make()
    for key in ("mo_coeff", "mo_occ", "mo_energy", "e_tot", "converged"):
        setattr(ref, key, getattr(mf, key))
    h = mf.Hessian()
    assert isinstance(h, mhess._MojoHessMixin)
    assert abs(h.kernel() - ref.Hessian().kernel()).max() < 1e-9
