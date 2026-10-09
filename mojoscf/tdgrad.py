"""Excited-state nuclear gradients (pyscf.grad.tdrks/tdrhf: TDA, TDDFT, TDHF) with the Mojo kernels.

pyscf's linear-response gradient spends its time in the exchange-correlation
kernel contractions (``_contract_xc_kernel``: Python loops over grid blocks
with AO values and their derivatives, up to the third functional derivative
for the transition density), in the derivative Coulomb/exchange matrices of
four densities (``td_grad.get_jk``, libcvhf direct) and in the Z-vector
equations.  This module keeps pyscf's driver (``grad_elec``) and replaces:

* ``_contract_xc_kernel`` by :func:`tdrks_xc_kernel`: the ground-state and
  response densities from one Mojo pass, libxc for vxc/fxc/kxc, and the
  potential and gradient matrices of the four weight sets from one pass each
  (LDA and GGA; meta-GGA keeps pyscf's code);
* ``get_jk``/``get_j``/``get_k`` by the Mojo derivative J/K matrices
  (exact integrals; :func:`mojoscf.grad._jk_ip1`, non-symmetric densities
  included);
* the Z-vector response runs through the accelerated SCF object's
  ``gen_response`` (Mojo J/K and XC kernels) as before.
"""
from __future__ import annotations

import numpy as np

from . import integrals


def tdrks_xc_kernel(td_grad, xc_code, dmvo, dmoo=None, with_vxc=True, with_kxc=True, singlet=True,
                    max_memory=2000):
    """pyscf's ``grad.tdrks._contract_xc_kernel`` from the Mojo kernels, or None (pyscf's code runs).

    Returns ``(f1vo, f1oo, v1ao, k1ao)``, each (4, nao, nao) or None: the XC
    matrix of the weights (component 0) and minus the gradient matrices
    (components 1-3) of fxc . rho1 (rho1 of the symmetrised ``dmvo``),
    fxc . rho2 (``dmoo``), vxc and kxc . rho1 . rho1, singlet or triplet
    coupling as pyscf forms them.
    """
    from . import dft

    mf = td_grad.base._scf
    mol = td_grad.mol
    ni = mf._numint
    if not isinstance(ni, dft.NumInt):
        return None
    kind = dft._kind(ni, xc_code)
    if kind not in (0, 1) or not dft._fxc_ok(ni, mol, xc_code) or mf.do_nlc():
        return None
    mo_coeff, mo_occ = np.asarray(mf.mo_coeff), np.asarray(mf.mo_occ)
    if mo_coeff.ndim != 2 or not np.isrealobj(mo_coeff):
        return None
    xctype = ni._xc_type(xc_code)
    coords, weights = dft._grid(mf.grids)
    nvar = (1, 4)[kind]
    dmvo = 0.5 * (np.asarray(dmvo) + np.asarray(dmvo).T)
    deriv = 3 if with_kxc else 2
    rho0 = dft._rho_orbitals(mol, coords, kind, [mo_coeff], [mo_occ])[0]           # (nvar, ngrid)
    if singlet:
        vxc, fxc, kxc = ni.eval_xc_eff(xc_code, rho0[0] if kind == 0 else rho0, deriv, xctype=xctype)[1:]
        f_vo = f_oo = np.asarray(fxc).reshape(nvar, nvar, -1)
        vxc = np.asarray(vxc).reshape(nvar, -1)
        k_vo = np.asarray(kxc).reshape(nvar, nvar, nvar, -1) if with_kxc else None
        scale = 2.0                     # alpha + beta
    else:
        r = 0.5 * (rho0[0] if kind == 0 else rho0)
        vxc, fxc, kxc = ni.eval_xc_eff(xc_code, np.array([r, r]), deriv, xctype=xctype, spin=1)[1:]
        fxc = np.asarray(fxc).reshape(2, nvar, 2, nvar, -1)
        f_vo = (fxc[:, :, 0] - fxc[:, :, 1])
        f_vo = f_vo[0] - f_vo[1]
        f_oo = fxc[0, :, 0] + fxc[0, :, 1]
        vxc = np.asarray(vxc).reshape(2, nvar, -1)[0]
        if with_kxc:
            kxc = np.asarray(kxc).reshape(2, nvar, 2, nvar, 2, nvar, -1)
            k_vo = kxc[0, :, 0] - kxc[0, :, 1]
            k_vo = k_vo[:, :, 0] - k_vo[:, :, 1]
        else:
            k_vo = None
        scale = 1.0
    dms = [dmvo] if dmoo is None else [dmvo, np.asarray(dmoo)]
    rhos = dft._rho(mol, coords, kind, np.ascontiguousarray(np.array(dms), dtype=np.float64)) * scale
    rho1 = rhos[0]
    wvs = [np.einsum("yg,xyg->xg", rho1, f_vo) * weights]
    if dmoo is not None:
        wvs.append(np.einsum("yg,xyg->xg", rhos[1], f_oo) * weights)
    if with_vxc:
        wvs.append(vxc * weights)
    if with_kxc:
        wvs.append(np.einsum("yg,zg,xyzg->xg", rho1, rho1, k_vo) * weights)
    it = iter(_xc_matrices(mol, coords, kind, wvs))
    f1vo = next(it)
    f1oo = next(it) if dmoo is not None else None
    v1ao = next(it) if with_vxc else None
    k1ao = next(it) if with_kxc else None
    return f1vo, f1oo, v1ao, k1ao


def _xc_matrices(mol, coords, kind, wvs):
    """[(4, nao, nao)] per weight set (nvar, ngrid): the XC matrix and minus the gradient matrices, as pyscf's
    ``_lda_eval_mat_``/``_gga_eval_mat_`` accumulate them (with the final sign flip), from one pass each."""
    from . import dft

    nao = mol.nao_nr()
    wv = np.ascontiguousarray(np.array(wvs, dtype=np.float64))               # (nset, nvar, ngrid)
    half = wv.copy()
    half[:, 0] *= 0.5
    v0 = dft._vmat(mol, coords, kind, half)                                      # symmetrised XC matrices
    vg = dft._xc_grad(mol, coords, kind == 1, half if kind == 1 else wv)        # gradient matrices
    out = []
    for v, g in zip(v0, vg):
        m = np.empty((4, nao, nao))
        m[0] = v
        m[1:] = -g
        out.append(m)
    return out


def tduks_xc_kernel(td_grad, xc_code, dmvo, dmoo=None, with_vxc=True, with_kxc=True, max_memory=2000):
    """pyscf's ``grad.tduks._contract_xc_kernel`` from the Mojo kernels, or None (pyscf's code runs).

    As :func:`tdrks_xc_kernel` with the spin-resolved kernels: each output
    (2, 4, nao, nao), per spin.
    """
    from . import dft

    mf = td_grad.base._scf
    mol = td_grad.mol
    ni = mf._numint
    if not isinstance(ni, dft.NumInt):
        return None
    kind = dft._kind(ni, xc_code)
    if kind not in (0, 1) or not dft._fxc_ok(ni, mol, xc_code) or mf.do_nlc():
        return None
    mo_coeff, mo_occ = np.asarray(mf.mo_coeff), np.asarray(mf.mo_occ)
    if mo_coeff.ndim != 3 or not np.isrealobj(mo_coeff):
        return None
    xctype = ni._xc_type(xc_code)
    coords, weights = dft._grid(mf.grids)
    nvar = (1, 4)[kind]
    deriv = 3 if with_kxc else 2
    rho0 = dft._rho_orbitals(mol, coords, kind, list(mo_coeff), list(mo_occ))          # (2, nvar, ngrid)
    r = (rho0[0, 0], rho0[1, 0]) if kind == 0 else (rho0[0], rho0[1])
    vxc, fxc, kxc = ni.eval_xc_eff(xc_code, r, deriv, xctype=xctype, spin=1)[1:]
    vxc = np.asarray(vxc).reshape(2, nvar, -1)
    fxc = np.asarray(fxc).reshape(2, nvar, 2, nvar, -1)
    dms = [0.5 * (np.asarray(d) + np.asarray(d).T) for d in dmvo]
    if dmoo is not None:
        dms += [np.asarray(d) for d in dmoo]
    rhos = dft._rho(mol, coords, kind, np.ascontiguousarray(np.array(dms), dtype=np.float64))
    rho1 = rhos[:2]
    wvs = list(np.einsum("axg,axbyg->byg", rho1, fxc) * weights)
    if dmoo is not None:
        wvs += list(np.einsum("axg,axbyg->byg", rhos[2:4], fxc) * weights)
    if with_vxc:
        wvs += list(vxc * weights)
    if with_kxc:
        kxc = np.asarray(kxc).reshape(2, nvar, 2, nvar, 2, nvar, -1)
        wvs += list(np.einsum("axg,byg,axbyczg->czg", rho1, rho1, kxc) * weights)
    mats = _xc_matrices(mol, coords, kind, wvs)
    it = iter([np.array(mats[k: k + 2]) for k in range(0, len(mats), 2)])
    f1vo = next(it)
    f1oo = next(it) if dmoo is not None else None
    v1ao = next(it) if with_vxc else None
    k1ao = next(it) if with_kxc else None
    return f1vo, f1oo, v1ao, k1ao


class _XCKernel:
    """Within the block, pyscf's ``grad.tdrks``/``tduks`` ``_contract_xc_kernel`` try the Mojo versions first."""

    def __enter__(self):
        from pyscf.grad import tdrks, tduks

        self.saved = (tdrks._contract_xc_kernel, tduks._contract_xc_kernel)
        rks_saved, uks_saved = self.saved

        def rks_kernel(td_grad, xc_code, dmvo, dmoo=None, with_vxc=True, with_kxc=True, singlet=True,
                       max_memory=2000):
            res = tdrks_xc_kernel(td_grad, xc_code, dmvo, dmoo, with_vxc, with_kxc, singlet, max_memory)
            if res is None:
                return rks_saved(td_grad, xc_code, dmvo, dmoo, with_vxc, with_kxc, singlet, max_memory)
            return res

        def uks_kernel(td_grad, xc_code, dmvo, dmoo=None, with_vxc=True, with_kxc=True, max_memory=2000):
            res = tduks_xc_kernel(td_grad, xc_code, dmvo, dmoo, with_vxc, with_kxc, max_memory)
            if res is None:
                return uks_saved(td_grad, xc_code, dmvo, dmoo, with_vxc, with_kxc, max_memory)
            return res

        tdrks._contract_xc_kernel = rks_kernel
        tduks._contract_xc_kernel = uks_kernel
        return self

    def __exit__(self, *exc):
        from pyscf.grad import tdrks, tduks

        tdrks._contract_xc_kernel, tduks._contract_xc_kernel = self.saved
        return False


class _MojoTDGradMixin:
    """In front of pyscf's TDHF/TDDFT gradient classes: Mojo derivative J/K and XC kernel contractions."""

    __name_mixin__ = "Mojo"

    def _mojo_jk_ok(self, mol, dm, omega):
        if (omega or 0) < 0 or integrals.engine() != "mojo" or integrals.unsupported_reason(mol, two_electron=True):
            return False
        if getattr(mol, "omega", 0):
            return False
        dm = np.asarray(dm)
        return dm.ndim >= 2 and np.isrealobj(dm)

    def get_jk(self, mol=None, dm=None, hermi=0, omega=None):
        from .grad import _jk_ip1

        mol = self.mol if mol is None else mol
        if dm is None or not self._mojo_jk_ok(mol, dm, omega):
            return super().get_jk(mol, dm, hermi, omega)
        return _jk_ip1(mol, np.asarray(dm), omega=omega)

    def get_j(self, mol=None, dm=None, hermi=0, omega=None):
        from .grad import _jk_ip1

        mol = self.mol if mol is None else mol
        if dm is None or not self._mojo_jk_ok(mol, dm, omega):
            return super().get_j(mol, dm, hermi, omega)
        return _jk_ip1(mol, np.asarray(dm), with_k=False, omega=omega)[0]

    def get_k(self, mol=None, dm=None, hermi=0, omega=None):
        from .grad import _jk_ip1

        mol = self.mol if mol is None else mol
        if dm is None or not self._mojo_jk_ok(mol, dm, omega):
            return super().get_k(mol, dm, hermi, omega)
        return _jk_ip1(mol, np.asarray(dm), with_j=False, omega=omega)[1]

    def grad_elec(self, *args, **kwargs):
        with _XCKernel():
            return super().grad_elec(*args, **kwargs)


def accelerate(g):
    """Give a pyscf TDHF/TDDFT gradient object the Mojo kernels, in place; returns it."""
    if not isinstance(g, _MojoTDGradMixin):
        from pyscf import lib

        lib.set_class(g, (_MojoTDGradMixin, type(g)))
    return g
