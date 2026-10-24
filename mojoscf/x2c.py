"""Nuclear gradients of the spin-free X2C one-electron Hamiltonian contracted with densities.

pyscf's ``x2c.sfx2c1e_grad.hcore_grad_generator`` returns, per atom, the
derivative of the X2C Hamiltonian ``h_FW = c_FW^T h c_FW`` (``c_FW = [R; X R]``
in the uncontracted basis of n functions): the first-order Dirac eigenvectors
``c1 = (C^T h1 C - e C^T s1 C) / (e_q - e_p)`` of the 2n x 2n modified Dirac
equation, the first-order coupling ``X1``, the first-order renormalisation
``R1`` (JCP 135, 084114 (2011), Eq. 34) and ``h_FW1``: dozens of products of
(2n x 2n) matrices for each of the three coordinates of every atom.  A
gradient needs only ``sum_ij D_ij h_FW1_ij`` per atom and coordinate, and
every step from the derivative integrals ``(h1, s1)`` to ``h_FW1`` is linear in
them, so here the density is taken backwards through that chain once (each
product transposed: the adjoint), which gives effective densities ``H`` and
``M`` with ``sum D h_FW1 = sum H h1 + sum M s1`` for any perturbation.  Those
are contracted with the derivative integrals of all atoms at the end: the
basis-function derivatives (``int1e_ipovlp``, ``ipkin``, ``ipnuc``,
``ippnucp``) once, the operator derivatives at each nucleus from
:func:`mojoscf.integrals.int1e_iprinv_dm` (all atoms in one pass) and
``int1e_ipprinvp``.  The zeroth-order quantities are pyscf's, so the result is
pyscf's to rounding (``hcore_grad_dm``).

>>> de1 = mojoscf.x2c.hcore_grad_dm(mf.with_x2c, mol, dm)     # (natm, 3)
"""
from __future__ import annotations

from functools import reduce

import numpy as np
import scipy.linalg
from pyscf import gto, lib
from pyscf.x2c import sfx2c1e, sfx2c1e_grad
from pyscf.x2c import x2c as x2c_mod


def grad_ok(x2cobj) -> bool:
    """True if :func:`hcore_grad_dm` reproduces ``x2cobj.hcore_deriv_generator(deriv=1)`` (pyscf's
    spin-free helper with its own generator)."""
    return (
        isinstance(x2cobj, sfx2c1e.SpinFreeX2CHelper)
        and type(x2cobj).hcore_deriv_generator is sfx2c1e.SpinFreeX2CHelper.hcore_deriv_generator
    )


def _contraction(x2cobj, mol):
    """The uncontracted molecule and the (nx, nao) matrix taking its functions to ``mol``'s (or None), as
    ``hcore_grad_generator`` forms them."""
    xmol, contr = x2cobj.get_xmol(mol)
    if x2cobj.basis is not None:
        s22 = xmol.intor_symmetric("int1e_ovlp")
        s21 = gto.intor_cross("int1e_ovlp", xmol, mol)
        contr = lib.cho_solve(s22, s21)
    return xmol, contr


class _Zeroth:
    """Zeroth-order quantities of pyscf's ``gen_sf_hfw`` and ``_gen_first_order_quantities``."""

    def __init__(self, xmol, approx):
        c = lib.param.LIGHT_SPEED
        self.atomic = "ATOM" in approx.upper()
        h0, s0 = sfx2c1e_grad._get_h0_s0(xmol)
        e0, c0 = scipy.linalg.eigh(h0, s0)
        n = xmol.nao_nr()
        if self.atomic:
            x0 = sfx2c1e._atomic_1e_x(xmol)
        else:
            x0 = scipy.linalg.solve(c0[:n, n:].T, c0[n:, n:].T).T
        # gen_sf_hfw
        s_nesc0 = s0[:n, :n] + reduce(np.dot, (x0.T, s0[n:, n:], x0))
        r0 = x2c_mod._get_r(s0[:n, :n], s_nesc0)
        self.c_fw0 = np.vstack((r0, x0 @ r0))
        self.h0_fw_half = h0 @ self.c_fw0
        # _gen_first_order_quantities
        epq = e0[:, None] - e0
        epq[abs(epq) < 1e-7] = 1e200
        ovlp = xmol.intor("int1e_ovlp")
        t0x0 = xmol.intor("int1e_kin") @ x0 * (0.5 / c**2)
        w_s, v_s = scipy.linalg.eigh(ovlp)
        w_sqrt = np.sqrt(w_s)
        sn_v = reduce(np.dot, (v_s.T, ovlp + x0.T @ t0x0, v_s))
        wr0, vr0 = scipy.linalg.eigh(np.einsum("i,ij,j->ij", 1.0 / w_sqrt, sn_v, 1.0 / w_sqrt))
        wr0_sqrt = np.sqrt(wr0)
        r0b = (vr0 / wr0_sqrt) @ vr0.T
        r0b *= w_sqrt
        r0b /= w_sqrt[:, None]
        self.r0b = reduce(np.dot, (v_s, r0b, v_s.T))
        self.n, self.e0, self.c0, self.x0, self.epq, self.t0x0 = n, e0, c0, x0, epq, t0x0
        self.w_sqrt, self.v_s, self.sn_v, self.wr0_sqrt, self.vr0 = w_sqrt, v_s, sn_v, wr0_sqrt, vr0

    def _r1_adjoint(self, rbar):
        """(S1bar, N1bar) with sum R1 rbar = sum S1 S1bar + sum N1 N1bar for pyscf's ``_get_r1``
        (s1 = S1, s_nesc1 = N1), its products and scalings taken in reverse."""
        w_sqrt, v_s, sn_v, vr0 = self.w_sqrt, self.v_s, self.sn_v, self.vr0
        wi = 1.0 / w_sqrt
        wri = 1.0 / self.wr0_sqrt
        vr0_wr0 = vr0 * wri
        p2 = vr0.T * w_sqrt
        p1 = vr0.T * wi
        abar = reduce(np.dot, (v_s.T, rbar, v_s))
        # R1 = p1^T M p2 + s1_invsqrt (vr0_wr0 p2) + (p1^T vr0_wr0^T) s1_sqrt
        mbar = reduce(np.dot, (p1, abar, p2.T))
        inv_bar = abar @ (vr0_wr0 @ p2).T
        sqrt_bar = (p1.T @ vr0_wr0.T).T @ abar
        # M = a_i [(vr0^T K vr0) / -(wri_i + wri_j)] a_j, a = wri^2
        a = wri**2
        kbar = reduce(np.dot, (vr0, a[:, None] * mbar * a / -(wri[:, None] + wri), vr0.T))
        # K = L + L^T + wi_i N1'_ij wi_j, L = (s1_invsqrt sn_v) wi_j
        inv_bar += ((kbar + kbar.T) * wi) @ sn_v.T
        n1bar = wi[:, None] * kbar * wi
        # s1_invsqrt = wi_i^2 s1'_ij wi_j^2 / -(wi_i + wi_j);  s1_sqrt = s1' / (w_sqrt_i + w_sqrt_j)
        c2 = wi**2
        s1bar = c2[:, None] * inv_bar / -(wi[:, None] + wi) * c2
        s1bar += sqrt_bar / (w_sqrt[:, None] + w_sqrt)
        return reduce(np.dot, (v_s, s1bar, v_s.T)), reduce(np.dot, (v_s, n1bar, v_s.T))

    def adjoint(self, dx):
        """(H, M), 2n x 2n, with sum dx h_FW1 = sum H h1 + sum M s1 for every first-order change (h1, s1)
        of the modified Dirac Hamiltonian and metric (dx: density in the uncontracted basis)."""
        n, c0, x0 = self.n, self.c0, self.x0
        ds = (dx + dx.T) * 0.5
        # h_FW1 = c_fw1^T h0_fw_half + transpose + c_fw0^T h1 c_fw0
        g = 2.0 * (self.h0_fw_half @ ds)
        hbar = reduce(np.dot, (self.c_fw0, ds, self.c_fw0.T))
        # c_fw1 = [R1; X0 R1 + X1 R0]
        rbar = g[:n] + x0.T @ g[n:]
        s1bar, n1bar = self._r1_adjoint(rbar)
        mbar = np.zeros_like(hbar)
        # s_nesc1 = X1^T t0x0 + t0x0^T X1 + X0^T s1_SS X0 + s1_LL
        mbar[:n, :n] = s1bar + n1bar
        mbar[n:, n:] = reduce(np.dot, (x0, n1bar, x0.T))
        if not self.atomic:
            xbar = g[n:] @ self.r0b.T + self.t0x0 @ (n1bar + n1bar.T)
            # X1 = (cs1 - X0 cl1) cl0^-1,  [cl1; cs1] = C0 c1
            tbar = scipy.linalg.solve(c0[:n, n:], xbar.T).T
            c1bar = c0.T @ np.vstack((-x0.T @ tbar, tbar))
            # c1 = (h1mo - s1mo e_i)[:, electronic] / -(e_p - e_i)
            f1bar = c1bar / -self.epq[:, n:]
            c0e = c0[:, n:]
            hbar += reduce(np.dot, (c0, f1bar, c0e.T))
            mbar -= reduce(np.dot, (c0, f1bar * self.e0[n:], c0e.T))
        return hbar, mbar


def _effective_densities(hbar, mbar, n):
    """The symmetric (n x n) densities contracted with the derivative overlap, kinetic, nuclear attraction and
    pVp integrals: the blocks of pyscf's h1 = [[V1, T1], [T1, W1/4c^2 - T1]], s1 = [[S1, 0], [0, T1/2c^2]]."""
    c = lib.param.LIGHT_SPEED
    hs = (hbar + hbar.T) * 0.5
    ms = (mbar + mbar.T) * 0.5
    d_v = hs[:n, :n]
    d_t = hs[:n, n:] + hs[n:, :n] - hs[n:, n:] + ms[n:, n:] * (0.5 / c**2)
    d_w = hs[n:, n:] * (0.25 / c**2)
    d_s = ms[:n, :n]
    return d_v, (d_t + d_t.T) * 0.5, d_w, d_s


def hcore_grad_dm(x2cobj, mol, dms):
    """``sum_ij D_ij dh_ij / dR_A`` of the spin-free X2C Hamiltonian for every atom A: pyscf's
    ``hcore_grad_generator(x2cobj, mol)(A)`` contracted with each density of ``dms`` (nao x nao, or a stack),
    as (natm, 3) or (nset, natm, 3)."""
    from . import integrals

    dms = np.asarray(dms, dtype=np.float64)
    single = dms.ndim == 2
    dms = dms.reshape(-1, mol.nao, mol.nao)
    xmol, contr = _contraction(x2cobj, mol)
    zeroth = _Zeroth(xmol, x2cobj.approx)
    n = xmol.nao_nr()
    sets = []
    for dm in dms:
        dx = dm if contr is None else reduce(np.dot, (contr, dm, contr.T))
        sets.append(_effective_densities(*zeroth.adjoint(dx), n))

    natm = xmol.natm
    charges = xmol.atom_charges().astype(np.float64)
    out = np.zeros((len(sets), natm, 3))
    # basis-function derivatives: rows of the atom's functions, 2 sum_{i in A} X_ij D_ij for each block
    aoslices = xmol.aoslice_by_atom()
    ints = [xmol.intor(name, comp=3) for name in ("int1e_ipnuc", "int1e_ipkin", "int1e_ippnucp", "int1e_ipovlp")]
    for s, dsets in enumerate(sets):
        rows = sum(np.einsum("xij,ij->xi", x, d) for x, d in zip(ints, dsets))
        for ia in range(natm):
            p0, p1 = aoslices[ia, 2:]
            if p1 > p0:
                out[s, ia] -= 2.0 * rows[:, p0:p1].sum(axis=1)
    # operator derivatives at each nucleus: -2 Z_A (<nabla i|1/r_A|j> D_V + <nabla p i|1/r_A|p j> D_W)
    if integrals.available(xmol):
        for s, dsets in enumerate(sets):
            out[s] -= 2.0 * charges[:, None] * integrals.int1e_iprinv_dm(xmol, dsets[0])
        need_rinv = False
    else:
        need_rinv = True
    for ia in range(natm):
        if charges[ia] == 0:
            continue
        with xmol.with_rinv_origin(xmol.atom_coord(ia)):
            prinvp = xmol.intor("int1e_ipprinvp", comp=3)
            rinv = xmol.intor("int1e_iprinv", comp=3) if need_rinv else None
        for s, dsets in enumerate(sets):
            g = np.einsum("xij,ij->x", prinvp, dsets[2])
            if rinv is not None:
                g += np.einsum("xij,ij->x", rinv, dsets[0])
            out[s, ia] -= 2.0 * charges[ia] * g
    return out[0] if single else out
