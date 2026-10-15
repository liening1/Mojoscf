"""The pair sums of pyscf's analytical VV10 Hessian (``pyscf.hessian.rks``) on Mojo kernels.

pyscf's RKS Hessian of a VV10 functional (wB97M-V, wB97X-V, B97M-V, or
``mf.nlc``) evaluates double sums over the NLC grid in C: the per-point
sums U, W, A, B, C, E (``VXC_vv10nlc_hessian_eval_UWABCE``), the kernel
applied to trial densities (``..._f_t``: the response of the potential in
every CPHF iteration and the nuclear-displacement terms), the
grid-response sums (``..._EUW_grid_response``) and the sums weighted with
the derivatives of the grid weights (``..._EUW_with_weight1``).  Each
loops over all ordered pairs of points one at a time with three divisions
per pair, and the trial and weight-derivative sums evaluate the pair
terms again for every six trials or every derivative: O(N^2) pair terms
times the number of nuclear displacements.

:func:`uwabce`, :func:`f_t`, :func:`euw_grid_response` and
:func:`euw_with_weight1` compute the same sums with the Mojo kernels in
``_mojo/nlc.mojo``: the symmetric sums take each unordered pair once, the
sums over trials and derivatives are GEMMs of tiles of pair coefficients
(one division per four vectors of pairs) with the trial vectors, and the
grid-response sums run over the points of one atom at a time.
:func:`install` puts them behind pyscf's calls (``libdft`` of
``pyscf.hessian.rks`` is wrapped; any other call or an unexpected
signature goes to pyscf's C function).
"""
from __future__ import annotations

import ctypes

import numpy as np

from ._backend import get_extension, worker_blas

HT = 128            # points per tile of the GEMM-based kernels (HT in nlc.mojo)
W = 8               # SIMD width of the kernels (atom runs of the grid response are padded to it)


def _c(a, dtype=np.float64):
    return np.ascontiguousarray(a, dtype=dtype)


def uwabce(coords, weights, rho, omega, kappa):
    """``VXC_vv10nlc_hessian_eval_UWABCE``: (U, W, A, B, C, E), each (ngrids,)."""
    n = len(rho)
    out = [np.empty(n) for _ in range(6)]
    get_extension().vv10_uwabce(_c(coords).reshape(n, 3), _c(weights), _c(rho), _c(omega), _c(kappa), *out)
    return tuple(out)


def _tiles(x, nt):
    """(n, nt) rows -> (ntile, HT, nt) zero-padded."""
    n = x.shape[0]
    ntile = (n + HT - 1) // HT
    y = np.zeros((ntile * HT, nt))
    y[:n] = x
    return y.reshape(ntile, HT, nt)


def f_t(coords, weights, rho, omega, kappa, U, W_, A, B, C, domega_drho, domega_dgamma, dkappa_drho, d2omega_drho2,
        d2omega_dgamma2, d2omega_drho_dgamma, d2kappa_drho2, rho_t, gamma_t):
    """``VXC_vv10nlc_hessian_eval_f_t``: (f_rho_t, f_gamma_t), each (ntrial, ngrids), for the trial
    densities rho_t and gamma_t (ntrial, ngrids)."""
    rho = np.asarray(rho)
    n = rho.size
    rho_t = np.asarray(rho_t).reshape(-1, n)
    gamma_t = np.asarray(gamma_t).reshape(-1, n)
    nt = rho_t.shape[0]
    w = np.asarray(weights)
    y = np.empty(((n + HT - 1) // HT, 2, HT, nt))
    y[:, 0] = _tiles((rho_t * w).T, nt)
    y[:, 1] = _tiles((gamma_t * w).T, nt)
    o = np.empty_like(y)
    path, prefix = worker_blas()
    get_extension().vv10_fxc(_c(coords).reshape(n, 3), _c(omega), _c(kappa), _c(rho * domega_drho),
                             _c(rho * dkappa_drho), _c(rho * domega_dgamma), y, o, path, prefix)
    f_rho = o[:, 0].reshape(-1, nt)[:n].T
    f_gamma = o[:, 1].reshape(-1, nt)[:n].T
    # the point's own terms (the i-dependence of g_ij), as pyscf adds them
    frr = (2 * domega_drho * W_ + 2 * dkappa_drho * U
           + rho * (d2omega_drho2 * W_ + d2kappa_drho2 * U + dkappa_drho ** 2 * A + domega_drho ** 2 * C
                    + 2 * domega_drho * dkappa_drho * B))
    fgr = domega_dgamma * W_ + rho * (d2omega_drho_dgamma * W_ + domega_dgamma * (dkappa_drho * B + domega_drho * C))
    fgg = rho * (d2omega_dgamma2 * W_ + domega_dgamma ** 2 * C)
    return f_rho + frr * rho_t + fgr * gamma_t, f_gamma + fgr * rho_t + fgg * gamma_t


def euw_with_weight1(coords, weight1, rho, omega, kappa):
    """``VXC_vv10nlc_hessian_eval_EUW_with_weight1``: (Ew, Uw, Ww), each (nderivative, ngrids), for the
    weight derivatives weight1 (nderivative, ngrids)."""
    rho = np.asarray(rho)
    n = rho.size
    weight1 = np.asarray(weight1).reshape(-1, n)
    nd = weight1.shape[0]
    y = _tiles((weight1 * rho).T, nd)
    o = np.empty((y.shape[0], 3, HT, nd))
    path, prefix = worker_blas()
    get_extension().vv10_euw_w1(_c(coords).reshape(n, 3), _c(omega), _c(kappa), y, o, path, prefix)
    return tuple(o[:, k].reshape(-1, nd)[:n].T for k in range(3))


def euw_grid_response(coords, weights, rho, omega, kappa, atom, natm):
    """``VXC_vv10nlc_hessian_eval_EUW_grid_response``: (Egr, Ugr, Wgr), each (natm, 3, ngrids), for the
    points' atoms ``atom`` (negative: no atom, zero rows as in pyscf)."""
    rho = np.asarray(rho)
    n = rho.size
    atom = np.asarray(atom).ravel()
    keep = np.flatnonzero(atom >= 0)
    order = keep[np.argsort(atom[keep], kind="stable")]
    counts = np.bincount(atom[keep], minlength=natm)[:natm]
    padded = (counts + W - 1) // W * W
    seg = np.zeros(natm + 1, dtype=np.int64)
    np.cumsum(padded, out=seg[1:])
    npad = int(seg[-1])
    pos = np.concatenate([seg[b] + np.arange(counts[b]) for b in range(natm)]).astype(np.int64)
    xyz = np.zeros((npad, 3))
    w = np.zeros(npad)
    r = np.zeros(npad)
    om = np.zeros(npad)
    ka = np.ones(npad)                          # padding points: zero density, finite pair terms
    coords = np.asarray(coords).reshape(n, 3)
    xyz[pos] = coords[order]
    w[pos] = np.asarray(weights)[order]
    r[pos] = rho[order]
    om[pos] = np.asarray(omega)[order]
    ka[pos] = np.asarray(kappa)[order]
    out = np.zeros((3, natm, 3, npad))
    get_extension().vv10_euw_gr(xyz, w, r, om, ka, seg, out)
    res = np.zeros((3, natm, 3, n))
    res[..., order] = out[..., pos]
    return res[0], res[1], res[2]


def omega_derivatives(rho, gamma, c):
    """pyscf's ``VXC_vv10nlc_hessian_eval_omega_derivative``: (domega/drho, domega/dgamma, d2omega/drho2,
    d2omega/dgamma2, d2omega/drho dgamma) of omega = sqrt(C gamma^2 / rho^4 + 4 pi rho / 3)."""
    from pyscf import lib

    rho, gamma = _c(rho), _c(gamma)
    n = rho.size
    out = [np.empty(n) for _ in range(5)]
    lib.load_library("libdft").VXC_vv10nlc_hessian_eval_omega_derivative(
        *(o.ctypes.data_as(ctypes.c_void_p) for o in out), rho.ctypes.data_as(ctypes.c_void_p),
        gamma.ctypes.data_as(ctypes.c_void_p), ctypes.c_double(c), ctypes.c_int(n))
    return out


class Response:
    """pyscf's ``get_vnlc_resp`` (the VV10 term of ``gen_response`` for an RKS ground state: the potential
    matrices of first-order densities) with the ground-state quantities computed once.

    pyscf evaluates the ground-state density on the NLC grid and the pair sums
    of :func:`uwabce` at every call, then the density of every first-order
    matrix and its potential matrix block by block.  Here the densities and
    the potential matrices of all matrices come from the XC kernels in one
    pass each (``dft._rho``, ``dft._vmat``) and the kernel from :func:`f_t`.
    """

    def __init__(self, mf, mo_coeff, mo_occ, max_set=32):
        from . import dft

        ni = mf._numint
        grids = mf.nlcgrids
        if grids.coords is None:
            grids.build()
        xc_code = mf.xc if ni.libxc.is_nlc(mf.xc) else mf.nlc
        coefs = ni.nlc_coeff(xc_code)
        if len(coefs) != 1:
            raise NotImplementedError("Additive NLC")
        (bvv, cvv), _ = coefs[0]             # pyscf's get_vnlc_resp does not scale by the factor either
        c = np.asarray(mo_coeff)
        occ = np.asarray(mo_occ)
        co = np.ascontiguousarray(c[:, occ > 0])
        dm0 = (co * occ[occ > 0]) @ co.T
        self.mol = mf.mol
        self.max_set = max_set
        self.coords, weights = dft._grid(grids)
        self.ngrid = weights.size
        rho = dft._rho(self.mol, self.coords, 1, np.ascontiguousarray(dm0[None]), (co[None], occ[occ > 0][None]))[0]
        self.mask = rho[0] >= 1e-8                  # pyscf's NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD
        r = np.ascontiguousarray(rho[0, self.mask])
        self.nab = np.ascontiguousarray(rho[1:4, self.mask])
        self.w = np.ascontiguousarray(weights[self.mask])
        xyz = np.ascontiguousarray(self.coords[self.mask])
        gamma = (self.nab ** 2).sum(axis=0)
        kpref = bvv * 1.5 * np.pi * (9 * np.pi) ** (-1.0 / 6.0)
        omega = np.sqrt(cvv * gamma ** 2 / r ** 4 + (4.0 / 3.0 * np.pi) * r)
        kappa = kpref * r ** (1.0 / 6.0)
        u, wq, a, b, cc, _ = uwabce(xyz, self.w, r, omega, kappa)
        der = omega_derivatives(r, gamma, cvv)
        dk = kpref * (1.0 / 6.0) * r ** (-5.0 / 6.0)
        d2k = kpref * (-5.0 / 36.0) * r ** (-11.0 / 6.0)
        self.f_gamma = r * der[1] * wq
        self.args = (xyz, self.w, r, omega, kappa, u, wq, a, b, cc, der[0], der[1], dk, der[2], der[3], der[4], d2k)

    def __call__(self, dm1s):
        """The VV10 response potential (nset, nao, nao) of first-order density matrices (nset, nao, nao);
        their symmetric parts carry the density."""
        from . import dft

        d = np.asarray(dm1s, dtype=np.float64)
        single = d.ndim == 2
        nao = d.shape[-1]
        d = d.reshape(-1, nao, nao)
        out = np.empty_like(d)
        for p0 in range(0, d.shape[0], self.max_set):
            p1 = min(p0 + self.max_set, d.shape[0])
            dd = np.ascontiguousarray(0.5 * (d[p0:p1] + d[p0:p1].transpose(0, 2, 1)))
            rt = dft._rho(self.mol, self.coords, 1, dd)
            rho_t = rt[:, 0, self.mask]
            nab_t = rt[:, 1:4, self.mask]
            gamma_t = 2.0 * np.einsum("dg,tdg->tg", self.nab, nab_t)
            f_rho, f_gamma = f_t(*self.args, rho_t, gamma_t)
            wv = np.zeros((p1 - p0, 4, self.ngrid))
            wv[:, 0, self.mask] = 0.5 * f_rho * self.w
            wv[:, 1:4, self.mask] = (2.0 * self.w) * (self.nab * f_gamma[:, None] + nab_t * self.f_gamma)
            out[p0:p1] = dft._vmat(self.mol, self.coords, 1, wv)
        return out[0] if single else out


# --------------------------------------------------------------------------------------------------------
# pyscf's ctypes calls


def _array(arg, n, dtype=ctypes.c_double):
    """A NumPy view of the n elements a ctypes pointer argument points to."""
    addr = arg.value if isinstance(arg, ctypes.c_void_p) else None
    if addr is None:
        raise TypeError("not a pointer")
    return np.ctypeslib.as_array((dtype * n).from_address(addr))


def _int(arg):
    if not isinstance(arg, ctypes.c_int):
        raise TypeError("not an int")
    return arg.value


def _uwabce_call(args):
    if len(args) != 12:
        raise TypeError
    n = _int(args[11])
    outs = [_array(a, n) for a in args[:6]]
    res = uwabce(_array(args[6], 3 * n), *(_array(a, n) for a in args[7:11]))
    for o, r in zip(outs, res):
        o[:] = r


def _f_t_call(args):
    if len(args) != 23:
        raise TypeError
    n, nt = _int(args[21]), _int(args[22])
    f_rho, f_gamma = _array(args[0], nt * n), _array(args[1], nt * n)
    per_point = [_array(a, n) for a in args[3:19]]
    res = f_t(_array(args[2], 3 * n), *per_point, _array(args[19], nt * n).reshape(nt, n),
              _array(args[20], nt * n).reshape(nt, n))
    f_rho[:] = res[0].ravel()
    f_gamma[:] = res[1].ravel()


def _euw_gr_call(args):
    if len(args) != 11:
        raise TypeError
    n, natm = _int(args[9]), _int(args[10])
    outs = [_array(a, natm * 3 * n) for a in args[:3]]
    res = euw_grid_response(_array(args[3], 3 * n), *(_array(a, n) for a in args[4:8]),
                            _array(args[8], n, ctypes.c_int), natm)
    for o, r in zip(outs, res):
        o[:] = r.ravel()


def _euw_w1_call(args):
    if len(args) != 10:
        raise TypeError
    n, nd = _int(args[8]), _int(args[9])
    outs = [_array(a, nd * n) for a in args[:3]]
    res = euw_with_weight1(_array(args[3], 3 * n), _array(args[4], nd * n).reshape(nd, n),
                           *(_array(a, n) for a in args[5:8]))
    for o, r in zip(outs, res):
        o[:] = r.ravel()


_CALLS = {
    "VXC_vv10nlc_hessian_eval_UWABCE": _uwabce_call,
    "VXC_vv10nlc_hessian_eval_f_t": _f_t_call,
    "VXC_vv10nlc_hessian_eval_EUW_grid_response": _euw_gr_call,
    "VXC_vv10nlc_hessian_eval_EUW_with_weight1": _euw_w1_call,
}


class _Libdft:
    """pyscf's ``libdft`` with the VV10 Hessian functions answered by the Mojo kernels."""

    def __init__(self, lib):
        self._mojoscf_orig = lib

    def __getattr__(self, name):
        orig = getattr(self._mojoscf_orig, name)
        call = _CALLS.get(name)
        if call is None:
            return orig

        def fn(*args):
            try:
                return call(args)
            except TypeError:               # a signature this module does not know: pyscf's C function
                return orig(*args)

        return fn


def install():
    """Route the VV10 Hessian sums of ``pyscf.hessian.rks`` to the Mojo kernels (idempotent)."""
    from pyscf.hessian import rks as rks_hess

    if not isinstance(rks_hess.libdft, _Libdft):
        rks_hess.libdft = _Libdft(rks_hess.libdft)
