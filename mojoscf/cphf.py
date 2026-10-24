"""Coupled-perturbed SCF equations with a Krylov solver in matrix products.

pyscf's ``lib.krylov`` keeps its trial vectors in a list and runs every
Gram-Schmidt projection, the QR of each new block and the subspace matrix as
Python loops over single vectors (a ``dot`` and a full-size temporary per
pair of vectors): for the 3N right-hand sides of a Hessian that is
O(n_vec^2) small calls per iteration, seconds for twenty atoms and growing
quadratically with the system.  :func:`krylov` runs the same algorithm
(classical Gram-Schmidt of each new block against the subspace, the block's
QR, the same convergence test and linear-dependence masking) with the trial
vectors as matrix blocks, so each step is a matrix product; :func:`solve` is
pyscf's ``cphf.solve_withs1`` and ``ucphf.solve_withs1`` on it.
"""
from __future__ import annotations

import numpy as np
from pyscf.lib import logger
from pyscf.lib.linalg_helper import DSOLVE_LINDEP, _qr


def _block_qr(x1):
    """The rows of ``x1`` orthogonalised in order, unnormalised (pyscf's ``_qr`` scaled by the diagonal of
    R), and their squared norms.  Householder QR; with a nearly dependent row (squared norm <= 1e-14)
    pyscf's modified Gram-Schmidt instead, which drops it before orthogonalising the later rows."""
    q, r = np.linalg.qr(x1.T)
    d = np.diagonal(r)
    if np.all(d * d > 1e-14):
        return np.ascontiguousarray((q * d).T), d * d
    qs, rmat = _qr(x1, np.dot)
    return qs * rmat.diagonal()[:, None], rmat.diagonal() ** 2


def krylov(aop, b, tol=1e-10, max_cycle=30, lindep=DSOLVE_LINDEP, verbose=logger.WARN):
    """Solve (1 + a) x = b for the rows of ``b`` as pyscf's ``lib.krylov`` (no initial guess, ``hermi``
    False, real vectors), with the trial vectors kept as blocks of a matrix."""
    log = logger.new_logger(verbose=verbose)
    b = np.asarray(b)
    single = b.ndim == 1
    x1 = b.reshape(1, -1) if single else b
    nroots, ndim = x1.shape
    if nroots > 1:
        x1, innerprod = _block_qr(x1)
    else:
        innerprod = np.array([np.dot(x1[0], x1[0])])
    if innerprod.max(initial=0.0) < max(lindep, tol ** 2):
        return np.zeros_like(b)
    xs, ax, ips = [], [], []
    for cycle in range(min(max_cycle, ndim)):
        axt = aop(x1)
        if axt.ndim == 1:
            axt = axt.reshape(1, ndim)
        xs.append(x1)
        ax.append(axt)
        ips.append(innerprod)
        # classical Gram-Schmidt of the new block against the subspace
        x1 = axt.copy()
        for xb, ib in zip(xs, ips):
            x1 -= (axt @ xb.T / ib) @ xb
        if nroots > 1:
            x1, innerprod1 = _block_qr(x1)
        else:
            innerprod1 = np.array([np.dot(x1[0], x1[0])])
        max_innerprod = innerprod1.max(initial=0.0)
        log.debug("krylov cycle %d  r = %g", cycle, max_innerprod ** 0.5)
        if max_innerprod < lindep or max_innerprod < tol ** 2:
            break
        mask = (innerprod1 > lindep) & (innerprod1 > tol ** 2)
        x1 = x1[mask]
        innerprod = innerprod1[mask]
    else:
        raise RuntimeError("Krylov solver failed to converge.")
    xsm = np.vstack(xs)
    h = xsm @ np.vstack(ax).T
    h[np.diag_indices_from(h)] += np.concatenate(ips)
    g = xsm @ b.reshape(nroots, ndim).T
    x = np.linalg.solve(h, g).T @ xsm
    return x[0] if single else x


def solve(fvind, mo_energy, mo_occ, h1, s1, max_cycle=50, tol=1e-9, level_shift=0, verbose=logger.WARN):
    """pyscf's ``cphf.solve_withs1`` (``mo_occ`` one array, RHF) or ``ucphf.solve_withs1`` (a pair, UHF)
    with :func:`krylov`: the first-order MO coefficients and orbital energies of field-dependent bases,
    returned as pyscf's are."""
    log = logger.new_logger(verbose=verbose)
    t0 = (logger.process_clock(), logger.perf_counter())
    unrestricted = isinstance(mo_occ, (tuple, list)) or np.asarray(mo_occ).ndim == 2
    occs = [np.asarray(o) for o in mo_occ] if unrestricted else [np.asarray(mo_occ)]
    ens = [np.asarray(e) for e in mo_energy] if unrestricted else [np.asarray(mo_energy)]
    h1s = list(h1) if unrestricted else [h1]
    s1s = list(s1) if unrestricted else [s1]
    single = np.asarray(h1s[0]).ndim == 2
    chans = []
    for occ, e, h, s in zip(occs, ens, h1s, s1s):
        occidx = occ > 0
        viridx = ~occidx
        e_i = e[occidx]
        e_a = e[viridx]
        e_ai = 1.0 / (e_a[:, None] + level_shift - e_i)
        nmo, nocc = occ.size, int(np.count_nonzero(occidx))
        s = np.asarray(s).reshape(-1, nmo, nocc)
        hs = np.asarray(h).reshape(-1, nmo, nocc) - s * e_i
        base = hs.copy()
        base[:, viridx] *= -e_ai
        base[:, occidx] = -s[:, occidx] * 0.5
        chans.append((occidx, viridx, e_i, e_a, e_ai, nmo, nocc, hs, base))
    nset = chans[0][7].shape[0]
    sizes = [c[5] * c[6] for c in chans]
    mo1base = np.hstack([c[8].reshape(nset, -1) for c in chans])

    def split(v, nd):
        out, off = [], 0
        for c, n in zip(chans, sizes):
            out.append(v[:, off: off + n].reshape(nd, c[5], c[6]))
            off += n
        return out

    def vind_vo(mo1):
        nd = mo1.shape[0]
        arg = mo1 if unrestricted else mo1.reshape(nd, chans[0][5], chans[0][6])
        v = np.asarray(fvind(arg)).reshape(nd, -1)
        if level_shift != 0:
            v = v - mo1 * level_shift
        for c, vs in zip(chans, split(v, nd)):
            vs[:, c[1]] *= c[4]
            vs[:, c[0]] = 0
        return v

    mo1 = krylov(vind_vo, mo1base, tol=tol, max_cycle=max_cycle, verbose=log)
    mo1s = split(mo1.reshape(nset, -1), nset)
    for c, m in zip(chans, mo1s):
        m[:, c[0]] = c[8][:, c[0]]
    log.timer("krylov solver in CPHF", *t0)
    arg = mo1 if unrestricted else mo1s[0]
    v1mo = split(np.asarray(fvind(arg)).reshape(nset, -1), nset)
    e1s = []
    for c, m, v in zip(chans, mo1s, v1mo):
        occidx, viridx, e_i, e_a = c[0], c[1], c[2], c[3]
        hs = c[7] + v
        m[:, viridx] = hs[:, viridx] / (e_i - e_a[:, None])
        e1 = hs[:, occidx]
        e1 += m[:, occidx] * (e_i[:, None] - e_i)
        e1s.append(e1)
    if single:
        mo1s = [m[0] for m in mo1s]
        e1s = [e[0] for e in e1s]
    if unrestricted:
        return tuple(mo1s), tuple(e1s)
    return mo1s[0], e1s[0]
