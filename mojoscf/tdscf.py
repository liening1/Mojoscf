"""Linear-response TDDFT, TDA and TDHF (pyscf.tdscf) with the response built in the occupied-virtual space.

pyscf's Davidson solvers need the product of the response matrices with a
batch of trial vectors.  pyscf forms it by expanding each vector into an AO
transition density, calling ``get_jk`` and the XC kernel in the AO basis and
projecting back; with density fitting and non-symmetric densities its
``get_jk`` is the bulk of a TDDFT run.  With an in-core DF tensor ``(pq|Q)``
the Coulomb and exchange parts are much cheaper directly in the MO basis
(i, j occupied, a, b virtual, ``(ia|Q)`` the transformed tensor):

    J[i,a]     = sum_Q (ia|Q) rho_Q,   rho_Q = sum_jb (jb|Q) (x + y)_jb (x 2 for RKS)
    (K_A x)_ia = sum_Q [(oo|Q) x (vv|Q)]_ia       ((ij|ab) x_jb)
    (K_B y)_ia = sum_Q [(ov|Q) y^T (ov|Q)]_ia     ((ib|ja) y_jb)

with the top block A x + B y = e x + J + XC - c (K_A x + K_B y) and the bottom
block B x + A y = e y + J + XC - c (K_B x + K_A y) (TDA: y = 0; c the
fraction of exact exchange, a second term with the long-range tensor for
range-separated hybrids).  :func:`mojoscf.kernels.df_mo` transforms the
tensor once per calculation and :func:`mojoscf.kernels.df_sandwich`
contracts the exchange terms for all trial vectors at once.

With exact integrals, ERIs pyscf keeps in core are transformed once to
(ia|jb), (ij|ab) and (ib|ja) as (ov) x (ov) matrices, which turns every
product into a GEMM; otherwise (integral-direct SCF, or MO matrices beyond
``max_memory``) J and K of the AO transition densities of all vectors come
from one call of the Mojo kernels (:func:`mojoscf.dft.exact_jk`: symmetric
parts with J and K, antisymmetric ones with K).

The XC response skips the AO densities as well: from the factors
``C_v x^T`` and ``C_o`` of the transition densities a Mojo pass forms
``V C_o`` of the response potential directly (:func:`mojoscf.dft.fxc_matrices`
with ``project``).  The eigensolvers, initial guesses, normalisation and
analysis stay pyscf's; they run with SciPy's OpenBLAS on one thread
(:class:`mojoscf._backend.serial_scipy_blas`), as their ``scipy.linalg`` and
``numpy.linalg`` calls on the subspace matrices otherwise wait for each
other's spinning BLAS threads.

Applies to RHF/UHF and RKS/UKS objects with pyscf's in-core density fitting
or exact integrals (the objects :func:`mojoscf.dft.accelerate` returns create
these classes from ``mf.TDA()``, ``mf.TDDFT()``, ``mf.CasidaTDDFT()``);
point-group restrictions (``wfnsym``), solvent models, short-range-only
hybrids, NLC response, DF tensors that do not fit in ``max_memory`` and
objects with a ``get_jk`` of their own fall back to pyscf's operator.

>>> mf = mojoscf.dft.accelerate(dft.RKS(mol, xc="pbe0").density_fit()).run()
>>> td = mf.TDA(); td.nstates = 10; td.kernel()
>>> td = mojoscf.tdscf.TDDFT(mf).run()            # the same classes, created explicitly
"""
from __future__ import annotations

import numpy as np
from pyscf import lib
from pyscf.lib import logger

from . import integrals, kernels
from ._backend import serial_scipy_blas

__all__ = ["TDA", "TDDFT", "TDHF", "CasidaTDDFT", "TDDFTNoHybrid", "RPA"]

_CLASSES: dict = {}


class _MojoTD:
    """In front of a pyscf TD class: ``gen_vind`` builds the response in the occupied-virtual space, and the
    nuclear gradients (``Gradients``/``nuc_grad_method``) carry the Mojo kernels (:mod:`mojoscf.tdgrad`)."""

    def gen_vind(self, mf=None):
        assert mf is None or mf is self._scf
        op = _operator(self)
        if op is None:
            return super().gen_vind(mf)
        return op

    def kernel(self, *args, **kwargs):
        # pyscf's eigensolvers alternate numpy.linalg and scipy.linalg calls on the subspace matrices
        with serial_scipy_blas():
            return super().kernel(*args, **kwargs)

    def Gradients(self):
        from . import tdgrad

        return tdgrad.accelerate(super().Gradients())


def _td_class(mf, name):
    """The subclass of pyscf's TD class ``name`` for ``mf`` (RHF/UHF/RKS/UKS) with the MO-space operator."""
    from pyscf.scf import hf, rohf, uhf
    from pyscf.tdscf import rhf as td_rhf
    from pyscf.tdscf import rks as td_rks
    from pyscf.tdscf import uhf as td_uhf
    from pyscf.tdscf import uks as td_uks

    ks = isinstance(mf, hf.KohnShamDFT)
    unrestricted = isinstance(mf, uhf.UHF)
    if not (unrestricted or isinstance(mf, hf.RHF)) or isinstance(mf, rohf.ROHF):
        raise TypeError(f"mojoscf.tdscf supports RHF/UHF/RKS/UKS objects, not {type(mf).__name__}")
    module = {(False, False): td_rhf, (False, True): td_uhf, (True, False): td_rks, (True, True): td_uks}[ks, unrestricted]
    base = getattr(module, name)
    cls = _CLASSES.get(base)
    if cls is None:
        cls = _CLASSES[base] = type(base.__name__, (_MojoTD, base), {"__module__": __name__})
    return cls


def TDA(mf, frozen=None):
    """Tamm-Dancoff approximation (pyscf's ``mf.TDA()``) with the MO-space response."""
    return _td_class(mf, "TDA")(mf, frozen)


def TDDFT(mf, frozen=None):
    """pyscf's ``mf.TDDFT()``: full TDDFT for hybrids and HF, the Casida form for pure functionals."""
    from pyscf.scf import hf

    if not isinstance(mf, hf.KohnShamDFT):
        return _td_class(mf, "TDHF")(mf, frozen)
    if mf._numint.libxc.is_hybrid_xc(mf.xc):
        return _td_class(mf, "TDDFT")(mf, frozen)
    return _td_class(mf, "CasidaTDDFT")(mf, frozen)


def CasidaTDDFT(mf, frozen=None):
    """pyscf's ``mf.CasidaTDDFT()`` ((A-B)(A+B) for pure functionals) with the MO-space response."""
    return _td_class(mf, "CasidaTDDFT")(mf, frozen)


TDHF = RPA = TDDFT
TDDFTNoHybrid = CasidaTDDFT


# ---------------------------------------------------------------------- set-up


def _kind(td):
    from pyscf.tdscf import rhf as td_rhf
    from pyscf.tdscf import rks as td_rks
    from pyscf.tdscf import uhf as td_uhf
    from pyscf.tdscf import uks as td_uks

    if isinstance(td, (td_rks.CasidaTDDFT, td_uks.CasidaTDDFT)):
        return "casida"
    if isinstance(td, (td_rhf.TDHF, td_uhf.TDHF)):
        return "rpa"
    if isinstance(td, (td_rhf.TDA, td_uhf.TDA)):
        return "tda"
    return None


def _exchange_terms(mf):
    """[(coefficient, omega)] of the exact exchange in the response, or None if pyscf's operator must run."""
    from pyscf.scf import hf

    if not isinstance(mf, hf.KohnShamDFT):
        return [(1.0, 0.0)]
    ni = mf._numint
    if not ni.libxc.is_hybrid_xc(mf.xc):
        return []
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mf.mol.spin)
    if omega == 0:
        return [(hyb, 0.0)]
    if omega < 0 or alpha == 0:          # short-range exchange only: pyscf fits it with its own metric
        return None
    if hyb == 0:
        return [(alpha, omega)]
    return [(hyb, 0.0), (alpha - hyb, omega)]


class _Channel:
    """One spin channel: orbitals, orbital-energy differences and the MO-basis DF tensors."""

    def __init__(self, orbo, orbv, e_ia):
        self.orbo = np.ascontiguousarray(orbo)
        self.orbv = np.ascontiguousarray(orbv)
        self.nocc = orbo.shape[1]
        self.nvir = orbv.shape[1]
        self.e_ia = e_ia
        self.lov = None          # (naux, nocc, nvir) for J (full-range tensor)
        self.k = []              # [(c, (oo|Q), (vv|Q), (ov|Q) or None)] per exchange term


def _channels(td):
    """The spin channels of the (frozen-masked) reference, as pyscf's operators set them up."""
    from pyscf.scf import uhf

    mf = td._scf
    if isinstance(mf, uhf.UHF):
        masks = td.get_frozen_mask()
        out = []
        for s in range(2):
            c = mf.mo_coeff[s][:, masks[s]]
            e = mf.mo_energy[s][masks[s]]
            occ = mf.mo_occ[s][masks[s]]
            o, v = np.where(occ > 0)[0], np.where(occ == 0)[0]
            out.append(_Channel(c[:, o], c[:, v], e[v] - e[o, None]))
        return out
    mask = td.get_frozen_mask()
    c = mf.mo_coeff[:, mask]
    e = mf.mo_energy[mask]
    occ = mf.mo_occ[mask]
    o, v = np.where(occ == 2)[0], np.where(occ == 0)[0]
    return [_Channel(c[:, o], c[:, v], e[v] - e[o, None])]


def _operator(td):
    """``(vind, hdiag)`` of the MO-space response for ``td``, or None when pyscf's operator has to run."""
    from pyscf.scf import hf, rohf, uhf

    from . import dft

    mf = td._scf
    mol = mf.mol
    kind = _kind(td)
    if kind is None or integrals.engine() != "mojo":
        return None
    if td.wfnsym is not None and mol.symmetry:
        return None
    if mf.mo_coeff is None or not np.isrealobj(mf.mo_coeff):
        return None
    try:
        from pyscf.solvent._attach_solvent import _Solvation

        if isinstance(mf, _Solvation):
            return None
    except ImportError:  # pragma: no cover - older pyscf
        pass
    ks = isinstance(mf, hf.KohnShamDFT)
    unrestricted = isinstance(mf, uhf.UHF)
    if not (unrestricted or isinstance(mf, hf.RHF)) or isinstance(mf, rohf.ROHF):
        return None
    singlet = True
    if not unrestricted:
        singlet = td.singlet
        if singlet is None:
            return None
    if ks:
        mf._numint.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
        if not td.exclude_nlc and mf.do_nlc():
            return None
    kterms = _exchange_terms(mf)
    if kterms is None:
        return None
    chans = _channels(td)
    cderi = dft._df_tensor(mf)
    if cderi is not None:
        two = _df_two_electron(td, chans, kterms, cderi, need_j=singlet or unrestricted, need_b=kind == "rpa")
    elif dft.exact_jk_applies(mf):
        two = _exact_mo_two_electron(td, chans, kterms, need_j=singlet or unrestricted, need_b=kind == "rpa")
        if two is None:
            two = _exact_two_electron(mf, chans, kterms)
    else:
        two = None
    if two is None:
        return None
    xc = _xc_response(td, ks, unrestricted, singlet, chans)
    if unrestricted:
        return _uks_operator(kind, chans, xc, two)
    return _rks_operator(kind, chans[0], singlet, xc, two)


def _df_two_electron(td, chans, kterms, cderi, need_j, need_b):
    """``two(xs, ys, jscale)`` (:func:`_rks_operator`) from the MO-basis DF tensors, which this sets up on the
    channels; None when a tensor is missing or they would not fit in ``max_memory``."""
    from . import dft

    mf = td._scf
    log = logger.new_logger(td)
    tensors = {0.0: cderi}
    for _, omega in kterms:
        if omega not in tensors:
            t = dft._df_tensor(mf, omega)
            if t is None:
                return None
            tensors[omega] = t
    naux = cderi.shape[0]
    words = 0
    for ch in chans:
        no, nv = ch.nocc, ch.nvir
        words += naux * no * nv if need_j else 0
        for c, omega in kterms:
            words += tensors[omega].shape[0] * (no * no + nv * nv + (no * nv if need_b else 0))
    avail = td.max_memory - lib.current_memory()[0]
    if words * 8e-6 > 0.7 * avail:
        log.info("mojoscf.tdscf: the MO-basis DF tensors (%.0f MB) exceed max_memory; pyscf's operator runs",
                 words * 8e-6)
        return None

    t0 = (logger.process_clock(), logger.perf_counter())
    for ch in chans:
        if ch.nocc == 0 or ch.nvir == 0:
            continue
        if need_j:
            ch.lov = kernels.df_mo(cderi, ch.orbo, ch.orbv)
        for c, omega in kterms:
            if c == 0:
                continue
            t = tensors[omega]
            loo = kernels.df_mo(t, ch.orbo, ch.orbo)
            lvv = kernels.df_mo(t, ch.orbv, ch.orbv)
            lov = None
            if need_b:
                lov = ch.lov if omega == 0 and ch.lov is not None else kernels.df_mo(t, ch.orbo, ch.orbv)
            ch.k.append((c, loo, lvv, lov))
    log.timer("mojoscf.tdscf MO-basis DF tensors", *t0)

    def two(xs, ys, jscale):
        n = len(xs[0])
        ws = xs if ys is None else [x + y for x, y in zip(xs, ys)]
        tops = _coulomb(chans, ws, jscale) if jscale else [np.zeros_like(x) for x in xs]
        bots = None if ys is None else [t.copy() for t in tops]
        for s, ch in enumerate(chans):
            if not _empty(ch, n):
                _exchange(ch, xs[s], None if ys is None else ys[s], tops[s], None if bots is None else bots[s])
        return tops, bots

    return two


def _exact_mo_two_electron(td, chans, kterms, need_j, need_b):
    """``two(xs, ys, jscale)`` (:func:`_rks_operator`) from exact MO-basis integrals, transformed once from the
    in-core 8-fold ERIs: (ia|jb) for J, and per exchange term (ij|ab) for K_A and (ib|ja) for K_B, each stored
    as an (ov) x (ov) matrix, so that every product is one GEMM for all vectors.  Range-separated hybrids
    take their long-range ERIs from pyscf (libcint).  None when the ERIs are not kept in core or the MO
    matrices would not fit in ``max_memory`` (then :func:`_exact_two_electron`).
    """
    from pyscf import ao2mo

    mf = td._scf
    mol = mf.mol
    log = logger.new_logger(td)
    if getattr(mf, "_eri", None) is None and not (mol.incore_anyway or mf._is_mem_enough()):
        return None
    nao = mol.nao_nr()
    npair = nao * (nao + 1) // 2
    nov = [ch.nocc * ch.nvir for ch in chans]
    kts = [(c, omega) for c, omega in kterms if c != 0]
    words = sum(nov[s] * nov[t] for s in range(len(chans)) for t in range(s + 1)) if need_j else 0
    words += sum(n * n for n in nov) * len(kts) * (2 if need_b else 1) + max(nov) ** 2
    words += npair * (npair + 1) // 2 * (1 if any(omega for _, omega in kts) else 0)
    if getattr(mf, "_eri", None) is None:
        words += npair * (npair + 1) // 2
    avail = td.max_memory - lib.current_memory()[0]
    if words * 8e-6 > 0.7 * avail:
        log.info("mojoscf.tdscf: the MO-basis ERIs (%.0f MB) exceed max_memory; J/K of AO densities run", words * 8e-6)
        return None
    t0 = (logger.process_clock(), logger.perf_counter())
    if getattr(mf, "_eri", None) is None:
        mf._eri = integrals.int2e_s8(mol)
    eri0 = mf._eri
    nch = len(chans)
    live = [ch.nocc > 0 and ch.nvir > 0 for ch in chans]

    def ovov(eri, s, t):
        a, b = chans[s], chans[t]
        return ao2mo.incore.general(eri, (a.orbo, a.orbv, b.orbo, b.orbv), compact=False)

    gj = {}
    if need_j:
        for s in range(nch):
            for t in range(s + 1):
                if live[s] and live[t]:
                    gj[s, t] = ovov(eri0, s, t)
    kmats = [[] for _ in chans]           # per channel [(c, K_A matrix, K_B matrix or None)]
    for c, omega in kts:
        if omega:
            with mol.with_range_coulomb(omega):
                eri = mol.intor("int2e", aosym="s8")
        else:
            eri = eri0
        for s, ch in enumerate(chans):
            if not live[s]:
                continue
            no, nv = ch.nocc, ch.nvir
            ka = ao2mo.incore.general(eri, (ch.orbo, ch.orbo, ch.orbv, ch.orbv), compact=False)
            ka = np.ascontiguousarray(ka.reshape(no, no, nv, nv).transpose(0, 2, 1, 3)).reshape(no * nv, no * nv)
            kb = None
            if need_b:
                g = gj[s, s] if omega == 0 and (s, s) in gj else ovov(eri, s, s)
                kb = np.ascontiguousarray(g.reshape(no, nv, no, nv).transpose(0, 3, 2, 1)).reshape(no * nv, no * nv)
                g = None
            kmats[s].append((c, ka, kb))
        eri = None
    log.timer("mojoscf.tdscf MO-basis ERIs", *t0)

    def two(xs, ys, jscale):
        n = len(xs[0])
        xm = [x.reshape(n, -1) for x in xs]
        ym = None if ys is None else [y.reshape(n, -1) for y in ys]
        tops = [np.zeros((n, k)) for k in nov]
        bots = None if ys is None else [np.zeros((n, k)) for k in nov]
        if jscale and gj:
            ws = xm if ym is None else [x + y for x, y in zip(xm, ym)]
            for (s, t), g in gj.items():
                # sum_jb (ia|jb) w_jb: rows of g are ia (channel s), columns jb (channel t)
                j = ws[t] @ g.T
                tops[s] += jscale * j
                if bots is not None:
                    bots[s] += jscale * j
                if s != t:
                    j = ws[s] @ g
                    tops[t] += jscale * j
                    if bots is not None:
                        bots[t] += jscale * j
        for s in range(nch):
            for c, ka, kb in kmats[s]:
                if ym is None:
                    tops[s] -= c * (xm[s] @ ka)
                    continue
                xy = np.vstack((xm[s], ym[s]))
                pa = xy @ ka
                pb = xy @ kb
                tops[s] -= c * (pa[:n] + pb[n:])
                bots[s] -= c * (pb[:n] + pa[n:])
        tops = [t.reshape(x.shape) for t, x in zip(tops, xs)]
        if bots is not None:
            bots = [b.reshape(x.shape) for b, x in zip(bots, xs)]
        return tops, bots

    return two


def _exact_two_electron(mf, chans, kterms):
    """``two(xs, ys, jscale)`` (:func:`_rks_operator`) with exact integrals: J and K of the AO transition
    densities ``C_v x^T C_o^T + C_o y C_v^T`` of all vectors in one call of the Mojo kernels
    (:func:`mojoscf.dft.exact_jk`, in-core or integral-direct; their symmetric parts give J and K, the
    antisymmetric ones K), projected on the occupied-virtual block: ``C_v^T V C_o`` for the top block,
    ``C_o^T V C_v`` for the bottom one.  The long-range exchange of range-separated hybrids is pyscf's
    ``get_k(omega=...)``.
    """
    from . import dft

    mol = mf.mol
    cfull = sum(c for c, omega in kterms if omega == 0)
    lr = [(c, omega) for c, omega in kterms if omega != 0 and c != 0]

    def top(ch, m):
        return (ch.orbv.T @ m @ ch.orbo).transpose(0, 2, 1)

    def bot(ch, m):
        return ch.orbo.T @ m @ ch.orbv

    def two(xs, ys, jscale):
        n = len(xs[0])
        dms = []
        for s, ch in enumerate(chans):
            d = ch.orbv @ (xs[s].transpose(0, 2, 1) @ ch.orbo.T)
            if ys is not None:
                d += ch.orbo @ (ys[s] @ ch.orbv.T)
            dms.append(d)
        tops = [np.zeros_like(x) for x in xs]
        bots = None if ys is None else [np.zeros_like(x) for x in xs]
        vj = vk = None
        if jscale and cfull == 0:
            # Coulomb only: of the symmetric part of the summed densities
            d = dms[0] if len(dms) == 1 else sum(dms)
            vj = dft.exact_jk(mf, 0.5 * (d + d.transpose(0, 2, 1)), 1, True, False)[0]
        elif jscale or cfull != 0:
            vj, vk = dft.exact_jk(mf, np.concatenate(dms), 0, bool(jscale), cfull != 0)
            if vj is not None:
                vj = vj.reshape(len(chans), n, *vj.shape[1:]).sum(axis=0)
        for s, ch in enumerate(chans):
            if _empty(ch, n):
                continue
            if vj is not None:
                j = top(ch, jscale * vj)
                tops[s] += j
                if bots is not None:
                    bots[s] += j
            if vk is not None:
                k = vk[s * n:(s + 1) * n]
                tops[s] -= top(ch, cfull * k)
                if bots is not None:
                    bots[s] -= bot(ch, cfull * k)
        for c, omega in lr:
            vk = np.asarray(mf.get_k(mol, np.concatenate(dms), hermi=0, omega=omega))
            for s, ch in enumerate(chans):
                if _empty(ch, n):
                    continue
                k = vk[s * n:(s + 1) * n]
                tops[s] -= top(ch, c * k)
                if bots is not None:
                    bots[s] -= bot(ch, c * k)
        return tops, bots

    return two


def _xc_response(td, ks, unrestricted, singlet, chans):
    """``f(factors) -> [(n, nocc, nvir) per channel]``: the XC kernel response to transition densities.

    ``factors`` holds one ``(L (n, nao, nocc), R = C_o)`` pair per spin
    channel (only the symmetric part of ``L R^T`` enters); the result is
    ``C_o^T V C_v`` of pyscf's response matrix V (``nr_rks_fxc_st`` with the
    halved spin kernel for RKS, ``nr_uks_fxc`` for UKS), the same in the top
    and bottom blocks since V is symmetric.  With
    :class:`mojoscf.dft.NumInt` one fused Mojo pass per batch forms ``V C_o``
    straight from the factors (:func:`mojoscf.dft.fxc_matrices` with
    ``project``); otherwise pyscf's NumInt builds V.
    """
    from . import dft

    if not ks:
        return None
    mf = td._scf
    ni = mf._numint
    if ni._xc_type(mf.xc) == "HF":
        return None
    mol = mf.mol
    mem_now = lib.current_memory()[0]
    max_memory = max(2000, mf.max_memory * 0.8 - mem_now)
    rho0, vxc, fxc = ni.cache_xc_kernel(mol, mf.grids, mf.xc, mf.mo_coeff, mf.mo_occ, 1)
    fast = isinstance(ni, dft.NumInt) and dft._fxc_ok(ni, mol, mf.xc)
    kind = dft._kind(ni, mf.xc) if fast else None
    if not unrestricted:
        fxc *= 0.5
        if fast:
            fxc = fxc[0, :, 0] + fxc[0, :, 1] if singlet else fxc[0, :, 0] - fxc[0, :, 1]
    else:
        fxc = np.asarray(fxc)

    def f(factors):
        if fast:
            w = dft.fxc_matrices(mol, mf.grids, kind, fxc, factors=factors, project=True)
            return [(ch.orbv.T @ w[a][:, :, : ch.nocc]).transpose(0, 2, 1) for a, ch in enumerate(chans)]
        if unrestricted:
            dms = np.asarray([lf @ rf.T for lf, rf in factors])
            v1ao = ni.nr_uks_fxc(mol, mf.grids, mf.xc, None, dms, 0, 0, rho0, vxc, fxc, max_memory=max_memory)
        else:
            lf, rf = factors[0]
            v1ao = [ni.nr_rks_fxc_st(mol, mf.grids, mf.xc, None, lf @ rf.T, 0, singlet, rho0, vxc, fxc,
                                     max_memory=max_memory)]
        return [_top(ch, v) for ch, v in zip(chans, v1ao)]

    return f


# ------------------------------------------------------------------ contractions


def _coulomb(chans, ws, scale):
    """J of the vectors ``ws`` (one (n, nocc, nvir) array per channel): sum_Q (ia|Q) rho_Q per channel."""
    rho = 0.0
    for ch, w in zip(chans, ws):
        if ch.lov is not None:
            rho = rho + ch.lov.reshape(ch.lov.shape[0], -1) @ w.reshape(len(w), -1).T
    out = []
    for ch, w in zip(chans, ws):
        if ch.lov is None:
            out.append(np.zeros_like(w))
        else:
            out.append((scale * rho.T @ ch.lov.reshape(ch.lov.shape[0], -1)).reshape(w.shape))
    return out


def _k_a(loo, lvv, z):
    """sum_Q (oo|Q) z_n (vv|Q) for every vector, (n, nocc, nvir)."""
    n, no, nv = z.shape
    x = np.ascontiguousarray(z.transpose(1, 0, 2).reshape(no, n * nv))
    return kernels.df_sandwich(loo, x, lvv, n).reshape(no, n, nv).transpose(1, 0, 2)


def _k_b(lov, y):
    """sum_Q (ov|Q) y_n^T (ov|Q) for every vector, (n, nocc, nvir)."""
    n, no, nv = y.shape
    x = np.ascontiguousarray(y.transpose(2, 0, 1).reshape(nv, n * no))
    return kernels.df_sandwich(lov, x, lov, n).reshape(no, n, nv).transpose(1, 0, 2)


def _exchange(ch, x, y, top, bot):
    """Subtract the exact-exchange terms: top -= c (K_A x + K_B y), bot -= c (K_B x + K_A y) (TDA: y None)."""
    n = len(x)
    for c, loo, lvv, lov in ch.k:
        if y is None:
            top -= c * _k_a(loo, lvv, x)
            continue
        ka = _k_a(loo, lvv, np.concatenate((x, y)))
        kb = _k_b(lov, np.concatenate((y, x)))
        top -= c * (ka[:n] + kb[:n])
        bot -= c * (ka[n:] + kb[n:])


def _factor(ch, w, scale):
    """(L, R) with L R^T = scale C_v w_n^T C_o^T for every vector: L (n, nao, nocc), R = C_o."""
    return (scale * (ch.orbv @ w.transpose(0, 2, 1)), ch.orbo)


def _top(ch, v1ao):
    """pyscf's einsum('xpq,qo,pv->xov', v1ao, orbo, orbv)."""
    return (ch.orbv.T @ v1ao @ ch.orbo).transpose(0, 2, 1)


def _empty(ch, n):
    return ch.nocc == 0 or ch.nvir == 0 or n == 0


def _rks_operator(kind, ch, singlet, xc, two):
    """``(vind, hdiag)`` for one closed-shell channel.  ``two(xs, ys, jscale)`` (per channel lists of
    (n, nocc, nvir) vectors; ``ys`` None for TDA) returns the two-electron parts of the top and bottom blocks
    (bottom None for TDA): ``jscale`` J(x + y) - c (K_A x + K_B y) and ``jscale`` J(x + y) - c (K_B x + K_A y)."""
    no, nv = ch.nocc, ch.nvir
    e_ia = ch.e_ia
    jscale = 2.0 if singlet else 0.0

    if kind == "tda":
        def vind(zs):
            zs = np.asarray(zs).reshape(-1, no, nv)
            v = zs * e_ia
            if _empty(ch, len(zs)):
                return v.reshape(len(zs), -1)
            v += two([zs], None, jscale)[0][0]
            if xc is not None:
                v += xc([_factor(ch, zs, 2.0)])[0]
            return v.reshape(len(zs), -1)

        return vind, e_ia.ravel()

    if kind == "rpa":
        def vind(xys):
            xys = np.asarray(xys).reshape(-1, 2, no, nv)
            nz = len(xys)
            xs, ys = xys[:, 0], xys[:, 1]
            top = xs * e_ia
            bot = ys * e_ia
            if not _empty(ch, nz):
                t2, b2 = two([xs], [ys], jscale)
                top += t2[0]
                bot += b2[0]
                if xc is not None:
                    p = xc([_factor(ch, xs + ys, 2.0)])[0]
                    top += p
                    bot += p
            return np.hstack((top.reshape(nz, -1), -bot.reshape(nz, -1)))

        hdiag = e_ia.ravel()
        return vind, np.hstack((hdiag, -hdiag))

    # Casida: (A-B)^1/2 (A+B) (A-B)^1/2 for pure functionals (no exchange)
    d_ia = np.sqrt(e_ia)
    ed_ia = e_ia * d_ia

    def vind(zs):
        zs = np.asarray(zs).reshape(-1, no, nv)
        nz = len(zs)
        v = zs * ed_ia
        if not _empty(ch, nz):
            w = zs * d_ia
            if singlet:
                v += two([w], None, 4.0)[0][0]
            if xc is not None:
                v += xc([_factor(ch, w, 4.0)])[0]
        v *= d_ia
        return v.reshape(nz, -1)

    return vind, (e_ia ** 2).ravel()


def _uks_operator(kind, chans, xc, two):
    a, b = chans
    nova = a.nocc * a.nvir
    e_ia = np.hstack((a.e_ia.ravel(), b.e_ia.ravel()))

    def split(v, n):
        return [v[:, :nova].reshape(n, a.nocc, a.nvir), v[:, nova:].reshape(n, b.nocc, b.nvir)]

    def xc_terms(ws, n, symmetric):
        return xc([_factor(ch, w, 2.0 if symmetric else 1.0) for ch, w in zip(chans, ws)])

    if kind in ("tda", "rpa"):
        def vind(zs):
            zs = np.asarray(zs)
            nz = len(zs)
            if kind == "tda":
                xs, ys = split(zs.reshape(nz, -1), nz), None
            else:
                zs = zs.reshape(nz, 2, -1)
                xs, ys = split(zs[:, 0], nz), split(zs[:, 1], nz)
            ws = xs if ys is None else [x + y for x, y in zip(xs, ys)]
            tops = [x * ch.e_ia for x, ch in zip(xs, chans)]
            bots = None if ys is None else [y * ch.e_ia for y, ch in zip(ys, chans)]
            if nz:
                t2, b2 = two(xs, ys, 1.0)
                pxc = xc_terms(ws, nz, False) if xc is not None else None
                for s, ch in enumerate(chans):
                    if _empty(ch, nz):
                        continue
                    tops[s] += t2[s]
                    if bots is not None:
                        bots[s] += b2[s]
                    if pxc is not None:
                        tops[s] += pxc[s]
                        if bots is not None:
                            bots[s] += pxc[s]
            top = np.hstack([t.reshape(nz, -1) for t in tops])
            if bots is None:
                return top
            return np.hstack((top, -np.hstack([t.reshape(nz, -1) for t in bots])))

        if kind == "tda":
            return vind, e_ia
        return vind, np.hstack((e_ia, -e_ia))

    # Casida
    d_ia = np.sqrt(e_ia)
    ed_ia = e_ia * d_ia

    def vind(zs):
        zs = np.asarray(zs)
        nz = len(zs)
        zs = zs.reshape(nz, -1)
        hx = zs * ed_ia
        if nz:
            ws = split(zs * d_ia, nz)
            js = two(ws, None, 2.0)[0]
            pxc = xc_terms(ws, nz, True) if xc is not None else None
            parts = []
            for s, ch in enumerate(chans):
                v = js[s]
                if pxc is not None and not _empty(ch, nz):
                    v = v + pxc[s]
                parts.append(v.reshape(nz, -1))
            hx += np.hstack(parts)
        hx *= d_ia
        return hx

    return vind, e_ia ** 2
