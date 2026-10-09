"""SCF stability analysis (``mf.stability()``, :mod:`pyscf.scf.stability`) with the occupied-virtual response.

pyscf finds the lowest eigenvalues of the orbital Hessian with Davidson's
method and forms one Hessian-vector product per call: an AO first-order
density through ``gen_response`` (J/K and the XC kernel in the AO basis) and
a projection back.  The Hessians are those of linear response (up to
pyscf's overall factors):

    internal (real orbitals)     F_vv x - x F_oo + V(x + x^T)              (TDDFT A + B, y = x)
    RHF -> UHF                   F_vv x - x F_oo + triplet V(x + x^T)
    real -> complex              F_vv x - x F_oo - c (K_A - K_B) x         (A - B, no Coulomb or XC)

so the two-electron and XC parts come from the operators of
:mod:`mojoscf.tdscf` (MO-basis DF tensors, MO-basis exact ERIs or one
call of the integral-direct kernel, the projected XC kernel), for all
vectors of a Davidson iteration at once (``lib.davidson1`` with a batched
operator in place of pyscf's per-vector ``lib.davidson``).  The Fock blocks,
initial guesses, preconditioners, thresholds and the orbital rotation are
pyscf's.  RHF/RKS (internal and external) and UHF/UKS (internal) objects
take this route; ROHF, GHF, the UHF -> GHF analysis, point-group symmetry
labels, solvent models and the cases :mod:`mojoscf.tdscf` leaves to pyscf
run pyscf's code.

>>> mf = mojoscf.dft.accelerate(dft.UKS(mol, xc="b3lyp").density_fit()).run()
>>> mo_i, mo_e, stable_i, stable_e = mf.stability(return_status=True)
"""
from __future__ import annotations

from functools import reduce

import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import stability as pstab

from . import integrals
from ._backend import serial_scipy_blas

__all__ = ["rhf_stability", "uhf_stability", "rhf_internal", "uhf_internal", "rhf_external",
           "internal_hessian", "external_hessians"]


def _supported(mf):
    """None if the occupied-virtual operators cannot stand in for pyscf's ``gen_response`` of ``mf``."""
    from pyscf.scf import hf, rohf, uhf

    if integrals.engine() != "mojo" or mf.mo_coeff is None or not np.isrealobj(mf.mo_coeff):
        return False
    if getattr(mf, "_scf", None) is not None and mf._scf.mol is not mf.mol:
        return False
    if mf.mol.symmetry:
        return False
    try:
        from pyscf.solvent._attach_solvent import _Solvation

        if isinstance(mf, _Solvation):
            return False
    except ImportError:  # pragma: no cover - older pyscf
        pass
    if isinstance(mf, rohf.ROHF) or not isinstance(mf, (hf.RHF, uhf.UHF)):
        return False
    if isinstance(mf, hf.KohnShamDFT):
        mf._numint.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
    return True


def _channels(mf):
    from pyscf.scf import uhf

    from .tdscf import _Channel

    if isinstance(mf, uhf.UHF):
        out = []
        for s in range(2):
            c, e, occ = mf.mo_coeff[s], mf.mo_energy[s], mf.mo_occ[s]
            o, v = np.where(occ > 0)[0], np.where(occ == 0)[0]
            out.append(_Channel(c[:, o], c[:, v], e[v] - e[o, None]))
        return out
    c, e, occ = mf.mo_coeff, mf.mo_energy, mf.mo_occ
    o, v = np.where(occ == 2)[0], np.where(occ == 0)[0]
    return [_Channel(c[:, o], c[:, v], e[v] - e[o, None])]


def _parts(mf, chans, singlet):
    from . import tdscf

    return tdscf._response_parts(mf, chans, singlet, True, mf.max_memory, logger.new_logger(mf))


def _fock_blocks(mf, chans):
    """[(F_vv, F_oo, g = F_vo)] per channel in the MO basis, the Fock matrix built as pyscf's ``gen_g_hop_*``."""
    from pyscf.scf import uhf

    dm0 = mf.make_rdm1(mf.mo_coeff, mf.mo_occ)
    if isinstance(mf, uhf.UHF):
        focks = mf.get_fock(dm=dm0)
    else:
        focks = [mf.get_hcore() + mf.get_veff(mf.mol, dm0)]
    out = []
    for f, ch in zip(focks, chans):
        out.append((
            reduce(np.dot, (ch.orbv.T, f, ch.orbv)), reduce(np.dot, (ch.orbo.T, f, ch.orbo)),
            reduce(np.dot, (ch.orbv.T, f, ch.orbo)),
        ))
    return out


def _davidson(aop, x0, precond, tol, log, nroots):
    """pyscf's ``lib.davidson`` (same solver and return values) with a batched operator."""
    e, x = lib.davidson1(aop, x0, precond, tol=tol, verbose=log, nroots=nroots)[1:]
    if nroots == 1:
        return e[0], x[0]
    return e, x


def _vo_op(chans, fock, two, xc, ys_sign, jscale, xc_scale, scale):
    """Batched ``x -> scale [(F_vv x - x F_oo) + two-electron + XC]`` on vo-ordered vectors of all channels.

    The two-electron part is the top block of :func:`mojoscf.tdscf._rks_operator` with ``y = ys_sign x``;
    the XC part (``xc`` None: none) that of the transition density factors ``xc_scale`` C_v x^T, C_o.
    """
    from .tdscf import _factor

    sizes = [ch.nvir * ch.nocc for ch in chans]

    def aop(xs):
        xs = np.asarray(xs).reshape(len(xs), -1)
        n = len(xs)
        out = np.empty_like(xs)
        xo = []
        p0 = 0
        for ch, k in zip(chans, sizes):
            xo.append(np.ascontiguousarray(xs[:, p0:p0 + k].reshape(n, ch.nvir, ch.nocc).transpose(0, 2, 1)))
            p0 += k
        tops = two(xo, None if ys_sign == 0 else [ys_sign * x for x in xo], jscale)[0]
        pxc = xc([_factor(ch, x, xc_scale) for ch, x in zip(chans, xo)]) if xc is not None else None
        p0 = 0
        for s, (ch, k) in enumerate(zip(chans, sizes)):
            fvv, foo, _ = fock[s]
            x = xo[s].transpose(0, 2, 1)
            v = fvv @ x - x @ foo
            v += tops[s].transpose(0, 2, 1)
            if pxc is not None:
                v += pxc[s].transpose(0, 2, 1)
            out[:, p0:p0 + k] = scale * v.reshape(n, k)
            p0 += k
        return list(out)

    return aop


def internal_hessian(mf):
    """``(aop, g, hdiag)`` of the internal stability analysis of an RHF/RKS or UHF/UKS object, or None.

    ``aop`` maps a list of vectors (vo order, alpha then beta) to the products
    with the orbital Hessian pyscf's ``rhf_internal``/``uhf_internal`` diagonalise
    (``2 h_op`` of ``newton_ah.gen_g_hop_rhf``/``gen_g_hop_uhf``, whose RHF
    ``h_op`` carries a factor 2 of its own); ``g`` and ``hdiag`` are the
    gradient and the diagonal those functions start from.
    """
    from pyscf.scf import uhf

    if not _supported(mf):
        return None
    chans = _channels(mf)
    parts = _parts(mf, chans, True)
    if parts is None:
        return None
    two, xc = parts
    fock = _fock_blocks(mf, chans)
    if isinstance(mf, uhf.UHF):
        g = np.hstack([f[2].ravel() for f in fock])
        hdiag = np.hstack([(f[0].diagonal()[:, None] - f[1].diagonal()).ravel() for f in fock]) * 2
        # h_op: dm1_s = d_s + d_s^T, d_s = C_v x_s C_o^T
        return _vo_op(chans, fock, two, xc, 1.0, 1.0, 2.0, 2.0), g, hdiag
    fvv, foo, gvo = fock[0]
    g = gvo.ravel() * 2
    hdiag = (fvv.diagonal()[:, None] - foo.diagonal()).ravel() * 4
    # h_op: dm1 = 2 (d + d^T), and pyscf's RHF h_op returns twice the product
    return _vo_op(chans, fock, two, xc, 1.0, 2.0, 4.0, 4.0), g, hdiag


def external_hessians(mf):
    """``(hop1, hop2, hdiag)`` of pyscf's ``rhf_external`` (real -> complex, RHF -> UHF) for an RHF/RKS
    object (batched operators as in :func:`internal_hessian`), or None."""
    from pyscf.scf import uhf

    if isinstance(mf, uhf.UHF) or not _supported(mf):
        return None
    chans = _channels(mf)
    parts = _parts(mf, chans, False)          # triplet kernel; its exchange serves both analyses
    if parts is None:
        return None
    two, xc = parts
    fock = _fock_blocks(mf, chans)
    fvv, foo, _ = fock[0]
    hdiag = (fvv.diagonal()[:, None] - foo.diagonal()).ravel()
    # real -> complex: dm1 = 2 (d - d^T), no Coulomb or XC; RHF -> UHF: dm1 = 2 (d + d^T), triplet kernel
    hop1 = _vo_op(chans, fock, two, None, -1.0, 0.0, 0.0, 1.0)
    hop2 = _vo_op(chans, fock, two, xc, 1.0, 0.0, 4.0, 1.0)
    return hop1, hop2, hdiag


def rhf_internal(mf, with_symmetry=True, verbose=None, return_status=False, nroots=pstab.STAB_NROOTS,
                 tol=pstab.STAB_TOL):
    """pyscf's ``stability.rhf_internal`` with the batched occupied-virtual orbital Hessian."""
    ops = internal_hessian(mf)
    if ops is None:
        return pstab.rhf_internal(mf, with_symmetry, verbose, return_status, nroots, tol)
    aop, g, hdiag = ops
    log = logger.new_logger(mf, verbose)

    def precond(dx, e, x0):
        hdiagd = hdiag - e
        hdiagd[abs(hdiagd) < 1e-8] = 1e-8
        return dx / hdiagd

    x0 = np.zeros_like(g)
    x0[g != 0] = 1.0 / hdiag[g != 0]
    if not with_symmetry:
        x0[np.argmin(hdiag)] = 1
    with serial_scipy_blas():
        e, v = _davidson(aop, x0, precond, tol, log, nroots)
    log.info("rhf_internal: lowest eigs of H = %s", e)
    if nroots != 1:
        e, v = e[0], v[0]
    stable = not (e < -1e-5)
    pstab.dump_status(log, stable, f"{mf.__class__}", "internal")
    mo = mf.mo_coeff if stable else pstab._rotate_mo(mf.mo_coeff, mf.mo_occ, v)
    return (mo, stable) if return_status else mo


def uhf_internal(mf, with_symmetry=True, verbose=None, return_status=False, nroots=pstab.STAB_NROOTS,
                 tol=pstab.STAB_TOL):
    """pyscf's ``stability.uhf_internal`` with the batched occupied-virtual orbital Hessian."""
    ops = internal_hessian(mf)
    if ops is None:
        return pstab.uhf_internal(mf, with_symmetry, verbose, return_status, nroots, tol)
    aop, g, hdiag = ops
    log = logger.new_logger(mf, verbose)

    def precond(dx, e, x0):
        hdiagd = hdiag - e
        hdiagd[abs(hdiagd) < 1e-8] = 1e-8
        return dx / hdiagd

    x0 = np.zeros_like(g)
    x0[g != 0] = 1.0 / hdiag[g != 0]
    if not with_symmetry:
        x0[np.argmin(hdiag)] = 1
    with serial_scipy_blas():
        e, v = _davidson(aop, x0, precond, tol, log, nroots)
    log.info("uhf_internal: lowest eigs of H = %s", e)
    if nroots != 1:
        e, v = e[0], v[0]
    stable = not (e < -1e-5)
    pstab.dump_status(log, stable, f"{mf.__class__}", "internal")
    if stable:
        mo = mf.mo_coeff
    else:
        nova = np.count_nonzero(mf.mo_occ[0] > 0) * np.count_nonzero(mf.mo_occ[0] == 0)
        mo = (pstab._rotate_mo(mf.mo_coeff[0], mf.mo_occ[0], v[:nova]),
              pstab._rotate_mo(mf.mo_coeff[1], mf.mo_occ[1], v[nova:]))
    return (mo, stable) if return_status else mo


def rhf_external(mf, with_symmetry=True, verbose=None, return_status=False, nroots=pstab.STAB_NROOTS,
                 tol=pstab.STAB_TOL):
    """pyscf's ``stability.rhf_external`` (real -> complex, then RHF -> UHF) with batched operators."""
    ops = external_hessians(mf)
    if ops is None:
        return pstab.rhf_external(mf, with_symmetry, verbose, return_status, nroots, tol)
    hop1, hop2, hdiag = ops
    log = logger.new_logger(mf, verbose)

    def precond(dx, e, x0):
        hdiagd = hdiag - e
        hdiagd[abs(hdiagd) < 1e-8] = 1e-8
        return dx / hdiagd

    with serial_scipy_blas():
        x0 = np.zeros_like(hdiag)
        x0[hdiag > 1e-5] = 1.0 / hdiag[hdiag > 1e-5]
        if not with_symmetry:
            x0[np.argmin(hdiag)] = 1
        e1, v1 = _davidson(hop1, x0, precond, tol, log, nroots)
        log.info("rhf_real2complex: lowest eigs of H = %s", e1)
        if nroots != 1:
            e1, v1 = e1[0], v1[0]
        pstab.dump_status(log, not (e1 < -1e-5), f"{mf.__class__}", "real -> complex")
        x0 = np.zeros_like(hdiag)
        x0[hdiag > 1e-5] = 1.0 / hdiag[hdiag > 1e-5]
        e3, v3 = _davidson(hop2, x0, precond, tol, log, nroots)
    log.info("rhf_external: lowest eigs of H = %s", e3)
    if nroots != 1:
        e3, v3 = e3[0], v3[0]
    stable = not (e3 < -1e-5)
    pstab.dump_status(log, stable, f"{mf.__class__}", "RHF/RKS -> UHF/UKS")
    mo = (mf.mo_coeff, mf.mo_coeff) if stable else (pstab._rotate_mo(mf.mo_coeff, mf.mo_occ, v3), mf.mo_coeff)
    return (mo, stable) if return_status else mo


def rhf_stability(mf, internal=True, external=False, verbose=None, return_status=False,
                  nroots=pstab.STAB_NROOTS, tol=pstab.STAB_TOL):
    """pyscf's ``stability.rhf_stability`` with :func:`rhf_internal` and :func:`rhf_external`."""
    mo_i = mo_e = None
    stable_i = stable_e = None
    if internal:
        mo_i, stable_i = rhf_internal(mf, verbose=verbose, return_status=True, nroots=nroots, tol=tol)
    if external:
        mo_e, stable_e = rhf_external(mf, verbose=verbose, return_status=True, nroots=nroots, tol=tol)
    if return_status:
        return mo_i, mo_e, stable_i, stable_e
    return mo_i, mo_e


def uhf_stability(mf, internal=True, external=False, verbose=None, return_status=False,
                  nroots=pstab.STAB_NROOTS, tol=pstab.STAB_TOL):
    """pyscf's ``stability.uhf_stability`` with :func:`uhf_internal` (the UHF -> GHF analysis is pyscf's)."""
    mo_i = mo_e = None
    stable_i = stable_e = None
    if internal:
        mo_i, stable_i = uhf_internal(mf, verbose=verbose, return_status=True, nroots=nroots, tol=tol)
    if external:
        mo_e, stable_e = pstab.uhf_external(mf, verbose=verbose, return_status=True, nroots=nroots, tol=tol)
    if return_status:
        return mo_i, mo_e, stable_i, stable_e
    return mo_i, mo_e


def stability(mf, internal=True, external=False, verbose=None, return_status=False, **kwargs):
    """``mf.stability()`` of the mojoscf classes: :func:`rhf_stability` or :func:`uhf_stability`."""
    from pyscf.scf import rohf, uhf

    if isinstance(mf, uhf.UHF):
        return uhf_stability(mf, internal, external, verbose, return_status, **kwargs)
    if isinstance(mf, rohf.ROHF):
        return pstab.rohf_stability(mf, internal, external, verbose, return_status, **kwargs)
    return rhf_stability(mf, internal, external, verbose, return_status, **kwargs)
