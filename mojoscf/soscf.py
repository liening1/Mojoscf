"""Second-order SCF (``mf.newton()``, :mod:`pyscf.soscf.newton_ah`) with the occupied-virtual orbital Hessian.

pyscf's co-iterative augmented-Hessian solver applies the orbital Hessian to
one vector at a time, several times per macro iteration: every product
builds an AO first-order density and runs it through ``gen_response`` (J/K
and the XC kernel in the AO basis) before projecting back.  For RHF/RKS and
UHF/UKS objects accelerated by mojoscf, ``gen_g_hop`` (installed on pyscf's
``_SecondOrderRHF``/``_SecondOrderUHF``) returns the same gradient, Hessian
product and diagonal with the product from the operators of
:mod:`mojoscf.stability`, set up once per macro iteration for its orbitals
and Fock matrix: Coulomb and exchange from the MO-basis DF tensors (or
MO-basis ERIs, or one call of the integral-direct kernel), the XC kernel
response from the factors of the first-order density (:mod:`mojoscf.tdscf`).
Symmetry-adapted, solvent, dual-basis, ROHF/GHF objects and the cases
:mod:`mojoscf.tdscf` leaves to pyscf keep pyscf's ``gen_g_hop``; the solver,
its step control and the orbital update are pyscf's.
"""
from __future__ import annotations

from functools import reduce

import numpy as np
from pyscf.lib import logger

from . import integrals

__all__ = ["gen_g_hop", "install"]


def _applies(mf, mo_coeff, with_symmetry):
    from pyscf.scf import hf, rohf, uhf

    from .dft import _MojoKSHook
    from .scf import _MojoGlueMixin, _MojoHFHook

    if integrals.engine() != "mojo" or not isinstance(mf, (_MojoKSHook, _MojoHFHook, _MojoGlueMixin)):
        return False
    if isinstance(mf, rohf.ROHF) or not isinstance(mf, (hf.RHF, uhf.UHF)):
        return False
    if not np.isrealobj(mo_coeff) or (with_symmetry and mf.mol.symmetry):
        return False
    if getattr(mf, "_scf", None) is not None and mf._scf.mol is not mf.mol:
        return False
    try:
        from pyscf.solvent._attach_solvent import _Solvation

        if isinstance(mf, _Solvation):
            return False
    except ImportError:  # pragma: no cover - older pyscf
        pass
    if isinstance(mf, hf.KohnShamDFT):
        try:
            mf._numint.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
        except NotImplementedError:
            return False
    return True


def gen_g_hop(mf, mo_coeff, mo_occ, fock_ao=None, h1e=None, with_symmetry=True):
    """``(g, h_op, h_diag)`` of pyscf's ``gen_g_hop_rhf``/``gen_g_hop_uhf`` with the occupied-virtual Hessian
    product, or None where pyscf's has to run."""
    from pyscf.scf import uhf

    from . import stability, tdscf

    if not _applies(mf, mo_coeff, with_symmetry):
        return None
    unrestricted = isinstance(mf, uhf.UHF)
    if unrestricted:
        idx = [(np.where(mo_occ[s] > 0)[0], np.where(mo_occ[s] == 0)[0]) for s in range(2)]
        cs = [np.asarray(mo_coeff[s]) for s in range(2)]
    else:
        idx = [(np.where(mo_occ == 2)[0], np.where(mo_occ == 0)[0])]
        cs = [np.asarray(mo_coeff)]
    chans = [tdscf._Channel(c[:, o], c[:, v], np.zeros((len(o), len(v)))) for c, (o, v) in zip(cs, idx)]
    if any(ch.nocc == 0 or ch.nvir == 0 for ch in chans):
        return None
    parts = tdscf._response_parts(mf, chans, True, True, mf.max_memory, logger.new_logger(mf),
                                  mo_coeff=mo_coeff, mo_occ=mo_occ)
    if parts is None:
        return None
    two, xc = parts
    if fock_ao is None:
        dm0 = mf.make_rdm1(mo_coeff, mo_occ)
        fock_ao = mf.get_fock(h1e, dm=dm0)
    focks = [np.asarray(fock_ao[s]) for s in range(2)] if unrestricted else [np.asarray(fock_ao)]
    fock = [(reduce(np.dot, (ch.orbv.T, f, ch.orbv)), reduce(np.dot, (ch.orbo.T, f, ch.orbo)),
             reduce(np.dot, (ch.orbv.T, f, ch.orbo))) for ch, f in zip(chans, focks)]
    if unrestricted:
        g = np.hstack([f[2].ravel() for f in fock])
        h_diag = np.hstack([(f[0].diagonal()[:, None] - f[1].diagonal()).ravel() for f in fock])
        # pyscf's h_op: dm1_s = d_s + d_s^T with d_s = C_v x_s C_o^T
        aop = stability._vo_op(chans, fock, two, xc, 1.0, 1.0, 2.0, 1.0)
    else:
        fvv, foo, gvo = fock[0]
        g = gvo.ravel() * 2
        h_diag = (fvv.diagonal()[:, None] - foo.diagonal()).ravel() * 2
        # pyscf's h_op: dm1 = 2 (d + d^T), and it returns twice the product
        aop = stability._vo_op(chans, fock, two, xc, 1.0, 2.0, 4.0, 2.0)

    def h_op(x):
        return aop([np.asarray(x).ravel()])[0]

    return g, h_op, h_diag


def install():
    """Put the dispatching ``gen_g_hop`` on pyscf's second-order SCF classes of RHF/RKS and UHF/UKS objects."""
    from pyscf.scf import uhf
    from pyscf.soscf import newton_ah

    for cls, unrestricted in ((newton_ah._SecondOrderRHF, False), (newton_ah._SecondOrderUHF, True)):
        orig = cls.__dict__.get("gen_g_hop")
        if orig is None or getattr(orig, "_mojoscf_orig", None) is not None:
            continue

        def dispatch(mf, mo_coeff, mo_occ, fock_ao=None, h1e=None, with_symmetry=True, _orig=orig,
                     _unrestricted=unrestricted):
            out = None
            if isinstance(mf, uhf.UHF) == _unrestricted:
                out = gen_g_hop(mf, mo_coeff, mo_occ, fock_ao, h1e, with_symmetry)
            return _orig(mf, mo_coeff, mo_occ, fock_ao, h1e, with_symmetry) if out is None else out

        dispatch.__doc__ = orig.__doc__
        dispatch._mojoscf_orig = orig
        cls.gen_g_hop = dispatch
