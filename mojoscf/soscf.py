"""Second-order SCF (``mf.newton()``, :mod:`pyscf.soscf.newton_ah`) with the occupied-virtual orbital Hessian.

pyscf's co-iterative augmented-Hessian solver applies the orbital Hessian to
one vector at a time, several times per macro iteration: every product
builds an AO first-order density and runs it through ``gen_response`` (J/K
and the XC kernel in the AO basis) before projecting back.  For RHF/RKS,
UHF/UKS and ROHF/ROKS objects accelerated by mojoscf, ``gen_g_hop``
(installed on pyscf's ``_SecondOrderRHF``/``_SecondOrderUHF``/``_SecondOrderROHF``)
returns the same gradient, Hessian product and diagonal with the product
from the operators of :mod:`mojoscf.stability`, set up once per macro
iteration for its orbitals and Fock matrix: Coulomb and exchange from the
MO-basis DF tensors, or with exact integrals from the J/K of the first-order
AO density (Mojo kernels, in-core ERIs or integral-direct: with a handful of
products per macro iteration this beats transforming the ERIs to the MO
basis each time), the XC kernel response from the factors of the
first-order density (:mod:`mojoscf.tdscf`).  ROHF objects take pyscf's
route through the UHF Hessian of the alpha (occupied: occupation > 0) and
beta (occupation 2) orbitals, folded onto the ROHF rotations.  With a
solvent model (``mf.PCM()``, SMD, ...) the products are those of pyscf's
solvated ``gen_response``: the gas-phase response, plus with
``with_solvent.equilibrium_solvation`` the solvent's ``_B_dot_x`` of the
first-order density for all vectors at once (the Mojo PCM kernels when the
solvent has them, :mod:`mojoscf.solvent`).  Symmetry-adapted, dual-basis and
GHF objects, ROHF with equilibrium solvation and the cases
:mod:`mojoscf.tdscf` leaves to pyscf keep pyscf's ``gen_g_hop``; the solver,
its step control and the orbital update are pyscf's.
"""
from __future__ import annotations

from functools import reduce

import numpy as np
from pyscf.lib import logger

from . import integrals

__all__ = ["gen_g_hop", "install"]


def _kind(mf):
    from pyscf.scf import rohf, uhf

    if isinstance(mf, uhf.UHF):
        return "uhf"
    return "rohf" if isinstance(mf, rohf.ROHF) else "rhf"


def _applies(mf, mo_coeff, with_symmetry):
    from pyscf.scf import hf, uhf

    from .dft import _MojoKSHook
    from .scf import _MojoGlueMixin, _MojoHFHook

    if integrals.engine() != "mojo" or not isinstance(mf, (_MojoKSHook, _MojoHFHook, _MojoGlueMixin)):
        return False
    if not isinstance(mf, (hf.RHF, uhf.UHF)):
        return False
    if not np.isrealobj(mo_coeff) or (with_symmetry and mf.mol.symmetry):
        return False
    if getattr(mf, "_scf", None) is not None and mf._scf.mol is not mf.mol:
        return False
    solvent = _solvent(mf)
    if solvent is not None and (_kind(mf) == "rohf" or not hasattr(solvent, "_B_dot_x")):
        # pyscf's ROHF route adds the solvent response of each spin's density to that spin only
        return False
    if isinstance(mf, hf.KohnShamDFT):
        try:
            mf._numint.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
        except NotImplementedError:
            return False
    return True


def _solvent(mf):
    """The solvent object whose response pyscf's ``gen_response`` of ``mf`` adds (equilibrium solvation),
    or None."""
    try:
        from pyscf.solvent._attach_solvent import _Solvation
    except ImportError:  # pragma: no cover - older pyscf
        return None
    if not isinstance(mf, _Solvation) or not mf.with_solvent.equilibrium_solvation:
        return None
    return mf.with_solvent


def _with_solvent(aop, chans, solvent, dm_scale, scale):
    """``aop`` plus pyscf's equilibrium solvent response: ``solvent._B_dot_x`` of the first-order density
    ``dm_scale sum_s (d_s + d_s^T)``, d_s = C_v x_s C_o^T, projected on every channel and scaled by
    ``scale`` like the rest of the product."""
    sizes = [ch.nvir * ch.nocc for ch in chans]

    def op(xs):
        xs = np.asarray(xs).reshape(len(xs), -1)
        out = np.array(aop(xs))
        dm1 = 0.0
        p0 = 0
        for ch, k in zip(chans, sizes):
            d = ch.orbv @ xs[:, p0:p0 + k].reshape(-1, ch.nvir, ch.nocc) @ ch.orbo.T
            dm1 = dm1 + d + d.transpose(0, 2, 1)
            p0 += k
        v = solvent._B_dot_x(dm_scale * dm1)
        p0 = 0
        for ch, k in zip(chans, sizes):
            out[:, p0:p0 + k] += scale * (ch.orbv.T @ v @ ch.orbo).reshape(len(xs), k)
            p0 += k
        return list(out)

    return op


def _vo_parts(mf, cs, idx, focks, unrestricted, xc_coeff, xc_occ):
    """``(g, h_diag, aop)`` of pyscf's ``gen_g_hop_uhf`` (``unrestricted``, one channel per spin) or
    ``gen_g_hop_rhf`` for the orbitals ``cs`` with the occupied and virtual columns ``idx`` ((occ, vir) per
    channel) and the AO Fock matrices ``focks``, the XC kernel that of ``xc_coeff``, ``xc_occ``; or None."""
    from . import stability, tdscf

    chans = [tdscf._Channel(c[:, o], c[:, v], np.zeros((len(o), len(v)))) for c, (o, v) in zip(cs, idx)]
    if any(ch.nocc == 0 or ch.nvir == 0 for ch in chans):
        return None
    parts = tdscf._response_parts(mf, chans, None, True, mf.max_memory, logger.new_logger(mf),
                                  mo_coeff=xc_coeff, mo_occ=xc_occ, unrestricted=unrestricted, exact_mo=False)
    if parts is None:
        return None
    two, xc = parts
    fock = [(reduce(np.dot, (ch.orbv.T, f, ch.orbv)), reduce(np.dot, (ch.orbo.T, f, ch.orbo)),
             reduce(np.dot, (ch.orbv.T, f, ch.orbo))) for ch, f in zip(chans, focks)]
    g = np.hstack([f[2].ravel() for f in fock])
    h_diag = np.hstack([(f[0].diagonal()[:, None] - f[1].diagonal()).ravel() for f in fock])
    solvent = _solvent(mf)
    if unrestricted:
        # pyscf's h_op: dm1_s = d_s + d_s^T with d_s = C_v x_s C_o^T
        aop = stability._vo_op(chans, fock, two, xc, 1.0, 1.0, 2.0, 1.0)
        return g, h_diag, aop if solvent is None else _with_solvent(aop, chans, solvent, 1.0, 1.0)
    # pyscf's h_op: dm1 = 2 (d + d^T), and it returns twice the product
    aop = stability._vo_op(chans, fock, two, xc, 1.0, 2.0, 4.0, 2.0)
    return g * 2, h_diag * 2, aop if solvent is None else _with_solvent(aop, chans, solvent, 2.0, 2.0)


def gen_g_hop(mf, mo_coeff, mo_occ, fock_ao=None, h1e=None, with_symmetry=True):
    """``(g, h_op, h_diag)`` of pyscf's ``gen_g_hop_rhf``/``gen_g_hop_uhf``/``gen_g_hop_rohf`` with the
    occupied-virtual Hessian product, or None where pyscf's has to run."""
    if not _applies(mf, mo_coeff, with_symmetry):
        return None
    kind = _kind(mf)
    if kind == "rohf":
        return _gen_g_hop_rohf(mf, np.asarray(mo_coeff), np.asarray(mo_occ), fock_ao, h1e)
    if fock_ao is None:
        dm0 = mf.make_rdm1(mo_coeff, mo_occ)
        fock_ao = mf.get_fock(h1e, dm=dm0)
    if kind == "uhf":
        cs = [np.asarray(mo_coeff[s]) for s in range(2)]
        idx = [(np.where(np.asarray(mo_occ[s]) > 0)[0], np.where(np.asarray(mo_occ[s]) == 0)[0]) for s in range(2)]
        focks = [np.asarray(fock_ao[s]) for s in range(2)]
    else:
        occ = np.asarray(mo_occ)
        cs, idx, focks = [np.asarray(mo_coeff)], [(np.where(occ == 2)[0], np.where(occ == 0)[0])], [np.asarray(fock_ao)]
    parts = _vo_parts(mf, cs, idx, focks, kind == "uhf", mo_coeff, mo_occ)
    if parts is None:
        return None
    g, h_diag, aop = parts

    def h_op(x):
        return aop([np.asarray(x).ravel()])[0]

    return g, h_op, h_diag


def _gen_g_hop_rohf(mf, mo_coeff, mo_occ, fock_ao, h1e):
    """pyscf's ``gen_g_hop_rohf``: the UHF Hessian of the alpha (occupation > 0) and beta (occupation 2)
    orbitals of the same coefficients, folded onto the unique ROHF rotations."""
    if getattr(fock_ao, "focka", None) is None:
        dm0 = mf.make_rdm1(mo_coeff, mo_occ)
        fock_ao = mf.get_fock(h1e, dm=dm0)
    occa, occb = mo_occ > 0, mo_occ == 2
    idx = [(np.where(occa)[0], np.where(~occa)[0]), (np.where(occb)[0], np.where(~occb)[0])]
    parts = _vo_parts(mf, [mo_coeff, mo_coeff], idx, [np.asarray(fock_ao.focka), np.asarray(fock_ao.fockb)],
                      True, (mo_coeff, mo_coeff), (occa.astype(np.float64), occb.astype(np.float64)))
    if parts is None:
        return None
    ug, uh_diag, aop = parts
    uniq_a = ~occa[:, None] & occa
    uniq_b = ~occb[:, None] & occb
    uniq = uniq_a | uniq_b
    nmo = mo_coeff.shape[-1]
    nova = np.count_nonzero(uniq_a)

    def sum_ab(x):
        x1 = np.zeros((nmo, nmo), dtype=x.dtype)
        x1[uniq_a] = x[:nova]
        x1[uniq_b] += x[nova:]
        return x1[uniq]

    def h_op(x):
        x1 = np.zeros((nmo, nmo), dtype=np.result_type(x, np.float64))
        x1[uniq] = x
        return sum_ab(aop([np.hstack((x1[uniq_a], x1[uniq_b]))])[0])

    return sum_ab(ug), h_op, sum_ab(uh_diag)


def install():
    """Put the dispatching ``gen_g_hop`` on pyscf's second-order SCF classes of RHF/RKS, UHF/UKS and ROHF/ROKS
    objects."""
    from pyscf.soscf import newton_ah

    for cls, kind in ((newton_ah._SecondOrderRHF, "rhf"), (newton_ah._SecondOrderUHF, "uhf"),
                      (newton_ah._SecondOrderROHF, "rohf")):
        orig = cls.__dict__.get("gen_g_hop")
        if orig is None or getattr(orig, "_mojoscf_orig", None) is not None:
            continue

        def dispatch(mf, mo_coeff, mo_occ, fock_ao=None, h1e=None, with_symmetry=True, _orig=orig, _kind_of=kind):
            out = None
            if _kind(mf) == _kind_of:
                out = gen_g_hop(mf, mo_coeff, mo_occ, fock_ao, h1e, with_symmetry)
            return _orig(mf, mo_coeff, mo_occ, fock_ao, h1e, with_symmetry) if out is None else out

        dispatch.__doc__ = orig.__doc__
        dispatch._mojoscf_orig = orig
        cls.gen_g_hop = dispatch
