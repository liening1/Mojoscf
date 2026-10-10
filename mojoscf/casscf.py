"""DF-CASSCF integrals from the Mojo DF kernels (pyscf's ``mcscf.df._ERIS`` without the disk round trip).

pyscf's DF-CASSCF (``mcscf.CASSCF`` of a density-fitted reference) rebuilds
its MO-basis integrals in every macro iteration: the DF tensor is
transformed to ``(Q|pq)`` for all MO pairs, written to a temporary HDF5 file
(``naux * nmo**2`` doubles) and read back to form ``(pq|uv)`` with the
active orbitals u, v.  :class:`ERIS` transforms the tensor with
:func:`mojoscf.kernels.df_mo` and forms from it the Coulomb and exchange
diagonals ``j_pc``/``k_pc``, ``ppaa = (pq|uv)`` and ``papa = (pu|qv)``.
When the transformed tensor fits in memory it is kept for the macro
iteration, and the core potential ``vhf_c`` and the J/K of the orbital
Hessian steps (``update_jk_in_ah``, two AO J/K builds per step in pyscf) are
contracted in the MO basis (:meth:`ERIS.update_jk_in_ah`); otherwise the
tensor is processed in blocks of auxiliary functions and those J/K builds
run on the DF object (:class:`mojoscf.dft.MojoDF`).  The integrals are the
same as pyscf's; :func:`install` makes pyscf's DF-CASSCF use them for
:class:`~mojoscf.dft.MojoDF` objects whose tensor is in memory (or
memory-mapped) when they fit in ``max_memory``, and keeps pyscf's otherwise.
"""
from __future__ import annotations

import numpy as np
from pyscf import lib
from pyscf.lib import logger

from . import kernels


class ERIS:
    """pyscf's DF-CASSCF integrals: ``j_pc``, ``k_pc`` (nmo, ncore), ``ppaa`` (nmo, nmo, ncas, ncas),
    ``papa`` (nmo, ncas, nmo, ncas) and ``vhf_c`` (nmo, nmo), from the DF tensor ``cderi`` (naux, npair).

    With ``keep`` the MO-basis tensor ``bmo`` (naux, nmo, nmo) is kept for
    :meth:`update_jk_in_ah` (``bcore``, ``bact``: views of its core and active rows).
    """

    def __init__(self, casscf, mo, cderi, max_memory, keep=False):
        log = logger.new_logger(casscf)
        t0 = (logger.process_clock(), logger.perf_counter())
        self._mo_in = mo
        self.mo = mo = np.ascontiguousarray(mo, dtype=np.float64)
        self.ncore = ncore = casscf.ncore
        self.ncas = ncas = casscf.ncas
        self.bmo = self.bcore = self.bact = None
        if keep:
            self._build_kept(mo, cderi, ncore, ncas)
            log.timer("mojoscf DF-CASSCF integrals (MO-basis tensor kept)", *t0)
        else:
            self._build_blocked(mo, cderi, ncore, ncas, max_memory)
            log.timer("mojoscf DF-CASSCF integrals", *t0)
            dm_core = mo[:, :ncore] @ mo[:, :ncore].T
            vj, vk = casscf.get_jk(casscf.mol, dm_core)
            self.vhf_c = mo.T @ (vj * 2 - vk) @ mo

    def _build_kept(self, mo, cderi, ncore, ncas):
        nmo = mo.shape[1]
        nocc = ncore + ncas
        naux = cderi.shape[0]
        b = self.bmo = kernels.df_mo(cderi, mo, mo)                      # (Q|pq)
        self.bcore = b[:, :ncore]
        self.bact = b[:, ncore:nocc]
        d = np.einsum("kii->ki", b)
        self.j_pc = d.T @ d[:, :ncore]
        self.k_pc = np.ascontiguousarray(np.einsum("kij,kij->ij", self.bcore, self.bcore).T)
        b2 = b.reshape(naux, nmo * nmo)
        baa = np.ascontiguousarray(self.bact[:, :, ncore:nocc]).reshape(naux, ncas * ncas)
        self.ppaa = lib.dot(b2.T, baa).reshape(nmo, nmo, ncas, ncas)
        pa = np.ascontiguousarray(self.bact.transpose(0, 2, 1)).reshape(naux, nmo * ncas)
        self.papa = lib.dot(pa.T, pa).reshape(nmo, ncas, nmo, ncas)
        # vhf_c = 2 J - K of the core density in the MO basis: J from the core diagonals,
        # K = sum_Q B_Q[:, c] B_Q[c, :] (the core columns picked by a selector)
        rho = d[:, :ncore].sum(axis=1)
        self.vhf_c = (rho @ b2).reshape(nmo, nmo) * 2
        kernels.df_sandwich(b, np.eye(nmo, ncore), self.bcore, 1, -1.0, out=self.vhf_c)

    def _build_blocked(self, mo, cderi, ncore, ncas, max_memory):
        nmo = mo.shape[1]
        nocc = ncore + ncas
        naux = cderi.shape[0]
        self.j_pc = np.zeros((nmo, ncore))
        k_cp = np.zeros((ncore, nmo))
        ppaa = np.zeros((nmo * nmo, ncas * ncas))
        bufpa = np.empty((naux, nmo, ncas))
        blk = max(1, min(naux, int(max_memory * 0.25e6 / 8 / (nmo * nmo))))
        for q0 in range(0, naux, blk):
            q1 = min(q0 + blk, naux)
            b = kernels.df_mo(cderi[q0:q1], mo, mo)                     # (Q|pq) of the block
            d = np.einsum("kii->ki", b)
            self.j_pc += d.T @ d[:, :ncore]
            bc = b[:, :ncore]
            k_cp += np.einsum("kij,kij->ij", bc, bc)
            bufpa[q0:q1] = b[:, :, ncore:nocc]
            baa = np.ascontiguousarray(b[:, ncore:nocc, ncore:nocc]).reshape(q1 - q0, ncas * ncas)
            lib.dot(b.reshape(q1 - q0, -1).T, baa, 1.0, ppaa, 1.0)
            b = d = bc = baa = None
        self.k_pc = np.ascontiguousarray(k_cp.T)
        self.ppaa = ppaa.reshape(nmo, nmo, ncas, ncas)
        pa = bufpa.reshape(naux, nmo * ncas)
        self.papa = lib.dot(pa.T, pa).reshape(nmo, ncas, nmo, ncas)

    def same_mo(self, mo) -> bool:
        """True when ``mo`` are the orbitals of these integrals."""
        return mo is self._mo_in or (np.shape(mo) == self.mo.shape and np.array_equal(mo, self.mo))

    def update_jk_in_ah(self, r, casdm1):
        """pyscf's ``CASSCF.update_jk_in_ah`` from the kept MO-basis tensor B_Q = (pq|Q).

        For the rotation ``r`` (nmo, nmo) and the active density ``casdm1``
        pyscf builds the AO densities D3 = C X3 C^T, X3 = R + R^T with R the
        core-noncore block of ``r``, and D4 = C (S + S^T) C^T, S = casdm1
        r[active] in the active rows, and returns ``va = casdm1 C_a^T (2J - K)[D3] C``
        and ``vc = C_c^T (2J - K)[2 D3 + D4] C_x``.  Only these rows of the
        MO-basis matrices are needed: J from rho_Q = sum_rs B_Q,rs X_rs, K
        from sum_Q (B_Q[rows] X) B_Q (:func:`mojoscf.kernels.df_sandwich`).
        """
        ncore, ncas = self.ncore, self.ncas
        nocc = ncore + ncas
        naux, nmo = self.bmo.shape[:2]
        r = np.asarray(r, dtype=np.float64)
        rc = np.zeros((ncore, nmo))
        rc[:, ncore:] = r[:ncore, ncore:]
        sa = np.asarray(casdm1, dtype=np.float64) @ r[ncore:nocc]
        x3 = np.zeros((nmo, nmo))
        x3[:ncore] = rc
        x3 += x3.T
        x34 = x3 * 2
        x34[ncore:nocc] += sa
        x34[:, ncore:nocc] += sa.T
        b2 = self.bmo.reshape(naux, nmo * nmo)
        bc2 = b2[:, :ncore * nmo]
        ba2 = b2[:, ncore * nmo:nocc * nmo]
        rho3 = bc2 @ rc.ravel() * 2
        rho34 = rho3 * 2 + ba2 @ sa.ravel() * 2
        v3 = (rho3 @ ba2).reshape(ncas, nmo) * 2
        v34 = (rho34 @ bc2).reshape(ncore, nmo) * 2
        kernels.df_sandwich(self.bact, x3, self.bmo, 1, -1.0, out=v3)
        kernels.df_sandwich(self.bcore, x34, self.bmo, 1, -1.0, out=v34)
        return casdm1 @ v3, v34[:, ncore:]


def _memory_mb(nmo, ncore, ncas, naux, keep):
    """MB that :class:`ERIS` keeps (ppaa, papa and the active columns, and with ``keep`` the MO-basis
    tensor and the work arrays of :meth:`ERIS.update_jk_in_ah`)."""
    words = 2 * nmo * nmo * ncas * ncas + naux * nmo * ncas
    if keep:
        words += naux * nmo * nmo + 4 * nmo * nmo
    return words * 8e-6


def make_eris(casscf, mo, with_df):
    """:class:`ERIS` for a :class:`~mojoscf.dft.MojoDF` whose tensor is in memory or memory-mapped and
    when the integrals fit in ``max_memory`` (the MO-basis tensor kept when it fits too); None otherwise
    (pyscf's ``_ERIS`` then runs)."""
    from . import dft

    if not isinstance(with_df, dft.MojoDF) or not np.isrealobj(mo) or np.ndim(mo) != 2:
        return None
    if not isinstance(casscf.ncore, (int, np.integer)) or not isinstance(casscf.ncas, (int, np.integer)):
        return None
    cderi = with_df._cderi
    if not isinstance(cderi, np.ndarray):
        cderi = dft.ondisk_tensor(with_df) if cderi is not None else None
    nao, nmo = np.shape(mo)
    if (not isinstance(cderi, np.ndarray) or cderi.ndim != 2 or cderi.dtype != np.float64
            or cderi.shape[1] != nao * (nao + 1) // 2):
        return None
    naux = cderi.shape[0]
    free = casscf.max_memory - lib.current_memory()[0]
    keep = _memory_mb(nmo, casscf.ncore, casscf.ncas, naux, True)
    if keep < 0.8 * free:
        return ERIS(casscf, mo, cderi, free - keep, keep=True)
    blocked = _memory_mb(nmo, casscf.ncore, casscf.ncas, naux, False)
    if blocked + nmo * nmo * 8e-6 * 64 > 0.8 * free:
        return None
    return ERIS(casscf, mo, cderi, free - blocked)


_installed = False


def install():
    """Make pyscf's DF-CASSCF build its integrals with :func:`make_eris` for :class:`~mojoscf.dft.MojoDF`
    objects (``pyscf.mcscf.df._ERIS``), and run the J/K of its orbital Hessian steps with
    :meth:`ERIS.update_jk_in_ah` when they have the MO-basis tensor (``_DFCASSCF.update_jk_in_ah``).
    Idempotent; other DF objects keep pyscf's code."""
    global _installed
    if _installed:
        return
    from pyscf.mcscf import df as mcscf_df

    orig = mcscf_df._ERIS

    def _ERIS(casscf, mo, with_df):
        eris = make_eris(casscf, mo, with_df)
        return eris if eris is not None else orig(casscf, mo, with_df)

    _ERIS._mojoscf_orig = orig
    mcscf_df._ERIS = _ERIS
    cls = getattr(mcscf_df, "_DFCASSCF", None)
    if cls is not None:
        own = cls.__dict__.get("update_jk_in_ah")

        def update_jk_in_ah(self, mo, r, casdm1, eris):
            if isinstance(eris, ERIS) and eris.bmo is not None and eris.same_mo(mo):
                return eris.update_jk_in_ah(r, casdm1)
            if own is not None:
                return own(self, mo, r, casdm1, eris)
            return super(cls, self).update_jk_in_ah(mo, r, casdm1, eris)

        update_jk_in_ah._mojoscf_orig = own
        cls.update_jk_in_ah = update_jk_in_ah
    _installed = True
