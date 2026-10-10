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


def _incore_tensor(mc, mo, with_df):
    """The DF tensor (naux, nao*(nao+1)//2) of a :class:`~mojoscf.dft.MojoDF` in memory or memory-mapped,
    for real orbitals ``mo`` (nao, nmo) and integer ``ncore``/``ncas`` of ``mc``; None otherwise."""
    from . import dft

    if not isinstance(with_df, dft.MojoDF) or not np.isrealobj(mo) or np.ndim(mo) != 2:
        return None
    if not isinstance(mc.ncore, (int, np.integer)) or not isinstance(mc.ncas, (int, np.integer)):
        return None
    cderi = with_df._cderi
    if not isinstance(cderi, np.ndarray):
        cderi = dft.ondisk_tensor(with_df) if cderi is not None else None
    nao = np.shape(mo)[0]
    if (not isinstance(cderi, np.ndarray) or cderi.ndim != 2 or cderi.dtype != np.float64
            or cderi.shape[1] != nao * (nao + 1) // 2):
        return None
    return cderi


def make_eris(casscf, mo, with_df):
    """:class:`ERIS` for a :class:`~mojoscf.dft.MojoDF` whose tensor is in memory or memory-mapped and
    when the integrals fit in ``max_memory`` (the MO-basis tensor kept when it fits too); None otherwise
    (pyscf's ``_ERIS`` then runs)."""
    cderi = _incore_tensor(casscf, mo, with_df)
    if cderi is None:
        return None
    nmo = np.shape(mo)[1]
    naux = cderi.shape[0]
    free = casscf.max_memory - lib.current_memory()[0]
    keep = _memory_mb(nmo, casscf.ncore, casscf.ncas, naux, True)
    if keep < 0.8 * free:
        return ERIS(casscf, mo, cderi, free - keep, keep=True)
    blocked = _memory_mb(nmo, casscf.ncore, casscf.ncas, naux, False)
    if blocked + nmo * nmo * 8e-6 * 64 > 0.8 * free:
        return None
    return ERIS(casscf, mo, cderi, free - blocked)


class NEVPT2ERIS(dict):
    """pyscf's DF-NEVPT2 integrals (the dict of ``pyscf.mrpt.dfnevpt2._ERIS``) without ``cvcv``.

    (cv|cv), (ncore nvir)^2 doubles, is used only by the Sijrs subspace;
    the DF factors ``cv`` = (Q|cv) (naux, ncore, nvir) are kept instead and
    :func:`sijrs` evaluates that subspace from them.  The Srsi and Srs
    subspaces of such integrals run :func:`srsi` and :func:`srs`.
    """


def nevpt2_eris(mc, mo, with_df):
    """pyscf's DF-NEVPT2 integrals (``vhf_c``, ``ppaa``, ``papa``, ``pacv``, ``h1eff``) for a
    :class:`~mojoscf.dft.MojoDF` whose tensor is in memory or memory-mapped, as :class:`NEVPT2ERIS`; None
    when that does not apply or they do not fit in ``max_memory`` (pyscf's ``_ERIS`` then runs).

    Two partial transforms replace pyscf's four: (Q|up) with the active
    orbitals u (``papa``, ``pacv`` and the (Q|uv) block) and (Q|cv).
    ``ppaa`` = (pq|uv) for all orbital pairs comes from the AO matrices
    sum_Q E_Q (Q|uv), one per active pair, transformed to the MO basis: no
    full (Q|pq) transform.
    """
    cderi = _incore_tensor(mc, mo, with_df)
    if cderi is None:
        return None
    log = logger.new_logger(mc)
    t0 = (logger.process_clock(), logger.perf_counter())
    mo = np.ascontiguousarray(mo, dtype=np.float64)
    nao, nmo = mo.shape
    ncore, ncas = mc.ncore, mc.ncas
    nocc = ncore + ncas
    nvir = nmo - nocc
    naux = cderi.shape[0]
    words = (2 * nmo * nmo * ncas * ncas + nmo * ncas * ncore * nvir + naux * ncore * nvir
             + naux * ncas * nmo + 3 * ncas * ncas * nao * nao)
    if words * 8e-6 > 0.8 * (mc.max_memory - lib.current_memory()[0]):
        return None
    ap = kernels.df_mo(cderi, mo[:, ncore:nocc], mo)                      # (Q|up)
    cv = kernels.df_mo(cderi, mo[:, :ncore], mo[:, nocc:])                # (Q|cv)
    pa = np.ascontiguousarray(ap.transpose(0, 2, 1)).reshape(naux, nmo * ncas)
    papa = lib.dot(pa.T, pa).reshape(nmo, ncas, nmo, ncas)
    pacv = lib.dot(pa.T, cv.reshape(naux, ncore * nvir)).reshape(nmo, ncas, ncore, nvir)
    aa = np.ascontiguousarray(ap[:, :, ncore:nocc]).reshape(naux, ncas * ncas)
    ap = pa = None
    w = lib.unpack_tril(lib.dot(aa.T, cderi))                             # sum_Q (Q|uv) E_Q
    ppaa = np.matmul(mo.T, np.matmul(w, mo))                              # (uv, p, q)
    w = None
    ppaa = np.ascontiguousarray(ppaa.reshape(ncas * ncas, nmo * nmo).T).reshape(nmo, nmo, ncas, ncas)
    dmcore = mo[:, :ncore] @ mo[:, :ncore].T
    vj, vk = mc._scf.get_jk(mc.mol, dmcore)
    vhf_c = mo.T @ (vj * 2 - vk) @ mo
    eris = NEVPT2ERIS(vhf_c=vhf_c, ppaa=ppaa, papa=papa, pacv=pacv, cvcv=None,
                      h1eff=mo.T @ mc.get_hcore() @ mo + vhf_c)
    eris.cv = cv
    log.timer("mojoscf DF-NEVPT2 integrals", *t0)
    return eris


def sijrs(nevpt, eris, verbose=None):
    """pyscf's Sijrs subspace (``pyscf.mrpt.nevpt2.Sijrs``: norm and energy of the doubly external
    core -> virtual excitations) from the DF factors of :class:`NEVPT2ERIS`.

    For each core orbital i, (ia|jb) = sum_Q (Q|ia)(Q|jb) for j <= i is one
    GEMM and is contracted with 2(ia|jb) - (ib|ja) as pyscf does with the
    stored (cv|cv); the terms are symmetric under (ia) <-> (jb), so the pairs
    j < i count twice and half of (cv|cv) is ever formed.
    """
    cv = eris.cv
    naux, ncore, nvir = cv.shape
    nocc = nevpt.ncore + nevpt.ncas
    mo_energy = np.asarray(nevpt.mo_energy)
    eia = mo_energy[:ncore, None] - mo_energy[None, nocc:]
    cv2 = cv.reshape(naux, ncore * nvir)
    norm = e = 0.0
    for i in range(ncore):
        k = i + 1
        g = (cv[:, i].T @ cv2[:, :k * nvir]).reshape(nvir, k, nvir)          # (ia|jb), j <= i
        theta = g * 2 - g.transpose(2, 1, 0)
        wgt = np.full(k, 2.0)
        wgt[i] = 1.0
        norm += np.einsum("ajb,ajb->j", g, theta) @ wgt
        g /= eia[i][:, None, None] + eia[None, :k, :]
        e += np.einsum("ajb,ajb->j", g, theta) @ wgt
    return norm, e


def srsi(nevpt, dms, eris, verbose=None):
    """pyscf's Srsi subspace (``pyscf.mrpt.nevpt2.Srsi``, the r <= s sums of its current version) with
    the three-index contractions sum_pa (rs|ip) m_pa (rs|ia) as a GEMM over p and a two-index reduction."""
    from pyscf.mrpt import nevpt2 as pnev

    ncore, ncas = nevpt.ncore, nevpt.ncas
    nocc = ncore + ncas
    dm1, dm2 = dms["1"], dms["2"]
    h1e = eris["h1eff"][ncore:nocc, ncore:nocc]
    h2e = eris["ppaa"][ncore:nocc, ncore:nocc].transpose(0, 2, 1, 3)
    h2e_v = np.ascontiguousarray(eris["pacv"][nocc:].transpose(3, 0, 2, 1))     # (r, s, i, p)
    nvir = h2e_v.shape[0]
    k27 = pnev.make_k27(h1e, h2e, dm1, dm2)
    vi_diag = np.diag_indices(nvir)
    vi_triu = np.triu_indices(nvir)

    def contract(m):
        x = (h2e_v.reshape(-1, ncas) @ m).reshape(h2e_v.shape)
        out = 2.0 * np.einsum("rsia,rsia->rsi", x, h2e_v) - np.einsum("rsia,sria->rsi", x, h2e_v)
        out += out.transpose(1, 0, 2)
        out[vi_diag] *= 0.5
        return out

    norm = contract(dm1)
    h = contract(k27)
    mo_energy = np.asarray(nevpt.mo_energy)
    diff = mo_energy[nocc:, None, None] + mo_energy[None, nocc:, None] - mo_energy[None, None, :ncore]
    return pnev._norm_to_energy(norm[vi_triu], h[vi_triu], diff[vi_triu])


def srs(nevpt, dms, eris, verbose=None):
    """pyscf's Srs subspace (``pyscf.mrpt.nevpt2.Srs``) with sum_{pq,ab} (rs|qp)(rs|ba) m_pqab as one GEMM
    (nvir^2 x ncas^2 x ncas^2) and a row-wise dot product."""
    from pyscf.mrpt import nevpt2 as pnev

    ncore, ncas = nevpt.ncore, nevpt.ncas
    nocc = ncore + ncas
    nvir = eris["papa"].shape[0] - nocc
    if nvir == 0:
        return 0, 0
    h1e = eris["h1eff"][ncore:nocc, ncore:nocc]
    h2e = eris["ppaa"][ncore:nocc, ncore:nocc].transpose(0, 2, 1, 3)
    hv = np.ascontiguousarray(eris["papa"][nocc:, :, nocc:].transpose(0, 2, 1, 3)).reshape(nvir * nvir, ncas * ncas)
    rm2, a7 = pnev.make_a7(h1e, h2e, dms["1"], dms["2"], dms["3"])
    n2 = ncas * ncas
    norm = 0.5 * np.einsum("xy,xy->x", hv @ rm2.transpose(1, 0, 2, 3).reshape(n2, n2), hv).reshape(nvir, nvir)
    h = 0.5 * np.einsum("xy,xy->x", hv @ a7.transpose(1, 0, 3, 2).reshape(n2, n2), hv).reshape(nvir, nvir)
    mo_energy = np.asarray(nevpt.mo_energy)
    return pnev._norm_to_energy(norm, h, mo_energy[nocc:, None] + mo_energy[None, nocc:])


_installed = False


def install():
    """Make pyscf's DF-CASSCF build its integrals with :func:`make_eris` for :class:`~mojoscf.dft.MojoDF`
    objects (``pyscf.mcscf.df._ERIS``) and run the J/K of its orbital Hessian steps with
    :meth:`ERIS.update_jk_in_ah` when they have the MO-basis tensor (``_DFCASSCF.update_jk_in_ah``), and
    DF-NEVPT2 build its integrals with :func:`nevpt2_eris` (``pyscf.mrpt.dfnevpt2._ERIS``; the Sijrs,
    Srsi and Srs subspaces of those integrals from :func:`sijrs`, :func:`srsi`, :func:`srs`).  Idempotent;
    other DF objects keep pyscf's code."""
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
    from pyscf.mrpt import dfnevpt2, nevpt2

    orig_nevpt2 = dfnevpt2._ERIS

    def _NEVPT2_ERIS(mc, mo, with_df, method="incore"):
        eris = nevpt2_eris(mc, mo, with_df)
        return eris if eris is not None else orig_nevpt2(mc, mo, with_df, method)

    _NEVPT2_ERIS._mojoscf_orig = orig_nevpt2
    dfnevpt2._ERIS = _NEVPT2_ERIS
    orig_sijrs, orig_srsi, orig_srs = nevpt2.Sijrs, nevpt2.Srsi, nevpt2.Srs

    def Sijrs(mc, eris, verbose=None):
        if isinstance(eris, NEVPT2ERIS):
            return sijrs(mc, eris, verbose)
        return orig_sijrs(mc, eris, verbose)

    def Srsi(mc, dms, eris, verbose=None):
        if isinstance(eris, NEVPT2ERIS):
            return srsi(mc, dms, eris, verbose)
        return orig_srsi(mc, dms, eris, verbose)

    def Srs(mc, dms, eris=None, verbose=None):
        if isinstance(eris, NEVPT2ERIS):
            return srs(mc, dms, eris, verbose)
        return orig_srs(mc, dms, eris, verbose)

    Sijrs._mojoscf_orig, Srsi._mojoscf_orig, Srs._mojoscf_orig = orig_sijrs, orig_srsi, orig_srs
    nevpt2.Sijrs, nevpt2.Srsi, nevpt2.Srs = Sijrs, Srsi, Srs
    _installed = True
