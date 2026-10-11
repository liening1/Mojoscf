"""Spin-traced 3-particle density matrices of FCI vectors and NEVPT2's 4-particle contractions with Mojo
kernels (``pyscf.fci.rdm.make_dm123``, ``pyscf.mrpt.nevpt2._contract4pdm``).

pyscf's ``make_dm123`` (strongly contracted NEVPT2, ``direct_spin1.make_rdm123``)
evaluates <p^+ q r^+ s t^+ u> in C for p >= r >= t: 68 s for a CAS(12,12)
vector here, against 0.3 s for the 2-particle matrix.  :func:`make_dm123`
forms the same entries from the vectors E_tu|ket> and E_qp|bra> over all
determinants with one GEMM per single excitation of each string
(:mod:`mojoscf._mojo.fcirdm`) and completes the other orderings with pyscf's
``_complete_dm3_``.  The 4-particle contractions of strongly contracted
NEVPT2 (89 s each for CAS(12,12) in pyscf) are 3-particle matrices with the
third vector O_ef|ci> = sum_x E_ex W_fx|ci> (W_fx = sum_yz (fx|yz) E_yz) or
sum_x E_xf W_ex|ci> (:func:`contract4pdm`), formed with the same kernel.
:func:`install` makes pyscf use both (when the two work vectors, 2 ndet
norb^2 doubles, fit in memory).
"""
from __future__ import annotations

import numpy as np
from pyscf import lib
from pyscf.fci import cistring
from pyscf.fci import rdm as fci_rdm

from ._backend import get_extension, worker_blas

_orig = {}


def _links(norb, nelec):
    """pyscf's link table (nstr, nlink, 4) [c, d, J, sign] (E_cd|I> = sign|J>) as int64."""
    return np.ascontiguousarray(cistring.gen_linkstr_index(range(norb), nelec), dtype=np.int64)


def _available_mb() -> float:
    """Memory the system reports available (Linux ``MemAvailable``), MB; 0 when unknown."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return 0.0


def _fits(ndet, norb) -> bool:
    """The two work vectors (ndet, norb^2) and the 6-index results fit: within pyscf's ``MAX_MEMORY`` or
    half of the memory the system has available, whichever is larger (pyscf's own C kernels allocate
    their buffers without consulting ``MAX_MEMORY`` either)."""
    need = 2 * ndet * norb * norb * 8e-6 + norb ** 6 * 8e-6 * 3
    room = max(lib.param.MAX_MEMORY - lib.current_memory()[0], 0.5 * _available_mb())
    return need < 0.8 * room


def make_dm123(fname, cibra, ciket, norb, nelec):
    """pyscf's ``fci.rdm.make_dm123``: ``(rdm1, rdm2, rdm3)`` with rdm2[p,q,r,s] = <p^+ q r^+ s> and
    rdm3[p,q,r,s,t,u] = <p^+ q r^+ s t^+ u> (spin-traced, before ``reorder_dm123``); the Mojo kernel for
    pyscf's spin-traced kernel ``FCI3pdm_kern_sf`` when the work vectors fit, pyscf's code otherwise."""
    orig = _orig.get("make_dm123", fci_rdm.make_dm123)
    if fname != "FCI3pdm_kern_sf":
        return orig(fname, cibra, ciket, norb, nelec)
    neleca, nelecb = fci_rdm._unpack_nelec(nelec)
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    bra = np.asarray(cibra)
    ket = np.asarray(ciket)
    if (not (np.isrealobj(bra) and np.isrealobj(ket)) or bra.size != na * nb or ket.size != na * nb
            or not _fits(na * nb, norb)):
        return orig(fname, cibra, ciket, norb, nelec)
    bra = np.ascontiguousarray(bra, dtype=np.float64).reshape(na, nb)
    ket = np.ascontiguousarray(ket, dtype=np.float64).reshape(na, nb)
    linka = _links(norb, neleca)
    linkb = _links(norb, nelecb)
    n2 = norb * norb
    ak = np.empty((na * nb, n2))
    ab = np.empty((na * nb, n2))
    dm1 = np.empty((norb, norb))
    dm2 = np.empty((norb,) * 4)
    dm3 = np.empty((norb,) * 6)
    path, prefix = worker_blas()
    get_extension().fci_dm123(linka, linkb, bra, ket, ak, ab, dm1, dm2, dm3, path, prefix)
    ak = ab = None
    dm3 = fci_rdm._complete_dm3_(dm2, dm3)
    return dm1.T, dm2, dm3


_F3_MODES = {"NEVPTkern_cedf_aedf": 0, "NEVPTkern_aedf_ecdf": 1}


def contract4pdm(kern, eri, civec, norb, nelec, link_index=None):
    """pyscf's ``mrpt.nevpt2._contract4pdm``: for ``NEVPTkern_cedf_aedf`` sum_xyz (fx|yz)
    <E_ab E_cd E_ex E_yz> and for ``NEVPTkern_aedf_ecdf`` sum_xyz (ex|yz) <E_ab E_cd E_xf E_yz>
    (index order [a, b, c, d, e, f]) for real ``eri`` (ncas^4, 8-fold symmetric (pq|rs)); computed for
    a >= c by the Mojo kernel and completed as pyscf does.  pyscf's code for other kernels or when the
    work vectors do not fit."""
    from pyscf.mrpt import nevpt2

    orig = _orig.get("_contract4pdm", nevpt2._contract4pdm)
    mode = _F3_MODES.get(kern)
    neleca, nelecb = fci_rdm._unpack_nelec(nelec)
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    ci = np.asarray(civec)
    eri = np.asarray(eri)
    if (mode is None or not (np.isrealobj(ci) and np.isrealobj(eri)) or ci.size != na * nb
            or eri.size != norb ** 4 or not _fits(na * nb, norb)):
        return orig(kern, eri, civec, norb, nelec, link_index)
    n2 = norb * norb
    ci = np.ascontiguousarray(ci, dtype=np.float64).reshape(na, nb)
    eri = np.ascontiguousarray(eri, dtype=np.float64).reshape(n2, n2)
    linka = _links(norb, neleca)
    linkb = _links(norb, nelecb)
    buf1 = np.empty((na * nb, n2))
    buf2 = np.empty((na * nb, n2))
    fdm2 = np.empty((norb,) * 4)
    fdm3 = np.empty((norb,) * 6)
    path, prefix = worker_blas()
    get_extension().fci_f3(linka, linkb, ci, eri, mode, buf1, buf2, fdm2, fdm3, path, prefix)
    buf1 = buf2 = None
    for i in range(norb):
        for j in range(i):
            fdm3[j, :, i] = fdm3[i, :, j].transpose(1, 0, 2, 3)
            fdm3[j, i, i, :] += fdm2[j, :]
            fdm3[j, :, i, j] -= fdm2[i, :]
    return fdm3


def install():
    """Make ``pyscf.fci.rdm.make_dm123`` use :func:`make_dm123` and ``pyscf.mrpt.nevpt2._contract4pdm``
    :func:`contract4pdm` (idempotent; the originals stay as ``_mojoscf_orig``)."""
    if "make_dm123" in _orig:
        return
    from pyscf.mrpt import nevpt2

    _orig["make_dm123"] = fci_rdm.make_dm123
    _orig["_contract4pdm"] = nevpt2._contract4pdm
    make_dm123._mojoscf_orig = fci_rdm.make_dm123
    contract4pdm._mojoscf_orig = nevpt2._contract4pdm
    fci_rdm.make_dm123 = make_dm123
    nevpt2._contract4pdm = contract4pdm
