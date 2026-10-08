"""NumPy-facing wrappers around the Mojo kernels.

These functions accept array-likes, enforce C-contiguous float64 storage and
allocate outputs.  They are drop-in equivalents of the corresponding pyscf
helpers (``pyscf.scf.hf.make_rdm1``, ``get_occ``, ``get_grad``, ...) and are
what ``mojoscf.scf`` uses to accelerate an SCF object.
"""
from __future__ import annotations

import numpy as np

from ._backend import blas_args, df_block_mb, get_extension, worker_blas

__all__ = [
    "gemm",
    "eigh",
    "make_rdm1",
    "trace_prod",
    "energy_elec",
    "get_occ",
    "get_grad",
    "damping",
    "level_shift",
    "diis_errvec",
    "norm_diff",
    "jk_dense",
    "df_jk",
    "jk_s8",
    "factorize_density",
    "df_mo",
    "df_sandwich",
    "cphf_k",
]


def _c(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64)


def _square(a, name: str) -> np.ndarray:
    a = _c(a)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape {a.shape}")
    return a


def gemm(a, b, transa: bool = False, transb: bool = False, alpha: float = 1.0, beta: float = 0.0, out=None):
    """Row-major ``C = alpha * op(A) @ op(B) + beta * C``."""
    a = _c(a)
    b = _c(b)
    m = a.shape[1] if transa else a.shape[0]
    k = a.shape[0] if transa else a.shape[1]
    n = b.shape[0] if transb else b.shape[1]
    if out is None:
        out = np.zeros((m, n))
        beta = 0.0
    else:
        out = _c(out)
        if out.shape != (m, n):
            raise ValueError(f"out has shape {out.shape}, expected {(m, n)}")
    path, prefix = blas_args(max(m, n, k))
    get_extension().gemm(a, b, out, bool(transa), bool(transb), float(alpha), float(beta), path, prefix)
    return out


def eigh(h, s=None, x=None):
    """Symmetric eigensolver with pyscf's conventions.

    * ``eigh(h)``        : ``h c = c diag(w)``
    * ``eigh(h, s)``     : ``h c = s c diag(w)``
    * ``eigh(h, x=x)``   : ``c = x c'`` where ``x^T h x c' = c' diag(w)`` (what
      pyscf does once it has computed the orthogonaliser ``x`` of ``s``).

    Eigenvalues ascending; the largest component of each eigenvector is positive.
    """
    h = _square(h, "h")
    nao = h.shape[0]
    path, prefix = blas_args(nao)
    ext = get_extension()
    if x is not None:
        x = _c(x)
        if x.shape[0] != nao:
            raise ValueError("x must have shape (nao, nmo)")
        nmo = x.shape[1]
        w = np.empty(nmo)
        c = np.empty((nao, nmo))
        ext.eigh(h, None, x, w, c, path, prefix)
        return w, c
    w = np.empty(nao)
    c = np.empty((nao, nao))
    if s is None:
        ext.eigh(h, None, None, w, c, path, prefix)
    else:
        s = _square(s, "s")
        if s.shape != h.shape:
            raise ValueError("h and s must have the same shape")
        ext.eigh(h, s, None, w, c, path, prefix)
    return w, c


def make_rdm1(mo_coeff, mo_occ):
    """One-particle density matrix ``C_occ diag(occ) C_occ^T``."""
    mo_coeff = _c(mo_coeff)
    mo_occ = _c(mo_occ).ravel()
    if mo_coeff.ndim != 2 or mo_coeff.shape[1] != mo_occ.shape[0]:
        raise ValueError("mo_coeff must be (nao, nmo) and mo_occ (nmo,)")
    dm = np.empty((mo_coeff.shape[0], mo_coeff.shape[0]))
    path, prefix = blas_args(mo_coeff.shape[0])
    get_extension().make_rdm1(mo_coeff, mo_occ, dm, path, prefix)
    return dm


def trace_prod(a, b) -> float:
    """``sum_ij a[i, j] * b[j, i]`` i.e. ``numpy.einsum('ij,ji->', a, b)``."""
    a = _square(a, "a")
    b = _square(b, "b")
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape")
    return float(get_extension().trace_prod(a, b))


def energy_elec(h1e, vhf, dm):
    """``(e1 + e2, e2)`` with ``e1 = tr(h1e dm)`` and ``e2 = tr(vhf dm) / 2``."""
    h1e = _square(h1e, "h1e")
    vhf = _square(vhf, "vhf")
    dm = _square(dm, "dm")
    if not (h1e.shape == vhf.shape == dm.shape):
        raise ValueError("h1e, vhf and dm must have the same shape")
    e1, e2 = get_extension().energy_elec(h1e, vhf, dm)
    return float(e1) + float(e2), float(e2)


def get_occ(mo_energy, nocc: int, occ: float = 2.0):
    """Aufbau occupations; returns ``(mo_occ, (homo, lumo) or None)``.

    ``occ`` is the occupation of the filled orbitals: 2 for RHF (default), 1 for
    one spin channel of UHF.
    """
    mo_energy = _c(mo_energy).ravel()
    mo_occ = np.zeros_like(mo_energy)
    gap = get_extension().get_occ(mo_energy, int(nocc), mo_occ, float(occ))
    return mo_occ, gap


def get_grad(mo_coeff, mo_occ, fock, prefactor: float = 2.0):
    """Orbital gradient ``prefactor * C_vir^T F C_occ`` flattened to ``nvir * nocc``.

    ``prefactor`` is 2 for RHF (default) and 1 for one spin channel of UHF.
    """
    mo_coeff = _c(mo_coeff)
    mo_occ = _c(mo_occ).ravel()
    fock = _square(fock, "fock")
    nocc = int(np.count_nonzero(mo_occ > 0))
    nvir = mo_occ.shape[0] - nocc
    g = np.empty(max(nvir * nocc, 1))
    path, prefix = blas_args(mo_coeff.shape[0])
    n = get_extension().get_grad(mo_coeff, mo_occ, fock, g, float(prefactor), path, prefix)
    return g[:n]


def damping(f, f_prev, factor: float):
    f = _c(f)
    f_prev = _c(f_prev)
    out = np.empty_like(f)
    get_extension().damping(f, f_prev, float(factor), out)
    return out


def level_shift(s, dm, f, factor: float, dm_scale: float = 0.5):
    """``f + factor * (s - s (dm_scale * dm) s)``.

    The default ``dm_scale = 0.5`` is pyscf's RHF call ``level_shift(s, dm*.5, f,
    factor)``; use ``dm_scale = 1`` for a UHF spin density.
    """
    s = _square(s, "s")
    dm = _square(dm, "dm")
    f = _square(f, "f")
    out = np.empty_like(f)
    path, prefix = blas_args(f.shape[0])
    get_extension().level_shift(s, dm, f, float(factor), out, float(dm_scale), path, prefix)
    return out


def diis_errvec(s, dm, f, x=None):
    """CDIIS error vector ``(SDF)^T - SDF`` (projected with ``x`` when given)."""
    s = _square(s, "s")
    dm = _square(dm, "dm")
    f = _square(f, "f")
    nao = s.shape[0]
    path, prefix = blas_args(nao)
    if x is None:
        err = np.empty(nao * nao)
        get_extension().diis_errvec(s, dm, f, None, err, path, prefix)
        return err
    x = _c(x)
    nmo = x.shape[1]
    err = np.empty(nmo * nmo)
    get_extension().diis_errvec(s, dm, f, x, err, path, prefix)
    return err


def norm_diff(a, b) -> float:
    """Frobenius norm of ``a - b``."""
    a = _c(a)
    b = _c(b)
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape")
    return float(get_extension().norm_diff(a, b))


def jk_dense(eri, dm):
    """Coulomb and exchange matrices from a full ``(n, n, n, n)`` ERI tensor."""
    dm = _square(dm, "dm")
    n = dm.shape[0]
    eri = _c(eri)
    if eri.size != n**4:
        raise ValueError("eri must hold nao**4 elements")
    vj = np.empty((n, n))
    vk = np.empty((n, n))
    get_extension().jk_dense(eri, dm, vj, vk)
    return vj, vk


def factorize_density(dm, rel_tol: float = 1e-14):
    """Weighted orbitals of a symmetric density: ``dm = (orb * sign) @ orb.T``.

    Returns ``(orb, sign)`` with ``orb`` of shape ``(nao, m)``; eigenvalues below
    ``rel_tol`` times the largest one are dropped.  Used to build the exchange
    matrix of a density that does not come with occupied orbitals.
    """
    dm = _square(dm, "dm")
    nao = dm.shape[0]
    orb = np.zeros((nao, nao))
    sign = np.zeros(nao)
    path, prefix = blas_args(nao)
    m = int(get_extension().factorize_density(dm, orb, sign, float(rel_tol), path, prefix))
    return orb[:, :m].copy(), sign[:m].copy()


def df_jk(cderi, dm, mo_coeff=None, mo_occ=None, with_j=True, with_k=True, block_mb=None, fact_tol=1e-14):
    """Density-fitted ``(vj, vk)`` with the semantics of ``pyscf.df.df_jk.get_jk``.

    ``cderi`` is pyscf's in-core ``(naux, nao*(nao+1)//2)`` tensor; ``dm`` is one
    symmetric density or a stack of them.  When ``mo_coeff``/``mo_occ`` are
    given (stacked like ``dm``) the exchange part uses the occupied orbitals;
    otherwise each density is factorised.  Returns arrays shaped like ``dm``
    (``None`` for a part that was not requested).
    """
    cderi = _c(cderi)
    dm_in = np.asarray(dm, dtype=np.float64)
    dms = np.ascontiguousarray(dm_in.reshape(-1, dm_in.shape[-1], dm_in.shape[-1]))
    nset, nao, _ = dms.shape
    if cderi.ndim != 2 or cderi.shape[1] != nao * (nao + 1) // 2:
        raise ValueError("cderi must have shape (naux, nao*(nao+1)//2)")
    vj = np.zeros((nset, nao, nao))
    vk = np.zeros((nset, nao, nao))
    orbs = ms = signs = None
    if with_k and mo_coeff is not None:
        if mo_occ is None:
            raise ValueError("mo_occ is required with mo_coeff")
        c_all = np.asarray(mo_coeff, dtype=np.float64)
        o_all = np.asarray(mo_occ, dtype=np.float64)
        c_all = c_all.reshape(-1, c_all.shape[-2], c_all.shape[-1])
        o_all = o_all.reshape(-1, o_all.shape[-1])
        if c_all.shape[0] != nset or o_all.shape[0] != nset:
            raise ValueError("mo_coeff/mo_occ must be stacked like dm")
        orbs = np.zeros((nset, nao, nao))
        signs = np.zeros((nset, nao))
        ms = np.zeros(nset, dtype=np.int64)
        for s in range(nset):
            occ = o_all[s] > 0
            m = int(occ.sum())
            orbs[s, :, :m] = c_all[s][:, occ] * np.sqrt(o_all[s][occ])
            signs[s, :m] = 1.0
            ms[s] = m
    if block_mb is None:
        block_mb = df_block_mb()
    path, prefix = blas_args(nao)
    seq_path, seq_prefix = worker_blas()
    get_extension().df_jk(
        cderi, dms, orbs, ms, signs, vj, vk, bool(with_j), bool(with_k), int(block_mb), float(fact_tol),
        path, prefix, seq_path, seq_prefix,
    )
    vj = vj.reshape(dm_in.shape) if with_j else None
    vk = vk.reshape(dm_in.shape) if with_k else None
    return vj, vk


def df_mo(cderi, cl, cr):
    """The DF tensor in an orbital basis: ``out[Q] = cl^T E_Q cr``, shape ``(naux, nl, nr)``.

    ``cderi`` is pyscf's in-core ``(naux, nao*(nao+1)//2)`` tensor (packed
    symmetric ``E_Q``), ``cl`` and ``cr`` are ``(nao, nl)`` and ``(nao, nr)``
    coefficient matrices (``(ia|Q)`` with the occupied and virtual orbitals).
    """
    cderi = _c(cderi)
    cl = _c(cl)
    cr = _c(cr)
    nao = cl.shape[0]
    if cderi.ndim != 2 or cderi.shape[1] != nao * (nao + 1) // 2 or cr.shape[0] != nao:
        raise ValueError("cderi must have shape (naux, nao*(nao+1)//2) and cl, cr nao rows")
    out = np.empty((cderi.shape[0], cl.shape[1], cr.shape[1]))
    seq_path, seq_prefix = worker_blas()
    get_extension().df_mo(cderi, cl, cr, out, seq_path, seq_prefix)
    return out


def df_sandwich(a, x, b, nvec, alpha=1.0, out=None):
    """``out (m*nvec, p) += alpha * sum_Q reshape(a[Q] @ x, (m*nvec, k2)) @ b[Q]``.

    ``a`` is ``(nq, m, k1)``, ``x`` is ``(k1, nvec*k2)`` and ``b`` is
    ``(nq, k2, p)``: the exchange contractions of the linear-response
    operators (:mod:`mojoscf.tdscf`).  A new zero ``out`` is used when none is
    given; returns ``out``.
    """
    a = _c(a)
    x = _c(x)
    b = _c(b)
    nq, m, k1 = a.shape
    k2, p = b.shape[1], b.shape[2]
    if b.shape[0] != nq or x.shape != (k1, nvec * k2):
        raise ValueError(f"inconsistent shapes a {a.shape}, x {x.shape}, b {b.shape}, nvec {nvec}")
    if out is None:
        out = np.zeros((m * nvec, p))
    elif out.shape != (m * nvec, p) or out.dtype != np.float64 or not out.flags.c_contiguous:
        raise ValueError(f"out must be a C-contiguous float64 array of shape {(m * nvec, p)}")
    seq_path, seq_prefix = worker_blas()
    get_extension().df_sandwich(a, x, b, int(nvec), float(alpha), out, seq_path, seq_prefix)
    return out


def cphf_k(lfull, lmo, loo, x, alpha=1.0, out=None):
    """``out (nset, nmo, nocc) += alpha * sum_Q L_Q (x_n (oo|Q) + E_o x_n^T (po|Q))``.

    The projected exchange ``C^T K[C x C_o^T + C_o x^T C^T] C_o`` of first-order
    orbitals ``x`` (nset, nmo, nocc) with the occupied orbitals first:
    ``lfull`` (naux, nmo, nmo) is the DF tensor in the MO basis, ``lmo`` and
    ``loo`` its (naux, nmo, nocc) and (naux, nocc, nocc) blocks.
    """
    lfull, lmo, loo, x = _c(lfull), _c(lmo), _c(loo), _c(x)
    nq, nmo, _ = lfull.shape
    nset, _, nocc = x.shape
    if lmo.shape != (nq, nmo, nocc) or loo.shape != (nq, nocc, nocc) or x.shape[1] != nmo:
        raise ValueError(f"inconsistent shapes lfull {lfull.shape}, lmo {lmo.shape}, loo {loo.shape}, x {x.shape}")
    if out is None:
        out = np.zeros((nset, nmo, nocc))
    elif out.shape != (nset, nmo, nocc) or out.dtype != np.float64 or not out.flags.c_contiguous:
        raise ValueError(f"out must be a C-contiguous float64 array of shape {(nset, nmo, nocc)}")
    xts = np.ascontiguousarray(x.transpose(0, 2, 1))
    seq_path, seq_prefix = worker_blas()
    get_extension().cphf_k(lfull, lmo, loo, x, xts, float(alpha), out, seq_path, seq_prefix)
    return out


def jk_s8(eri, dm, with_j=True, with_k=True):
    """``(vj, vk)`` from 8-fold packed ERIs (``mol.intor('int2e', aosym='s8')``).

    Same semantics as ``pyscf.scf.hf.dot_eri_dm(eri, dm, hermi=1)``: ``dm`` must be
    symmetric (one matrix or a stack); the outputs are shaped like ``dm``.
    """
    eri = _c(eri)
    dm_in = np.asarray(dm, dtype=np.float64)
    dms = np.ascontiguousarray(dm_in.reshape(-1, dm_in.shape[-1], dm_in.shape[-1]))
    nset, nao, _ = dms.shape
    npair = nao * (nao + 1) // 2
    if eri.ndim != 1 or eri.size != npair * (npair + 1) // 2:
        raise ValueError("eri must be the 8-fold packed int2e vector for this nao")
    vj = np.zeros((nset, nao, nao))
    vk = np.zeros((nset, nao, nao))
    get_extension().jk_s8(eri, dms, vj, vk, bool(with_j), bool(with_k))
    vj = vj.reshape(dm_in.shape) if with_j else None
    vk = vk.reshape(dm_in.shape) if with_k else None
    return vj, vk
