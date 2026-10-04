"""Density-fitted Coulomb and exchange matrices from an in-core 3-index tensor.

``cderi`` is pyscf's Cholesky-decomposed DF tensor: ``(naux, npair)`` row-major
with ``npair = nao (nao + 1) / 2`` and the AO pair packed in pyscf's
``lib.pack_tril`` order, ``pair(i, j) = i (i + 1) / 2 + j`` for ``i >= j``.

J is built in two streaming passes (``rho_Q = sum_pair (Q|pair) D_pair`` and
``J_pair = sum_Q (Q|pair) rho_Q``).  K uses weighted "orbitals" ``c_k`` with
signs ``s_k`` such that ``D = sum_k s_k c_k c_k^T``:

    K = sum_Q sum_k s_k (E_Q c_k) (E_Q c_k)^T

For a converged-style density these are the occupied orbitals scaled by
``sqrt(occ)`` (all signs positive); an arbitrary symmetric density is
factorised by diagonalisation (``factorize_density``).  Per block of auxiliary
functions, worker threads unpack one ``E_Q`` each into a thread-local buffer
and immediately form ``U_Q = C^T E_Q`` with a *sequential* BLAS GEMM (the data
stays in cache, as in pyscf's C transform); the block is then closed with one
multi-threaded rank-k update ``K += U^T U`` (``dsyrk``, half the flops of a
GEMM; a GEMM is used when some weights are negative).
"""
from std.memory import Pointer
from std.math import sqrt
from std.runtime import parallelism_level
from max.algorithm import parallelize
from _mojo.linalg import F64Ptr, Blas, list_ptr, vfill, vcopy, vaxpy, vscale, vdot_serial, symmetrize_upper


def pack_tril_dm(dm: F64Ptr, nao: Int, dst: F64Ptr):
    """pyscf's ``dmtril``: dst[pair(i,j)] = dm[i,j] + dm[j,i] (i > j), dm[i,i] on the diagonal."""
    var p = 0
    for i in range(nao):
        for j in range(i):
            dst[unsafe_offset=p] = dm[unsafe_offset=i * nao + j] + dm[unsafe_offset=j * nao + i]
            p += 1
        dst[unsafe_offset=p] = dm[unsafe_offset=i * nao + i]
        p += 1


def unpack_tril_sym(src: F64Ptr, nao: Int, dst: F64Ptr):
    """Symmetric (nao x nao) matrix from a packed lower triangle."""
    var p = 0
    for i in range(nao):
        for j in range(i + 1):
            var v = src[unsafe_offset=p]
            dst[unsafe_offset=i * nao + j] = v
            dst[unsafe_offset=j * nao + i] = v
            p += 1


def unpack_row(src: F64Ptr, nao: Int, dst: F64Ptr):
    """dst (nao x nao, symmetric) from one packed lower triangle ``src``.

    The lower triangle is copied contiguously; the upper one is filled in
    32 x 32 tiles so the strided writes stay cache resident.
    """
    comptime TB = 32
    var p = 0
    for i in range(nao):
        vcopy(dst.unsafe_offset(i * nao), src.unsafe_offset(p), i + 1)
        p += i + 1
    var i0 = 0
    while i0 < nao:
        var i1 = min(i0 + TB, nao)
        var j0 = 0
        while j0 <= i0:
            var j1 = min(j0 + TB, nao)
            for i in range(i0, i1):
                var jmax = min(j1, i)
                for j in range(j0, jmax):
                    dst[unsafe_offset=j * nao + i] = dst[unsafe_offset=i * nao + j]
            j0 = j1
        i0 = i1


def block_size(naux: Int, nao: Int, nk: Int, mmax: Int, budget_bytes: Int) -> Int:
    """Auxiliary functions per block so the work buffers stay within ``budget_bytes``."""
    var per_q = 8 * (2 * max(nk, 1) * nao * max(mmax, 1))
    var fixed = 8 * nao * nao * max(1, 2 * parallelism_level())
    var b = (budget_bytes - fixed) // per_q
    if b < 1:
        b = 1
    if b > naux:
        b = naux
    return b


def factorize_density(
    blas: Blas, nao: Int, dm: F64Ptr, orb: F64Ptr, sign: F64Ptr, rel_tol: Float64
) raises -> Int:
    """Weighted orbitals of a symmetric density: D = sum_k sign_k c_k c_k^T.

    Writes ``m`` columns ``c_k = v_k sqrt(|w_k|)`` into ``orb`` (nao x m, row-major
    with row stride ``nao``; columns beyond ``m`` are unused) and the signs of
    the eigenvalues into ``sign``.  Eigenvalues below ``rel_tol * max|w|`` are
    dropped.  Returns ``m``.
    """
    var w = List[Float64](length=nao, fill=0.0)
    var v = List[Float64](length=nao * nao, fill=0.0)
    var pw = list_ptr(w)
    var pv = list_ptr(v)
    blas.eigh(nao, dm, pw, pv)
    var wmax = 0.0
    for k in range(nao):
        if abs(w[k]) > wmax:
            wmax = abs(w[k])
    var thresh = rel_tol * wmax
    var m = 0
    for k in range(nao):
        if abs(w[k]) > thresh:
            var scale = sqrt(abs(w[k]))
            for i in range(nao):
                orb[unsafe_offset=i * nao + m] = pv[unsafe_offset=i * nao + k] * scale
            sign[unsafe_offset=m] = 1.0 if w[k] > 0.0 else -1.0
            m += 1
    _ = w^
    _ = v^
    return m


def orbitals_from_mo(
    nao: Int, nmo: Int, mo_coeff: F64Ptr, mo_occ: F64Ptr, orb: F64Ptr, sign: F64Ptr
) -> Int:
    """Occupied orbitals scaled by sqrt(occ) (nao x m, row stride nao); returns m."""
    var m = 0
    for k in range(nmo):
        var o = mo_occ[unsafe_offset=k]
        if o > 0.0:
            var scale = sqrt(o)
            for i in range(nao):
                orb[unsafe_offset=i * nao + m] = mo_coeff[unsafe_offset=i * nmo + k] * scale
            sign[unsafe_offset=m] = 1.0
            m += 1
    return m


def df_jk_core(
    blas: Blas,
    blas_seq: Blas,
    cderi: F64Ptr,
    naux: Int,
    nao: Int,
    blk: Int,
    nj: Int,
    dms_j: F64Ptr,
    vj: F64Ptr,
    nk: Int,
    orbs: F64Ptr,
    orb_stride: Int,
    ms: Pointer[Int64, MutAnyOrigin],
    signs: F64Ptr,
    vk: F64Ptr,
) raises:
    """J for ``nj`` densities and K for ``nk`` orbital sets from the DF tensor.

    blas      : library for the large multi-threaded updates
    blas_seq  : library that is safe to call from several threads at once
                (a sequential build); used for the per-Q transforms
    dms_j : nj x nao x nao symmetric densities  ->  vj : nj x nao x nao (overwritten)
    orbs  : nk orbital sets, set s at ``orbs + s * orb_stride`` as (nao x m_s) with
            row stride ``nao``; ``signs`` holds the m_s signs of set s at
            ``signs + s * nao``  ->  vk : nk x nao x nao (overwritten)
    """
    var npair = nao * (nao + 1) // 2
    var n2 = nao * nao
    var nthreads = max(1, parallelism_level())

    # ---------------------------------------------------------------- Coulomb
    # Two memory-streaming passes over the tensor: rho_Q = E_Q . Dt, then
    # J_pair = sum_Q rho_Q E_Q, each parallel over aux functions / pair ranges.
    if nj > 0:
        var dt = List[Float64](length=npair * nj, fill=0.0)
        var rho = List[Float64](length=naux * nj, fill=0.0)
        var vjp = List[Float64](length=npair * nj, fill=0.0)
        var pdt = list_ptr(dt)
        var prho = list_ptr(rho)
        var pvjp = list_ptr(vjp)
        for s in range(nj):
            pack_tril_dm(dms_j.unsafe_offset(s * n2), nao, pdt.unsafe_offset(s * npair))

        def rho_work(q: Int) {imm cderi, imm pdt, imm prho, imm npair, imm nj}:
            var row = cderi.unsafe_offset(q * npair)
            for s in range(nj):
                prho[unsafe_offset=q * nj + s] = vdot_serial(row, pdt.unsafe_offset(s * npair), npair)

        parallelize(rho_work, naux)

        var nchunks = 4 * nthreads
        if nchunks > npair:
            nchunks = npair
        var chunk = (npair + nchunks - 1) // nchunks

        def j_work(c: Int) {imm cderi, imm prho, imm pvjp, imm npair, imm naux, imm nj, imm chunk}:
            var p0 = c * chunk
            var p1 = min(p0 + chunk, npair)
            if p1 <= p0:
                return
            for q in range(naux):
                var row = cderi.unsafe_offset(q * npair + p0)
                for s in range(nj):
                    vaxpy(pvjp.unsafe_offset(s * npair + p0), p1 - p0, prho[unsafe_offset=q * nj + s], row)

        parallelize(j_work, nchunks)
        for s in range(nj):
            unpack_tril_sym(pvjp.unsafe_offset(s * npair), nao, vj.unsafe_offset(s * n2))
        _ = dt^
        _ = rho^
        _ = vjp^

    # --------------------------------------------------------------- exchange
    if nk == 0:
        return
    var mmax = 0
    var neg = List[Bool](length=nk, fill=False)
    for s in range(nk):
        var m = Int(ms[unsafe_offset=s])
        if m > mmax:
            mmax = m
        for k in range(m):
            if signs[unsafe_offset=s * nao + k] < 0.0:
                neg[s] = True
    vfill(vk, nk * n2, 0.0)
    if mmax == 0:
        return

    # Compact copies of the orbital sets: (nao x m_s) with row stride m_s.
    var corb = List[Float64](length=nk * nao * mmax, fill=0.0)
    var pcorb = list_ptr(corb)
    for s in range(nk):
        var m = Int(ms[unsafe_offset=s])
        var src = orbs.unsafe_offset(s * orb_stride)
        var dst = pcorb.unsafe_offset(s * nao * mmax)
        for i in range(nao):
            for k in range(m):
                dst[unsafe_offset=i * m + k] = src[unsafe_offset=i * nao + k]

    var nwork = 2 * nthreads
    var ebuf = List[Float64](length=nwork * n2, fill=0.0)              # one unpacked E_Q per worker
    var ubuf = List[Float64](length=nk * blk * mmax * nao, fill=0.0)   # U rows (Q, k) per set
    var pe = list_ptr(ebuf)
    var pu = list_ptr(ubuf)
    var pms = ms
    var ustride = blk * mmax * nao

    var q0 = 0
    while q0 < naux:
        var q1 = min(q0 + blk, naux)
        var nb = q1 - q0

        def work(c: Int) {imm blas_seq, imm cderi, imm pe, imm pu, imm pcorb, imm pms, imm nao, imm npair, imm n2, imm nk, imm mmax, imm ustride, imm q0, imm q1, imm nwork}:
            var e = pe.unsafe_offset(c * n2)
            var q = q0 + c
            while q < q1:
                unpack_row(cderi.unsafe_offset(q * npair), nao, e)
                for s in range(nk):
                    var m = Int(pms[unsafe_offset=s])
                    if m > 0:
                        # U_q (m x nao) = C_s^T (m x nao) . E_q (nao x nao)
                        try:
                            blas_seq.gemm(
                                True, False, m, nao, nao, 1.0, pcorb.unsafe_offset(s * nao * mmax), e, 0.0,
                                pu.unsafe_offset(s * ustride + (q - q0) * m * nao),
                            )
                        except:
                            pass
                q += nwork

        parallelize(work, nwork)

        for s in range(nk):
            var m = Int(ms[unsafe_offset=s])
            if m == 0:
                continue
            var u = pu.unsafe_offset(s * ustride)
            var kout = vk.unsafe_offset(s * n2)
            if neg[s]:
                # K += A^T U with the rows of A = U scaled by the signs
                var a = List[Float64](length=nb * m * nao, fill=0.0)
                var pa = list_ptr(a)
                vcopy(pa, u, nb * m * nao)
                var sg = signs.unsafe_offset(s * nao)
                for q in range(nb):
                    for k in range(m):
                        if sg[unsafe_offset=k] < 0.0:
                            vscale(pa.unsafe_offset((q * m + k) * nao), nao, -1.0)
                blas.gemm(True, False, nao, nao, nb * m, 1.0, pa, u, 1.0, kout)
                _ = a^
            else:
                blas.syrk_upper(nao, nb * m, 1.0, u, 1.0, kout)
        q0 = q1
    for s in range(nk):
        if not neg[s]:
            symmetrize_upper(vk.unsafe_offset(s * n2), nao)
    _ = neg^
    _ = corb^
    _ = ebuf^
    _ = ubuf^
