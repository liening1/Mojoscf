"""Density-fitting tensors in a molecular-orbital basis and their contractions (linear response).

``df_mo_core`` transforms pyscf's in-core DF tensor (``(naux, npair)``, packed
lower triangles ``E_Q``) to ``L_Q = C_l^T E_Q C_r`` for every auxiliary
function: worker threads unpack one ``E_Q`` each and apply two sequential
GEMMs, as the exchange part of ``dfjk.mojo`` does.

``df_sandwich_core`` contracts two such tensors with a batch of trial vectors,

    R[(i, n), a] += alpha * sum_Q sum_b (A_Q X)[i, (n, b)] B_Q[b, a],

the exchange part of the TDA/TDDFT response in the occupied-virtual space:
with A_Q = (oo|Q), X[j, (n, b)] = z_n[j, b] and B_Q = (vv|Q) it is
sum_Q (oo|Q) z_n (vv|Q) for every vector n at once, with A_Q = B_Q = (ov|Q)
and X[b, (n, j)] = y_n[j, b] it is sum_Q (ov|Q) y_n^T (ov|Q).  Both products
per Q are plain row-major GEMMs ((A_Q X) is already laid out as the
(i, n) x b matrix the second one needs); workers take every nwork-th Q with
a sequential BLAS and private accumulators that are summed at the end.
"""
from std.runtime import parallelism_level
from max.algorithm import parallelize
from _mojo.linalg import F64Ptr, Blas, list_ptr, vaxpy
from _mojo.dfjk import unpack_row


def df_mo_core(
    blas_seq: Blas, cderi: F64Ptr, naux: Int, nao: Int, cl: F64Ptr, nl: Int, cr: F64Ptr, nr: Int, dst: F64Ptr
) raises:
    """``dst[Q]`` (nl x nr) = C_l^T E_Q C_r for Q < naux; ``cl`` (nao x nl), ``cr`` (nao x nr) row-major."""
    if naux == 0 or nl == 0 or nr == 0:
        return
    var npair = nao * (nao + 1) // 2
    var nwork = max(1, min(parallelism_level(), naux))
    var stride = nao * nao + nl * nao
    var buf = List[Float64](length=nwork * stride, fill=0.0)
    var pb = list_ptr(buf)

    def work(c: Int) {imm blas_seq, imm cderi, imm pb, imm cl, imm cr, imm dst, imm nao, imm npair, imm nl, imm nr, imm naux, imm nwork, imm stride}:
        var e = pb.unsafe_offset(c * stride)
        var u = e.unsafe_offset(nao * nao)
        var q = c
        while q < naux:
            unpack_row(cderi.unsafe_offset(q * npair), nao, e)
            try:
                # U (nl x nao) = C_l^T E_Q, then L_Q = U C_r
                blas_seq.gemm(True, False, nl, nao, nao, 1.0, cl, e, 0.0, u)
                blas_seq.gemm(False, False, nl, nr, nao, 1.0, u, cr, 0.0, dst.unsafe_offset(q * nl * nr))
            except:
                pass
            q += nwork

    var nthr = blas_seq.serial_begin()      # the per-Q GEMMs run concurrently, one BLAS thread each
    parallelize(work, nwork)
    blas_seq.serial_end(nthr)
    _ = buf^


def df_sandwich_core(
    blas_seq: Blas,
    nq: Int,
    a: F64Ptr,
    m: Int,
    k1: Int,
    x: F64Ptr,
    nvec: Int,
    k2: Int,
    b: F64Ptr,
    p: Int,
    alpha: Float64,
    r: F64Ptr,
) raises:
    """r (m*nvec x p) += alpha * sum_Q reshape(A_Q X, (m*nvec, k2)) B_Q.

    a : nq x m x k1,  x : k1 x (nvec*k2),  b : nq x k2 x p  (row-major).
    """
    if nq == 0 or m == 0 or nvec == 0 or p == 0 or k1 == 0 or k2 == 0:
        return
    var nwork = max(1, min(parallelism_level(), nq))
    var tsize = m * nvec * k2
    var rsize = m * nvec * p
    var tbuf = List[Float64](length=nwork * tsize, fill=0.0)
    var rbuf = List[Float64](length=nwork * rsize, fill=0.0)
    var pt = list_ptr(tbuf)
    var pr = list_ptr(rbuf)

    def work(c: Int) {imm blas_seq, imm a, imm x, imm b, imm pt, imm pr, imm nq, imm m, imm k1, imm nvec, imm k2, imm p, imm tsize, imm rsize, imm nwork}:
        var t = pt.unsafe_offset(c * tsize)
        var acc = pr.unsafe_offset(c * rsize)
        var q = c
        while q < nq:
            try:
                blas_seq.gemm(False, False, m, nvec * k2, k1, 1.0, a.unsafe_offset(q * m * k1), x, 0.0, t)
                blas_seq.gemm(False, False, m * nvec, p, k2, 1.0, t, b.unsafe_offset(q * k2 * p), 1.0, acc)
            except:
                pass
            q += nwork

    var nthr = blas_seq.serial_begin()
    parallelize(work, nwork)
    blas_seq.serial_end(nthr)
    for c in range(nwork):
        vaxpy(r, rsize, alpha, pr.unsafe_offset(c * rsize))
    _ = tbuf^
    _ = rbuf^
