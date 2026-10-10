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
from _mojo.linalg import F64Ptr, Blas, list_ptr, vaxpy, vscale, vdot_serial, symmetrize_upper
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


def casscf_mo_core(
    blas_seq: Blas,
    cderi: F64Ptr,
    naux: Int,
    nao: Int,
    mo: F64Ptr,
    nmo: Int,
    ncore: Int,
    dst: F64Ptr,
    diag: F64Ptr,
    vj: F64Ptr,
    vk: F64Ptr,
    kpc: F64Ptr,
) raises:
    """``df_mo_core`` with mo on both sides and, while each L_Q = C^T E_Q C is in cache, the core terms of
    DF-CASSCF: ``diag`` (naux x nmo) the diagonals of L_Q, ``vj`` (nmo x nmo) sum_Q rho_Q L_Q with
    rho_Q = sum_{i < ncore} L_Q[i, i], ``vk`` (nmo x nmo) sum_Q L_Q[c, :]^T L_Q[c, :] (a rank-ncore SYRK per Q)
    and ``kpc`` (ncore x nmo) sum_Q L_Q[c, :]^2 (elementwise).  vj, vk, kpc are overwritten."""
    if naux == 0 or nmo == 0:
        return
    var npair = nao * (nao + 1) // 2
    var nwork = max(1, min(parallelism_level(), naux))
    var n2 = nmo * nmo
    var stride = nao * nao + nmo * nao + 2 * n2 + ncore * nmo
    var buf = List[Float64](length=nwork * stride, fill=0.0)
    var pb = list_ptr(buf)

    def work(c: Int) {imm blas_seq, imm cderi, imm pb, imm mo, imm dst, imm diag, imm nao, imm npair, imm nmo, imm ncore, imm naux, imm nwork, imm stride, imm n2}:
        var e = pb.unsafe_offset(c * stride)
        var u = e.unsafe_offset(nao * nao)
        var jacc = u.unsafe_offset(nmo * nao)
        var kacc = jacc.unsafe_offset(n2)
        var kpacc = kacc.unsafe_offset(n2)
        var q = c
        while q < naux:
            var l = dst.unsafe_offset(q * n2)
            unpack_row(cderi.unsafe_offset(q * npair), nao, e)
            try:
                blas_seq.gemm(True, False, nmo, nao, nao, 1.0, mo, e, 0.0, u)
                blas_seq.gemm(False, False, nmo, nmo, nao, 1.0, u, mo, 0.0, l)
                if ncore > 0:
                    blas_seq.syrk_upper(nmo, ncore, 1.0, l, 1.0, kacc)
            except:
                pass
            var rho = 0.0
            for p in range(nmo):
                var x = l[unsafe_offset=p * nmo + p]
                diag[unsafe_offset=q * nmo + p] = x
                if p < ncore:
                    rho += x
            if ncore > 0:
                vaxpy(jacc, n2, rho, l)
                for i in range(ncore * nmo):
                    var x = l[unsafe_offset=i]
                    kpacc[unsafe_offset=i] += x * x
            q += nwork

    var nthr = blas_seq.serial_begin()
    parallelize(work, nwork)
    blas_seq.serial_end(nthr)
    for i in range(n2):
        vj[unsafe_offset=i] = 0.0
        vk[unsafe_offset=i] = 0.0
    for i in range(ncore * nmo):
        kpc[unsafe_offset=i] = 0.0
    for c in range(nwork):
        var jacc = pb.unsafe_offset(c * stride + nao * nao + nmo * nao)
        vaxpy(vj, n2, 1.0, jacc)
        vaxpy(vk, n2, 1.0, jacc.unsafe_offset(n2))
        vaxpy(kpc, ncore * nmo, 1.0, jacc.unsafe_offset(2 * n2))
    symmetrize_upper(vk, nmo)
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
    a_qs: Int,
    b_qs: Int,
) raises:
    """r (m*nvec x p) += alpha * sum_Q reshape(A_Q X, (m*nvec, k2)) B_Q.

    a : nq x m x k1,  x : k1 x (nvec*k2),  b : nq x k2 x p  (row-major, A_Q at a + Q a_qs and
    B_Q at b + Q b_qs: blocks of rows of larger per-Q matrices need no copy).
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

    def work(c: Int) {imm blas_seq, imm a, imm x, imm b, imm pt, imm pr, imm nq, imm m, imm k1, imm nvec, imm k2, imm p, imm tsize, imm rsize, imm nwork, imm a_qs, imm b_qs}:
        var t = pt.unsafe_offset(c * tsize)
        var acc = pr.unsafe_offset(c * rsize)
        var q = c
        while q < nq:
            try:
                blas_seq.gemm(False, False, m, nvec * k2, k1, 1.0, a.unsafe_offset(q * a_qs), x, 0.0, t)
                blas_seq.gemm(False, False, m * nvec, p, k2, 1.0, t, b.unsafe_offset(q * b_qs), 1.0, acc)
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


def cphf_k_core(
    blas_seq: Blas,
    nq: Int,
    nmo: Int,
    nocc: Int,
    nset: Int,
    lfull: F64Ptr,
    lmo: F64Ptr,
    loo: F64Ptr,
    xs: F64Ptr,
    xts: F64Ptr,
    alpha: Float64,
    r: F64Ptr,
) raises:
    """Exchange of the coupled-perturbed (orbital Hessian) response in the MO basis.

    For first-order orbitals x_n (nmo x nocc, occupied orbitals first) the
    projected exchange of D = C x C_o^T + C_o x^T C^T is

        C^T K[D] C_o = sum_Q [L_Q x (oo|Q) + (po|Q) x^T (po|Q)] = sum_Q L_Q T_Q,
        T_Q = x (oo|Q) + E_o x^T (po|Q)

    (L_Q = (pq|Q) over all orbitals, E_o puts a nocc x nocc block into the
    occupied rows).  The vectors are laid out (p, n, i): per auxiliary
    function T for all of them is one GEMM (nmo nset x nocc), the x^T (po|Q)
    blocks another, and L_Q T one nmo x nmo x (nset nocc) GEMM, half the work
    of the two separate exchange terms and in shapes that keep the BLAS busy.
    r (nmo x nset x nocc) += alpha * sum_Q L_Q T_Q.
    lfull: nq x nmo x nmo, lmo: nq x nmo x nocc, loo: nq x nocc x nocc,
    xs: nmo x nset x nocc, xts: nset x nocc x nmo (the transposes).
    """
    if nq == 0 or nset == 0 or nocc == 0 or nmo == 0:
        return
    var nwork = max(1, min(parallelism_level(), nq))
    var tsize = nmo * nset * nocc
    var psize = nset * nocc * nocc
    var buf = List[Float64](length=nwork * (2 * tsize + psize), fill=0.0)
    var pb = list_ptr(buf)

    def work(c: Int) {imm blas_seq, imm nq, imm nmo, imm nocc, imm nset, imm lfull, imm lmo, imm loo, imm xs, imm xts, imm pb, imm tsize, imm psize, imm nwork}:
        var t = pb.unsafe_offset(c * (2 * tsize + psize))
        var acc = t.unsafe_offset(tsize)
        var p = acc.unsafe_offset(tsize)
        var q = c
        while q < nq:
            try:
                # T (p, n, i) = x (oo|Q) for all vectors, P (n, j, i) = x_n^T (po|Q)
                blas_seq.gemm(False, False, nmo * nset, nocc, nocc, 1.0, xs, loo.unsafe_offset(q * nocc * nocc), 0.0, t)
                blas_seq.gemm(False, False, nset * nocc, nocc, nmo, 1.0, xts, lmo.unsafe_offset(q * nmo * nocc), 0.0, p)
                for j in range(nocc):
                    for n in range(nset):
                        vaxpy(t.unsafe_offset((j * nset + n) * nocc), nocc, 1.0, p.unsafe_offset((n * nocc + j) * nocc))
                blas_seq.gemm(False, False, nmo, nset * nocc, nmo, 1.0, lfull.unsafe_offset(q * nmo * nmo), t, 1.0, acc)
            except:
                pass
            q += nwork

    var nthr = blas_seq.serial_begin()
    parallelize(work, nwork)
    blas_seq.serial_end(nthr)
    for c in range(nwork):
        vaxpy(r, tsize, alpha, pb.unsafe_offset(c * (2 * tsize + psize) + tsize))
    _ = buf^


def casscf_ah_core(
    blas_seq: Blas,
    nq: Int,
    nmo: Int,
    ncore: Int,
    ncas: Int,
    b: F64Ptr,
    rmat: F64Ptr,
    sa: F64Ptr,
    v3: F64Ptr,
    v34: F64Ptr,
) raises:
    """The J/K of a CASSCF orbital-Hessian step (pyscf's ``update_jk_in_ah``) in the MO basis.

    For the core-noncore rotation R (ncore x nx, nx = nmo - ncore) and Sa =
    casdm1 r[active] (ncas x nmo) the densities are X3 = E_c R E_x^T + its
    transpose and X34 = 2 X3 + E_a Sa + Sa^T E_a^T, and the rows needed are

        v3  (ncas x nmo)  += sum_Q 2 rho3_Q B_Q[a, :] - (B_Q X3 B_Q)[a, :]
        v34 (ncore x nx)  += sum_Q 2 rho34_Q B_Q[c, x] - (B_Q X34 B_Q)[c, x]

    with rho_Q = sum_rs B_Q,rs X_rs.  X3 has only the core-noncore blocks, so
    Y = B_Q[occ, :] X3 is Y[:, x] = B_Q[occ, c] R and Y[:, c] = B_Q[occ, x] R^T
    (4 nocc ncore nx flops instead of 2 nocc nmo^2 for a dense X).  With
    T = 2 Y[c] + B_Q[c, a] Sa + B_Q[c, :] Sa^T (into the active columns),
    [T; Y[a]] B_Q[:, x] gives the K of v34 and of the noncore columns of v3,
    and the core columns of v3 come from the symmetry of B X3 B as
    (Y[c] B_Q[:, a])^T.  The J terms use B_Q while it is in cache.  Workers
    take chunks of ``kq`` consecutive Q: the [T; Y[a]] of a chunk are laid side
    by side so that its K is one GEMM with nmo kq terms (B_Q[:, x] of
    consecutive Q are rows of one strided matrix), with a sequential BLAS and
    private accumulators.  b: nq x nmo x nmo (symmetric B_Q), rmat: ncore x nx,
    sa: ncas x nmo, v3: ncas x nmo, v34: ncore x nx.
    """
    var nx = nmo - ncore
    var nocc = ncore + ncas
    if nq == 0 or ncore == 0 or nx == 0:
        return
    comptime kq = 4
    var nchunk = (nq + kq - 1) // kq
    var nwork = max(1, min(parallelism_level(), nchunk))
    var ldy = kq * nmo
    var ysize = nocc * ldy
    var asize = nocc * nx
    var csize = ncore * ncas
    var wsize = ysize + asize + csize
    var buf = List[Float64](length=nwork * wsize + nx * ncore, fill=0.0)
    var pb = list_ptr(buf)
    var rt = pb.unsafe_offset(nwork * wsize)                  # R^T (nx x ncore)
    for i in range(ncore):
        for a in range(nx):
            rt[unsafe_offset=a * ncore + i] = rmat[unsafe_offset=i * nx + a]

    def work(c: Int) {imm blas_seq, imm nq, imm nmo, imm ncore, imm ncas, imm nx, imm nocc, imm b, imm rmat, imm rt, imm sa, imm pb, imm wsize, imm ysize, imm asize, imm ldy, imm nchunk, imm nwork}:
        var y = pb.unsafe_offset(c * wsize)
        var acc = y.unsafe_offset(ysize)                      # nocc x nx: K34 rows, then v3[:, x]
        var acc3c = acc.unsafe_offset(asize)                  # ncore x ncas: v3[:, c]^T
        var ch = c
        while ch < nchunk:
            var q0 = ch * kq
            var nk = min(kq, nq - q0)
            try:
                for j in range(nk):
                    var bq = b.unsafe_offset((q0 + j) * nmo * nmo)
                    var yj = y.unsafe_offset(j * nmo)
                    var rho3 = 0.0
                    for i in range(ncore):
                        rho3 += vdot_serial(bq.unsafe_offset(i * nmo + ncore), rmat.unsafe_offset(i * nx), nx)
                    rho3 *= 2.0
                    var rho34 = 2.0 * rho3
                    if ncas > 0:
                        rho34 += 2.0 * vdot_serial(bq.unsafe_offset(ncore * nmo), sa, ncas * nmo)
                    # Y = B_Q[occ, :] X3
                    blas_seq.gemm_ld(False, False, nocc, nx, ncore, 1.0, bq, nmo, rmat, nx, 0.0,
                                     yj.unsafe_offset(ncore), ldy)
                    blas_seq.gemm_ld(False, False, nocc, ncore, nx, 1.0, bq.unsafe_offset(ncore), nmo, rt, ncore,
                                     0.0, yj, ldy)
                    for i in range(ncore):
                        vaxpy(acc.unsafe_offset(i * nx), nx, 2.0 * rho34, bq.unsafe_offset(i * nmo + ncore))
                    if ncas > 0:
                        # v3: J, and K of the core columns (Y[c] B_Q[:, a])^T
                        for t in range(ncas):
                            vaxpy(acc.unsafe_offset((ncore + t) * nx), nx, 2.0 * rho3,
                                  bq.unsafe_offset((ncore + t) * nmo + ncore))
                        for i in range(ncore):
                            vaxpy(acc3c.unsafe_offset(i * ncas), ncas, 2.0 * rho3, bq.unsafe_offset(i * nmo + ncore))
                        blas_seq.gemm_ld(False, False, ncore, ncas, nmo, -1.0, yj, ldy, bq.unsafe_offset(ncore), nmo,
                                         1.0, acc3c, ncas)
                    # T = 2 Y[c] + B_Q[c, a] Sa + B_Q[c, :] Sa^T E_a^T
                    for i in range(ncore):
                        vscale(yj.unsafe_offset(i * ldy), nmo, 2.0)
                    if ncas > 0:
                        blas_seq.gemm_ld(False, False, ncore, nmo, ncas, 1.0, bq.unsafe_offset(ncore), nmo, sa, nmo,
                                         1.0, yj, ldy)
                        blas_seq.gemm_ld(False, True, ncore, ncas, nmo, 1.0, bq, nmo, sa, nmo, 1.0,
                                         yj.unsafe_offset(ncore), ldy)
                # [T; Y[a]] B_Q[:, x] summed over the chunk
                blas_seq.gemm_ld(False, False, nocc, nx, nk * nmo, -1.0, y, ldy,
                                 b.unsafe_offset(q0 * nmo * nmo + ncore), nmo, 1.0, acc, nx)
            except:
                pass
            ch += nwork

    var nthr = blas_seq.serial_begin()
    parallelize(work, nwork)
    blas_seq.serial_end(nthr)
    for c in range(nwork):
        var acc = pb.unsafe_offset(c * wsize + ysize)
        var acc3c = acc.unsafe_offset(asize)
        vaxpy(v34, ncore * nx, 1.0, acc)
        for t in range(ncas):
            vaxpy(v3.unsafe_offset(t * nmo + ncore), nx, 1.0, acc.unsafe_offset((ncore + t) * nx))
            for i in range(ncore):
                v3[unsafe_offset=t * nmo + i] += acc3c[unsafe_offset=i * ncas + t]
    _ = buf^
