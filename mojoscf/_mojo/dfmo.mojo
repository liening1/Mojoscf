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
from std.memory import Pointer
from _mojo.linalg import F64Ptr, Blas, list_ptr, vaxpy, vfill, vscale, vdot_serial, symmetrize_upper
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
                blas_seq.gemm_small(True, False, nl, nao, nao, 1.0, cl, e, 0.0, u)
                blas_seq.gemm_small(False, False, nl, nr, nao, 1.0, u, cr, 0.0, dst.unsafe_offset(q * nl * nr))
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
                blas_seq.gemm_small(True, False, nmo, nao, nao, 1.0, mo, e, 0.0, u)
                blas_seq.gemm_small(False, False, nmo, nmo, nao, 1.0, u, mo, 0.0, l)
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
                blas_seq.gemm_small(False, False, m, nvec * k2, k1, 1.0, a.unsafe_offset(q * a_qs), x, 0.0, t)
                blas_seq.gemm_small(False, False, m * nvec, p, k2, 1.0, t, b.unsafe_offset(q * b_qs), 1.0, acc)
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
                blas_seq.gemm_small(False, False, nmo * nset, nocc, nocc, 1.0, xs, loo.unsafe_offset(q * nocc * nocc), 0.0, t)
                blas_seq.gemm_small(False, False, nset * nocc, nocc, nmo, 1.0, xts, lmo.unsafe_offset(q * nmo * nocc), 0.0, p)
                for j in range(nocc):
                    for n in range(nset):
                        vaxpy(t.unsafe_offset((j * nset + n) * nocc), nocc, 1.0, p.unsafe_offset((n * nocc + j) * nocc))
                blas_seq.gemm_small(False, False, nmo, nset * nocc, nmo, 1.0, lfull.unsafe_offset(q * nmo * nmo), t, 1.0, acc)
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


comptime I64Ptr = Pointer[Int64, MutAnyOrigin]


def packed_gather_core(nq: Int, npk: Int, src: F64Ptr, nidx: Int, idx: I64Ptr, dst: F64Ptr):
    """dst[P, k] = src[P, idx[k]] for P < nq (src rows npk apart, dst rows nidx apart): blocks of the
    packed per-auxiliary-function matrices of the DF gradient terms, rows in parallel."""
    if nq == 0 or nidx == 0:
        return

    def row(q: Int) {imm src, imm dst, imm idx, imm npk, imm nidx}:
        var s = src.unsafe_offset(q * npk)
        var d = dst.unsafe_offset(q * nidx)
        for k in range(nidx):
            d[unsafe_offset=k] = s[unsafe_offset=Int(idx[unsafe_offset=k])]

    parallelize(row, nq)


def packed_scatter_core(nq: Int, npk: Int, dst: F64Ptr, nidx: Int, idx: I64Ptr, wts: F64Ptr, src: F64Ptr):
    """dst[P, idx[k]] += wts[k] src[P, k] for P < nq (repeated indices accumulate), rows in parallel."""
    if nq == 0 or nidx == 0:
        return

    def row(q: Int) {imm src, imm dst, imm idx, imm wts, imm npk, imm nidx}:
        var s = src.unsafe_offset(q * nidx)
        var d = dst.unsafe_offset(q * npk)
        for k in range(nidx):
            d[unsafe_offset=Int(idx[unsafe_offset=k])] += wts[unsafe_offset=k] * s[unsafe_offset=k]

    parallelize(row, nq)


def hess_dfk_rk_core(
    blas_seq: Blas, nao: Int, naux: Int, nocc: Int, natm: Int, i0: Int, nb: Int, co: F64Ptr, g: F64Ptr,
    ck: F64Ptr, v1: F64Ptr, v1ck: F64Ptr, aoloc: I64Ptr, auxloc: I64Ptr, rk: F64Ptr,
) raises:
    """The exchange response vectors of the DF Hessian (``mojoscf.hessian._df_jk_pass``) for the occupied
    rows i0 <= i < i0 + nb:

        rk[P, A, x, i, j] = -(Z_AxP[i, j] + Z_AxP[j, i]) + sum_{Q on A} V1_x[Q, P] ck_Q[i, j]
                            + [P on A] (Z_xP[i, j] + Z_xP[j, i] + (V1_x ck)_P[i, j])

    with Z_AxP = C_o[A]^T g_xP[A] (the rows of the functions of atom A), Z_xP = sum_A Z_AxP,
    g (3, nao, naux, nocc) = (nabla_x mu nu|P) C_o, V1 (3, naux, naux) = (nabla_x P|Q), ck (naux, nocc, nocc),
    v1ck (3, naux, nocc, nocc) = V1 ck; atom A owns the functions aoloc[A] <= mu < aoloc[A + 1] and the
    auxiliary functions auxloc[A] <= P < auxloc[A + 1].  rk (naux, natm, 3, nb, nocc) is overwritten.
    Tiles of auxiliary functions run in parallel with one BLAS thread each: the V1 term is one GEMM per
    (tile, A, x), the Z terms two small GEMMs per (P, A, x).
    """
    comptime PT = 32
    if naux == 0 or nb == 0 or nocc == 0:
        return
    var nbo = nb * nocc
    var ldr = natm * 3 * nbo
    var ntile = (naux + PT - 1) // PT
    var nwork = max(1, min(parallelism_level(), ntile))
    var pat = List[Int](length=naux, fill=0)
    for a in range(natm):
        for p in range(Int(auxloc[unsafe_offset=a]), Int(auxloc[unsafe_offset=a + 1])):
            pat[p] = a
    var buf = List[Float64](length=nwork * 2 * nbo + 1, fill=0.0)
    var pb = list_ptr(buf)
    var ppat = Pointer(to=pat)

    def work(w: Int) {imm blas_seq, imm nao, imm naux, imm nocc, imm natm, imm i0, imm nb, imm co, imm g, imm ck, imm v1, imm v1ck, imm aoloc, imm auxloc, imm rk, imm nbo, imm ldr, imm ntile, imm nwork, imm pb, imm ppat}:
        var za = pb.unsafe_offset(w * 2 * nbo)
        var zs = za.unsafe_offset(nbo)
        var t = w
        while t < ntile:
            var p0 = t * PT
            var np_ = min(PT, naux - p0)
            try:
                for a in range(natm):
                    var q0 = Int(auxloc[unsafe_offset=a])
                    var q1 = Int(auxloc[unsafe_offset=a + 1])
                    for x in range(3):
                        # sum_{Q on A} V1_x[Q, P] ck_Q[i, j] for the tile (initialises its rows)
                        blas_seq.gemm_ld(
                            True, False, np_, nbo, q1 - q0, 1.0, v1.unsafe_offset(x * naux * naux + q0 * naux + p0),
                            naux, ck.unsafe_offset(q0 * nocc * nocc + i0 * nocc), nocc * nocc, 0.0,
                            rk.unsafe_offset(p0 * ldr + (a * 3 + x) * nbo), ldr,
                        )
                for p in range(p0, p0 + np_):
                    var ap = ppat[][p]
                    for x in range(3):
                        vfill(zs, nbo, 0.0)
                        var gp = g.unsafe_offset((x * nao * naux + p) * nocc)        # row mu at mu naux nocc
                        for a in range(natm):
                            var m0 = Int(aoloc[unsafe_offset=a])
                            var m1 = Int(aoloc[unsafe_offset=a + 1])
                            if m1 == m0:
                                continue
                            var ga = gp.unsafe_offset(m0 * naux * nocc)
                            var ca = co.unsafe_offset(m0 * nocc)
                            # Z_A[i, j] = sum_{mu on A} C_o[mu, i] g[mu, j], then + Z_A[j, i] = sum_mu g[mu, i] C_o[mu, j]
                            blas_seq.gemm_ld(True, False, nb, nocc, m1 - m0, 1.0, ca.unsafe_offset(i0), nocc, ga,
                                             naux * nocc, 0.0, za, nocc)
                            blas_seq.gemm_ld(True, False, nb, nocc, m1 - m0, 1.0, ga.unsafe_offset(i0), naux * nocc,
                                             ca, nocc, 1.0, za, nocc)
                            vaxpy(rk.unsafe_offset(p * ldr + (a * 3 + x) * nbo), nbo, -1.0, za)
                            vaxpy(zs, nbo, 1.0, za)
                        var dst = rk.unsafe_offset(p * ldr + (ap * 3 + x) * nbo)
                        vaxpy(dst, nbo, 1.0, zs)
                        vaxpy(dst, nbo, 1.0, v1ck.unsafe_offset(((x * naux + p) * nocc + i0) * nocc))
            except:
                pass
            t += nwork

    var nthr = blas_seq.serial_begin()
    parallelize(work, nwork)
    blas_seq.serial_end(nthr)
    _ = buf^
    _ = pat^


def hess_dfk_contract_core(
    blas_seq: Blas, nw: Int, nat3: Int, nb: Int, nocc: Int, nmo: Int, i0: Int, rt: F64Ptr, lt: F64Ptr,
    want_h: Bool, gram: F64Ptr, hr: F64Ptr,
) raises:
    """gram (nat3 x nat3) += sum_P rt_P rt_P^T and, with ``want_h``, hr[a] (nmo x nocc) += sum_P
    lt[:, P, i0:i0 + nb] rt[P, a] for the whitened exchange response vectors rt (nw, nat3, nb, nocc) of
    :func:`hess_dfk_rk_core` and lt (nmo, nw, nocc) (the whitened C^T B C_o); hr is (nat3, nmo, nocc).
    The Gram matrix over tiles of P (one accumulator per thread), hr over the perturbations a: no
    transposed copy of rt.
    """
    if nw == 0 or nb == 0 or nocc == 0 or nat3 == 0:
        return
    var nbo = nb * nocc
    var nwork = max(1, min(parallelism_level(), nw))
    var g2 = nat3 * nat3
    var buf = List[Float64](length=nwork * g2 + 1, fill=0.0)
    var pb = list_ptr(buf)

    def work_gram(w: Int) {imm blas_seq, imm nw, imm nat3, imm nbo, imm rt, imm pb, imm g2, imm nwork}:
        var acc = pb.unsafe_offset(w * g2)
        var p = w
        while p < nw:
            try:
                var r = rt.unsafe_offset(p * nat3 * nbo)
                blas_seq.gemm_small(False, True, nat3, nat3, nbo, 1.0, r, r, 1.0, acc)
            except:
                pass
            p += nwork

    var nthr = blas_seq.serial_begin()
    parallelize(work_gram, nwork)
    if want_h:
        var nwa = max(1, min(parallelism_level(), nat3))

        def work_h(w: Int) {imm blas_seq, imm nw, imm nat3, imm nb, imm nocc, imm nmo, imm i0, imm rt, imm lt, imm hr, imm nbo, imm nwa}:
            var a = w
            while a < nat3:
                try:
                    var h = hr.unsafe_offset(a * nmo * nocc)
                    for p in range(nw):
                        blas_seq.gemm_ld(False, False, nmo, nocc, nb, 1.0, lt.unsafe_offset(p * nocc + i0), nw * nocc,
                                         rt.unsafe_offset((p * nat3 + a) * nbo), nocc, 1.0, h, nocc)
                except:
                    pass
                a += nwa

        parallelize(work_h, nwa)
    blas_seq.serial_end(nthr)
    for w in range(nwork):
        vaxpy(gram, g2, 1.0, pb.unsafe_offset(w * g2))
    _ = buf^


def hess_df_ip1_j_core(
    nao: Int, npf: Int, p0: Int, naux: Int, n3: F64Ptr, dm: F64Ptr, ndm: F64Ptr, want_nt: Bool, coef: F64Ptr,
    nt: F64Ptr, natm: Int, auxloc: I64Ptr,
):
    """Coulomb contractions of a block of first-derivative integrals n3 (3, npf, nao, nao) = (nabla mu nu|P)
    for p0 <= P < p0 + npf: ndm[x, P, mu] = sum_nu n3[x, P, mu, nu] D[mu, nu] (ndm (3, naux, nao), the
    block's rows written) and, with ``want_nt``, nt[A, x] (natm, 3, nao, nao) += sum_{P on A} c_P n3[x, P].
    Tasks are (x, block of rows mu), so every output row has one owner."""
    comptime RB = 16
    var nrb = (nao + RB - 1) // RB
    var ntask = 3 * nrb
    var pat = List[Int](length=max(npf, 1), fill=0)
    for a in range(natm):
        for p in range(max(Int(auxloc[unsafe_offset=a]), p0), min(Int(auxloc[unsafe_offset=a + 1]), p0 + npf)):
            pat[p - p0] = a
    var ppat = Pointer(to=pat)

    def work(t: Int) {imm nao, imm npf, imm p0, imm naux, imm n3, imm dm, imm ndm, imm want_nt, imm coef, imm nt, imm nrb, imm ppat}:
        var x = t // nrb
        var m0 = (t % nrb) * RB
        var m1 = min(nao, m0 + RB)
        for p in range(npf):
            var blk = n3.unsafe_offset((x * npf + p) * nao * nao)
            var drow = ndm.unsafe_offset((x * naux + p0 + p) * nao)
            for m in range(m0, m1):
                drow[unsafe_offset=m] = vdot_serial(blk.unsafe_offset(m * nao), dm.unsafe_offset(m * nao), nao)
            if want_nt:
                var c = coef[unsafe_offset=p0 + p]
                var ntp = nt.unsafe_offset((ppat[][p] * 3 + x) * nao * nao)
                for m in range(m0, m1):
                    vaxpy(ntp.unsafe_offset(m * nao), nao, c, blk.unsafe_offset(m * nao))

    parallelize(work, ntask)
    _ = pat^


def hess_df_ip1_k_core(
    blas_seq: Blas, nao: Int, npf: Int, p0: Int, naux: Int, nocc: Int, n3: F64Ptr, co: F64Ptr, g: F64Ptr,
    want_h: Bool, ck: F64Ptr, hb: F64Ptr, hd: F64Ptr, natm: Int, aoloc: I64Ptr, auxloc: I64Ptr,
) raises:
    """Exchange contractions of a block of first-derivative integrals n3 (3, npf, nao, nao) for one spin:
    g[x, :, P, :] = n3[x, P] C_o (g (3, nao, naux, nocc), the block's columns written) and, with ``want_h``,
    with mk_P = C_o ck_P (ck (naux, nocc, nocc)):
        hb[A, x] (natm, 3, nao, nocc) += sum_P n3[x, P][A rows]^T mk_P[A rows]
        hd[A, x] += sum_{P on A} n3[x, P]^T mk_P
    Each output block has one owner: g over (x, P), hb over (A, x, rows), hd over (A, x, rows); one BLAS
    thread per worker."""
    comptime RB = 32
    if npf == 0 or nocc == 0:
        return
    var nn = nao * nao
    var nthr = blas_seq.serial_begin()

    def work_g(t: Int) {imm blas_seq, imm nao, imm npf, imm p0, imm naux, imm nocc, imm n3, imm co, imm g, imm nn}:
        var x = t // npf
        var p = t % npf
        try:
            blas_seq.gemm_ld(False, False, nao, nocc, nao, 1.0, n3.unsafe_offset((x * npf + p) * nn), nao, co, nocc,
                             0.0, g.unsafe_offset((x * nao * naux + p0 + p) * nocc), naux * nocc)
        except:
            pass

    parallelize(work_g, 3 * npf)
    if want_h:
        var mkl = List[Float64](length=npf * nao * nocc + 1, fill=0.0)
        var mk = list_ptr(mkl)

        def work_mk(p: Int) {imm blas_seq, imm nao, imm p0, imm nocc, imm co, imm ck, imm mk}:
            try:
                blas_seq.gemm_small(False, False, nao, nocc, nocc, 1.0, co, ck.unsafe_offset((p0 + p) * nocc * nocc), 0.0,
                              mk.unsafe_offset(p * nao * nocc))
            except:
                pass

        parallelize(work_mk, npf)
        var nrb = (nao + RB - 1) // RB

        def work_hb(t: Int) {imm blas_seq, imm nao, imm npf, imm nocc, imm n3, imm mk, imm hb, imm nn, imm nrb, imm aoloc}:
            var a = t // (3 * nrb)
            var x = (t // nrb) % 3
            var r0 = (t % nrb) * RB
            var nr = min(RB, nao - r0)
            var m0 = Int(aoloc[unsafe_offset=a])
            var m1 = Int(aoloc[unsafe_offset=a + 1])
            if m1 == m0:
                return
            try:
                var dst = hb.unsafe_offset(((a * 3 + x) * nao + r0) * nocc)
                for p in range(npf):
                    blas_seq.gemm_ld(True, False, nr, nocc, m1 - m0, 1.0,
                                     n3.unsafe_offset((x * npf + p) * nn + m0 * nao + r0), nao,
                                     mk.unsafe_offset((p * nao + m0) * nocc), nocc, 1.0, dst, nocc)
            except:
                pass

        parallelize(work_hb, natm * 3 * nrb)
        var aq = List[Int]()
        for a in range(natm):
            if max(Int(auxloc[unsafe_offset=a]), p0) < min(Int(auxloc[unsafe_offset=a + 1]), p0 + npf):
                aq.append(a)
        var naq = len(aq)
        var paq = Pointer(to=aq)

        def work_hd(t: Int) {imm blas_seq, imm nao, imm npf, imm p0, imm nocc, imm n3, imm mk, imm hd, imm nn, imm nrb, imm auxloc, imm paq}:
            var a = paq[][t // (3 * nrb)]
            var x = (t // nrb) % 3
            var r0 = (t % nrb) * RB
            var nr = min(RB, nao - r0)
            var s0 = max(Int(auxloc[unsafe_offset=a]), p0) - p0
            var s1 = min(Int(auxloc[unsafe_offset=a + 1]), p0 + npf) - p0
            try:
                var dst = hd.unsafe_offset(((a * 3 + x) * nao + r0) * nocc)
                for p in range(s0, s1):
                    blas_seq.gemm_ld(True, False, nr, nocc, nao, 1.0, n3.unsafe_offset((x * npf + p) * nn + r0), nao,
                                     mk.unsafe_offset(p * nao * nocc), nocc, 1.0, dst, nocc)
            except:
                pass

        parallelize(work_hd, naq * 3 * nrb)
        _ = aq^
        _ = mkl^
    blas_seq.serial_end(nthr)
