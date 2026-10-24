"""Coulomb and exchange matrices from 8-fold symmetric packed two-electron integrals.

``eri`` is pyscf's ``aosym='s8'`` tensor: for AO pairs ``ij >= kl`` (pair index
``pair(i, j) = i (i + 1) / 2 + j``, ``i >= j``) the value ``(ij|kl)`` is stored at
``ij (ij + 1) / 2 + kl``.  Densities must be symmetric (pyscf ``hermi=1``).

Each unique integral stands for up to eight index permutations.  Rows ``ij`` are
distributed round-robin over work items, every item accumulates into private
buffers, and the results are reduced at the end:

* J: ``J'_ij += sum_kl (ij|kl) Dt_kl`` and ``J'_kl += (ij|kl) Dt_ij`` (kl < ij) with
  ``Dt`` the packed ``D + D^T`` (diagonal not doubled); J = unpack(J').
* K: for the tuples ``(ij|kl), (ij|lk), (ji|kl), (ji|lk)`` the updates
  ``K'_ik += v D_jl, K'_il += v D_jk, K'_jk += v D_il, K'_jl += v D_ik`` are made
  with ``v`` halved for each coincidence ``i == j``, ``k == l``, ``ij == kl``; the
  remaining four tuples are their transposes, so ``K = K' + K'^T``.
  For fixed ``(ij, k)`` the ``l`` loop is contiguous in memory: two dot products
  and two AXPYs over the integral row segment.

With NK_LANES or more densities (the CPHF of a Hessian: three per atom) the
loops over densities inside every row segment, short dot products and AXPYs
with their overheads, give way to ``_jk_s8_multi``: groups of W densities are
interleaved as the SIMD lanes (``D[a][b][s]``), every integral updates all of
them with vector FMAs, and each row (eight side by side for J) is applied to
all groups while it is in cache.
"""
from std.memory import Pointer
from std.runtime import parallelism_level
from std.sys import simd_width_of
from max.algorithm import parallelize
from _mojo.linalg import F64Ptr, list_ptr, vfill, vcopy, vaxpy, vdot_serial
from _mojo.dfjk import pack_tril_dm, unpack_tril_sym

comptime W = simd_width_of[DType.float64]()
comptime NK_LANES = 3       # densities from which jk_s8_core runs them as SIMD lanes
comptime IntPtr = Pointer[Int, MutAnyOrigin]


def _aligned(mut buf: List[Float64]) -> F64Ptr:
    """The first 64-byte aligned element of ``buf`` (allocated W values longer)."""
    var a = Int(buf.unsafe_ptr())
    return F64Ptr(unsafe_from_address=(a + 63) // 64 * 64)


def _j_block(eri: F64Ptr, npair: Int, r0: Int, dtl: F64Ptr, jc: F64Ptr):
    """J' of the eight rows from r0 (fewer at the end) for W densities as SIMD lanes (``dtl`` the packed D + D^T
    [pair][W], ``jc`` J' [pair][W]): the rows are read side by side, so every element of D + D^T and J' is
    loaded once per block instead of once per row."""
    comptime V = SIMD[DType.float64, W]
    var nr = min(8, npair - r0)
    if nr < 8:
        # the last, short block row by row
        for r in range(nr):
            var ij = r0 + r
            var row = eri.unsafe_offset(ij * (ij + 1) // 2)
            var dtij = dtl.unsafe_load[width=W](ij * W)
            var acc = V(0.0)
            for kl in range(ij):
                var v = row[unsafe_offset=kl]
                acc += v * dtl.unsafe_load[width=W](kl * W)
                jc.unsafe_store(kl * W, jc.unsafe_load[width=W](kl * W) + v * dtij)
            acc += row[unsafe_offset=ij] * dtij
            jc.unsafe_store(ij * W, jc.unsafe_load[width=W](ij * W) + acc)
        return
    var w0 = eri.unsafe_offset(r0 * (r0 + 1) // 2)
    var w1 = eri.unsafe_offset((r0 + 1) * (r0 + 2) // 2)
    var w2 = eri.unsafe_offset((r0 + 2) * (r0 + 3) // 2)
    var w3 = eri.unsafe_offset((r0 + 3) * (r0 + 4) // 2)
    var w4 = eri.unsafe_offset((r0 + 4) * (r0 + 5) // 2)
    var w5 = eri.unsafe_offset((r0 + 5) * (r0 + 6) // 2)
    var w6 = eri.unsafe_offset((r0 + 6) * (r0 + 7) // 2)
    var w7 = eri.unsafe_offset((r0 + 7) * (r0 + 8) // 2)
    var d0 = dtl.unsafe_load[width=W](r0 * W)
    var d1 = dtl.unsafe_load[width=W]((r0 + 1) * W)
    var d2 = dtl.unsafe_load[width=W]((r0 + 2) * W)
    var d3 = dtl.unsafe_load[width=W]((r0 + 3) * W)
    var d4 = dtl.unsafe_load[width=W]((r0 + 4) * W)
    var d5 = dtl.unsafe_load[width=W]((r0 + 5) * W)
    var d6 = dtl.unsafe_load[width=W]((r0 + 6) * W)
    var d7 = dtl.unsafe_load[width=W]((r0 + 7) * W)
    var a0 = V(0.0)
    var a1 = V(0.0)
    var a2 = V(0.0)
    var a3 = V(0.0)
    var a4 = V(0.0)
    var a5 = V(0.0)
    var a6 = V(0.0)
    var a7 = V(0.0)
    # columns before the block: every row of it
    for kl in range(r0):
        var dk = dtl.unsafe_load[width=W](kl * W)
        var v0 = w0[unsafe_offset=kl]
        var v1 = w1[unsafe_offset=kl]
        var v2 = w2[unsafe_offset=kl]
        var v3 = w3[unsafe_offset=kl]
        var v4 = w4[unsafe_offset=kl]
        var v5 = w5[unsafe_offset=kl]
        var v6 = w6[unsafe_offset=kl]
        var v7 = w7[unsafe_offset=kl]
        a0 += v0 * dk
        a1 += v1 * dk
        a2 += v2 * dk
        a3 += v3 * dk
        a4 += v4 * dk
        a5 += v5 * dk
        a6 += v6 * dk
        a7 += v7 * dk
        var js = ((v0 * d0 + v1 * d1) + (v2 * d2 + v3 * d3)) + ((v4 * d4 + v5 * d5) + (v6 * d6 + v7 * d7))
        jc.unsafe_store(kl * W, jc.unsafe_load[width=W](kl * W) + js)
    jc.unsafe_store(r0 * W, jc.unsafe_load[width=W](r0 * W) + a0)
    jc.unsafe_store((r0 + 1) * W, jc.unsafe_load[width=W]((r0 + 1) * W) + a1)
    jc.unsafe_store((r0 + 2) * W, jc.unsafe_load[width=W]((r0 + 2) * W) + a2)
    jc.unsafe_store((r0 + 3) * W, jc.unsafe_load[width=W]((r0 + 3) * W) + a3)
    jc.unsafe_store((r0 + 4) * W, jc.unsafe_load[width=W]((r0 + 4) * W) + a4)
    jc.unsafe_store((r0 + 5) * W, jc.unsafe_load[width=W]((r0 + 5) * W) + a5)
    jc.unsafe_store((r0 + 6) * W, jc.unsafe_load[width=W]((r0 + 6) * W) + a6)
    jc.unsafe_store((r0 + 7) * W, jc.unsafe_load[width=W]((r0 + 7) * W) + a7)
    # the triangle of the block's own columns
    for r in range(8):
        var ij = r0 + r
        var row = eri.unsafe_offset(ij * (ij + 1) // 2)
        var dtij = dtl.unsafe_load[width=W](ij * W)
        var acc = V(0.0)
        for kl in range(r0, ij):
            var v = row[unsafe_offset=kl]
            acc += v * dtl.unsafe_load[width=W](kl * W)
            jc.unsafe_store(kl * W, jc.unsafe_load[width=W](kl * W) + v * dtij)
        acc += row[unsafe_offset=ij] * dtij
        jc.unsafe_store(ij * W, jc.unsafe_load[width=W](ij * W) + acc)


def _jk_s8_lanes_j(eri: F64Ptr, npair: Int, dtl: F64Ptr, jc: F64Ptr, c: Int, nchunks: Int, ng: Int, gs: Int):
    """J' of work item ``c`` (``_j_block`` for the blocks b of eight rows with b mod nchunks == c) for ``ng`` groups
    of W densities, ``gs`` values apart in ``dtl`` and ``jc``: each block's rows are applied to every group while
    they are in cache."""
    var nblk = (npair + 7) // 8
    var b = c
    while b < nblk:
        for g in range(ng):
            _j_block(eri, npair, 8 * b, dtl.unsafe_offset(g * gs), jc.unsafe_offset(g * gs))
        b += nchunks


def _k_elem(
    v: Float64, l: Int, k: Int, kl: Int, ij: Int, w_ij: Float64, di: F64Ptr, dj: F64Ptr, ki: F64Ptr, kj: F64Ptr,
    dik: SIMD[DType.float64, W], djk: SIMD[DType.float64, W], mut aik: SIMD[DType.float64, W],
    mut ajk: SIMD[DType.float64, W],
):
    """One integral v = (ij|kl) of ``_jk_s8_lanes_k`` with its weight (halved for l == k and for kl == ij)."""
    var w = w_ij
    if l == k:
        w *= 0.5
    if kl == ij:
        w *= 0.5
    var vw = SIMD[DType.float64, W](v * w)
    var o = l * W
    aik += vw * dj.unsafe_load[width=W](o)
    ajk += vw * di.unsafe_load[width=W](o)
    ki.unsafe_store(o, ki.unsafe_load[width=W](o) + vw * djk)
    kj.unsafe_store(o, kj.unsafe_load[width=W](o) + vw * dik)


def _k_row(row: F64Ptr, ij: Int, i: Int, j: Int, di: F64Ptr, dj: F64Ptr, ki: F64Ptr, kj: F64Ptr):
    """K' of row ij = (i, j) of the integrals for W densities as SIMD lanes (``di``, ``dj`` the rows i, j of the
    densities [nao][W], ``ki``, ``kj`` those of K'): the updates of the module docstring with every integral
    read once for all densities.

    The segments of four consecutive k of a row (at the end one at a time) are taken together over their
    common l: K'_il and K'_jl get all four contributions in one update and D_jl, D_il serve all four, and
    the row's weight goes into D_ik, D_jk and the sums, so that an integral is one broadcast load and four
    FMAs; the triangle each group leaves (l from k to k + t in segment t) is unrolled with its D values in
    registers.  Single-thread, data in cache, W = 8: 9-10 cycles per integral one segment at a time, 6-7
    with the groups."""
    comptime V = SIMD[DType.float64, W]
    var w_ij = 0.5 if i == j else 1.0
    var wv = V(w_ij)
    var k = 0
    while k <= i:
        var kl0 = k * (k + 1) // 2
        if kl0 > ij:
            break
        var kl1 = kl0 + k + 1               # first pairs of k + 1, k + 2, k + 3
        var kl2 = kl1 + k + 2
        var kl3 = kl2 + k + 3
        var lm0 = min(k, ij - kl0)          # last l of k (inclusive)
        var seg0 = row.unsafe_offset(kl0)
        var dik0 = di.unsafe_load[width=W](k * W)
        var djk0 = dj.unsafe_load[width=W](k * W)
        var a0 = V(0.0)
        var b0 = V(0.0)
        if k + 3 <= i and kl3 <= ij:
            # k .. k + 3: l < m plain for all four
            var lm1 = min(k + 1, ij - kl1)
            var lm2 = min(k + 2, ij - kl2)
            var lm3 = min(k + 3, ij - kl3)
            var m = min(min(lm0, lm1), min(lm2, lm3))
            var seg1 = row.unsafe_offset(kl1)
            var seg2 = row.unsafe_offset(kl2)
            var seg3 = row.unsafe_offset(kl3)
            var dik1 = di.unsafe_load[width=W]((k + 1) * W)
            var djk1 = dj.unsafe_load[width=W]((k + 1) * W)
            var dik2 = di.unsafe_load[width=W]((k + 2) * W)
            var djk2 = dj.unsafe_load[width=W]((k + 2) * W)
            var dik3 = di.unsafe_load[width=W]((k + 3) * W)
            var djk3 = dj.unsafe_load[width=W]((k + 3) * W)
            var wik0 = dik0 * wv
            var wjk0 = djk0 * wv
            var wik1 = dik1 * wv
            var wjk1 = djk1 * wv
            var wik2 = dik2 * wv
            var wjk2 = djk2 * wv
            var wik3 = dik3 * wv
            var wjk3 = djk3 * wv
            var a1 = V(0.0)
            var b1 = V(0.0)
            var a2 = V(0.0)
            var b2 = V(0.0)
            var a3 = V(0.0)
            var b3 = V(0.0)
            for l in range(m):
                var o = l * W
                var u0 = V(seg0[unsafe_offset=l])
                var u1 = V(seg1[unsafe_offset=l])
                var u2 = V(seg2[unsafe_offset=l])
                var u3 = V(seg3[unsafe_offset=l])
                var djl = dj.unsafe_load[width=W](o)
                var dil = di.unsafe_load[width=W](o)
                a0 += u0 * djl
                b0 += u0 * dil
                a1 += u1 * djl
                b1 += u1 * dil
                a2 += u2 * djl
                b2 += u2 * dil
                a3 += u3 * djl
                b3 += u3 * dil
                # K'_il, then K'_jl: the same element when i == j, so the second load follows the first store
                var xi = ki.unsafe_load[width=W](o)
                xi = u0.fma(wjk0, xi)
                xi = u1.fma(wjk1, xi)
                xi = u2.fma(wjk2, xi)
                xi = u3.fma(wjk3, xi)
                ki.unsafe_store(o, xi)
                var xj = kj.unsafe_load[width=W](o)
                xj = u0.fma(wik0, xj)
                xj = u1.fma(wik1, xj)
                xj = u2.fma(wik2, xj)
                xj = u3.fma(wik3, xj)
                kj.unsafe_store(o, xj)
            a0 *= wv
            b0 *= wv
            a1 *= wv
            b1 *= wv
            a2 *= wv
            b2 *= wv
            a3 *= wv
            b3 *= wv
            if m == k and kl3 + k + 3 < ij:
                # complete segments short of the row's last integral: the rest is the triangle l = k + s,
                # s <= t, of segment t, whose D_jl, D_il are D_jk_s, D_ik_s; K'_il at l = k + s receives
                # with the sums of segment s (halved weight on the diagonal s == t)
                var hv = wv * 0.5
                var u00 = V(seg0[unsafe_offset=k]) * hv
                var u10 = V(seg1[unsafe_offset=k]) * wv
                var u11 = V(seg1[unsafe_offset=k + 1]) * hv
                var u20 = V(seg2[unsafe_offset=k]) * wv
                var u21 = V(seg2[unsafe_offset=k + 1]) * wv
                var u22 = V(seg2[unsafe_offset=k + 2]) * hv
                var u30 = V(seg3[unsafe_offset=k]) * wv
                var u31 = V(seg3[unsafe_offset=k + 1]) * wv
                var u32 = V(seg3[unsafe_offset=k + 2]) * wv
                var u33 = V(seg3[unsafe_offset=k + 3]) * hv
                a0 += u00 * djk0
                b0 += u00 * dik0
                a1 += u10 * djk0 + u11 * djk1
                b1 += u10 * dik0 + u11 * dik1
                a2 += (u20 * djk0 + u21 * djk1) + u22 * djk2
                b2 += (u20 * dik0 + u21 * dik1) + u22 * dik2
                a3 += (u30 * djk0 + u31 * djk1) + (u32 * djk2 + u33 * djk3)
                b3 += (u30 * dik0 + u31 * dik1) + (u32 * dik2 + u33 * dik3)
                # scatter at l = k + s from the segments t >= s
                a0 += (u00 * djk0 + u10 * djk1) + (u20 * djk2 + u30 * djk3)
                b0 += (u00 * dik0 + u10 * dik1) + (u20 * dik2 + u30 * dik3)
                a1 += (u11 * djk1 + u21 * djk2) + u31 * djk3
                b1 += (u11 * dik1 + u21 * dik2) + u31 * dik3
                a2 += u22 * djk2 + u32 * djk3
                b2 += u22 * dik2 + u32 * dik3
                a3 += u33 * djk3
                b3 += u33 * dik3
            else:
                # the rest of each segment one integral at a time (its last with its weight)
                for l2 in range(m, lm0 + 1):
                    _k_elem(seg0[unsafe_offset=l2], l2, k, kl0 + l2, ij, w_ij, di, dj, ki, kj, dik0, djk0, a0, b0)
                for l2 in range(m, lm1 + 1):
                    _k_elem(seg1[unsafe_offset=l2], l2, k + 1, kl1 + l2, ij, w_ij, di, dj, ki, kj, dik1, djk1, a1, b1)
                for l2 in range(m, lm2 + 1):
                    _k_elem(seg2[unsafe_offset=l2], l2, k + 2, kl2 + l2, ij, w_ij, di, dj, ki, kj, dik2, djk2, a2, b2)
                for l2 in range(m, lm3 + 1):
                    _k_elem(seg3[unsafe_offset=l2], l2, k + 3, kl3 + l2, ij, w_ij, di, dj, ki, kj, dik3, djk3, a3, b3)
            ki.unsafe_store((k + 3) * W, ki.unsafe_load[width=W]((k + 3) * W) + a3)
            kj.unsafe_store((k + 3) * W, kj.unsafe_load[width=W]((k + 3) * W) + b3)
            ki.unsafe_store((k + 2) * W, ki.unsafe_load[width=W]((k + 2) * W) + a2)
            kj.unsafe_store((k + 2) * W, kj.unsafe_load[width=W]((k + 2) * W) + b2)
            ki.unsafe_store((k + 1) * W, ki.unsafe_load[width=W]((k + 1) * W) + a1)
            kj.unsafe_store((k + 1) * W, kj.unsafe_load[width=W]((k + 1) * W) + b1)
            ki.unsafe_store(k * W, ki.unsafe_load[width=W](k * W) + a0)
            kj.unsafe_store(k * W, kj.unsafe_load[width=W](k * W) + b0)
            k += 4
        else:
            for l2 in range(lm0 + 1):
                _k_elem(seg0[unsafe_offset=l2], l2, k, kl0 + l2, ij, w_ij, di, dj, ki, kj, dik0, djk0, a0, b0)
            ki.unsafe_store(k * W, ki.unsafe_load[width=W](k * W) + a0)
            kj.unsafe_store(k * W, kj.unsafe_load[width=W](k * W) + b0)
            k += 1


def _jk_s8_lanes_k(
    eri: F64Ptr, nao: Int, npair: Int, ppi: IntPtr, ppj: IntPtr, dl: F64Ptr, kc: F64Ptr, c: Int, nchunks: Int,
    ng: Int,
):
    """K' of work item ``c`` (``_k_row`` for the rows ij = c, c + nchunks, ...) for ``ng`` groups of W densities
    (``dl`` the densities [g][nao][nao][W], ``kc`` the item's K' likewise): each row is applied to every group
    while it is in cache."""
    var gs = nao * nao * W
    var ij = c
    while ij < npair:
        var row = eri.unsafe_offset(ij * (ij + 1) // 2)
        var i = ppi[unsafe_offset=ij]
        var j = ppj[unsafe_offset=ij]
        for g in range(ng):
            var dg = dl.unsafe_offset(g * gs)
            var kg = kc.unsafe_offset(g * gs)
            _k_row(
                row, ij, i, j, dg.unsafe_offset(i * nao * W), dg.unsafe_offset(j * nao * W),
                kg.unsafe_offset(i * nao * W), kg.unsafe_offset(j * nao * W),
            )
        ij += nchunks


comptime GMAX = 8          # lane groups (W densities each) of one sweep of _jk_s8_multi over the integrals


def _jk_s8_multi(
    eri: F64Ptr, nao: Int, nj: Int, dms_j: F64Ptr, vj: F64Ptr, nk: Int, dms_k: F64Ptr, vk: F64Ptr, nanti: Int,
):
    """``jk_s8_core`` with the densities as SIMD lanes: groups of W (density g W + t of the J and of the K
    densities in lane t of group g), up to GMAX groups applied together in one sweep over the integrals (each
    row, or block of eight for J, goes to all groups while it is in cache; one sweep per group read the
    tensor from memory again), J and K in separate passes (``_jk_s8_lanes_j``, ``_jk_s8_lanes_k``)."""
    var npair = nao * (nao + 1) // 2
    var n2 = nao * nao
    var nchunks = max(1, parallelism_level())
    if npair * (npair + 1) // 2 < (1 << 18):
        nchunks = 1
    if nchunks > npair:
        nchunks = npair
    var pi = List[Int](length=npair, fill=0)
    var pj = List[Int](length=npair, fill=0)
    var p = 0
    for i in range(nao):
        for j in range(i + 1):
            pi[p] = i
            pj[p] = j
            p += 1
    var ppi = IntPtr(unsafe_from_address=Int(pi.unsafe_ptr()))
    var ppj = IntPtr(unsafe_from_address=Int(pj.unsafe_ptr()))
    var ngj_all = (nj + W - 1) // W
    var ngk_all = (nk + W - 1) // W
    var gmax = min(GMAX, max(ngj_all, ngk_all))
    var gj = min(gmax, ngj_all)         # room for the J and K groups of a sweep
    var gk = min(gmax, ngk_all)
    var sj = npair * W                  # group strides
    var sk = n2 * W
    # the lane arrays 64-byte aligned (every element one cache line)
    var dtlb = List[Float64](length=gj * sj + 2 * W, fill=0.0)
    var dlb = List[Float64](length=gk * sk + 2 * W, fill=0.0)
    var jbuf = List[Float64](length=nchunks * gj * sj + 2 * W, fill=0.0)
    var kbuf = List[Float64](length=nchunks * gk * sk + 2 * W, fill=0.0)
    var tmp = List[Float64](length=npair, fill=0.0)
    var pdtl = _aligned(dtlb)
    var pdl = _aligned(dlb)
    var pjb = _aligned(jbuf)
    var pkb = _aligned(kbuf)
    var ptmp = list_ptr(tmp)
    var g0 = 0
    while g0 < max(ngj_all, ngk_all):
        var ngj = min(gj, max(ngj_all - g0, 0))     # groups of this sweep
        var ngk = min(gk, max(ngk_all - g0, 0))
        var s0 = g0 * W
        var mj = min(ngj * W, max(nj - s0, 0))      # densities of this sweep
        var mk = min(ngk * W, max(nk - s0, 0))
        # interleave the sweep's densities (lanes past mj, mk zero)
        if mj > 0:
            vfill(pdtl, ngj * sj, 0.0)
            for t in range(mj):
                pack_tril_dm(dms_j.unsafe_offset((s0 + t) * n2), nao, ptmp)
                var dst = pdtl.unsafe_offset((t // W) * sj + t % W)
                for q in range(npair):
                    dst[unsafe_offset=q * W] = ptmp[unsafe_offset=q]
            vfill(pjb, nchunks * ngj * sj, 0.0)
        if mk > 0:
            vfill(pdl, ngk * sk, 0.0)
            for t in range(mk):
                var d = dms_k.unsafe_offset((s0 + t) * n2)
                var dst = pdl.unsafe_offset((t // W) * sk + t % W)
                for q in range(n2):
                    dst[unsafe_offset=q * W] = d[unsafe_offset=q]
            vfill(pkb, nchunks * ngk * sk, 0.0)

        def work(c: Int) {imm eri, imm nao, imm npair, imm ppi, imm ppj, imm pdtl, imm pdl, imm pjb, imm pkb, imm nchunks, imm ngj, imm ngk, imm sj, imm sk}:
            if ngj > 0:
                _jk_s8_lanes_j(eri, npair, pdtl, pjb.unsafe_offset(c * ngj * sj), c, nchunks, ngj, sj)
            if ngk > 0:
                _jk_s8_lanes_k(eri, nao, npair, ppi, ppj, pdl, pkb.unsafe_offset(c * ngk * sk), c, nchunks, ngk)

        parallelize(work, nchunks)
        # reductions over the work items, then out of the lanes
        for c in range(1, nchunks):
            if mj > 0:
                vaxpy(pjb, ngj * sj, 1.0, pjb.unsafe_offset(c * ngj * sj))
            if mk > 0:
                vaxpy(pkb, ngk * sk, 1.0, pkb.unsafe_offset(c * ngk * sk))
        for t in range(mj):
            var src = pjb.unsafe_offset((t // W) * sj + t % W)
            for q in range(npair):
                ptmp[unsafe_offset=q] = src[unsafe_offset=q * W]
            unpack_tril_sym(ptmp, nao, vj.unsafe_offset((s0 + t) * n2))
        for t in range(mk):
            var src = pkb.unsafe_offset((t // W) * sk + t % W)
            var out = vk.unsafe_offset((s0 + t) * n2)
            var sgn = -1.0 if s0 + t >= nk - nanti else 1.0
            for a in range(nao):
                for b in range(nao):
                    out[unsafe_offset=a * nao + b] = (
                        src[unsafe_offset=(a * nao + b) * W] + sgn * src[unsafe_offset=(b * nao + a) * W]
                    )
        g0 += gmax
    _ = pi^
    _ = pj^
    _ = dtlb^
    _ = dlb^
    _ = jbuf^
    _ = kbuf^
    _ = tmp^


def jk_s8_core(
    eri: F64Ptr,
    nao: Int,
    nj: Int,
    dms_j: F64Ptr,
    vj: F64Ptr,
    nk: Int,
    dms_k: F64Ptr,
    vk: F64Ptr,
    nanti: Int = 0,
):
    """J for ``nj`` densities and K for ``nk`` densities (symmetric, nao x nao; the last ``nanti`` exchange
    densities antisymmetric: their K = K' - K'^T, the transposed tuples changing sign).

    vj : nj x nao x nao and vk : nk x nao x nao are overwritten.
    """
    if max(nj, nk) >= NK_LANES:
        _jk_s8_multi(eri, nao, nj, dms_j, vj, nk, dms_k, vk, nanti)
        return
    var npair = nao * (nao + 1) // 2
    var n2 = nao * nao
    # Small problems (a few hundred AO pairs) are faster on one thread than with
    # the dispatch and reduction overhead of several work items.
    var nchunks = max(1, 4 * parallelism_level())
    if npair * (npair + 1) // 2 < (1 << 18):
        nchunks = 1
    if nchunks > npair:
        nchunks = npair

    # pair index -> (i, j)
    var pi = List[Int](length=npair, fill=0)
    var pj = List[Int](length=npair, fill=0)
    var p = 0
    for i in range(nao):
        for j in range(i + 1):
            pi[p] = i
            pj[p] = j
            p += 1
    var ppi = Pointer[Int, MutAnyOrigin](unsafe_from_address=Int(pi.unsafe_ptr()))
    var ppj = Pointer[Int, MutAnyOrigin](unsafe_from_address=Int(pj.unsafe_ptr()))

    # packed D + D^T for the Coulomb part
    var dt = List[Float64](length=max(nj, 1) * npair, fill=0.0)
    var pdt = list_ptr(dt)
    for s in range(nj):
        pack_tril_dm(dms_j.unsafe_offset(s * n2), nao, pdt.unsafe_offset(s * npair))

    var jbuf = List[Float64](length=nchunks * max(nj, 1) * npair, fill=0.0)
    var kbuf = List[Float64](length=nchunks * max(nk, 1) * n2, fill=0.0)
    var pjb = list_ptr(jbuf)
    var pkb = list_ptr(kbuf)

    def work(c: Int) {imm eri, imm nao, imm npair, imm n2, imm nchunks, imm nj, imm nk, imm pdt, imm dms_k, imm pjb, imm pkb, imm ppi, imm ppj}:
        var jc = pjb.unsafe_offset(c * nj * npair)
        var kc = pkb.unsafe_offset(c * nk * n2)
        var ij = c
        while ij < npair:
            var row = eri.unsafe_offset(ij * (ij + 1) // 2)
            var i = ppi[unsafe_offset=ij]
            var j = ppj[unsafe_offset=ij]
            # Coulomb
            for s in range(nj):
                var dts = pdt.unsafe_offset(s * npair)
                var js = jc.unsafe_offset(s * npair)
                js[unsafe_offset=ij] = js[unsafe_offset=ij] + vdot_serial(row, dts, ij + 1)
                vaxpy(js, ij, dts[unsafe_offset=ij], row)
            # Exchange
            if nk > 0:
                var w_ij = 0.5 if i == j else 1.0
                var k = 0
                while k < nao:
                    var kl0 = k * (k + 1) // 2
                    if kl0 > ij:
                        break
                    var lmax = min(k, ij - kl0)          # last l of this k (inclusive)
                    var seg = row.unsafe_offset(kl0)
                    var nfull = lmax                     # l in [0, lmax) carry the plain weight
                    var w_last = w_ij
                    if lmax == k:
                        w_last *= 0.5
                    if kl0 + lmax == ij:
                        w_last *= 0.5
                    var vlast = seg[unsafe_offset=lmax] * w_last
                    for s in range(nk):
                        var d = dms_k.unsafe_offset(s * n2)
                        var ks = kc.unsafe_offset(s * n2)
                        var di = d.unsafe_offset(i * nao)
                        var dj = d.unsafe_offset(j * nao)
                        var ki = ks.unsafe_offset(i * nao)
                        var kj = ks.unsafe_offset(j * nao)
                        var dik = di[unsafe_offset=k]
                        var djk = dj[unsafe_offset=k]
                        # K'_ik += sum_l v_l D_jl ;  K'_jk += sum_l v_l D_il
                        var acc_ik = w_ij * vdot_serial(seg, dj, nfull) + vlast * dj[unsafe_offset=lmax]
                        var acc_jk = w_ij * vdot_serial(seg, di, nfull) + vlast * di[unsafe_offset=lmax]
                        ki[unsafe_offset=k] = ki[unsafe_offset=k] + acc_ik
                        kj[unsafe_offset=k] = kj[unsafe_offset=k] + acc_jk
                        # K'_il += v_l D_jk ;  K'_jl += v_l D_ik   (l < lmax vectorised, l = lmax scalar)
                        vaxpy(ki, nfull, w_ij * djk, seg)
                        vaxpy(kj, nfull, w_ij * dik, seg)
                        ki[unsafe_offset=lmax] = ki[unsafe_offset=lmax] + vlast * djk
                        kj[unsafe_offset=lmax] = kj[unsafe_offset=lmax] + vlast * dik
                    k += 1
            ij += nchunks

    parallelize(work, nchunks)

    # reductions
    if nj > 0:
        var jsum = List[Float64](length=npair, fill=0.0)
        var pjs = list_ptr(jsum)
        for s in range(nj):
            vfill(pjs, npair, 0.0)
            for c in range(nchunks):
                vaxpy(pjs, npair, 1.0, pjb.unsafe_offset((c * nj + s) * npair))
            unpack_tril_sym(pjs, nao, vj.unsafe_offset(s * n2))
        _ = jsum^
    if nk > 0:
        var ksum = List[Float64](length=n2, fill=0.0)
        var pks = list_ptr(ksum)
        for s in range(nk):
            vfill(pks, n2, 0.0)
            for c in range(nchunks):
                vaxpy(pks, n2, 1.0, pkb.unsafe_offset((c * nk + s) * n2))
            var out = vk.unsafe_offset(s * n2)
            var sgn = -1.0 if s >= nk - nanti else 1.0
            for a in range(nao):
                for b in range(nao):
                    out[unsafe_offset=a * nao + b] = pks[unsafe_offset=a * nao + b] + sgn * pks[unsafe_offset=b * nao + a]
        _ = ksum^
    _ = pi^
    _ = pj^
    _ = dt^
    _ = jbuf^
    _ = kbuf^
