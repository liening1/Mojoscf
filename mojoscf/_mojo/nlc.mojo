"""VV10 non-local correlation: the pair sums of pyscf's ``numint._vv10nlc``.

For every outer grid point i (coordinates r_i, W0_i, K_i) and the inner
points j (r_j, W0'_j, K'_j and RpW_j = rho_j w_j), with R2 = |r_i - r_j|^2,

    g = R2 W0_i + K_i,  g' = R2 W0'_j + K'_j,  T = RpW_j / (g g' (g + g')),
    F_i = -3/2 sum_j T,  U_i = sum_j T (1/g + 1/(g + g')),  W_i = sum_j T (1/g + 1/(g + g')) R2.

One reciprocal per pair: with inv = 1 / (g g' (g + g')), T = RpW inv and
T (1/g + 1/(g + g')) = T inv g' (2 g + g').  The double division bounds a
loop this short (16 cycles per vector, against about 10 for everything else),
so four vectors of pairs share one: 1 / (d1 d2 d3 d4) times the other three
denominators gives each reciprocal (a few roundings more than a division;
every denominator is at least 2 K^3 > 0.5 for densities above pyscf's 1e-8
threshold and far below 1e70, so the product of four stays in range).  The inner points are taken in tiles (structure of arrays) that
stay in cache while a block of outer points runs over them, SIMD over the
inner points; workers take blocks of outer points.
"""
from std.runtime import parallelism_level
from max.algorithm import parallelize
from _mojo.linalg import F64Ptr, list_ptr

comptime W = 8
comptime F64V = SIMD[DType.float64, W]
comptime OUTER_BLOCK = 64
comptime INNER_TILE = 512


@always_inline
def _pair_terms(
    j: Int, xi: F64V, yi: F64V, zi: F64V, w0i: F64V, ki: F64V, px: F64Ptr, py: F64Ptr, pz: F64Ptr, pw: F64Ptr,
    pk: F64Ptr,
) -> Tuple[F64V, F64V, F64V, F64V]:
    """(R2, g_ij, g_ji, D = g_ij g_ji (g_ij + g_ji)) of point i with the vector of points at j."""
    var dx = px.unsafe_load[width=W](j) - xi
    var dy = py.unsafe_load[width=W](j) - yi
    var dz = pz.unsafe_load[width=W](j) - zi
    var r2 = dx * dx + dy * dy + dz * dz
    var gij = r2 * w0i + ki
    var gji = r2 * pw.unsafe_load[width=W](j) + pk.unsafe_load[width=W](j)
    return (r2, gij, gji, gij * gji * (gij + gji))


@always_inline
def _add_outer(
    j: Int, t: Tuple[F64V, F64V, F64V, F64V], inv: F64V, pr: F64Ptr, mut sf: F64V, mut su: F64V, mut sw: F64V
):
    """The terms of point i (``vv10_core``) for the pairs ``t`` with inv = 1 / D."""
    var tj = pr.unsafe_load[width=W](j) * inv
    var t2 = tj * inv * t[2] * (t[1] + (t[1] + t[2]))
    sf += tj
    su += t2
    sw += t2 * t[0]


def vv10_core(
    nout: Int,
    xyz: F64Ptr,
    w0: F64Ptr,
    k: F64Ptr,
    nin: Int,
    vxyz: F64Ptr,
    w0p: F64Ptr,
    kp: F64Ptr,
    rpw: F64Ptr,
    f_out: F64Ptr,
    u_out: F64Ptr,
    w_out: F64Ptr,
):
    """F, U, W (nout) of the VV10 kernel; xyz (nout, 3), vxyz (nin, 3) row-major."""
    if nout == 0:
        return
    # inner points as structure of arrays, padded to a multiple of W with RpW = 0 (and K' = 1: no 0/0)
    var npad = ((nin + W - 1) // W) * W
    var soa = List[Float64](length=6 * max(npad, W), fill=0.0)
    var ps = list_ptr(soa)
    var px = ps
    var py = ps.unsafe_offset(npad)
    var pz = ps.unsafe_offset(2 * npad)
    var pw = ps.unsafe_offset(3 * npad)
    var pk = ps.unsafe_offset(4 * npad)
    var pr = ps.unsafe_offset(5 * npad)
    for j in range(npad):
        if j < nin:
            px[unsafe_offset=j] = vxyz[unsafe_offset=3 * j]
            py[unsafe_offset=j] = vxyz[unsafe_offset=3 * j + 1]
            pz[unsafe_offset=j] = vxyz[unsafe_offset=3 * j + 2]
            pw[unsafe_offset=j] = w0p[unsafe_offset=j]
            pk[unsafe_offset=j] = kp[unsafe_offset=j]
            pr[unsafe_offset=j] = rpw[unsafe_offset=j]
        else:
            pk[unsafe_offset=j] = 1.0
    var nblock = (nout + OUTER_BLOCK - 1) // OUTER_BLOCK

    def block(b: Int) {imm nout, imm xyz, imm w0, imm k, imm npad, imm px, imm py, imm pz, imm pw, imm pk, imm pr, imm f_out, imm u_out, imm w_out}:
        var i0 = b * OUTER_BLOCK
        var ni = min(OUTER_BLOCK, nout - i0)
        var acc = List[Float64](length=3 * OUTER_BLOCK * W, fill=0.0)   # F, U, W partial sums per outer point
        var pa = list_ptr(acc)
        var j0 = 0
        while j0 < npad:
            var j1 = min(j0 + INNER_TILE, npad)
            for ii in range(ni):
                var i = i0 + ii
                var xi = F64V(xyz[unsafe_offset=3 * i])
                var yi = F64V(xyz[unsafe_offset=3 * i + 1])
                var zi = F64V(xyz[unsafe_offset=3 * i + 2])
                var w0i = F64V(w0[unsafe_offset=i])
                var ki = F64V(k[unsafe_offset=i])
                var sf = pa.unsafe_load[width=W](3 * ii * W)
                var su = pa.unsafe_load[width=W]((3 * ii + 1) * W)
                var sw = pa.unsafe_load[width=W]((3 * ii + 2) * W)
                var j = j0
                while j + 3 * W < j1:                           # four vectors, one division
                    var a = _pair_terms(j, xi, yi, zi, w0i, ki, px, py, pz, pw, pk)
                    var b2 = _pair_terms(j + W, xi, yi, zi, w0i, ki, px, py, pz, pw, pk)
                    var c = _pair_terms(j + 2 * W, xi, yi, zi, w0i, ki, px, py, pz, pw, pk)
                    var e = _pair_terms(j + 3 * W, xi, yi, zi, w0i, ki, px, py, pz, pw, pk)
                    var dab = a[3] * b2[3]
                    var dce = c[3] * e[3]
                    var inv = 1.0 / (dab * dce)
                    var iab = inv * dce
                    var ice = inv * dab
                    _add_outer(j, a, iab * b2[3], pr, sf, su, sw)
                    _add_outer(j + W, b2, iab * a[3], pr, sf, su, sw)
                    _add_outer(j + 2 * W, c, ice * e[3], pr, sf, su, sw)
                    _add_outer(j + 3 * W, e, ice * c[3], pr, sf, su, sw)
                    j += 4 * W
                while j < j1:
                    var a = _pair_terms(j, xi, yi, zi, w0i, ki, px, py, pz, pw, pk)
                    _add_outer(j, a, 1.0 / a[3], pr, sf, su, sw)
                    j += W
                pa.unsafe_store[width=W](3 * ii * W, sf)
                pa.unsafe_store[width=W]((3 * ii + 1) * W, su)
                pa.unsafe_store[width=W]((3 * ii + 2) * W, sw)
            j0 = j1
        for ii in range(ni):
            f_out[unsafe_offset=i0 + ii] = -1.5 * pa.unsafe_load[width=W](3 * ii * W).reduce_add()
            u_out[unsafe_offset=i0 + ii] = pa.unsafe_load[width=W]((3 * ii + 1) * W).reduce_add()
            w_out[unsafe_offset=i0 + ii] = pa.unsafe_load[width=W]((3 * ii + 2) * W).reduce_add()
        _ = acc^

    parallelize(block, nblock)
    _ = soa^


comptime SYM_BLOCK = 256


@always_inline
def _add_sym(
    j: Int, t: Tuple[F64V, F64V, F64V, F64V], inv: F64V, ri: F64V, pr: F64Ptr, mut sf: F64V, mut su: F64V,
    mut sw: F64V, af: F64Ptr, au: F64Ptr, aw: F64Ptr,
):
    """The terms of both points of the pairs ``t`` (``vv10_sym_core``) with inv = 1 / D: point i into the
    registers, the points j into their accumulators."""
    var q = inv * inv
    var p = t[1] * t[2]
    var p2 = p + p
    var rj = pr.unsafe_load[width=W](j)
    var ti = rj * q * (t[2] * t[2] + p2)          # point i: RpW_j g_ji (2 g_ij + g_ji) / D^2
    var tj = ri * q * (t[1] * t[1] + p2)          # point j: RpW_i g_ij (2 g_ji + g_ij) / D^2
    sf += rj * inv
    su += ti
    sw += ti * t[0]
    af.unsafe_store(j, af.unsafe_load[width=W](j) + ri * inv)
    au.unsafe_store(j, au.unsafe_load[width=W](j) + tj)
    aw.unsafe_store(j, aw.unsafe_load[width=W](j) + tj * t[0])


def vv10_sym_core(
    n: Int, xyz: F64Ptr, w0: F64Ptr, k: F64Ptr, rpw: F64Ptr, f_out: F64Ptr, u_out: F64Ptr, w_out: F64Ptr
):
    """``vv10_core`` for one grid as both the outer and the inner points (pyscf's ``nr_nlc_vxc`` and the
    NLC gradient): with g_ij = R2 W0_i + K_i the denominator D_ij = g_ij g_ji (g_ij + g_ji) is symmetric,
    so every unordered pair takes one reciprocal and adds RpW_j / D, RpW_j g_ji (2 g_ij + g_ji) / D^2 (and
    times R2) to point i and the same with i and j exchanged to point j.  Blocks of points (structure of
    arrays) are paired I <= J; within a diagonal block a lane mask keeps j > i, and the j == i terms are
    added once.  Workers take every nwork-th row block, with private accumulators for the j side."""
    if n == 0:
        return
    var npad = ((n + SYM_BLOCK - 1) // SYM_BLOCK) * SYM_BLOCK
    var soa = List[Float64](length=6 * npad, fill=0.0)
    var ps = list_ptr(soa)
    var px = ps
    var py = ps.unsafe_offset(npad)
    var pz = ps.unsafe_offset(2 * npad)
    var pw = ps.unsafe_offset(3 * npad)
    var pk = ps.unsafe_offset(4 * npad)
    var pr = ps.unsafe_offset(5 * npad)
    for j in range(npad):
        if j < n:
            px[unsafe_offset=j] = xyz[unsafe_offset=3 * j]
            py[unsafe_offset=j] = xyz[unsafe_offset=3 * j + 1]
            pz[unsafe_offset=j] = xyz[unsafe_offset=3 * j + 2]
            pw[unsafe_offset=j] = w0[unsafe_offset=j]
            pk[unsafe_offset=j] = k[unsafe_offset=j]
            pr[unsafe_offset=j] = rpw[unsafe_offset=j]
        else:
            pk[unsafe_offset=j] = 1.0                    # padding: RpW = 0, no 0/0
    var nblk = npad // SYM_BLOCK
    var nwork = max(1, min(parallelism_level(), nblk))
    var acc = List[Float64](length=nwork * 3 * npad, fill=0.0)    # per worker F, U, W for every point
    var pa = list_ptr(acc)

    def work(wk: Int) {imm npad, imm nblk, imm nwork, imm px, imm py, imm pz, imm pw, imm pk, imm pr, imm pa}:
        var af = pa.unsafe_offset(wk * 3 * npad)
        var au = af.unsafe_offset(npad)
        var aw = au.unsafe_offset(npad)
        var part = List[Float64](length=3 * SYM_BLOCK * W, fill=0.0)   # SIMD partial sums of the block's points
        var pp = list_ptr(part)
        var bi = wk
        while bi < nblk:
            var i0 = bi * SYM_BLOCK
            for t in range(3 * SYM_BLOCK * W):
                pp[unsafe_offset=t] = 0.0
            var j0 = i0
            while j0 < npad:                                    # tiles of later points, cached for the block
                var j1 = min(j0 + INNER_TILE, npad)
                for i in range(i0, i0 + SYM_BLOCK):
                    var jstart = max(j0, (i // W) * W)
                    if jstart >= j1:
                        continue
                    var ii = i - i0
                    var xi = F64V(px[unsafe_offset=i])
                    var yi = F64V(py[unsafe_offset=i])
                    var zi = F64V(pz[unsafe_offset=i])
                    var w0i = F64V(pw[unsafe_offset=i])
                    var ki = F64V(pk[unsafe_offset=i])
                    var ri = F64V(pr[unsafe_offset=i])
                    var sf = pp.unsafe_load[width=W](3 * ii * W)
                    var su = pp.unsafe_load[width=W]((3 * ii + 1) * W)
                    var sw = pp.unsafe_load[width=W]((3 * ii + 2) * W)
                    var j = jstart
                    if j <= i:                                  # the vector holding i: lanes j > i only
                        var a = _pair_terms(j, xi, yi, zi, w0i, ki, px, py, pz, pw, pk)
                        var lane = F64V(0.0)
                        for l in range(W):
                            if j + l > i:
                                lane[l] = 1.0
                        _add_sym(j, a, lane / a[3], ri, pr, sf, su, sw, af, au, aw)
                        j += W
                    while j + 3 * W < j1:                       # four vectors, one division
                        var a = _pair_terms(j, xi, yi, zi, w0i, ki, px, py, pz, pw, pk)
                        var b2 = _pair_terms(j + W, xi, yi, zi, w0i, ki, px, py, pz, pw, pk)
                        var c = _pair_terms(j + 2 * W, xi, yi, zi, w0i, ki, px, py, pz, pw, pk)
                        var e = _pair_terms(j + 3 * W, xi, yi, zi, w0i, ki, px, py, pz, pw, pk)
                        var dab = a[3] * b2[3]
                        var dce = c[3] * e[3]
                        var inv = 1.0 / (dab * dce)
                        var iab = inv * dce
                        var ice = inv * dab
                        _add_sym(j, a, iab * b2[3], ri, pr, sf, su, sw, af, au, aw)
                        _add_sym(j + W, b2, iab * a[3], ri, pr, sf, su, sw, af, au, aw)
                        _add_sym(j + 2 * W, c, ice * e[3], ri, pr, sf, su, sw, af, au, aw)
                        _add_sym(j + 3 * W, e, ice * c[3], ri, pr, sf, su, sw, af, au, aw)
                        j += 4 * W
                    while j < j1:
                        var a = _pair_terms(j, xi, yi, zi, w0i, ki, px, py, pz, pw, pk)
                        _add_sym(j, a, 1.0 / a[3], ri, pr, sf, su, sw, af, au, aw)
                        j += W
                    pp.unsafe_store(3 * ii * W, sf)
                    pp.unsafe_store((3 * ii + 1) * W, su)
                    pp.unsafe_store((3 * ii + 2) * W, sw)
                j0 = j1
            for ii in range(SYM_BLOCK):
                var i = i0 + ii
                # the pair (i, i): R2 = 0, D = 2 K_i^3
                var kk = pk[unsafe_offset=i]
                var dii = 2.0 * kk * kk * kk
                af[unsafe_offset=i] += pp.unsafe_load[width=W](3 * ii * W).reduce_add() + pr[unsafe_offset=i] / dii
                au[unsafe_offset=i] += (pp.unsafe_load[width=W]((3 * ii + 1) * W).reduce_add()
                                        + pr[unsafe_offset=i] * 3.0 * kk * kk / (dii * dii))
                aw[unsafe_offset=i] += pp.unsafe_load[width=W]((3 * ii + 2) * W).reduce_add()
            bi += nwork
        _ = part^

    parallelize(work, nwork)
    for wk in range(1, nwork):
        var src = pa.unsafe_offset(wk * 3 * npad)
        for t in range(3 * npad):
            pa[unsafe_offset=t] += src[unsafe_offset=t]
    for i in range(n):
        f_out[unsafe_offset=i] = -1.5 * pa[unsafe_offset=i]
        u_out[unsafe_offset=i] = pa[unsafe_offset=npad + i]
        w_out[unsafe_offset=i] = pa[unsafe_offset=2 * npad + i]
    _ = soa^
    _ = acc^
