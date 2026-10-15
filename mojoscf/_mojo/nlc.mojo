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
from _mojo.linalg import Blas, F64Ptr, list_ptr

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


# ---------------------------------------------------------------------------------------------------------------
# The pair sums of pyscf's analytical VV10 Hessian (``pyscf.hessian.rks``: ``_get_enlc_deriv2``,
# ``_get_vnlc_deriv1``, ``get_vnlc_resp``), each with the inputs and outputs of the C function it replaces.
# pyscf's loops take every ordered pair one point at a time, with three divisions per pair and, for the
# sums over trial densities and nuclear displacements, the pair terms recomputed for every six trials or
# for every displacement.  Here the symmetric sums take each unordered pair once, the trial sums are GEMMs
# with tiles of pair coefficients, and the grid-response sums run over the points of one atom at a time.

comptime NE = 6     # E, U, W, A, B, C


@always_inline
def _add_uwabce(
    j: Int, t: Tuple[F64V, F64V, F64V, F64V], inv: F64V, ri: F64V, pr: F64Ptr, mut se: F64V, mut su: F64V,
    mut sw: F64V, mut sa: F64V, mut sb: F64V, mut sc: F64V, acc: F64Ptr, npad: Int,
):
    """The terms of both points of the pairs ``t`` (inv = 1 / D, Phi = -3/2 inv): E = RpW Phi,
    U = E (1/(g_ij + g_ji) + 1/g_ij), W = U R2, A = E (1/(g_ij + g_ji)^2 + 1/((g_ij + g_ji) g_ij) + 1/g_ij^2),
    B = A R2, C = A R2^2; point i into the registers, the points j into acc[k npad + j]."""
    var r2 = t[0]
    var gij = t[1]
    var gji = t[2]
    var sg = gij + gji
    var phi = -1.5 * inv
    var gs1 = inv * (gij * gji)
    var gij1 = inv * (gji * sg)
    var gji1 = inv * (gij * sg)
    var gs2 = gs1 * gs1
    var ei = pr.unsafe_load[width=W](j) * phi
    var si = gs1 + gij1
    var ui = ei * si
    var ai = ei * (gs2 + gij1 * si)
    var bi = ai * r2
    se += ei
    su += ui
    sw += ui * r2
    sa += ai
    sb += bi
    sc += bi * r2
    var ej = ri * phi
    var sj = gs1 + gji1
    var uj = ej * sj
    var aj = ej * (gs2 + gji1 * sj)
    var bj = aj * r2
    acc.unsafe_store(j, acc.unsafe_load[width=W](j) + ej)
    acc.unsafe_store(npad + j, acc.unsafe_load[width=W](npad + j) + uj)
    acc.unsafe_store(2 * npad + j, acc.unsafe_load[width=W](2 * npad + j) + uj * r2)
    acc.unsafe_store(3 * npad + j, acc.unsafe_load[width=W](3 * npad + j) + aj)
    acc.unsafe_store(4 * npad + j, acc.unsafe_load[width=W](4 * npad + j) + bj)
    acc.unsafe_store(5 * npad + j, acc.unsafe_load[width=W](5 * npad + j) + bj * r2)


def vv10_uwabce_core(
    n: Int, xyz: F64Ptr, w: F64Ptr, rho: F64Ptr, omega: F64Ptr, kappa: F64Ptr, u_out: F64Ptr, w_out: F64Ptr,
    a_out: F64Ptr, b_out: F64Ptr, c_out: F64Ptr, e_out: F64Ptr,
):
    """pyscf's ``VXC_vv10nlc_hessian_eval_UWABCE``: for every point i the sums over all points j (j = i
    included) of E_ij = w_j rho_j Phi_ij and the U, W, A, B, C terms of ``_add_uwabce``; out U = -sum U,
    W = -sum W, A = 2 sum A, B = 2 sum B, C = 2 sum C, E = sum E.  The pair terms are symmetric up to
    exchanging g_ij and g_ji, so every unordered pair is evaluated once for both points, as in
    ``vv10_sym_core`` (one division per four vectors of pairs; the pair (i, i) added once)."""
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
            pw[unsafe_offset=j] = omega[unsafe_offset=j]
            pk[unsafe_offset=j] = kappa[unsafe_offset=j]
            pr[unsafe_offset=j] = w[unsafe_offset=j] * rho[unsafe_offset=j]
        else:
            pk[unsafe_offset=j] = 1.0                    # padding: RpW = 0, no 0/0
    var nblk = npad // SYM_BLOCK
    var nwork = max(1, min(parallelism_level(), nblk))
    var acc = List[Float64](length=nwork * NE * npad, fill=0.0)
    var pa = list_ptr(acc)

    def work(wk: Int) {imm npad, imm nblk, imm nwork, imm px, imm py, imm pz, imm pw, imm pk, imm pr, imm pa}:
        var acc_w = pa.unsafe_offset(wk * NE * npad)
        var part = List[Float64](length=NE * SYM_BLOCK * W, fill=0.0)
        var pp = list_ptr(part)
        var bi = wk
        while bi < nblk:
            var i0 = bi * SYM_BLOCK
            for t in range(NE * SYM_BLOCK * W):
                pp[unsafe_offset=t] = 0.0
            var j0 = i0
            while j0 < npad:
                var j1 = min(j0 + INNER_TILE, npad)
                for i in range(i0, i0 + SYM_BLOCK):
                    var jstart = max(j0, (i // W) * W)
                    if jstart >= j1:
                        continue
                    var q = pp.unsafe_offset((i - i0) * NE * W)
                    var xi = F64V(px[unsafe_offset=i])
                    var yi = F64V(py[unsafe_offset=i])
                    var zi = F64V(pz[unsafe_offset=i])
                    var w0i = F64V(pw[unsafe_offset=i])
                    var ki = F64V(pk[unsafe_offset=i])
                    var ri = F64V(pr[unsafe_offset=i])
                    var se = q.unsafe_load[width=W](0)
                    var su = q.unsafe_load[width=W](W)
                    var sw = q.unsafe_load[width=W](2 * W)
                    var sa = q.unsafe_load[width=W](3 * W)
                    var sb = q.unsafe_load[width=W](4 * W)
                    var sc = q.unsafe_load[width=W](5 * W)
                    var j = jstart
                    if j <= i:                                  # the vector holding i: lanes j > i only
                        var a = _pair_terms(j, xi, yi, zi, w0i, ki, px, py, pz, pw, pk)
                        var lane = F64V(0.0)
                        for l in range(W):
                            if j + l > i:
                                lane[l] = 1.0
                        _add_uwabce(j, a, lane / a[3], ri, pr, se, su, sw, sa, sb, sc, acc_w, npad)
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
                        _add_uwabce(j, a, iab * b2[3], ri, pr, se, su, sw, sa, sb, sc, acc_w, npad)
                        _add_uwabce(j + W, b2, iab * a[3], ri, pr, se, su, sw, sa, sb, sc, acc_w, npad)
                        _add_uwabce(j + 2 * W, c, ice * e[3], ri, pr, se, su, sw, sa, sb, sc, acc_w, npad)
                        _add_uwabce(j + 3 * W, e, ice * c[3], ri, pr, se, su, sw, sa, sb, sc, acc_w, npad)
                        j += 4 * W
                    while j < j1:
                        var a = _pair_terms(j, xi, yi, zi, w0i, ki, px, py, pz, pw, pk)
                        _add_uwabce(j, a, 1.0 / a[3], ri, pr, se, su, sw, sa, sb, sc, acc_w, npad)
                        j += W
                    q.unsafe_store(0, se)
                    q.unsafe_store(W, su)
                    q.unsafe_store(2 * W, sw)
                    q.unsafe_store(3 * W, sa)
                    q.unsafe_store(4 * W, sb)
                    q.unsafe_store(5 * W, sc)
                j0 = j1
            for ii in range(SYM_BLOCK):
                var i = i0 + ii
                var q = pp.unsafe_offset(ii * NE * W)
                # the pair (i, i): R2 = 0, g = K_i, Phi = -3/4 / K^3, U = E 3/(2 K), A = E 7/(4 K^2)
                var kk = pk[unsafe_offset=i]
                var ei = pr[unsafe_offset=i] * (-0.75 / (kk * kk * kk))
                acc_w[unsafe_offset=i] += q.unsafe_load[width=W](0).reduce_add() + ei
                acc_w[unsafe_offset=npad + i] += q.unsafe_load[width=W](W).reduce_add() + ei * 1.5 / kk
                acc_w[unsafe_offset=2 * npad + i] += q.unsafe_load[width=W](2 * W).reduce_add()
                acc_w[unsafe_offset=3 * npad + i] += q.unsafe_load[width=W](3 * W).reduce_add() + ei * 1.75 / (kk * kk)
                acc_w[unsafe_offset=4 * npad + i] += q.unsafe_load[width=W](4 * W).reduce_add()
                acc_w[unsafe_offset=5 * npad + i] += q.unsafe_load[width=W](5 * W).reduce_add()
            bi += nwork
        _ = part^

    parallelize(work, nwork)
    for wk in range(1, nwork):
        var src = pa.unsafe_offset(wk * NE * npad)
        for t in range(NE * npad):
            pa[unsafe_offset=t] += src[unsafe_offset=t]
    for i in range(n):
        e_out[unsafe_offset=i] = pa[unsafe_offset=i]
        u_out[unsafe_offset=i] = -pa[unsafe_offset=npad + i]
        w_out[unsafe_offset=i] = -pa[unsafe_offset=2 * npad + i]
        a_out[unsafe_offset=i] = 2.0 * pa[unsafe_offset=3 * npad + i]
        b_out[unsafe_offset=i] = 2.0 * pa[unsafe_offset=4 * npad + i]
        c_out[unsafe_offset=i] = 2.0 * pa[unsafe_offset=5 * npad + i]
    _ = soa^
    _ = acc^


comptime HT = 128   # points per tile of the GEMM-based sums


struct _Points(Movable):
    """Per-point arrays of the Hessian sums, padded to whole tiles of HT (structure of arrays): x, y, z,
    omega, kappa and up to three more per-point factors; padding points have kappa = 1 (no 0/0)."""

    var buf: List[Float64]
    var npad: Int
    var ntile: Int

    def __init__(out self, n: Int, nextra: Int):
        self.ntile = (n + HT - 1) // HT
        self.npad = self.ntile * HT
        self.buf = List[Float64](length=(5 + nextra) * max(self.npad, HT), fill=0.0)
        var p = list_ptr(self.buf)
        for j in range(n, self.npad):
            p[unsafe_offset=4 * self.npad + j] = 1.0

    def ptr(self, k: Int) -> F64Ptr:
        return list_ptr(self.buf).unsafe_offset(k * self.npad)

    def fill(self, k: Int, src: F64Ptr, n: Int, stride: Int, offset: Int):
        var d = self.ptr(k)
        for j in range(n):
            d[unsafe_offset=j] = src[unsafe_offset=j * stride + offset]


@always_inline
def _tile_inverses(
    j: Int, xi: F64V, yi: F64V, zi: F64V, w0i: F64V, ki: F64V, px: F64Ptr, py: F64Ptr, pz: F64Ptr, pw: F64Ptr,
    pk: F64Ptr,
) -> Tuple[Tuple[F64V, F64V, F64V, F64V], Tuple[F64V, F64V, F64V, F64V], Tuple[F64V, F64V, F64V, F64V],
           Tuple[F64V, F64V, F64V, F64V], F64V, F64V, F64V, F64V]:
    """``_pair_terms`` of four vectors of points from j and the reciprocals of their denominators (one division)."""
    var a = _pair_terms(j, xi, yi, zi, w0i, ki, px, py, pz, pw, pk)
    var b = _pair_terms(j + W, xi, yi, zi, w0i, ki, px, py, pz, pw, pk)
    var c = _pair_terms(j + 2 * W, xi, yi, zi, w0i, ki, px, py, pz, pw, pk)
    var e = _pair_terms(j + 3 * W, xi, yi, zi, w0i, ki, px, py, pz, pw, pk)
    var dab = a[3] * b[3]
    var dce = c[3] * e[3]
    var inv = 1.0 / (dab * dce)
    var iab = inv * dce
    var ice = inv * dab
    return (a, b, c, e, iab * b[3], iab * a[3], ice * e[3], ice * c[3])


@always_inline
def _fxc_coef(
    jj: Int, j: Int, t: Tuple[F64V, F64V, F64V, F64V], inv: F64V, p1i: F64V, p0i: F64V, cgi: F64V, pp1: F64Ptr,
    pp0: F64Ptr, pcg: F64Ptr, row_r: F64Ptr, row_g: F64Ptr,
):
    """The 2 x 2 coefficients of ``vv10_fxc_core`` for the pairs ``t``, stored at column jj of the rho row
    [frr | frg] and the gamma row [fgr | fgg] of the tile (HT columns per block)."""
    var r2 = t[0]
    var gij = t[1]
    var gji = t[2]
    var sg = gij + gji
    var phi = -1.5 * inv
    var gs1 = inv * (gij * gji)
    var gij1 = inv * (gji * sg)
    var gji1 = inv * (gij * sg)
    var ai = r2 * p1i + p0i
    var aj = r2 * pp1.unsafe_load[width=W](j) + pp0.unsafe_load[width=W](j)
    var d2 = 2.0 * (gs1 * gs1 + gij1 * gji1)
    var si = gs1 + gij1
    var sj = gs1 + gji1
    var cri = cgi * r2
    var crj = pcg.unsafe_load[width=W](j) * r2
    row_r.unsafe_store(jj, phi * (ai * aj * d2 - ai * si - aj * sj + 1.0))
    row_r.unsafe_store(HT + jj, crj * phi * (ai * d2 - sj))
    row_g.unsafe_store(jj, cri * phi * (aj * d2 - si))
    row_g.unsafe_store(HT + jj, cri * crj * phi * d2)


def vv10_fxc_core(
    blas_seq: Blas, n: Int, xyz: F64Ptr, omega: F64Ptr, kappa: F64Ptr, p1: F64Ptr, p0: F64Ptr, cg: F64Ptr,
    nt: Int, y: F64Ptr, o: F64Ptr,
) raises:
    """The pair part of pyscf's ``VXC_vv10nlc_hessian_eval_f_t`` for ``nt`` trial densities, as GEMMs.

    For all ordered pairs of points (i, j), j = i included as in pyscf, the coefficients
        frr = Phi (a_i a_j D2 - a_i S_i - a_j S_j + 1),   frg = c_j R2 Phi (a_i D2 - S_j),
        fgr = c_i R2 Phi (a_j D2 - S_i),                   fgg = c_i c_j R2^2 Phi D2,
    with Phi = -3/2 / (g_ij g_ji (g_ij + g_ji)), S_i = 1/(g_ij + g_ji) + 1/g_ij, D2 = 2 (1/(g_ij + g_ji)^2 +
    1/(g_ij g_ji)), a = R2 p1 + p0 (p1 = rho domega/drho, p0 = rho dkappa/drho) and c = rho domega/dgamma, are
    formed for tiles of HT x HT points (one division per four vectors) and multiplied with the trial tiles:
    o[I] (2 HT x nt) = sum_J C_IJ (2 HT x 2 HT) y[J] (2 HT x nt), y and o (ntile, 2, HT, nt) with the rho
    rows before the gamma rows (y holding w_j rho_t and w_j gamma_t).  Workers take tiles of rows I."""
    var pts = _Points(n, 3)
    pts.fill(0, xyz, n, 3, 0)
    pts.fill(1, xyz, n, 3, 1)
    pts.fill(2, xyz, n, 3, 2)
    pts.fill(3, omega, n, 1, 0)
    pts.fill(4, kappa, n, 1, 0)
    pts.fill(5, p1, n, 1, 0)
    pts.fill(6, p0, n, 1, 0)
    pts.fill(7, cg, n, 1, 0)
    var ntile = pts.ntile
    var px = pts.ptr(0)
    var py = pts.ptr(1)
    var pz = pts.ptr(2)
    var pw = pts.ptr(3)
    var pk = pts.ptr(4)
    var pp1 = pts.ptr(5)
    var pp0 = pts.ptr(6)
    var pcg = pts.ptr(7)
    var nwork = max(1, min(parallelism_level(), ntile))
    var tsize = 2 * HT * nt

    def work(wk: Int) {imm blas_seq, imm ntile, imm nwork, imm nt, imm tsize, imm px, imm py, imm pz, imm pw, imm pk, imm pp1, imm pp0, imm pcg, imm y, imm o}:
        var coef = List[Float64](length=4 * HT * HT, fill=0.0)
        var pc = list_ptr(coef)
        var it = wk
        while it < ntile:
            for jt in range(ntile):
                for ii in range(HT):
                    var i = it * HT + ii
                    var xi = F64V(px[unsafe_offset=i])
                    var yi = F64V(py[unsafe_offset=i])
                    var zi = F64V(pz[unsafe_offset=i])
                    var w0i = F64V(pw[unsafe_offset=i])
                    var ki = F64V(pk[unsafe_offset=i])
                    var p1i = F64V(pp1[unsafe_offset=i])
                    var p0i = F64V(pp0[unsafe_offset=i])
                    var cgi = F64V(pcg[unsafe_offset=i])
                    var row_r = pc.unsafe_offset(ii * 2 * HT)
                    var row_g = pc.unsafe_offset((HT + ii) * 2 * HT)
                    var jj = 0
                    while jj < HT:
                        var j = jt * HT + jj
                        var q = _tile_inverses(j, xi, yi, zi, w0i, ki, px, py, pz, pw, pk)
                        _fxc_coef(jj, j, q[0], q[4], p1i, p0i, cgi, pp1, pp0, pcg, row_r, row_g)
                        _fxc_coef(jj + W, j + W, q[1], q[5], p1i, p0i, cgi, pp1, pp0, pcg, row_r, row_g)
                        _fxc_coef(jj + 2 * W, j + 2 * W, q[2], q[6], p1i, p0i, cgi, pp1, pp0, pcg, row_r, row_g)
                        _fxc_coef(jj + 3 * W, j + 3 * W, q[3], q[7], p1i, p0i, cgi, pp1, pp0, pcg, row_r, row_g)
                        jj += 4 * W
                try:
                    blas_seq.gemm(False, False, 2 * HT, nt, 2 * HT, 1.0, pc, y.unsafe_offset(jt * tsize),
                                  0.0 if jt == 0 else 1.0, o.unsafe_offset(it * tsize))
                except:
                    pass
            it += nwork
        _ = coef^

    var nthr = blas_seq.serial_begin()
    parallelize(work, nwork)
    blas_seq.serial_end(nthr)
    _ = pts^


@always_inline
def _w1_coef(jj: Int, t: Tuple[F64V, F64V, F64V, F64V], inv: F64V, row_e: F64Ptr, row_u: F64Ptr, row_w: F64Ptr):
    """Phi, -Phi S_i and -Phi S_i R2 of the pairs ``t`` at column jj of three rows of the tile."""
    var gij = t[1]
    var gji = t[2]
    var sg = gij + gji
    var phi = -1.5 * inv
    var s = inv * (gij * gji) + inv * (gji * sg)
    var u = -phi * s
    row_e.unsafe_store(jj, phi)
    row_u.unsafe_store(jj, u)
    row_w.unsafe_store(jj, u * t[0])


def vv10_euw_w1_core(
    blas_seq: Blas, n: Int, xyz: F64Ptr, omega: F64Ptr, kappa: F64Ptr, nd: Int, y: F64Ptr, o: F64Ptr,
) raises:
    """pyscf's ``VXC_vv10nlc_hessian_eval_EUW_with_weight1`` as GEMMs: for each weight derivative d
    Ew[d, i] = sum_j y_dj Phi_ij, Uw = -sum_j y_dj Phi_ij S_ij, Ww = -sum_j y_dj Phi_ij S_ij R2 with
    y_dj = rho_j dw_j/dR_d.  Tiles of the three pair coefficients (3 HT x HT) times the tiles of y
    (ntile, HT, nd) give o (ntile, 3, HT, nd)."""
    var pts = _Points(n, 0)
    pts.fill(0, xyz, n, 3, 0)
    pts.fill(1, xyz, n, 3, 1)
    pts.fill(2, xyz, n, 3, 2)
    pts.fill(3, omega, n, 1, 0)
    pts.fill(4, kappa, n, 1, 0)
    var ntile = pts.ntile
    var px = pts.ptr(0)
    var py = pts.ptr(1)
    var pz = pts.ptr(2)
    var pw = pts.ptr(3)
    var pk = pts.ptr(4)
    var nwork = max(1, min(parallelism_level(), ntile))

    def work(wk: Int) {imm blas_seq, imm ntile, imm nwork, imm nd, imm px, imm py, imm pz, imm pw, imm pk, imm y, imm o}:
        var coef = List[Float64](length=3 * HT * HT, fill=0.0)
        var pc = list_ptr(coef)
        var it = wk
        while it < ntile:
            for jt in range(ntile):
                for ii in range(HT):
                    var i = it * HT + ii
                    var xi = F64V(px[unsafe_offset=i])
                    var yi = F64V(py[unsafe_offset=i])
                    var zi = F64V(pz[unsafe_offset=i])
                    var w0i = F64V(pw[unsafe_offset=i])
                    var ki = F64V(pk[unsafe_offset=i])
                    var row_e = pc.unsafe_offset(ii * HT)
                    var row_u = pc.unsafe_offset((HT + ii) * HT)
                    var row_w = pc.unsafe_offset((2 * HT + ii) * HT)
                    var jj = 0
                    while jj < HT:
                        var q = _tile_inverses(jt * HT + jj, xi, yi, zi, w0i, ki, px, py, pz, pw, pk)
                        _w1_coef(jj, q[0], q[4], row_e, row_u, row_w)
                        _w1_coef(jj + W, q[1], q[5], row_e, row_u, row_w)
                        _w1_coef(jj + 2 * W, q[2], q[6], row_e, row_u, row_w)
                        _w1_coef(jj + 3 * W, q[3], q[7], row_e, row_u, row_w)
                        jj += 4 * W
                try:
                    blas_seq.gemm(False, False, 3 * HT, nd, HT, 1.0, pc, y.unsafe_offset(jt * HT * nd),
                                  0.0 if jt == 0 else 1.0, o.unsafe_offset(it * 3 * HT * nd))
                except:
                    pass
            it += nwork
        _ = coef^

    var nthr = blas_seq.serial_begin()
    parallelize(work, nwork)
    blas_seq.serial_end(nthr)
    _ = pts^


def vv10_euw_gr_core(
    n: Int, xyz: F64Ptr, w: F64Ptr, rho: F64Ptr, omega: F64Ptr, kappa: F64Ptr, natm: Int, seg: Pointer[Int64, MutAnyOrigin],
    dst: F64Ptr,
):
    """pyscf's ``VXC_vv10nlc_hessian_eval_EUW_grid_response`` for points sorted by their atom: atom B owns the
    points seg[B] <= j < seg[B + 1] (each run padded to whole vectors with zero-density points).  For a point
    i of atom A and every other atom B the sums over the points j of B of E_ij (dPhi/dr_j / Phi) r_ji,
    E_ij (dPhi/dr_j dPhi/dg_ij / Phi^2 + d2Phi/dg_ij dr_j / Phi) r_ji and the same times R2 minus
    E_ij dPhi/dg_ij / Phi r_ji (pyscf's Egr, Ugr, Wgr terms) go to dst[(k, B, x), i] (k = E, U, W, scaled by
    -2, 2, 2), and minus their total over B to dst[(k, A, x), i].  dst is (3, natm, 3, n), zero on entry."""
    if n == 0:
        return
    var npad = Int(seg[unsafe_offset=natm])
    var soa = List[Float64](length=6 * npad, fill=0.0)
    var ps = list_ptr(soa)
    var px = ps
    var py = ps.unsafe_offset(npad)
    var pz = ps.unsafe_offset(2 * npad)
    var pw = ps.unsafe_offset(3 * npad)
    var pk = ps.unsafe_offset(4 * npad)
    var pr = ps.unsafe_offset(5 * npad)
    for j in range(npad):
        px[unsafe_offset=j] = xyz[unsafe_offset=3 * j]
        py[unsafe_offset=j] = xyz[unsafe_offset=3 * j + 1]
        pz[unsafe_offset=j] = xyz[unsafe_offset=3 * j + 2]
        pw[unsafe_offset=j] = omega[unsafe_offset=j]
        pk[unsafe_offset=j] = kappa[unsafe_offset=j]
        pr[unsafe_offset=j] = w[unsafe_offset=j] * rho[unsafe_offset=j]
    var nblk = (npad + OUTER_BLOCK - 1) // OUTER_BLOCK

    def block(bk: Int) {imm npad, imm natm, imm seg, imm px, imm py, imm pz, imm pw, imm pk, imm pr, imm dst}:
        var i0 = bk * OUTER_BLOCK
        var i1 = min(i0 + OUTER_BLOCK, npad)
        var tot = List[Float64](length=9, fill=0.0)
        var pt = list_ptr(tot)
        var ai = 0
        for i in range(i0, i1):                             # padding points too: their rows are discarded
            while Int(seg[unsafe_offset=ai + 1]) <= i:
                ai += 1
            var xi = F64V(px[unsafe_offset=i])
            var yi = F64V(py[unsafe_offset=i])
            var zi = F64V(pz[unsafe_offset=i])
            var w0i = F64V(pw[unsafe_offset=i])
            var ki = F64V(pk[unsafe_offset=i])
            for t in range(9):
                pt[unsafe_offset=t] = 0.0
            for b in range(natm):
                if b == ai:
                    continue
                var ex = F64V(0.0)
                var ey = F64V(0.0)
                var ez = F64V(0.0)
                var ux = F64V(0.0)
                var uy = F64V(0.0)
                var uz = F64V(0.0)
                var wx = F64V(0.0)
                var wy = F64V(0.0)
                var wz = F64V(0.0)
                var j = Int(seg[unsafe_offset=b])
                var jend = Int(seg[unsafe_offset=b + 1])
                while j < jend:
                    var rx = px.unsafe_load[width=W](j) - xi
                    var ry = py.unsafe_load[width=W](j) - yi
                    var rz = pz.unsafe_load[width=W](j) - zi
                    var r2 = rx * rx + ry * ry + rz * rz
                    var wj = pw.unsafe_load[width=W](j)
                    var gij = r2 * w0i + ki
                    var gji = r2 * wj + pk.unsafe_load[width=W](j)
                    var sg = gij + gji
                    var inv = 1.0 / (gij * gji * sg)
                    var gij1 = inv * (gji * sg)
                    var gji1 = inv * (gij * sg)
                    var gs1 = inv * (gij * gji)
                    var e = pr.unsafe_load[width=W](j) * (-1.5 * inv)
                    var ws = w0i + wj
                    var dr = w0i * gij1 + wj * gji1 + ws * gs1         # dPhi/dr_j / Phi
                    var d2 = w0i * gij1 * gij1 + ws * gs1 * gs1         # d2Phi/dg_ij dr_j / Phi
                    var dg = gs1 + gij1                                 # dPhi/dg_ij / Phi
                    var te = e * dr
                    var tu = e * (dr * dg + d2)
                    var tw = r2 * tu - e * dg
                    ex += te * rx
                    ey += te * ry
                    ez += te * rz
                    ux += tu * rx
                    uy += tu * ry
                    uz += tu * rz
                    wx += tw * rx
                    wy += tw * ry
                    wz += tw * rz
                    j += W
                var v = SIMD[DType.float64, 16](0.0)
                v[0] = ex.reduce_add()
                v[1] = ey.reduce_add()
                v[2] = ez.reduce_add()
                v[3] = ux.reduce_add()
                v[4] = uy.reduce_add()
                v[5] = uz.reduce_add()
                v[6] = wx.reduce_add()
                v[7] = wy.reduce_add()
                v[8] = wz.reduce_add()
                for t in range(9):
                    pt[unsafe_offset=t] += v[t]
                    var scale = -2.0 if t < 3 else 2.0
                    dst[unsafe_offset=((t // 3 * natm + b) * 3 + t % 3) * npad + i] = scale * v[t]
            for t in range(9):
                var scale = -2.0 if t < 3 else 2.0
                dst[unsafe_offset=((t // 3 * natm + ai) * 3 + t % 3) * npad + i] = -scale * pt[unsafe_offset=t]
        _ = tot^

    parallelize(block, nblk)
    _ = soa^
