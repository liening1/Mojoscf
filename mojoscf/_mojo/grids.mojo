"""Becke partition of pyscf's molecular grids and its nuclear derivatives (the weight response).

    becke_response_core   for the grid of one atom: the Becke weights
                          w0 = vol P_A / sum_B P_B and, contracted with an
                          energy density e(r), the derivative sum_r e(r) dw(r)/dR
                          of the weights with respect to every nucleus; the
                          weights alone are pyscf's ``gen_grid.get_partition``
                          (``mojoscf.dft.get_partition``)

The weights alone of molecules from SCREEN_MIN atoms go point by point,
SIMD over the atoms (``_becke_weights_screened``): a row of steps gives a
cell function P_C in full and, from the same polynomial, one factor of every
other P_X, so that beta_X, the product of the factors met so far, bounds
P_X.  Rows are taken for the owner and then for the atom of the largest
bound, until every bound is below SCREEN_EPS times the largest cell
function: about ten rows per point instead of the natm^2 / 2 pairs.

pyscf's ``grad.rks.grids_response_becke`` forms, for every point of an
atom's grid, the cell functions P_B of all atoms (products of the smoothed
step function s(mu_BC) over the other atoms C) and their derivatives with
respect to all nuclei, an (natm, natm, 3) array per point, as NumPy arrays
over the grid.  Here every point (SIMD over W points) needs only the
contractions T_x = P_A L_x,A and S_x = sum_B P_B L_x,B of the logarithmic
derivatives L_x,B = d ln P_B / dR_x, accumulated pair by pair in a second
sweep over the atom pairs once the cell functions are known; then
dw/dR_x = vol (z T_x - P_A z^2 S_x) with z = 1 / sum_B P_B.  The terms are
pyscf's, including those of the radii adjustment and the grid points moving
with their atom.  The step function and its derivative of every pair are
kept from the first sweep: near |mu| = 1 the Becke step rounds to 1 within
1e-200 (pyscf's guard), and its large logarithmic derivative t / s only
cancels against the s inside P_A when both sweeps see the same rounding.
Points are tasks for the worker threads in chunks.
"""
from std.math import sqrt
from std.runtime import parallelism_level
from max.algorithm import parallelize
from _mojo.linalg import F64Ptr, list_ptr
from _mojo.integrals import W

comptime F64V = SIMD[DType.float64, W]
comptime CHUNK = 8              # SIMD vectors of points per task
comptime SCREEN_MIN = 40        # atoms from which the weights alone take the screened sweep (mode 0)
comptime SCREEN_EPS = 1e-20     # cell functions bounded below this times the largest are left out of the sum


@always_inline
def _smooth(g: F64V, scheme: Int) -> Tuple[F64V, F64V]:
    """pyscf's smoothed step function of the Becke partition and its derivative: the original Becke
    scheme (three iterations of p(x) = (3 - x^2) x / 2, ``scheme`` 0) or Stratmann's (``scheme`` 1)."""
    if scheme == 0:
        var p0 = g
        var p1 = (3.0 - p0 * p0) * p0 * 0.5
        var p2 = (3.0 - p1 * p1) * p1 * 0.5
        var p3 = (3.0 - p2 * p2) * p2 * 0.5
        var t = 27.0 / 8.0 * (1.0 - p2 * p2) * (1.0 - p1 * p1) * (1.0 - p0 * p0)
        return (p3, t)
    var a = 0.64
    var ma = g / a
    var ma2 = ma * ma
    var g1 = (1.0 / 16.0) * (ma * (35.0 + ma2 * (-35.0 + ma2 * (21.0 - 5.0 * ma2))))
    var t = (35.0 / 16.0 / a) * (1.0 - ma2 * (3.0 - ma2 * (3.0 - ma2)))
    var lo = g.le(F64V(-a))
    var hi = g.ge(F64V(a))
    g1 = lo.select(F64V(-1.0), g1)
    g1 = hi.select(F64V(1.0), g1)
    t = lo.select(F64V(0.0), t)
    t = hi.select(F64V(0.0), t)
    return (g1, t)


@always_inline
def _half_smooth0[scheme: Int](g: F64V) -> F64V:
    """Half of ``_smooth0`` for a scheme known at compile time: the halving taken into the last step of
    Becke's polynomial, with the same rounding."""
    comptime if scheme == 0:
        var p1 = (1.5 - 0.5 * (g * g)) * g
        var p2 = (1.5 - 0.5 * (p1 * p1)) * p1
        return (0.75 - 0.25 * (p2 * p2)) * p2
    return 0.5 * _smooth0(g, 1)


@always_inline
def _smooth0(g: F64V, scheme: Int) -> F64V:
    """The value of ``_smooth`` alone (the weights without their derivatives), each step as
    (1.5 - 0.5 x^2) x: the same rounding as (3 - x^2) x / 2 in one operation less."""
    if scheme == 0:
        var p1 = (1.5 - 0.5 * (g * g)) * g
        var p2 = (1.5 - 0.5 * (p1 * p1)) * p1
        return (1.5 - 0.5 * (p2 * p2)) * p2
    var a = 0.64
    var ma = g / a
    var ma2 = ma * ma
    var g1 = (1.0 / 16.0) * (ma * (35.0 + ma2 * (-35.0 + ma2 * (21.0 - 5.0 * ma2))))
    g1 = g.le(F64V(-a)).select(F64V(-1.0), g1)
    return g.ge(F64V(a)).select(F64V(1.0), g1)


def _becke_weights_screened(
    natm: Int, atm: F64Ptr, adj: F64Ptr, use_adj: Int, scheme: Int, npts: Int, coords: F64Ptr, vol: F64Ptr,
    owner: Int, w0_out: F64Ptr,
):
    """The Becke weights of ``becke_response_core`` alone, point by point and SIMD over the atoms X.  The row
    of atom C, the steps s(mu_CX) over all X, gives P_C and, from the same polynomials, the factors s(mu_XC)
    of the other cell functions, multiplied into their bounds beta_X >= P_X.  The owner's row comes first,
    then the row of the atom of the largest bound, until all bounds are below SCREEN_EPS times the largest
    P_C: the atoms left out add at most natm SCREEN_EPS of sum_B P_B, so the weights are pyscf's to rounding
    (C40H82: 11 rows per point, where the full sweep has the work of 61)."""
    if scheme == 0 and use_adj == 1:
        _screened[0, 1](natm, atm, adj, npts, coords, vol, owner, w0_out)
    elif scheme == 0:
        _screened[0, 0](natm, atm, adj, npts, coords, vol, owner, w0_out)
    elif use_adj == 1:
        _screened[1, 1](natm, atm, adj, npts, coords, vol, owner, w0_out)
    else:
        _screened[1, 0](natm, atm, adj, npts, coords, vol, owner, w0_out)


def _screened[scheme: Int, use_adj: Int](
    natm: Int, atm: F64Ptr, adj: F64Ptr, npts: Int, coords: F64Ptr, vol: F64Ptr, owner: Int, w0_out: F64Ptr,
):
    var na = ((natm + W - 1) // W) * W
    var nv = na // W
    # rows of 1/R_CX and a_CX, zero on the diagonal and beyond natm: there the step is 1/2, so every cell
    # function carries the factor 2^-(1 + na - natm), taken out by ``scale``
    var ml = List[Float64](length=(2 * natm + 4) * na, fill=0.0)
    var invr = list_ptr(ml)
    var adjr = invr.unsafe_offset(natm * na)
    var ax = invr.unsafe_offset(2 * natm * na)
    var ay = ax.unsafe_offset(na)
    var az = ay.unsafe_offset(na)
    var valid = az.unsafe_offset(na)        # 1 for the atoms, 0 for the padding: the initial bounds
    for a in range(natm):
        ax[unsafe_offset=a] = atm[unsafe_offset=3 * a]
        ay[unsafe_offset=a] = atm[unsafe_offset=3 * a + 1]
        az[unsafe_offset=a] = atm[unsafe_offset=3 * a + 2]
        valid[unsafe_offset=a] = 1.0
        for b in range(natm):
            if b == a:
                continue
            var dx = atm[unsafe_offset=3 * a] - atm[unsafe_offset=3 * b]
            var dy = atm[unsafe_offset=3 * a + 1] - atm[unsafe_offset=3 * b + 1]
            var dz = atm[unsafe_offset=3 * a + 2] - atm[unsafe_offset=3 * b + 2]
            invr[unsafe_offset=a * na + b] = 1.0 / sqrt(dx * dx + dy * dy + dz * dz)
            comptime if use_adj == 1:
                adjr[unsafe_offset=a * na + b] = adj[unsafe_offset=a * natm + b]
    var scale = 2.0
    for _ in range(na - natm):
        scale *= 2.0
    var nchunk = (npts + CHUNK * W - 1) // (CHUNK * W)
    var nworkers = max(1, min(parallelism_level(), nchunk))

    def work(w: Int) {imm natm, imm npts, imm coords, imm vol, imm owner, imm w0_out, imm na, imm nv, imm invr, imm adjr, imm ax, imm ay, imm az, imm valid, imm scale, imm nchunk, imm nworkers}:
        # the captured pointers as locals: the stores to the bounds below cannot then alias them
        var pinv = invr
        var padj = adjr
        var pax = ax
        var pay = ay
        var paz = az
        var pval = valid
        var nvl = nv
        var nal = na
        var sl = List[Float64](length=2 * na + W, fill=0.0)
        var dist = list_ptr(sl)
        var beta = dist.unsafe_offset(na)
        var it = 0
        while True:
            var ch = w + it * nworkers
            it += 1
            if ch >= nchunk:
                break
            for p in range(ch * CHUNK * W, min((ch + 1) * CHUNK * W, npts)):
                var x = coords[unsafe_offset=3 * p]
                var y = coords[unsafe_offset=3 * p + 1]
                var z = coords[unsafe_offset=3 * p + 2]
                for v in range(nvl):
                    var vx = pax.unsafe_load[width=W](v * W) - x
                    var vy = pay.unsafe_load[width=W](v * W) - y
                    var vz = paz.unsafe_load[width=W](v * W) - z
                    dist.unsafe_store(v * W, sqrt(vx * vx + vy * vy + vz * vz) + 1e-200)
                    beta.unsafe_store(v * W, pval.unsafe_load[width=W](v * W))
                var c = owner
                var pown = 0.0
                var psum = 0.0
                var pmax = 0.0
                while True:
                    beta[unsafe_offset=c] = 0.0
                    var dc = F64V(dist[unsafe_offset=c])
                    var row = c * nal
                    var acc = F64V(1.0)
                    var bmax = F64V(0.0)
                    for v in range(nvl):
                        var mu = (dc - dist.unsafe_load[width=W](v * W)) * pinv.unsafe_load[width=W](row + v * W)
                        var g = mu
                        comptime if use_adj == 1:
                            g = mu + padj.unsafe_load[width=W](row + v * W) * (1.0 - mu * mu)
                        # the steps 0.5 (1 -+ s), without pyscf's guard 1e-200 against exact zeros: those of the
                        # nearest atom are positive (mu <= 0, |a| <= 1/2), so the sum of the cell functions is
                        var hv = _half_smooth0[scheme](g)
                        acc = acc * (0.5 - hv)
                        var bv = beta.unsafe_load[width=W](v * W) * (0.5 + hv)
                        beta.unsafe_store(v * W, bv)
                        bmax = max(bmax, bv)
                    var pc = acc.reduce_mul() * scale
                    psum += pc
                    if c == owner:
                        pown = pc
                    pmax = max(pmax, pc)
                    var m = bmax.reduce_max()
                    if m <= SCREEN_EPS * pmax or m <= 0.0:
                        break
                    # the next row: the atom of the largest bound
                    for v in range(nvl):
                        var bv = beta.unsafe_load[width=W](v * W)
                        if bv.eq(F64V(m)).reduce_or():
                            for l in range(W):
                                if bv[l] == m:
                                    c = v * W + l
                                    break
                            break
                w0_out[unsafe_offset=p] = vol[unsafe_offset=p] * pown / psum
        _ = sl^

    if nworkers == 1:
        work(0)
    else:
        parallelize(work, nworkers)
    _ = ml^


def becke_response_core(
    natm: Int, atm: F64Ptr, adj: F64Ptr, use_adj: Int, scheme: Int, npts: Int, coords: F64Ptr, vol: F64Ptr,
    owner: Int, eot: F64Ptr, mode: Int, w0_out: F64Ptr, de_out: F64Ptr,
):
    """The grid of atom ``owner`` (``npts`` points ``coords``, quadrature weights ``vol``): the Becke weights
    into ``w0_out`` and, for ``mode`` 1, de_out[x][c] = sum_r eot(r) d w(r) / d R_x,c (natm x 3, overwritten),
    as pyscf's ``grids_response_becke`` (w0, and its w1 contracted with eot).  The weights alone (``mode`` 0)
    come from the sweep over all pairs below SCREEN_MIN atoms and from the screened sweep from there; ``mode``
    2 and 3 force the one or the other.  ``adj`` (natm x natm) is the radii adjustment a_AB of pyscf's
    ``_radii_adjust`` when ``use_adj`` is 1."""
    if mode == 3 or (mode == 0 and natm >= SCREEN_MIN):
        _becke_weights_screened(natm, atm, adj, use_adj, scheme, npts, coords, vol, owner, w0_out)
        return
    var npair = natm * (natm - 1) // 2
    # pair data for a > b: 1/R_ab and (R_a - R_b) / R_ab^3
    var pl = List[Float64](length=max(npair, 1) * 4, fill=0.0)
    var pd = list_ptr(pl)
    var k = 0
    for a in range(natm):
        for b in range(a):
            var dx = atm[unsafe_offset=3 * a] - atm[unsafe_offset=3 * b]
            var dy = atm[unsafe_offset=3 * a + 1] - atm[unsafe_offset=3 * b + 1]
            var dz = atm[unsafe_offset=3 * a + 2] - atm[unsafe_offset=3 * b + 2]
            var r = sqrt(dx * dx + dy * dy + dz * dz)
            var r3 = r * r * r
            pd[unsafe_offset=4 * k] = 1.0 / r
            pd[unsafe_offset=4 * k + 1] = dx / r3
            pd[unsafe_offset=4 * k + 2] = dy / r3
            pd[unsafe_offset=4 * k + 3] = dz / r3
            k += 1
    var nvec = (npts + W - 1) // W
    var nchunk = (nvec + CHUNK - 1) // CHUNK
    var nworkers = max(1, min(parallelism_level(), nchunk))
    var accl = List[Float64](length=nworkers * natm * 3 + 1, fill=0.0)
    var pacc = list_ptr(accl)

    def work(w: Int) {imm natm, imm npair, imm atm, imm adj, imm use_adj, imm scheme, imm npts, imm coords, imm vol, imm owner, imm eot, imm mode, imm w0_out, imm pd, imm nvec, imm nchunk, imm nworkers, imm pacc}:
        var acc = pacc.unsafe_offset(w * natm * 3)
        # per atom, W lanes each: distance, unit vector (3), cell function, S (3), T (3)
        var sl = List[Float64](length=11 * natm * W + W, fill=0.0)
        var base = list_ptr(sl)
        # per pair (mode 1): s_ab, s_ba and the derivative of the step, W lanes each
        var ql = List[Float64](length=(3 * max(npair, 1) if mode == 1 else 1) * W + W, fill=0.0)
        var pairs = list_ptr(ql)
        var dist = base
        var unit = base.unsafe_offset(natm * W)
        var cell = base.unsafe_offset(4 * natm * W)
        var sacc = base.unsafe_offset(5 * natm * W)
        var tacc = base.unsafe_offset(8 * natm * W)
        var it = 0
        while True:
            var ch = w + it * nworkers
            it += 1
            if ch >= nchunk:
                break
            for vi in range(ch * CHUNK, min((ch + 1) * CHUNK, nvec)):
                var p0 = vi * W
                var nl = min(W, npts - p0)
                var x = F64V(0.0)
                var y = F64V(0.0)
                var z = F64V(0.0)
                var vw = F64V(0.0)
                var ew = F64V(0.0)
                for l in range(W):
                    var p = p0 + min(l, nl - 1)      # padding lanes repeat the last point, with zero weight
                    x[l] = coords[unsafe_offset=3 * p]
                    y[l] = coords[unsafe_offset=3 * p + 1]
                    z[l] = coords[unsafe_offset=3 * p + 2]
                    if l < nl:
                        vw[l] = vol[unsafe_offset=p]
                        if mode == 1:
                            ew[l] = eot[unsafe_offset=p]
                for a in range(natm):
                    var vx = atm[unsafe_offset=3 * a] - x
                    var vy = atm[unsafe_offset=3 * a + 1] - y
                    var vz = atm[unsafe_offset=3 * a + 2] - z
                    var d = sqrt(vx * vx + vy * vy + vz * vz) + 1e-200
                    dist.unsafe_store(a * W, d)
                    if mode == 1:
                        unit.unsafe_store((3 * a) * W, vx / d)
                        unit.unsafe_store((3 * a + 1) * W, vy / d)
                        unit.unsafe_store((3 * a + 2) * W, vz / d)
                    cell.unsafe_store(a * W, F64V(1.0))
                var kp = 0
                if mode != 1:
                    # the weights alone (mode 0 or 2): the cell function of a stays in a register over its pairs
                    for a in range(natm):
                        var da = dist.unsafe_load[width=W](a * W)
                        var ca = cell.unsafe_load[width=W](a * W)
                        for b in range(a):
                            var g = (da - dist.unsafe_load[width=W](b * W)) * pd[unsafe_offset=4 * kp]
                            if use_adj == 1:
                                g = g + adj[unsafe_offset=a * natm + b] * (1.0 - g * g)
                            var sv = _smooth0(g, scheme)
                            ca = ca * (0.5 * (1.0 - sv + 1e-200))
                            cell.unsafe_store(b * W, cell.unsafe_load[width=W](b * W) * (0.5 * (1.0 + sv + 1e-200)))
                            kp += 1
                        cell.unsafe_store(a * W, ca)
                for a in range(natm if mode == 1 else 0):
                    for b in range(a):
                        var da = dist.unsafe_load[width=W](a * W)
                        var db = dist.unsafe_load[width=W](b * W)
                        var g = (da - db) * pd[unsafe_offset=4 * kp]
                        var gp = g
                        var gadj = F64V(1.0)
                        if use_adj == 1:
                            var aab = adj[unsafe_offset=a * natm + b]
                            gp = g + aab * (1.0 - g * g)
                            gadj = 1.0 - 2.0 * aab * g
                        var st = _smooth(gp, scheme)
                        var s_ab = 0.5 * (1.0 - st[0] + 1e-200)
                        var s_ba = 0.5 * (1.0 + st[0] + 1e-200)
                        cell.unsafe_store(a * W, cell.unsafe_load[width=W](a * W) * s_ab)
                        cell.unsafe_store(b * W, cell.unsafe_load[width=W](b * W) * s_ba)
                        pairs.unsafe_store((3 * kp) * W, s_ab)
                        pairs.unsafe_store((3 * kp + 1) * W, s_ba)
                        pairs.unsafe_store((3 * kp + 2) * W, st[1] * (0.5 * gadj))
                        kp += 1
                var tot = F64V(0.0)
                for a in range(natm):
                    tot += cell.unsafe_load[width=W](a * W)
                var zi = 1.0 / tot
                var pown = cell.unsafe_load[width=W](owner * W)
                var w0 = vw * pown * zi
                for l in range(nl):
                    w0_out[unsafe_offset=p0 + l] = w0[l]
                if mode != 1:
                    continue
                for i in range(6 * natm):
                    sacc.unsafe_store(i * W, F64V(0.0))
                kp = 0
                for a in range(natm):
                    for b in range(a):
                        var da = dist.unsafe_load[width=W](a * W)
                        var db = dist.unsafe_load[width=W](b * W)
                        var ir = pd[unsafe_offset=4 * kp]
                        var s_ab = pairs.unsafe_load[width=W]((3 * kp) * W)
                        var s_ba = pairs.unsafe_load[width=W]((3 * kp + 1) * W)
                        var t = pairs.unsafe_load[width=W]((3 * kp + 2) * W)
                        var pt_ab = -t / s_ab
                        var pt_ba = t / s_ba
                        var pa = cell.unsafe_load[width=W](a * W)
                        var pb = cell.unsafe_load[width=W](b * W)
                        var dd = (da - db)
                        for c in range(3):
                            var ucomp = pd[unsafe_offset=4 * kp + 1 + c]
                            var na = unit.unsafe_load[width=W]((3 * a + c) * W)
                            var nb = unit.unsafe_load[width=W]((3 * b + c) * W)
                            var du_ab = ir * na - ucomp * dd
                            var du_ba = ir * nb - ucomp * dd
                            # rows a and b of the logarithmic derivatives, weighted by the cell functions
                            var dua = du_ba if a == owner else du_ab
                            var dub = du_ab if b == owner else du_ba
                            var sa = sacc.unsafe_load[width=W]((3 * a + c) * W)
                            sacc.unsafe_store((3 * a + c) * W, sa + (pa * pt_ab + pb * pt_ba) * dua)
                            var sb = sacc.unsafe_load[width=W]((3 * b + c) * W)
                            sacc.unsafe_store((3 * b + c) * W, sb - (pb * pt_ba + pa * pt_ab) * dub)
                            if a == owner:
                                var ta = tacc.unsafe_load[width=W]((3 * a + c) * W)
                                tacc.unsafe_store((3 * a + c) * W, ta + pa * pt_ab * dua)
                                var tb = tacc.unsafe_load[width=W]((3 * b + c) * W)
                                tacc.unsafe_store((3 * b + c) * W, tb - pa * pt_ab * dub)
                            elif b == owner:
                                var ta = tacc.unsafe_load[width=W]((3 * a + c) * W)
                                tacc.unsafe_store((3 * a + c) * W, ta + pb * pt_ba * dua)
                                var tb = tacc.unsafe_load[width=W]((3 * b + c) * W)
                                tacc.unsafe_store((3 * b + c) * W, tb - pb * pt_ba * dub)
                            else:
                                # the grid moves with its atom
                                var uab = (na - nb) * ir
                                var so = sacc.unsafe_load[width=W]((3 * owner + c) * W)
                                sacc.unsafe_store((3 * owner + c) * W, so - (pa * pt_ab + pb * pt_ba) * uab)
                        kp += 1
                var f = ew * vw
                var fz = f * zi
                var fs = f * pown * zi * zi
                for i in range(3 * natm):
                    var v = fz * tacc.unsafe_load[width=W](i * W) - fs * sacc.unsafe_load[width=W](i * W)
                    acc[unsafe_offset=i] += v.reduce_add()
        _ = sl^
        _ = ql^

    if nworkers == 1:
        work(0)
    else:
        parallelize(work, nworkers)
    if mode == 1:
        for i in range(natm * 3):
            var v = 0.0
            for w2 in range(nworkers):
                v += pacc[unsafe_offset=w2 * natm * 3 + i]
            de_out[unsafe_offset=i] = v
    _ = accl^
    _ = pl^
