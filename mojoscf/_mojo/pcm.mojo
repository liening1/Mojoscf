"""PCM surface matrices and their geometry derivatives, contracted (pyscf.solvent.pcm conventions).

Surface point i has position r_i, Gaussian exponent zeta_i (pyscf's
``charge_exp``), switching function F_i, unit normal n_i and the radius
R_i of its sphere.  With xi_ij = zeta_i zeta_j / sqrt(zeta_i^2 + zeta_j^2),
r = |r_i - r_j|, x = xi_ij r and n_ij = (r_i - r_j) . n_j:

    S_ij = erf(x) / r = (2 xi / sqrt(pi)) F_0(x^2),          S_ii = zeta_i sqrt(2/pi) / F_i
    D_ij = (erf(x) - 2 x e^{-x^2} / sqrt(pi)) n_ij / r^3
         = (4 x^3 / sqrt(pi)) F_1(x^2) n_ij / r^3,          D_ii = -zeta_i sqrt(2/pi) / (2 R_i)
    d S_ij / d r_i = s'(r) (r_i - r_j) / r,  s'(r) = -(4 x^3 / sqrt(pi)) F_1(x^2) / r^2
    d D_ij / d r_i = d'(r) (r_i - r_j) / r + s'(r) (-n_j / r + 3 n_ij (r_i - r_j) / r^3),
                     d'(r) = 4 x^2 xi e^{-x^2} n_ij / (sqrt(pi) r^3)

(F_n the Boys function of the integral engine, accurate to ~1e-14, where
std's vector erf is not).  ``pcm_ds_core`` fills S and D; ``pcm_pair_core``
returns G_p = a_p sum_j dX_pj b_j - b_p sum_i a_i dX_ip (X = S or D,
dX_ij = dX_ij / dr_i), the per-point form of pyscf's pairs of einsums
'i,xij,j->ix' - 'i,xij,j->jx' over the (3, n, n) derivative arrays, without
forming them.  For the Hessian, ``pcm_dx_atoms_core`` gives the first
derivatives of S, D or D^T contracted with a vector for every atom of the
partner point, and ``pcm_d2x_atoms_core`` the second derivatives,
d^2 S_ij / dr_i dr_i = c (2 e^2 F_2 d d^T - F_1 I) and
d^2 D_ij / dr_i dr_i = c (4 e^4 F_3 (d.n_j) d d^T - 2 e^2 F_2 ((d.n_j) I + d n_j^T + n_j d^T))
(c = 4 e^3 / sqrt(pi), e = xi_ij, d = r_i - r_j, F_n at e^2 r^2), contracted
v_i X_ij q_j and summed per pair of atoms.  All run over rows in parallel.
"""
from std.math import sqrt, exp
from std.memory import Pointer
from std.runtime import parallelism_level
from max.algorithm import parallelize
from _mojo.linalg import F64Ptr, list_ptr
from _mojo.integrals import BoysTable, PI


def _xi(zi: Float64, zj: Float64) -> Float64:
    return zi * zj / sqrt(zi * zi + zj * zj)


def pcm_ds_core(
    boys: BoysTable, n: Int, coords: F64Ptr, zeta: F64Ptr, switch: F64Ptr, norm: F64Ptr, rvdw: F64Ptr, with_d: Bool,
    s_out: F64Ptr, d_out: F64Ptr,
):
    """S (and with ``with_d`` D) into the dense n x n row-major ``s_out`` / ``d_out``."""
    var nworkers = max(1, min(parallelism_level(), n // 64))
    var c2p = sqrt(2.0 / PI)
    var isp = 1.0 / sqrt(PI)

    def work(w: Int) {imm boys, imm n, imm coords, imm zeta, imm switch, imm norm, imm rvdw, imm with_d, imm s_out, imm d_out, imm nworkers, imm c2p, imm isp}:
        var fl = List[Float64](length=4, fill=0.0)
        var f = list_ptr(fl)
        var i = w
        while i < n:
            var xi_ = coords[unsafe_offset=3 * i]
            var yi = coords[unsafe_offset=3 * i + 1]
            var zi_ = coords[unsafe_offset=3 * i + 2]
            for j in range(n):
                if j == i:
                    s_out[unsafe_offset=i * n + i] = zeta[unsafe_offset=i] * c2p / switch[unsafe_offset=i]
                    if with_d:
                        d_out[unsafe_offset=i * n + i] = -zeta[unsafe_offset=i] * c2p / (2.0 * rvdw[unsafe_offset=i])
                    continue
                var dx = xi_ - coords[unsafe_offset=3 * j]
                var dy = yi - coords[unsafe_offset=3 * j + 1]
                var dz = zi_ - coords[unsafe_offset=3 * j + 2]
                var r2 = dx * dx + dy * dy + dz * dz
                var r = sqrt(r2)
                var xi = _xi(zeta[unsafe_offset=i], zeta[unsafe_offset=j])
                var t = xi * xi * r2
                boys.eval(1 if with_d else 0, t, f)
                s_out[unsafe_offset=i * n + j] = 2.0 * xi * isp * f[0]
                if with_d:
                    var x = xi * r
                    var nij = dx * norm[unsafe_offset=3 * j] + dy * norm[unsafe_offset=3 * j + 1] + dz * norm[unsafe_offset=3 * j + 2]
                    d_out[unsafe_offset=i * n + j] = 4.0 * x * x * x * isp * f[1] * nij / (r2 * r)
            i += nworkers
        _ = fl^

    if nworkers == 1:
        work(0)
    else:
        parallelize(work, nworkers)


def _dpair(
    boys: BoysTable, kind: Int, i: Int, j: Int, coords: F64Ptr, zeta: F64Ptr, norm: F64Ptr, f: F64Ptr, isp: Float64
) -> SIMD[DType.float64, 4]:
    """d X_ij / d r_i (x, y, z, 0) for X = S (kind 0) or D (kind 1), i != j."""
    var dx = coords[unsafe_offset=3 * i] - coords[unsafe_offset=3 * j]
    var dy = coords[unsafe_offset=3 * i + 1] - coords[unsafe_offset=3 * j + 1]
    var dz = coords[unsafe_offset=3 * i + 2] - coords[unsafe_offset=3 * j + 2]
    var r2 = dx * dx + dy * dy + dz * dz
    var r = sqrt(r2)
    var xi = _xi(zeta[unsafe_offset=i], zeta[unsafe_offset=j])
    var x = xi * r
    var t = x * x
    boys.eval(1, t, f)
    var sp = -4.0 * x * t * isp * f[1] / r2          # s'(r)
    var u = SIMD[DType.float64, 4](dx / r, dy / r, dz / r, 0.0)
    if kind == 0:
        return u * sp
    var nx = norm[unsafe_offset=3 * j]
    var ny = norm[unsafe_offset=3 * j + 1]
    var nz = norm[unsafe_offset=3 * j + 2]
    var nij = dx * nx + dy * ny + dz * nz
    var dp = 4.0 * t * xi * isp * exp(-t) * nij / (r2 * r)    # d'(r)
    var nvec = SIMD[DType.float64, 4](nx, ny, nz, 0.0)
    return u * dp + (nvec * (-1.0 / r) + u * (3.0 * nij / r2)) * sp


def pcm_pair_core(
    boys: BoysTable, n: Int, coords: F64Ptr, zeta: F64Ptr, norm: F64Ptr, kind: Int, a: F64Ptr, b: F64Ptr,
    g_out: F64Ptr,
):
    """G_p = a_p sum_j dX_pj b_j - b_p sum_i a_i dX_ip into ``g_out`` (n x 3, overwritten)."""
    var nworkers = max(1, min(parallelism_level(), n // 64))
    var isp = 1.0 / sqrt(PI)

    def work(w: Int) {imm boys, imm n, imm coords, imm zeta, imm norm, imm kind, imm a, imm b, imm g_out, imm nworkers, imm isp}:
        var fl = List[Float64](length=4, fill=0.0)
        var f = list_ptr(fl)
        var p = w
        while p < n:
            var s1 = SIMD[DType.float64, 4](0.0)
            var s2 = SIMD[DType.float64, 4](0.0)
            for j in range(n):
                if j == p:
                    continue
                s1 += _dpair(boys, kind, p, j, coords, zeta, norm, f, isp) * b[unsafe_offset=j]
                s2 += _dpair(boys, kind, j, p, coords, zeta, norm, f, isp) * a[unsafe_offset=j]
            var g = s1 * a[unsafe_offset=p] - s2 * b[unsafe_offset=p]
            g_out[unsafe_offset=3 * p] = g[0]
            g_out[unsafe_offset=3 * p + 1] = g[1]
            g_out[unsafe_offset=3 * p + 2] = g[2]
            p += nworkers
        _ = fl^

    if nworkers == 1:
        work(0)
    else:
        parallelize(work, nworkers)


def _dpair_t(
    boys: BoysTable, kind: Int, i: Int, j: Int, coords: F64Ptr, zeta: F64Ptr, norm: F64Ptr, f: F64Ptr, isp: Float64
) -> SIMD[DType.float64, 4]:
    """d X_ij / d r_i for X = S (kind 0), D (1) or D^T (2: d D_ji / d r_i = -d D_ji / d r_j), i != j."""
    if kind == 2:
        return -_dpair(boys, 1, j, i, coords, zeta, norm, f, isp)
    return _dpair(boys, kind, i, j, coords, zeta, norm, f, isp)


def _d2pair(
    boys: BoysTable, kind: Int, i: Int, j: Int, coords: F64Ptr, zeta: F64Ptr, norm: F64Ptr, f: F64Ptr, isp: Float64
) -> SIMD[DType.float64, 8]:
    """d^2 X_ij / d r_i d r_i (xx, xy, xz, yy, yz, zz) for X = S (kind 0), D (1) or D^T (2), i != j.

    With e = xi_ij, t = e^2 |r_i - r_j|^2, c = 4 e^3 / sqrt(pi), d = r_i - r_j and the
    normal n of point j (D: S_ij's derivative along n_j, D_ij = c F_1 d.n):
        S: c (2 e^2 F_2 d_a d_b - F_1 delta_ab)
        D: c (4 e^4 F_3 d_a d_b (d.n) - 2 e^2 F_2 (delta_ab (d.n) + d_a n_b + n_a d_b))
    (pyscf's get_d2D_d2S elements); D^T takes D_ji (d = r_j - r_i, the normal of i).
    """
    var a = i
    var b = j
    if kind == 2:
        a = j
        b = i
    var dx = coords[unsafe_offset=3 * a] - coords[unsafe_offset=3 * b]
    var dy = coords[unsafe_offset=3 * a + 1] - coords[unsafe_offset=3 * b + 1]
    var dz = coords[unsafe_offset=3 * a + 2] - coords[unsafe_offset=3 * b + 2]
    var r2 = dx * dx + dy * dy + dz * dz
    var e = _xi(zeta[unsafe_offset=i], zeta[unsafe_offset=j])
    var e2 = e * e
    var c = 4.0 * e2 * e * isp
    var dd = SIMD[DType.float64, 8](dx * dx, dx * dy, dx * dz, dy * dy, dy * dz, dz * dz, 0.0, 0.0)
    var delta = SIMD[DType.float64, 8](1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0)
    if kind == 0:
        boys.eval(2, e2 * r2, f)
        return (dd * (2.0 * e2 * f[2]) - delta * f[1]) * c
    boys.eval(3, e2 * r2, f)
    var nx = norm[unsafe_offset=3 * b]
    var ny = norm[unsafe_offset=3 * b + 1]
    var nz = norm[unsafe_offset=3 * b + 2]
    var dn = dx * nx + dy * ny + dz * nz
    var sym = SIMD[DType.float64, 8](
        2.0 * dx * nx, dx * ny + nx * dy, dx * nz + nx * dz, 2.0 * dy * ny, dy * nz + ny * dz, 2.0 * dz * nz, 0.0, 0.0
    )
    return (dd * (4.0 * e2 * e2 * f[3] * dn) - (delta * dn + sym) * (2.0 * e2 * f[2])) * c


def pcm_dx_atoms_core(
    boys: BoysTable, n: Int, coords: F64Ptr, zeta: F64Ptr, norm: F64Ptr, kind: Int, q: F64Ptr,
    catom: Pointer[Int64, MutAnyOrigin], natm: Int, dst: F64Ptr,
):
    """P[A, x, i] = sum_{j on atom A, j != i} d X_ij / d r_i,x q_j into ``dst`` (natm x 3 x n, overwritten), X = S,
    D or D^T (kind 0, 1, 2), point j on atom ``catom[j]``: the pair part of pyscf's dX contracted with a
    vector for every atom (``get_dS_dot_q``, ``get_dD_dot_q``, ``get_dDT_dot_q``) without the (n, n, 3) array."""
    var nworkers = max(1, min(parallelism_level(), n // 64))
    var isp = 1.0 / sqrt(PI)

    def work(w: Int) {imm boys, imm n, imm coords, imm zeta, imm norm, imm kind, imm q, imm catom, imm natm, imm dst, imm nworkers, imm isp}:
        var fl = List[Float64](length=4, fill=0.0)
        var f = list_ptr(fl)
        var accl = List[Float64](length=3 * natm + 1, fill=0.0)
        var acc = list_ptr(accl)
        var i = w
        while i < n:
            for x in range(3 * natm):
                acc[unsafe_offset=x] = 0.0
            for j in range(n):
                if j == i:
                    continue
                var g = _dpair_t(boys, kind, i, j, coords, zeta, norm, f, isp) * q[unsafe_offset=j]
                var aj = 3 * Int(catom[unsafe_offset=j])
                acc[unsafe_offset=aj] += g[0]
                acc[unsafe_offset=aj + 1] += g[1]
                acc[unsafe_offset=aj + 2] += g[2]
            for x in range(3 * natm):
                dst[unsafe_offset=x * n + i] = acc[unsafe_offset=x]
            i += nworkers
        _ = fl^
        _ = accl^

    if nworkers == 1:
        work(0)
    else:
        parallelize(work, nworkers)


def _add_sym(dst: F64Ptr, natm: Int, a: Int, b: Int, x: Int, y: Int, s: Float64):
    """dst[a, b, x, y] += s and, for x != y, dst[a, b, y, x] += s."""
    dst[unsafe_offset=((a * natm + b) * 3 + x) * 3 + y] += s
    if x != y:
        dst[unsafe_offset=((a * natm + b) * 3 + y) * 3 + x] += s


def pcm_d2x_atoms_core(
    boys: BoysTable, n: Int, coords: F64Ptr, zeta: F64Ptr, norm: F64Ptr, kind: Int, v: F64Ptr, q: F64Ptr,
    catom: Pointer[Int64, MutAnyOrigin], natm: Int, dst: F64Ptr,
):
    """X_IJ = sum_{i on I, j on J, i != j} v_i (d^2 X_ij / d r_i d r_i) q_j, scattered as pyscf's
    ``get_v_dot_d2S_dot_q`` / ``get_v_dot_d2D_dot_q`` (pair part): dst[I, I] += X_IJ, dst[J, J] += X_IJ,
    dst[I, J] -= X_IJ, dst[J, I] -= X_IJ into ``dst`` (natm x natm x 3 x 3, overwritten); X = S, D or D^T."""
    var nworkers = max(1, min(parallelism_level(), n // 64))
    var isp = 1.0 / sqrt(PI)
    var per = natm * natm * 6
    var accl = List[Float64](length=nworkers * per + 1, fill=0.0)
    var pacc = list_ptr(accl)

    def work(w: Int) {imm boys, imm n, imm coords, imm zeta, imm norm, imm kind, imm v, imm q, imm catom, imm natm, imm nworkers, imm isp, imm per, imm pacc}:
        var fl = List[Float64](length=4, fill=0.0)
        var f = list_ptr(fl)
        var rowl = List[Float64](length=6 * natm + 1, fill=0.0)
        var row = list_ptr(rowl)
        var acc = pacc.unsafe_offset(w * per)
        var i = w
        while i < n:
            for x in range(6 * natm):
                row[unsafe_offset=x] = 0.0
            for j in range(n):
                if j == i:
                    continue
                var g = _d2pair(boys, kind, i, j, coords, zeta, norm, f, isp) * q[unsafe_offset=j]
                var aj = 6 * Int(catom[unsafe_offset=j])
                for c in range(6):
                    row[unsafe_offset=aj + c] += g[c]
            var vi = v[unsafe_offset=i]
            var dst = acc.unsafe_offset(Int(catom[unsafe_offset=i]) * natm * 6)
            for x in range(6 * natm):
                dst[unsafe_offset=x] += vi * row[unsafe_offset=x]
            i += nworkers
        _ = fl^
        _ = rowl^

    if nworkers == 1:
        work(0)
    else:
        parallelize(work, nworkers)
    for x in range(natm * natm * 9):
        dst[unsafe_offset=x] = 0.0
    for ia in range(natm):
        for ja in range(natm):
            for c in range(6):
                var s = 0.0
                for w2 in range(nworkers):
                    s += pacc[unsafe_offset=w2 * per + (ia * natm + ja) * 6 + c]
                var x = 0 if c < 3 else (1 if c < 5 else 2)
                var y = c if c < 3 else (c - 2 if c < 5 else 2)
                _add_sym(dst, natm, ia, ia, x, y, s)
                _add_sym(dst, natm, ja, ja, x, y, s)
                _add_sym(dst, natm, ia, ja, x, y, -s)
                _add_sym(dst, natm, ja, ia, x, y, -s)
    _ = accl^
