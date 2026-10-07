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
forming them.  Both run over rows in parallel.
"""
from std.math import sqrt, exp
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
