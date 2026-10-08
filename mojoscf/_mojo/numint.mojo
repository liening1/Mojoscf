"""Exchange-correlation integration on pyscf's grids: basis functions, densities and the XC potential matrix.

The functional itself (libxc) is evaluated by pyscf between two passes over
the grid; this module does the expensive parts around it, for LDA and GGA:

    xc_rho_core    rho_s(r) = sum_ij D_s,ij phi_i(r) phi_j(r) and, for GGA,
                   nabla rho_s = 2 sum_ij D_s,ij (nabla phi_i) phi_j, for every
                   grid point and symmetric density D_s
    xc_vmat_core   V_s,ij = sum_r phi_i(r) [sum_c w_s,c(r) phi_c,j(r)] with
                   phi_0 = phi and phi_1..3 = nabla phi (GGA); the caller
                   symmetrises (pyscf's convention: w_0 halved, V + V^T)

The grid is cut into blocks of BLK consecutive points (pyscf sorts the grid
points spatially).  For each block only the shells whose functions exceed
AO_CUT somewhere within the block's bounding sphere are evaluated: their
values and gradients, SIMD over the points (contraction, Cartesian
polynomial, spherical transform; the exponential to one ulp with ``vexp``).
The density then is one GEMM per block, Y = D_sub phi, and the potential
matrix one GEMM per block, V_sub += phi Z^T with Z = sum_c w_c phi_c,
scattered into per-thread matrices.  Blocks are tasks for the worker threads,
each with the sequential BLAS.
"""
from std.atomic import Atomic
from std.math import sqrt, floor, exp
from std.memory import Pointer, bitcast
from std.runtime import parallelism_level
from max.algorithm import parallelize
from _mojo.linalg import F64Ptr, Blas, list_ptr, vfill, vcopy, vaxpy
from _mojo.integrals import Basis, ncart, W, aligned_addr

comptime BLK = 128              # grid points per block
comptime NV = BLK // W          # SIMD vectors per block
comptime AO_CUT = 1.0e-16       # functions (and gradients) below this on a whole block are skipped
comptime F64V = SIMD[DType.float64, W]


def vexp(x0: F64V) -> F64V:
    """exp(x) for x <= 0 to about one ulp, 0 below -700 (std's vector exp is off by ~1e-10 there)."""
    var x = max(x0, F64V(-700.0))
    var n = floor(x * 1.4426950408889634 + 0.5)
    var r = x - n * 0.6931471803691238 - n * 1.9082149292705877e-10
    var p = F64V(1.0 / 6227020800.0)
    p = p * r + 1.0 / 479001600.0
    p = p * r + 1.0 / 39916800.0
    p = p * r + 1.0 / 3628800.0
    p = p * r + 1.0 / 362880.0
    p = p * r + 1.0 / 40320.0
    p = p * r + 1.0 / 5040.0
    p = p * r + 1.0 / 720.0
    p = p * r + 1.0 / 120.0
    p = p * r + 1.0 / 24.0
    p = p * r + 1.0 / 6.0
    p = p * r + 0.5
    p = p * r + 1.0
    p = p * r + 1.0
    var scale = bitcast[DType.float64, W]((n.cast[DType.int64]() + 1023) << 52)
    return x0.lt(-700.0).select(F64V(0.0), p * scale)


def _ncomp(nderiv: Int) -> Int:
    """Number of AO components up to derivative order ``nderiv`` (1, 4, 10, 20)."""
    return 1 if nderiv == 0 else (4 if nderiv == 1 else (10 if nderiv == 2 else 20))


def shell_cutoff_radius(basis: Basis, b: Int, nderiv: Int) -> Float64:
    """Distance from the shell's centre beyond which its functions and derivatives stay below AO_CUT.

    Bound: f(r) = s sum_p max_c |c_cp| r^l e^{-a_p r^2} g^n, g = 1 + l/r + 2 a_p r,
    n = max(nderiv, 1) the derivative order, s the largest coefficient of the
    Cartesian-to-spherical transform.
    """
    var l = basis.l[b]
    var np = basis.nprim[b]
    var nc = basis.nctr[b]
    var s = 1.0
    var off = basis.c2s_off[l]
    for i in range(ncart(l) * basis.nf[l]):
        s = max(s, abs(basis.c2s[off + i]))
    var rmax = 0.0
    var r = 0.05
    while r < 60.0:
        var f = 0.0
        for p in range(np):
            var a = basis.env[basis.pexp[b] + p]
            var cm = 0.0
            for c in range(nc):
                cm = max(cm, abs(basis.env[basis.pcoef[b] + c * np + p]))
            var rl = 1.0
            for _ in range(l):
                rl *= r
            var g1 = 1.0 + Float64(l) / r + 2.0 * a * r
            var g = g1
            for _ in range(1, max(nderiv, 1)):
                g *= g1
            f += cm * rl * exp(-a * r * r) * g
        if s * f >= AO_CUT:
            rmax = r
        r += 0.05
    return rmax + 0.05


struct Grid(Movable):
    """Grid coordinates as structure of arrays, padded to whole blocks, and each block's bounding sphere."""

    var buf: List[Float64]
    var xaddr: Int
    var npad: Int
    var ngrid: Int
    var nblk: Int
    var centre: List[Float64]       # 3 per block
    var radius: List[Float64]

    def __init__(out self, ngrid: Int, coords: F64Ptr):
        self.ngrid = ngrid
        self.nblk = (ngrid + BLK - 1) // BLK
        var npad = max(self.nblk, 1) * BLK
        self.npad = npad
        self.buf = List[Float64](length=3 * npad + 2 * W, fill=0.0)
        self.xaddr = aligned_addr(self.buf)
        var gx = F64Ptr(unsafe_from_address=self.xaddr)
        var gy = gx.unsafe_offset(npad)
        var gz = gx.unsafe_offset(2 * npad)
        for i in range(ngrid):
            gx[unsafe_offset=i] = coords[unsafe_offset=3 * i]
            gy[unsafe_offset=i] = coords[unsafe_offset=3 * i + 1]
            gz[unsafe_offset=i] = coords[unsafe_offset=3 * i + 2]
        # padding points repeat the last real point (their weights are zero)
        for i in range(ngrid, npad):
            var j = max(ngrid - 1, 0)
            gx[unsafe_offset=i] = gx[unsafe_offset=j]
            gy[unsafe_offset=i] = gy[unsafe_offset=j]
            gz[unsafe_offset=i] = gz[unsafe_offset=j]
        self.centre = List[Float64](length=3 * max(self.nblk, 1), fill=0.0)
        self.radius = List[Float64](length=max(self.nblk, 1), fill=0.0)
        for k in range(self.nblk):
            var p0 = k * BLK
            var p1 = min(p0 + BLK, ngrid)
            var cx = 0.0
            var cy = 0.0
            var cz = 0.0
            for i in range(p0, p1):
                cx += gx[unsafe_offset=i]
                cy += gy[unsafe_offset=i]
                cz += gz[unsafe_offset=i]
            var inv = 1.0 / Float64(p1 - p0)
            cx *= inv
            cy *= inv
            cz *= inv
            var r2 = 0.0
            for i in range(p0, p1):
                var dx = gx[unsafe_offset=i] - cx
                var dy = gy[unsafe_offset=i] - cy
                var dz = gz[unsafe_offset=i] - cz
                r2 = max(r2, dx * dx + dy * dy + dz * dz)
            self.centre[3 * k] = cx
            self.centre[3 * k + 1] = cy
            self.centre[3 * k + 2] = cz
            self.radius[k] = sqrt(r2)

    def x(self) -> F64Ptr:
        return F64Ptr(unsafe_from_address=self.xaddr)

    def y(self) -> F64Ptr:
        return F64Ptr(unsafe_from_address=self.xaddr).unsafe_offset(self.npad)

    def z(self) -> F64Ptr:
        return F64Ptr(unsafe_from_address=self.xaddr).unsafe_offset(2 * self.npad)


struct AOWork(Movable):
    """Per-thread scratch: the block's AO values [comp][row][BLK], index lists and GEMM buffers."""

    var buf: List[Float64]
    var aoaddr: Int         # ncomp x nao x BLK
    var yaddr: Int          # 3 x nao x BLK
    var daddr: Int          # nao x nao
    var cartaddr: Int       # small: Cartesian values per comp
    var pwaddr: Int         # powers and radial sums
    var rows: List[Int]     # global AO index of each local row
    var shells: List[Int]
    var run_row: List[Int]  # runs of consecutive AO indices: first local row,
    var run_ao: List[Int]   # first global AO index,
    var run_len: List[Int]  # length
    var nrun: Int

    def __init__(out self, nao: Int, nbas: Int, ncomp: Int, lmax: Int, nctr_max: Int):
        var nc = ncart(lmax)
        var n_ao = ncomp * nao * BLK
        var n_y = 3 * nao * BLK
        var n_d = nao * nao
        var n_cart = ncomp * nc * W
        var n_pw = (3 * (lmax + 4) + 4 * nctr_max) * W
        self.buf = List[Float64](length=n_ao + n_y + n_d + n_cart + n_pw + 8 * W, fill=0.0)
        var base = aligned_addr(self.buf)
        self.aoaddr = base
        self.yaddr = base + n_ao * 8
        self.daddr = self.yaddr + n_y * 8
        var c0 = (n_ao + n_y + n_d + W - 1) // W * W
        self.cartaddr = base + c0 * 8
        self.pwaddr = self.cartaddr + n_cart * 8
        self.rows = List[Int](length=max(nao, 1), fill=0)
        self.shells = List[Int](length=max(nbas, 1), fill=0)
        self.run_row = List[Int](length=max(nbas, 1), fill=0)
        self.run_ao = List[Int](length=max(nbas, 1), fill=0)
        self.run_len = List[Int](length=max(nbas, 1), fill=0)
        self.nrun = 0

    def ao(self) -> F64Ptr:
        return F64Ptr(unsafe_from_address=self.aoaddr)

    def y(self) -> F64Ptr:
        return F64Ptr(unsafe_from_address=self.yaddr)

    def y2(self, nrow: Int) -> F64Ptr:
        """A second nrow x BLK buffer after ``y``."""
        return F64Ptr(unsafe_from_address=self.yaddr).unsafe_offset(nrow * BLK)

    def y3(self, nrow: Int) -> F64Ptr:
        """A third nrow x BLK buffer."""
        return F64Ptr(unsafe_from_address=self.yaddr).unsafe_offset(2 * nrow * BLK)

    def dsub(self) -> F64Ptr:
        return F64Ptr(unsafe_from_address=self.daddr)


def select_shells(basis: Basis, grid: Grid, blk: Int, rcut: List[Float64], mut ws: AOWork) -> Tuple[Int, Int]:
    """Shells significant on block ``blk`` into ``ws.shells`` and their AO rows into ``ws.rows``.

    Returns (number of shells, number of AO rows).
    """
    var cx = grid.centre[3 * blk]
    var cy = grid.centre[3 * blk + 1]
    var cz = grid.centre[3 * blk + 2]
    var rad = grid.radius[blk]
    var ns = 0
    var nrow = 0
    var nrun = 0
    for b in range(basis.nbas):
        var dx = basis.shell_coord(b, 0) - cx
        var dy = basis.shell_coord(b, 1) - cy
        var dz = basis.shell_coord(b, 2) - cz
        var d = sqrt(dx * dx + dy * dy + dz * dz) - rad
        if d > rcut[b]:
            continue
        ws.shells[ns] = b
        ns += 1
        var i0 = basis.ao_loc[b]
        var i1 = basis.ao_loc[b + 1]
        if nrun > 0 and ws.run_ao[nrun - 1] + ws.run_len[nrun - 1] == i0:
            ws.run_len[nrun - 1] += i1 - i0
        else:
            ws.run_row[nrun] = nrow
            ws.run_ao[nrun] = i0
            ws.run_len[nrun] = i1 - i0
            nrun += 1
        for i in range(i0, i1):
            ws.rows[nrow] = i
            nrow += 1
    ws.nrun = nrun
    return (ns, nrow)


def eval_block(basis: Basis, grid: Grid, blk: Int, nderiv: Int, nshell: Int, nrow: Int, mut ws: AOWork):
    """Values, gradients (``nderiv`` >= 1) and second derivatives (``nderiv`` = 2) of the selected shells
    on block ``blk`` into ws.ao[comp][row][BLK]; components in pyscf's order (1, x, y, z, xx, xy, xz, yy, yz, zz).

    A Cartesian function x^i y^j z^k R(r^2), R = sum_p c_p e^{-a_p r^2}, with the
    radial sums S = sum_p c_p (-2 a_p) e^{-a_p r^2} and T = sum_p c_p (4 a_p^2) e^{-a_p r^2}:
    d/dx = (i x^{i-1} R + x^{i+1} S) y^j z^k,
    d2/dx2 = (i(i-1) x^{i-2} R + (2i+1) x^i S + x^{i+2} T) y^j z^k,
    d2/dxdy = (a_x a_y R + (a_x b_y + b_x a_y) S + b_x b_y T) z^k with a_x = i x^{i-1}, b_x = x^{i+1}.
    """
    if nderiv == 3:
        _eval_block3(basis, grid, blk, nshell, nrow, ws)
        return
    var p0 = blk * BLK
    var cstride = nrow * BLK
    var cart = F64Ptr(unsafe_from_address=ws.cartaddr)
    var pw = F64Ptr(unsafe_from_address=ws.pwaddr)
    var ncomp = _ncomp(nderiv)
    var row = 0
    for si in range(nshell):
        var b = ws.shells[si]
        var l = basis.l[b]
        var np = basis.nprim[b]
        var nc = basis.nctr[b]
        var nca = ncart(l)
        var nf = basis.nf[l]
        var coff = basis.cart_off[l]
        var scal = basis.c2s_scale[l]
        var c2s = list_ptr(basis.c2s).unsafe_offset(basis.c2s_off[l])
        var ax = basis.shell_coord(b, 0)
        var ay = basis.shell_coord(b, 1)
        var az = basis.shell_coord(b, 2)
        var npw = l + 3
        var xp = pw                                 # dx^0 .. dx^(l+2)
        var yp = pw.unsafe_offset(npw * W)
        var zp = pw.unsafe_offset(2 * npw * W)
        var rs = pw.unsafe_offset(3 * npw * W)      # R_c, S_c, T_c
        var ss = rs.unsafe_offset(nc * W)
        var ts = ss.unsafe_offset(nc * W)
        for v in range(NV):
            var dx = grid.x().unsafe_load[width=W](p0 + v * W) - ax
            var dy = grid.y().unsafe_load[width=W](p0 + v * W) - ay
            var dz = grid.z().unsafe_load[width=W](p0 + v * W) - az
            var r2 = dx * dx + dy * dy + dz * dz
            for c in range(nc):
                rs.unsafe_store(c * W, F64V(0.0))
                ss.unsafe_store(c * W, F64V(0.0))
                ts.unsafe_store(c * W, F64V(0.0))
            for p in range(np):
                var a = basis.env[basis.pexp[b] + p]
                var e = vexp(r2 * (-a))
                for c in range(nc):
                    var cf = basis.env[basis.pcoef[b] + c * np + p]
                    rs.unsafe_store(c * W, rs.unsafe_load[width=W](c * W) + e * cf)
                    if nderiv > 0:
                        ss.unsafe_store(c * W, ss.unsafe_load[width=W](c * W) + e * (-2.0 * a * cf))
                    if nderiv > 1:
                        ts.unsafe_store(c * W, ts.unsafe_load[width=W](c * W) + e * (4.0 * a * a * cf))
            var tx = F64V(1.0)
            var ty = F64V(1.0)
            var tz = F64V(1.0)
            for k in range(npw):
                xp.unsafe_store(k * W, tx)
                yp.unsafe_store(k * W, ty)
                zp.unsafe_store(k * W, tz)
                tx *= dx
                ty *= dy
                tz *= dz
            for c in range(nc):
                var rr = rs.unsafe_load[width=W](c * W)
                var sv = ss.unsafe_load[width=W](c * W)
                var tv = ts.unsafe_load[width=W](c * W)
                for cc in range(nca):
                    var i = basis.cx[coff + cc]
                    var j = basis.cy[coff + cc]
                    var k = basis.cz[coff + cc]
                    var xi = xp.unsafe_load[width=W](i * W)
                    var yj = yp.unsafe_load[width=W](j * W)
                    var zk = zp.unsafe_load[width=W](k * W)
                    var yz = yj * zk
                    cart.unsafe_store(cc * W, xi * yz * rr)
                    if nderiv > 0:
                        # a_x = i x^{i-1}, b_x = x^{i+1} (likewise y, z)
                        var axv = F64V(0.0)
                        if i > 0:
                            axv = xp.unsafe_load[width=W]((i - 1) * W) * Float64(i)
                        var ayv = F64V(0.0)
                        if j > 0:
                            ayv = yp.unsafe_load[width=W]((j - 1) * W) * Float64(j)
                        var azv = F64V(0.0)
                        if k > 0:
                            azv = zp.unsafe_load[width=W]((k - 1) * W) * Float64(k)
                        var bxv = xp.unsafe_load[width=W]((i + 1) * W)
                        var byv = yp.unsafe_load[width=W]((j + 1) * W)
                        var bzv = zp.unsafe_load[width=W]((k + 1) * W)
                        var gxv = axv * rr + bxv * sv
                        var gyv = ayv * rr + byv * sv
                        var gzv = azv * rr + bzv * sv
                        cart.unsafe_store((nca + cc) * W, gxv * yz)
                        cart.unsafe_store((2 * nca + cc) * W, xi * zk * gyv)
                        cart.unsafe_store((3 * nca + cc) * W, xi * yj * gzv)
                        if nderiv > 1:
                            var hxx = F64V(0.0)
                            if i > 1:
                                hxx = xp.unsafe_load[width=W]((i - 2) * W) * (Float64(i * (i - 1)) * rr)
                            hxx += xi * (Float64(2 * i + 1) * sv) + xp.unsafe_load[width=W]((i + 2) * W) * tv
                            var hyy = F64V(0.0)
                            if j > 1:
                                hyy = yp.unsafe_load[width=W]((j - 2) * W) * (Float64(j * (j - 1)) * rr)
                            hyy += yj * (Float64(2 * j + 1) * sv) + yp.unsafe_load[width=W]((j + 2) * W) * tv
                            var hzz = F64V(0.0)
                            if k > 1:
                                hzz = zp.unsafe_load[width=W]((k - 2) * W) * (Float64(k * (k - 1)) * rr)
                            hzz += zk * (Float64(2 * k + 1) * sv) + zp.unsafe_load[width=W]((k + 2) * W) * tv
                            var hxy = axv * ayv * rr + (axv * byv + bxv * ayv) * sv + bxv * byv * tv
                            var hxz = axv * azv * rr + (axv * bzv + bxv * azv) * sv + bxv * bzv * tv
                            var hyz = ayv * azv * rr + (ayv * bzv + byv * azv) * sv + byv * bzv * tv
                            cart.unsafe_store((4 * nca + cc) * W, hxx * yz)
                            cart.unsafe_store((5 * nca + cc) * W, hxy * zk)
                            cart.unsafe_store((6 * nca + cc) * W, hxz * yj)
                            cart.unsafe_store((7 * nca + cc) * W, xi * zk * hyy)
                            cart.unsafe_store((8 * nca + cc) * W, xi * hyz)
                            cart.unsafe_store((9 * nca + cc) * W, xi * yj * hzz)
                var r0 = row + c * nf
                for comp in range(ncomp):
                    var cb = cart.unsafe_offset(comp * nca * W)
                    var dst = ws.ao().unsafe_offset(comp * cstride + r0 * BLK + v * W)
                    if scal != 0.0:
                        for m in range(nf):
                            dst.unsafe_store(m * BLK, cb.unsafe_load[width=W](m * W) * scal)
                    else:
                        for m in range(nf):
                            var acc = F64V(0.0)
                            for cc in range(nca):
                                var t = c2s[unsafe_offset=cc * nf + m]
                                if t != 0.0:
                                    acc += cb.unsafe_load[width=W](cc * W) * t
                            dst.unsafe_store(m * BLK, acc)
        row += nc * nf


def _d1(p: Int, n: Int, i: Int) -> Tuple[Float64, Int]:
    """Term n of d^p/dx^p (x^i g_0) = sum_n c x^(i - p + 2n) g_n, g_{n+1} = (1/x) d g_n / dx: (c, power)."""
    var fi = Float64(i)
    var c = 0.0
    if p == 0:
        c = 1.0
    elif p == 1:
        c = fi if n == 0 else 1.0
    elif p == 2:
        c = fi * (fi - 1.0) if n == 0 else (2.0 * fi + 1.0 if n == 1 else 1.0)
    else:
        if n == 0:
            c = fi * (fi - 1.0) * (fi - 2.0)
        elif n == 1:
            c = 3.0 * fi * fi
        elif n == 2:
            c = 3.0 * fi + 3.0
        else:
            c = 1.0
    return (c, i - p + 2 * n)


def _comp_powers(comp: Int) -> Tuple[Int, Int, Int]:
    """Derivative orders (x, y, z) of AO component ``comp`` in pyscf's order up to third derivatives."""
    var code = 0
    if comp == 0:
        code = 0
    elif comp == 1:
        code = 100
    elif comp == 2:
        code = 10
    elif comp == 3:
        code = 1
    elif comp == 4:
        code = 200
    elif comp == 5:
        code = 110
    elif comp == 6:
        code = 101
    elif comp == 7:
        code = 20
    elif comp == 8:
        code = 11
    elif comp == 9:
        code = 2
    elif comp == 10:
        code = 300
    elif comp == 11:
        code = 210
    elif comp == 12:
        code = 201
    elif comp == 13:
        code = 120
    elif comp == 14:
        code = 111
    elif comp == 15:
        code = 102
    elif comp == 16:
        code = 30
    elif comp == 17:
        code = 21
    elif comp == 18:
        code = 12
    else:
        code = 3
    return (code // 100, (code // 10) % 10, code % 10)


def _eval_block3(basis: Basis, grid: Grid, blk: Int, nshell: Int, nrow: Int, mut ws: AOWork):
    """``eval_block`` with derivatives up to third order: 20 components in pyscf's order
    (1, x, y, z, xx, xy, xz, yy, yz, zz, xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz).

    A Cartesian function x^i y^j z^k R(r^2) with the radial sums
    F_n = sum_p c_p (-2 a_p)^n e^{-a_p r^2} (d F_n / dx = x F_{n+1}) has
    d^px d^py d^pz phi = sum_{nx,ny,nz} X_px,nx Y_py,ny Z_pz,nz F_{nx+ny+nz}, the
    one-dimensional factors from ``_d1``.
    """
    var p0 = blk * BLK
    var cstride = nrow * BLK
    var cart = F64Ptr(unsafe_from_address=ws.cartaddr)
    var pw = F64Ptr(unsafe_from_address=ws.pwaddr)
    comptime NC3 = 20
    var cpx = List[Int](length=NC3, fill=0)
    var cpy = List[Int](length=NC3, fill=0)
    var cpz = List[Int](length=NC3, fill=0)
    for comp in range(NC3):
        var t = _comp_powers(comp)
        cpx[comp] = t[0]
        cpy[comp] = t[1]
        cpz[comp] = t[2]
    # 1D factors for p = 0..3, n = 0..p: 10 per axis
    var fac = List[Float64](length=3 * 10 * W, fill=0.0)
    var pf = list_ptr(fac)
    var row = 0
    for si in range(nshell):
        var b = ws.shells[si]
        var l = basis.l[b]
        var np = basis.nprim[b]
        var nc = basis.nctr[b]
        var nca = ncart(l)
        var nf = basis.nf[l]
        var coff = basis.cart_off[l]
        var scal = basis.c2s_scale[l]
        var c2s = list_ptr(basis.c2s).unsafe_offset(basis.c2s_off[l])
        var ax = basis.shell_coord(b, 0)
        var ay = basis.shell_coord(b, 1)
        var az = basis.shell_coord(b, 2)
        var npw = l + 4
        var xp = pw                                 # dx^0 .. dx^(l+3)
        var yp = pw.unsafe_offset(npw * W)
        var zp = pw.unsafe_offset(2 * npw * W)
        var fs = pw.unsafe_offset(3 * npw * W)      # F_n of contraction c at (c * 4 + n) * W
        for v in range(NV):
            var dx = grid.x().unsafe_load[width=W](p0 + v * W) - ax
            var dy = grid.y().unsafe_load[width=W](p0 + v * W) - ay
            var dz = grid.z().unsafe_load[width=W](p0 + v * W) - az
            var r2 = dx * dx + dy * dy + dz * dz
            for c in range(nc):
                for n in range(4):
                    fs.unsafe_store((c * 4 + n) * W, F64V(0.0))
            for p in range(np):
                var a = basis.env[basis.pexp[b] + p]
                var e = vexp(r2 * (-a))
                for c in range(nc):
                    var t = e * basis.env[basis.pcoef[b] + c * np + p]
                    for n in range(4):
                        fs.unsafe_store((c * 4 + n) * W, fs.unsafe_load[width=W]((c * 4 + n) * W) + t)
                        t *= -2.0 * a
            var tx = F64V(1.0)
            var ty = F64V(1.0)
            var tz = F64V(1.0)
            for k in range(npw):
                xp.unsafe_store(k * W, tx)
                yp.unsafe_store(k * W, ty)
                zp.unsafe_store(k * W, tz)
                tx *= dx
                ty *= dy
                tz *= dz
            for c in range(nc):
                var f = fs.unsafe_offset(c * 4 * W)
                for cc in range(nca):
                    var ii = basis.cx[coff + cc]
                    var jj = basis.cy[coff + cc]
                    var kk = basis.cz[coff + cc]
                    # 1D factors of each axis: slot p (p + 1) / 2 + n
                    for p in range(4):
                        for n in range(p + 1):
                            var slot = p * (p + 1) // 2 + n
                            var tx_ = _d1(p, n, ii)
                            var ty_ = _d1(p, n, jj)
                            var tz_ = _d1(p, n, kk)
                            var vx = F64V(0.0)
                            if tx_[1] >= 0:
                                vx = xp.unsafe_load[width=W](tx_[1] * W) * tx_[0]
                            var vy = F64V(0.0)
                            if ty_[1] >= 0:
                                vy = yp.unsafe_load[width=W](ty_[1] * W) * ty_[0]
                            var vz = F64V(0.0)
                            if tz_[1] >= 0:
                                vz = zp.unsafe_load[width=W](tz_[1] * W) * tz_[0]
                            pf.unsafe_store(slot * W, vx)
                            pf.unsafe_store((10 + slot) * W, vy)
                            pf.unsafe_store((20 + slot) * W, vz)
                    for comp in range(NC3):
                        var px = cpx[comp]
                        var py = cpy[comp]
                        var pz = cpz[comp]
                        var acc = F64V(0.0)
                        for nx in range(px + 1):
                            var fx = pf.unsafe_load[width=W]((px * (px + 1) // 2 + nx) * W)
                            for ny in range(py + 1):
                                var fxy = fx * pf.unsafe_load[width=W]((10 + py * (py + 1) // 2 + ny) * W)
                                for nz in range(pz + 1):
                                    var fz = pf.unsafe_load[width=W]((20 + pz * (pz + 1) // 2 + nz) * W)
                                    acc += fxy * fz * f.unsafe_load[width=W]((nx + ny + nz) * W)
                        cart.unsafe_store((comp * nca + cc) * W, acc)
                var r0 = row + c * nf
                for comp in range(NC3):
                    var cb = cart.unsafe_offset(comp * nca * W)
                    var dst = ws.ao().unsafe_offset(comp * cstride + r0 * BLK + v * W)
                    if scal != 0.0:
                        for m in range(nf):
                            dst.unsafe_store(m * BLK, cb.unsafe_load[width=W](m * W) * scal)
                    else:
                        for m in range(nf):
                            var acc = F64V(0.0)
                            for cc in range(nca):
                                var t = c2s[unsafe_offset=cc * nf + m]
                                if t != 0.0:
                                    acc += cb.unsafe_load[width=W](cc * W) * t
                            dst.unsafe_store(m * BLK, acc)
        row += nc * nf
    _ = fac^
    _ = cpx^
    _ = cpy^
    _ = cpz^


def gather_block(dm: F64Ptr, nao: Int, nrow: Int, ws: AOWork):
    """ws.dsub[i][j] = dm[rows_i][rows_j], copying runs of consecutive AO indices."""
    var dsub = ws.dsub()
    for ri in range(ws.nrun):
        for a in range(ws.run_len[ri]):
            var src = dm.unsafe_offset((ws.run_ao[ri] + a) * nao)
            var dst = dsub.unsafe_offset((ws.run_row[ri] + a) * nrow)
            for rj in range(ws.nrun):
                var s = src.unsafe_offset(ws.run_ao[rj])
                var d = dst.unsafe_offset(ws.run_row[rj])
                var n = ws.run_len[rj]
                var k = 0
                while k + W <= n:
                    d.unsafe_store(k, s.unsafe_load[width=W](k))
                    k += W
                while k < n:
                    d[unsafe_offset=k] = s[unsafe_offset=k]
                    k += 1


def scatter_add_block(v: F64Ptr, nao: Int, nrow: Int, ws: AOWork):
    """v[rows_i][rows_j] += ws.dsub[i][j], by runs of consecutive AO indices."""
    var dsub = ws.dsub()
    for ri in range(ws.nrun):
        for a in range(ws.run_len[ri]):
            var dst = v.unsafe_offset((ws.run_ao[ri] + a) * nao)
            var src = dsub.unsafe_offset((ws.run_row[ri] + a) * nrow)
            for rj in range(ws.nrun):
                var d = dst.unsafe_offset(ws.run_ao[rj])
                var s = src.unsafe_offset(ws.run_row[rj])
                var n = ws.run_len[rj]
                var k = 0
                while k + W <= n:
                    d.unsafe_store(k, d.unsafe_load[width=W](k) + s.unsafe_load[width=W](k))
                    k += W
                while k < n:
                    d[unsafe_offset=k] += s[unsafe_offset=k]
                    k += 1


def _rcuts(basis: Basis, nderiv: Int) -> List[Float64]:
    var rc = List[Float64](length=max(basis.nbas, 1), fill=0.0)
    for b in range(basis.nbas):
        rc[b] = shell_cutoff_radius(basis, b, nderiv)
    return rc^


def eval_ao_core(basis: Basis, ngrid: Int, coords: F64Ptr, nderiv: Int, ao_out: F64Ptr):
    """All AO values and derivatives up to order ``nderiv`` on the grid: out[comp][point][ao] (``eval_gto``'s layout).

    For tests: screened functions are left at zero, as pyscf does.
    """
    var grid = Grid(ngrid, coords)
    var rcut = _rcuts(basis, nderiv)
    var nao = basis.nao
    var ncomp = _ncomp(nderiv)
    vfill(ao_out, ncomp * ngrid * nao, 0.0)
    var ws = AOWork(nao, basis.nbas, ncomp, basis.lmax, basis.nctr_max)
    for blk in range(grid.nblk):
        var sel = select_shells(basis, grid, blk, rcut, ws)
        var nrow = sel[1]
        eval_block(basis, grid, blk, nderiv, sel[0], nrow, ws)
        var p0 = blk * BLK
        for comp in range(ncomp):
            for r in range(nrow):
                var src = ws.ao().unsafe_offset((comp * nrow + r) * BLK)
                var col = ws.rows[r]
                for p in range(min(BLK, ngrid - p0)):
                    ao_out[unsafe_offset=(comp * ngrid + p0 + p) * nao + col] = src[unsafe_offset=p]
    _ = ws^
    _ = grid^
    _ = rcut^


def gather_rows(c: F64Ptr, ncol: Int, ws: AOWork):
    """ws.dsub[i][:] = c[rows_i][:] (rows of an nao x ncol matrix), by runs of consecutive AO indices."""
    gather_rows_to(c, ncol, ws, ws.dsub())


def gather_rows_to(c: F64Ptr, ncol: Int, ws: AOWork, out_rows: F64Ptr):
    """out_rows[i][:] = c[rows_i][:] (rows of an nao x ncol matrix), by runs of consecutive AO indices."""
    for ri in range(ws.nrun):
        var src = c.unsafe_offset(ws.run_ao[ri] * ncol)
        var dst = out_rows.unsafe_offset(ws.run_row[ri] * ncol)
        var n = ws.run_len[ri] * ncol
        var k = 0
        while k + W <= n:
            dst.unsafe_store(k, src.unsafe_load[width=W](k))
            k += W
        while k < n:
            dst[unsafe_offset=k] = src[unsafe_offset=k]
            k += 1


def _store_pts(dst: F64Ptr, v: Int, npt: Int, acc: F64V):
    """Store the W values of block vector ``v`` at dst[v W:], only those before ``npt``."""
    if v * W + W <= npt:
        dst.unsafe_store(v * W, acc)
    else:
        for lane in range(W):
            if v * W + lane < npt:
                dst[unsafe_offset=v * W + lane] = acc[lane]


def _nrho(kind: Int) -> Int:
    """Density components per set: 1 (LDA), 4 (GGA: rho, grad rho), 5 (meta-GGA: + tau)."""
    return 1 if kind == 0 else (4 if kind == 1 else 5)


def xc_rho_core(
    blas: Blas, basis: Basis, ngrid: Int, coords: F64Ptr, kind: Int, nset: Int, dms: F64Ptr, norb: Int,
    orbs: F64Ptr, occs: F64Ptr, rho_out: F64Ptr,
) raises:
    """rho_out[s][c][p] for each symmetric D_s: the density (c = 0); for ``kind`` 1 (GGA) and 2
    (meta-GGA) its gradient (c = 1..3); for meta-GGA tau = 1/2 sum_c sum_ij D_ij d_c phi_i d_c phi_j
    (c = 4, pyscf's convention).

    With ``norb`` > 0, D_s = C_s diag(n_s) C_s^T is also given by orbitals
    ``orbs[s]`` (nao x norb, row-major) and occupations ``occs[s]`` (norb);
    blocks where it is cheaper use them: psi = C_sub^T phi, rho = sum_k n_k
    psi_k^2 and, for GGA, Y = D_sub phi = C_sub (n psi), two GEMMs of
    norb x nrow instead of one of nrow x nrow; for meta-GGA psi_c = C_sub^T
    d_c phi too, grad rho = 2 sum_k n_k psi_k psi_c,k and tau = 1/2 sum_k n_k
    sum_c psi_c,k^2 (four GEMMs of norb x nrow instead of four of nrow x nrow).
    """
    var nderiv = 0 if kind == 0 else 1
    var nout = _nrho(kind)
    var grid = Grid(ngrid, coords)
    var rcut = _rcuts(basis, nderiv)
    var nao = basis.nao
    var n2 = nao * nao
    var ncomp = _ncomp(nderiv)
    var nblk = grid.nblk
    var nworkers = max(1, min(parallelism_level(), nblk))
    var counter = Atomic[Int64](0)
    var pcount = Pointer(to=counter)

    def work(w: Int) {imm blas, imm basis, imm grid, imm rcut, imm pcount, imm nblk, imm nao, imm n2, imm ncomp, imm nderiv, imm nout, imm kind, imm nset, imm dms, imm norb, imm orbs, imm occs, imm rho_out, imm ngrid}:
        var ws = AOWork(nao, basis.nbas, ncomp, basis.lmax, basis.nctr_max)
        var pl = List[Float64](length=(4 if kind == 2 else 1) * max(norb, 1) * BLK + W, fill=0.0)
        var pbuf = F64Ptr(unsafe_from_address=aligned_addr(pl))     # psi_c (meta-GGA orbital route)
        var tl = List[Float64](length=BLK + W, fill=0.0)
        var tau = F64Ptr(unsafe_from_address=aligned_addr(tl))
        # orbitals when their GEMMs are cheaper than the products with D_sub
        var ngemm_orb = 1 if kind == 0 else (2 if kind == 1 else 4)
        var ngemm_d = 4 if kind == 2 else 1
        while True:
            var blk = Int(pcount[].fetch_add(1))
            if blk >= nblk:
                break
            var sel = select_shells(basis, grid, blk, rcut, ws)
            var nrow = sel[1]
            var p0 = blk * BLK
            var npt = min(BLK, ngrid - p0)
            if nrow == 0:
                for s in range(nset):
                    for c in range(nout):
                        vfill(rho_out.unsafe_offset((s * nout + c) * ngrid + p0), npt, 0.0)
                continue
            eval_block(basis, grid, blk, nderiv, sel[0], nrow, ws)
            var cs = nrow * BLK
            var use_orb = norb > 0 and ngemm_orb * norb * 10 < ngemm_d * nrow * 9
            for s in range(nset):
                var out = rho_out.unsafe_offset((s * nout) * ngrid + p0)
                var c0 = 0
                if use_orb:
                    var nk = occs.unsafe_offset(s * norb)
                    gather_rows(orbs.unsafe_offset(s * nao * norb), norb, ws)
                    if kind == 2:
                        var pb = norb * BLK
                        for c in range(4):
                            try:
                                blas.gemm(True, False, norb, BLK, nrow, 1.0, ws.dsub(), ws.ao().unsafe_offset(c * cs), 0.0,
                                          pbuf.unsafe_offset(c * pb))
                            except:
                                pass
                        for v in range(NV):
                            var r = F64V(0.0)
                            var gx = F64V(0.0)
                            var gy = F64V(0.0)
                            var gz = F64V(0.0)
                            var t = F64V(0.0)
                            for k in range(norb):
                                var o = k * BLK + v * W
                                var n = nk[unsafe_offset=k]
                                var q = pbuf.unsafe_load[width=W](o)
                                var qx = pbuf.unsafe_load[width=W](pb + o)
                                var qy = pbuf.unsafe_load[width=W](2 * pb + o)
                                var qz = pbuf.unsafe_load[width=W](3 * pb + o)
                                var nq = q * n
                                r += nq * q
                                gx += nq * qx
                                gy += nq * qy
                                gz += nq * qz
                                t += (qx * qx + qy * qy + qz * qz) * n
                            _store_pts(out, v, npt, r)
                            _store_pts(out.unsafe_offset(ngrid), v, npt, gx * 2.0)
                            _store_pts(out.unsafe_offset(2 * ngrid), v, npt, gy * 2.0)
                            _store_pts(out.unsafe_offset(3 * ngrid), v, npt, gz * 2.0)
                            _store_pts(out.unsafe_offset(4 * ngrid), v, npt, t * 0.5)
                        continue
                    var psi = ws.y2(nrow)
                    try:
                        blas.gemm(True, False, norb, BLK, nrow, 1.0, ws.dsub(), ws.ao(), 0.0, psi)
                    except:
                        pass
                    for v in range(NV):
                        var acc = F64V(0.0)
                        for k in range(norb):
                            var t = psi.unsafe_load[width=W](k * BLK + v * W)
                            acc += t * t * nk[unsafe_offset=k]
                        _store_pts(out, v, npt, acc)
                    if kind == 0:
                        continue
                    for k in range(norb):
                        var f = nk[unsafe_offset=k]
                        for v in range(NV):
                            psi.unsafe_store(k * BLK + v * W, psi.unsafe_load[width=W](k * BLK + v * W) * f)
                    try:
                        blas.gemm(False, False, nrow, BLK, norb, 1.0, ws.dsub(), psi, 0.0, ws.y())
                    except:
                        pass
                    c0 = 1
                else:
                    gather_block(dms.unsafe_offset(s * n2), nao, nrow, ws)
                    try:
                        blas.gemm(False, False, nrow, BLK, nrow, 1.0, ws.dsub(), ws.ao(), 0.0, ws.y())
                    except:
                        pass
                for c in range(c0, ncomp):
                    var fac = 1.0 if c == 0 else 2.0
                    var ac = ws.ao().unsafe_offset(c * cs)
                    for v in range(NV):
                        var acc = F64V(0.0)
                        for i in range(nrow):
                            acc += ac.unsafe_load[width=W](i * BLK + v * W) * ws.y().unsafe_load[width=W](i * BLK + v * W)
                        _store_pts(out.unsafe_offset(c * ngrid), v, npt, acc * fac)
                if kind == 2:
                    # tau = 1/2 sum_c sum_i d_c phi_i (D_sub d_c phi)_i (D_sub still in ws.dsub)
                    vfill(tau, BLK, 0.0)
                    var yc = ws.y2(nrow)
                    for c in range(1, 4):
                        var ac = ws.ao().unsafe_offset(c * cs)
                        try:
                            blas.gemm(False, False, nrow, BLK, nrow, 1.0, ws.dsub(), ac, 0.0, yc)
                        except:
                            pass
                        for v in range(NV):
                            var acc = tau.unsafe_load[width=W](v * W)
                            for i in range(nrow):
                                acc += ac.unsafe_load[width=W](i * BLK + v * W) * yc.unsafe_load[width=W](i * BLK + v * W)
                            tau.unsafe_store(v * W, acc)
                    for v in range(NV):
                        _store_pts(out.unsafe_offset(4 * ngrid), v, npt, tau.unsafe_load[width=W](v * W) * 0.5)
        _ = ws^
        _ = pl^
        _ = tl^

    var nthr = blas.serial_begin()
    if nworkers == 1:
        work(0)
    else:
        parallelize(work, nworkers)
    blas.serial_end(nthr)
    _ = counter^
    _ = grid^
    _ = rcut^


def _load_w(wv: F64Ptr, off: Int, v: Int, npt: Int) -> F64V:
    """W weights of block vector ``v`` from ``wv[off:]`` (zero past ``npt``)."""
    if v * W + W <= npt:
        return wv.unsafe_load[width=W](off + v * W)
    var t = F64V(0.0)
    for lane in range(W):
        if v * W + lane < npt:
            t[lane] = wv[unsafe_offset=off + v * W + lane]
    return t


def xc_vmat_core(
    blas: Blas, basis: Basis, ngrid: Int, coords: F64Ptr, kind: Int, nset: Int, wv: F64Ptr, vmat_out: F64Ptr
) raises:
    """vmat_out[s] = sum_p phi(p) [sum_c wv[s][c][p] phi_c(p)]^T (c = 0 for LDA, 0..3 for GGA), plus for
    meta-GGA (``kind`` 2) sum_c d_c phi (wv[s][4] d_c phi)^T; nao x nao, not symmetrised, overwritten."""
    var nderiv = 0 if kind == 0 else 1
    var nw = _nrho(kind)
    var grid = Grid(ngrid, coords)
    var rcut = _rcuts(basis, nderiv)
    var nao = basis.nao
    var n2 = nao * nao
    var ncomp = _ncomp(nderiv)
    var nblk = grid.nblk
    var nworkers = max(1, min(parallelism_level(), nblk))
    var accl = List[Float64](length=nworkers * nset * n2 + 1, fill=0.0)
    var pacc = list_ptr(accl)
    var counter = Atomic[Int64](0)
    var pcount = Pointer(to=counter)

    def work(w: Int) {imm blas, imm basis, imm grid, imm rcut, imm pcount, imm nblk, imm nao, imm n2, imm ncomp, imm nderiv, imm nw, imm kind, imm nset, imm wv, imm pacc, imm ngrid}:
        var ws = AOWork(nao, basis.nbas, ncomp, basis.lmax, basis.nctr_max)
        var acc = pacc.unsafe_offset(w * nset * n2)
        while True:
            var blk = Int(pcount[].fetch_add(1))
            if blk >= nblk:
                break
            var sel = select_shells(basis, grid, blk, rcut, ws)
            var nrow = sel[1]
            if nrow == 0:
                continue
            var p0 = blk * BLK
            var npt = min(BLK, ngrid - p0)
            eval_block(basis, grid, blk, nderiv, sel[0], nrow, ws)
            var cs = nrow * BLK
            for s in range(nset):
                var off = (s * nw) * ngrid + p0
                # Z[i][p] = sum_c wv_c(p) phi_c,i(p) into ws.y (padding points weigh zero)
                for v in range(NV):
                    var w0 = _load_w(wv, off, v, npt)
                    if ncomp == 1:
                        for i in range(nrow):
                            ws.y().unsafe_store(i * BLK + v * W, ws.ao().unsafe_load[width=W](i * BLK + v * W) * w0)
                    else:
                        var w1 = _load_w(wv, off + ngrid, v, npt)
                        var w2 = _load_w(wv, off + 2 * ngrid, v, npt)
                        var w3 = _load_w(wv, off + 3 * ngrid, v, npt)
                        var ax = ws.ao().unsafe_offset(cs)
                        var ay = ws.ao().unsafe_offset(2 * cs)
                        var az = ws.ao().unsafe_offset(3 * cs)
                        for i in range(nrow):
                            var o = i * BLK + v * W
                            var z = ws.ao().unsafe_load[width=W](o) * w0 + ax.unsafe_load[width=W](o) * w1
                            z += ay.unsafe_load[width=W](o) * w2 + az.unsafe_load[width=W](o) * w3
                            ws.y().unsafe_store(o, z)
                try:
                    blas.gemm(False, True, nrow, nrow, BLK, 1.0, ws.ao(), ws.y(), 0.0, ws.dsub())
                except:
                    pass
                if kind == 2:
                    # + sum_c d_c phi (w_tau d_c phi)^T
                    for c in range(1, 4):
                        var ac = ws.ao().unsafe_offset(c * cs)
                        for v in range(NV):
                            var w4 = _load_w(wv, off + 4 * ngrid, v, npt)
                            for i in range(nrow):
                                var o = i * BLK + v * W
                                ws.y().unsafe_store(o, ac.unsafe_load[width=W](o) * w4)
                        try:
                            blas.gemm(False, True, nrow, nrow, BLK, 1.0, ac, ws.y(), 1.0, ws.dsub())
                        except:
                            pass
                scatter_add_block(acc.unsafe_offset(s * n2), nao, nrow, ws)
        _ = ws^

    var nthr = blas.serial_begin()
    if nworkers == 1:
        work(0)
    else:
        parallelize(work, nworkers)
    blas.serial_end(nthr)
    var total = nset * n2
    for i in range(total):
        var v = 0.0
        for w2 in range(nworkers):
            v += pacc[unsafe_offset=w2 * total + i]
        vmat_out[unsafe_offset=i] = v
    _ = accl^
    _ = counter^
    _ = grid^
    _ = rcut^


def xc_grad_core(
    blas: Blas, basis: Basis, ngrid: Int, coords: F64Ptr, gga: Bool, nset: Int, wv: F64Ptr, vmat_out: F64Ptr
) raises:
    """XC gradient matrices vmat_out[s][x] (nao x nao) of pyscf's ``grad.rks.get_vxc`` (before its sign flip).

    LDA (wv[s] = w v_rho): V_x = sum_p d_x phi(p) [w0(p) phi(p)]^T.
    GGA (wv[s][0..3] with w0 halved, as pyscf):
    V_x = sum_p d_x phi(p) Z(p)^T + Y_x(p) phi(p)^T with Z = sum_c w_c phi_c and
    Y_x = w0 d_x phi + sum_i w_i d_i d_x phi.  Overwritten.
    """
    var grid = Grid(ngrid, coords)
    var nderiv = 2 if gga else 1
    var rcut = _rcuts(basis, nderiv)
    var nao = basis.nao
    var n2 = nao * nao
    var ncomp = _ncomp(nderiv)
    var nw = 4 if gga else 1
    var nblk = grid.nblk
    var nworkers = max(1, min(parallelism_level(), nblk))
    var accl = List[Float64](length=nworkers * nset * 3 * n2 + 1, fill=0.0)
    var pacc = list_ptr(accl)
    var counter = Atomic[Int64](0)
    var pcount = Pointer(to=counter)

    def work(w: Int) {imm blas, imm basis, imm grid, imm rcut, imm pcount, imm nblk, imm nao, imm n2, imm ncomp, imm nderiv, imm nw, imm gga, imm nset, imm wv, imm pacc, imm ngrid}:
        var ws = AOWork(nao, basis.nbas, ncomp, basis.lmax, basis.nctr_max)
        var acc = pacc.unsafe_offset(w * nset * 3 * n2)
        while True:
            var blk = Int(pcount[].fetch_add(1))
            if blk >= nblk:
                break
            var sel = select_shells(basis, grid, blk, rcut, ws)
            var nrow = sel[1]
            if nrow == 0:
                continue
            var p0 = blk * BLK
            var npt = min(BLK, ngrid - p0)
            eval_block(basis, grid, blk, nderiv, sel[0], nrow, ws)
            var cs = nrow * BLK
            var ao = ws.ao()
            var zb = ws.y()
            var yb = ws.y2(nrow)
            for s in range(nset):
                var off = s * nw * ngrid + p0
                # Z = sum_c w_c phi_c
                for v in range(NV):
                    var w0 = _load_w(wv, off, v, npt)
                    if not gga:
                        for i in range(nrow):
                            zb.unsafe_store(i * BLK + v * W, ao.unsafe_load[width=W](i * BLK + v * W) * w0)
                    else:
                        var w1 = _load_w(wv, off + ngrid, v, npt)
                        var w2 = _load_w(wv, off + 2 * ngrid, v, npt)
                        var w3 = _load_w(wv, off + 3 * ngrid, v, npt)
                        for i in range(nrow):
                            var o = i * BLK + v * W
                            var z = ao.unsafe_load[width=W](o) * w0 + ao.unsafe_load[width=W](cs + o) * w1
                            z += ao.unsafe_load[width=W](2 * cs + o) * w2 + ao.unsafe_load[width=W](3 * cs + o) * w3
                            zb.unsafe_store(o, z)
                for x in range(3):
                    var dx_ao = ao.unsafe_offset((1 + x) * cs)
                    try:
                        blas.gemm(False, True, nrow, nrow, BLK, 1.0, dx_ao, zb, 0.0, ws.dsub())
                    except:
                        pass
                    if gga:
                        # Y_x = w0 d_x phi + sum_i w_i d_i d_x phi; components xx 4, xy 5, xz 6, yy 7, yz 8, zz 9
                        var c1 = 4 + x
                        var c2 = 5 if x == 0 else (7 if x == 1 else 8)
                        var c3 = 6 if x == 0 else (8 if x == 1 else 9)
                        for v in range(NV):
                            var w0 = _load_w(wv, off, v, npt)
                            var w1 = _load_w(wv, off + ngrid, v, npt)
                            var w2 = _load_w(wv, off + 2 * ngrid, v, npt)
                            var w3 = _load_w(wv, off + 3 * ngrid, v, npt)
                            for i in range(nrow):
                                var o = i * BLK + v * W
                                var y = dx_ao.unsafe_load[width=W](o) * w0 + ao.unsafe_load[width=W](c1 * cs + o) * w1
                                y += ao.unsafe_load[width=W](c2 * cs + o) * w2 + ao.unsafe_load[width=W](c3 * cs + o) * w3
                                yb.unsafe_store(o, y)
                        try:
                            blas.gemm(False, True, nrow, nrow, BLK, 1.0, yb, ao, 1.0, ws.dsub())
                        except:
                            pass
                    scatter_add_block(acc.unsafe_offset((s * 3 + x) * n2), nao, nrow, ws)
        _ = ws^

    var nthr = blas.serial_begin()
    if nworkers == 1:
        work(0)
    else:
        parallelize(work, nworkers)
    blas.serial_end(nthr)
    var total = nset * 3 * n2
    for i in range(total):
        var v = 0.0
        for w2 in range(nworkers):
            v += pacc[unsafe_offset=w2 * total + i]
        vmat_out[unsafe_offset=i] = v
    _ = accl^
    _ = counter^
    _ = grid^
    _ = rcut^


def _orb_apply(blas: Blas, nrow: Int, norb: Int, nk: F64Ptr, src: F64Ptr, tmp: F64Ptr, dst: F64Ptr, ws: AOWork):
    """dst = C_sub diag(n) C_sub^T src (nrow x BLK) with C_sub in ws.dsub (nrow x norb); tmp holds norb x BLK."""
    try:
        blas.gemm(True, False, norb, BLK, nrow, 1.0, ws.dsub(), src, 0.0, tmp)
    except:
        pass
    for k in range(norb):
        var f = nk[unsafe_offset=k]
        for v in range(NV):
            tmp.unsafe_store(k * BLK + v * W, tmp.unsafe_load[width=W](k * BLK + v * W) * f)
    try:
        blas.gemm(False, False, nrow, BLK, norb, 1.0, ws.dsub(), tmp, 0.0, dst)
    except:
        pass


def xc_grad_dm_core(
    blas: Blas, basis: Basis, ngrid: Int, coords: F64Ptr, kind: Int, nset: Int, wv: F64Ptr, dms: F64Ptr,
    norb: Int, orbs: F64Ptr, occs: F64Ptr, de_out: F64Ptr,
) raises:
    """XC term of the nuclear gradient, de_out[A][x] = -2 sum_s sum_{mu on A, nu} D_s,mu nu V_s,x,mu nu
    (natm x 3, overwritten), with V_s,x the matrices of ``xc_grad_core``, without forming them.

    With G0 = D phi and G1 = D Z (one GEMM each per block and set):
    sum_nu D_mu nu V_x,mu nu = sum_p d_x phi_mu G1_mu + Y_x,mu G0_mu
                             = sum_p d_x phi_mu (G1_mu + w0 G0_mu) + G0_mu sum_i w_i d_i d_x phi_mu
    for GGA; for LDA (w0 not halved) sum_p d_x phi_mu w0 G0_mu.  With ``norb`` > 0
    the densities are also given by orbitals (as in ``xc_rho_core``) and
    blocks where it is cheaper form G0 = C (n C^T phi) and G1 = C (n C^T Z).
    Meta-GGA (``kind`` 2, wv[4] the tau weight halved as in pyscf) adds
    sum_c sum_p d_x d_c phi_mu w_4 (D d_c phi)_mu (pyscf's ``_tau_grad_dot_``).
    """
    var gga = kind >= 1
    var grid = Grid(ngrid, coords)
    var nderiv = 2 if gga else 1
    var rcut = _rcuts(basis, nderiv)
    var nao = basis.nao
    var n2 = nao * nao
    var natm = basis.natm
    var ncomp = _ncomp(nderiv)
    var nw = _nrho(kind)
    var nblk = grid.nblk
    var nworkers = max(1, min(parallelism_level(), nblk))
    var accl = List[Float64](length=nworkers * natm * 3 + 1, fill=0.0)
    var pacc = list_ptr(accl)
    var counter = Atomic[Int64](0)
    var pcount = Pointer(to=counter)

    def work(w: Int) {imm blas, imm basis, imm grid, imm rcut, imm pcount, imm nblk, imm nao, imm n2, imm ncomp, imm nderiv, imm nw, imm gga, imm kind, imm nset, imm wv, imm dms, imm norb, imm orbs, imm occs, imm pacc, imm ngrid}:
        var ws = AOWork(nao, basis.nbas, ncomp, basis.lmax, basis.nctr_max)
        var acc = pacc.unsafe_offset(w * basis.natm * 3)
        var tl = List[Float64](length=max(norb, 1) * BLK + W, fill=0.0)
        var tmp = F64Ptr(unsafe_from_address=aligned_addr(tl))
        var wl = List[Float64](length=5 * BLK + W, fill=0.0)    # the block's weights, zero-padded
        var wb = list_ptr(wl)
        var rowatm = List[Int](length=max(nao, 1), fill=0)
        while True:
            var blk = Int(pcount[].fetch_add(1))
            if blk >= nblk:
                break
            var sel = select_shells(basis, grid, blk, rcut, ws)
            var nrow = sel[1]
            if nrow == 0:
                continue
            var r = 0
            for si in range(sel[0]):
                var b = ws.shells[si]
                for _ in range(basis.ao_loc[b + 1] - basis.ao_loc[b]):
                    rowatm[r] = basis.atom[b]
                    r += 1
            var p0 = blk * BLK
            var npt = min(BLK, ngrid - p0)
            eval_block(basis, grid, blk, nderiv, sel[0], nrow, ws)
            var cs = nrow * BLK
            var ao = ws.ao()
            var zb = ws.y()
            var g0 = ws.y2(nrow)
            var g1 = ws.y3(nrow)
            # orbitals when their GEMMs (two per product) are cheaper than D_sub times phi and Z
            var use_orb = norb > 0 and 2 * norb * 10 < nrow * 9
            for s in range(nset):
                var off = s * nw * ngrid + p0
                for c in range(nw):
                    for v in range(NV):
                        wb.unsafe_store(c * BLK + v * W, _load_w(wv, off + c * ngrid, v, npt))
                var nk = occs.unsafe_offset(s * norb)
                if use_orb:
                    gather_rows(orbs.unsafe_offset(s * nao * norb), norb, ws)
                    _orb_apply(blas, nrow, norb, nk, ao, tmp, g0, ws)
                else:
                    gather_block(dms.unsafe_offset(s * n2), nao, nrow, ws)
                    try:
                        blas.gemm(False, False, nrow, BLK, nrow, 1.0, ws.dsub(), ao, 0.0, g0)
                    except:
                        pass
                if not gga:
                    for i in range(nrow):
                        var tx = F64V(0.0)
                        var ty = F64V(0.0)
                        var tz = F64V(0.0)
                        for v in range(NV):
                            var o = i * BLK + v * W
                            var h = g0.unsafe_load[width=W](o) * wb.unsafe_load[width=W](v * W)
                            tx += ao.unsafe_load[width=W](cs + o) * h
                            ty += ao.unsafe_load[width=W](2 * cs + o) * h
                            tz += ao.unsafe_load[width=W](3 * cs + o) * h
                        var a = rowatm[i]
                        acc[unsafe_offset=3 * a] -= 2.0 * tx.reduce_add()
                        acc[unsafe_offset=3 * a + 1] -= 2.0 * ty.reduce_add()
                        acc[unsafe_offset=3 * a + 2] -= 2.0 * tz.reduce_add()
                    continue
                # Z = sum_c w_c phi_c, G1 = D Z
                for i in range(nrow):
                    for v in range(NV):
                        var o = i * BLK + v * W
                        var z = ao.unsafe_load[width=W](o) * wb.unsafe_load[width=W](v * W)
                        z += ao.unsafe_load[width=W](cs + o) * wb.unsafe_load[width=W](BLK + v * W)
                        z += ao.unsafe_load[width=W](2 * cs + o) * wb.unsafe_load[width=W](2 * BLK + v * W)
                        z += ao.unsafe_load[width=W](3 * cs + o) * wb.unsafe_load[width=W](3 * BLK + v * W)
                        zb.unsafe_store(o, z)
                if use_orb:
                    _orb_apply(blas, nrow, norb, nk, zb, tmp, g1, ws)
                else:
                    try:
                        blas.gemm(False, False, nrow, BLK, nrow, 1.0, ws.dsub(), zb, 0.0, g1)
                    except:
                        pass
                # second derivatives: xx 4, xy 5, xz 6, yy 7, yz 8, zz 9
                for i in range(nrow):
                    var tx = F64V(0.0)
                    var ty = F64V(0.0)
                    var tz = F64V(0.0)
                    for v in range(NV):
                        var o = i * BLK + v * W
                        var w0 = wb.unsafe_load[width=W](v * W)
                        var w1 = wb.unsafe_load[width=W](BLK + v * W)
                        var w2 = wb.unsafe_load[width=W](2 * BLK + v * W)
                        var w3 = wb.unsafe_load[width=W](3 * BLK + v * W)
                        var gg0 = g0.unsafe_load[width=W](o)
                        var h = g1.unsafe_load[width=W](o) + w0 * gg0
                        var w1g = w1 * gg0
                        var w2g = w2 * gg0
                        var w3g = w3 * gg0
                        var dxx = ao.unsafe_load[width=W](4 * cs + o)
                        var dxy = ao.unsafe_load[width=W](5 * cs + o)
                        var dxz = ao.unsafe_load[width=W](6 * cs + o)
                        var dyy = ao.unsafe_load[width=W](7 * cs + o)
                        var dyz = ao.unsafe_load[width=W](8 * cs + o)
                        var dzz = ao.unsafe_load[width=W](9 * cs + o)
                        tx += ao.unsafe_load[width=W](cs + o) * h + dxx * w1g + dxy * w2g + dxz * w3g
                        ty += ao.unsafe_load[width=W](2 * cs + o) * h + dxy * w1g + dyy * w2g + dyz * w3g
                        tz += ao.unsafe_load[width=W](3 * cs + o) * h + dxz * w1g + dyz * w2g + dzz * w3g
                    var a = rowatm[i]
                    acc[unsafe_offset=3 * a] -= 2.0 * tx.reduce_add()
                    acc[unsafe_offset=3 * a + 1] -= 2.0 * ty.reduce_add()
                    acc[unsafe_offset=3 * a + 2] -= 2.0 * tz.reduce_add()
                if kind == 2:
                    # tau: G_c = D d_c phi into g1; d_x d_c phi components (xx 4, xy 5, xz 6, yy 7, yz 8, zz 9)
                    for c in range(1, 4):
                        var ac = ao.unsafe_offset(c * cs)
                        if use_orb:
                            _orb_apply(blas, nrow, norb, nk, ac, tmp, g1, ws)
                        else:
                            try:
                                blas.gemm(False, False, nrow, BLK, nrow, 1.0, ws.dsub(), ac, 0.0, g1)
                            except:
                                pass
                        var kx = 3 + c
                        var ky = 5 if c == 1 else (7 if c == 2 else 8)
                        var kz = 6 if c == 1 else (8 if c == 2 else 9)
                        for i in range(nrow):
                            var tx = F64V(0.0)
                            var ty = F64V(0.0)
                            var tz = F64V(0.0)
                            for v in range(NV):
                                var o = i * BLK + v * W
                                var h = g1.unsafe_load[width=W](o) * wb.unsafe_load[width=W](4 * BLK + v * W)
                                tx += ao.unsafe_load[width=W](kx * cs + o) * h
                                ty += ao.unsafe_load[width=W](ky * cs + o) * h
                                tz += ao.unsafe_load[width=W](kz * cs + o) * h
                            var a = rowatm[i]
                            acc[unsafe_offset=3 * a] -= 2.0 * tx.reduce_add()
                            acc[unsafe_offset=3 * a + 1] -= 2.0 * ty.reduce_add()
                            acc[unsafe_offset=3 * a + 2] -= 2.0 * tz.reduce_add()
        _ = ws^
        _ = wl^
        _ = tl^
        _ = rowatm^

    var nthr = blas.serial_begin()
    if nworkers == 1:
        work(0)
    else:
        parallelize(work, nworkers)
    blas.serial_end(nthr)
    for i in range(natm * 3):
        var v = 0.0
        for w2 in range(nworkers):
            v += pacc[unsafe_offset=w2 * natm * 3 + i]
        de_out[unsafe_offset=i] = v
    _ = accl^
    _ = counter^
    _ = grid^
    _ = rcut^


def xc_fxc_core(
    blas: Blas,
    basis: Basis,
    ngrid: Int,
    coords: F64Ptr,
    weights: F64Ptr,
    kind: Int,
    nspin: Int,
    fxc: F64Ptr,
    nset: Int,
    dms: F64Ptr,
    rank: Int,
    lfac: F64Ptr,
    rfac: F64Ptr,
    project: Bool,
    vmat_out: F64Ptr,
) raises:
    """The XC kernel contracted with first-order densities, in one pass (TDDFT, CPHF response).

    For every set s and spin a, rho1_{a,s} (the density, and for GGA and
    meta-GGA its gradient and tau, as ``xc_rho_core``) is that of the symmetric
    dms[a][s] (nspin x nset x nao x nao).  With ``rank`` > 0 the same density
    is also the symmetric part of L_{a,s} R_a^T (lfac: nspin x nset x nao x
    rank, rfac: nspin x nao x rank; transition densities C_v z^T C_o^T), and
    blocks where that is cheaper use the factors: with A = L_sub^T phi and
    B = R_sub^T phi (shared by all sets), rho = sum_k A_k B_k (LDA), or
    Y = D phi = (L_sub B + R_sub A) / 2 (GGA; three GEMMs of rank x nrow
    instead of one of nrow x nrow) and tau = 1/2 sum_c sum_k (L^T d_c phi)_k
    (R^T d_c phi)_k (meta-GGA).  On each block the response weights are
    w_(b,y) = weight sum_(a,x) fxc[(a,x)][(b,y)] rho1_(a,x) (fxc:
    (nspin nvar) x (nspin nvar) x ngrid, pyscf's layout), with w_0 halved and
    w_tau quartered as for the potential, and vmat_out[b][s] += phi Z^T (and the
    tau term) as in ``xc_vmat_core``: not symmetrised, overwritten.  The
    basis functions are evaluated once per block for both halves.

    With ``project`` (``rank`` > 0) only V_{b,s} R_b is formed, V the
    symmetrised matrix: vmat_out is (nspin x nset x nao x rank) and on a
    block (V + V^T)_sub R_sub = phi U^T + Z B_0^T (+ 2 sum_c (w_tau d_c phi)
    B_c^T) with U = sum_c w_c B_c and B_c = R_sub^T d_c phi shared by all
    sets: two GEMMs of nrow x rank instead of one of nrow x nrow, and the
    per-thread sums are nao x rank.  The linear-response operators need no
    more (C_v^T V C_o).
    """
    var nderiv = 0 if kind == 0 else 1
    var nvar = _nrho(kind)
    var nv2 = nspin * nvar
    var nb = 4 if kind == 2 else 1             # components of B (and A) kept per spin
    var grid = Grid(ngrid, coords)
    var rcut = _rcuts(basis, nderiv)
    var nao = basis.nao
    var n2 = nao * nao
    var ncomp = _ncomp(nderiv)
    var nblk = grid.nblk
    var nworkers = max(1, min(parallelism_level(), nblk))
    var total = nspin * nset * (nao * rank if project else n2)
    var nbc = (1 if kind == 0 else 4) if project else nb     # components of B computed per spin
    var accl = List[Float64](length=nworkers * total + 1, fill=0.0)
    var pacc = list_ptr(accl)
    var counter = Atomic[Int64](0)
    var pcount = Pointer(to=counter)
    var rk = max(rank, 1)

    def work(w: Int) {imm blas, imm basis, imm grid, imm rcut, imm pcount, imm nblk, imm nao, imm n2, imm ncomp, imm nderiv, imm nvar, imm nv2, imm nb, imm kind, imm nspin, imm nset, imm fxc, imm weights, imm dms, imm rank, imm rk, imm lfac, imm rfac, imm pacc, imm total, imm ngrid, imm project, imm nbc}:
        var ws = AOWork(nao, basis.nbas, ncomp, basis.lmax, basis.nctr_max)
        var acc = pacc.unsafe_offset(w * total)
        var n_rsub = nspin * nao * rk
        var n_b = nspin * 4 * rk * BLK
        var n_lsub = nao * rk
        var n_a = nb * rk * BLK
        var n_r1 = nv2 * BLK
        var n_u = rk * BLK
        var n_wb = nao * rk
        var scratch = List[Float64](length=n_rsub + n_b + n_lsub + n_a + 2 * n_r1 + BLK + n_u + n_wb + 10 * W, fill=0.0)
        var base = F64Ptr(unsafe_from_address=aligned_addr(scratch))
        var rsub = base
        var bbuf = rsub.unsafe_offset((n_rsub + W - 1) // W * W)
        var lsub = bbuf.unsafe_offset((n_b + W - 1) // W * W)
        var abuf = lsub.unsafe_offset((n_lsub + W - 1) // W * W)
        var rho1 = abuf.unsafe_offset((n_a + W - 1) // W * W)
        var wvb = rho1.unsafe_offset(n_r1)
        var tau = wvb.unsafe_offset(n_r1)
        var ubuf = tau.unsafe_offset(BLK)
        var wblk = ubuf.unsafe_offset(n_u)
        while True:
            var blk = Int(pcount[].fetch_add(1))
            if blk >= nblk:
                break
            var sel = select_shells(basis, grid, blk, rcut, ws)
            var nrow = sel[1]
            if nrow == 0:
                continue
            var p0 = blk * BLK
            var npt = min(BLK, ngrid - p0)
            eval_block(basis, grid, blk, nderiv, sel[0], nrow, ws)
            var cs = nrow * BLK
            var use_lr = False
            if rank > 0:
                if kind == 0:
                    use_lr = rank * 10 < nrow * 9
                elif kind == 1:
                    use_lr = 3 * rank * 10 < nrow * 9
                else:
                    use_lr = 6 * rank * 10 < 4 * nrow * 9
            if use_lr or project:
                for a in range(nspin):
                    var ra = rsub.unsafe_offset(a * nao * rk)
                    gather_rows_to(rfac.unsafe_offset(a * nao * rank), rank, ws, ra)
                    for c in range(nbc if project else nb):
                        try:
                            blas.gemm(True, False, rank, BLK, nrow, 1.0, ra, ws.ao().unsafe_offset(c * cs), 0.0,
                                      bbuf.unsafe_offset((a * 4 + c) * rk * BLK))
                        except:
                            pass
            for s in range(nset):
                # ---- first-order densities of every spin on this block
                for a in range(nspin):
                    var r1 = rho1.unsafe_offset(a * nvar * BLK)
                    var y = ws.y()
                    if use_lr:
                        var ra = rsub.unsafe_offset(a * nao * rk)
                        var b0 = bbuf.unsafe_offset(a * 4 * rk * BLK)
                        gather_rows_to(lfac.unsafe_offset((a * nset + s) * nao * rank), rank, ws, lsub)
                        for c in range(nb):
                            try:
                                blas.gemm(True, False, rank, BLK, nrow, 1.0, lsub, ws.ao().unsafe_offset(c * cs), 0.0,
                                          abuf.unsafe_offset(c * rk * BLK))
                            except:
                                pass
                        if kind == 0:
                            for v in range(NV):
                                var t = F64V(0.0)
                                for k in range(rank):
                                    t += abuf.unsafe_load[width=W](k * BLK + v * W) * b0.unsafe_load[width=W](k * BLK + v * W)
                                r1.unsafe_store(v * W, t)
                            continue
                        try:
                            blas.gemm(False, False, nrow, BLK, rank, 0.5, lsub, b0, 0.0, y)
                            blas.gemm(False, False, nrow, BLK, rank, 0.5, ra, abuf, 1.0, y)
                        except:
                            pass
                    else:
                        gather_block(dms.unsafe_offset((a * nset + s) * n2), nao, nrow, ws)
                        try:
                            blas.gemm(False, False, nrow, BLK, nrow, 1.0, ws.dsub(), ws.ao(), 0.0, y)
                        except:
                            pass
                    for c in range(ncomp):
                        var fac = 1.0 if c == 0 else 2.0
                        var ac = ws.ao().unsafe_offset(c * cs)
                        for v in range(NV):
                            var t = F64V(0.0)
                            for i in range(nrow):
                                t += ac.unsafe_load[width=W](i * BLK + v * W) * y.unsafe_load[width=W](i * BLK + v * W)
                            r1.unsafe_store(c * BLK + v * W, t * fac)
                    if kind == 2:
                        vfill(tau, BLK, 0.0)
                        if use_lr:
                            var b0 = bbuf.unsafe_offset(a * 4 * rk * BLK)
                            for c in range(1, 4):
                                var ab = abuf.unsafe_offset(c * rk * BLK)
                                var bb = b0.unsafe_offset(c * rk * BLK)
                                for v in range(NV):
                                    var t = tau.unsafe_load[width=W](v * W)
                                    for k in range(rank):
                                        t += ab.unsafe_load[width=W](k * BLK + v * W) * bb.unsafe_load[width=W](k * BLK + v * W)
                                    tau.unsafe_store(v * W, t)
                        else:
                            var yc = ws.y2(nrow)
                            for c in range(1, 4):
                                var ac = ws.ao().unsafe_offset(c * cs)
                                try:
                                    blas.gemm(False, False, nrow, BLK, nrow, 1.0, ws.dsub(), ac, 0.0, yc)
                                except:
                                    pass
                                for v in range(NV):
                                    var t = tau.unsafe_load[width=W](v * W)
                                    for i in range(nrow):
                                        t += ac.unsafe_load[width=W](i * BLK + v * W) * yc.unsafe_load[width=W](i * BLK + v * W)
                                    tau.unsafe_store(v * W, t)
                        for v in range(NV):
                            r1.unsafe_store(4 * BLK + v * W, tau.unsafe_load[width=W](v * W) * 0.5)
                # ---- response weights w_(b,y) = weight sum_(a,x) fxc[(a,x)][(b,y)] rho1_(a,x)
                for v in range(NV):
                    var wt = _load_w(weights, p0, v, npt)
                    for j in range(nv2):
                        var t = F64V(0.0)
                        for i in range(nv2):
                            t += _load_w(fxc, (i * nv2 + j) * ngrid + p0, v, npt) * rho1.unsafe_load[width=W](i * BLK + v * W)
                        t *= wt
                        var var_j = j % nvar
                        if var_j == 0:
                            t *= 0.5
                        elif var_j == 4:
                            t *= 0.25
                        wvb.unsafe_store(j * BLK + v * W, t)
                # ---- potential matrices of every spin
                for b in range(nspin):
                    var wb = wvb.unsafe_offset(b * nvar * BLK)
                    var z = ws.y()
                    if project:
                        var bb = bbuf.unsafe_offset(b * 4 * rk * BLK)
                        # U = sum_c w_c B_c (rank x BLK)
                        for k in range(rank):
                            for v in range(NV):
                                var o = k * BLK + v * W
                                var t = wb.unsafe_load[width=W](v * W) * bb.unsafe_load[width=W](o)
                                if kind > 0:
                                    for c in range(1, 4):
                                        t += wb.unsafe_load[width=W](c * BLK + v * W) * bb.unsafe_load[width=W](c * rk * BLK + o)
                                ubuf.unsafe_store(o, t)
                    for v in range(NV):
                        var w0 = wb.unsafe_load[width=W](v * W)
                        if ncomp == 1:
                            for i in range(nrow):
                                z.unsafe_store(i * BLK + v * W, ws.ao().unsafe_load[width=W](i * BLK + v * W) * w0)
                        else:
                            var w1 = wb.unsafe_load[width=W](BLK + v * W)
                            var w2 = wb.unsafe_load[width=W](2 * BLK + v * W)
                            var w3 = wb.unsafe_load[width=W](3 * BLK + v * W)
                            var ax = ws.ao().unsafe_offset(cs)
                            var ay = ws.ao().unsafe_offset(2 * cs)
                            var az = ws.ao().unsafe_offset(3 * cs)
                            for i in range(nrow):
                                var o = i * BLK + v * W
                                var t = ws.ao().unsafe_load[width=W](o) * w0 + ax.unsafe_load[width=W](o) * w1
                                t += ay.unsafe_load[width=W](o) * w2 + az.unsafe_load[width=W](o) * w3
                                z.unsafe_store(o, t)
                    if project:
                        # (V + V^T)_sub R_sub = phi U^T + Z B_0^T (+ 2 sum_c (w_tau d_c phi) B_c^T)
                        var bb = bbuf.unsafe_offset(b * 4 * rk * BLK)
                        try:
                            blas.gemm(False, True, nrow, rank, BLK, 1.0, ws.ao(), ubuf, 0.0, wblk)
                            blas.gemm(False, True, nrow, rank, BLK, 1.0, z, bb, 1.0, wblk)
                        except:
                            pass
                        if kind == 2:
                            for c in range(1, 4):
                                var ac = ws.ao().unsafe_offset(c * cs)
                                for v in range(NV):
                                    var w4 = wb.unsafe_load[width=W](4 * BLK + v * W)
                                    for i in range(nrow):
                                        var o = i * BLK + v * W
                                        z.unsafe_store(o, ac.unsafe_load[width=W](o) * w4)
                                try:
                                    blas.gemm(False, True, nrow, rank, BLK, 2.0, z, bb.unsafe_offset(c * rk * BLK), 1.0, wblk)
                                except:
                                    pass
                        var dst = acc.unsafe_offset((b * nset + s) * nao * rank)
                        for ri in range(ws.nrun):
                            var d = dst.unsafe_offset(ws.run_ao[ri] * rank)
                            var src = wblk.unsafe_offset(ws.run_row[ri] * rank)
                            var n = ws.run_len[ri] * rank
                            var k = 0
                            while k + W <= n:
                                d.unsafe_store(k, d.unsafe_load[width=W](k) + src.unsafe_load[width=W](k))
                                k += W
                            while k < n:
                                d[unsafe_offset=k] += src[unsafe_offset=k]
                                k += 1
                        continue
                    try:
                        blas.gemm(False, True, nrow, nrow, BLK, 1.0, ws.ao(), z, 0.0, ws.dsub())
                    except:
                        pass
                    if kind == 2:
                        for c in range(1, 4):
                            var ac = ws.ao().unsafe_offset(c * cs)
                            for v in range(NV):
                                var w4 = wb.unsafe_load[width=W](4 * BLK + v * W)
                                for i in range(nrow):
                                    var o = i * BLK + v * W
                                    z.unsafe_store(o, ac.unsafe_load[width=W](o) * w4)
                            try:
                                blas.gemm(False, True, nrow, nrow, BLK, 1.0, ac, z, 1.0, ws.dsub())
                            except:
                                pass
                    scatter_add_block(acc.unsafe_offset((b * nset + s) * n2), nao, nrow, ws)
        _ = ws^
        _ = scratch^

    var nthr = blas.serial_begin()
    if nworkers == 1:
        work(0)
    else:
        parallelize(work, nworkers)
    blas.serial_end(nthr)
    for i in range(total):
        var v = 0.0
        for w2 in range(nworkers):
            v += pacc[unsafe_offset=w2 * total + i]
        vmat_out[unsafe_offset=i] = v
    _ = accl^
    _ = counter^
    _ = grid^
    _ = rcut^


def _didx(px: Int, py: Int, pz: Int) -> Int:
    """AO component index of the derivative d^px d^py d^pz (orders up to 3, pyscf's order)."""
    var code = px * 100 + py * 10 + pz
    for comp in range(20):
        var t = _comp_powers(comp)
        if t[0] * 100 + t[1] * 10 + t[2] == code:
            return comp
    return -1


def xc_hess_core(
    blas: Blas,
    basis: Basis,
    ngrid: Int,
    coords: F64Ptr,
    weights: F64Ptr,
    kind: Int,
    nspin: Int,
    dms: F64Ptr,
    vxc: F64Ptr,
    fxc: F64Ptr,
    aoatm: Pointer[Int64, MutAnyOrigin],
    natm: Int,
    de2_out: F64Ptr,
) raises:
    """XC part of the nuclear Hessian at fixed density matrix (pyscf's ``_get_vxc_diag`` and
    ``_get_vxc_deriv2`` contracted with the density, no grid response), LDA (``kind`` 0) and GGA (1).

    dms: nspin x nao x nao symmetric (RKS: the total density, UKS: alpha and beta);
    vxc: (nspin nvar) x ngrid and fxc: (nspin nvar) x (nspin nvar) x ngrid, pyscf's
    eval_xc_eff layout (unweighted); aoatm: atom of every AO.  With Y_c = D phi_c,
    v = w vxc and rho1_{A,x} = -d rho / d R_{A,x} (2 sum_{q in A} (d_x phi_q Y_0,q) for
    rho; GGA: 2 sum_{q in A} (d_x d_c phi_q Y_0,q + d_x phi_q Y_c,q) for d_c rho):

        de2[A,A][x,y] += 2 sum_{p in A} int [d_x d_y phi_p (v_0 Y_0 + sum_c v_c Y_c)_p
                                              + sum_c v_c d_x d_y d_c phi_p Y_0,p]
        de2[A,B][x,y] += 2 sum_{s in A, r in B} D_sr int [d_x phi_s G_y,r + H_x,s d_y phi_r]
                         G_y = v_0 d_y phi + sum_c v_c d_y d_c phi,  H_x = sum_c v_c d_x d_c phi
        de2[A,B][x,y] += int w rho1_{A,x} . fxc . rho1_{B,y}

    per spin (the fxc term over both).  The basis functions need derivatives up
    to order 2 (LDA) or 3 (GGA).  For every atom B on a block one GEMM
    (D_sub[:, B] [G | d phi]_B) gives the cross term; the fxc term is one
    GEMM over the (atom, direction) pairs of the block.  de2_out: natm x natm x 3 x 3,
    overwritten.
    """
    var gga = kind == 1
    var nderiv = 3 if gga else 2
    var nvar = 4 if gga else 1
    var nv2 = nspin * nvar
    var ncy = 4 if gga else 1                  # Y_c computed per spin
    var ngb = 6 if gga else 3                  # [G_x G_y G_z (d_x d_y d_z phi)] per row
    var grid = Grid(ngrid, coords)
    var rcut = _rcuts(basis, nderiv)
    var nao = basis.nao
    var n2 = nao * nao
    var ncomp = _ncomp(nderiv)
    var nblk = grid.nblk
    var nworkers = max(1, min(parallelism_level(), nblk))
    var nde = natm * natm * 9
    var accl = List[Float64](length=nworkers * nde + 1, fill=0.0)
    var pacc = list_ptr(accl)
    var counter = Atomic[Int64](0)
    var pcount = Pointer(to=counter)
    # derivative component tables
    var d2t = List[Int](length=9, fill=0)
    var d3t = List[Int](length=27, fill=0)
    for x in range(3):
        for y in range(3):
            var px = (1 if x == 0 else 0) + (1 if y == 0 else 0)
            var py = (1 if x == 1 else 0) + (1 if y == 1 else 0)
            var pz = (1 if x == 2 else 0) + (1 if y == 2 else 0)
            d2t[x * 3 + y] = _didx(px, py, pz)
            for c in range(3):
                d3t[(x * 3 + y) * 3 + c] = _didx(px + (1 if c == 0 else 0), py + (1 if c == 1 else 0), pz + (1 if c == 2 else 0))
    var pd2 = Pointer[Int, MutAnyOrigin](unsafe_from_address=Int(d2t.unsafe_ptr()))
    var pd3 = Pointer[Int, MutAnyOrigin](unsafe_from_address=Int(d3t.unsafe_ptr()))

    def work(w: Int) {imm blas, imm basis, imm grid, imm rcut, imm pcount, imm nblk, imm nao, imm n2, imm ncomp, imm nderiv, imm gga, imm nvar, imm nv2, imm ncy, imm ngb, imm nspin, imm dms, imm vxc, imm fxc, imm weights, imm aoatm, imm natm, imm pacc, imm nde, imm ngrid, imm pd2, imm pd3}:
        var ws = AOWork(nao, basis.nbas, ncomp, basis.lmax, basis.nctr_max)
        var acc = pacc.unsafe_offset(w * nde)
        var n_y = nspin * ncy * nao * BLK
        var n_gb = nao * ngb * BLK
        var n_r1 = natm * 3 * nv2 * BLK
        var n_h = 3 * nao * BLK
        var scratch = List[Float64](length=n_y + 2 * n_gb + n_h + 2 * n_r1 + nv2 * BLK + 9 * natm * natm + 8 * W, fill=0.0)
        var ybuf = F64Ptr(unsafe_from_address=aligned_addr(scratch))
        var gbuf = ybuf.unsafe_offset(n_y)
        var zbuf = gbuf.unsafe_offset(n_gb)
        var hbuf = zbuf.unsafe_offset(n_gb)              # H_x per row (3 x BLK), GGA
        var r1 = hbuf.unsafe_offset(n_h)
        var ub = r1.unsafe_offset(n_r1)
        var vb = ub.unsafe_offset(n_r1)                  # v = w vxc of the block (nv2 x BLK)
        var mm = vb.unsafe_offset(nv2 * BLK)             # (3 na) x (3 na) pair matrix
        var alist = List[Int](length=natm + 1, fill=0)   # atoms of the block
        var a0l = List[Int](length=natm + 1, fill=0)     # their local row ranges
        var a1l = List[Int](length=natm + 1, fill=0)
        while True:
            var blk = Int(pcount[].fetch_add(1))
            if blk >= nblk:
                break
            var sel = select_shells(basis, grid, blk, rcut, ws)
            var nrow = sel[1]
            if nrow == 0:
                continue
            var p0 = blk * BLK
            var npt = min(BLK, ngrid - p0)
            eval_block(basis, grid, blk, nderiv, sel[0], nrow, ws)
            var cs = nrow * BLK
            var ao = ws.ao()
            # atoms on the block and their (contiguous) local rows
            var na = 0
            for r in range(nrow):
                var a = Int(aoatm[unsafe_offset=ws.rows[r]])
                if na == 0 or alist[na - 1] != a:
                    alist[na] = a
                    a0l[na] = r
                    na += 1
                a1l[na - 1] = r + 1
            # weighted potential on the block
            for i in range(nv2):
                for v in range(NV):
                    vb.unsafe_store(i * BLK + v * W, _load_w(vxc, i * ngrid + p0, v, npt) * _load_w(weights, p0, v, npt))
            vfill(r1, na * 3 * nv2 * BLK, 0.0)
            for s in range(nspin):
                var vs = vb.unsafe_offset(s * nvar * BLK)
                var ys = ybuf.unsafe_offset(s * ncy * nao * BLK)
                gather_block(dms.unsafe_offset(s * n2), nao, nrow, ws)
                for c in range(ncy):
                    try:
                        blas.gemm(False, False, nrow, BLK, nrow, 1.0, ws.dsub(), ao.unsafe_offset(c * cs), 0.0, ys.unsafe_offset(c * nrow * BLK))
                    except:
                        pass
                var y0 = ys
                # ---- diagonal term and rho1 per row
                for ia in range(na):
                    var a = alist[ia]
                    var dd = acc.unsafe_offset((a * natm + a) * 9)
                    for p in range(a0l[ia], a1l[ia]):
                        var o = p * BLK
                        for x in range(3):
                            for y in range(x, 3):
                                var hxy = ao.unsafe_offset(pd2[unsafe_offset=x * 3 + y] * cs)
                                var t = F64V(0.0)
                                for v in range(NV):
                                    var q = v * W
                                    var tt = vs.unsafe_load[width=W](q) * y0.unsafe_load[width=W](o + q)
                                    if gga:
                                        for c in range(3):
                                            tt += vs.unsafe_load[width=W]((c + 1) * BLK + q) * ys.unsafe_load[width=W]((c + 1) * nrow * BLK + o + q)
                                    var u = hxy.unsafe_load[width=W](o + q) * tt
                                    if gga:
                                        for c in range(3):
                                            var h3 = ao.unsafe_offset(pd3[unsafe_offset=(x * 3 + y) * 3 + c] * cs)
                                            u += vs.unsafe_load[width=W]((c + 1) * BLK + q) * h3.unsafe_load[width=W](o + q) * y0.unsafe_load[width=W](o + q)
                                    t += u
                                var sxy = 2.0 * t.reduce_add()
                                dd[unsafe_offset=x * 3 + y] += sxy
                                if y != x:
                                    dd[unsafe_offset=y * 3 + x] += sxy
                        # rho1_{A,x} (= -d rho / d R_{A,x}) of this spin
                        for x in range(3):
                            var dx = ao.unsafe_offset((1 + x) * cs)
                            var rr = r1.unsafe_offset(((ia * 3 + x) * nv2 + s * nvar) * BLK)
                            for v in range(NV):
                                var q = v * W
                                var dxp = dx.unsafe_load[width=W](o + q)
                                var y0p = y0.unsafe_load[width=W](o + q)
                                rr.unsafe_store(q, rr.unsafe_load[width=W](q) + dxp * y0p * 2.0)
                                if gga:
                                    for c in range(3):
                                        var hxc = ao.unsafe_offset(pd2[unsafe_offset=x * 3 + c] * cs)
                                        var t = hxc.unsafe_load[width=W](o + q) * y0p + dxp * ys.unsafe_load[width=W]((c + 1) * nrow * BLK + o + q)
                                        rr.unsafe_store((c + 1) * BLK + q, rr.unsafe_load[width=W]((c + 1) * BLK + q) + t * 2.0)
                # ---- cross term: G_y (and d_y phi) per row
                for r in range(nrow):
                    var o = r * BLK
                    var g = gbuf.unsafe_offset(r * ngb * BLK)
                    for y in range(3):
                        var dy = ao.unsafe_offset((1 + y) * cs)
                        for v in range(NV):
                            var q = v * W
                            var t = vs.unsafe_load[width=W](q) * dy.unsafe_load[width=W](o + q)
                            if gga:
                                for c in range(3):
                                    var hyc = ao.unsafe_offset(pd2[unsafe_offset=y * 3 + c] * cs)
                                    t += vs.unsafe_load[width=W]((c + 1) * BLK + q) * hyc.unsafe_load[width=W](o + q)
                            g.unsafe_store(y * BLK + q, t)
                            if gga:
                                g.unsafe_store((3 + y) * BLK + q, dy.unsafe_load[width=W](o + q))
                    if gga:
                        var h = hbuf.unsafe_offset(r * 3 * BLK)
                        for x in range(3):
                            for v in range(NV):
                                var q = v * W
                                var t = F64V(0.0)
                                for c in range(3):
                                    var hxc = ao.unsafe_offset(pd2[unsafe_offset=x * 3 + c] * cs)
                                    t += vs.unsafe_load[width=W]((c + 1) * BLK + q) * hxc.unsafe_load[width=W](o + q)
                                h.unsafe_store(x * BLK + q, t)
                for ib in range(na):
                    var b = alist[ib]
                    var b0 = a0l[ib]
                    var nb = a1l[ib] - b0
                    # Z (nrow x ngb BLK) = D_sub[:, B] [G | d phi]_B  (D symmetric: rows B of D_sub, transposed)
                    try:
                        blas.gemm(True, False, nrow, ngb * BLK, nb, 1.0, ws.dsub().unsafe_offset(b0 * nrow), gbuf.unsafe_offset(b0 * ngb * BLK), 0.0, zbuf)
                    except:
                        pass
                    for ia in range(na):
                        var a = alist[ia]
                        var dd = acc.unsafe_offset((a * natm + b) * 9)
                        for x in range(3):
                            var dx = ao.unsafe_offset((1 + x) * cs)
                            for y in range(3):
                                var t = F64V(0.0)
                                for sr in range(a0l[ia], a1l[ia]):
                                    var o = sr * BLK
                                    var z = zbuf.unsafe_offset(sr * ngb * BLK)
                                    var hx = hbuf.unsafe_offset((sr * 3 + x) * BLK)
                                    for v in range(NV):
                                        var q = v * W
                                        t += dx.unsafe_load[width=W](o + q) * z.unsafe_load[width=W](y * BLK + q)
                                        if gga:
                                            t += hx.unsafe_load[width=W](q) * z.unsafe_load[width=W]((3 + y) * BLK + q)
                                dd[unsafe_offset=x * 3 + y] += 2.0 * t.reduce_add()
            # ---- fxc term: U = (w fxc) rho1, M = rho1 U^T over the (atom, direction) pairs
            for k in range(na * 3):
                var rk = r1.unsafe_offset(k * nv2 * BLK)
                var uk = ub.unsafe_offset(k * nv2 * BLK)
                for j in range(nv2):
                    for v in range(NV):
                        var q = v * W
                        var t = F64V(0.0)
                        for i in range(nv2):
                            t += _load_w(fxc, (i * nv2 + j) * ngrid + p0, v, npt) * rk.unsafe_load[width=W](i * BLK + q)
                        uk.unsafe_store(j * BLK + q, t * _load_w(weights, p0, v, npt))
            try:
                blas.gemm(False, True, na * 3, na * 3, nv2 * BLK, 1.0, r1, ub, 0.0, mm)
            except:
                pass
            for ia in range(na):
                for ib in range(na):
                    var dd = acc.unsafe_offset((alist[ia] * natm + alist[ib]) * 9)
                    for x in range(3):
                        for y in range(3):
                            dd[unsafe_offset=x * 3 + y] += mm[unsafe_offset=(ia * 3 + x) * (na * 3) + ib * 3 + y]
        _ = ws^
        _ = scratch^
        _ = alist^
        _ = a0l^
        _ = a1l^

    var nthr = blas.serial_begin()
    if nworkers == 1:
        work(0)
    else:
        parallelize(work, nworkers)
    blas.serial_end(nthr)
    for i in range(nde):
        var v = 0.0
        for w2 in range(nworkers):
            v += pacc[unsafe_offset=w2 * nde + i]
        de2_out[unsafe_offset=i] = v
    _ = accl^
    _ = counter^
    _ = grid^
    _ = rcut^
    _ = d2t^
    _ = d3t^


def xc_h1_core(
    blas: Blas,
    basis: Basis,
    ngrid: Int,
    coords: F64Ptr,
    weights: F64Ptr,
    kind: Int,
    nspin: Int,
    dms: F64Ptr,
    fxc: F64Ptr,
    aoatm: Pointer[Int64, MutAnyOrigin],
    natm: Int,
    cmo: F64Ptr,
    nmo: Int,
    nocc: Int,
    h1_out: F64Ptr,
) raises:
    """Kernel part of the XC first-derivative Fock matrices of the Hessian, projected to the MO basis.

    pyscf's ``_get_vxc_deriv1`` forms, for every atom A and direction x, the
    nao x nao matrix F + F^T with F = Z phi^T, Z = sum_c u_c d_c phi and
    u = w fxc rho1_{A,x} (rho1 = -d rho / d R_{A,x} as in ``xc_hess_core``, the
    density component of u halved), per spin from both spins' rho1.  Only
    C^T (F + F^T) C_o is needed (the Hessian projects every first-order Fock
    matrix): with Phi_c = C_sub^T d_c phi, shared by all perturbations, it is
    Z_m B^T + Phi_0 Z_o^T, Z_m = sum_c u_c Phi_c and Z_o, B its occupied rows
    (the orbitals are ordered occupied first): two GEMMs over all
    perturbations of a block instead of one nao x nao GEMM each.

    dms: nspin x nao x nao; fxc: (nspin nvar) x (nspin nvar) x ngrid (unweighted);
    cmo: nspin x nao x nmo (occupied first); h1_out: nspin x natm x 3 x nmo x nocc
    with nocc the largest occupied count (a spin with fewer occupied orbitals
    uses its first columns), overwritten (the gradient-matrix part and the
    overall sign are the caller's).  LDA (``kind`` 0) and GGA (1).
    """
    var gga = kind == 1
    var nderiv = 2 if gga else 1
    var nvar = 4 if gga else 1
    var nv2 = nspin * nvar
    var ncy = 4 if gga else 1
    var grid = Grid(ngrid, coords)
    var rcut = _rcuts(basis, nderiv)
    var nao = basis.nao
    var n2 = nao * nao
    var ncomp = _ncomp(nderiv)
    var nblk = grid.nblk
    var nworkers = max(1, min(parallelism_level(), nblk))
    var nh = nspin * natm * 3 * nmo * nocc
    var accl = List[Float64](length=nworkers * nh + 1, fill=0.0)
    var pacc = list_ptr(accl)
    var counter = Atomic[Int64](0)
    var pcount = Pointer(to=counter)
    var d2t = List[Int](length=9, fill=0)
    for x in range(3):
        for y in range(3):
            d2t[x * 3 + y] = _didx((1 if x == 0 else 0) + (1 if y == 0 else 0), (1 if x == 1 else 0) + (1 if y == 1 else 0),
                                   (1 if x == 2 else 0) + (1 if y == 2 else 0))
    var pd2 = Pointer[Int, MutAnyOrigin](unsafe_from_address=Int(d2t.unsafe_ptr()))

    def work(w: Int) {imm blas, imm basis, imm grid, imm rcut, imm pcount, imm nblk, imm nao, imm n2, imm ncomp, imm nderiv, imm gga, imm nvar, imm nv2, imm ncy, imm nspin, imm dms, imm fxc, imm weights, imm aoatm, imm natm, imm cmo, imm nmo, imm nocc, imm pacc, imm nh, imm ngrid, imm pd2}:
        var ws = AOWork(nao, basis.nbas, ncomp, basis.lmax, basis.nctr_max)
        var acc = pacc.unsafe_offset(w * nh)
        var kmax = 3 * natm
        var n_y = nspin * ncy * nao * BLK
        var n_r1 = kmax * nv2 * BLK
        var n_phi = nvar * nmo * BLK
        var n_z = kmax * nmo * BLK
        var n_zo = kmax * nocc * BLK
        var n_o1 = kmax * nmo * nocc
        var scratch = List[Float64](length=n_y + 2 * n_r1 + nao * nmo + n_phi + n_z + n_zo + 2 * n_o1 + 10 * W, fill=0.0)
        var ybuf = F64Ptr(unsafe_from_address=aligned_addr(scratch))
        var r1 = ybuf.unsafe_offset(n_y)
        var ub = r1.unsafe_offset(n_r1)
        var csub = ub.unsafe_offset(n_r1)
        var phi = csub.unsafe_offset(nao * nmo)
        var zb = phi.unsafe_offset(n_phi)
        var zo = zb.unsafe_offset(n_z)
        var o1 = zo.unsafe_offset(n_zo)
        var o2 = o1.unsafe_offset(n_o1)
        var alist = List[Int](length=natm + 1, fill=0)
        var a0l = List[Int](length=natm + 1, fill=0)
        var a1l = List[Int](length=natm + 1, fill=0)
        while True:
            var blk = Int(pcount[].fetch_add(1))
            if blk >= nblk:
                break
            var sel = select_shells(basis, grid, blk, rcut, ws)
            var nrow = sel[1]
            if nrow == 0:
                continue
            var p0 = blk * BLK
            var npt = min(BLK, ngrid - p0)
            eval_block(basis, grid, blk, nderiv, sel[0], nrow, ws)
            var cs = nrow * BLK
            var ao = ws.ao()
            var na = 0
            for r in range(nrow):
                var a = Int(aoatm[unsafe_offset=ws.rows[r]])
                if na == 0 or alist[na - 1] != a:
                    alist[na] = a
                    a0l[na] = r
                    na += 1
                a1l[na - 1] = r + 1
            var nk = 3 * na
            vfill(r1, nk * nv2 * BLK, 0.0)
            # ---- rho1_{A,x} of every spin
            for s in range(nspin):
                var ys = ybuf.unsafe_offset(s * ncy * nao * BLK)
                gather_block(dms.unsafe_offset(s * n2), nao, nrow, ws)
                for c in range(ncy):
                    try:
                        blas.gemm(False, False, nrow, BLK, nrow, 1.0, ws.dsub(), ao.unsafe_offset(c * cs), 0.0, ys.unsafe_offset(c * nrow * BLK))
                    except:
                        pass
                for ia in range(na):
                    for p in range(a0l[ia], a1l[ia]):
                        var o = p * BLK
                        for x in range(3):
                            var dx = ao.unsafe_offset((1 + x) * cs)
                            var rr = r1.unsafe_offset(((ia * 3 + x) * nv2 + s * nvar) * BLK)
                            for v in range(NV):
                                var q = v * W
                                var dxp = dx.unsafe_load[width=W](o + q)
                                var y0p = ys.unsafe_load[width=W](o + q)
                                rr.unsafe_store(q, rr.unsafe_load[width=W](q) + dxp * y0p * 2.0)
                                if gga:
                                    for c in range(3):
                                        var hxc = ao.unsafe_offset(pd2[unsafe_offset=x * 3 + c] * cs)
                                        var t = hxc.unsafe_load[width=W](o + q) * y0p + dxp * ys.unsafe_load[width=W]((c + 1) * nrow * BLK + o + q)
                                        rr.unsafe_store((c + 1) * BLK + q, rr.unsafe_load[width=W]((c + 1) * BLK + q) + t * 2.0)
            # ---- u = w fxc rho1 (density component halved)
            for k in range(nk):
                var rk = r1.unsafe_offset(k * nv2 * BLK)
                var uk = ub.unsafe_offset(k * nv2 * BLK)
                for j in range(nv2):
                    var half = 0.5 if j % nvar == 0 else 1.0
                    for v in range(NV):
                        var q = v * W
                        var t = F64V(0.0)
                        for i in range(nv2):
                            t += _load_w(fxc, (i * nv2 + j) * ngrid + p0, v, npt) * rk.unsafe_load[width=W](i * BLK + q)
                        uk.unsafe_store(j * BLK + q, t * _load_w(weights, p0, v, npt) * half)
            # ---- projected matrices per output spin
            for t in range(nspin):
                gather_rows_to(cmo.unsafe_offset(t * nao * nmo), nmo, ws, csub)
                for c in range(nvar):
                    try:
                        blas.gemm(True, False, nmo, BLK, nrow, 1.0, csub, ao.unsafe_offset(c * cs), 0.0, phi.unsafe_offset(c * nmo * BLK))
                    except:
                        pass
                for k in range(nk):
                    var uk = ub.unsafe_offset((k * nv2 + t * nvar) * BLK)
                    var zk = zb.unsafe_offset(k * nmo * BLK)
                    for m in range(nmo):
                        for v in range(NV):
                            var q = v * W
                            var z = uk.unsafe_load[width=W](q) * phi.unsafe_load[width=W](m * BLK + q)
                            if gga:
                                for c in range(1, 4):
                                    z += uk.unsafe_load[width=W](c * BLK + q) * phi.unsafe_load[width=W]((c * nmo + m) * BLK + q)
                            zk.unsafe_store(m * BLK + q, z)
                    vcopy(zo.unsafe_offset(k * nocc * BLK), zk, nocc * BLK)
                try:
                    # o1 = Z_m B^T (all perturbations), o2 = Phi_0 Z_o^T
                    blas.gemm(False, True, nk * nmo, nocc, BLK, 1.0, zb, phi, 0.0, o1)
                    blas.gemm(False, True, nmo, nk * nocc, BLK, 1.0, phi, zo, 0.0, o2)
                except:
                    pass
                for ia in range(na):
                    for x in range(3):
                        var k = ia * 3 + x
                        var dst = acc.unsafe_offset((((t * natm + alist[ia]) * 3 + x) * nmo) * nocc)
                        vaxpy(dst, nmo * nocc, 1.0, o1.unsafe_offset(k * nmo * nocc))
                        for m in range(nmo):
                            vaxpy(dst.unsafe_offset(m * nocc), nocc, 1.0, o2.unsafe_offset(m * nk * nocc + k * nocc))
        _ = ws^
        _ = scratch^
        _ = alist^
        _ = a0l^
        _ = a1l^

    var nthr = blas.serial_begin()
    if nworkers == 1:
        work(0)
    else:
        parallelize(work, nworkers)
    blas.serial_end(nthr)
    for i in range(nh):
        var v = 0.0
        for w2 in range(nworkers):
            v += pacc[unsafe_offset=w2 * nh + i]
        h1_out[unsafe_offset=i] = v
    _ = accl^
    _ = counter^
    _ = grid^
    _ = rcut^
    _ = d2t^
