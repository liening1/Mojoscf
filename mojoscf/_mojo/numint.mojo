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
from _mojo.linalg import F64Ptr, Blas, list_ptr, vfill
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


def shell_cutoff_radius(basis: Basis, b: Int) -> Float64:
    """Distance from the shell's centre beyond which its functions and gradients stay below AO_CUT.

    Bound: f(r) = s sum_p max_c |c_cp| r^l e^{-a_p r^2} (1 + l/r + 2 a_p r), with
    s the largest coefficient of the Cartesian-to-spherical transform.
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
            f += cm * rl * exp(-a * r * r) * (1.0 + Float64(l) / r + 2.0 * a * r)
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
    var yaddr: Int          # nao x BLK
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
        var n_y = nao * BLK
        var n_d = nao * nao
        var n_cart = 4 * nc * W
        var n_pw = (3 * (lmax + 2) + 2 * nctr_max) * W
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


def eval_block(basis: Basis, grid: Grid, blk: Int, deriv: Bool, nshell: Int, nrow: Int, mut ws: AOWork):
    """Values (and gradients) of the selected shells on block ``blk`` into ws.ao[comp][row][BLK]."""
    var p0 = blk * BLK
    var cstride = nrow * BLK
    var cart = F64Ptr(unsafe_from_address=ws.cartaddr)
    var pw = F64Ptr(unsafe_from_address=ws.pwaddr)
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
        var xp = pw                                 # dx^0 .. dx^(l+1)
        var yp = pw.unsafe_offset((l + 2) * W)
        var zp = pw.unsafe_offset(2 * (l + 2) * W)
        var rs = pw.unsafe_offset(3 * (l + 2) * W)  # R_c, then S_c
        var ss = rs.unsafe_offset(nc * W)
        for v in range(NV):
            var dx = grid.x().unsafe_load[width=W](p0 + v * W) - ax
            var dy = grid.y().unsafe_load[width=W](p0 + v * W) - ay
            var dz = grid.z().unsafe_load[width=W](p0 + v * W) - az
            var r2 = dx * dx + dy * dy + dz * dz
            for c in range(nc):
                rs.unsafe_store(c * W, F64V(0.0))
                ss.unsafe_store(c * W, F64V(0.0))
            for p in range(np):
                var a = basis.env[basis.pexp[b] + p]
                var e = vexp(r2 * (-a))
                for c in range(nc):
                    var cf = basis.env[basis.pcoef[b] + c * np + p]
                    rs.unsafe_store(c * W, rs.unsafe_load[width=W](c * W) + e * cf)
                    if deriv:
                        ss.unsafe_store(c * W, ss.unsafe_load[width=W](c * W) + e * (-2.0 * a * cf))
            var tx = F64V(1.0)
            var ty = F64V(1.0)
            var tz = F64V(1.0)
            for k in range(l + 2):
                xp.unsafe_store(k * W, tx)
                yp.unsafe_store(k * W, ty)
                zp.unsafe_store(k * W, tz)
                tx *= dx
                ty *= dy
                tz *= dz
            for c in range(nc):
                var rr = rs.unsafe_load[width=W](c * W)
                var sv = ss.unsafe_load[width=W](c * W)
                for cc in range(nca):
                    var i = basis.cx[coff + cc]
                    var j = basis.cy[coff + cc]
                    var k = basis.cz[coff + cc]
                    var xi = xp.unsafe_load[width=W](i * W)
                    var yj = yp.unsafe_load[width=W](j * W)
                    var zk = zp.unsafe_load[width=W](k * W)
                    var yz = yj * zk
                    cart.unsafe_store(cc * W, xi * yz * rr)
                    if deriv:
                        var gxv = xp.unsafe_load[width=W]((i + 1) * W) * sv
                        if i > 0:
                            gxv += xp.unsafe_load[width=W]((i - 1) * W) * (Float64(i) * rr)
                        var gyv = yp.unsafe_load[width=W]((j + 1) * W) * sv
                        if j > 0:
                            gyv += yp.unsafe_load[width=W]((j - 1) * W) * (Float64(j) * rr)
                        var gzv = zp.unsafe_load[width=W]((k + 1) * W) * sv
                        if k > 0:
                            gzv += zp.unsafe_load[width=W]((k - 1) * W) * (Float64(k) * rr)
                        cart.unsafe_store((nca + cc) * W, gxv * yz)
                        cart.unsafe_store((2 * nca + cc) * W, xi * zk * gyv)
                        cart.unsafe_store((3 * nca + cc) * W, xi * yj * gzv)
                var ncomp = 4 if deriv else 1
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


def _rcuts(basis: Basis) -> List[Float64]:
    var rc = List[Float64](length=max(basis.nbas, 1), fill=0.0)
    for b in range(basis.nbas):
        rc[b] = shell_cutoff_radius(basis, b)
    return rc^


def eval_ao_core(basis: Basis, ngrid: Int, coords: F64Ptr, deriv: Bool, ao_out: F64Ptr):
    """All AO values (and gradients) on the grid: out[comp][point][ao] (pyscf's ``eval_gto`` layout).

    For tests: screened functions are left at zero, as pyscf does.
    """
    var grid = Grid(ngrid, coords)
    var rcut = _rcuts(basis)
    var nao = basis.nao
    var ncomp = 4 if deriv else 1
    vfill(ao_out, ncomp * ngrid * nao, 0.0)
    var ws = AOWork(nao, basis.nbas, ncomp, basis.lmax, basis.nctr_max)
    for blk in range(grid.nblk):
        var sel = select_shells(basis, grid, blk, rcut, ws)
        var nrow = sel[1]
        eval_block(basis, grid, blk, deriv, sel[0], nrow, ws)
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


def xc_rho_core(
    blas: Blas, basis: Basis, ngrid: Int, coords: F64Ptr, deriv: Bool, nset: Int, dms: F64Ptr, rho_out: F64Ptr
) raises:
    """rho_out[s][c][p]: density (c = 0) and, with ``deriv``, its gradient (c = 1..3) of each D_s (symmetric)."""
    var grid = Grid(ngrid, coords)
    var rcut = _rcuts(basis)
    var nao = basis.nao
    var n2 = nao * nao
    var ncomp = 4 if deriv else 1
    var nblk = grid.nblk
    var nworkers = max(1, min(parallelism_level(), nblk))
    var counter = Atomic[Int64](0)
    var pcount = Pointer(to=counter)

    def work(w: Int) {imm blas, imm basis, imm grid, imm rcut, imm pcount, imm nblk, imm nao, imm n2, imm ncomp, imm deriv, imm nset, imm dms, imm rho_out, imm ngrid}:
        var ws = AOWork(nao, basis.nbas, ncomp, basis.lmax, basis.nctr_max)
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
                    for c in range(ncomp):
                        vfill(rho_out.unsafe_offset((s * ncomp + c) * ngrid + p0), npt, 0.0)
                continue
            eval_block(basis, grid, blk, deriv, sel[0], nrow, ws)
            for s in range(nset):
                var dm = dms.unsafe_offset(s * n2)
                gather_block(dm, nao, nrow, ws)
                try:
                    blas.gemm(False, False, nrow, BLK, nrow, 1.0, ws.dsub(), ws.ao(), 0.0, ws.y())
                except:
                    pass
                for c in range(ncomp):
                    var fac = 1.0 if c == 0 else 2.0
                    var ac = ws.ao().unsafe_offset(c * nrow * BLK)
                    var dst = rho_out.unsafe_offset((s * ncomp + c) * ngrid + p0)
                    for v in range(NV):
                        var acc = F64V(0.0)
                        for i in range(nrow):
                            acc += ac.unsafe_load[width=W](i * BLK + v * W) * ws.y().unsafe_load[width=W](i * BLK + v * W)
                        acc *= fac
                        if v * W + W <= npt:
                            dst.unsafe_store(v * W, acc)
                        else:
                            for lane in range(W):
                                if v * W + lane < npt:
                                    dst[unsafe_offset=v * W + lane] = acc[lane]
        _ = ws^

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
    blas: Blas, basis: Basis, ngrid: Int, coords: F64Ptr, deriv: Bool, nset: Int, wv: F64Ptr, vmat_out: F64Ptr
) raises:
    """vmat_out[s] = sum_p phi(p) [sum_c wv[s][c][p] phi_c(p)]^T (nao x nao, not symmetrised), overwritten."""
    var grid = Grid(ngrid, coords)
    var rcut = _rcuts(basis)
    var nao = basis.nao
    var n2 = nao * nao
    var ncomp = 4 if deriv else 1
    var nblk = grid.nblk
    var nworkers = max(1, min(parallelism_level(), nblk))
    var accl = List[Float64](length=nworkers * nset * n2 + 1, fill=0.0)
    var pacc = list_ptr(accl)
    var counter = Atomic[Int64](0)
    var pcount = Pointer(to=counter)

    def work(w: Int) {imm blas, imm basis, imm grid, imm rcut, imm pcount, imm nblk, imm nao, imm n2, imm ncomp, imm deriv, imm nset, imm wv, imm pacc, imm ngrid}:
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
            eval_block(basis, grid, blk, deriv, sel[0], nrow, ws)
            for s in range(nset):
                # Z[i][p] = sum_c wv_c(p) phi_c,i(p) into ws.y (padding points weigh zero)
                for v in range(NV):
                    var w0 = _load_w(wv, (s * ncomp) * ngrid + p0, v, npt)
                    if ncomp == 1:
                        for i in range(nrow):
                            ws.y().unsafe_store(i * BLK + v * W, ws.ao().unsafe_load[width=W](i * BLK + v * W) * w0)
                    else:
                        var w1 = _load_w(wv, (s * ncomp + 1) * ngrid + p0, v, npt)
                        var w2 = _load_w(wv, (s * ncomp + 2) * ngrid + p0, v, npt)
                        var w3 = _load_w(wv, (s * ncomp + 3) * ngrid + p0, v, npt)
                        var ax = ws.ao().unsafe_offset(nrow * BLK)
                        var ay = ws.ao().unsafe_offset(2 * nrow * BLK)
                        var az = ws.ao().unsafe_offset(3 * nrow * BLK)
                        for i in range(nrow):
                            var o = i * BLK + v * W
                            var z = ws.ao().unsafe_load[width=W](o) * w0 + ax.unsafe_load[width=W](o) * w1
                            z += ay.unsafe_load[width=W](o) * w2 + az.unsafe_load[width=W](o) * w3
                            ws.y().unsafe_store(o, z)
                try:
                    blas.gemm(False, True, nrow, nrow, BLK, 1.0, ws.ao(), ws.y(), 0.0, ws.dsub())
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
