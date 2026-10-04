"""Gaussian integral engine: overlap, kinetic, nuclear attraction and
electron-repulsion integrals over contracted Gaussian basis functions.

The engine implements the McMurchie-Davidson scheme.  A product of two
Cartesian primitives ``x_A^i x_B^j exp(-a r_A^2) exp(-b r_B^2)`` is expanded
in Hermite Gaussians centred at P with coefficients ``E_t^{ij}``; one-electron
integrals then reduce to ``E_0^{ij}`` (overlap, kinetic energy) or to the
Hermite Coulomb integrals ``R_{tuv}`` (nuclear attraction), and electron
repulsion integrals to

    (ab|cd) = 2 pi^{5/2} / (p q sqrt(p + q))
              sum_{tuv} E^{ab}_{tuv} sum_{TUV} (-1)^{T+U+V} E^{cd}_{TUV} R_{t+T, u+U, v+V}(alpha, P - Q)

with ``alpha = p q / (p + q)``.  ``R_{tuv}`` are generated from the Boys
function, which is evaluated from a pretabulated Taylor expansion (eight terms
on a 0.05 grid up to T = 36) followed by the downward recursion, and from the
asymptotic form with the upward recursion beyond.

Conventions follow pyscf/libcint exactly: the basis is read from pyscf's
``_atm``/``_bas``/``_env`` tables (coefficients already carry the primitive
normalisation ``gto_norm``), Cartesian components are ordered ``x^l``,
``x^{l-1} y``, ..., ``z^l``, and the Cartesian block of every shell is
transformed with the ``cart2sph`` matrices supplied by the caller (for s and p
shells these are multiples of the identity and are folded into the contraction
coefficients, as libcint does with ``CINTcommon_fac_sp``).

Two-electron integrals are produced directly in pyscf's 8-fold packed layout
``eri[ij (ij + 1) / 2 + kl]`` with ``ij = i (i + 1) / 2 + j`` (``i >= j``,
``ij >= kl``).  Shell pairs are processed in parallel; every packed element is
written by exactly one shell quartet, so no synchronisation is needed.
"""
from std.atomic import Atomic
from std.ffi import external_call
from std.math import sqrt
from std.memory import Pointer
from std.runtime import parallelism_level
from std.sys import simd_width_of
from max.algorithm import parallelize
from _mojo.linalg import F64Ptr, list_ptr, vfill, vaxpy

comptime W = simd_width_of[DType.float64]()
comptime I64Ptr = Pointer[Int64, MutAnyOrigin]
comptime IntPtr = Pointer[Int, MutAnyOrigin]
comptime PI = 3.141592653589793
comptime TWO_PI_52 = 34.986836655249725    # 2 pi^{5/2}
# Primitive pairs with mu |AB|^2 beyond this are dropped (exp(-60) ~ 1e-26),
# which is libcint's default EXPCUTOFF.
comptime EXP_CUTOFF = 60.0


def ncart(l: Int) -> Int:
    return (l + 1) * (l + 2) // 2


def exp(x: Float64) -> Float64:
    """Correctly rounded exponential from the C library (``std.math.exp`` is only accurate to ~1e-11)."""
    return external_call["exp", Float64](x)


def nherm(l: Int) -> Int:
    """Number of Hermite indices (t, u, v) with t + u + v <= l."""
    return (l + 1) * (l + 2) * (l + 3) // 6


def padded(n: Int) -> Int:
    """``n`` rounded up to a multiple of the SIMD width."""
    return ((n + W - 1) // W) * W


def int_ptr(ref buf: List[Int]) -> IntPtr:
    return IntPtr(unsafe_from_address=Int(buf.unsafe_ptr()))


# --------------------------------------------------------------------------
# Basis set
# --------------------------------------------------------------------------


struct Basis(Movable):
    """A basis set in pyscf's internal representation.

    ``atm`` is pyscf's ``_atm`` (charge, pointer to the coordinates, ...),
    ``bas`` its ``_bas`` (atom, l, nprim, nctr, kappa, pointer to exponents,
    pointer to coefficients) and ``env`` the float table both point into.
    ``nf[l]`` is the number of functions a shell of angular momentum ``l``
    contributes (``2l+1`` or ``ncart``) and ``c2s`` holds, for l = 0..lmax,
    the ``(ncart(l), nf[l])`` transformation matrices back to back.
    """

    var nbas: Int
    var natm: Int
    var lmax: Int
    var l: List[Int]
    var atom: List[Int]
    var nprim: List[Int]
    var nctr: List[Int]
    var pexp: List[Int]
    var pcoef: List[Int]
    var env: List[Float64]
    var charge: List[Float64]
    var coord: List[Float64]       # 3 natm
    var nf: List[Int]              # per l
    var c2s: List[Float64]
    var c2s_off: List[Int]         # per l
    var c2s_scale: List[Float64]   # per l: w when c2s = w I (then no transform is needed), else 0
    var ao_loc: List[Int]          # nbas + 1
    var cx: List[Int]              # Cartesian exponents per l, offset cart_off[l]
    var cy: List[Int]
    var cz: List[Int]
    var cart_off: List[Int]
    var nprim_max: Int
    var nctr_max: Int
    var dmax: Int                  # max over shells of nctr * ncart
    var nao: Int

    def __init__(
        out self, natm: Int, atm: I64Ptr, nbas: Int, bas: I64Ptr, nenv: Int, env: F64Ptr, nl: Int, nf: I64Ptr, c2s: F64Ptr
    ):
        self.nbas = nbas
        self.natm = natm
        self.l = List[Int](length=nbas, fill=0)
        self.atom = List[Int](length=nbas, fill=0)
        self.nprim = List[Int](length=nbas, fill=0)
        self.nctr = List[Int](length=nbas, fill=0)
        self.pexp = List[Int](length=nbas, fill=0)
        self.pcoef = List[Int](length=nbas, fill=0)
        self.env = List[Float64](length=nenv, fill=0.0)
        for i in range(nenv):
            self.env[i] = env[unsafe_offset=i]
        self.charge = List[Float64](length=natm, fill=0.0)
        self.coord = List[Float64](length=3 * natm, fill=0.0)
        for a in range(natm):
            self.charge[a] = Float64(atm[unsafe_offset=a * 6])
            var pc = Int(atm[unsafe_offset=a * 6 + 1])
            for d in range(3):
                self.coord[3 * a + d] = self.env[pc + d]
        self.lmax = 0
        self.nprim_max = 1
        self.nctr_max = 1
        for b in range(nbas):
            self.atom[b] = Int(bas[unsafe_offset=b * 8])
            self.l[b] = Int(bas[unsafe_offset=b * 8 + 1])
            self.nprim[b] = Int(bas[unsafe_offset=b * 8 + 2])
            self.nctr[b] = Int(bas[unsafe_offset=b * 8 + 3])
            self.pexp[b] = Int(bas[unsafe_offset=b * 8 + 5])
            self.pcoef[b] = Int(bas[unsafe_offset=b * 8 + 6])
            self.lmax = max(self.lmax, self.l[b])
            self.nprim_max = max(self.nprim_max, self.nprim[b])
            self.nctr_max = max(self.nctr_max, self.nctr[b])
        self.nf = List[Int](length=nl, fill=0)
        self.c2s_off = List[Int](length=nl + 1, fill=0)
        self.c2s_scale = List[Float64](length=nl, fill=0.0)
        for l in range(nl):
            self.nf[l] = Int(nf[unsafe_offset=l])
            self.c2s_off[l + 1] = self.c2s_off[l] + ncart(l) * self.nf[l]
        self.c2s = List[Float64](length=self.c2s_off[nl], fill=0.0)
        for i in range(self.c2s_off[nl]):
            self.c2s[i] = c2s[unsafe_offset=i]
        for l in range(nl):
            var n = ncart(l)
            if self.nf[l] != n:
                continue
            var off = self.c2s_off[l]
            var w = self.c2s[off]
            var scalar = True
            for i in range(n):
                for j in range(n):
                    var v = self.c2s[off + i * n + j]
                    if (i == j and v != w) or (i != j and v != 0.0):
                        scalar = False
            if scalar:
                self.c2s_scale[l] = w
        self.ao_loc = List[Int](length=nbas + 1, fill=0)
        self.dmax = 1
        for b in range(nbas):
            var d = self.nctr[b] * self.nf[self.l[b]]
            self.ao_loc[b + 1] = self.ao_loc[b] + d
            self.dmax = max(self.dmax, self.nctr[b] * max(self.nf[self.l[b]], ncart(self.l[b])))
        self.nao = self.ao_loc[nbas]
        self.cart_off = List[Int](length=nl + 1, fill=0)
        for l in range(nl):
            self.cart_off[l + 1] = self.cart_off[l] + ncart(l)
        var ntot = self.cart_off[nl]
        self.cx = List[Int](length=ntot, fill=0)
        self.cy = List[Int](length=ntot, fill=0)
        self.cz = List[Int](length=ntot, fill=0)
        for l in range(nl):
            var c = self.cart_off[l]
            for i in range(l, -1, -1):
                for j in range(l - i, -1, -1):
                    self.cx[c] = i
                    self.cy[c] = j
                    self.cz[c] = l - i - j
                    c += 1

    def shell_coord(self, b: Int, d: Int) -> Float64:
        return self.coord[3 * self.atom[b] + d]

    def needs_transform(self, b: Int) -> Bool:
        return self.c2s_scale[self.l[b]] == 0.0

    def func_scale(self, b: Int) -> Float64:
        """Factor folded into the contraction coefficients of shell ``b`` (1 when a transform is needed)."""
        var w = self.c2s_scale[self.l[b]]
        return 1.0 if w == 0.0 else w


# --------------------------------------------------------------------------
# Boys function
# --------------------------------------------------------------------------


struct BoysTable(Movable):
    """``F_n(T)`` for n <= NMAX via a Taylor table (T < 36) or the asymptotic form."""

    comptime NROWS = 40          # n = 0 .. 39 in the table
    comptime NTERMS = 8          # Taylor terms
    comptime NMAX = Self.NROWS - Self.NTERMS
    comptime NT = 721            # grid points T0 = 0, 0.05, ..., 36
    comptime DT = 0.05
    comptime TMAX = 36.0

    var table: List[Float64]     # [k * NROWS + n] = F_n(k DT)

    def __init__(out self):
        self.table = List[Float64](length=Self.NT * Self.NROWS, fill=0.0)
        var n = Self.NROWS - 1
        for k in range(Self.NT):
            var t0 = Float64(k) * Self.DT
            # F_n(T) = exp(-T) sum_i (2T)^i / ((2n+1)(2n+3)...(2n+2i+1))
            var term = 1.0 / Float64(2 * n + 1)
            var s = term
            var i = 1
            while True:
                term *= 2.0 * t0 / Float64(2 * n + 2 * i + 1)
                s += term
                if term < 1e-17 * s:
                    break
                i += 1
            var et = exp(-t0)
            self.table[k * Self.NROWS + n] = s * et
            for m in range(n - 1, -1, -1):
                self.table[k * Self.NROWS + m] = (2.0 * t0 * self.table[k * Self.NROWS + m + 1] + et) / Float64(2 * m + 1)

    def eval(self, nmax: Int, t: Float64, f: F64Ptr):
        """f[n] = F_n(t) for n = 0..nmax (nmax <= NMAX)."""
        if t >= Self.TMAX:
            var f0 = 0.5 * sqrt(PI / t)
            var et = exp(-t)
            var inv2t = 0.5 / t
            f[unsafe_offset=0] = f0
            for n in range(nmax):
                f[unsafe_offset=n + 1] = (Float64(2 * n + 1) * f[unsafe_offset=n] - et) * inv2t
            return
        var k = Int(t / Self.DT + 0.5)
        var dt = Float64(k) * Self.DT - t
        var base = k * Self.NROWS + nmax
        # Taylor expansion of F_nmax about T0 = k DT: F_n(T0 - dt) = sum_j F_{n+j}(T0) dt^j / j!
        var acc = self.table[base + Self.NTERMS - 1]
        for j in range(Self.NTERMS - 1, 0, -1):
            acc = acc * dt / Float64(j) + self.table[base + j - 1]
        f[unsafe_offset=nmax] = acc
        if nmax == 0:
            return
        var et = exp(-t)
        var t2 = 2.0 * t
        for n in range(nmax - 1, -1, -1):
            f[unsafe_offset=n] = (t2 * f[unsafe_offset=n + 1] + et) / Float64(2 * n + 1)


# --------------------------------------------------------------------------
# Hermite expansion coefficients and Hermite Coulomb integrals
# --------------------------------------------------------------------------


def hermite_e(la: Int, lb: Int, p: Float64, xpa: Float64, xpb: Float64, kab: Float64, e: F64Ptr):
    """E_t^{ij} for i <= la, j <= lb, t <= i + j, laid out ``e[(i (lb+1) + j) (la+lb+1) + t]``."""
    var s = la + lb + 1
    vfill(e, (la + 1) * (lb + 1) * s, 0.0)
    e[unsafe_offset=0] = kab
    var inv2p = 0.5 / p
    for i in range(la + 1):
        for j in range(lb + 1):
            if i == 0 and j == 0:
                continue
            var row = (i * (lb + 1) + j) * s
            var prev: Int
            var x: Float64
            if j == 0:
                prev = ((i - 1) * (lb + 1)) * s
                x = xpa
            else:
                prev = (i * (lb + 1) + j - 1) * s
                x = xpb
            var n = i + j
            for t in range(n + 1):
                var v = x * e[unsafe_offset=prev + t]
                if t > 0:
                    v += inv2p * e[unsafe_offset=prev + t - 1]
                if t + 1 <= n - 1:
                    v += Float64(t + 1) * e[unsafe_offset=prev + t + 1]
                e[unsafe_offset=row + t] = v


def hermite_r(
    big_l: Int, alpha: Float64, x: Float64, y: Float64, z: Float64, boys: BoysTable, f: F64Ptr, r0: F64Ptr, r1: F64Ptr
):
    """R_{tuv} = R^0_{tuv}(alpha, (x, y, z)) for t + u + v <= L into the cube ``r0`` of side L + 1.

    ``f`` is scratch for L + 1 Boys values and ``r1`` a second cube; level n of
    the recursion lives in r0 for even n and in r1 for odd n, so the result
    (n = 0) always ends in r0.
    """
    var s1 = big_l + 1
    var s2 = s1 * s1
    boys.eval(big_l, alpha * (x * x + y * y + z * z), f)
    var m2a = -2.0 * alpha
    var pw = 1.0
    for n in range(big_l + 1):
        f[unsafe_offset=n] = f[unsafe_offset=n] * pw
        pw *= m2a
    for n in range(big_l, -1, -1):
        var even = (n % 2) == 0
        var cur = r0 if even else r1
        var prev = r1 if even else r0
        cur[unsafe_offset=0] = f[unsafe_offset=n]
        var lm = big_l - n
        for t in range(lm + 1):
            for u in range(lm - t + 1):
                for v in range(lm - t - u + 1):
                    if t + u + v == 0:
                        continue
                    var idx = t * s2 + u * s1 + v
                    var val: Float64
                    if t > 0:
                        val = x * prev[unsafe_offset=idx - s2]
                        if t > 1:
                            val += Float64(t - 1) * prev[unsafe_offset=idx - 2 * s2]
                    elif u > 0:
                        val = y * prev[unsafe_offset=idx - s1]
                        if u > 1:
                            val += Float64(u - 1) * prev[unsafe_offset=idx - 2 * s1]
                    else:
                        val = z * prev[unsafe_offset=idx - 1]
                        if v > 1:
                            val += Float64(v - 1) * prev[unsafe_offset=idx - 2]
                    cur[unsafe_offset=idx] = val


# --------------------------------------------------------------------------
# Shell-pair data
# --------------------------------------------------------------------------


struct PairData(Movable):
    """Primitive-pair quantities of one shell pair, in reusable storage.

    For each surviving primitive pair k: the total exponent ``p[k]``, the
    centre ``pc[3k..]``, the contraction coefficient products
    ``coef[(k nca + ca) ncb + cb]`` and the Hermite coefficient matrix
    ``hmat[k][h][cp] = E^x_t E^y_u E^z_v`` over the Hermite indices
    ``h = (t, u, v)`` with ``t + u + v <= la + lb`` (``ht/hu/hv[h]``) and the
    Cartesian component pairs ``cp`` (row stride ``stride``, padded to the
    SIMD width).  When ``signed`` the entries carry ``(-1)^{t+u+v}`` (ket side
    of an ERI).

    ``b < 0`` denotes the dummy s partner (exponent 0, coefficient 1) used for
    three- and two-centre integrals: the "pair" is then a single shell.
    """

    var a: Int
    var b: Int
    var la: Int
    var lb: Int
    var lab: Int
    var nca: Int
    var ncb: Int
    var ncarta: Int
    var ncartb: Int
    var ncab: Int
    var nh: Int
    var stride: Int
    var hsize: Int           # nh * stride
    var np: Int
    var p: List[Float64]
    var pc: List[Float64]
    var coef: List[Float64]
    var e: List[Float64]        # scratch: 3 E arrays of one primitive pair
    var hmat: List[Float64]
    var ht: List[Int]
    var hu: List[Int]
    var hv: List[Int]
    var hidx: List[Int]         # (t (lab+1) + u) (lab+1) + v -> h
    var hidx_lab: Int           # lab the index tables were built for (-1: none)
    var signed: Bool

    def __init__(out self, lmax: Int, nprim_max: Int, nctr_max: Int, signed: Bool):
        var npmax = nprim_max * nprim_max
        var ncmax = ncart(lmax)
        var l21 = 2 * lmax + 1
        self.a = 0
        self.b = 0
        self.la = 0
        self.lb = 0
        self.lab = 0
        self.nca = 1
        self.ncb = 1
        self.ncarta = 1
        self.ncartb = 1
        self.ncab = 1
        self.nh = 1
        self.stride = W
        self.hsize = W
        self.np = 0
        self.signed = signed
        self.p = List[Float64](length=npmax, fill=0.0)
        self.pc = List[Float64](length=3 * npmax, fill=0.0)
        self.coef = List[Float64](length=npmax * nctr_max * nctr_max, fill=0.0)
        self.e = List[Float64](length=3 * (lmax + 1) * (lmax + 1) * l21, fill=0.0)
        self.hmat = List[Float64](length=min(npmax, 16) * nherm(2 * lmax) * padded(ncmax * ncmax), fill=0.0)
        self.ht = List[Int](length=nherm(2 * lmax), fill=0)
        self.hu = List[Int](length=nherm(2 * lmax), fill=0)
        self.hv = List[Int](length=nherm(2 * lmax), fill=0)
        self.hidx = List[Int](length=l21 * l21 * l21, fill=0)
        self.hidx_lab = -1

    def fill(mut self, ba: Basis, a: Int, bb: Basis, b: Int):
        """Set up the pair (shell ``a`` of ``ba``, shell ``b`` of ``bb``); ``b < 0`` is the dummy partner."""
        self.a = a
        self.b = b
        self.la = ba.l[a]
        self.lb = bb.l[b] if b >= 0 else 0
        self.lab = self.la + self.lb
        self.nca = ba.nctr[a]
        self.ncb = bb.nctr[b] if b >= 0 else 1
        self.ncarta = ncart(self.la)
        self.ncartb = ncart(self.lb)
        self.ncab = self.ncarta * self.ncartb
        self.stride = padded(self.ncab)
        var npa = ba.nprim[a]
        var npb = bb.nprim[b] if b >= 0 else 1
        var ax = ba.shell_coord(a, 0)
        var ay = ba.shell_coord(a, 1)
        var az = ba.shell_coord(a, 2)
        var bx = bb.shell_coord(b, 0) if b >= 0 else ax
        var by = bb.shell_coord(b, 1) if b >= 0 else ay
        var bz = bb.shell_coord(b, 2) if b >= 0 else az
        var rab2 = (ax - bx) * (ax - bx) + (ay - by) * (ay - by) + (az - bz) * (az - bz)
        var fac = ba.func_scale(a) * (bb.func_scale(b) if b >= 0 else 1.0)
        # Hermite index set (rebuilt only when la + lb changes)
        var s = self.lab + 1
        var lb1 = self.lb + 1
        if self.hidx_lab != self.lab:
            var h = 0
            for t in range(s):
                for u in range(s - t):
                    for v in range(s - t - u):
                        self.ht[h] = t
                        self.hu[h] = u
                        self.hv[h] = v
                        self.hidx[(t * s + u) * s + v] = h
                        h += 1
            self.hidx_lab = self.lab
        self.nh = nherm(self.lab)
        self.hsize = self.nh * self.stride
        if npa * npb * self.hsize > len(self.hmat):
            self.hmat.resize(npa * npb * self.hsize, 0.0)
        var offa = ba.cart_off[self.la]
        var offb = bb.cart_off[self.lb] if b >= 0 else 0
        var esz = (self.la + 1) * lb1 * s
        var pe = list_ptr(self.e)
        var ph = list_ptr(self.hmat)
        var phidx = int_ptr(self.hidx)
        var kk = 0
        for ia in range(npa):
            var ea = ba.env[ba.pexp[a] + ia]
            for ib in range(npb):
                var eb = bb.env[bb.pexp[b] + ib] if b >= 0 else 0.0
                var p = ea + eb
                var mu = ea * eb / p
                if mu * rab2 > EXP_CUTOFF:
                    continue
                var px = (ea * ax + eb * bx) / p
                var py = (ea * ay + eb * by) / p
                var pz = (ea * az + eb * bz) / p
                self.p[kk] = p
                self.pc[3 * kk] = px
                self.pc[3 * kk + 1] = py
                self.pc[3 * kk + 2] = pz
                hermite_e(self.la, self.lb, p, px - ax, px - bx, exp(-mu * (ax - bx) * (ax - bx)), pe)
                hermite_e(self.la, self.lb, p, py - ay, py - by, exp(-mu * (ay - by) * (ay - by)), pe.unsafe_offset(esz))
                hermite_e(self.la, self.lb, p, pz - az, pz - bz, exp(-mu * (az - bz) * (az - bz)), pe.unsafe_offset(2 * esz))
                for ca in range(self.nca):
                    var cfa = ba.env[ba.pcoef[a] + ca * npa + ia] * fac
                    for cb in range(self.ncb):
                        var cfb = bb.env[bb.pcoef[b] + cb * npb + ib] if b >= 0 else 1.0
                        self.coef[(kk * self.nca + ca) * self.ncb + cb] = cfa * cfb
                var hm = ph.unsafe_offset(kk * self.hsize)
                vfill(hm, self.hsize, 0.0)
                for ca in range(self.ncarta):
                    var i = ba.cx[offa + ca]
                    var k = ba.cy[offa + ca]
                    var m = ba.cz[offa + ca]
                    for cb in range(self.ncartb):
                        var cp = ca * self.ncartb + cb
                        var j = 0
                        var l = 0
                        var n = 0
                        if b >= 0:
                            j = bb.cx[offb + cb]
                            l = bb.cy[offb + cb]
                            n = bb.cz[offb + cb]
                        var ex = (i * lb1 + j) * s
                        var ey = esz + (k * lb1 + l) * s
                        var ez = 2 * esz + (m * lb1 + n) * s
                        for t in range(i + j + 1):
                            var vx = pe[unsafe_offset=ex + t]
                            for u in range(k + l + 1):
                                var vxy = vx * pe[unsafe_offset=ey + u]
                                var hrow = phidx.unsafe_offset((t * s + u) * s)
                                for v in range(m + n + 1):
                                    var val = vxy * pe[unsafe_offset=ez + v]
                                    if self.signed and ((t + u + v) % 2 == 1):
                                        val = -val
                                    hm[unsafe_offset=hrow[unsafe_offset=v] * self.stride + cp] = val
                kk += 1
        self.np = kk

    def center(self, k: Int, d: Int) -> Float64:
        return self.pc[3 * k + d]


# --------------------------------------------------------------------------
# Spherical transformation and scratch space
# --------------------------------------------------------------------------


def transform_axis(src: F64Ptr, dst: F64Ptr, pre: Int, ncart: Int, nf: Int, post: Int, c2s: F64Ptr):
    """dst[a, f, b] = sum_c c2s[c, f] src[a, c, b] for a < pre, b < post."""
    if post == 1:
        for a in range(pre):
            var srow = src.unsafe_offset(a * ncart)
            var drow = dst.unsafe_offset(a * nf)
            for f in range(nf):
                var acc = 0.0
                for c in range(ncart):
                    acc += c2s[unsafe_offset=c * nf + f] * srow[unsafe_offset=c]
                drow[unsafe_offset=f] = acc
        return
    for a in range(pre):
        for f in range(nf):
            var d = dst.unsafe_offset((a * nf + f) * post)
            vfill(d, post, 0.0)
            for c in range(ncart):
                var w = c2s[unsafe_offset=c * nf + f]
                if w != 0.0:
                    vaxpy(d, post, w, src.unsafe_offset((a * ncart + c) * post))


struct Workspace(Movable):
    """Per-thread scratch for one shell quartet (or one-electron shell pair)."""

    var f: List[Float64]
    var r0: List[Float64]
    var r1: List[Float64]
    var bm: List[Float64]       # [m][cpb] (ket Hermite index x bra component pair), stride padded(ncab)
    var blk: List[Float64]      # [cpb][cpk], stride padded(ncd)
    var acc: List[Float64]
    var out: List[Float64]
    var tmp: List[Float64]
    var hoff: List[Int]
    var koff: List[Int]
    var midx: List[Int]

    def __init__(out self, lmax_bra: Int, lmax_ket: Int, dmax_bra: Int, dmax_ket: Int, nctr_ket: Int):
        var big_l = 2 * lmax_bra + 2 * lmax_ket
        var cube = (big_l + 1) * (big_l + 1) * (big_l + 1)
        var ncab = ncart(lmax_bra) * ncart(lmax_bra)
        var ncd = ncart(lmax_ket) * ncart(lmax_ket)
        var nm = nherm(2 * lmax_ket)
        var nout = dmax_bra * dmax_bra * dmax_ket * dmax_ket
        self.f = List[Float64](length=big_l + 1 + BoysTable.NTERMS, fill=0.0)
        self.r0 = List[Float64](length=cube, fill=0.0)
        self.r1 = List[Float64](length=cube, fill=0.0)
        self.bm = List[Float64](length=(nm + 1) * padded(ncab), fill=0.0)
        self.blk = List[Float64](length=(ncab + 1) * padded(ncd), fill=0.0)
        self.acc = List[Float64](length=nctr_ket * nctr_ket * ncab * ncd, fill=0.0)
        self.out = List[Float64](length=nout, fill=0.0)
        self.tmp = List[Float64](length=nout, fill=0.0)
        self.hoff = List[Int](length=nherm(2 * lmax_bra), fill=0)
        self.koff = List[Int](length=nm, fill=0)
        self.midx = List[Int](length=nm, fill=0)


# --------------------------------------------------------------------------
# Register-tiled contraction kernels
# --------------------------------------------------------------------------


def contract_tile[NACC: Int, TWO: Bool](
    nk: Int, idx: IntPtr, xbase: F64Ptr, xbase2: F64Ptr, e: F64Ptr, estride: Int, dst: F64Ptr, dst2: F64Ptr
):
    """dst[j] = sum_k xbase[idx[k]] e[k estride + j] for j < NACC W; with TWO also for (xbase2, dst2).

    Up to eight independent SIMD accumulators keep the FMA pipeline full.
    """
    var a0 = SIMD[DType.float64, W](0.0)
    var a1 = SIMD[DType.float64, W](0.0)
    var a2 = SIMD[DType.float64, W](0.0)
    var a3 = SIMD[DType.float64, W](0.0)
    var b0 = SIMD[DType.float64, W](0.0)
    var b1 = SIMD[DType.float64, W](0.0)
    var b2 = SIMD[DType.float64, W](0.0)
    var b3 = SIMD[DType.float64, W](0.0)
    for k in range(nk):
        var o = idx[unsafe_offset=k]
        var x = xbase[unsafe_offset=o]
        var y = 0.0
        comptime if TWO:
            y = xbase2[unsafe_offset=o]
        var row = e.unsafe_offset(k * estride)
        var e0 = row.unsafe_load[width=W](0)
        a0 += e0 * x
        comptime if TWO:
            b0 += e0 * y
        comptime if NACC > 1:
            var e1 = row.unsafe_load[width=W](W)
            a1 += e1 * x
            comptime if TWO:
                b1 += e1 * y
        comptime if NACC > 2:
            var e2 = row.unsafe_load[width=W](2 * W)
            a2 += e2 * x
            comptime if TWO:
                b2 += e2 * y
        comptime if NACC > 3:
            var e3 = row.unsafe_load[width=W](3 * W)
            a3 += e3 * x
            comptime if TWO:
                b3 += e3 * y
    dst.unsafe_store(0, a0)
    comptime if TWO:
        dst2.unsafe_store(0, b0)
    comptime if NACC > 1:
        dst.unsafe_store(W, a1)
        comptime if TWO:
            dst2.unsafe_store(W, b1)
    comptime if NACC > 2:
        dst.unsafe_store(2 * W, a2)
        comptime if TWO:
            dst2.unsafe_store(2 * W, b2)
    comptime if NACC > 3:
        dst.unsafe_store(3 * W, a3)
        comptime if TWO:
            dst2.unsafe_store(3 * W, b3)


def contract_rows[TWO: Bool](
    nk: Int, idx: IntPtr, xbase: F64Ptr, xbase2: F64Ptr, e: F64Ptr, estride: Int, dst: F64Ptr, dst2: F64Ptr, nvec: Int
):
    """One (or two) output rows of ``nvec`` SIMD vectors: dst[j] = sum_k xbase[idx[k]] e[k][j]."""
    var j = 0
    while j + 4 <= nvec:
        contract_tile[4, TWO](
            nk, idx, xbase, xbase2, e.unsafe_offset(j * W), estride, dst.unsafe_offset(j * W), dst2.unsafe_offset(j * W)
        )
        j += 4
    var rem = nvec - j
    if rem == 3:
        contract_tile[3, TWO](
            nk, idx, xbase, xbase2, e.unsafe_offset(j * W), estride, dst.unsafe_offset(j * W), dst2.unsafe_offset(j * W)
        )
    elif rem == 2:
        contract_tile[2, TWO](
            nk, idx, xbase, xbase2, e.unsafe_offset(j * W), estride, dst.unsafe_offset(j * W), dst2.unsafe_offset(j * W)
        )
    elif rem == 1:
        contract_tile[1, TWO](
            nk, idx, xbase, xbase2, e.unsafe_offset(j * W), estride, dst.unsafe_offset(j * W), dst2.unsafe_offset(j * W)
        )


# --------------------------------------------------------------------------
# Electron repulsion integrals
# --------------------------------------------------------------------------


def eri_quartet(bra: PairData, ket: PairData, boys: BoysTable, mut ws: Workspace):
    """Contracted Cartesian (ab|cd) block into ``ws.out``.

    Layout ``[ca][carta][cb][cartb][cc][cartc][cd][cartd]`` (contraction index
    outer, Cartesian component inner, as in pyscf).  Per primitive quartet the
    Hermite Coulomb integrals are contracted with the bra Hermite matrix for
    every ket Hermite index (``Bm[m][cpb] = sum_h R[h + m] E^{ab}[h][cpb]``)
    and then with the signed ket Hermite matrix
    (``blk[cpb][cpk] = sum_m Bm[m][cpb] E^{cd}[m][cpk]``); both are small dense
    matrix products run through the register-tiled kernels above.  Primitive
    blocks are contracted over the ket per bra primitive pair and over the bra
    once per bra primitive pair, so general contractions cost little extra.
    """
    var big_l = bra.lab + ket.lab
    var s1 = big_l + 1
    var s2 = s1 * s1
    var nh = bra.nh
    var nm = ket.nh
    var sb = bra.stride
    var sk = ket.stride
    var nvec_b = sb // W
    var nvec_k = sk // W
    for h in range(nh):
        ws.hoff[h] = bra.ht[h] * s2 + bra.hu[h] * s1 + bra.hv[h]
    for m in range(nm):
        ws.koff[m] = ket.ht[m] * s2 + ket.hu[m] * s1 + ket.hv[m]
        ws.midx[m] = m * sb
    var ncab = bra.ncab
    var ncd = ket.ncab
    var nblk = ncab * ncd
    var nkc = ket.nca * ket.ncb
    var nout = bra.nca * bra.ncb * nkc * nblk
    var pf = list_ptr(ws.f)
    var r0 = list_ptr(ws.r0)
    var r1 = list_ptr(ws.r1)
    var bm = list_ptr(ws.bm)
    var blk = list_ptr(ws.blk)
    var acc = list_ptr(ws.acc)
    var out = list_ptr(ws.out)
    var hoff = int_ptr(ws.hoff)
    var koff = int_ptr(ws.koff)
    var midx = int_ptr(ws.midx)
    var hb_all = list_ptr(bra.hmat)
    var hk_all = list_ptr(ket.hmat)
    vfill(out, nout, 0.0)
    var segmented = bra.nca == 1 and bra.ncb == 1 and nkc == 1
    var ncartb = bra.ncartb
    var ncartc = ket.ncarta
    var ncartd = ket.ncartb
    if big_l == 0:
        # (ss|ss): the Hermite matrices are the single numbers K_ab and K_cd.
        var nbc = bra.nca * bra.ncb
        for kb in range(bra.np):
            var p = bra.p[kb]
            var px = bra.center(kb, 0)
            var py = bra.center(kb, 1)
            var pz = bra.center(kb, 2)
            var kab = hb_all[unsafe_offset=kb * bra.hsize]
            vfill(acc, nkc, 0.0)
            for kk in range(ket.np):
                var q = ket.p[kk]
                var dx = px - ket.center(kk, 0)
                var dy = py - ket.center(kk, 1)
                var dz = pz - ket.center(kk, 2)
                var alpha = p * q / (p + q)
                boys.eval(0, alpha * (dx * dx + dy * dy + dz * dz), pf)
                var val = TWO_PI_52 / (p * q * sqrt(p + q)) * pf[unsafe_offset=0] * kab * hk_all[unsafe_offset=kk * ket.hsize]
                for kc in range(nkc):
                    acc[unsafe_offset=kc] += val * ket.coef[kk * nkc + kc]
            for bc in range(nbc):
                var w = bra.coef[kb * nbc + bc]
                for kc in range(nkc):
                    out[unsafe_offset=bc * nkc + kc] += w * acc[unsafe_offset=kc]
        return
    for kb in range(bra.np):
        var p = bra.p[kb]
        var px = bra.center(kb, 0)
        var py = bra.center(kb, 1)
        var pz = bra.center(kb, 2)
        var hb = hb_all.unsafe_offset(kb * bra.hsize)
        vfill(acc, nkc * nblk, 0.0)
        for kk in range(ket.np):
            var q = ket.p[kk]
            var alpha = p * q / (p + q)
            var pref = TWO_PI_52 / (p * q * sqrt(p + q))
            hermite_r(big_l, alpha, px - ket.center(kk, 0), py - ket.center(kk, 1), pz - ket.center(kk, 2), boys, pf, r0, r1)
            # Step A: Bm[m][cpb] = sum_h R[koff[m] + hoff[h]] E^{ab}[h][cpb], two m at a time.
            var m = 0
            while m + 2 <= nm:
                contract_rows[True](
                    nh, hoff, r0.unsafe_offset(koff[m]), r0.unsafe_offset(koff[m + 1]), hb, sb,
                    bm.unsafe_offset(m * sb), bm.unsafe_offset((m + 1) * sb), nvec_b,
                )
                m += 2
            if m < nm:
                contract_rows[False](nh, hoff, r0.unsafe_offset(koff[m]), r0, hb, sb, bm.unsafe_offset(m * sb), bm, nvec_b)
            # Step B: blk[cpb][cpk] = sum_m Bm[m][cpb] E^{cd}[m][cpk], two cpb at a time.
            var hk = hk_all.unsafe_offset(kk * ket.hsize)
            var cpb = 0
            while cpb + 2 <= ncab:
                contract_rows[True](
                    nm, midx, bm.unsafe_offset(cpb), bm.unsafe_offset(cpb + 1), hk, sk,
                    blk.unsafe_offset(cpb * sk), blk.unsafe_offset((cpb + 1) * sk), nvec_k,
                )
                cpb += 2
            if cpb < ncab:
                contract_rows[False](nm, midx, bm.unsafe_offset(cpb), bm, hk, sk, blk.unsafe_offset(cpb * sk), blk, nvec_k)
            # Step C1: ket contraction coefficients (and the prefactor).
            for kc in range(nkc):
                var w = pref * ket.coef[kk * nkc + kc]
                var a = acc.unsafe_offset(kc * nblk)
                for cpb2 in range(ncab):
                    vaxpy(a.unsafe_offset(cpb2 * ncd), ncd, w, blk.unsafe_offset(cpb2 * sk))
        # Step C2: bra contraction coefficients.
        if segmented:
            vaxpy(out, nblk, bra.coef[kb], acc)
        else:
            for ca in range(bra.nca):
                for cb in range(bra.ncb):
                    var w = bra.coef[(kb * bra.nca + ca) * bra.ncb + cb]
                    for cc in range(ket.nca):
                        for cd in range(ket.ncb):
                            var a = acc.unsafe_offset((cc * ket.ncb + cd) * nblk)
                            for ia in range(bra.ncarta):
                                for ib in range(ncartb):
                                    var cpb2 = ia * ncartb + ib
                                    for ic in range(ncartc):
                                        var src = a.unsafe_offset((cpb2 * ncartc + ic) * ncartd)
                                        var dst = out.unsafe_offset(
                                            (((((ca * bra.ncarta + ia) * bra.ncb + cb) * ncartb + ib) * ket.nca + cc) * ncartc + ic) * ket.ncb * ncartd
                                            + cd * ncartd
                                        )
                                        vaxpy(dst, ncartd, w, src)


def _axis_shell(bra: PairData, ket: PairData, ax: Int) -> Int:
    if ax == 0:
        return bra.a
    if ax == 1:
        return bra.b
    if ax == 2:
        return ket.a
    return ket.b


def quartet_to_sph(bra: PairData, ket: PairData, ba: Basis, bk: Basis, mut ws: Workspace) -> Int:
    """Transform ``ws.out`` to the final (spherical or Cartesian) functions; result in ``ws.out``.

    Returns the block length.  Dimensions after the transform are
    ``nctr * nf`` per index.
    """
    var d0 = bra.nca * bra.ncarta
    var d1 = bra.ncb * bra.ncartb
    var d2 = ket.nca * ket.ncarta
    var d3 = ket.ncb * ket.ncartb
    var src = list_ptr(ws.out)
    var dst = list_ptr(ws.tmp)
    var in_out = True
    for ax in range(4):
        var sh = _axis_shell(bra, ket, ax)
        if sh < 0:
            continue
        var is_bra = ax < 2
        var l = ba.l[sh] if is_bra else bk.l[sh]
        var needs = ba.needs_transform(sh) if is_bra else bk.needs_transform(sh)
        if not needs:
            continue
        var nc = ba.nctr[sh] if is_bra else bk.nctr[sh]
        var nf = ba.nf[l] if is_bra else bk.nf[l]
        var c2s = list_ptr(ba.c2s).unsafe_offset(ba.c2s_off[l]) if is_bra else list_ptr(bk.c2s).unsafe_offset(bk.c2s_off[l])
        var ncartx = ncart(l)
        var pre = nc
        var post = 1
        if ax >= 1:
            pre *= d0
        if ax >= 2:
            pre *= d1
        if ax >= 3:
            pre *= d2
        if ax <= 0:
            post *= d1
        if ax <= 1:
            post *= d2
        if ax <= 2:
            post *= d3
        if in_out:
            transform_axis(src, dst, pre, ncartx, nf, post, c2s)
        else:
            transform_axis(dst, src, pre, ncartx, nf, post, c2s)
        in_out = not in_out
        var nd = nc * nf
        if ax == 0:
            d0 = nd
        elif ax == 1:
            d1 = nd
        elif ax == 2:
            d2 = nd
        else:
            d3 = nd
    var n = d0 * d1 * d2 * d3
    if not in_out:
        for i in range(n):
            src[unsafe_offset=i] = dst[unsafe_offset=i]
    return n


def block_max_abs(ws: Workspace, n: Int) -> Float64:
    var m = 0.0
    for i in range(n):
        var v = ws.out[i]
        if v < 0.0:
            v = -v
        if v > m:
            m = v
    return m


def scatter_s8(
    basis: Basis, a: Int, b: Int, c: Int, d: Int, blk: F64Ptr, na: Int, nb: Int, nc: Int, nd: Int, eri: F64Ptr
):
    """Store the canonical elements of the spherical block (ab|cd) in the 8-fold packed vector."""
    var i0 = basis.ao_loc[a]
    var j0 = basis.ao_loc[b]
    var k0 = basis.ao_loc[c]
    var l0 = basis.ao_loc[d]
    var same_pair = (a == c) and (b == d)
    for fa in range(na):
        var i = i0 + fa
        for fb in range(nb):
            var j = j0 + fb
            if j > i:
                continue
            var ij = i * (i + 1) // 2 + j
            var row = (fa * nb + fb) * nc
            for fc in range(nc):
                var k = k0 + fc
                var src = blk.unsafe_offset((row + fc) * nd)
                var kl0 = k * (k + 1) // 2 + l0
                # eligible l range: l <= k always; l <= k also bounds kl <= ij when i == k
                var ndd = nd
                if l0 + ndd - 1 > k:
                    ndd = k - l0 + 1
                if i > k:
                    # every kl is below ij: contiguous row segment
                    var dst = eri.unsafe_offset(ij * (ij + 1) // 2 + kl0)
                    for fd in range(ndd):
                        dst[unsafe_offset=fd] = src[unsafe_offset=fd]
                else:
                    for fd in range(ndd):
                        var kl = kl0 + fd
                        if same_pair and kl > ij:
                            break
                        if ij >= kl:
                            eri[unsafe_offset=ij * (ij + 1) // 2 + kl] = src[unsafe_offset=fd]
                        else:
                            eri[unsafe_offset=kl * (kl + 1) // 2 + ij] = src[unsafe_offset=fd]


def pair_shells(sp: Int) -> Tuple[Int, Int]:
    """(a, b) with a >= b for the shell-pair index sp = a (a + 1) / 2 + b."""
    var a = Int((sqrt(8.0 * Float64(sp) + 1.0) - 1.0) / 2.0)
    while a * (a + 1) // 2 > sp:
        a -= 1
    while (a + 1) * (a + 2) // 2 <= sp:
        a += 1
    return (a, sp - a * (a + 1) // 2)


# Bra shell pairs handled together by one worker; every ket pair is set up
# once per chunk instead of once per quartet.  Small systems use smaller
# chunks so that every worker still gets several tasks.
comptime BRA_CHUNK = 8


def chunk_size(npairs: Int, nworkers: Int) -> Int:
    return max(1, min(BRA_CHUNK, npairs // (8 * nworkers)))


def eri_s8_core(basis: Basis, boys: BoysTable, eri: F64Ptr, schwarz_tol: Float64):
    """8-fold packed ERIs of ``basis`` into ``eri`` (length npair (npair + 1) / 2).

    Shell quartets whose Schwarz bound ``sqrt((ab|ab)) sqrt((cd|cd))`` falls
    below ``schwarz_tol`` are skipped (their elements are left at zero), so a
    non-positive tolerance computes every quartet.  Worker threads pull chunks
    of bra shell pairs (largest pair index, i.e. most ket pairs, first) from a
    shared counter; each worker owns one set of scratch buffers.
    """
    var nbas = basis.nbas
    var npairs = nbas * (nbas + 1) // 2
    var nao = basis.nao
    var npair_ao = nao * (nao + 1) // 2
    var lmax = basis.lmax
    var nworkers = min(parallelism_level(), npairs)
    # Zero the output in parallel chunks (also spreads the first touch of a
    # freshly allocated array over the threads).
    var neri = npair_ao * (npair_ao + 1) // 2
    var nfill = (neri + (1 << 20) - 1) >> 20

    def fill(c: Int) {imm eri, imm neri}:
        var start = c << 20
        vfill(eri.unsafe_offset(start), min(1 << 20, neri - start), 0.0)

    parallelize(fill, nfill)

    # Schwarz bounds per shell pair.
    var qb = List[Float64](length=npairs, fill=0.0)
    var pq = list_ptr(qb)
    var counter = Atomic[Int64](0)
    var pcount = Pointer(to=counter)

    def schwarz(w: Int) {imm basis, imm boys, imm pq, imm lmax, imm pcount, imm npairs}:
        var bra = PairData(lmax, basis.nprim_max, basis.nctr_max, False)
        var ket = PairData(lmax, basis.nprim_max, basis.nctr_max, True)
        var ws = Workspace(lmax, lmax, basis.dmax, basis.dmax, basis.nctr_max)
        while True:
            var sp = Int(pcount[].fetch_add(1))
            if sp >= npairs:
                break
            var ab = pair_shells(sp)
            bra.fill(basis, ab[0], basis, ab[1])
            ket.fill(basis, ab[0], basis, ab[1])
            if bra.np == 0:
                continue
            eri_quartet(bra, ket, boys, ws)
            var n = quartet_to_sph(bra, ket, basis, basis, ws)
            pq[unsafe_offset=sp] = sqrt(block_max_abs(ws, n))
        _ = bra^
        _ = ket^
        _ = ws^

    parallelize(schwarz, nworkers)
    counter.store(0)
    var chunk = chunk_size(npairs, nworkers)
    var ntasks = (npairs + chunk - 1) // chunk

    def work(w: Int) {imm basis, imm boys, imm pq, imm lmax, imm eri, imm schwarz_tol, imm npairs, imm pcount, imm ntasks, imm chunk}:
        var bras = List[PairData]()
        for g in range(chunk):
            bras.append(PairData(lmax, basis.nprim_max, basis.nctr_max, False))
        var ket = PairData(lmax, basis.nprim_max, basis.nctr_max, True)
        var ws = Workspace(lmax, lmax, basis.dmax, basis.dmax, basis.nctr_max)
        var sps = List[Int](length=chunk, fill=-1)
        while True:
            var task = Int(pcount[].fetch_add(1))
            if task >= ntasks:
                break
            # chunk of bra pairs sp = spmax, spmax - 1, ...
            var spmax = npairs - 1 - task * chunk
            var ng = 0
            var qmax = 0.0
            for g in range(chunk):
                var sp = spmax - g
                if sp < 0:
                    break
                var ab = pair_shells(sp)
                bras[g].fill(basis, ab[0], basis, ab[1])
                sps[g] = sp
                ng += 1
                qmax = max(qmax, pq[unsafe_offset=sp])
            for spk in range(spmax + 1):
                var qk = pq[unsafe_offset=spk]
                if qmax * qk < schwarz_tol:
                    continue
                var cd = pair_shells(spk)
                var c = cd[0]
                var d = cd[1]
                var filled = False
                for g in range(ng):
                    var sp = sps[g]
                    if spk > sp or bras[g].np == 0 or pq[unsafe_offset=sp] * qk < schwarz_tol:
                        continue
                    if not filled:
                        ket.fill(basis, c, basis, d)
                        filled = True
                    if ket.np == 0:
                        break
                    var ab = pair_shells(sp)
                    var a = ab[0]
                    var b = ab[1]
                    eri_quartet(bras[g], ket, boys, ws)
                    _ = quartet_to_sph(bras[g], ket, basis, basis, ws)
                    scatter_s8(
                        basis, a, b, c, d, list_ptr(ws.out),
                        basis.ao_loc[a + 1] - basis.ao_loc[a], basis.ao_loc[b + 1] - basis.ao_loc[b],
                        basis.ao_loc[c + 1] - basis.ao_loc[c], basis.ao_loc[d + 1] - basis.ao_loc[d], eri,
                    )
        _ = bras^
        _ = ket^
        _ = ws^
        _ = sps^

    parallelize(work, nworkers)
    _ = qb^
    _ = counter^


def int3c2e_core(basis: Basis, aux: Basis, boys: BoysTable, dst: F64Ptr):
    """(ab|P) into ``dst[P * npair + pair(a, b)]`` (naux x npair, pyscf's cderi layout before the Cholesky step)."""
    var nbas = basis.nbas
    var npairs = nbas * (nbas + 1) // 2
    var nao = basis.nao
    var npair_ao = nao * (nao + 1) // 2
    var lmax = basis.lmax
    var lmax_aux = aux.lmax
    var nworkers = min(parallelism_level(), npairs)
    var counter = Atomic[Int64](0)
    var pcount = Pointer(to=counter)
    var chunk = chunk_size(npairs, nworkers)
    var ntasks = (npairs + chunk - 1) // chunk

    def work(w: Int) {imm basis, imm aux, imm boys, imm lmax, imm lmax_aux, imm dst, imm npairs, imm npair_ao, imm pcount, imm ntasks, imm chunk}:
        var bras = List[PairData]()
        for g in range(chunk):
            bras.append(PairData(lmax, basis.nprim_max, basis.nctr_max, False))
        var ket = PairData(lmax_aux, aux.nprim_max, aux.nctr_max, True)
        var ws = Workspace(lmax, lmax_aux, basis.dmax, aux.dmax, aux.nctr_max)
        var sps = List[Int](length=chunk, fill=-1)
        while True:
            var task = Int(pcount[].fetch_add(1))
            if task >= ntasks:
                break
            var spmax = npairs - 1 - task * chunk
            var ng = 0
            for g in range(chunk):
                var sp = spmax - g
                if sp < 0:
                    break
                var ab = pair_shells(sp)
                bras[g].fill(basis, ab[0], basis, ab[1])
                sps[g] = sp
                ng += 1
            for pshell in range(aux.nbas):
                ket.fill(aux, pshell, aux, -1)
                var p0 = aux.ao_loc[pshell]
                var npf = aux.ao_loc[pshell + 1] - p0
                for g in range(ng):
                    if bras[g].np == 0:
                        continue
                    var ab = pair_shells(sps[g])
                    var a = ab[0]
                    var b = ab[1]
                    eri_quartet(bras[g], ket, boys, ws)
                    _ = quartet_to_sph(bras[g], ket, basis, aux, ws)
                    var i0 = basis.ao_loc[a]
                    var j0 = basis.ao_loc[b]
                    var na = basis.ao_loc[a + 1] - i0
                    var nb = basis.ao_loc[b + 1] - j0
                    for fa in range(na):
                        var i = i0 + fa
                        for fb in range(nb):
                            var j = j0 + fb
                            if j > i:
                                continue
                            var ij = i * (i + 1) // 2 + j
                            var src = (fa * nb + fb) * npf
                            for fp in range(npf):
                                dst[unsafe_offset=(p0 + fp) * npair_ao + ij] = ws.out[src + fp]
        _ = bras^
        _ = ket^
        _ = ws^
        _ = sps^

    parallelize(work, nworkers)
    _ = counter^


def int2c2e_core(aux: Basis, boys: BoysTable, dst: F64Ptr):
    """(P|Q) into the dense ``(naux, naux)`` matrix ``dst``."""
    var naux = aux.nao
    var lmax = aux.lmax
    var nworkers = min(parallelism_level(), aux.nbas)
    var counter = Atomic[Int64](0)
    var pcount = Pointer(to=counter)

    def work(w: Int) {imm aux, imm boys, imm lmax, imm dst, imm naux, imm pcount}:
        var bra = PairData(lmax, aux.nprim_max, aux.nctr_max, False)
        var ket = PairData(lmax, aux.nprim_max, aux.nctr_max, True)
        var ws = Workspace(lmax, lmax, aux.dmax, aux.dmax, aux.nctr_max)
        while True:
            var task = Int(pcount[].fetch_add(1))
            if task >= aux.nbas:
                break
            var pshell = aux.nbas - 1 - task
            bra.fill(aux, pshell, aux, -1)
            var p0 = aux.ao_loc[pshell]
            var npf = aux.ao_loc[pshell + 1] - p0
            for qshell in range(pshell + 1):
                ket.fill(aux, qshell, aux, -1)
                eri_quartet(bra, ket, boys, ws)
                _ = quartet_to_sph(bra, ket, aux, aux, ws)
                var q0 = aux.ao_loc[qshell]
                var nqf = aux.ao_loc[qshell + 1] - q0
                for fp in range(npf):
                    for fq in range(nqf):
                        var v = ws.out[fp * nqf + fq]
                        dst[unsafe_offset=(p0 + fp) * naux + q0 + fq] = v
                        dst[unsafe_offset=(q0 + fq) * naux + p0 + fp] = v
        _ = bra^
        _ = ket^
        _ = ws^

    parallelize(work, nworkers)
    _ = counter^


# --------------------------------------------------------------------------
# One-electron integrals
# --------------------------------------------------------------------------


def int1e_core(basis: Basis, boys: BoysTable, s_out: F64Ptr, t_out: F64Ptr, v_out: F64Ptr):
    """Overlap, kinetic energy and nuclear attraction matrices (nao x nao, row-major)."""
    var nbas = basis.nbas
    var npairs = nbas * (nbas + 1) // 2
    var nao = basis.nao
    var lmax = basis.lmax
    var natm = basis.natm

    def work(sp: Int) {imm basis, imm boys, imm lmax, imm natm, imm nao, imm s_out, imm t_out, imm v_out}:
        var ab = pair_shells(sp)
        var a = ab[0]
        var b = ab[1]
        var la = basis.l[a]
        var lb = basis.l[b]
        var lab = la + lb
        var nca = basis.nctr[a]
        var ncb = basis.nctr[b]
        var ncarta = ncart(la)
        var ncartb = ncart(lb)
        var npa = basis.nprim[a]
        var npb = basis.nprim[b]
        var ax = basis.shell_coord(a, 0)
        var ay = basis.shell_coord(a, 1)
        var az = basis.shell_coord(a, 2)
        var bx = basis.shell_coord(b, 0)
        var by = basis.shell_coord(b, 1)
        var bz = basis.shell_coord(b, 2)
        var rab2 = (ax - bx) * (ax - bx) + (ay - by) * (ay - by) + (az - bz) * (az - bz)
        var fac = basis.func_scale(a) * basis.func_scale(b)
        var offa = basis.cart_off[la]
        var offb = basis.cart_off[lb]
        # E arrays with lb + 2 (kinetic energy needs j + 2)
        var lb3 = lb + 3
        var s = lab + 3
        var esz = (la + 1) * lb3 * s
        var e = List[Float64](length=3 * esz, fill=0.0)
        var pe = list_ptr(e)
        var cube = (lab + 1) * (lab + 1) * (lab + 1)
        var r0 = List[Float64](length=cube, fill=0.0)
        var r1 = List[Float64](length=cube, fill=0.0)
        var pr0 = list_ptr(r0)
        var pr1 = list_ptr(r1)
        var fb = List[Float64](length=lab + 1 + BoysTable.NTERMS, fill=0.0)
        var pfb = list_ptr(fb)
        var nblk = ncarta * ncartb
        var ncab = nca * ncarta * ncb * ncartb
        var sprim = List[Float64](length=nblk, fill=0.0)
        var tprim = List[Float64](length=nblk, fill=0.0)
        var vprim = List[Float64](length=nblk, fill=0.0)
        var sblk = List[Float64](length=ncab, fill=0.0)
        var tblk = List[Float64](length=ncab, fill=0.0)
        var vblk = List[Float64](length=ncab, fill=0.0)
        var s1 = lab + 1
        var s2 = s1 * s1
        for ia in range(npa):
            var ea = basis.env[basis.pexp[a] + ia]
            for ib in range(npb):
                var eb = basis.env[basis.pexp[b] + ib]
                var p = ea + eb
                var mu = ea * eb / p
                if mu * rab2 > EXP_CUTOFF:
                    continue
                var px = (ea * ax + eb * bx) / p
                var py = (ea * ay + eb * by) / p
                var pz = (ea * az + eb * bz) / p
                hermite_e(la, lb + 2, p, px - ax, px - bx, exp(-mu * (ax - bx) * (ax - bx)), pe)
                hermite_e(la, lb + 2, p, py - ay, py - by, exp(-mu * (ay - by) * (ay - by)), pe.unsafe_offset(esz))
                hermite_e(la, lb + 2, p, pz - az, pz - bz, exp(-mu * (az - bz) * (az - bz)), pe.unsafe_offset(2 * esz))
                var sp_ = sqrt(PI / p)
                var vpref = -2.0 * PI / p
                for ca in range(ncarta):
                    var i = basis.cx[offa + ca]
                    var k = basis.cy[offa + ca]
                    var m = basis.cz[offa + ca]
                    for cb in range(ncartb):
                        var j = basis.cx[offb + cb]
                        var l = basis.cy[offb + cb]
                        var n = basis.cz[offb + cb]
                        # one-dimensional overlap and kinetic factors
                        var ex = (i * lb3 + j) * s
                        var ey = esz + (k * lb3 + l) * s
                        var ez = 2 * esz + (m * lb3 + n) * s
                        var sx = e[ex] * sp_
                        var sy = e[ey] * sp_
                        var sz = e[ez] * sp_
                        var tx = -2.0 * eb * eb * e[ex + 2 * s] * sp_ + eb * Float64(2 * j + 1) * sx
                        if j >= 2:
                            tx -= 0.5 * Float64(j * (j - 1)) * e[ex - 2 * s] * sp_
                        var ty = -2.0 * eb * eb * e[ey + 2 * s] * sp_ + eb * Float64(2 * l + 1) * sy
                        if l >= 2:
                            ty -= 0.5 * Float64(l * (l - 1)) * e[ey - 2 * s] * sp_
                        var tz = -2.0 * eb * eb * e[ez + 2 * s] * sp_ + eb * Float64(2 * n + 1) * sz
                        if n >= 2:
                            tz -= 0.5 * Float64(n * (n - 1)) * e[ez - 2 * s] * sp_
                        sprim[ca * ncartb + cb] = sx * sy * sz
                        tprim[ca * ncartb + cb] = tx * sy * sz + sx * ty * sz + sx * sy * tz
                        vprim[ca * ncartb + cb] = 0.0
                # nuclear attraction: one R cube per atom
                for at in range(natm):
                    var z = basis.charge[at]
                    if z == 0.0:
                        continue
                    hermite_r(
                        lab, p, px - basis.coord[3 * at], py - basis.coord[3 * at + 1], pz - basis.coord[3 * at + 2], boys, pfb, pr0, pr1
                    )
                    for ca in range(ncarta):
                        var i = basis.cx[offa + ca]
                        var k = basis.cy[offa + ca]
                        var m = basis.cz[offa + ca]
                        for cb in range(ncartb):
                            var j = basis.cx[offb + cb]
                            var l = basis.cy[offb + cb]
                            var n = basis.cz[offb + cb]
                            var ex = (i * lb3 + j) * s
                            var ey = esz + (k * lb3 + l) * s
                            var ez = 2 * esz + (m * lb3 + n) * s
                            var acc = 0.0
                            for t in range(i + j + 1):
                                var vx = e[ex + t]
                                for u in range(k + l + 1):
                                    var vxy = vx * e[ey + u]
                                    var rr = pr0.unsafe_offset(t * s2 + u * s1)
                                    for v in range(m + n + 1):
                                        acc += vxy * e[ez + v] * rr[unsafe_offset=v]
                            vprim[ca * ncartb + cb] += vpref * z * acc
                # contraction
                for ca in range(nca):
                    var cfa = basis.env[basis.pcoef[a] + ca * npa + ia] * fac
                    for cb in range(ncb):
                        var w = cfa * basis.env[basis.pcoef[b] + cb * npb + ib]
                        for ia2 in range(ncarta):
                            var dst = ((ca * ncarta + ia2) * ncb + cb) * ncartb
                            var src = ia2 * ncartb
                            for ib2 in range(ncartb):
                                sblk[dst + ib2] += w * sprim[src + ib2]
                                tblk[dst + ib2] += w * tprim[src + ib2]
                                vblk[dst + ib2] += w * vprim[src + ib2]
        # transform and store
        var na = basis.ao_loc[a + 1] - basis.ao_loc[a]
        var nb = basis.ao_loc[b + 1] - basis.ao_loc[b]
        var i0 = basis.ao_loc[a]
        var j0 = basis.ao_loc[b]
        var tmp = List[Float64](length=max(ncab, na * nb), fill=0.0)
        var ptmp = list_ptr(tmp)
        for which in range(3):
            var blk = list_ptr(sblk) if which == 0 else (list_ptr(tblk) if which == 1 else list_ptr(vblk))
            var mat = s_out if which == 0 else (t_out if which == 1 else v_out)
            var cur = blk
            var oth = ptmp
            var d0 = nca * ncarta
            var d1 = ncb * ncartb
            if basis.needs_transform(a):
                transform_axis(cur, oth, nca, ncarta, basis.nf[la], d1, list_ptr(basis.c2s).unsafe_offset(basis.c2s_off[la]))
                d0 = na
                var sw = cur
                cur = oth
                oth = sw
            if basis.needs_transform(b):
                transform_axis(cur, oth, d0 * ncb, ncartb, basis.nf[lb], 1, list_ptr(basis.c2s).unsafe_offset(basis.c2s_off[lb]))
                d1 = nb
                var sw = cur
                cur = oth
                oth = sw
            for fa in range(na):
                for fbb in range(nb):
                    var v = cur[unsafe_offset=fa * nb + fbb]
                    mat[unsafe_offset=(i0 + fa) * nao + j0 + fbb] = v
                    mat[unsafe_offset=(j0 + fbb) * nao + i0 + fa] = v
        _ = e^
        _ = r0^
        _ = r1^
        _ = fb^
        _ = sprim^
        _ = tprim^
        _ = vprim^
        _ = sblk^
        _ = tblk^
        _ = vblk^
        _ = tmp^

    parallelize(work, npairs)
