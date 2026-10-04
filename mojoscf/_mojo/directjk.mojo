"""Direct-SCF Coulomb and exchange matrices from the Mojo integral engine.

``DirectJK`` builds the shell-pair table and the Schwarz bounds of a basis
once; ``jk`` then evaluates J and K of symmetric densities by recomputing the
electron repulsion integrals of every significant shell quartet and folding
them into the matrices (the integrals are never stored).  A quartet
(ab|cd) with a >= b, c >= d, ab >= cd stands for its eight permutations;
it is skipped when

    sqrt((ab|ab)) sqrt((cd|cd)) max(4 D_ab, 4 D_cd, D_ac, D_ad, D_bc, D_bd) < tol

with D_xy the largest density element in shell block (x, y), the same test
as pyscf's libcvhf (``direct_scf_tol``).  Contributions are accumulated with
the weight 1/2 for each coinciding index pair into per-thread matrices A,
and J = A_J + A_J^T, K = A_K + A_K^T at the end:

    A_J[i,j] += 2 s sum_kl (ij|kl) D_kl      A_J[k,l] += 2 s sum_ij (ij|kl) D_ij
    A_K[i,k] += s sum_jl (ij|kl) D_jl        A_K[j,k] += s sum_il (ij|kl) D_il
    A_K[i,l] += s sum_jk (ij|kl) D_jk        A_K[j,l] += s sum_ik (ij|kl) D_ik
"""
from std.atomic import Atomic
from std.math import sqrt
from std.memory import Pointer
from std.python import PythonObject
from std.runtime import parallelism_level
from max.algorithm import parallelize
from _mojo.linalg import F64Ptr, list_ptr, vfill, vaxpy
from _mojo.integrals import (
    Basis, BoysTable, HermTable, PairTable, EriWork, I64Ptr, eri_quartet, schwarz_bounds, shell_nfunc,
    I_A, I_B, I_NP,
)


def _ptr_f64(arr: PythonObject) raises -> F64Ptr:
    return F64Ptr(unsafe_from_address=Int(py=arr.__array_interface__["data"][0]))


def _ptr_i64(arr: PythonObject) raises -> I64Ptr:
    return I64Ptr(unsafe_from_address=Int(py=arr.__array_interface__["data"][0]))


def basis_from_py(b: PythonObject) raises -> Basis:
    """Basis from the tuple (atm, bas, env, nf, c2s) prepared by ``mojoscf.integrals.basis_tables``."""
    var atm = b[0]
    var bas = b[1]
    var env = b[2]
    var nf = b[3]
    var c2s = b[4]
    return Basis(
        Int(py=atm.shape[0]), _ptr_i64(atm), Int(py=bas.shape[0]), _ptr_i64(bas), Int(py=env.shape[0]), _ptr_f64(env),
        Int(py=nf.shape[0]), _ptr_i64(nf), _ptr_f64(c2s),
    )


def empty_basis() -> Basis:
    var di = List[Int64](length=8, fill=0)
    var df = List[Float64](length=8, fill=0.0)
    var pi = I64Ptr(unsafe_from_address=Int(di.unsafe_ptr()))
    var pf = list_ptr(df)
    var b = Basis(0, pi, 0, pi, 0, pf, 0, pi, pf)
    _ = di^
    _ = df^
    return b^


def digest[DO_J: Bool, DO_K: Bool](
    blk: F64Ptr, na: Int, nb: Int, nc: Int, nd: Int, i0: Int, j0: Int, k0: Int, l0: Int, nao: Int, scale: Float64,
    dj: F64Ptr, aj: F64Ptr, dk: F64Ptr, ak: F64Ptr, loc: F64Ptr,
):
    """Fold the integral block (ij|kl) of one unique shell quartet into the accumulators (module docstring)."""
    var jab = loc
    var jcd = jab.unsafe_offset(na * nb)
    var kac = jcd.unsafe_offset(nc * nd)
    var kbc = kac.unsafe_offset(na * nc)
    var kad = kbc.unsafe_offset(nb * nc)
    var kbd = kad.unsafe_offset(na * nd)
    vfill(loc, na * nb + nc * nd + na * nc + nb * nc + na * nd + nb * nd, 0.0)
    for i in range(na):
        var dad = dk.unsafe_offset((i0 + i) * nao + l0)
        var kadr = kad.unsafe_offset(i * nd)
        for j in range(nb):
            var dij = 0.0
            comptime if DO_J:
                dij = dj[unsafe_offset=(i0 + i) * nao + j0 + j]
            var dbd = dk.unsafe_offset((j0 + j) * nao + l0)
            var kbdr = kbd.unsafe_offset(j * nd)
            var sj = 0.0
            for k in range(nc):
                var g = blk.unsafe_offset(((i * nb + j) * nc + k) * nd)
                var dcd = dj.unsafe_offset((k0 + k) * nao + l0)
                var jrow = jcd.unsafe_offset(k * nd)
                var dbc = 0.0
                var dac = 0.0
                comptime if DO_K:
                    dbc = dk[unsafe_offset=(j0 + j) * nao + k0 + k]
                    dac = dk[unsafe_offset=(i0 + i) * nao + k0 + k]
                var sac = 0.0
                var sbc = 0.0
                for l in range(nd):
                    var v = g[unsafe_offset=l]
                    comptime if DO_J:
                        sj += v * dcd[unsafe_offset=l]
                        jrow[unsafe_offset=l] += v * dij
                    comptime if DO_K:
                        sac += v * dbd[unsafe_offset=l]
                        sbc += v * dad[unsafe_offset=l]
                        kadr[unsafe_offset=l] += v * dbc
                        kbdr[unsafe_offset=l] += v * dac
                comptime if DO_K:
                    kac[unsafe_offset=i * nc + k] += sac
                    kbc[unsafe_offset=j * nc + k] += sbc
            comptime if DO_J:
                jab[unsafe_offset=i * nb + j] += sj
    comptime if DO_J:
        var s2 = 2.0 * scale
        for i in range(na):
            vaxpy(aj.unsafe_offset((i0 + i) * nao + j0), nb, s2, jab.unsafe_offset(i * nb))
        for k in range(nc):
            vaxpy(aj.unsafe_offset((k0 + k) * nao + l0), nd, s2, jcd.unsafe_offset(k * nd))
    comptime if DO_K:
        for i in range(na):
            vaxpy(ak.unsafe_offset((i0 + i) * nao + k0), nc, scale, kac.unsafe_offset(i * nc))
            vaxpy(ak.unsafe_offset((i0 + i) * nao + l0), nd, scale, kad.unsafe_offset(i * nd))
        for j in range(nb):
            vaxpy(ak.unsafe_offset((j0 + j) * nao + k0), nc, scale, kbc.unsafe_offset(j * nc))
            vaxpy(ak.unsafe_offset((j0 + j) * nao + l0), nd, scale, kbd.unsafe_offset(j * nd))


struct DirectJK(Movable):
    """Integral-direct J and K for one basis (see the module docstring)."""

    var active: Bool
    var basis: Basis
    var boys: BoysTable
    var ht: HermTable
    var tab: PairTable
    var q: List[Float64]      # Schwarz bound per shell pair a >= b (index a (a + 1) / 2 + b)
    var nbas: Int
    var nao: Int
    var nfmax: Int

    def __init__(out self, var basis: Basis, var boys: BoysTable):
        var nbas = basis.nbas
        var npairs = nbas * (nbas + 1) // 2
        var ht = HermTable(2 * basis.lmax)
        var sa = List[Int](capacity=npairs)
        var sb = List[Int](capacity=npairs)
        for a in range(nbas):
            for b in range(a + 1):
                sa.append(a)
                sb.append(b)
        var tab = PairTable(basis, basis, sa, sb, ht)
        var q = schwarz_bounds(boys, tab, ht)
        var nfmax = 1
        for a in range(nbas):
            nfmax = max(nfmax, shell_nfunc(basis, a))
        self.active = True
        self.nbas = nbas
        self.nao = basis.nao
        self.nfmax = nfmax
        self.basis = basis^
        self.boys = boys^
        self.ht = ht^
        self.tab = tab^
        self.q = q^
        _ = sa^
        _ = sb^

    def __init__(out self, *, inactive: Bool):
        """Placeholder for drivers that build J and K differently."""
        var basis = empty_basis()
        var ht = HermTable(0)
        var sa = List[Int]()
        var sb = List[Int]()
        self.tab = PairTable(basis, basis, sa, sb, ht)
        self.ht = ht^
        self.active = False
        self.nbas = 0
        self.nao = 0
        self.nfmax = 1
        self.boys = BoysTable(empty=True)
        self.q = List[Float64]()
        self.basis = basis^
        _ = sa^
        _ = sb^

    def jk(self, nj: Int, dmj: F64Ptr, vj: F64Ptr, nk: Int, dmk: F64Ptr, vk: F64Ptr, tol: Float64):
        """vj = J[dmj] (nj <= 1 densities) and vk[s] = K[dmk[s]] (nk densities); outputs overwritten.

        All densities must be symmetric (nao x nao, row-major).
        """
        var nao = self.nao
        var n2 = nao * nao
        var nbas = self.nbas
        var npairs = nbas * (nbas + 1) // 2
        var nacc = nj + nk
        if nacc == 0:
            return
        # largest density element per shell block, over all densities
        var cond = List[Float64](length=max(nbas * nbas, 1), fill=0.0)
        var pcond = list_ptr(cond)
        for a in range(nbas):
            var i0 = self.basis.ao_loc[a]
            var i1 = self.basis.ao_loc[a + 1]
            for b in range(nbas):
                var j0 = self.basis.ao_loc[b]
                var j1 = self.basis.ao_loc[b + 1]
                var m = 0.0
                for s in range(nacc):
                    var d = dmj if s < nj else dmk.unsafe_offset((s - nj) * n2)
                    for i in range(i0, i1):
                        for j in range(j0, j1):
                            var v = abs(d[unsafe_offset=i * nao + j])
                            if v > m:
                                m = v
                pcond[unsafe_offset=a * nbas + b] = m
        var nworkers = min(parallelism_level(), npairs) if npairs >= 16 else 1
        var acc = List[Float64](length=nworkers * nacc * n2, fill=0.0)
        var pacc = list_ptr(acc)
        var pq = list_ptr(self.q)
        var counter = Atomic[Int64](0)
        var pcount = Pointer(to=counter)
        var nfmax = self.nfmax

        def work(w: Int) {imm self, imm pacc, imm pcond, imm pq, imm pcount, imm npairs, imm nbas, imm nao, imm n2, imm nacc, imm nj, imm nk, imm dmj, imm dmk, imm tol, imm nfmax}:
            var ws = EriWork(self.tab.maxcomp, self.tab.maxlab, self.tab.maxcomp, self.tab.maxlab)
            var loc = List[Float64](length=6 * nfmax * nfmax + 8, fill=0.0)
            var ploc = list_ptr(loc)
            var aj = pacc.unsafe_offset(w * nacc * n2)
            var ak = aj.unsafe_offset(nj * n2)
            while True:
                var task = Int(pcount[].fetch_add(1))
                if task >= npairs:
                    break
                var sp = npairs - 1 - task
                var qab = pq[unsafe_offset=sp]
                if qab == 0.0:
                    continue
                var a = self.tab.get(sp, I_A)
                var b = self.tab.get(sp, I_B)
                var i0 = self.basis.ao_loc[a]
                var na = self.basis.ao_loc[a + 1] - i0
                var j0 = self.basis.ao_loc[b]
                var nb = self.basis.ao_loc[b + 1] - j0
                var dab = pcond[unsafe_offset=a * nbas + b]
                for spk in range(sp + 1):
                    var qq = qab * pq[unsafe_offset=spk]
                    if qq < tol:
                        continue
                    var c = self.tab.get(spk, I_A)
                    var d = self.tab.get(spk, I_B)
                    var dmax = 4.0 * max(dab, pcond[unsafe_offset=c * nbas + d])
                    dmax = max(dmax, max(pcond[unsafe_offset=a * nbas + c], pcond[unsafe_offset=a * nbas + d]))
                    dmax = max(dmax, max(pcond[unsafe_offset=b * nbas + c], pcond[unsafe_offset=b * nbas + d]))
                    if qq * dmax < tol:
                        continue
                    if not eri_quartet(self.tab, sp, self.tab, spk, self.ht, self.boys, ws):
                        continue
                    var scale = 1.0
                    if a == b:
                        scale *= 0.5
                    if c == d:
                        scale *= 0.5
                    if sp == spk:
                        scale *= 0.5
                    var k0 = self.basis.ao_loc[c]
                    var nc = self.basis.ao_loc[c + 1] - k0
                    var l0 = self.basis.ao_loc[d]
                    var nd = self.basis.ao_loc[d + 1] - l0
                    var blk = list_ptr(ws.out)
                    if nk == 0:
                        digest[True, False](blk, na, nb, nc, nd, i0, j0, k0, l0, nao, scale, dmj, aj, dmj, ak, ploc)
                    elif nj == 0:
                        for s in range(nk):
                            digest[False, True](
                                blk, na, nb, nc, nd, i0, j0, k0, l0, nao, scale, dmj, aj,
                                dmk.unsafe_offset(s * n2), ak.unsafe_offset(s * n2), ploc,
                            )
                    else:
                        digest[True, True](blk, na, nb, nc, nd, i0, j0, k0, l0, nao, scale, dmj, aj, dmk, ak, ploc)
                        for s in range(1, nk):
                            digest[False, True](
                                blk, na, nb, nc, nd, i0, j0, k0, l0, nao, scale, dmj, aj,
                                dmk.unsafe_offset(s * n2), ak.unsafe_offset(s * n2), ploc,
                            )
            _ = ws^
            _ = loc^

        if nworkers == 1:
            work(0)
        else:
            parallelize(work, nworkers)

        # sum the per-thread accumulators into the first one (rows in parallel)
        var total = nacc * n2
        if nworkers > 1:
            var nchunk = min(64, max(1, total // 65536))

            def reduce(cidx: Int) {imm pacc, imm total, imm nchunk, imm nworkers}:
                var lo = total * cidx // nchunk
                var hi = total * (cidx + 1) // nchunk
                for w2 in range(1, nworkers):
                    vaxpy(pacc.unsafe_offset(lo), hi - lo, 1.0, pacc.unsafe_offset(w2 * total + lo))

            if nchunk == 1:
                reduce(0)
            else:
                parallelize(reduce, nchunk)
        # J = A + A^T, K = A + A^T (tiled transpose)
        comptime TB = 32
        for s in range(nacc):
            var src = pacc.unsafe_offset(s * n2)
            var dst = vj if s < nj else vk.unsafe_offset((s - nj) * n2)
            var i0 = 0
            while i0 < nao:
                var i1 = min(i0 + TB, nao)
                var j0 = 0
                while j0 < nao:
                    var j1 = min(j0 + TB, nao)
                    for i in range(i0, i1):
                        for j in range(j0, j1):
                            dst[unsafe_offset=i * nao + j] = src[unsafe_offset=i * nao + j] + src[unsafe_offset=j * nao + i]
                    j0 = j1
                i0 = i1
        _ = acc^
        _ = cond^
        _ = counter^
