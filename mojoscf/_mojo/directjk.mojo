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
from _mojo.linalg import F64Ptr, list_ptr, vfill, vaxpy, vdot_serial
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


# --------------------------------------------------------------------------
# Gradient contractions with first-derivative ERIs
# --------------------------------------------------------------------------


def digest_ip1(
    blk: F64Ptr, na: Int, nb: Int, nc: Int, nd: Int, i0: Int, j0: Int, k0: Int, l0: Int, nao: Int,
    same_cd: Bool, nset: Int, dms: F64Ptr, acc: F64Ptr, with_j: Bool, with_k: Bool,
):
    """Fold the block (nabla_x i j|kl), layout [x][i][j][k][l], into J1/K1 accumulators.

    ``acc`` holds, per density s, J1 (3 nao^2) then K1 (3 nao^2):
    J1[x][i][j] += sum_kl (nabla_x i j|kl) D_lk and K1[x][i][l] += sum_jk (nabla_x i j|kl) D_jk
    over all k, l, i.e. also over the (lk) half of the ket pair when c != d.
    """
    var n2 = nao * nao
    var nab = na * nb
    var ncd = nc * nd
    var fac = 1.0 if same_cd else 2.0
    for s in range(nset):
        var dm = dms.unsafe_offset(s * n2)
        var vj = acc.unsafe_offset(s * 6 * n2)
        var vk = vj.unsafe_offset(3 * n2)
        for x in range(3):
            var vjx = vj.unsafe_offset(x * n2)
            var vkx = vk.unsafe_offset(x * n2)
            for i in range(na):
                var vki = vkx.unsafe_offset((i0 + i) * nao)
                for j in range(nb):
                    var row = blk.unsafe_offset((x * nab + i * nb + j) * ncd)
                    var djrow = dm.unsafe_offset((j0 + j) * nao)
                    if with_j:
                        var sj = 0.0
                        for k in range(nc):
                            var dkrow = dm.unsafe_offset((k0 + k) * nao + l0)
                            var g = row.unsafe_offset(k * nd)
                            for l in range(nd):
                                sj += g[unsafe_offset=l] * dkrow[unsafe_offset=l]
                        vjx[unsafe_offset=(i0 + i) * nao + j0 + j] += fac * sj
                    if with_k:
                        for k in range(nc):
                            var g = row.unsafe_offset(k * nd)
                            vaxpy(vki.unsafe_offset(l0), nd, djrow[unsafe_offset=k0 + k], g)
                            if not same_cd:
                                var sk = 0.0
                                for l in range(nd):
                                    sk += g[unsafe_offset=l] * djrow[unsafe_offset=l0 + l]
                                vki[unsafe_offset=k0 + k] += sk


def jk_ip1_core(
    var basis: Basis, var boys: BoysTable, nset: Int, dms: F64Ptr, vj: F64Ptr, vk: F64Ptr,
    with_j: Bool, with_k: Bool, tol: Float64,
):
    """J1 = sum_kl (nabla i j|kl) D_lk and K1 = sum_jk (nabla i j|kl) D_jk, each (nset, 3, nao, nao).

    These are pyscf's ``grad.rhf.get_jk`` matrices up to the overall sign
    (pyscf returns -J1, -K1).  Bra pairs are the ordered pairs (nabla a, b),
    ket pairs c >= d; a quartet is skipped when
    q'_ab q_cd max(2 D_cd, D_bc, D_bd) < tol with q' the Schwarz bound of the
    derivative pair and D the largest density element per shell block.
    """
    var nbas = basis.nbas
    var nao = basis.nao
    var n2 = nao * nao
    var ht = HermTable(2 * basis.lmax + 1)
    var sa = List[Int](capacity=nbas * (nbas + 1) // 2)
    var sb = List[Int](capacity=nbas * (nbas + 1) // 2)
    for a in range(nbas):
        for b in range(a + 1):
            sa.append(a)
            sb.append(b)
    var tab = PairTable(basis, basis, sa, sb, ht)
    var q = schwarz_bounds(boys, tab, ht)
    var da = List[Int](capacity=nbas * nbas)
    var db = List[Int](capacity=nbas * nbas)
    for a in range(nbas):
        for b in range(nbas):
            da.append(a)
            db.append(b)
    var dtab = PairTable(basis, basis, da, db, ht, 1)
    var qd = schwarz_bounds(boys, dtab, ht)
    var nket = nbas * (nbas + 1) // 2
    var nbra = nbas * nbas
    # largest density element per shell block, over all densities
    var cond = List[Float64](length=max(nbas * nbas, 1), fill=0.0)
    var pcond = list_ptr(cond)
    for a in range(nbas):
        for b in range(nbas):
            var m = 0.0
            for s in range(nset):
                var d = dms.unsafe_offset(s * n2)
                for i in range(basis.ao_loc[a], basis.ao_loc[a + 1]):
                    for j in range(basis.ao_loc[b], basis.ao_loc[b + 1]):
                        m = max(m, abs(d[unsafe_offset=i * nao + j]))
            pcond[unsafe_offset=a * nbas + b] = m
    var nworkers = min(parallelism_level(), nbra) if nbra >= 16 else 1
    var per = nset * 6 * n2
    var accl = List[Float64](length=nworkers * per, fill=0.0)
    var pacc = list_ptr(accl)
    var pq = list_ptr(q)
    var pqd = list_ptr(qd)
    var counter = Atomic[Int64](0)
    var pcount = Pointer(to=counter)

    def work(w: Int) {imm basis, imm boys, imm ht, imm tab, imm dtab, imm pq, imm pqd, imm pcond, imm pacc, imm pcount, imm nbra, imm nket, imm nbas, imm nao, imm per, imm nset, imm dms, imm with_j, imm with_k, imm tol}:
        var ws = EriWork(dtab.maxcomp, dtab.maxlab, tab.maxcomp, tab.maxlab)
        var acc = pacc.unsafe_offset(w * per)
        while True:
            var ib = Int(pcount[].fetch_add(1))
            if ib >= nbra:
                break
            var qab = pqd[unsafe_offset=ib]
            if qab == 0.0:
                continue
            var a = ib // nbas
            var b = ib % nbas
            var i0 = basis.ao_loc[a]
            var na = basis.ao_loc[a + 1] - i0
            var j0 = basis.ao_loc[b]
            var nb = basis.ao_loc[b + 1] - j0
            for ik in range(nket):
                var qq = qab * pq[unsafe_offset=ik]
                if qq < tol:
                    continue
                var c = tab.get(ik, I_A)
                var d = tab.get(ik, I_B)
                var dmax = 2.0 * pcond[unsafe_offset=c * nbas + d]
                dmax = max(dmax, max(pcond[unsafe_offset=b * nbas + c], pcond[unsafe_offset=b * nbas + d]))
                if qq * dmax < tol:
                    continue
                if not eri_quartet(dtab, ib, tab, ik, ht, boys, ws):
                    continue
                var k0 = basis.ao_loc[c]
                var nc = basis.ao_loc[c + 1] - k0
                var l0 = basis.ao_loc[d]
                var nd = basis.ao_loc[d + 1] - l0
                digest_ip1(list_ptr(ws.out), na, nb, nc, nd, i0, j0, k0, l0, nao, c == d, nset, dms, acc, with_j, with_k)
        _ = ws^

    if nworkers == 1:
        work(0)
    else:
        parallelize(work, nworkers)
    for w2 in range(1, nworkers):
        vaxpy(pacc, per, 1.0, pacc.unsafe_offset(w2 * per))
    for s in range(nset):
        var src = pacc.unsafe_offset(s * 6 * n2)
        for i in range(3 * n2):
            vj[unsafe_offset=s * 3 * n2 + i] = src[unsafe_offset=i]
            vk[unsafe_offset=s * 3 * n2 + i] = src[unsafe_offset=3 * n2 + i]
    _ = accl^
    _ = cond^
    _ = counter^
    _ = tab^
    _ = dtab^
    _ = q^
    _ = qd^
    _ = sa^
    _ = sb^
    _ = da^
    _ = db^
    _ = ht^
    _ = basis^
    _ = boys^


# --------------------------------------------------------------------------
# Two-electron part of the energy gradient
# --------------------------------------------------------------------------


def _block_max(d: F64Ptr, nao: Int, i0: Int, i1: Int, j0: Int, j1: Int) -> Float64:
    var m = 0.0
    for i in range(i0, i1):
        for j in range(j0, j1):
            m = max(m, abs(d[unsafe_offset=i * nao + j]))
    return m


def grad2e_core(
    var basis: Basis, var boys: BoysTable, dmj: F64Ptr, nk: Int, dmk: F64Ptr, jfac: Float64, kfac: Float64,
    tol: Float64, de: F64Ptr,
):
    """de[A][x] = d/dR_Ax of E2 = 1/2 sum_ijkl (ij|kl) G_ijkl at fixed densities, for every atom A.

    G_ijkl = jfac Dj_ij Dj_kl - kfac/2 sum_s (Dk_s,ik Dk_s,jl + Dk_s,il Dk_s,jk), with Dj and
    the ``nk`` matrices Dk_s symmetric (RHF: Dj = D, Dk = D, kfac = 1/2; UHF:
    Dj = Da + Db, Dk = (Da, Db), kfac = 1).  ``de`` (natm x 3) is overwritten.

    Each quartet a >= b, c >= d, ab >= cd is evaluated once with its eight
    permutations folded into a weight: one pass with the six-component pair
    table (nabla a, nabla b) of the bra, one with (nabla c) of the ket, and the
    derivative with respect to the centre of d from translational invariance
    (the four centre derivatives of an integral sum to zero).  The integrals
    are contracted with G as they are produced, so no derivative matrices are
    formed.  A quartet is skipped when
    max(q'_ab q_cd, q_ab q'_cd) max|G| < tol, q' the Schwarz bound of the
    six-component pair.
    """
    var nbas = basis.nbas
    var nao = basis.nao
    var natm = basis.natm
    var n2 = nao * nao
    var npairs = nbas * (nbas + 1) // 2
    var ht = HermTable(2 * basis.lmax + 1)
    var sa = List[Int](capacity=npairs)
    var sb = List[Int](capacity=npairs)
    for a in range(nbas):
        for b in range(a + 1):
            sa.append(a)
            sb.append(b)
    var tab = PairTable(basis, basis, sa, sb, ht)
    var q = schwarz_bounds(boys, tab, ht)
    var tab2 = PairTable(basis, basis, sa, sb, ht, 2)
    var q2 = schwarz_bounds(boys, tab2, ht)
    var tab1 = PairTable(basis, basis, sa, sb, ht, 1)
    # largest |Dj| and largest |Dk_s| per shell block
    var cj = List[Float64](length=max(nbas * nbas, 1), fill=0.0)
    var ck = List[Float64](length=max(nbas * nbas, 1), fill=0.0)
    for a in range(nbas):
        var i0 = basis.ao_loc[a]
        var i1 = basis.ao_loc[a + 1]
        for b in range(nbas):
            var j0 = basis.ao_loc[b]
            var j1 = basis.ao_loc[b + 1]
            cj[a * nbas + b] = _block_max(dmj, nao, i0, i1, j0, j1)
            var m = 0.0
            for s in range(nk):
                m = max(m, _block_max(dmk.unsafe_offset(s * n2), nao, i0, i1, j0, j1))
            ck[a * nbas + b] = m
    var nfmax = 1
    for a in range(nbas):
        nfmax = max(nfmax, shell_nfunc(basis, a))
    var nworkers = min(parallelism_level(), npairs) if npairs >= 16 else 1
    var per = 3 * natm
    var accl = List[Float64](length=max(nworkers * per, 1), fill=0.0)
    var pacc = list_ptr(accl)
    var pq = list_ptr(q)
    var pq2 = list_ptr(q2)
    var pcj = list_ptr(cj)
    var pck = list_ptr(ck)
    var counter = Atomic[Int64](0)
    var pcount = Pointer(to=counter)
    var ajf = abs(jfac)
    var akf = 0.5 * Float64(nk) * abs(kfac)

    def work(w: Int) {imm basis, imm boys, imm ht, imm tab, imm tab1, imm tab2, imm pq, imm pq2, imm pcj, imm pck, imm pacc, imm pcount, imm npairs, imm nbas, imm nao, imm n2, imm per, imm nk, imm dmj, imm dmk, imm jfac, imm kfac, imm ajf, imm akf, imm tol, imm nfmax}:
        var ws = EriWork(tab2.maxcomp, tab2.maxlab, tab.maxcomp, tab.maxlab)
        var gbuf = List[Float64](length=nfmax * nfmax * nfmax * nfmax + 8, fill=0.0)
        var g = list_ptr(gbuf)
        var gabuf = List[Float64](length=16, fill=0.0)
        var ga = list_ptr(gabuf)
        var acc = pacc.unsafe_offset(w * per)
        while True:
            var task = Int(pcount[].fetch_add(1))
            if task >= npairs:
                break
            var sp = npairs - 1 - task
            var qab = pq[unsafe_offset=sp]
            var q2ab = pq2[unsafe_offset=sp]
            if qab == 0.0:
                continue
            var a = tab.get(sp, I_A)
            var b = tab.get(sp, I_B)
            var i0 = basis.ao_loc[a]
            var na = basis.ao_loc[a + 1] - i0
            var j0 = basis.ao_loc[b]
            var nb = basis.ao_loc[b + 1] - j0
            var nab = na * nb
            var djab = ajf * pcj[unsafe_offset=a * nbas + b]
            for spk in range(sp + 1):
                var qcd = pq[unsafe_offset=spk]
                var qq = max(q2ab * qcd, qab * pq2[unsafe_offset=spk])
                if qq < tol:
                    continue
                var c = tab.get(spk, I_A)
                var d = tab.get(spk, I_B)
                var gmax = djab * pcj[unsafe_offset=c * nbas + d] + akf * (
                    pck[unsafe_offset=a * nbas + c] * pck[unsafe_offset=b * nbas + d]
                    + pck[unsafe_offset=a * nbas + d] * pck[unsafe_offset=b * nbas + c]
                )
                if qq * gmax < tol:
                    continue
                var k0 = basis.ao_loc[c]
                var nc = basis.ao_loc[c + 1] - k0
                var l0 = basis.ao_loc[d]
                var nd = basis.ao_loc[d + 1] - l0
                var ncd = nc * nd
                # G[ij][kl] of the block
                for i in range(na):
                    for j in range(nb):
                        var row = g.unsafe_offset((i * nb + j) * ncd)
                        var dij = jfac * dmj[unsafe_offset=(i0 + i) * nao + j0 + j]
                        for k in range(nc):
                            var dkrow = dmj.unsafe_offset((k0 + k) * nao + l0)
                            for l in range(nd):
                                row[unsafe_offset=k * nd + l] = dij * dkrow[unsafe_offset=l]
                        for s in range(nk):
                            var dm = dmk.unsafe_offset(s * n2)
                            var di = dm.unsafe_offset((i0 + i) * nao)
                            var dj = dm.unsafe_offset((j0 + j) * nao)
                            for k in range(nc):
                                var dik = 0.5 * kfac * di[unsafe_offset=k0 + k]
                                var djk = 0.5 * kfac * dj[unsafe_offset=k0 + k]
                                var r = row.unsafe_offset(k * nd)
                                for l in range(nd):
                                    r[unsafe_offset=l] -= dik * dj[unsafe_offset=l0 + l] + djk * di[unsafe_offset=l0 + l]
                var nabcd = nab * ncd
                var wgt = 0.5
                if a != b:
                    wgt *= 2.0
                if c != d:
                    wgt *= 2.0
                if sp != spk:
                    wgt *= 2.0
                vfill(ga, 9, 0.0)
                # (nabla a b|cd) and (a nabla b|cd): layout [6][ij][kl]
                if not eri_quartet(tab2, sp, tab, spk, ht, boys, ws):
                    continue
                var blk = list_ptr(ws.out)
                for x in range(6):
                    ga[unsafe_offset=x] = vdot_serial(blk.unsafe_offset(x * nabcd), g, nabcd)
                # (ab|nabla c d): layout [ij][3][kl]
                _ = eri_quartet(tab, sp, tab1, spk, ht, boys, ws)
                for ij in range(nab):
                    var grow = g.unsafe_offset(ij * ncd)
                    for x in range(3):
                        ga[unsafe_offset=6 + x] += vdot_serial(blk.unsafe_offset((ij * 3 + x) * ncd), grow, ncd)
                # d/dR = -nabla: atom(a) gets -ga[0:3], atom(b) -ga[3:6], atom(c) -ga[6:9], atom(d) the rest
                var pa = acc.unsafe_offset(3 * basis.atom[a])
                var pb = acc.unsafe_offset(3 * basis.atom[b])
                var pc = acc.unsafe_offset(3 * basis.atom[c])
                var pd = acc.unsafe_offset(3 * basis.atom[d])
                for x in range(3):
                    var gx = ga[unsafe_offset=x]
                    var gy = ga[unsafe_offset=3 + x]
                    var gz = ga[unsafe_offset=6 + x]
                    pa[unsafe_offset=x] -= wgt * gx
                    pb[unsafe_offset=x] -= wgt * gy
                    pc[unsafe_offset=x] -= wgt * gz
                    pd[unsafe_offset=x] += wgt * (gx + gy + gz)
        _ = ws^
        _ = gbuf^
        _ = gabuf^

    if nworkers == 1:
        work(0)
    else:
        parallelize(work, nworkers)
    for i in range(per):
        var v = 0.0
        for w2 in range(nworkers):
            v += pacc[unsafe_offset=w2 * per + i]
        de[unsafe_offset=i] = v
    _ = accl^
    _ = cj^
    _ = ck^
    _ = counter^
    _ = tab^
    _ = tab1^
    _ = tab2^
    _ = q^
    _ = q2^
    _ = sa^
    _ = sb^
    _ = ht^
    _ = basis^
    _ = boys^
