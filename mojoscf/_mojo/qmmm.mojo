"""QM/MM electrostatics in the AO basis: the potential of MM charges and its derivatives.

Every MM charge k (position C_k, weight w_k) is a unit charge distribution:
a point charge, or pyscf's Gaussian (zeta_k / pi)^{3/2} exp(-zeta_k |r - C_k|^2).
Both are the s-type ket of ``eri_kernel_lanes`` with one charge per SIMD
lane: a point charge is the exponent POINT_ZETA = 1e30, for which (ab|k)
equals <a| 1/|r - C_k| |b> to double precision (the Boys argument is
p q/(p + q) |P - C|^2 = p |P - C|^2 exactly in floating point).  The AO pair
is the bra; the lanes carry (q_k, w_k sqrt(q_k) / pi^{3/2}, C_k), which makes
the kernel's U[o][lane] equal to w_k (o|k).  Tasks are (shell pair, group of
charges) with per-thread accumulators, so a small QM region with many MM
charges still spreads over all threads.

    mm_potential_core  V_ij = sum_k w_k (ij|k)                         (nao x nao)
    mm_esp_core        phi_s,k = sum_ij D_s,ij (ij|k)                  (nset x nch: the potential
                       of the densities at the charges; the pair's Hermite matrices are
                       contracted with the densities first)
    mm_grad_core       with the six-component pair table (nabla a, nabla b):
                       M_x,ij = sum_k w_k (nabla_x i j|k)              (3 x nao x nao, all i, j)
                       F_k,x = sum_ij D_ij w_k (ij|nabla_x k)
                             = -sum_ij D_ij w_k [(nabla_x i j|k) + (i nabla_x j|k)]   (nch x 3)
                       G_A,x = 2 sum_{i on A, j} M_x,ij D_ij              (natm x 3)
                       (the last two from one pass, without forming M: the pair's
                       derivative Hermite matrices are contracted with D first, so
                       the kernel transforms six components instead of 6 n_a n_b)

(nabla acts on the electron coordinate, as libcint's ``ip`` integrals;
the last line is translational invariance.)
"""
from std.atomic import Atomic
from std.math import sqrt
from std.memory import Pointer
from std.runtime import parallelism_level
from max.algorithm import parallelize
from _mojo.linalg import F64Ptr, list_ptr, vfill
from _mojo.integrals import (
    Basis, BoysTable, HermTable, PairTable, lanes_dispatch, aligned_addr, nherm, padded, PFIELDS, W, PI, LMAX_POT,
    I_A, I_B, I_LAB, I_NCOMP, I_STRIDE, I_NP,
)

comptime CHUNK = 8 * W          # charges per kernel call (eight lane vectors)
comptime GROUP = 16             # chunks per task
comptime POINT_ZETA = 1.0e30    # exponent standing in for a point charge
comptime RBUF = (2 * 220 + 2 * LMAX_POT + 9) * W    # recursion scratch for degrees up to LMAX_POT (nherm(9) = 220)


struct ChargeLanes(Movable):
    """The charges in the lane layout of ``eri_kernel_lanes``, CHUNK per block.

    Block c holds five arrays of ipad = nvec(c) W values (exponent,
    w sqrt(q) / pi^{3/2}, x, y, z); unused lanes are zero, so they add nothing.
    """

    var buf: List[Float64]
    var base: Int
    var nch: Int
    var nchunk: Int

    def __init__(out self, nch: Int, coords: F64Ptr, weights: F64Ptr, zetas: F64Ptr, point: Bool):
        self.nch = nch
        self.nchunk = (nch + CHUNK - 1) // CHUNK
        self.buf = List[Float64](length=max(self.nchunk, 1) * PFIELDS * CHUNK + W, fill=0.0)
        self.base = aligned_addr(self.buf)
        var p = F64Ptr(unsafe_from_address=self.base)
        var inv_pi32 = 1.0 / (PI * sqrt(PI))
        for c in range(self.nchunk):
            var k0 = c * CHUNK
            var n = min(CHUNK, nch - k0)
            var ipad = ((n + W - 1) // W) * W
            var blk = p.unsafe_offset(c * PFIELDS * CHUNK)
            for x in range(n):
                var k = k0 + x
                var q = POINT_ZETA if point else zetas[unsafe_offset=k]
                blk[unsafe_offset=x] = q
                blk[unsafe_offset=ipad + x] = weights[unsafe_offset=k] * sqrt(q) * inv_pi32
                blk[unsafe_offset=2 * ipad + x] = coords[unsafe_offset=3 * k]
                blk[unsafe_offset=3 * ipad + x] = coords[unsafe_offset=3 * k + 1]
                blk[unsafe_offset=4 * ipad + x] = coords[unsafe_offset=3 * k + 2]

    def chunk(self, c: Int) -> F64Ptr:
        return F64Ptr(unsafe_from_address=self.base).unsafe_offset(c * PFIELDS * CHUNK)

    def nvec(self, c: Int) -> Int:
        return (min(CHUNK, self.nch - c * CHUNK) + W - 1) // W


def _all_pairs(nbas: Int, mut sa: List[Int], mut sb: List[Int]):
    for a in range(nbas):
        for b in range(a + 1):
            sa.append(a)
            sb.append(b)


def _reduce_threads(acc: F64Ptr, nworkers: Int, n: Int, dst: F64Ptr):
    for i in range(n):
        var v = 0.0
        for w in range(nworkers):
            v += acc[unsafe_offset=w * n + i]
        dst[unsafe_offset=i] = v


def mm_potential_core(
    basis: Basis, boys: BoysTable, nch: Int, coords: F64Ptr, weights: F64Ptr, zetas: F64Ptr, point: Bool,
    v_out: F64Ptr,
):
    """V_ij = sum_k w_k (ij|k) into ``v_out`` (nao x nao, overwritten); see the module docstring."""
    var nbas = basis.nbas
    var nao = basis.nao
    var n2 = nao * nao
    var npairs = nbas * (nbas + 1) // 2
    var ht = HermTable(2 * basis.lmax)
    var sa = List[Int](capacity=npairs)
    var sb = List[Int](capacity=npairs)
    _all_pairs(nbas, sa, sb)
    var tab = PairTable(basis, basis, sa, sb, ht)
    var lanes = ChargeLanes(nch, coords, weights, zetas, point)
    var ngroup = max(1, (lanes.nchunk + GROUP - 1) // GROUP)
    var ntask = npairs * ngroup
    var nworkers = max(1, min(parallelism_level(), ntask))
    var accl = List[Float64](length=nworkers * n2 + 1, fill=0.0)
    var pacc = list_ptr(accl)
    var btab = list_ptr(boys.table)
    var counter = Atomic[Int64](0)
    var pcount = Pointer(to=counter)

    def work(w: Int) {imm basis, imm tab, imm lanes, imm btab, imm pacc, imm pcount, imm ntask, imm ngroup, imm npairs, imm nao, imm n2}:
        var maxc = tab.maxcomp
        var ub = List[Float64](length=maxc * CHUNK + 2 * W, fill=0.0)
        var rbl = List[Float64](length=RBUF + 2 * W, fill=0.0)
        var vacc = List[Float64](length=maxc * W + 2 * W, fill=0.0)
        var pu = F64Ptr(unsafe_from_address=aligned_addr(ub))
        var prb = F64Ptr(unsafe_from_address=aligned_addr(rbl))
        var pv = F64Ptr(unsafe_from_address=aligned_addr(vacc))
        var acc = pacc.unsafe_offset(w * n2)
        while True:
            var task = Int(pcount[].fetch_add(1))
            if task >= ntask:
                break
            var sp = npairs - 1 - task // ngroup
            var g = task % ngroup
            var np = tab.get(sp, I_NP)
            if np == 0:
                continue
            var lo = tab.get(sp, I_LAB)
            var no = tab.get(sp, I_NCOMP)
            var so = tab.get(sp, I_STRIDE)
            vfill(pv, no * W, 0.0)
            for c in range(g * GROUP, min((g + 1) * GROUP, lanes.nchunk)):
                var nv = lanes.nvec(c)
                lanes_dispatch(lo, 0, np, tab.prim_ptr(sp), tab.e_ptr(sp), so, no, lanes.chunk(c), nv, btab, pu, prb)
                var ipad = nv * W
                for o in range(no):
                    var s = pv.unsafe_load[width=W](o * W)
                    for v in range(nv):
                        s += pu.unsafe_load[width=W](o * ipad + v * W)
                    pv.unsafe_store(o * W, s)
            var a = tab.get(sp, I_A)
            var b = tab.get(sp, I_B)
            var i0 = basis.ao_loc[a]
            var na = basis.ao_loc[a + 1] - i0
            var j0 = basis.ao_loc[b]
            var nb = basis.ao_loc[b + 1] - j0
            for i in range(na):
                for j in range(nb):
                    var val = pv.unsafe_load[width=W]((i * nb + j) * W).reduce_add()
                    acc[unsafe_offset=(i0 + i) * nao + j0 + j] += val
                    if a != b:
                        acc[unsafe_offset=(j0 + j) * nao + i0 + i] += val
        _ = ub^
        _ = rbl^
        _ = vacc^

    if nworkers == 1:
        work(0)
    else:
        parallelize(work, nworkers)
    _reduce_threads(pacc, nworkers, n2, v_out)
    _ = accl^
    _ = counter^
    _ = lanes^
    _ = tab^
    _ = ht^
    _ = sa^
    _ = sb^


def mm_esp_core(
    basis: Basis, boys: BoysTable, nch: Int, coords: F64Ptr, zetas: F64Ptr, point: Bool, nset: Int, dms: F64Ptr,
    esp_out: F64Ptr,
):
    """phi_s,k = sum_ij D_s,ij (ij|k) into ``esp_out`` (nset x nch, overwritten) for symmetric ``dms``.

    Per shell pair the Hermite matrices are contracted with the densities,
    E~[ko][h][s] = w_ab sum_ij D_s,ij E[ko][h][ij] (w_ab = 2 for a != b), so the
    kernel transforms ``nset`` components and its U[s][lane] is the pair's
    share of phi_s at the lane's charge.
    """
    var nbas = basis.nbas
    var nao = basis.nao
    var npairs = nbas * (nbas + 1) // 2
    var ht = HermTable(2 * basis.lmax)
    var sa = List[Int](capacity=npairs)
    var sb = List[Int](capacity=npairs)
    _all_pairs(nbas, sa, sb)
    var tab = PairTable(basis, basis, sa, sb, ht)
    var ones = List[Float64](length=max(nch, 1), fill=1.0)
    var lanes = ChargeLanes(nch, coords, list_ptr(ones), zetas, point)
    var ngroup = max(1, (lanes.nchunk + GROUP - 1) // GROUP)
    var ntask = npairs * ngroup
    var nworkers = max(1, min(parallelism_level(), ntask))
    var per = nset * nch
    var maxnp = 1
    for sp in range(npairs):
        maxnp = max(maxnp, tab.get(sp, I_NP))
    var st = padded(nset)
    var esize = maxnp * nherm(tab.maxlab) * st
    var accl = List[Float64](length=nworkers * per + 1, fill=0.0)
    var pacc = list_ptr(accl)
    var btab = list_ptr(boys.table)
    var counter = Atomic[Int64](0)
    var pcount = Pointer(to=counter)

    def work(w: Int) {imm basis, imm tab, imm lanes, imm btab, imm pacc, imm pcount, imm ntask, imm ngroup, imm npairs, imm nao, imm per, imm nset, imm nch, imm dms, imm st, imm esize}:
        var ub = List[Float64](length=max(nset, 1) * CHUNK + 2 * W, fill=0.0)
        var rbl = List[Float64](length=RBUF + 2 * W, fill=0.0)
        var econ = List[Float64](length=esize + 2 * W, fill=0.0)
        var pe = F64Ptr(unsafe_from_address=aligned_addr(econ))
        var pu = F64Ptr(unsafe_from_address=aligned_addr(ub))
        var prb = F64Ptr(unsafe_from_address=aligned_addr(rbl))
        var acc = pacc.unsafe_offset(w * per)
        var n2 = nao * nao
        while True:
            var task = Int(pcount[].fetch_add(1))
            if task >= ntask:
                break
            var sp = npairs - 1 - task // ngroup
            var g = task % ngroup
            var np = tab.get(sp, I_NP)
            if np == 0:
                continue
            var lo = tab.get(sp, I_LAB)
            var so = tab.get(sp, I_STRIDE)
            var a = tab.get(sp, I_A)
            var b = tab.get(sp, I_B)
            var i0 = basis.ao_loc[a]
            var na = basis.ao_loc[a + 1] - i0
            var j0 = basis.ao_loc[b]
            var nb = basis.ao_loc[b + 1] - j0
            var wpair = 2.0 if a != b else 1.0
            # E~[ko][h][s] = w_ab sum_ij D_s,ij E[ko][h][ij]
            var nh = nherm(lo)
            var e = tab.e_ptr(sp)
            for r in range(np * nh):
                var row = e.unsafe_offset(r * so)
                var dst = pe.unsafe_offset(r * st)
                for s in range(nset):
                    var dmat = dms.unsafe_offset(s * n2)
                    var acc_s = 0.0
                    for i in range(na):
                        var drow = dmat.unsafe_offset((i0 + i) * nao + j0)
                        var erow = row.unsafe_offset(i * nb)
                        for j in range(nb):
                            acc_s += drow[unsafe_offset=j] * erow[unsafe_offset=j]
                    dst[unsafe_offset=s] = acc_s * wpair
            for c in range(g * GROUP, min((g + 1) * GROUP, lanes.nchunk)):
                var nv = lanes.nvec(c)
                var ipad = nv * W
                lanes_dispatch(lo, 0, np, tab.prim_ptr(sp), pe, st, nset, lanes.chunk(c), nv, btab, pu, prb)
                var k0 = c * CHUNK
                var nk = min(CHUNK, nch - k0)
                for s in range(nset):
                    var dst = acc.unsafe_offset(s * nch + k0)
                    var src = pu.unsafe_offset(s * ipad)
                    for k in range(nk):
                        dst[unsafe_offset=k] += src[unsafe_offset=k]
        _ = ub^
        _ = rbl^
        _ = econ^

    if nworkers == 1:
        work(0)
    else:
        parallelize(work, nworkers)
    _reduce_threads(pacc, nworkers, per, esp_out)
    _ = accl^
    _ = counter^
    _ = lanes^
    _ = ones^
    _ = tab^
    _ = ht^
    _ = sa^
    _ = sb^


def mm_grad_core(
    basis: Basis, boys: BoysTable, nch: Int, coords: F64Ptr, weights: F64Ptr, zetas: F64Ptr, point: Bool,
    dm: F64Ptr, want_mat: Bool, mat_out: F64Ptr, want_force: Bool, force_out: F64Ptr,
    want_atoms: Bool, atom_out: F64Ptr,
):
    """M_x,ij into ``mat_out`` (3 x nao x nao), F_k,x into ``force_out`` (nch x 3) and/or G_A,x into ``atom_out``.

    Each requested output is overwritten.  ``dm`` (nao x nao, symmetric) is
    read for the forces and the atom gradient.  See the module docstring
    for the definitions.
    """
    var nbas = basis.nbas
    var nao = basis.nao
    var n2 = nao * nao
    var npairs = nbas * (nbas + 1) // 2
    var ht = HermTable(2 * basis.lmax + 1)
    var sa = List[Int](capacity=npairs)
    var sb = List[Int](capacity=npairs)
    _all_pairs(nbas, sa, sb)
    var tab2 = PairTable(basis, basis, sa, sb, ht, 2)
    var lanes = ChargeLanes(nch, coords, weights, zetas, point)
    var ngroup = max(1, (lanes.nchunk + GROUP - 1) // GROUP)
    var ntask = npairs * ngroup
    var nworkers = max(1, min(parallelism_level(), ntask))
    var natm = basis.natm
    var nmat = 3 * n2 if want_mat else 0
    var nfor = 3 * nch if want_force else 0
    var per = nmat + nfor + (3 * natm if want_atoms else 0)
    # D-contracted Hermite matrices (six components, padded) when M is not wanted
    var maxnp = 1
    for sp in range(npairs):
        maxnp = max(maxnp, tab2.get(sp, I_NP))
    var st = padded(6)
    var esize = maxnp * nherm(tab2.maxlab) * st
    var accl = List[Float64](length=nworkers * per + 1, fill=0.0)
    var pacc = list_ptr(accl)
    var btab = list_ptr(boys.table)
    var counter = Atomic[Int64](0)
    var pcount = Pointer(to=counter)

    def work(w: Int) {imm basis, imm tab2, imm lanes, imm btab, imm pacc, imm pcount, imm ntask, imm ngroup, imm npairs, imm nao, imm n2, imm per, imm nmat, imm nfor, imm dm, imm want_mat, imm want_force, imm want_atoms, imm st, imm esize}:
        var maxc = tab2.maxcomp
        var ub = List[Float64](length=maxc * CHUNK + 2 * W, fill=0.0)
        var rbl = List[Float64](length=RBUF + 2 * W, fill=0.0)
        var vacc = List[Float64](length=maxc * W + 2 * W, fill=0.0)
        var dloc = List[Float64](length=maxc + 1, fill=0.0)
        var econ = List[Float64](length=(esize if not want_mat else 1) + 2 * W, fill=0.0)
        var pe = F64Ptr(unsafe_from_address=aligned_addr(econ))
        var pu = F64Ptr(unsafe_from_address=aligned_addr(ub))
        var prb = F64Ptr(unsafe_from_address=aligned_addr(rbl))
        var pv = F64Ptr(unsafe_from_address=aligned_addr(vacc))
        var pd = list_ptr(dloc)
        var macc = pacc.unsafe_offset(w * per)
        var facc = macc.unsafe_offset(nmat)
        var aacc = facc.unsafe_offset(nfor)
        var contract = want_force or want_atoms
        while True:
            var task = Int(pcount[].fetch_add(1))
            if task >= ntask:
                break
            var sp = npairs - 1 - task // ngroup
            var g = task % ngroup
            var np = tab2.get(sp, I_NP)
            if np == 0:
                continue
            var lo = tab2.get(sp, I_LAB)
            var no = tab2.get(sp, I_NCOMP)
            var so = tab2.get(sp, I_STRIDE)
            var a = tab2.get(sp, I_A)
            var b = tab2.get(sp, I_B)
            var i0 = basis.ao_loc[a]
            var na = basis.ao_loc[a + 1] - i0
            var j0 = basis.ao_loc[b]
            var nb = basis.ao_loc[b + 1] - j0
            var nab = na * nb
            var wpair = 2.0 if a != b else 1.0
            if contract:
                for i in range(na):
                    for j in range(nb):
                        pd[unsafe_offset=i * nb + j] = dm[unsafe_offset=(i0 + i) * nao + j0 + j]
            # sum over the charges of the D-contracted (nabla a) and (nabla b) components, per x
            var ga0 = SIMD[DType.float64, W](0.0)
            var ga1 = SIMD[DType.float64, W](0.0)
            var ga2 = SIMD[DType.float64, W](0.0)
            var gb0 = SIMD[DType.float64, W](0.0)
            var gb1 = SIMD[DType.float64, W](0.0)
            var gb2 = SIMD[DType.float64, W](0.0)
            if want_mat:
                vfill(pv, no * W, 0.0)
            else:
                # E~[ko][h][c] = sum_ij D_ij E[ko][h][c nab + ij], c = (nabla a)_x, (nabla b)_x
                var nh = nherm(lo)
                var e2 = tab2.e_ptr(sp)
                for r in range(np * nh):
                    var row = e2.unsafe_offset(r * so)
                    var dst = pe.unsafe_offset(r * st)
                    for cc in range(6):
                        var acc = 0.0
                        var src = row.unsafe_offset(cc * nab)
                        for ij in range(nab):
                            acc += pd[unsafe_offset=ij] * src[unsafe_offset=ij]
                        dst[unsafe_offset=cc] = acc
            for c in range(g * GROUP, min((g + 1) * GROUP, lanes.nchunk)):
                var nv = lanes.nvec(c)
                var ipad = nv * W
                if not want_mat:
                    lanes_dispatch(lo, 0, np, tab2.prim_ptr(sp), pe, st, 6, lanes.chunk(c), nv, btab, pu, prb)
                    var k0 = c * CHUNK
                    var nk = min(CHUNK, lanes.nch - k0)
                    for v in range(nv):
                        var a0 = pu.unsafe_load[width=W](v * W)
                        var a1 = pu.unsafe_load[width=W](ipad + v * W)
                        var a2 = pu.unsafe_load[width=W](2 * ipad + v * W)
                        var b0 = pu.unsafe_load[width=W](3 * ipad + v * W)
                        var b1 = pu.unsafe_load[width=W](4 * ipad + v * W)
                        var b2 = pu.unsafe_load[width=W](5 * ipad + v * W)
                        ga0 += a0
                        ga1 += a1
                        ga2 += a2
                        gb0 += b0
                        gb1 += b1
                        gb2 += b2
                        if want_force:
                            var f0 = (a0 + b0) * wpair
                            var f1 = (a1 + b1) * wpair
                            var f2 = (a2 + b2) * wpair
                            for lane in range(min(W, nk - v * W)):
                                var fk = facc.unsafe_offset(3 * (k0 + v * W + lane))
                                fk[unsafe_offset=0] -= f0[lane]
                                fk[unsafe_offset=1] -= f1[lane]
                                fk[unsafe_offset=2] -= f2[lane]
                    continue
                lanes_dispatch(lo, 0, np, tab2.prim_ptr(sp), tab2.e_ptr(sp), so, no, lanes.chunk(c), nv, btab, pu, prb)
                if want_mat:
                    for o in range(no):
                        var s = pv.unsafe_load[width=W](o * W)
                        for v in range(nv):
                            s += pu.unsafe_load[width=W](o * ipad + v * W)
                        pv.unsafe_store(o * W, s)
                if contract:
                    var k0 = c * CHUNK
                    var nk = min(CHUNK, lanes.nch - k0)
                    for v in range(nv):
                        for x in range(3):
                            var s0 = SIMD[DType.float64, W](0.0)
                            var s1 = SIMD[DType.float64, W](0.0)
                            var ua = pu.unsafe_offset(x * nab * ipad + v * W)
                            var ubb = pu.unsafe_offset((x + 3) * nab * ipad + v * W)
                            for ij in range(nab):
                                var d = SIMD[DType.float64, W](pd[unsafe_offset=ij])
                                s0 += ua.unsafe_load[width=W](ij * ipad) * d
                                s1 += ubb.unsafe_load[width=W](ij * ipad) * d
                            if want_force:
                                var s = (s0 + s1) * wpair
                                for lane in range(min(W, nk - v * W)):
                                    facc[unsafe_offset=3 * (k0 + v * W + lane) + x] -= s[lane]
                            if x == 0:
                                ga0 += s0
                                gb0 += s1
                            elif x == 1:
                                ga1 += s0
                                gb1 += s1
                            else:
                                ga2 += s0
                                gb2 += s1
            if want_atoms:
                # G_A = 2 sum_{i on A} M_ij D_ij: (nabla a) rows for atom(a), (nabla b) rows (a != b) for atom(b)
                var pa = aacc.unsafe_offset(3 * basis.atom[a])
                pa[unsafe_offset=0] += 2.0 * ga0.reduce_add()
                pa[unsafe_offset=1] += 2.0 * ga1.reduce_add()
                pa[unsafe_offset=2] += 2.0 * ga2.reduce_add()
                if a != b:
                    var pb = aacc.unsafe_offset(3 * basis.atom[b])
                    pb[unsafe_offset=0] += 2.0 * gb0.reduce_add()
                    pb[unsafe_offset=1] += 2.0 * gb1.reduce_add()
                    pb[unsafe_offset=2] += 2.0 * gb2.reduce_add()
            if want_mat:
                for x in range(3):
                    var mx = macc.unsafe_offset(x * n2)
                    for i in range(na):
                        for j in range(nb):
                            mx[unsafe_offset=(i0 + i) * nao + j0 + j] += pv.unsafe_load[width=W](
                                (x * nab + i * nb + j) * W
                            ).reduce_add()
                    if a != b:
                        for i in range(na):
                            for j in range(nb):
                                mx[unsafe_offset=(j0 + j) * nao + i0 + i] += pv.unsafe_load[width=W](
                                    ((x + 3) * nab + i * nb + j) * W
                                ).reduce_add()
        _ = ub^
        _ = rbl^
        _ = vacc^
        _ = dloc^
        _ = econ^

    if nworkers == 1:
        work(0)
    else:
        parallelize(work, nworkers)
    if want_mat:
        for i in range(3 * n2):
            var v = 0.0
            for w2 in range(nworkers):
                v += pacc[unsafe_offset=w2 * per + i]
            mat_out[unsafe_offset=i] = v
    if want_force:
        for i in range(3 * nch):
            var v = 0.0
            for w2 in range(nworkers):
                v += pacc[unsafe_offset=w2 * per + nmat + i]
            force_out[unsafe_offset=i] = v
    if want_atoms:
        for i in range(3 * natm):
            var v = 0.0
            for w2 in range(nworkers):
                v += pacc[unsafe_offset=w2 * per + nmat + nfor + i]
            atom_out[unsafe_offset=i] = v
    _ = accl^
    _ = counter^
    _ = lanes^
    _ = tab2^
    _ = ht^
    _ = sa^
    _ = sb^
