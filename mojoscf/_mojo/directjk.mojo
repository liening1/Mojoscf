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

Each unique quartet is evaluated by the task of its pair with more
primitive pairs (the bra).  The bra's kets are queued by class (Hermite
degree and component count) and evaluated together by ``eri_batch``: their
primitive pairs, up to BATCH_LANES of them, are the SIMD lanes of one
kernel call, so even single-primitive quartets of segmented basis sets run
vectorised.  Kets of one primitive pair and two shells of degree lc + ld
<= LMAX1 (most kets of segmented basis sets) are queued by (lc, ld) instead
and stay one per lane after the kernel: their integrals and all six J/K
contractions are vector operations over the lanes (``lanes_digest1``).
Kets beyond the lane kernels' degrees go through ``eri_quartet`` one at a
time.  From JK_LANES_MIN densities on the densities are the SIMD lanes
instead (``DirectJK._jk_lanes``): each integral updates eight of them with
vector FMAs, and every batched ket goes through ``eri_batch``.

With ``omega`` > 0 the integrals are those of the long-range operator
erf(omega r12) / r12: the bra side of every quartet comes from a second,
attenuated pair table (``attenuate``); the Schwarz bounds of the full
operator remain valid upper bounds.
"""
from std.atomic import Atomic
from std.math import sqrt
from std.memory import Pointer, stack_allocation
from std.python import PythonObject
from std.runtime import parallelism_level
from max.algorithm import parallelize
from _mojo.linalg import F64Ptr, list_ptr, vcopy, vfill, vaxpy
from _mojo.integrals import (
    Basis, BoysTable, HermTable, PairTable, EriWork, EriBatch, I64Ptr, IntPtr, eri_quartet, eri_batch, eri_batch_u,
    schwarz_bounds, W, aligned_addr,
    shell_nfunc, batch_supported, batch_classes, lanes_preferred, int_ptr, KetQueue, attenuate, I_A, I_B, I_LAB,
    I_NCOMP, I_NP, I_STRIDE, nherm, bras_by_nprim, LAB_BATCH,
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


def digest_nd[DO_J: Bool, DO_K: Bool, ND: Int](
    blk: F64Ptr, na: Int, nb: Int, nc: Int, i0: Int, j0: Int, k0: Int, l0: Int, nao: Int, scale: Float64,
    dj: F64Ptr, aj: F64Ptr, dk: F64Ptr, ak: F64Ptr, loc: F64Ptr,
):
    """``digest`` for a last shell of ND functions: its density rows and K rows stay in registers."""
    var jab = loc
    var jcd = jab.unsafe_offset(na * nb)
    var kac = jcd.unsafe_offset(nc * ND)
    var kbc = kac.unsafe_offset(na * nc)
    var kad = kbc.unsafe_offset(nb * nc)
    var kbd = kad.unsafe_offset(na * ND)
    vfill(loc, na * nb + nc * ND + na * nc + nb * nc + na * ND + nb * ND, 0.0)
    for i in range(na):
        var dad = SIMD[DType.float64, 8](0.0)
        var kadv = SIMD[DType.float64, 8](0.0)
        comptime if DO_K:
            comptime for l in range(ND):
                dad[l] = dk[unsafe_offset=(i0 + i) * nao + l0 + l]
        for j in range(nb):
            var dij = 0.0
            comptime if DO_J:
                dij = dj[unsafe_offset=(i0 + i) * nao + j0 + j]
            var dbd = SIMD[DType.float64, 8](0.0)
            var kbdv = SIMD[DType.float64, 8](0.0)
            comptime if DO_K:
                comptime for l in range(ND):
                    dbd[l] = dk[unsafe_offset=(j0 + j) * nao + l0 + l]
            var sj = 0.0
            for k in range(nc):
                var g = blk.unsafe_offset(((i * nb + j) * nc + k) * ND)
                var dcd = dj.unsafe_offset((k0 + k) * nao + l0)
                var jrow = jcd.unsafe_offset(k * ND)
                var dbc = 0.0
                var dac = 0.0
                comptime if DO_K:
                    dbc = dk[unsafe_offset=(j0 + j) * nao + k0 + k]
                    dac = dk[unsafe_offset=(i0 + i) * nao + k0 + k]
                var sac = 0.0
                var sbc = 0.0
                comptime for l in range(ND):
                    var v = g[unsafe_offset=l]
                    comptime if DO_J:
                        sj += v * dcd[unsafe_offset=l]
                        jrow[unsafe_offset=l] += v * dij
                    comptime if DO_K:
                        sac += v * dbd[l]
                        sbc += v * dad[l]
                        kadv[l] += v * dbc
                        kbdv[l] += v * dac
                comptime if DO_K:
                    kac[unsafe_offset=i * nc + k] += sac
                    kbc[unsafe_offset=j * nc + k] += sbc
            comptime if DO_J:
                jab[unsafe_offset=i * nb + j] += sj
            comptime if DO_K:
                comptime for l in range(ND):
                    kbd[unsafe_offset=j * ND + l] += kbdv[l]
        comptime if DO_K:
            comptime for l in range(ND):
                kad[unsafe_offset=i * ND + l] = kadv[l]
    comptime if DO_J:
        var s2 = 2.0 * scale
        for i in range(na):
            vaxpy(aj.unsafe_offset((i0 + i) * nao + j0), nb, s2, jab.unsafe_offset(i * nb))
        for k in range(nc):
            vaxpy(aj.unsafe_offset((k0 + k) * nao + l0), ND, s2, jcd.unsafe_offset(k * ND))
    comptime if DO_K:
        for i in range(na):
            vaxpy(ak.unsafe_offset((i0 + i) * nao + k0), nc, scale, kac.unsafe_offset(i * nc))
            vaxpy(ak.unsafe_offset((i0 + i) * nao + l0), ND, scale, kad.unsafe_offset(i * ND))
        for j in range(nb):
            vaxpy(ak.unsafe_offset((j0 + j) * nao + k0), nc, scale, kbc.unsafe_offset(j * nc))
            vaxpy(ak.unsafe_offset((j0 + j) * nao + l0), ND, scale, kbd.unsafe_offset(j * ND))


def digest_cd[DO_J: Bool, DO_K: Bool, NC: Int, ND: Int](
    blk: F64Ptr, na: Int, nb: Int, i0: Int, j0: Int, k0: Int, l0: Int, nao: Int, scale: Float64,
    dj: F64Ptr, aj: F64Ptr, dk: F64Ptr, ak: F64Ptr,
):
    """``digest`` for a ket of NC x ND functions, unrolled over the ket.

    Sums over the ket stay in registers and every sum is added to the
    matrices once it is complete, so there is no scratch to clear and the only
    loops (over the bra's functions) have the same length for all kets of a
    bra.  The density elements of the ket block are read once per quartet.
    """
    comptime NCD = NC * ND
    var s2 = 2.0 * scale
    var dcd = stack_allocation[NCD, Float64]()    # D_cd (J)
    var jcd = stack_allocation[NCD, Float64]()    # 2 s sum_ij (ij|cd) D_ij (J)
    var dad = stack_allocation[ND, Float64]()     # D_il, D_ik of the current bra row i (K)
    var dac = stack_allocation[NC, Float64]()
    var dbd = stack_allocation[ND, Float64]()     # D_jl, D_jk of the current j (K)
    var dbc = stack_allocation[NC, Float64]()
    var kad = stack_allocation[ND, Float64]()     # K_il over j, k; K_ik over j, l (row i)
    var kac = stack_allocation[NC, Float64]()
    var kbd = stack_allocation[ND, Float64]()     # K_jl over k (one (i, j))
    comptime if DO_J:
        comptime for k in range(NC):
            comptime for l in range(ND):
                dcd[unsafe_offset=k * ND + l] = dj[unsafe_offset=(k0 + k) * nao + l0 + l]
                jcd[unsafe_offset=k * ND + l] = 0.0
    for i in range(na):
        var ri = (i0 + i) * nao
        comptime if DO_K:
            comptime for l in range(ND):
                dad[unsafe_offset=l] = dk[unsafe_offset=ri + l0 + l]
                kad[unsafe_offset=l] = 0.0
            comptime for k in range(NC):
                dac[unsafe_offset=k] = dk[unsafe_offset=ri + k0 + k]
                kac[unsafe_offset=k] = 0.0
        for j in range(nb):
            var rj = (j0 + j) * nao
            var g = blk.unsafe_offset((i * nb + j) * NCD)
            var dij = 0.0
            comptime if DO_J:
                dij = s2 * dj[unsafe_offset=ri + j0 + j]
            comptime if DO_K:
                comptime for l in range(ND):
                    dbd[unsafe_offset=l] = dk[unsafe_offset=rj + l0 + l]
                    kbd[unsafe_offset=l] = 0.0
                comptime for k in range(NC):
                    dbc[unsafe_offset=k] = dk[unsafe_offset=rj + k0 + k]
            var sj = 0.0
            comptime for k in range(NC):
                var sac = 0.0
                var sbc = 0.0
                comptime for l in range(ND):
                    var v = g[unsafe_offset=k * ND + l]
                    comptime if DO_J:
                        sj += v * dcd[unsafe_offset=k * ND + l]
                        jcd[unsafe_offset=k * ND + l] += v * dij
                    comptime if DO_K:
                        sac += v * dbd[unsafe_offset=l]
                        sbc += v * dad[unsafe_offset=l]
                        kad[unsafe_offset=l] += v * dbc[unsafe_offset=k]
                        kbd[unsafe_offset=l] += v * dac[unsafe_offset=k]
                comptime if DO_K:
                    kac[unsafe_offset=k] += sac
                    ak[unsafe_offset=rj + k0 + k] += scale * sbc
            comptime if DO_J:
                aj[unsafe_offset=ri + j0 + j] += s2 * sj
            comptime if DO_K:
                comptime for l in range(ND):
                    ak[unsafe_offset=rj + l0 + l] += scale * kbd[unsafe_offset=l]
        comptime if DO_K:
            comptime for k in range(NC):
                ak[unsafe_offset=ri + k0 + k] += scale * kac[unsafe_offset=k]
            comptime for l in range(ND):
                ak[unsafe_offset=ri + l0 + l] += scale * kad[unsafe_offset=l]
    comptime if DO_J:
        comptime for k in range(NC):
            comptime for l in range(ND):
                aj[unsafe_offset=(k0 + k) * nao + l0 + l] += jcd[unsafe_offset=k * ND + l]


def digest_any[DO_J: Bool, DO_K: Bool](
    blk: F64Ptr, na: Int, nb: Int, nc: Int, nd: Int, i0: Int, j0: Int, k0: Int, l0: Int, nao: Int, scale: Float64,
    dj: F64Ptr, aj: F64Ptr, dk: F64Ptr, ak: F64Ptr, loc: F64Ptr,
):
    """``digest``, through ``digest_cd`` for kets of s to f shells (spherical, one contraction) and
    ``digest_nd`` for the common function counts of the last shell."""
    comptime for KC in range(4):
        comptime for KD in range(4):
            if nc == 2 * KC + 1 and nd == 2 * KD + 1:
                digest_cd[DO_J, DO_K, 2 * KC + 1, 2 * KD + 1](blk, na, nb, i0, j0, k0, l0, nao, scale, dj, aj, dk, ak)
                return
    if nd == 1:
        digest_nd[DO_J, DO_K, 1](blk, na, nb, nc, i0, j0, k0, l0, nao, scale, dj, aj, dk, ak, loc)
    elif nd == 3:
        digest_nd[DO_J, DO_K, 3](blk, na, nb, nc, i0, j0, k0, l0, nao, scale, dj, aj, dk, ak, loc)
    elif nd == 5:
        digest_nd[DO_J, DO_K, 5](blk, na, nb, nc, i0, j0, k0, l0, nao, scale, dj, aj, dk, ak, loc)
    else:
        digest[DO_J, DO_K](blk, na, nb, nc, nd, i0, j0, k0, l0, nao, scale, dj, aj, dk, ak, loc)


def digest_quartet(
    basis: Basis, a: Int, b: Int, c: Int, d: Int, same_pair: Bool, blk: F64Ptr, nao: Int, n2: Int,
    nj: Int, nk: Int, dmj: F64Ptr, dmk: F64Ptr, aj: F64Ptr, ak: F64Ptr, ploc: F64Ptr,
):
    """Fold the block (ab|cd) of a quartet a >= b, c >= d, ab >= cd into the J/K accumulators."""
    var i0 = basis.ao_loc[a]
    var na = basis.ao_loc[a + 1] - i0
    var j0 = basis.ao_loc[b]
    var nb = basis.ao_loc[b + 1] - j0
    var k0 = basis.ao_loc[c]
    var nc = basis.ao_loc[c + 1] - k0
    var l0 = basis.ao_loc[d]
    var nd = basis.ao_loc[d + 1] - l0
    var scale = 1.0
    if a == b:
        scale *= 0.5
    if c == d:
        scale *= 0.5
    if same_pair:
        scale *= 0.5
    # density s: J and K together while both have one, then the rest of either
    for s in range(max(nj, nk)):
        var pj = dmj.unsafe_offset(min(s, max(nj - 1, 0)) * n2)
        var qj = aj.unsafe_offset(min(s, max(nj - 1, 0)) * n2)
        var pk = dmk.unsafe_offset(min(s, max(nk - 1, 0)) * n2)
        var qk = ak.unsafe_offset(min(s, max(nk - 1, 0)) * n2)
        if s < nj and s < nk:
            digest_any[True, True](blk, na, nb, nc, nd, i0, j0, k0, l0, nao, scale, pj, qj, pk, qk, ploc)
        elif s < nj:
            digest_any[True, False](blk, na, nb, nc, nd, i0, j0, k0, l0, nao, scale, pj, qj, pj, qk, ploc)
        else:
            digest_any[False, True](blk, na, nb, nc, nd, i0, j0, k0, l0, nao, scale, pj, qj, pk, qk, ploc)


comptime LMAX1 = 3      # kets of ``flush_batch1``: shells of degree lc + ld <= LMAX1
comptime NCLASS1 = 16   # their classes lc * 4 + ld


def single_class(basis: Basis, tab: PairTable, sp: Int) -> Int:
    """Class ``lc * 4 + ld`` of a pair ``flush_batch1`` takes as a ket, else -1.

    That is a pair with one primitive pair, one contraction per shell and
    2l + 1 functions per shell, of Hermite degree at most LMAX1.
    """
    if tab.get(sp, I_NP) != 1:
        return -1
    var c = tab.get(sp, I_A)
    var d = tab.get(sp, I_B)
    var lc = basis.l[c]
    var ld = basis.l[d]
    if lc + ld > LMAX1 or shell_nfunc(basis, c) != 2 * lc + 1 or shell_nfunc(basis, d) != 2 * ld + 1:
        return -1
    return lc * 4 + ld


def lanes1_scratch(nfmax: Int, nset: Int) -> Int:
    """Floats of scratch ``flush_batch1`` needs (bras of at most nfmax functions per shell, nset densities)."""
    var ncd = 15    # largest ket block of degree <= LMAX1 (d x p)
    var nc = 7
    var nomax = nfmax * nfmax
    return (nherm(LMAX1) * ncd + nomax * ncd + 2 * ncd + 6 * nfmax * nc + 2 * nc + nset * nomax) * W


def lanes_digest1[LC: Int, LD: Int](
    jk: DirectJK, sp: Int, kets: IntPtr, nket: Int, ipad: Int, ubuf: F64Ptr, scr: F64Ptr, nfmax: Int, n2: Int,
    nj: Int, nk: Int, dmj: F64Ptr, dmk: F64Ptr, aj: F64Ptr, ak: F64Ptr,
):
    """``flush_batch1`` for kets of shells of degree LC, LD: one ket per SIMD lane.

    For each W kets, the integrals (o|kl) = sum_h U[o][h] E[h][kl] of all bra
    functions o are formed with the lanes as the vector dimension, the
    densities each lane needs are gathered into lane-major blocks, the six
    J/K contractions of ``digest`` run as vector FMAs over the lanes, and the
    lanes' sums are added to the matrices at the end; the bra's Coulomb sums
    are reduced over the lanes once per batch.
    """
    comptime NC = 2 * LC + 1
    comptime ND = 2 * LD + 1
    comptime NCD = NC * ND
    comptime NHI = nherm(LC + LD)
    comptime V = SIMD[DType.float64, W]
    var nao = jk.nao
    var a = jk.tab.get(sp, I_A)
    var b = jk.tab.get(sp, I_B)
    var i0 = jk.basis.ao_loc[a]
    var na = jk.basis.ao_loc[a + 1] - i0
    var j0 = jk.basis.ao_loc[b]
    var nb = jk.basis.ao_loc[b + 1] - j0
    var no = na * nb
    var nomax = nfmax * nfmax
    var sab = 0.5 if a == b else 1.0
    var nset = max(nj, nk)
    var el = scr                                        # E of the lanes, [h][kl][lane]
    var gb = el.unsafe_offset(nherm(LMAX1) * 15 * W)    # (o|kl), [o][kl][lane]
    var dcd = gb.unsafe_offset(nomax * 15 * W)          # D_kl times the lane's weight, [kl][lane]
    var jcd = dcd.unsafe_offset(15 * W)                 # sum_ij (ij|kl) D_ij, [kl][lane]
    var dac = jcd.unsafe_offset(15 * W)                 # D_ik, D_il, D_jk, D_jl, [i or j][k or l][lane]
    var dad = dac.unsafe_offset(nfmax * 7 * W)
    var dbc = dad.unsafe_offset(nfmax * 7 * W)
    var dbd = dbc.unsafe_offset(nfmax * 7 * W)
    var kbc = dbd.unsafe_offset(nfmax * 7 * W)          # K_jk, K_jl over i, [j][k or l][lane]
    var kbd = kbc.unsafe_offset(nfmax * 7 * W)
    var kac = kbd.unsafe_offset(nfmax * 7 * W)          # K_ik, K_il over j, [k or l][lane]
    var kad = kac.unsafe_offset(7 * W)
    var jab = kad.unsafe_offset(7 * W)                  # sum_kl (ij|kl) D_kl, [set][o][lane]
    var k0s = stack_allocation[W, Int]()
    var l0s = stack_allocation[W, Int]()
    var sls = stack_allocation[W, Float64]()
    var acc = stack_allocation[NCD * W, Float64]()
    for x in range(nset * nomax):
        jab.unsafe_store(x * W, V(0.0))
    var v = 0
    while v * W < nket:
        var nx = min(W, nket - v * W)
        for x in range(nx):
            var ket = kets[unsafe_offset=v * W + x]
            var c = jk.tab.get(ket, I_A)
            var d = jk.tab.get(ket, I_B)
            k0s[unsafe_offset=x] = jk.basis.ao_loc[c]
            l0s[unsafe_offset=x] = jk.basis.ao_loc[d]
            var sl = 0.5 if c == d else 1.0
            if ket == sp:
                sl *= 0.5
            sls[unsafe_offset=x] = sl
            var e = jk.tab.e_ptr(ket)
            var sk = jk.tab.get(ket, I_STRIDE)
            comptime for h in range(NHI):
                comptime for kl in range(NCD):
                    el[unsafe_offset=(h * NCD + kl) * W + x] = e[unsafe_offset=h * sk + kl]
        # lanes past the batch: U is zero there and the stale E values are finite
        for o in range(no):
            var u = ubuf.unsafe_offset(o * NHI * ipad + v * W)
            var g = gb.unsafe_offset(o * NCD * W)
            comptime for kl in range(NCD):
                acc.unsafe_store(kl * W, V(0.0))
            comptime for h in range(NHI):
                var uh = u.unsafe_load[width=W](h * ipad)
                comptime for kl in range(NCD):
                    acc.unsafe_store(kl * W, acc.unsafe_load[width=W](kl * W) + uh * el.unsafe_load[width=W]((h * NCD + kl) * W))
            comptime for kl in range(NCD):
                g.unsafe_store(kl * W, acc.unsafe_load[width=W](kl * W))
        for st in range(nset):
            var do_j = st < nj
            var do_k = st < nk
            var pj = dmj.unsafe_offset(st * n2)
            var qj = aj.unsafe_offset(st * n2)
            var pk = dmk.unsafe_offset(st * n2)
            var qk = ak.unsafe_offset(st * n2)
            if do_j:
                for x in range(nx):
                    var sl = sls[unsafe_offset=x]
                    var k0 = k0s[unsafe_offset=x]
                    var l0 = l0s[unsafe_offset=x]
                    comptime for k in range(NC):
                        comptime for l in range(ND):
                            dcd[unsafe_offset=(k * ND + l) * W + x] = sl * pj[unsafe_offset=(k0 + k) * nao + l0 + l]
                comptime for kl in range(NCD):
                    jcd.unsafe_store(kl * W, V(0.0))
            if do_k:
                for x in range(nx):
                    var k0 = k0s[unsafe_offset=x]
                    var l0 = l0s[unsafe_offset=x]
                    for i in range(na):
                        var r = pk.unsafe_offset((i0 + i) * nao)
                        comptime for k in range(NC):
                            dac[unsafe_offset=(i * NC + k) * W + x] = r[unsafe_offset=k0 + k]
                        comptime for l in range(ND):
                            dad[unsafe_offset=(i * ND + l) * W + x] = r[unsafe_offset=l0 + l]
                    for j in range(nb):
                        var r = pk.unsafe_offset((j0 + j) * nao)
                        comptime for k in range(NC):
                            dbc[unsafe_offset=(j * NC + k) * W + x] = r[unsafe_offset=k0 + k]
                        comptime for l in range(ND):
                            dbd[unsafe_offset=(j * ND + l) * W + x] = r[unsafe_offset=l0 + l]
                for x in range(nb * NC):
                    kbc.unsafe_store(x * W, V(0.0))
                for x in range(nb * ND):
                    kbd.unsafe_store(x * W, V(0.0))
            for i in range(na):
                if do_k:
                    comptime for k in range(NC):
                        kac.unsafe_store(k * W, V(0.0))
                    comptime for l in range(ND):
                        kad.unsafe_store(l * W, V(0.0))
                for j in range(nb):
                    var o = i * nb + j
                    var g = gb.unsafe_offset(o * NCD * W)
                    if do_j:
                        var dij = pj[unsafe_offset=(i0 + i) * nao + j0 + j]
                        var sj = V(0.0)
                        comptime for kl in range(NCD):
                            var gv = g.unsafe_load[width=W](kl * W)
                            sj += gv * dcd.unsafe_load[width=W](kl * W)
                            jcd.unsafe_store(kl * W, jcd.unsafe_load[width=W](kl * W) + gv * dij)
                        var pjab = jab.unsafe_offset((st * nomax + o) * W)
                        pjab.unsafe_store(0, pjab.unsafe_load[width=W](0) + sj)
                    if do_k:
                        comptime for k in range(NC):
                            var sac = V(0.0)
                            var sbc = V(0.0)
                            var dbck = dbc.unsafe_load[width=W]((j * NC + k) * W)
                            var dack = dac.unsafe_load[width=W]((i * NC + k) * W)
                            comptime for l in range(ND):
                                var gv = g.unsafe_load[width=W]((k * ND + l) * W)
                                sac += gv * dbd.unsafe_load[width=W]((j * ND + l) * W)
                                sbc += gv * dad.unsafe_load[width=W]((i * ND + l) * W)
                                kad.unsafe_store(l * W, kad.unsafe_load[width=W](l * W) + gv * dbck)
                                kbd.unsafe_store((j * ND + l) * W, kbd.unsafe_load[width=W]((j * ND + l) * W) + gv * dack)
                            kac.unsafe_store(k * W, kac.unsafe_load[width=W](k * W) + sac)
                            kbc.unsafe_store((j * NC + k) * W, kbc.unsafe_load[width=W]((j * NC + k) * W) + sbc)
                if do_k:
                    var r = qk.unsafe_offset((i0 + i) * nao)
                    for x in range(nx):
                        var f = sab * sls[unsafe_offset=x]
                        var k0 = k0s[unsafe_offset=x]
                        var l0 = l0s[unsafe_offset=x]
                        comptime for k in range(NC):
                            r[unsafe_offset=k0 + k] += f * kac[unsafe_offset=k * W + x]
                        comptime for l in range(ND):
                            r[unsafe_offset=l0 + l] += f * kad[unsafe_offset=l * W + x]
            if do_k:
                for x in range(nx):
                    var f = sab * sls[unsafe_offset=x]
                    var k0 = k0s[unsafe_offset=x]
                    var l0 = l0s[unsafe_offset=x]
                    for j in range(nb):
                        var r = qk.unsafe_offset((j0 + j) * nao)
                        comptime for k in range(NC):
                            r[unsafe_offset=k0 + k] += f * kbc[unsafe_offset=(j * NC + k) * W + x]
                        comptime for l in range(ND):
                            r[unsafe_offset=l0 + l] += f * kbd[unsafe_offset=(j * ND + l) * W + x]
            if do_j:
                for x in range(nx):
                    var f = 2.0 * sab * sls[unsafe_offset=x]
                    var k0 = k0s[unsafe_offset=x]
                    var l0 = l0s[unsafe_offset=x]
                    comptime for k in range(NC):
                        comptime for l in range(ND):
                            qj[unsafe_offset=(k0 + k) * nao + l0 + l] += f * jcd[unsafe_offset=(k * ND + l) * W + x]
        v += 1
    for st in range(nj):
        var qj = aj.unsafe_offset(st * n2)
        for i in range(na):
            for j in range(nb):
                var sj = jab.unsafe_load[width=W]((st * nomax + i * nb + j) * W).reduce_add()
                qj[unsafe_offset=(i0 + i) * nao + j0 + j] += 2.0 * sab * sj


def flush_batch1(
    jk: DirectJK, tb: PairTable, sp: Int, kets: IntPtr, nket: Int, cls: Int, mut ws: EriWork, mut wb: EriBatch,
    scr: F64Ptr, nfmax: Int, n2: Int, nj: Int, nk: Int, dmj: F64Ptr, dmk: F64Ptr, aj: F64Ptr, ak: F64Ptr,
):
    """Integrals of bra ``sp`` with queued kets of class ``cls`` (``single_class``) folded into the J/K
    accumulators, one ket per SIMD lane (``lanes_digest1``)."""
    var ipad = eri_batch_u(tb, sp, jk.tab, kets, nket, jk.boys, ws, wb)
    var ubuf = F64Ptr(unsafe_from_address=wb.uaddr)
    comptime for LC in range(LMAX1 + 1):
        comptime for LD in range(LMAX1 + 1 - LC):
            if cls == LC * 4 + LD:
                lanes_digest1[LC, LD](jk, sp, kets, nket, ipad, ubuf, scr, nfmax, n2, nj, nk, dmj, dmk, aj, ak)
                return


comptime JK_LANES_MIN = 3   # densities from which ``DirectJK.jk`` digests them as SIMD lanes
comptime LANES_BUDGET = 1536 * 1024 * 1024    # bytes of the workers' lane accumulators in one pass


def digest_lanes[DO_J: Bool, DO_K: Bool, NC: Int, ND: Int](
    blk: F64Ptr, na: Int, nb: Int, i0: Int, j0: Int, k0: Int, l0: Int, nao: Int, scale: Float64,
    dj: F64Ptr, aj: F64Ptr, dk: F64Ptr, ak: F64Ptr,
):
    """``digest_cd`` for W densities at once.

    ``dj``, ``dk`` and the accumulators ``aj``, ``ak`` are lane arrays:
    element (r, c) of density t of the group at (r nao + c) W + t.  Every
    integral is broadcast and updates the W densities with vector FMAs.
    """
    comptime NCD = NC * ND
    comptime V = SIMD[DType.float64, W]
    var s2 = 2.0 * scale
    var dcd = stack_allocation[NCD * W, Float64, alignment=64]()    # D_cd (J)
    var jcd = stack_allocation[NCD * W, Float64, alignment=64]()    # 2 s sum_ij (ij|cd) D_ij (J)
    var dad = stack_allocation[ND * W, Float64, alignment=64]()     # D_il, D_ik of the current bra row i (K)
    var dac = stack_allocation[NC * W, Float64, alignment=64]()
    var dbd = stack_allocation[ND * W, Float64, alignment=64]()     # D_jl, D_jk of the current j (K)
    var dbc = stack_allocation[NC * W, Float64, alignment=64]()
    var kad = stack_allocation[ND * W, Float64, alignment=64]()     # K_il over j, k; K_ik over j, l (row i)
    var kac = stack_allocation[NC * W, Float64, alignment=64]()
    var kbd = stack_allocation[ND * W, Float64, alignment=64]()     # K_jl over k (one (i, j))
    comptime if DO_J:
        comptime for k in range(NC):
            comptime for l in range(ND):
                dcd.unsafe_store((k * ND + l) * W, dj.unsafe_load[width=W](((k0 + k) * nao + l0 + l) * W))
                jcd.unsafe_store((k * ND + l) * W, V(0.0))
    for i in range(na):
        var ri = (i0 + i) * nao
        comptime if DO_K:
            comptime for l in range(ND):
                dad.unsafe_store(l * W, dk.unsafe_load[width=W]((ri + l0 + l) * W))
                kad.unsafe_store(l * W, V(0.0))
            comptime for k in range(NC):
                dac.unsafe_store(k * W, dk.unsafe_load[width=W]((ri + k0 + k) * W))
                kac.unsafe_store(k * W, V(0.0))
        for j in range(nb):
            var rj = (j0 + j) * nao
            var g = blk.unsafe_offset((i * nb + j) * NCD)
            var dij = V(0.0)
            comptime if DO_J:
                dij = dj.unsafe_load[width=W]((ri + j0 + j) * W) * s2
            comptime if DO_K:
                comptime for l in range(ND):
                    dbd.unsafe_store(l * W, dk.unsafe_load[width=W]((rj + l0 + l) * W))
                    kbd.unsafe_store(l * W, V(0.0))
                comptime for k in range(NC):
                    dbc.unsafe_store(k * W, dk.unsafe_load[width=W]((rj + k0 + k) * W))
            var sj = V(0.0)
            comptime for k in range(NC):
                var sac = V(0.0)
                var sbc = V(0.0)
                comptime for l in range(ND):
                    var v = g[unsafe_offset=k * ND + l]
                    comptime if DO_J:
                        sj += v * dcd.unsafe_load[width=W]((k * ND + l) * W)
                        jcd.unsafe_store((k * ND + l) * W, jcd.unsafe_load[width=W]((k * ND + l) * W) + v * dij)
                    comptime if DO_K:
                        sac += v * dbd.unsafe_load[width=W](l * W)
                        sbc += v * dad.unsafe_load[width=W](l * W)
                        kad.unsafe_store(l * W, kad.unsafe_load[width=W](l * W) + v * dbc.unsafe_load[width=W](k * W))
                        kbd.unsafe_store(l * W, kbd.unsafe_load[width=W](l * W) + v * dac.unsafe_load[width=W](k * W))
                comptime if DO_K:
                    kac.unsafe_store(k * W, kac.unsafe_load[width=W](k * W) + sac)
                    var o = (rj + k0 + k) * W
                    ak.unsafe_store(o, ak.unsafe_load[width=W](o) + sbc * scale)
            comptime if DO_J:
                var o = (ri + j0 + j) * W
                aj.unsafe_store(o, aj.unsafe_load[width=W](o) + sj * s2)
            comptime if DO_K:
                comptime for l in range(ND):
                    var o = (rj + l0 + l) * W
                    ak.unsafe_store(o, ak.unsafe_load[width=W](o) + kbd.unsafe_load[width=W](l * W) * scale)
        comptime if DO_K:
            comptime for k in range(NC):
                var o = (ri + k0 + k) * W
                ak.unsafe_store(o, ak.unsafe_load[width=W](o) + kac.unsafe_load[width=W](k * W) * scale)
            comptime for l in range(ND):
                var o = (ri + l0 + l) * W
                ak.unsafe_store(o, ak.unsafe_load[width=W](o) + kad.unsafe_load[width=W](l * W) * scale)
    comptime if DO_J:
        comptime for k in range(NC):
            comptime for l in range(ND):
                var o = ((k0 + k) * nao + l0 + l) * W
                aj.unsafe_store(o, aj.unsafe_load[width=W](o) + jcd.unsafe_load[width=W]((k * ND + l) * W))


def digest_lanes_gen[DO_J: Bool, DO_K: Bool](
    blk: F64Ptr, na: Int, nb: Int, nc: Int, nd: Int, i0: Int, j0: Int, k0: Int, l0: Int, nao: Int, scale: Float64,
    dj: F64Ptr, aj: F64Ptr, dk: F64Ptr, ak: F64Ptr, loc: F64Ptr,
):
    """``digest_lanes`` for any ket (run-time loops; ``loc``: 2 nc nd + 3 nd + 4 nc vectors of scratch)."""
    comptime V = SIMD[DType.float64, W]
    var ncd = nc * nd
    var s2 = 2.0 * scale
    var dcd = loc
    var jcd = dcd.unsafe_offset(ncd * W)
    var dad = jcd.unsafe_offset(ncd * W)
    var dac = dad.unsafe_offset(nd * W)
    var dbd = dac.unsafe_offset(nc * W)
    var dbc = dbd.unsafe_offset(nd * W)
    var kad = dbc.unsafe_offset(nc * W)
    var kac = kad.unsafe_offset(nd * W)
    var kbd = kac.unsafe_offset(nc * W)
    comptime if DO_J:
        for k in range(nc):
            for l in range(nd):
                dcd.unsafe_store((k * nd + l) * W, dj.unsafe_load[width=W](((k0 + k) * nao + l0 + l) * W))
                jcd.unsafe_store((k * nd + l) * W, V(0.0))
    for i in range(na):
        var ri = (i0 + i) * nao
        comptime if DO_K:
            for l in range(nd):
                dad.unsafe_store(l * W, dk.unsafe_load[width=W]((ri + l0 + l) * W))
                kad.unsafe_store(l * W, V(0.0))
            for k in range(nc):
                dac.unsafe_store(k * W, dk.unsafe_load[width=W]((ri + k0 + k) * W))
                kac.unsafe_store(k * W, V(0.0))
        for j in range(nb):
            var rj = (j0 + j) * nao
            var g = blk.unsafe_offset((i * nb + j) * ncd)
            var dij = V(0.0)
            comptime if DO_J:
                dij = dj.unsafe_load[width=W]((ri + j0 + j) * W) * s2
            comptime if DO_K:
                for l in range(nd):
                    dbd.unsafe_store(l * W, dk.unsafe_load[width=W]((rj + l0 + l) * W))
                    kbd.unsafe_store(l * W, V(0.0))
                for k in range(nc):
                    dbc.unsafe_store(k * W, dk.unsafe_load[width=W]((rj + k0 + k) * W))
            var sj = V(0.0)
            for k in range(nc):
                var sac = V(0.0)
                var sbc = V(0.0)
                var dbck = dbc.unsafe_load[width=W](k * W)
                var dack = dac.unsafe_load[width=W](k * W)
                for l in range(nd):
                    var v = g[unsafe_offset=k * nd + l]
                    comptime if DO_J:
                        sj += v * dcd.unsafe_load[width=W]((k * nd + l) * W)
                        jcd.unsafe_store((k * nd + l) * W, jcd.unsafe_load[width=W]((k * nd + l) * W) + v * dij)
                    comptime if DO_K:
                        sac += v * dbd.unsafe_load[width=W](l * W)
                        sbc += v * dad.unsafe_load[width=W](l * W)
                        kad.unsafe_store(l * W, kad.unsafe_load[width=W](l * W) + v * dbck)
                        kbd.unsafe_store(l * W, kbd.unsafe_load[width=W](l * W) + v * dack)
                comptime if DO_K:
                    kac.unsafe_store(k * W, kac.unsafe_load[width=W](k * W) + sac)
                    var o = (rj + k0 + k) * W
                    ak.unsafe_store(o, ak.unsafe_load[width=W](o) + sbc * scale)
            comptime if DO_J:
                var o = (ri + j0 + j) * W
                aj.unsafe_store(o, aj.unsafe_load[width=W](o) + sj * s2)
            comptime if DO_K:
                for l in range(nd):
                    var o = (rj + l0 + l) * W
                    ak.unsafe_store(o, ak.unsafe_load[width=W](o) + kbd.unsafe_load[width=W](l * W) * scale)
        comptime if DO_K:
            for k in range(nc):
                var o = (ri + k0 + k) * W
                ak.unsafe_store(o, ak.unsafe_load[width=W](o) + kac.unsafe_load[width=W](k * W) * scale)
            for l in range(nd):
                var o = (ri + l0 + l) * W
                ak.unsafe_store(o, ak.unsafe_load[width=W](o) + kad.unsafe_load[width=W](l * W) * scale)
    comptime if DO_J:
        for k in range(nc):
            for l in range(nd):
                var o = ((k0 + k) * nao + l0 + l) * W
                aj.unsafe_store(o, aj.unsafe_load[width=W](o) + jcd.unsafe_load[width=W]((k * nd + l) * W))


def _digest_lanes_any[DO_J: Bool, DO_K: Bool](
    blk: F64Ptr, na: Int, nb: Int, nc: Int, nd: Int, i0: Int, j0: Int, k0: Int, l0: Int, nao: Int, scale: Float64,
    dj: F64Ptr, aj: F64Ptr, dk: F64Ptr, ak: F64Ptr, loc: F64Ptr,
):
    """``digest_lanes`` for kets of s to f shells (spherical, one contraction), else ``digest_lanes_gen``."""
    comptime for KC in range(4):
        comptime for KD in range(4):
            if nc == 2 * KC + 1 and nd == 2 * KD + 1:
                digest_lanes[DO_J, DO_K, 2 * KC + 1, 2 * KD + 1](blk, na, nb, i0, j0, k0, l0, nao, scale, dj, aj, dk, ak)
                return
    digest_lanes_gen[DO_J, DO_K](blk, na, nb, nc, nd, i0, j0, k0, l0, nao, scale, dj, aj, dk, ak, loc)


comptime QBUF_WORDS = 1 << 17    # integrals of one bra's quartets kept for ``QuartetBuffer.digest``, at most


struct QuartetBuffer(Movable):
    """The integral blocks of one bra's quartets in ``_jk_bra`` with lanes, digested group by group.

    Digesting every quartet for all groups of densities in turn reads, per
    quartet, a few rows of every group's lane arrays (megabytes apart) and
    missed the caches; group by group, the bra's rows of one group stay in
    cache for all its quartets.
    """

    var blocks: List[Float64]
    var meta: List[Int]         # per quartet: c, d, same pair, offset of its block
    var used: Int
    var cap: Int

    def __init__(out self, cap: Int):
        self.cap = cap
        self.blocks = List[Float64](length=max(cap, 1) + 8, fill=0.0)
        self.meta = List[Int]()
        self.used = 0

    def room(self, n: Int) -> Bool:
        return self.used + n <= self.cap

    def push(mut self, c: Int, d: Int, same: Bool, src: F64Ptr, n: Int):
        var dst = list_ptr(self.blocks).unsafe_offset(self.used)
        for e in range(n):
            dst[unsafe_offset=e] = src[unsafe_offset=e]
        self.meta.append(c)
        self.meta.append(d)
        self.meta.append(1 if same else 0)
        self.meta.append(self.used)
        self.used += n

    def digest(
        mut self, basis: Basis, a: Int, b: Int, nao: Int, gst: Int, ngj: Int, ngk: Int, dlj: F64Ptr, dlk: F64Ptr,
        aj: F64Ptr, ak: F64Ptr, loc: F64Ptr,
    ):
        """Fold the buffered quartets of bra (a, b) into the lane accumulators, group by group
        (``digest_lanes``), and empty the buffer."""
        var nq = len(self.meta) // 4
        var pb = list_ptr(self.blocks)
        var i0 = basis.ao_loc[a]
        var na = basis.ao_loc[a + 1] - i0
        var j0 = basis.ao_loc[b]
        var nb = basis.ao_loc[b + 1] - j0
        var sab = 0.5 if a == b else 1.0
        for g in range(max(ngj, ngk)):
            var pj = dlj.unsafe_offset(min(g, max(ngj - 1, 0)) * gst)
            var qj = aj.unsafe_offset(min(g, max(ngj - 1, 0)) * gst)
            var pk = dlk.unsafe_offset(min(g, max(ngk - 1, 0)) * gst)
            var qk = ak.unsafe_offset(min(g, max(ngk - 1, 0)) * gst)
            for q in range(nq):
                var c = self.meta[4 * q]
                var d = self.meta[4 * q + 1]
                var blk = pb.unsafe_offset(self.meta[4 * q + 3])
                var k0 = basis.ao_loc[c]
                var nc = basis.ao_loc[c + 1] - k0
                var l0 = basis.ao_loc[d]
                var nd = basis.ao_loc[d + 1] - l0
                var scale = sab
                if c == d:
                    scale *= 0.5
                if self.meta[4 * q + 2] != 0:
                    scale *= 0.5
                if g < ngj and g < ngk:
                    _digest_lanes_any[True, True](blk, na, nb, nc, nd, i0, j0, k0, l0, nao, scale, pj, qj, pk, qk, loc)
                elif g < ngj:
                    _digest_lanes_any[True, False](blk, na, nb, nc, nd, i0, j0, k0, l0, nao, scale, pj, qj, pj, qk, loc)
                else:
                    _digest_lanes_any[False, True](blk, na, nb, nc, nd, i0, j0, k0, l0, nao, scale, pj, qj, pk, qk, loc)
        self.meta.clear()
        self.used = 0


def flush_batch[LANES: Bool](
    jk: DirectJK, tb: PairTable, sp: Int, kets: IntPtr, nket: Int, mut ws: EriWork, mut wb: EriBatch, n2: Int,
    nj: Int, nk: Int, dmj: F64Ptr, dmk: F64Ptr, aj: F64Ptr, ak: F64Ptr, ploc: F64Ptr, mut qb: QuartetBuffer,
):
    """Integrals of bra ``sp`` (of ``tb``: ``jk.tab`` or its attenuated copy) with the queued kets
    (``eri_batch``) folded into the J/K accumulators (LANES: into ``qb``, as in ``_jk_bra``)."""
    eri_batch(tb, sp, jk.tab, kets, nket, jk.boys, ws, wb)
    var a = jk.tab.get(sp, I_A)
    var b = jk.tab.get(sp, I_B)
    var nab = jk.tab.get(sp, I_NCOMP)
    for k in range(nket):
        var sk = kets[unsafe_offset=k]
        comptime if LANES:
            var n = nab * jk.tab.get(sk, I_NCOMP)
            if not qb.room(n):
                qb.digest(jk.basis, a, b, jk.nao, n2, nj, nk, dmj, dmk, aj, ak, ploc)
            qb.push(jk.tab.get(sk, I_A), jk.tab.get(sk, I_B), sk == sp, wb.block(k), n)
        else:
            digest_quartet(
                jk.basis, a, b, jk.tab.get(sk, I_A), jk.tab.get(sk, I_B), sk == sp, wb.block(k),
                jk.nao, n2, nj, nk, dmj, dmk, aj, ak, ploc,
            )


def _jk_bra[LANES: Bool](
    self: DirectJK, tb: PairTable, sp: Int, mut ws: EriWork, mut wb: EriBatch, mut queue: KetQueue, ploc: F64Ptr,
    pcond: F64Ptr, pq: F64Ptr, npairs: Int, nbas: Int, nao: Int, n2: Int, nj: Int, nk: Int, dmj: F64Ptr,
    dmk: F64Ptr, aj: F64Ptr, ak: F64Ptr, tol: Float64, lscr: F64Ptr, mut qb: QuartetBuffer,
):
    """The quartets of bra pair ``sp`` folded into a worker's J/K accumulators (``DirectJK.jk``); ``tb`` is the
    bra-side pair table (``self.tab``, or its attenuated copy for the long-range operator).  With LANES the
    densities are SIMD lanes (``digest_lanes``): ``nj``, ``nk`` count groups of W, ``dmj``, ``dmk``,
    ``aj``, ``ak`` are lane arrays of ``n2`` (= nao^2 W) values per group, ``ploc`` is that kernel's
    scratch, the kets of one primitive pair go through ``eri_batch`` with the others, and the integral
    blocks are collected in ``qb`` and digested group by group."""
    var qab = pq[unsafe_offset=sp]
    if qab == 0.0:
        return
    var pkcls = int_ptr(self.kcls)
    var pkone = int_ptr(self.kone)
    var a = self.tab.get(sp, I_A)
    var b = self.tab.get(sp, I_B)
    var dab = pcond[unsafe_offset=a * nbas + b]
    var npb = self.tab.get(sp, I_NP)
    var batch_bra = batch_supported(self.tab.get(sp, I_LAB), 0)
    # each unique quartet belongs to the pair with more primitive pairs (ties: the
    # higher index), so the kets are the less contracted side
    for spk in range(npairs):
        var npk = self.tab.get(spk, I_NP)
        if npk > npb or (npk == npb and spk > sp):
            continue
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
        var kc = pkcls[unsafe_offset=spk] if batch_bra else -1
        if kc >= 0 and lanes_preferred(self.tab, sp, self.tab, spk):
            comptime if not LANES:
                var k1 = pkone[unsafe_offset=spk]
                if k1 >= 0:
                    var q1 = self.nclass + k1
                    if queue.full(q1, 1):
                        flush_batch1(
                            self, tb, sp, queue.kets(q1), queue.cnt[q1], k1, ws, wb, lscr, self.nfmax, n2, nj, nk,
                            dmj, dmk, aj, ak,
                        )
                        queue.clear(q1)
                    queue.push(q1, spk, 1)
                    continue
            if queue.full(kc, npk):
                flush_batch[LANES](
                    self, tb, sp, queue.kets(kc), queue.cnt[kc], ws, wb, n2, nj, nk, dmj, dmk, aj, ak, ploc, qb
                )
                queue.clear(kc)
            queue.push(kc, spk, npk)
            continue
        if not eri_quartet(tb, sp, self.tab, spk, self.ht, self.boys, ws):
            continue
        comptime if LANES:
            var n = self.tab.get(sp, I_NCOMP) * self.tab.get(spk, I_NCOMP)
            if not qb.room(n):
                qb.digest(self.basis, a, b, nao, n2, nj, nk, dmj, dmk, aj, ak, ploc)
            qb.push(c, d, sp == spk, list_ptr(ws.out), n)
        else:
            digest_quartet(
                self.basis, a, b, c, d, sp == spk, list_ptr(ws.out), nao, n2, nj, nk, dmj, dmk, aj, ak, ploc
            )
    # the rest of this bra's batches
    for kc in range(self.nclass):
        if queue.cnt[kc] > 0:
            flush_batch[LANES](
                self, tb, sp, queue.kets(kc), queue.cnt[kc], ws, wb, n2, nj, nk, dmj, dmk, aj, ak, ploc, qb
            )
            queue.clear(kc)
    comptime if LANES:
        qb.digest(self.basis, a, b, nao, n2, nj, nk, dmj, dmk, aj, ak, ploc)
    else:
        for k1 in range(NCLASS1):
            var q1 = self.nclass + k1
            if queue.cnt[q1] > 0:
                flush_batch1(
                    self, tb, sp, queue.kets(q1), queue.cnt[q1], k1, ws, wb, lscr, self.nfmax, n2, nj, nk, dmj, dmk,
                    aj, ak,
                )
                queue.clear(q1)


struct DirectJK(Movable):
    """Integral-direct J and K for one basis (see the module docstring)."""

    var active: Bool
    var basis: Basis
    var boys: BoysTable
    var ht: HermTable
    var tab: PairTable
    var tabw: PairTable       # the bra side for omega > 0 (attenuated copy of tab), else empty
    var omega: Float64
    var q: List[Float64]      # Schwarz bound per shell pair a >= b (index a (a + 1) / 2 + b)
    var kcls: List[Int]       # batch class of each pair as a ket (-1: not batched)
    var kone: List[Int]       # class of each pair as a ket of ``flush_batch1`` (-1: not one)
    var order: List[Int]      # bra pairs in the order the workers take them: most primitive pairs first
    var nclass: Int
    var nbas: Int
    var nao: Int
    var nfmax: Int

    def __init__(out self, var basis: Basis, var boys: BoysTable, omega: Float64 = 0.0):
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
        var none = List[Int]()
        var tabw: PairTable
        if omega > 0.0:
            tabw = PairTable(basis, basis, sa, sb, ht)
            attenuate(tabw, omega)
        else:
            tabw = PairTable(basis, basis, none, none, ht)
        self.tabw = tabw^
        self.omega = omega
        var nfmax = 1
        for a in range(nbas):
            nfmax = max(nfmax, shell_nfunc(basis, a))
        var kcls = List[Int](length=max(npairs, 1), fill=-1)
        self.nclass = batch_classes(tab, kcls)
        self.kcls = kcls^
        var kone = List[Int](length=max(npairs, 1), fill=-1)
        for sp in range(npairs):
            kone[sp] = single_class(basis, tab, sp)
        self.kone = kone^
        # a bra takes the kets with at most as many primitive pairs, so the bras with the most primitive
        # pairs have the most work: they go first, which keeps the workers' last tasks short
        var maxnp = 0
        for sp in range(npairs):
            maxnp = max(maxnp, tab.get(sp, I_NP))
        var start = List[Int](length=maxnp + 2, fill=0)
        for sp in range(npairs):
            start[tab.get(sp, I_NP)] += 1
        var acc = 0
        for np in range(maxnp, -1, -1):
            var c = start[np]
            start[np] = acc
            acc += c
        var order = List[Int](length=max(npairs, 1), fill=0)
        for sp in range(npairs - 1, -1, -1):
            var np = tab.get(sp, I_NP)
            order[start[np]] = sp
            start[np] += 1
        self.order = order^
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
        self.tabw = PairTable(basis, basis, sa, sb, ht)
        self.omega = 0.0
        self.ht = ht^
        self.active = False
        self.nbas = 0
        self.nao = 0
        self.nfmax = 1
        self.boys = BoysTable(empty=True)
        self.q = List[Float64]()
        self.kcls = List[Int]()
        self.kone = List[Int]()
        self.order = List[Int]()
        self.nclass = 0
        self.basis = basis^
        _ = sa^
        _ = sb^

    def _cond(self, nj: Int, dmj: F64Ptr, nk: Int, dmk: F64Ptr) -> List[Float64]:
        """The largest density element of every shell block (a nbas + b), over all densities."""
        var nao = self.nao
        var n2 = nao * nao
        var nbas = self.nbas
        var cond = List[Float64](length=max(nbas * nbas, 1), fill=0.0)
        for a in range(nbas):
            var i0 = self.basis.ao_loc[a]
            var i1 = self.basis.ao_loc[a + 1]
            for b in range(nbas):
                var j0 = self.basis.ao_loc[b]
                var j1 = self.basis.ao_loc[b + 1]
                var m = 0.0
                for s in range(nj + nk):
                    var d = dmj.unsafe_offset(s * n2) if s < nj else dmk.unsafe_offset((s - nj) * n2)
                    for i in range(i0, i1):
                        for j in range(j0, j1):
                            var v = abs(d[unsafe_offset=i * nao + j])
                            if v > m:
                                m = v
                cond[a * nbas + b] = m
        return cond^

    def _jk_lanes(
        self, nj: Int, dmj: F64Ptr, vj: F64Ptr, nk: Int, dmk: F64Ptr, vk: F64Ptr, tol: Float64, nanti: Int
    ):
        """``jk`` with the densities as SIMD lanes.

        Groups of W densities (density g W + t in lane t of group g) are
        interleaved into lane arrays of nao^2 W values, element (r, c) at
        (r nao + c) W + t, and every quartet is digested for all of them with
        vector FMAs (``digest_lanes``) instead of once per density.
        A pass over the integrals takes as many groups as keep the workers'
        accumulators within LANES_BUDGET bytes.  When ``dmj`` and ``dmk`` are
        the same array (J of its first nj densities, K of all nk) the lanes
        are shared.
        """
        var nao = self.nao
        var n2 = nao * nao
        var gst = n2 * W
        var nbas = self.nbas
        var npairs = nbas * (nbas + 1) // 2
        var cond = self._cond(nj, dmj, nk, dmk)
        var pcond = list_ptr(cond)
        var ngj_all = (nj + W - 1) // W
        var ngk_all = (nk + W - 1) // W
        var shared = Int(dmj) == Int(dmk) and nj <= nk
        var nworkers = min(parallelism_level(), npairs) if npairs >= 16 else 1
        var gmax = max(1, LANES_BUDGET // (nworkers * 2 * gst * 8))
        var pq = list_ptr(self.q)
        var nfmax = self.nfmax
        var attenuated = self.omega > 0.0
        var g0 = 0
        while g0 < max(ngj_all, ngk_all):
            var ngj = min(gmax, max(ngj_all - g0, 0))     # groups of this pass
            var ngk = min(gmax, max(ngk_all - g0, 0))
            var s0 = g0 * W
            var mj = min(ngj * W, max(nj - s0, 0))         # densities of this pass
            var mk = min(ngk * W, max(nk - s0, 0))
            var nl = ngk if shared else ngj + ngk
            var dl = List[Float64](length=max(nl, 1) * gst + 2 * W, fill=0.0)    # lanes past the densities: zero
            var pdl = F64Ptr(unsafe_from_address=aligned_addr(dl))
            var dlk = pdl
            var dlj = pdl if shared else pdl.unsafe_offset(ngk * gst)
            for t in range(mk):
                var d = dmk.unsafe_offset((s0 + t) * n2)
                var dst = dlk.unsafe_offset((t // W) * gst + t % W)
                for e in range(n2):
                    dst[unsafe_offset=e * W] = d[unsafe_offset=e]
            if not shared:
                for t in range(mj):
                    var d = dmj.unsafe_offset((s0 + t) * n2)
                    var dst = dlj.unsafe_offset((t // W) * gst + t % W)
                    for e in range(n2):
                        dst[unsafe_offset=e * W] = d[unsafe_offset=e]
            var per = (ngj + ngk) * gst
            var acc = List[Float64](length=nworkers * per + 2 * W, fill=0.0)
            var pacc = F64Ptr(unsafe_from_address=aligned_addr(acc))
            var counter = Atomic[Int64](0)
            var pcount = Pointer(to=counter)

            def work(w: Int) {imm self, imm pacc, imm pcond, imm pq, imm pcount, imm npairs, imm nbas, imm nao, imm gst, imm per, imm ngj, imm ngk, imm dlj, imm dlk, imm tol, imm nfmax, imm attenuated}:
                var ws = EriWork(self.tab.maxcomp, self.tab.maxlab, self.tab.maxcomp, self.tab.maxlab)
                var wb = EriBatch(self.tab.maxcomp, self.tab.maxcomp)
                var loc = List[Float64](length=(2 * nfmax * nfmax + 7 * nfmax) * W + 2 * W, fill=0.0)
                var ploc = F64Ptr(unsafe_from_address=aligned_addr(loc))
                var aj = pacc.unsafe_offset(w * per)
                var ak = aj.unsafe_offset(ngj * gst)
                var queue = KetQueue(self.nclass + NCLASS1)
                var qb = QuartetBuffer(max(QBUF_WORDS, self.tab.maxcomp * self.tab.maxcomp))
                while True:
                    var task = Int(pcount[].fetch_add(1))
                    if task >= npairs:
                        break
                    var sp = self.order[task]
                    if attenuated:
                        _jk_bra[True](
                            self, self.tabw, sp, ws, wb, queue, ploc, pcond, pq, npairs, nbas, nao, gst, ngj, ngk, dlj,
                            dlk, aj, ak, tol, ploc, qb,
                        )
                    else:
                        _jk_bra[True](
                            self, self.tab, sp, ws, wb, queue, ploc, pcond, pq, npairs, nbas, nao, gst, ngj, ngk, dlj,
                            dlk, aj, ak, tol, ploc, qb,
                        )
                _ = wb^
                _ = queue^
                _ = ws^
                _ = loc^
                _ = qb^

            if nworkers == 1:
                work(0)
            else:
                parallelize(work, nworkers)
            if nworkers > 1:
                var nchunk = min(64, max(1, per // 65536))

                def reduce(cidx: Int) {imm pacc, imm per, imm nchunk, imm nworkers}:
                    var lo = per * cidx // nchunk
                    var hi = per * (cidx + 1) // nchunk
                    for w2 in range(1, nworkers):
                        vaxpy(pacc.unsafe_offset(lo), hi - lo, 1.0, pacc.unsafe_offset(w2 * per + lo))

                if nchunk == 1:
                    reduce(0)
                else:
                    parallelize(reduce, nchunk)
            # out of the lanes: J = A + A^T, K = A + A^T (A - A^T for the antisymmetric densities)
            var nout = mj + mk

            def unpack(t2: Int) {imm pacc, imm gst, imm ngj, imm mj, imm s0, imm nao, imm n2, imm vj, imm vk, imm nk, imm nanti}:
                var t = t2 if t2 < mj else t2 - mj
                var src = pacc.unsafe_offset(((t // W) if t2 < mj else ngj + t // W) * gst + t % W)
                var dst = vj.unsafe_offset((s0 + t) * n2) if t2 < mj else vk.unsafe_offset((s0 + t) * n2)
                var sgn = -1.0 if t2 >= mj and s0 + t >= nk - nanti else 1.0
                for a in range(nao):
                    for b in range(nao):
                        dst[unsafe_offset=a * nao + b] = (
                            src[unsafe_offset=(a * nao + b) * W] + sgn * src[unsafe_offset=(b * nao + a) * W]
                        )

            if nout > 0:
                parallelize(unpack, nout)
            _ = dl^
            _ = acc^
            _ = counter^
            g0 += gmax
        _ = cond^

    def jk(
        self, nj: Int, dmj: F64Ptr, vj: F64Ptr, nk: Int, dmk: F64Ptr, vk: F64Ptr, tol: Float64, nanti: Int = 0
    ):
        """vj[s] = J[dmj[s]] (nj densities) and vk[s] = K[dmk[s]] (nk densities); outputs overwritten.

        The densities are symmetric (nao x nao, row-major), except the last
        ``nanti`` exchange densities, which are antisymmetric: the four index
        orderings the digestion adds are then minus the transpose of the
        other four, K = A - A^T.  From JK_LANES_MIN densities on they are
        digested as SIMD lanes (``_jk_lanes``).
        """
        var nao = self.nao
        var n2 = nao * nao
        var nbas = self.nbas
        var npairs = nbas * (nbas + 1) // 2
        var nacc = nj + nk
        if nacc == 0:
            return
        if max(nj, nk) >= JK_LANES_MIN:
            self._jk_lanes(nj, dmj, vj, nk, dmk, vk, tol, nanti)
            return
        var cond = self._cond(nj, dmj, nk, dmk)
        var pcond = list_ptr(cond)
        var nworkers = min(parallelism_level(), npairs) if npairs >= 16 else 1
        var acc = List[Float64](length=nworkers * nacc * n2, fill=0.0)
        var pacc = list_ptr(acc)
        var pq = list_ptr(self.q)
        var counter = Atomic[Int64](0)
        var pcount = Pointer(to=counter)
        var nfmax = self.nfmax

        var attenuated = self.omega > 0.0

        def work(w: Int) {imm self, imm pacc, imm pcond, imm pq, imm pcount, imm npairs, imm nbas, imm nao, imm n2, imm nacc, imm nj, imm nk, imm dmj, imm dmk, imm tol, imm nfmax, imm attenuated}:
            var ws = EriWork(self.tab.maxcomp, self.tab.maxlab, self.tab.maxcomp, self.tab.maxlab)
            var wb = EriBatch(self.tab.maxcomp, self.tab.maxcomp)
            var loc = List[Float64](length=6 * nfmax * nfmax + 8, fill=0.0)
            var ploc = list_ptr(loc)
            var aj = pacc.unsafe_offset(w * nacc * n2)
            var ak = aj.unsafe_offset(nj * n2)
            var queue = KetQueue(self.nclass + NCLASS1)    # kets of the current bra waiting for a batch
            var lsl = List[Float64](length=lanes1_scratch(nfmax, max(nj, nk)) + W, fill=0.0)
            var lscr = F64Ptr(unsafe_from_address=aligned_addr(lsl))
            var qb = QuartetBuffer(0)
            while True:
                var task = Int(pcount[].fetch_add(1))
                if task >= npairs:
                    break
                var sp = self.order[task]
                if attenuated:
                    _jk_bra[False](self, self.tabw, sp, ws, wb, queue, ploc, pcond, pq, npairs, nbas, nao, n2, nj, nk, dmj, dmk, aj, ak, tol, lscr, qb)
                else:
                    _jk_bra[False](self, self.tab, sp, ws, wb, queue, ploc, pcond, pq, npairs, nbas, nao, n2, nj, nk, dmj, dmk, aj, ak, tol, lscr, qb)
            _ = wb^
            _ = queue^
            _ = ws^
            _ = loc^
            _ = lsl^
            _ = qb^

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
        # J = A + A^T, K = A + A^T (A - A^T for the antisymmetric densities) (tiled transpose)
        comptime TB = 32
        for s in range(nacc):
            var src = pacc.unsafe_offset(s * n2)
            var dst = vj.unsafe_offset(s * n2) if s < nj else vk.unsafe_offset((s - nj) * n2)
            var sgn = -1.0 if s >= nacc - nanti else 1.0
            var i0 = 0
            while i0 < nao:
                var i1 = min(i0 + TB, nao)
                var j0 = 0
                while j0 < nao:
                    var j1 = min(j0 + TB, nao)
                    for i in range(i0, i1):
                        for j in range(j0, j1):
                            dst[unsafe_offset=i * nao + j] = src[unsafe_offset=i * nao + j] + sgn * src[unsafe_offset=j * nao + i]
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
    with_j: Bool, with_k: Bool, tol: Float64, omega: Float64 = 0.0,
):
    """J1 = sum_kl (nabla i j|kl) D_lk and K1 = sum_jk (nabla i j|kl) D_jk, each (nset, 3, nao, nao).

    These are pyscf's ``grad.rhf.get_jk`` matrices up to the overall sign
    (pyscf returns -J1, -K1).  Bra pairs are the ordered pairs (nabla a, b),
    ket pairs c >= d; a quartet is skipped when
    q'_ab q_cd max(2 D_cd, D_bc, D_bd) < tol with q' the Schwarz bound of the
    derivative pair and D the largest density element per shell block.
    ``omega`` > 0: the long-range operator erf(omega r12) / r12 (the ket
    table attenuated after its Schwarz bounds are taken).
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
    if omega > 0.0:
        attenuate(tab, omega)
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


comptime H1_NBLK = 9    # derivative blocks of one quartet folded together: three atoms besides the remaining
                        # one, three directions each


def digest_h1[DO_J: Bool, DO_K: Bool, NC: Int, ND: Int](
    nblk: Int, bp: IntPtr, bst: IntPtr, aj: IntPtr, ak: IntPtr,
    na: Int, nb: Int, i0: Int, j0: Int, k0: Int, l0: Int, nao: Int, scale: Float64, dj: F64Ptr, dk: F64Ptr,
):
    """``digest_cd`` for all derivative blocks of one quartet in one pass over its functions.

    Block t = 3 g + x (atom g, direction x; ``nblk`` = 3 per atom) starts at
    address bp[t], with rows ij of bst[t] values and the ket's NC x ND values
    contiguous in each, and is folded into the matrices at addresses aj[t],
    ak[t].  The blocks of all atoms of a quartet sum to zero (translational
    invariance), so the remaining atom's matrices, at aj[nblk + x] and
    ak[nblk + x], receive minus the sums over g: its block is never formed
    or contracted.  The density elements are read once for all blocks.
    """
    comptime NCD = NC * ND
    var ng = nblk // 3
    var s2 = 2.0 * scale
    var dcd = stack_allocation[NCD, Float64]()            # D_cd (J)
    var jcd = stack_allocation[H1_NBLK * NCD, Float64]()  # per block: 2 s sum_ij (ij|cd) D_ij (J)
    var dad = stack_allocation[ND, Float64]()             # D_il, D_ik of the current bra row i (K)
    var dac = stack_allocation[NC, Float64]()
    var dbd = stack_allocation[ND, Float64]()             # D_jl, D_jk of the current j (K)
    var dbc = stack_allocation[NC, Float64]()
    var kad = stack_allocation[H1_NBLK * ND, Float64]()   # per block: K_il over j, k; K_ik over j, l (row i)
    var kac = stack_allocation[H1_NBLK * NC, Float64]()
    var kbd = stack_allocation[ND, Float64]()             # K_jl over k (one block, one (i, j))
    var ebc = stack_allocation[NC, Float64]()             # the remaining atom's K_jk, K_jl: sums over g
    var ebd = stack_allocation[ND, Float64]()
    comptime if DO_J:
        comptime for k in range(NC):
            comptime for l in range(ND):
                dcd[unsafe_offset=k * ND + l] = dj[unsafe_offset=(k0 + k) * nao + l0 + l]
        for t in range(nblk * NCD):
            jcd[unsafe_offset=t] = 0.0
    for i in range(na):
        var ri = (i0 + i) * nao
        comptime if DO_K:
            comptime for l in range(ND):
                dad[unsafe_offset=l] = dk[unsafe_offset=ri + l0 + l]
            comptime for k in range(NC):
                dac[unsafe_offset=k] = dk[unsafe_offset=ri + k0 + k]
            for t in range(nblk * ND):
                kad[unsafe_offset=t] = 0.0
            for t in range(nblk * NC):
                kac[unsafe_offset=t] = 0.0
        for j in range(nb):
            var rj = (j0 + j) * nao
            var ij = i * nb + j
            var dij = 0.0
            comptime if DO_J:
                dij = s2 * dj[unsafe_offset=ri + j0 + j]
            comptime if DO_K:
                comptime for l in range(ND):
                    dbd[unsafe_offset=l] = dk[unsafe_offset=rj + l0 + l]
                comptime for k in range(NC):
                    dbc[unsafe_offset=k] = dk[unsafe_offset=rj + k0 + k]
            for x in range(3):
                var esj = 0.0
                comptime if DO_K:
                    comptime for k in range(NC):
                        ebc[unsafe_offset=k] = 0.0
                    comptime for l in range(ND):
                        ebd[unsafe_offset=l] = 0.0
                for g in range(ng):
                    var t = 3 * g + x
                    var gv = F64Ptr(unsafe_from_address=bp[unsafe_offset=t]).unsafe_offset(ij * bst[unsafe_offset=t])
                    var pj = F64Ptr(unsafe_from_address=aj[unsafe_offset=t])
                    var pk = F64Ptr(unsafe_from_address=ak[unsafe_offset=t])
                    var jc = jcd.unsafe_offset(t * NCD)
                    var kd = kad.unsafe_offset(t * ND)
                    var kc = kac.unsafe_offset(t * NC)
                    var sj = 0.0
                    comptime if DO_K:
                        comptime for l in range(ND):
                            kbd[unsafe_offset=l] = 0.0
                    comptime for k in range(NC):
                        var sac = 0.0
                        var sbc = 0.0
                        comptime for l in range(ND):
                            var v = gv[unsafe_offset=k * ND + l]
                            comptime if DO_J:
                                sj += v * dcd[unsafe_offset=k * ND + l]
                                jc[unsafe_offset=k * ND + l] += v * dij
                            comptime if DO_K:
                                sac += v * dbd[unsafe_offset=l]
                                sbc += v * dad[unsafe_offset=l]
                                kd[unsafe_offset=l] += v * dbc[unsafe_offset=k]
                                kbd[unsafe_offset=l] += v * dac[unsafe_offset=k]
                        comptime if DO_K:
                            kc[unsafe_offset=k] += sac
                            pk[unsafe_offset=rj + k0 + k] += scale * sbc
                            ebc[unsafe_offset=k] += sbc
                    comptime if DO_J:
                        pj[unsafe_offset=ri + j0 + j] += s2 * sj
                        esj += sj
                    comptime if DO_K:
                        comptime for l in range(ND):
                            pk[unsafe_offset=rj + l0 + l] += scale * kbd[unsafe_offset=l]
                            ebd[unsafe_offset=l] += kbd[unsafe_offset=l]
                comptime if DO_J:
                    var qj = F64Ptr(unsafe_from_address=aj[unsafe_offset=nblk + x])
                    qj[unsafe_offset=ri + j0 + j] -= s2 * esj
                comptime if DO_K:
                    var qk = F64Ptr(unsafe_from_address=ak[unsafe_offset=nblk + x])
                    comptime for k in range(NC):
                        qk[unsafe_offset=rj + k0 + k] -= scale * ebc[unsafe_offset=k]
                    comptime for l in range(ND):
                        qk[unsafe_offset=rj + l0 + l] -= scale * ebd[unsafe_offset=l]
        comptime if DO_K:
            for x in range(3):
                var qk = F64Ptr(unsafe_from_address=ak[unsafe_offset=nblk + x])
                for g in range(ng):
                    var t = 3 * g + x
                    var pk = F64Ptr(unsafe_from_address=ak[unsafe_offset=t])
                    comptime for k in range(NC):
                        var v = scale * kac[unsafe_offset=t * NC + k]
                        pk[unsafe_offset=ri + k0 + k] += v
                        qk[unsafe_offset=ri + k0 + k] -= v
                    comptime for l in range(ND):
                        var v = scale * kad[unsafe_offset=t * ND + l]
                        pk[unsafe_offset=ri + l0 + l] += v
                        qk[unsafe_offset=ri + l0 + l] -= v
    comptime if DO_J:
        for x in range(3):
            var qj = F64Ptr(unsafe_from_address=aj[unsafe_offset=nblk + x])
            for g in range(ng):
                var t = 3 * g + x
                var pj = F64Ptr(unsafe_from_address=aj[unsafe_offset=t])
                comptime for k in range(NC):
                    comptime for l in range(ND):
                        var v = jcd[unsafe_offset=t * NCD + k * ND + l]
                        pj[unsafe_offset=(k0 + k) * nao + l0 + l] += v
                        qj[unsafe_offset=(k0 + k) * nao + l0 + l] -= v


def _h1_digest[DO_J: Bool, DO_K: Bool](
    nblk: Int, bp: IntPtr, bst: IntPtr, aj: IntPtr, ak: IntPtr, na: Int, nb: Int, nc: Int, nd: Int,
    i0: Int, j0: Int, k0: Int, l0: Int, nao: Int, scale: Float64, dj: F64Ptr, dk: F64Ptr, fb: F64Ptr, ploc: F64Ptr,
):
    """``digest_h1`` for kets of s to f shells (spherical, one contraction); otherwise every block and the
    remaining atom's (formed in ``fb``, 2 nab ncd values) through ``digest_any``."""
    comptime for KC in range(4):
        comptime for KD in range(4):
            if nc == 2 * KC + 1 and nd == 2 * KD + 1:
                digest_h1[DO_J, DO_K, 2 * KC + 1, 2 * KD + 1](
                    nblk, bp, bst, aj, ak, na, nb, i0, j0, k0, l0, nao, scale, dj, dk
                )
                return
    var nab = na * nb
    var ncd = nc * nd
    var blk = fb
    var rest = fb.unsafe_offset(nab * ncd)
    for x in range(3):
        vfill(rest, nab * ncd, 0.0)
        for g in range(nblk // 3):
            var t = 3 * g + x
            var src = F64Ptr(unsafe_from_address=bp[unsafe_offset=t])
            var st = bst[unsafe_offset=t]
            for ij in range(nab):
                for kl in range(ncd):
                    var v = src[unsafe_offset=ij * st + kl]
                    blk[unsafe_offset=ij * ncd + kl] = v
                    rest[unsafe_offset=ij * ncd + kl] -= v
            digest_any[DO_J, DO_K](
                blk, na, nb, nc, nd, i0, j0, k0, l0, nao, scale, dj, F64Ptr(unsafe_from_address=aj[unsafe_offset=t]),
                dk, F64Ptr(unsafe_from_address=ak[unsafe_offset=t]), ploc,
            )
        digest_any[DO_J, DO_K](
            rest, na, nb, nc, nd, i0, j0, k0, l0, nao, scale, dj, F64Ptr(unsafe_from_address=aj[unsafe_offset=nblk + x]),
            dk, F64Ptr(unsafe_from_address=ak[unsafe_offset=nblk + x]), ploc,
        )


def _set_cen(cen: IntPtr, m: Int, atom: Int, addr: Int, xs: Int, st: Int):
    """Centre m of ``_h1_quartet``'s list: its atom, the address of its x block, the offset between directions
    and the stride of rows ij (values)."""
    cen[unsafe_offset=4 * m] = atom
    cen[unsafe_offset=4 * m + 1] = addr
    cen[unsafe_offset=4 * m + 2] = xs
    cen[unsafe_offset=4 * m + 3] = st


def _h1_quartet(
    basis: Basis, a: Int, b: Int, c: Int, d: Int, same_pair: Bool, ncen: Int, cen: IntPtr, der: Int, scr: F64Ptr,
    meta: IntPtr, nao: Int, n2: Int, nj: Int, nk: Int, dmj: F64Ptr, dmk: F64Ptr, acc: F64Ptr, fb: F64Ptr,
    ploc: F64Ptr,
):
    """Fold the nabla blocks of one quartet into the matrices of the displaced atoms (d/dR = -nabla).

    ``cen`` lists ``ncen`` blocks, each nabla of one centre (``_set_cen``);
    the centres not listed sit on atom ``der``, which receives minus the sum
    over the other atoms (``digest_h1``), so blocks of centres on ``der`` are
    not needed.  Blocks of one atom are summed in ``scr`` (3 x 3 nab ncd
    values).  ``acc``: matrices [A][x][J sets, then K sets] of nao x nao.
    """
    var i0 = basis.ao_loc[a]
    var na = basis.ao_loc[a + 1] - i0
    var j0 = basis.ao_loc[b]
    var nb = basis.ao_loc[b + 1] - j0
    var k0 = basis.ao_loc[c]
    var nc = basis.ao_loc[c + 1] - k0
    var l0 = basis.ao_loc[d]
    var nd = basis.ao_loc[d + 1] - l0
    var nab = na * nb
    var ncd = nc * nd
    var nabcd = nab * ncd
    var scale = -1.0
    if a == b:
        scale *= 0.5
    if c == d:
        scale *= 0.5
    if same_pair:
        scale *= 0.5
    # blocks 3 g + x of the atoms g besides ``der``: views of the integrals, or their sums in ``scr``
    var bp = meta
    var bst = meta.unsafe_offset(H1_NBLK)
    var aj = meta.unsafe_offset(2 * H1_NBLK)
    var ak = meta.unsafe_offset(3 * H1_NBLK + 3)
    var gat = meta.unsafe_offset(4 * H1_NBLK + 6)
    var gsum = meta.unsafe_offset(4 * H1_NBLK + 9)
    var ng = 0
    for m in range(ncen):
        var at = cen[unsafe_offset=4 * m]
        if at == der:
            continue
        var base = cen[unsafe_offset=4 * m + 1]
        var xs = cen[unsafe_offset=4 * m + 2]
        var st = cen[unsafe_offset=4 * m + 3]
        var g = 0
        while g < ng and gat[unsafe_offset=g] != at:
            g += 1
        if g == ng:
            gat[unsafe_offset=g] = at
            gsum[unsafe_offset=g] = 0
            for x in range(3):
                bp[unsafe_offset=3 * g + x] = base + 8 * x * xs
                bst[unsafe_offset=3 * g + x] = st
            ng += 1
            continue
        for x in range(3):
            var s = scr.unsafe_offset((3 * g + x) * nabcd)
            var add = F64Ptr(unsafe_from_address=base + 8 * x * xs)
            if gsum[unsafe_offset=g] == 0:
                var src = F64Ptr(unsafe_from_address=bp[unsafe_offset=3 * g + x])
                var sst = bst[unsafe_offset=3 * g + x]
                for ij in range(nab):
                    for kl in range(ncd):
                        s[unsafe_offset=ij * ncd + kl] = src[unsafe_offset=ij * sst + kl] + add[unsafe_offset=ij * st + kl]
                bp[unsafe_offset=3 * g + x] = Int(s)
                bst[unsafe_offset=3 * g + x] = ncd
            else:
                for ij in range(nab):
                    for kl in range(ncd):
                        s[unsafe_offset=ij * ncd + kl] += add[unsafe_offset=ij * st + kl]
        gsum[unsafe_offset=g] = 1
    if ng == 0:
        return
    var nblk = 3 * ng
    var nacc = nj + nk
    # density s: J and K together while both have one, then the rest of either
    for s in range(max(nj, nk)):
        var sj = min(s, max(nj - 1, 0))
        var sk = min(s, max(nk - 1, 0))
        for t in range(nblk + 3):
            var at = gat[unsafe_offset=t // 3] if t < nblk else der
            var m0 = (at * 3 + t % 3) * nacc
            aj[unsafe_offset=t] = Int(acc.unsafe_offset((m0 + sj) * n2))
            ak[unsafe_offset=t] = Int(acc.unsafe_offset((m0 + nj + sk) * n2))
        var dj = dmj.unsafe_offset(sj * n2)
        var dk = dmk.unsafe_offset(sk * n2)
        if s < nj and s < nk:
            _h1_digest[True, True](nblk, bp, bst, aj, ak, na, nb, nc, nd, i0, j0, k0, l0, nao, scale, dj, dk, fb, ploc)
        elif s < nj:
            _h1_digest[True, False](nblk, bp, bst, aj, ak, na, nb, nc, nd, i0, j0, k0, l0, nao, scale, dj, dk, fb, ploc)
        else:
            _h1_digest[False, True](nblk, bp, bst, aj, ak, na, nb, nc, nd, i0, j0, k0, l0, nao, scale, dj, dk, fb, ploc)


def _h1_flush(
    basis: Basis, tab: PairTable, tab3: PairTable, tab6: PairTable, sp: Int, kets: IntPtr, nket: Int,
    boys: BoysTable, ws: EriWork, mut wb1: EriBatch, mut wb2: EriBatch, cen: IntPtr, scr: F64Ptr, meta: IntPtr,
    nao: Int, n2: Int, nj: Int, nk: Int, dmj: F64Ptr, dmk: F64Ptr, acc: F64Ptr, fb: F64Ptr, ploc: F64Ptr,
):
    """``h1_2e_core`` for bra ``sp`` and queued kets through ``eri_batch`` (the kets' primitive pairs as SIMD
    lanes): (nabla a b|cd) unless a and b share an atom, (ab|nabla c d) and (ab|c nabla d); b from translational
    invariance."""
    var a = tab.get(sp, I_A)
    var b = tab.get(sp, I_B)
    var nab = tab.get(sp, I_NCOMP)
    var with_ab = basis.atom[a] != basis.atom[b]
    var ncd = tab.get(kets[unsafe_offset=0], I_NCOMP)
    if with_ab:
        eri_batch(tab3, sp, tab, kets, nket, boys, ws, wb1)
    eri_batch(tab, sp, tab6, kets, nket, boys, ws, wb2)
    for k in range(nket):
        var spk = kets[unsafe_offset=k]
        var c = tab.get(spk, I_A)
        var d = tab.get(spk, I_B)
        var n = 0
        if with_ab:
            _set_cen(cen, 0, basis.atom[a], Int(wb1.block(k)), nab * ncd, ncd)
            n = 1
        var kb = Int(wb2.block(k))
        _set_cen(cen, n, basis.atom[c], kb, ncd, 6 * ncd)
        _set_cen(cen, n + 1, basis.atom[d], kb + 8 * 3 * ncd, ncd, 6 * ncd)
        _h1_quartet(
            basis, a, b, c, d, sp == spk, n + 2, cen, basis.atom[b], scr, meta, nao, n2, nj, nk, dmj, dmk, acc, fb,
            ploc,
        )


def h1_2e_core(
    var basis: Basis, var boys: BoysTable, nj: Int, dmj: F64Ptr, nk: Int, dmk: F64Ptr, tol: Float64, dst: F64Ptr,
    omega: Float64 = 0.0,
):
    """First-order Coulomb and exchange matrices of every nuclear displacement, at fixed densities.

    dst[A][x][s] (natm x 3 x (nj + nk) x nao x nao, overwritten) is
    d/dR_Ax J[Dj_s] for s < nj and d/dR_Ax K[Dk_(s - nj)] otherwise, with the
    integrals differentiated at all four centres: the two-electron part of
    the Hessian's first-order Fock matrices (pyscf's ``make_h1``).  The
    densities must be symmetric.  Each unique quartet a >= b, c >= d,
    ab >= cd is evaluated once, by the task of its pair with more primitive
    pairs (the bra), with the derivative integrals of ``grad2e_core``'s
    driver: (nabla a b|cd) and the six-component pair (nabla c, nabla d) of
    the ket, or the mirror split (nabla a, nabla b on the bra and nabla c)
    when the ket is the higher pair, a single pass when either pair sits on
    one atom (only the sum of its two derivatives is needed), and the bra's
    kets of degree <= LAB_BATCH as SIMD lanes of ``eri_batch``.  The
    derivative of the last centre follows from translational invariance:
    the blocks are summed per atom, and the atom of the missing centre
    receives minus the others' contractions (``digest_h1``: the derivative
    has the integral's eight-fold symmetry, and all blocks of a quartet are
    folded in one pass over its functions).  A quartet is skipped when
    max(q'_ab q_cd, q_ab q'_cd) times the largest density element it meets
    (as in the direct J/K) is below ``tol``, q' the Schwarz bound of the
    six-component pair, or when its four centres sit on one atom.
    ``omega`` > 0: the long-range operator erf(omega r12) / r12 (the plain
    pair table, one side of every pass, attenuated after the Schwarz bounds
    are taken).
    """
    var nbas = basis.nbas
    var nao = basis.nao
    var natm = basis.natm
    var n2 = nao * nao
    var npairs = nbas * (nbas + 1) // 2
    var nacc = nj + nk
    var ht = HermTable(2 * basis.lmax + 1)
    var sa = List[Int](capacity=npairs)
    var sb = List[Int](capacity=npairs)
    for a in range(nbas):
        for b in range(a + 1):
            sa.append(a)
            sb.append(b)
    var tab = PairTable(basis, basis, sa, sb, ht)
    var q = schwarz_bounds(boys, tab, ht)
    var tab6 = PairTable(basis, basis, sa, sb, ht, 2)
    var q6 = schwarz_bounds(boys, tab6, ht)
    var tab3 = PairTable(basis, basis, sa, sb, ht, 1)
    if omega > 0.0:
        attenuate(tab, omega)
    # largest density element per shell block, over all densities
    var cond = List[Float64](length=max(nbas * nbas, 1), fill=0.0)
    var pcond = list_ptr(cond)
    for a in range(nbas):
        for b in range(nbas):
            var m = 0.0
            for s in range(nacc):
                var d = dmj.unsafe_offset(s * n2) if s < nj else dmk.unsafe_offset((s - nj) * n2)
                for i in range(basis.ao_loc[a], basis.ao_loc[a + 1]):
                    for j in range(basis.ao_loc[b], basis.ao_loc[b + 1]):
                        m = max(m, abs(d[unsafe_offset=i * nao + j]))
            pcond[unsafe_offset=a * nbas + b] = m
    var nfmax = 1
    for a in range(nbas):
        nfmax = max(nfmax, shell_nfunc(basis, a))
    # ket classes of the lane batches, as in grad2e_core
    var kcls = List[Int](length=max(npairs, 1), fill=-1)
    var nclass = batch_classes(tab, kcls)
    var mck = 1     # largest ket of a batch
    var mcb = 1     # largest bra of a batch
    for sp in range(npairs):
        var lab = tab.get(sp, I_LAB)
        if lab > LAB_BATCH:
            kcls[sp] = -1
        else:
            mck = max(mck, tab.get(sp, I_NCOMP))
        if lab <= LAB_BATCH + 1:
            mcb = max(mcb, tab.get(sp, I_NCOMP))
    var order = bras_by_nprim(tab)
    var nworkers = min(parallelism_level(), npairs) if npairs >= 16 else 1
    var per = natm * 3 * nacc * n2
    var accl = List[Float64](length=max(nworkers * per, 1), fill=0.0)
    var pacc = list_ptr(accl)
    var pq = list_ptr(q)
    var pq6 = list_ptr(q6)
    var pkcls = int_ptr(kcls)
    var porder = int_ptr(order)
    var counter = Atomic[Int64](0)
    var pcount = Pointer(to=counter)

    def work(w: Int) {imm basis, imm boys, imm ht, imm tab, imm tab6, imm tab3, imm pq, imm pq6, imm pcond, imm pacc, imm pcount, imm npairs, imm nbas, imm nao, imm n2, imm nj, imm nk, imm dmj, imm dmk, imm tol, imm nfmax, imm per, imm pkcls, imm porder, imm nclass, imm mck, imm mcb}:
        var ws = EriWork(tab6.maxcomp, tab6.maxlab, tab6.maxcomp, tab6.maxlab)
        var wb1 = EriBatch(3 * mcb, mck)
        var wb2 = EriBatch(mcb, 6 * mck)
        var queue = KetQueue(nclass)
        var nf4 = nfmax * nfmax * nfmax * nfmax
        var dbuf = List[Float64](length=6 * nf4 + 8, fill=0.0)     # the first pass of a quartet
        var pd = list_ptr(dbuf)
        var sbuf = List[Float64](length=11 * nf4 + 8, fill=0.0)    # atom sums (9 blocks), fallback (2)
        var scr = list_ptr(sbuf)
        var fb = scr.unsafe_offset(9 * nf4)
        var ibuf = List[Int](length=16 + 4 * H1_NBLK + 12, fill=0)
        var cen = int_ptr(ibuf)
        var meta = cen.unsafe_offset(16)
        var loc = List[Float64](length=6 * nfmax * nfmax + 8, fill=0.0)
        var ploc = list_ptr(loc)
        var acc = pacc.unsafe_offset(w * per)
        while True:
            var task = Int(pcount[].fetch_add(1))
            if task >= npairs:
                break
            var sp = porder[unsafe_offset=task]
            var qab = pq[unsafe_offset=sp]
            if qab == 0.0:
                continue
            var a = tab.get(sp, I_A)
            var b = tab.get(sp, I_B)
            var aa = basis.atom[a]
            var ab = basis.atom[b]
            var nab = tab.get(sp, I_NCOMP)
            var npb = tab.get(sp, I_NP)
            var lab = tab.get(sp, I_LAB)
            var with_ab = aa != ab
            var batch_bra = lab <= LAB_BATCH or (lab == LAB_BATCH + 1 and not with_ab)
            var dab = pcond[unsafe_offset=a * nbas + b]
            for spk in range(npairs):
                var npk = tab.get(spk, I_NP)
                if npk > npb or (npk == npb and spk > sp):
                    continue
                var qq = max(pq6[unsafe_offset=sp] * pq[unsafe_offset=spk], qab * pq6[unsafe_offset=spk])
                if qq < tol:
                    continue
                var c = tab.get(spk, I_A)
                var d = tab.get(spk, I_B)
                var ac = basis.atom[c]
                var ad = basis.atom[d]
                if not with_ab and ac == aa and ad == aa:
                    continue
                var dmax = 4.0 * max(dab, pcond[unsafe_offset=c * nbas + d])
                dmax = max(dmax, max(pcond[unsafe_offset=a * nbas + c], pcond[unsafe_offset=a * nbas + d]))
                dmax = max(dmax, max(pcond[unsafe_offset=b * nbas + c], pcond[unsafe_offset=b * nbas + d]))
                if qq * dmax < tol:
                    continue
                var kc = pkcls[unsafe_offset=spk] if batch_bra else -1
                if kc >= 0 and lanes_preferred(tab, sp, tab, spk):
                    if queue.full(kc, npk):
                        _h1_flush(
                            basis, tab, tab3, tab6, sp, queue.kets(kc), queue.cnt[kc], boys, ws, wb1, wb2, cen, scr,
                            meta, nao, n2, nj, nk, dmj, dmk, acc, fb, ploc,
                        )
                        queue.clear(kc)
                    queue.push(kc, spk, npk)
                    continue
                var with_cd = ac != ad
                var lcd = tab.get(spk, I_LAB)
                var ncd = tab.get(spk, I_NCOMP)
                var nabcd = nab * ncd
                var ncen = 2
                var der = aa
                if with_ab and with_cd:
                    if lcd > lab:
                        # (nabla a, nabla b|cd) [6][ij][kl], then (ab|nabla c d) [ij][3][kl]; d by invariance
                        if not eri_quartet(tab6, sp, tab, spk, ht, boys, ws):
                            continue
                        vcopy(pd, list_ptr(ws.out), 6 * nabcd)
                        _ = eri_quartet(tab, sp, tab3, spk, ht, boys, ws)
                        _set_cen(cen, 0, aa, Int(pd), nabcd, ncd)
                        _set_cen(cen, 1, ab, Int(pd) + 8 * 3 * nabcd, nabcd, ncd)
                        _set_cen(cen, 2, ac, Int(list_ptr(ws.out)), ncd, 3 * ncd)
                        der = ad
                    else:
                        # (nabla a b|cd) [3][ij][kl], then (ab|nabla c d), (ab|c nabla d) [ij][6][kl]; b by invariance
                        if not eri_quartet(tab3, sp, tab, spk, ht, boys, ws):
                            continue
                        vcopy(pd, list_ptr(ws.out), 3 * nabcd)
                        _ = eri_quartet(tab, sp, tab6, spk, ht, boys, ws)
                        var o = Int(list_ptr(ws.out))
                        _set_cen(cen, 0, aa, Int(pd), nabcd, ncd)
                        _set_cen(cen, 1, ac, o, ncd, 6 * ncd)
                        _set_cen(cen, 2, ad, o + 8 * 3 * ncd, ncd, 6 * ncd)
                        der = ab
                    ncen = 3
                elif with_ab or (not with_cd and lcd > lab):
                    # the bra pass alone: c and d share an atom
                    if not eri_quartet(tab6, sp, tab, spk, ht, boys, ws):
                        continue
                    var o = Int(list_ptr(ws.out))
                    _set_cen(cen, 0, aa, o, nabcd, ncd)
                    _set_cen(cen, 1, ab, o + 8 * 3 * nabcd, nabcd, ncd)
                    der = ac
                else:
                    # the ket pass alone: a and b share an atom
                    if not eri_quartet(tab, sp, tab6, spk, ht, boys, ws):
                        continue
                    var o = Int(list_ptr(ws.out))
                    _set_cen(cen, 0, ac, o, ncd, 6 * ncd)
                    _set_cen(cen, 1, ad, o + 8 * 3 * ncd, ncd, 6 * ncd)
                _h1_quartet(
                    basis, a, b, c, d, sp == spk, ncen, cen, der, scr, meta, nao, n2, nj, nk, dmj, dmk, acc, fb, ploc
                )
            # the rest of this bra's batches
            for kc in range(nclass):
                if queue.cnt[kc] > 0:
                    _h1_flush(
                        basis, tab, tab3, tab6, sp, queue.kets(kc), queue.cnt[kc], boys, ws, wb1, wb2, cen, scr, meta,
                        nao, n2, nj, nk, dmj, dmk, acc, fb, ploc,
                    )
                    queue.clear(kc)
        _ = ws^
        _ = wb1^
        _ = wb2^
        _ = queue^
        _ = dbuf^
        _ = sbuf^
        _ = ibuf^
        _ = loc^

    if nworkers == 1:
        work(0)
    else:
        parallelize(work, nworkers)
    var total = per

    def reduce(cidx: Int) {imm pacc, imm total, imm nworkers}:
        var nchunk = 64
        var lo = total * cidx // nchunk
        var hi = total * (cidx + 1) // nchunk
        for w2 in range(1, nworkers):
            vaxpy(pacc.unsafe_offset(lo), hi - lo, 1.0, pacc.unsafe_offset(w2 * total + lo))

    if nworkers > 1:
        parallelize(reduce, 64)
    # each matrix: A + A^T (the digestion's half of the symmetric result)
    for m in range(natm * 3 * nacc):
        var src = pacc.unsafe_offset(m * n2)
        var d = dst.unsafe_offset(m * n2)
        for i in range(nao):
            for j in range(nao):
                d[unsafe_offset=i * nao + j] = src[unsafe_offset=i * nao + j] + src[unsafe_offset=j * nao + i]
    _ = accl^
    _ = cond^
    _ = kcls^
    _ = order^
    _ = counter^
    _ = tab^
    _ = tab6^
    _ = tab3^
    _ = q^
    _ = q6^
    _ = sa^
    _ = sb^
    _ = ht^
    _ = basis^
    _ = boys^
