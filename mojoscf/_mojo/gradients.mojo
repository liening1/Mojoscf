"""Nuclear-gradient contractions of derivative integrals from the Mojo engine.

All routines return the derivative of an energy with respect to the nuclear
coordinates, ``de[A][x]`` (natm x 3), at fixed densities: the integral
derivatives are contracted with the density factors as they are produced, so
no derivative integrals or derivative matrices are stored.

* ``grad2e_core``: exact two-electron term, ``1/2 sum (ij|kl) G_ijkl``.
* ``grad_df3c_core`` and ``grad2c_core``: the two terms of the density-fitted
  two-electron energy, ``sum (mu nu|P) Gamma_P,mu nu`` and
  ``-1/2 sum (P|Q) W_PQ`` (their derivatives give the DF gradient, including
  the response of the auxiliary basis); ``df_grad_rhs`` prepares the fit.

The derivative pair tables (``fill_pair`` with ``nderiv``) give
``nabla = -d/dR`` of the differentiated functions; the derivative with
respect to the remaining centre follows from translational invariance where
noted.
"""
from std.atomic import Atomic
from std.math import sqrt
from std.memory import Pointer
from std.runtime import parallelism_level
from max.algorithm import parallelize
from _mojo.linalg import F64Ptr, Blas, list_ptr, vfill, vdot_serial
from _mojo.dfjk import unpack_row
from _mojo.integrals import (
    Basis, BoysTable, HermTable, PairTable, EriWork, aux_table, eri_quartet, int3c2e_core, schwarz_bounds,
    shell_nfunc, I_A, I_B,
)


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



def df_grad_rhs(
    blas_seq: Blas, var basis: Basis, var aux: Basis, var boys: BoysTable, dm_tril: F64Ptr, nset: Int, m: Int,
    orbs: F64Ptr, blk: Int, rho: F64Ptr, q: F64Ptr,
) raises:
    """rho_P = sum (P|mu nu) D_mu nu and Q_s,P = C_s^T (P|..) C_s for the fit of the DF gradient.

    ``dm_tril`` is pyscf's packed density (off-diagonal elements doubled),
    ``orbs`` holds ``nset`` (nao x m) orbital blocks (zero columns allowed),
    ``q`` receives the packed lower triangles (nset, naux, m (m + 1) / 2) of
    the symmetric Q_s,P.  The three-centre integrals are
    evaluated for blocks of about ``blk`` auxiliary functions; within a block
    worker threads unpack one (P|mu nu) at a time and transform it with the
    sequential BLAS (two small GEMMs per set), as the DF exchange build does.
    """
    var nao = basis.nao
    var naux = aux.nao
    var npair = nao * (nao + 1) // 2
    var n2 = nao * nao
    var mp = m * (m + 1) // 2
    var nwork = 2 * max(1, parallelism_level())
    var wsz = n2 + m * nao + m * m
    var ebuf = List[Float64](length=nwork * wsz + 8, fill=0.0)
    var pe = list_ptr(ebuf)
    var s0 = 0
    while s0 < aux.nbas:
        var s1 = s0 + 1
        while s1 < aux.nbas and aux.ao_loc[s1 + 1] - aux.ao_loc[s0] <= blk:
            s1 += 1
        var p0 = aux.ao_loc[s0]
        var p1 = aux.ao_loc[s1]
        var b = List[Float64](length=(p1 - p0) * npair, fill=0.0)
        var pb = list_ptr(b)
        int3c2e_core(basis, aux, boys, pb, s0, s1)

        def work(c: Int) {imm blas_seq, imm pb, imm pe, imm dm_tril, imm orbs, imm rho, imm q, imm nao, imm naux, imm npair, imm n2, imm nset, imm m, imm mp, imm wsz, imm p0, imm p1, imm nwork}:
            var e = pe.unsafe_offset(c * wsz)
            var u = e.unsafe_offset(n2)
            var qm = u.unsafe_offset(m * nao)
            var pp = p0 + c
            while pp < p1:
                var row = pb.unsafe_offset((pp - p0) * npair)
                rho[unsafe_offset=pp] = vdot_serial(row, dm_tril, npair)
                if m > 0:
                    unpack_row(row, nao, e)
                    for st in range(nset):
                        var cs = orbs.unsafe_offset(st * nao * m)
                        try:
                            # U (m x nao) = C^T E, Q_P (m x m) = U C
                            blas_seq.gemm(True, False, m, nao, nao, 1.0, cs, e, 0.0, u)
                            blas_seq.gemm(False, False, m, m, nao, 1.0, u, cs, 0.0, qm)
                        except:
                            pass
                        var qp = q.unsafe_offset((st * naux + pp) * mp)
                        var k = 0
                        for i in range(m):
                            for j in range(i + 1):
                                qp[unsafe_offset=k + j] = qm[unsafe_offset=i * m + j]
                            k += i + 1
                pp += nwork

        parallelize(work, nwork)
        _ = b^
        s0 = s1
    _ = ebuf^
    _ = basis^
    _ = aux^
    _ = boys^


def grad_df3c_core(
    blas_seq: Blas, var basis: Basis, var aux: Basis, var boys: BoysTable, coef: F64Ptr, dpack: F64Ptr,
    jfac: Float64, kfac: Float64, nset: Int, m: Int, xs: F64Ptr, cns: F64Ptr, blk: Int, tol: Float64, de: F64Ptr,
) raises:
    """de[A][x] = d/dR_Ax sum_{P, mu nu} (mu nu|P) Gamma_P,mu nu, overwritten (natm x 3).

    Gamma_P = jfac coef_P D - kfac sum_s Cn_s X_s,P Cn_s^T, with ``dpack`` the
    packed lower triangle of D (not doubled), ``xs`` the packed lower
    triangles of the symmetric X_s,P (nset, naux, m (m + 1) / 2) and ``cns``
    (nset, nao x m).  Per block of about ``blk`` auxiliary functions
    the rows Gamma_P (packed, (np, npair)) are built by worker threads with a
    sequential GEMM and a rank-2k update per set; then each auxiliary shell is a task that
    runs over all AO shell pairs a >= b with the six-component derivative
    table (nabla a, nabla b) (weight 2 for a != b), the auxiliary centre
    taking minus their sum.  A triple is skipped when
    q'_ab q_P max|Gamma_P,ab| < tol.  The pair table and the Schwarz bounds
    are built once for all blocks.
    """
    var nbas = basis.nbas
    var natm = basis.natm
    var nao = basis.nao
    var n2 = nao * nao
    var npair = nao * (nao + 1) // 2
    var npairs = nbas * (nbas + 1) // 2
    var ht = HermTable(max(2 * basis.lmax + 1, aux.lmax))
    var sa = List[Int](capacity=npairs)
    var sb = List[Int](capacity=npairs)
    for a in range(nbas):
        for b in range(a + 1):
            sa.append(a)
            sb.append(b)
    var tab2 = PairTable(basis, basis, sa, sb, ht, 2)
    var q2 = schwarz_bounds(boys, tab2, ht)
    var atab = aux_table(aux, ht)
    var qa = schwarz_bounds(boys, atab, ht)
    var nthreads = max(1, parallelism_level())
    var nwork = 2 * nthreads
    var per = 3 * natm
    var accl = List[Float64](length=nthreads * per + 1, fill=0.0)
    var pacc = list_ptr(accl)
    var mp = m * (m + 1) // 2
    var wsz = n2 + m * nao + m * m
    var fbuf = List[Float64](length=nwork * wsz + 8, fill=0.0)
    var pf = list_ptr(fbuf)
    var pq2 = list_ptr(q2)
    var pqa = list_ptr(qa)
    var counter = Atomic[Int64](0)
    var pcount = Pointer(to=counter)
    var s0 = 0
    while s0 < aux.nbas:
        var s1 = s0 + 1
        while s1 < aux.nbas and aux.ao_loc[s1 + 1] - aux.ao_loc[s0] <= blk:
            s1 += 1
        var p0 = aux.ao_loc[s0]
        var p1 = aux.ao_loc[s1]
        var gbuf = List[Float64](length=(p1 - p0) * npair + 1, fill=0.0)
        var gam = list_ptr(gbuf)

        def build(c: Int) {imm blas_seq, imm pf, imm gam, imm coef, imm dpack, imm jfac, imm kfac, imm xs, imm cns, imm nao, imm n2, imm npair, imm nset, imm m, imm mp, imm wsz, imm p0, imm p1, imm nwork, imm aux}:
            var f = pf.unsafe_offset(c * wsz)
            var t = f.unsafe_offset(n2)
            var xm = t.unsafe_offset(m * nao)
            var naux = aux.nao
            var pp = p0 + c
            while pp < p1:
                var g = gam.unsafe_offset((pp - p0) * npair)
                var cp = jfac * coef[unsafe_offset=pp]
                for k in range(npair):
                    g[unsafe_offset=k] = cp * dpack[unsafe_offset=k]
                if m > 0:
                    for st in range(nset):
                        var cn = cns.unsafe_offset(st * nao * m)
                        var xp = xs.unsafe_offset((st * naux + pp) * mp)
                        var k = 0
                        for i in range(m):
                            for j in range(i + 1):
                                var v = xp[unsafe_offset=k + j]
                                xm[unsafe_offset=i * m + j] = v
                                xm[unsafe_offset=j * m + i] = v
                            k += i + 1
                        try:
                            # T (nao x m) = Cn X_P; lower triangle of F (+)= -kfac Cn X_P Cn^T
                            # = -kfac/2 (T Cn^T + Cn T^T) (X_P is symmetric)
                            blas_seq.gemm(False, False, nao, m, m, 1.0, cn, xm, 0.0, t)
                            blas_seq.syr2k_lower(nao, m, -0.5 * kfac, t, cn, 0.0 if st == 0 else 1.0, f)
                        except:
                            pass
                    var k = 0
                    for i in range(nao):
                        var frow = f.unsafe_offset(i * nao)
                        for j in range(i + 1):
                            g[unsafe_offset=k + j] += frow[unsafe_offset=j]
                        k += i + 1
                pp += nwork

        parallelize(build, nwork)
        counter.store(0)
        var ntask = s1 - s0

        def work(w: Int) {imm basis, imm aux, imm boys, imm ht, imm tab2, imm atab, imm pq2, imm pqa, imm pacc, imm pcount, imm npairs, imm per, imm s0, imm ntask, imm p0, imm npair, imm gam, imm tol}:
            var ws = EriWork(tab2.maxcomp, tab2.maxlab, atab.maxcomp, atab.maxlab)
            var gabuf = List[Float64](length=8, fill=0.0)
            var ga = list_ptr(gabuf)
            var acc = pacc.unsafe_offset(w * per)
            while True:
                var task = Int(pcount[].fetch_add(1))
                if task >= ntask:
                    break
                var pshell = s0 + task
                var qp = pqa[unsafe_offset=pshell]
                if qp == 0.0:
                    continue
                var f0 = aux.ao_loc[pshell] - p0
                var npf = aux.ao_loc[pshell + 1] - aux.ao_loc[pshell]
                var grows = gam.unsafe_offset(f0 * npair)
                var pp = acc.unsafe_offset(3 * aux.atom[pshell])
                for sp in range(npairs):
                    var qab = pq2[unsafe_offset=sp] * qp
                    if qab < tol:
                        continue
                    var a = tab2.get(sp, I_A)
                    var b = tab2.get(sp, I_B)
                    var i0 = basis.ao_loc[a]
                    var na = basis.ao_loc[a + 1] - i0
                    var j0 = basis.ao_loc[b]
                    var nb = basis.ao_loc[b + 1] - j0
                    var nab = na * nb
                    var gmax = 0.0
                    for fp in range(npf):
                        var grow = grows.unsafe_offset(fp * npair)
                        for i in range(na):
                            for j in range(nb):
                                var ii = max(i0 + i, j0 + j)
                                var jj = min(i0 + i, j0 + j)
                                gmax = max(gmax, abs(grow[unsafe_offset=ii * (ii + 1) // 2 + jj]))
                    if qab * gmax < tol:
                        continue
                    if not eri_quartet(tab2, sp, atab, pshell, ht, boys, ws):
                        continue
                    var blk3 = list_ptr(ws.out)
                    vfill(ga, 6, 0.0)
                    for fp in range(npf):
                        var grow = grows.unsafe_offset(fp * npair)
                        for i in range(na):
                            for j in range(nb):
                                var ii = max(i0 + i, j0 + j)
                                var jj = min(i0 + i, j0 + j)
                                var gv = grow[unsafe_offset=ii * (ii + 1) // 2 + jj]
                                var ij = i * nb + j
                                for x in range(6):
                                    ga[unsafe_offset=x] += blk3[unsafe_offset=(x * nab + ij) * npf + fp] * gv
                    var wgt = 2.0 if a != b else 1.0
                    var pa = acc.unsafe_offset(3 * basis.atom[a])
                    var pb = acc.unsafe_offset(3 * basis.atom[b])
                    for x in range(3):
                        var gx = wgt * ga[unsafe_offset=x]
                        var gy = wgt * ga[unsafe_offset=3 + x]
                        pa[unsafe_offset=x] -= gx
                        pb[unsafe_offset=x] -= gy
                        pp[unsafe_offset=x] += gx + gy
            _ = ws^
            _ = gabuf^

        var nw = min(nthreads, ntask)
        if nw <= 1:
            work(0)
        else:
            parallelize(work, nw)
        _ = gbuf^
        s0 = s1
    for i in range(per):
        var v = 0.0
        for w2 in range(nthreads):
            v += pacc[unsafe_offset=w2 * per + i]
        de[unsafe_offset=i] = v
    _ = accl^
    _ = fbuf^
    _ = counter^
    _ = tab2^
    _ = atab^
    _ = q2^
    _ = qa^
    _ = sa^
    _ = sb^
    _ = ht^
    _ = basis^
    _ = aux^
    _ = boys^


def grad2c_core(var aux: Basis, var boys: BoysTable, wmat: F64Ptr, natm: Int, de: F64Ptr):
    """de[A][x] = d/dR_Ax of -1/2 sum_PQ (P|Q) W_PQ for the symmetric (naux, naux) matrix ``wmat``.

    Shell pairs P > Q are evaluated once with the derivative table of P
    against the plain one (weight 2); Q gets minus the derivative of P.
    ``de`` (natm x 3; the atoms of ``aux``) is overwritten.
    """
    var naux = aux.nao
    var nsh = aux.nbas
    var ht = HermTable(aux.lmax + 1)
    var atab = aux_table(aux, ht)
    var sa = List[Int](capacity=nsh)
    var sb = List[Int](capacity=nsh)
    for p in range(nsh):
        sa.append(p)
        sb.append(-1)
    var dtab = PairTable(aux, aux, sa, sb, ht, 1)
    var nworkers = min(parallelism_level(), nsh) if nsh >= 8 else 1
    var per = 3 * natm
    var accl = List[Float64](length=max(nworkers * per, 1), fill=0.0)
    var pacc = list_ptr(accl)
    var counter = Atomic[Int64](0)
    var pcount = Pointer(to=counter)

    def work(w: Int) {imm aux, imm boys, imm ht, imm atab, imm dtab, imm pacc, imm pcount, imm nsh, imm naux, imm per, imm wmat}:
        var ws = EriWork(dtab.maxcomp, dtab.maxlab, atab.maxcomp, atab.maxlab)
        var acc = pacc.unsafe_offset(w * per)
        while True:
            var task = Int(pcount[].fetch_add(1))
            if task >= nsh:
                break
            var ps = nsh - 1 - task
            var p0 = aux.ao_loc[ps]
            var npf = aux.ao_loc[ps + 1] - p0
            var pp = acc.unsafe_offset(3 * aux.atom[ps])
            for qs in range(ps):
                if aux.atom[qs] == aux.atom[ps]:
                    continue        # equal and opposite on the same atom
                if not eri_quartet(dtab, ps, atab, qs, ht, boys, ws):
                    continue
                var q0 = aux.ao_loc[qs]
                var nqf = aux.ao_loc[qs + 1] - q0
                var blk = list_ptr(ws.out)
                var pq = acc.unsafe_offset(3 * aux.atom[qs])
                for x in range(3):
                    var g = 0.0
                    for fp in range(npf):
                        g += vdot_serial(blk.unsafe_offset((x * npf + fp) * nqf), wmat.unsafe_offset((p0 + fp) * naux + q0), nqf)
                    # -1/2 * 2 (P>Q) * d/dR_P (P|Q) = +(nabla P|Q)
                    pp[unsafe_offset=x] += g
                    pq[unsafe_offset=x] -= g
        _ = ws^

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
    _ = counter^
    _ = atab^
    _ = dtab^
    _ = sa^
    _ = sb^
    _ = ht^
    _ = aux^
    _ = boys^
