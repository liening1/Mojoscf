"""Coulomb and exchange matrices from 8-fold symmetric packed two-electron integrals.

``eri`` is pyscf's ``aosym='s8'`` tensor: for AO pairs ``ij >= kl`` (pair index
``pair(i, j) = i (i + 1) / 2 + j``, ``i >= j``) the value ``(ij|kl)`` is stored at
``ij (ij + 1) / 2 + kl``.  Densities must be symmetric (pyscf ``hermi=1``).

Each unique integral stands for up to eight index permutations.  Rows ``ij`` are
distributed round-robin over work items, every item accumulates into private
buffers, and the results are reduced at the end:

* J: ``J'_ij += sum_kl (ij|kl) Dt_kl`` and ``J'_kl += (ij|kl) Dt_ij`` (kl < ij) with
  ``Dt`` the packed ``D + D^T`` (diagonal not doubled); J = unpack(J').
* K: for the tuples ``(ij|kl), (ij|lk), (ji|kl), (ji|lk)`` the updates
  ``K'_ik += v D_jl, K'_il += v D_jk, K'_jk += v D_il, K'_jl += v D_ik`` are made
  with ``v`` halved for each coincidence ``i == j``, ``k == l``, ``ij == kl``; the
  remaining four tuples are their transposes, so ``K = K' + K'^T``.
  For fixed ``(ij, k)`` the ``l`` loop is contiguous in memory: two dot products
  and two AXPYs over the integral row segment.
"""
from std.memory import Pointer
from std.runtime import parallelism_level
from max.algorithm import parallelize
from _mojo.linalg import F64Ptr, list_ptr, vfill, vcopy, vaxpy, vdot_serial
from _mojo.dfjk import pack_tril_dm, unpack_tril_sym


def jk_s8_core(
    eri: F64Ptr,
    nao: Int,
    nj: Int,
    dms_j: F64Ptr,
    vj: F64Ptr,
    nk: Int,
    dms_k: F64Ptr,
    vk: F64Ptr,
):
    """J for ``nj`` densities and K for ``nk`` densities (all symmetric, nao x nao).

    vj : nj x nao x nao and vk : nk x nao x nao are overwritten.
    """
    var npair = nao * (nao + 1) // 2
    var n2 = nao * nao
    # Small problems (a few hundred AO pairs) are faster on one thread than with
    # the dispatch and reduction overhead of several work items.
    var nchunks = max(1, 4 * parallelism_level())
    if npair * (npair + 1) // 2 < (1 << 18):
        nchunks = 1
    if nchunks > npair:
        nchunks = npair

    # pair index -> (i, j)
    var pi = List[Int](length=npair, fill=0)
    var pj = List[Int](length=npair, fill=0)
    var p = 0
    for i in range(nao):
        for j in range(i + 1):
            pi[p] = i
            pj[p] = j
            p += 1
    var ppi = Pointer[Int, MutAnyOrigin](unsafe_from_address=Int(pi.unsafe_ptr()))
    var ppj = Pointer[Int, MutAnyOrigin](unsafe_from_address=Int(pj.unsafe_ptr()))

    # packed D + D^T for the Coulomb part
    var dt = List[Float64](length=max(nj, 1) * npair, fill=0.0)
    var pdt = list_ptr(dt)
    for s in range(nj):
        pack_tril_dm(dms_j.unsafe_offset(s * n2), nao, pdt.unsafe_offset(s * npair))

    var jbuf = List[Float64](length=nchunks * max(nj, 1) * npair, fill=0.0)
    var kbuf = List[Float64](length=nchunks * max(nk, 1) * n2, fill=0.0)
    var pjb = list_ptr(jbuf)
    var pkb = list_ptr(kbuf)

    def work(c: Int) {imm eri, imm nao, imm npair, imm n2, imm nchunks, imm nj, imm nk, imm pdt, imm dms_k, imm pjb, imm pkb, imm ppi, imm ppj}:
        var jc = pjb.unsafe_offset(c * nj * npair)
        var kc = pkb.unsafe_offset(c * nk * n2)
        var ij = c
        while ij < npair:
            var row = eri.unsafe_offset(ij * (ij + 1) // 2)
            var i = ppi[unsafe_offset=ij]
            var j = ppj[unsafe_offset=ij]
            # Coulomb
            for s in range(nj):
                var dts = pdt.unsafe_offset(s * npair)
                var js = jc.unsafe_offset(s * npair)
                js[unsafe_offset=ij] = js[unsafe_offset=ij] + vdot_serial(row, dts, ij + 1)
                vaxpy(js, ij, dts[unsafe_offset=ij], row)
            # Exchange
            if nk > 0:
                var w_ij = 0.5 if i == j else 1.0
                var k = 0
                while k < nao:
                    var kl0 = k * (k + 1) // 2
                    if kl0 > ij:
                        break
                    var lmax = min(k, ij - kl0)          # last l of this k (inclusive)
                    var seg = row.unsafe_offset(kl0)
                    var nfull = lmax                     # l in [0, lmax) carry the plain weight
                    var w_last = w_ij
                    if lmax == k:
                        w_last *= 0.5
                    if kl0 + lmax == ij:
                        w_last *= 0.5
                    var vlast = seg[unsafe_offset=lmax] * w_last
                    for s in range(nk):
                        var d = dms_k.unsafe_offset(s * n2)
                        var ks = kc.unsafe_offset(s * n2)
                        var di = d.unsafe_offset(i * nao)
                        var dj = d.unsafe_offset(j * nao)
                        var ki = ks.unsafe_offset(i * nao)
                        var kj = ks.unsafe_offset(j * nao)
                        var dik = di[unsafe_offset=k]
                        var djk = dj[unsafe_offset=k]
                        # K'_ik += sum_l v_l D_jl ;  K'_jk += sum_l v_l D_il
                        var acc_ik = w_ij * vdot_serial(seg, dj, nfull) + vlast * dj[unsafe_offset=lmax]
                        var acc_jk = w_ij * vdot_serial(seg, di, nfull) + vlast * di[unsafe_offset=lmax]
                        ki[unsafe_offset=k] = ki[unsafe_offset=k] + acc_ik
                        kj[unsafe_offset=k] = kj[unsafe_offset=k] + acc_jk
                        # K'_il += v_l D_jk ;  K'_jl += v_l D_ik   (l < lmax vectorised, l = lmax scalar)
                        vaxpy(ki, nfull, w_ij * djk, seg)
                        vaxpy(kj, nfull, w_ij * dik, seg)
                        ki[unsafe_offset=lmax] = ki[unsafe_offset=lmax] + vlast * djk
                        kj[unsafe_offset=lmax] = kj[unsafe_offset=lmax] + vlast * dik
                    k += 1
            ij += nchunks

    parallelize(work, nchunks)

    # reductions
    if nj > 0:
        var jsum = List[Float64](length=npair, fill=0.0)
        var pjs = list_ptr(jsum)
        for s in range(nj):
            vfill(pjs, npair, 0.0)
            for c in range(nchunks):
                vaxpy(pjs, npair, 1.0, pjb.unsafe_offset((c * nj + s) * npair))
            unpack_tril_sym(pjs, nao, vj.unsafe_offset(s * n2))
        _ = jsum^
    if nk > 0:
        var ksum = List[Float64](length=n2, fill=0.0)
        var pks = list_ptr(ksum)
        for s in range(nk):
            vfill(pks, n2, 0.0)
            for c in range(nchunks):
                vaxpy(pks, n2, 1.0, pkb.unsafe_offset((c * nk + s) * n2))
            var out = vk.unsafe_offset(s * n2)
            for a in range(nao):
                for b in range(nao):
                    out[unsafe_offset=a * nao + b] = pks[unsafe_offset=a * nao + b] + pks[unsafe_offset=b * nao + a]
        _ = ksum^
    _ = pi^
    _ = pj^
    _ = dt^
    _ = jbuf^
    _ = kbuf^
