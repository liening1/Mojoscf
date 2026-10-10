"""MC-PDFT on pyscf's grids: the on-top pair density of a multiconfigurational wave function and its
terms in the effective potential and the nuclear gradient.

    ontop_pi_core   the cumulant part of the on-top pair density,
                    1/2 sum_uvxy psi_u psi_v psi_x psi_y L_uvxy, and for GGA
                    translations its gradient, for every grid point; psi are
                    the active orbitals, L the 2-RDM cumulant
    ontop_grad_core its term in the nuclear gradient at fixed grids,
                    -2 sum_p v_Pi(p) sum_{mu on A} d_x phi_mu sum_u C_mu,u
                    sum_vxy L_uvxy psi_v psi_x psi_y, per atom
    ontop_paaa_core the AO-active-active-active block of the on-top
                    potential, sum_p phi_mu v_Pi psi_u psi_v psi_w

The blocks, the shell selection and the AO values are those of
``numint.mojo``.  Per block the active orbitals are one GEMM, psi = C_sub^T
phi; with the pair products q_kl = psi_k psi_l the cumulant contraction is a
second, t = L^T q (ncas^2 x ncas^2 times ncas^2 x BLK).
"""
from std.memory import Pointer
from std.runtime import parallelism_level
from max.algorithm import parallelize
from _mojo.linalg import F64Ptr, Blas, vfill, list_ptr
from _mojo.integrals import Basis, W, aligned_addr
from _mojo.numint import (
    BLK,
    NV,
    F64V,
    Grid,
    AOWork,
    select_shells,
    eval_block,
    prune_rows,
    gather_rows,
    _rcuts,
    _ncomp,
    _store_pts,
    _load_w,
)


def ontop_pi_core(
    blas: Blas, basis: Basis, ngrid: Int, coords: F64Ptr, nderiv: Int, ncas: Int, mo: F64Ptr, lt: F64Ptr,
    pi_out: F64Ptr,
) raises:
    """pi_out[c][p] for c = 0 (and c = 1..3 for ``nderiv`` 1): the cumulant part of the on-top pair density
    1/2 sum_kl q_kl t_kl and its gradient 2 sum_kl (d_c psi_k) psi_l t_kl, with q_kl = psi_k psi_l,
    t = lt q, psi = mo^T phi (``mo`` nao x ncas, row-major) and ``lt`` the transposed cumulant
    (ncas^2 x ncas^2: lt[kl][ij] = L_ijkl), as pyscf's ``get_ontop_pair_density``.
    """
    var grid = Grid(ngrid, coords)
    var rcut = _rcuts(basis, nderiv)
    var nao = basis.nao
    var ncomp = _ncomp(nderiv)
    var nout = 1 if nderiv == 0 else 4
    var n2 = ncas * ncas
    var nblk = grid.nblk
    var nworkers = max(1, min(parallelism_level(), nblk))

    def work(w: Int) {imm blas, imm basis, imm grid, imm rcut, imm nblk, imm nao, imm ncomp, imm nderiv, imm nout, imm ncas, imm n2, imm mo, imm lt, imm pi_out, imm ngrid, imm nworkers}:
        var ws = AOWork(nao, basis.nbas, ncomp, basis.lmax, basis.nctr_max)
        var bl = List[Float64](length=(ncomp * ncas + 2 * n2) * BLK + W, fill=0.0)
        var psi = F64Ptr(unsafe_from_address=aligned_addr(bl))      # [comp][k][BLK]
        var q = psi.unsafe_offset(ncomp * ncas * BLK)                 # [kl][BLK], then s[k][BLK]
        var t = q.unsafe_offset(n2 * BLK)                             # [kl][BLK]
        var it = 0
        while True:
            var blk = w + it * nworkers
            it += 1
            if blk >= nblk:
                break
            var sel = select_shells(basis, grid, blk, rcut, ws)
            var nrow = sel[1]
            var p0 = blk * BLK
            var npt = min(BLK, ngrid - p0)
            if nrow > 0:
                eval_block(basis, grid, blk, nderiv, sel[0], nrow, ws)
                nrow = prune_rows(nrow, ncomp, ws)
            if nrow == 0:
                for c in range(nout):
                    vfill(pi_out.unsafe_offset(c * ngrid + p0), npt, 0.0)
                continue
            var cs = nrow * BLK
            gather_rows(mo, ncas, ws)
            for c in range(ncomp):
                try:
                    blas.gemm(True, False, ncas, BLK, nrow, 1.0, ws.dsub(), ws.ao().unsafe_offset(c * cs), 0.0,
                              psi.unsafe_offset(c * ncas * BLK))
                except:
                    pass
            for k in range(ncas):
                for l in range(ncas):
                    var dst = q.unsafe_offset((k * ncas + l) * BLK)
                    for v in range(NV):
                        dst.unsafe_store(v * W, psi.unsafe_load[width=W](k * BLK + v * W)
                                         * psi.unsafe_load[width=W](l * BLK + v * W))
            try:
                blas.gemm(False, False, n2, BLK, n2, 1.0, lt, q, 0.0, t)
            except:
                pass
            # s_k = sum_l psi_l t_kl (into the first ncas rows of q)
            for k in range(ncas):
                for v in range(NV):
                    var acc = F64V(0.0)
                    for l in range(ncas):
                        acc += psi.unsafe_load[width=W](l * BLK + v * W) * t.unsafe_load[width=W]((k * ncas + l) * BLK + v * W)
                    q.unsafe_store(k * BLK + v * W, acc)
            for c in range(nout):
                var pc = psi.unsafe_offset(c * ncas * BLK)
                var fac = 0.5 if c == 0 else 2.0
                for v in range(NV):
                    var acc = F64V(0.0)
                    for k in range(ncas):
                        acc += pc.unsafe_load[width=W](k * BLK + v * W) * q.unsafe_load[width=W](k * BLK + v * W)
                    _store_pts(pi_out.unsafe_offset(c * ngrid + p0), v, npt, acc * fac)
        _ = ws^
        _ = bl^

    var nthr = blas.serial_begin()
    if nworkers == 1:
        work(0)
    else:
        parallelize(work, nworkers)
    blas.serial_end(nthr)
    _ = grid^
    _ = rcut^


def ontop_grad_core(
    blas: Blas, basis: Basis, ngrid: Int, coords: F64Ptr, wpi: F64Ptr, ncas: Int, mo: F64Ptr, lmat: F64Ptr,
    de_out: F64Ptr,
) raises:
    """On-top pair-density term of the MC-PDFT nuclear gradient at fixed grids (translated functionals):
    de_out[A][x] = -2 sum_p sum_{mu on A} d_x phi_mu(p) (C e(p))_mu (natm x 3, overwritten), with
    e_u = wpi(p) sum_v psi_v t_uv, t = lmat q, q_xy = psi_x psi_y, psi = C^T phi (C = ``mo``, nao x ncas,
    row-major; ``lmat`` ncas^2 x ncas^2, lmat[uv][xy] = L_uvxy; ``wpi`` the weighted v_Pi): pyscf's
    ``tmp_dv1`` of ``grad.mcpdft.xc_response``, summed over the AOs of each atom.
    """
    var grid = Grid(ngrid, coords)
    var rcut = _rcuts(basis, 1)
    var nao = basis.nao
    var natm = basis.natm
    var ncomp = 4
    var n2 = ncas * ncas
    var nblk = grid.nblk
    var nworkers = max(1, min(parallelism_level(), nblk))
    var accl = List[Float64](length=nworkers * natm * 3 + 1, fill=0.0)
    var pacc = list_ptr(accl)
    var aoatom_l = List[Int](length=max(nao, 1), fill=0)      # atom of every AO
    for b in range(basis.nbas):
        for i in range(basis.ao_loc[b], basis.ao_loc[b + 1]):
            aoatom_l[i] = basis.atom[b]
    var aoatom = Pointer[Int, MutAnyOrigin](unsafe_from_address=Int(aoatom_l.unsafe_ptr()))

    def work(w: Int) {imm blas, imm basis, imm grid, imm rcut, imm nblk, imm nao, imm ncomp, imm ncas, imm n2, imm mo, imm lmat, imm wpi, imm pacc, imm ngrid, imm aoatom, imm nworkers}:
        var ws = AOWork(nao, basis.nbas, ncomp, basis.lmax, basis.nctr_max)
        var acc = pacc.unsafe_offset(w * basis.natm * 3)
        var bl = List[Float64](length=(ncas + 2 * n2) * BLK + W, fill=0.0)
        var psi = F64Ptr(unsafe_from_address=aligned_addr(bl))      # [k][BLK]
        var q = psi.unsafe_offset(ncas * BLK)                         # [xy][BLK], then e[u][BLK]
        var t = q.unsafe_offset(n2 * BLK)                             # [uv][BLK]
        var rowatm = List[Int](length=max(nao, 1), fill=0)
        var it = 0
        while True:
            var blk = w + it * nworkers
            it += 1
            if blk >= nblk:
                break
            var sel = select_shells(basis, grid, blk, rcut, ws)
            var nrow = sel[1]
            if nrow == 0:
                continue
            var p0 = blk * BLK
            var npt = min(BLK, ngrid - p0)
            eval_block(basis, grid, blk, 1, sel[0], nrow, ws)
            nrow = prune_rows(nrow, ncomp, ws)
            if nrow == 0:
                continue
            for r in range(nrow):
                rowatm[r] = aoatom[unsafe_offset=ws.rows[r]]
            var cs = nrow * BLK
            var ao = ws.ao()
            gather_rows(mo, ncas, ws)
            try:
                blas.gemm(True, False, ncas, BLK, nrow, 1.0, ws.dsub(), ao, 0.0, psi)
            except:
                pass
            for k in range(ncas):
                for l in range(ncas):
                    var dst = q.unsafe_offset((k * ncas + l) * BLK)
                    for v in range(NV):
                        dst.unsafe_store(v * W, psi.unsafe_load[width=W](k * BLK + v * W)
                                         * psi.unsafe_load[width=W](l * BLK + v * W))
            try:
                blas.gemm(False, False, n2, BLK, n2, 1.0, lmat, q, 0.0, t)
            except:
                pass
            # e_u = wpi sum_v psi_v t_uv (into the first ncas rows of q)
            for u in range(ncas):
                for v in range(NV):
                    var acc_e = F64V(0.0)
                    for l in range(ncas):
                        acc_e += psi.unsafe_load[width=W](l * BLK + v * W) * t.unsafe_load[width=W]((u * ncas + l) * BLK + v * W)
                    q.unsafe_store(u * BLK + v * W, acc_e * _load_w(wpi, p0, v, npt))
            # Y = C_sub e (nrow x BLK)
            var yb = ws.y()
            try:
                blas.gemm(False, False, nrow, BLK, ncas, 1.0, ws.dsub(), q, 0.0, yb)
            except:
                pass
            for i in range(nrow):
                var tx = F64V(0.0)
                var ty = F64V(0.0)
                var tz = F64V(0.0)
                for v in range(NV):
                    var o = i * BLK + v * W
                    var y = yb.unsafe_load[width=W](o)
                    tx += ao.unsafe_load[width=W](cs + o) * y
                    ty += ao.unsafe_load[width=W](2 * cs + o) * y
                    tz += ao.unsafe_load[width=W](3 * cs + o) * y
                var a = rowatm[i]
                acc[unsafe_offset=3 * a] -= 2.0 * tx.reduce_add()
                acc[unsafe_offset=3 * a + 1] -= 2.0 * ty.reduce_add()
                acc[unsafe_offset=3 * a + 2] -= 2.0 * tz.reduce_add()
        _ = ws^
        _ = bl^
        _ = rowatm^

    var nthr = blas.serial_begin()
    if nworkers == 1:
        work(0)
    else:
        parallelize(work, nworkers)
    blas.serial_end(nthr)
    _ = aoatom_l^
    for i in range(natm * 3):
        var v = 0.0
        for w2 in range(nworkers):
            v += pacc[unsafe_offset=w2 * natm * 3 + i]
        de_out[unsafe_offset=i] = v
    _ = accl^
    _ = grid^
    _ = rcut^


def ontop_paaa_core(
    blas: Blas, basis: Basis, ngrid: Int, coords: F64Ptr, wpi: F64Ptr, ncas: Int, mo: F64Ptr, paaa_out: F64Ptr,
) raises:
    """paaa_out[mu][(u v w)] = sum_p phi_mu(p) wpi(p) psi_u(p) psi_v(p) psi_w(p) (nao x ncas^3, overwritten), with
    psi = C^T phi (C = ``mo``, nao x ncas, row-major) and ``wpi`` the weighted v_Pi: the ``paaa`` block of
    pyscf's ``pdft_eff._ERIS`` (translated functionals) before the MO transformation of its first index.
    Per block one GEMM, V_sub = phi Z^T with Z_(uvw) = wpi psi_u psi_v psi_w, added to per-thread
    matrices.
    """
    var grid = Grid(ngrid, coords)
    var rcut = _rcuts(basis, 0)
    var nao = basis.nao
    var n3 = ncas * ncas * ncas
    var nblk = grid.nblk
    var nworkers = max(1, min(parallelism_level(), nblk))
    var total = nao * n3
    var accl = List[Float64](length=nworkers * total + 1, fill=0.0)
    var pacc = list_ptr(accl)

    def work(w: Int) {imm blas, imm basis, imm grid, imm rcut, imm nblk, imm nao, imm ncas, imm n3, imm mo, imm wpi, imm pacc, imm total, imm ngrid, imm nworkers}:
        var ws = AOWork(nao, basis.nbas, 1, basis.lmax, basis.nctr_max)
        var acc = pacc.unsafe_offset(w * total)
        var bl = List[Float64](length=(ncas + n3) * BLK + nao * n3 + 2 * W, fill=0.0)
        var psi = F64Ptr(unsafe_from_address=aligned_addr(bl))      # [k][BLK]
        var z = psi.unsafe_offset(ncas * BLK)                         # [(uvw)][BLK]
        var vsub = z.unsafe_offset(n3 * BLK)                          # [row][(uvw)]
        var it = 0
        while True:
            var blk = w + it * nworkers
            it += 1
            if blk >= nblk:
                break
            var sel = select_shells(basis, grid, blk, rcut, ws)
            var nrow = sel[1]
            if nrow == 0:
                continue
            var p0 = blk * BLK
            var npt = min(BLK, ngrid - p0)
            eval_block(basis, grid, blk, 0, sel[0], nrow, ws)
            nrow = prune_rows(nrow, 1, ws)
            if nrow == 0:
                continue
            gather_rows(mo, ncas, ws)
            try:
                blas.gemm(True, False, ncas, BLK, nrow, 1.0, ws.dsub(), ws.ao(), 0.0, psi)
            except:
                pass
            for v in range(NV):
                var wp = _load_w(wpi, p0, v, npt)
                for a in range(ncas):
                    var pa = psi.unsafe_load[width=W](a * BLK + v * W) * wp
                    for b in range(ncas):
                        var pab = pa * psi.unsafe_load[width=W](b * BLK + v * W)
                        for c in range(ncas):
                            z.unsafe_store(((a * ncas + b) * ncas + c) * BLK + v * W,
                                           pab * psi.unsafe_load[width=W](c * BLK + v * W))
            try:
                blas.gemm(False, True, nrow, n3, BLK, 1.0, ws.ao(), z, 0.0, vsub)
            except:
                pass
            for ri in range(ws.nrun):
                for k in range(ws.run_len[ri]):
                    var dst = acc.unsafe_offset((ws.run_ao[ri] + k) * n3)
                    var src = vsub.unsafe_offset((ws.run_row[ri] + k) * n3)
                    var j = 0
                    while j + W <= n3:
                        dst.unsafe_store(j, dst.unsafe_load[width=W](j) + src.unsafe_load[width=W](j))
                        j += W
                    while j < n3:
                        dst[unsafe_offset=j] += src[unsafe_offset=j]
                        j += 1
        _ = ws^
        _ = bl^

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
        paaa_out[unsafe_offset=i] = v
    _ = accl^
    _ = grid^
    _ = rcut^
