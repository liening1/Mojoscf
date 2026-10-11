"""Spin-traced 1-, 2- and 3-particle density matrices of FCI vectors (pyscf's ``fci.rdm.make_dm123``).

With the single excitations of the alpha and beta strings (pyscf's link
tables) the vectors

    ak[I, (t, u)] = <I|E_tu|ket>,      ab[I, (p, q)] = <I|E_qp|bra>

over all determinants I give dm1 = bra^T ak and dm2[(p q), (r s)] =
sum_I ab[I, (p q)] ak[I, (r s)] (GEMMs), and the 3-particle matrix

    dm3[p, q, r, s, t, u] = <bra|E_pq E_rs E_tu|ket> = sum_I b[I, (r s p q)] ak[I, (t u)],
    b[I, (r s p q)] = <I|E_sr E_qp|bra> = sum_J <J|E_rs|I> ab[J, (p q)],

for p >= r >= t only (pyscf's ``_complete_dm3_`` fills the other orderings
from these with the 2-particle matrix).  A link (r, s, J, sign) of an alpha
string Ia, E_rs|Ia> = sign|J>, adds sign ab[(J, Ib), (p >= r, q)]^T
ak[(Ia, Ib), (t <= r, u)] summed over the beta strings Ib (contiguous rows
of ab and ak) to the block s of C_r, the n^2 (n - r) x (r + 1) n part of
dm3 with creation index r; a link of a beta string the same summed over the
alpha strings (rows nb n^2 apart).  One GEMM per link: about a sixth of the
n^6 ndet work of all orderings, less the (r, s) blocks without excitations.
Workers take every nwork-th string (alpha, then beta) with a sequential BLAS
and private C_r, summed and scattered into dm3 at the end.
"""
from std.runtime import parallelism_level
from max.algorithm import parallelize
from _mojo.linalg import F64Ptr, Blas, list_ptr, vaxpy

comptime I64Ptr = Pointer[Int64, MutAnyOrigin]


def _excite(
    norb: Int, na: Int, nb: Int, nlinka: Int, nlinkb: Int, linka: I64Ptr, linkb: I64Ptr, ci: F64Ptr,
    dst: F64Ptr, swap: Bool,
):
    """dst[I, (d, c)] (or [I, (c, d)] with ``swap``) = sum of sign ci[J] over the links (c, d, J, sign) of the
    strings of I = (Ia, Ib), i.e. dst[I, (t, u)] = <I|E_tu|ci> (``swap``: dst[I, (p, q)] = <I|E_qp|ci>)."""
    var n2 = norb * norb

    def row(ia: Int) {imm norb, imm na, imm nb, imm nlinka, imm nlinkb, imm linka, imm linkb, imm ci, imm dst, imm swap, imm n2}:
        var o = dst.unsafe_offset(ia * nb * n2)
        for k in range(nb * n2):
            o[unsafe_offset=k] = 0.0
        for l in range(nlinka):
            var e = linka.unsafe_offset((ia * nlinka + l) * 4)
            var c = Int(e[unsafe_offset=0])
            var d = Int(e[unsafe_offset=1])
            var j = Int(e[unsafe_offset=2])
            var sign = Float64(e[unsafe_offset=3])
            var col = c * norb + d if swap else d * norb + c
            var src = ci.unsafe_offset(j * nb)
            for ib in range(nb):
                o[unsafe_offset=ib * n2 + col] += sign * src[unsafe_offset=ib]
        var crow = ci.unsafe_offset(ia * nb)
        for ib in range(nb):
            var ob = o.unsafe_offset(ib * n2)
            for l in range(nlinkb):
                var e = linkb.unsafe_offset((ib * nlinkb + l) * 4)
                var c = Int(e[unsafe_offset=0])
                var d = Int(e[unsafe_offset=1])
                var j = Int(e[unsafe_offset=2])
                var col = c * norb + d if swap else d * norb + c
                ob[unsafe_offset=col] += Float64(e[unsafe_offset=3]) * crow[unsafe_offset=j]

    parallelize(row, na)


def _v_add(dst: F64Ptr, src: F64Ptr, n: Int, c: Int, d: Int, sign: Float64, mode: Int):
    """One link (c, d) of _v_excite: mode 0 dst[d, :] += sign src[c, :], mode 1 dst[:, c] += sign src[d, :]."""
    if mode == 0:
        vaxpy(dst.unsafe_offset(d * n), n, sign, src.unsafe_offset(c * n))
    else:
        var sd = src.unsafe_offset(d * n)
        for k in range(n):
            dst[unsafe_offset=k * n + c] += sign * sd[unsafe_offset=k]


def _v_excite(
    norb: Int, na: Int, nb: Int, nlinka: Int, nlinkb: Int, linka: I64Ptr, linkb: I64Ptr, w: F64Ptr, v: F64Ptr,
    mode: Int,
):
    """The third vectors of pyscf's NEVPT 4-particle contractions from w[I, (x, f)] = <I|W_fx|ci>,
    W_fx = sum_yz (fx|yz) E_yz (symmetric in (x, f)): over the links (c, d, J, sign) of the strings of I,
    mode 0 (``NEVPTkern_cedf_aedf``): v[I, (e, f)] = sum_x <I|E_ex|w_fx> (e = d, x = c), mode 1
    (``NEVPTkern_aedf_ecdf``): v[I, (e, f)] = sum_x <I|E_xf|w_ex> (x = d, f = c)."""
    var n = norb
    var n2 = n * n

    def row(ia: Int) {imm n, imm n2, imm na, imm nb, imm nlinka, imm nlinkb, imm linka, imm linkb, imm w, imm v, imm mode}:
        var o = v.unsafe_offset(ia * nb * n2)
        for k in range(nb * n2):
            o[unsafe_offset=k] = 0.0
        for l in range(nlinka):                                   # alpha links: sources (J, Ib)
            var e = linka.unsafe_offset((ia * nlinka + l) * 4)
            var c = Int(e[unsafe_offset=0])
            var d = Int(e[unsafe_offset=1])
            var j = Int(e[unsafe_offset=2])
            var sign = Float64(e[unsafe_offset=3])
            for ib in range(nb):
                _v_add(o.unsafe_offset(ib * n2), w.unsafe_offset((j * nb + ib) * n2), n, c, d, sign, mode)
        for ib in range(nb):                                      # beta links: sources (Ia, J)
            for l in range(nlinkb):
                var e = linkb.unsafe_offset((ib * nlinkb + l) * 4)
                _v_add(o.unsafe_offset(ib * n2), w.unsafe_offset((ia * nb + Int(e[unsafe_offset=2])) * n2), n,
                       Int(e[unsafe_offset=0]), Int(e[unsafe_offset=1]), Float64(e[unsafe_offset=3]), mode)

    parallelize(row, na)


def _pair_gemms(
    blas_seq: Blas, norb: Int, na: Int, nb: Int, nlinka: Int, nlinkb: Int, linka: I64Ptr, linkb: I64Ptr,
    ab: F64Ptr, third: F64Ptr, tcut: Bool, out6: F64Ptr,
) raises:
    """out6[p, q, r, s, t, u] = sum_I <bra|E_pq E_rs|I> third[I, (t, u)] for p >= r (and t <= r with ``tcut``),
    ab[I, (p, q)] = <I|E_qp|bra>.  <bra|E_pq E_rs|I> = sum_J <J|E_rs|I> ab[J, (p, q)]: a link (r, s, J, sign)
    of an alpha string Ia adds sign ab[(J, Ib), (p >= r, q)]^T third[(Ia, Ib), cols] summed over the beta
    strings Ib (contiguous rows) to the block s of C_r[(s, p - r, q), cols], a link of a beta string the same
    summed over the alpha strings (rows nb norb^2 apart).  One GEMM per link; workers take every nwork-th
    string with a sequential BLAS and private C_r, summed and scattered into out6 at the end."""
    var n = norb
    var n2 = n * n
    var coff = List[Int](length=n + 1, fill=0)
    for r in range(n):
        coff[r + 1] = coff[r] + n2 * (n - r) * ((r + 1) * n if tcut else n2)
    var csize = coff[n]
    var nwork = max(1, min(parallelism_level(), na + nb))
    var buf = List[Float64](length=nwork * csize, fill=0.0)
    var pb = list_ptr(buf)
    var pcoff = coff.unsafe_ptr()
    var ldab = nb * n2

    def work(w: Int) {imm blas_seq, imm n, imm n2, imm na, imm nb, imm nlinka, imm nlinkb, imm linka, imm linkb, imm third, imm ab, imm pb, imm csize, imm pcoff, imm nwork, imm ldab, imm tcut}:
        var cacc = pb.unsafe_offset(w * csize)
        var task = w
        while task < na + nb:
            var alpha_str = task < na
            var st = task if alpha_str else task - na
            var nlink = nlinka if alpha_str else nlinkb
            var links = linka.unsafe_offset(st * nlinka * 4) if alpha_str else linkb.unsafe_offset(st * nlinkb * 4)
            for l in range(nlink):
                var e = links.unsafe_offset(l * 4)
                var r = Int(e[unsafe_offset=0])
                var s = Int(e[unsafe_offset=1])
                var j = Int(e[unsafe_offset=2])
                var sign = Float64(e[unsafe_offset=3])
                var m = n * (n - r)
                var ncol = (r + 1) * n if tcut else n2
                var cdst = cacc.unsafe_offset(pcoff[unsafe_offset=r] + s * m * ncol)
                try:
                    if alpha_str:
                        blas_seq.gemm_ld(True, False, m, ncol, nb, sign, ab.unsafe_offset(j * ldab + r * n), n2,
                                         third.unsafe_offset(st * ldab), n2, 1.0, cdst, ncol)
                    else:
                        blas_seq.gemm_ld(True, False, m, ncol, na, sign, ab.unsafe_offset(j * n2 + r * n), ldab,
                                         third.unsafe_offset(st * n2), ldab, 1.0, cdst, ncol)
                except:
                    pass
            task += nwork

    var nthr = blas_seq.serial_begin()
    parallelize(work, nwork)
    blas_seq.serial_end(nthr)
    for w in range(1, nwork):
        vaxpy(pb, csize, 1.0, pb.unsafe_offset(w * csize))
    for r in range(n):
        var m = n * (n - r)
        var ldc = (r + 1) * n if tcut else n2
        var cr = pb.unsafe_offset(coff[r])
        for s in range(n):
            for p in range(r, n):
                for q in range(n):
                    var src = cr.unsafe_offset(((s * m) + (p - r) * n + q) * ldc)
                    var dst = out6.unsafe_offset((((p * n + q) * n + r) * n + s) * n2)
                    for k in range(ldc):
                        dst[unsafe_offset=k] = src[unsafe_offset=k]
    _ = buf^
    _ = coff^


def fci_dm123_core(
    blas_seq: Blas,
    norb: Int,
    na: Int,
    nb: Int,
    nlinka: Int,
    nlinkb: Int,
    linka: I64Ptr,
    linkb: I64Ptr,
    bra: F64Ptr,
    ket: F64Ptr,
    ak: F64Ptr,
    ab: F64Ptr,
    dm1: F64Ptr,
    dm2: F64Ptr,
    dm3: F64Ptr,
) raises:
    """dm1[p, q] = <bra|E_pq|ket>, dm2[p, q, r, s] = <bra|E_pq E_rs|ket> and dm3[p, q, r, s, t, u] =
    <bra|E_pq E_rs E_tu|ket> for p >= r >= t (other entries untouched).

    linka (na, nlinka, 4), linkb (nb, nlinkb, 4): pyscf's link tables [c, d, J, sign] (E_cd|I> = sign|J>)
    as int64.  bra, ket (na, nb); ak, ab (na nb, norb^2) work arrays.
    """
    var n2 = norb * norb
    var ndet = na * nb
    if norb == 0 or ndet == 0:
        return
    _excite(norb, na, nb, nlinka, nlinkb, linka, linkb, ket, ak, False)
    _excite(norb, na, nb, nlinka, nlinkb, linka, linkb, bra, ab, True)
    blas_seq.gemm(False, False, 1, n2, ndet, 1.0, bra, ak, 0.0, dm1)
    blas_seq.gemm(True, False, n2, n2, ndet, 1.0, ab, ak, 0.0, dm2)
    _pair_gemms(blas_seq, norb, na, nb, nlinka, nlinkb, linka, linkb, ab, ak, True, dm3)


def fci_f3_core(
    blas_seq: Blas,
    norb: Int,
    na: Int,
    nb: Int,
    nlinka: Int,
    nlinkb: Int,
    linka: I64Ptr,
    linkb: I64Ptr,
    ci: F64Ptr,
    eri: F64Ptr,
    mode: Int,
    buf1: F64Ptr,
    buf2: F64Ptr,
    fdm2: F64Ptr,
    fdm3: F64Ptr,
) raises:
    """pyscf's NEVPT 4-particle contractions (``mrpt.nevpt2._contract4pdm``, before its completion):
    fdm3[a, b, c, d, e, f] = <ci|E_ab E_cd O_ef|ci> for a >= c and fdm2[a, b, e, f] = <ci|E_ab O_ef|ci>, with
    O_ef|ci> = sum_x E_ex W_fx|ci> (mode 0, ``NEVPTkern_cedf_aedf``: sum_xyz (fx|yz) <E_ab E_cd E_ex E_yz>) or
    sum_x E_xf W_ex|ci> (mode 1, ``NEVPTkern_aedf_ecdf``), W_fx = sum_yz (fx|yz) E_yz; eri (norb^2, norb^2)
    with the 8-fold symmetry of real (pq|rs).  buf1, buf2 (na nb, norb^2) work arrays."""
    var n2 = norb * norb
    var ndet = na * nb
    if norb == 0 or ndet == 0:
        return
    _excite(norb, na, nb, nlinka, nlinkb, linka, linkb, ci, buf1, False)          # <I|E_yz|ci>
    blas_seq.gemm(False, False, ndet, n2, n2, 1.0, buf1, eri, 0.0, buf2)          # w = <I|W_fx|ci>
    _v_excite(norb, na, nb, nlinka, nlinkb, linka, linkb, buf2, buf1, mode)        # O_ef|ci>
    _excite(norb, na, nb, nlinka, nlinkb, linka, linkb, ci, buf2, True)           # <I|E_qp|ci>
    blas_seq.gemm(True, False, n2, n2, ndet, 1.0, buf2, buf1, 0.0, fdm2)
    _pair_gemms(blas_seq, norb, na, nb, nlinka, nlinkb, linka, linkb, buf2, buf1, False, fdm3)
