"""SCF "glue" kernels: the per-iteration pieces pyscf implements in Python.

Each function mirrors one small routine from ``pyscf.scf.hf`` /
``pyscf.scf.diis`` so results agree with pyscf to round-off.  Matrices are
row-major float64 buffers; ``nao`` is the AO dimension and ``nmo`` the number
of molecular orbitals (``nmo < nao`` only when linear dependencies in the
basis were projected out with an orthogonaliser ``x``).
"""
from std.memory import Pointer
from std.math import sqrt
from max.algorithm import parallelize
from _mojo.linalg import (
    F64Ptr,
    W,
    Blas,
    list_ptr,
    vcopy,
    vfill,
    vdot,
    vaxpy,
    vlincomb,
    vnorm2diff,
    transpose,
    trace_prod,
    adjust_phase,
)


def count_occupied(nmo: Int, mo_occ: F64Ptr) -> Int:
    var nocc = 0
    for k in range(nmo):
        if mo_occ[unsafe_offset=k] > 0.0:
            nocc += 1
    return nocc


def gather_columns(
    nao: Int, nmo: Int, mo_coeff: F64Ptr, mo_occ: F64Ptr, occupied: Bool, dst: F64Ptr, ncol: Int
):
    """Copy the occupied (or virtual) columns of mo_coeff into dst (nao x ncol)."""
    var j = 0
    for k in range(nmo):
        var is_occ = mo_occ[unsafe_offset=k] > 0.0
        if is_occ == occupied:
            for i in range(nao):
                dst[unsafe_offset=i * ncol + j] = mo_coeff[unsafe_offset=i * nmo + k]
            j += 1


def make_rdm1(blas: Blas, nao: Int, nmo: Int, mo_coeff: F64Ptr, mo_occ: F64Ptr, dm: F64Ptr) raises:
    """dm = C_occ diag(occ) C_occ^T   (pyscf.scf.hf.make_rdm1)."""
    var nocc = count_occupied(nmo, mo_occ)
    if nocc == 0:
        vfill(dm, nao * nao, 0.0)
        return
    var cocc = List[Float64](length=nao * nocc, fill=0.0)
    var cocc_w = List[Float64](length=nao * nocc, fill=0.0)
    var pc = list_ptr(cocc)
    var pw = list_ptr(cocc_w)
    var j = 0
    for k in range(nmo):
        var o = mo_occ[unsafe_offset=k]
        if o > 0.0:
            for i in range(nao):
                var v = mo_coeff[unsafe_offset=i * nmo + k]
                pc[unsafe_offset=i * nocc + j] = v
                pw[unsafe_offset=i * nocc + j] = v * o
            j += 1
    blas.gemm(False, True, nao, nao, nocc, 1.0, pw, pc, 0.0, dm)
    _ = cocc^
    _ = cocc_w^


def get_occ(
    nmo: Int, mo_energy: F64Ptr, nocc: Int, mo_occ: F64Ptr, homo_lumo: F64Ptr, occ_value: Float64 = 2.0
) -> Bool:
    """Aufbau occupations (``occ_value`` electrons in the ``nocc`` lowest orbitals).

    ``occ_value`` is 2 for restricted and 1 for each spin channel of UHF.

    Follows pyscf: orbitals are ranked by a *stable* sort of the energies
    rounded to 9 decimals.  Returns True and fills ``homo_lumo`` when both a
    HOMO and a LUMO exist.
    """
    var key = List[Float64](length=nmo, fill=0.0)
    var idx = List[Int](length=nmo, fill=0)
    for i in range(nmo):
        key[i] = round(mo_energy[unsafe_offset=i] * 1e9) / 1e9
        idx[i] = i
    # Stable insertion sort; LAPACK already returns the energies ascending.
    for i in range(1, nmo):
        var j = i
        while j > 0 and key[idx[j - 1]] > key[idx[j]]:
            var t = idx[j - 1]
            idx[j - 1] = idx[j]
            idx[j] = t
            j -= 1
    vfill(mo_occ, nmo, 0.0)
    for i in range(min(nocc, nmo)):
        mo_occ[unsafe_offset=idx[i]] = occ_value
    if 0 < nocc and nocc < nmo:
        homo_lumo[unsafe_offset=0] = mo_energy[unsafe_offset=idx[nocc - 1]]
        homo_lumo[unsafe_offset=1] = mo_energy[unsafe_offset=idx[nocc]]
        return True
    return False


def get_grad(
    blas: Blas,
    nao: Int,
    nmo: Int,
    mo_coeff: F64Ptr,
    mo_occ: F64Ptr,
    fock: F64Ptr,
    g: F64Ptr,
    prefactor: Float64 = 2.0,
) raises -> Int:
    """Orbital gradient g = prefactor * C_vir^T F C_occ, flattened (nvir x nocc).

    ``prefactor`` is 2 for RHF and 1 for a UHF spin channel.  Returns the
    gradient length; ``g`` must hold at least nvir*nocc values.
    """
    var nocc = count_occupied(nmo, mo_occ)
    var nvir = nmo - nocc
    if nocc == 0 or nvir == 0:
        return 0
    var cocc = List[Float64](length=nao * nocc, fill=0.0)
    var cvir = List[Float64](length=nao * nvir, fill=0.0)
    var fc = List[Float64](length=nao * nocc, fill=0.0)
    var pocc = list_ptr(cocc)
    var pvir = list_ptr(cvir)
    var pfc = list_ptr(fc)
    gather_columns(nao, nmo, mo_coeff, mo_occ, True, pocc, nocc)
    gather_columns(nao, nmo, mo_coeff, mo_occ, False, pvir, nvir)
    blas.gemm(False, False, nao, nocc, nao, 1.0, fock, pocc, 0.0, pfc)
    blas.gemm(True, False, nvir, nocc, nao, prefactor, pvir, pfc, 0.0, g)
    _ = cocc^
    _ = cvir^
    _ = fc^
    return nvir * nocc


def grad_sumsq(
    blas: Blas,
    nao: Int,
    nmo: Int,
    mo_coeff: F64Ptr,
    mo_occ: F64Ptr,
    fock: F64Ptr,
    prefactor: Float64 = 2.0,
) raises -> Float64:
    """Sum of squares of the orbital gradient of one spin channel."""
    var nocc = count_occupied(nmo, mo_occ)
    var nvir = nmo - nocc
    if nocc == 0 or nvir == 0:
        return 0.0
    var g = List[Float64](length=nvir * nocc, fill=0.0)
    var pg = list_ptr(g)
    var ng = get_grad(blas, nao, nmo, mo_coeff, mo_occ, fock, pg, prefactor)
    var result = vdot(pg, pg, ng)
    _ = g^
    return result


def damping(n2: Int, f: F64Ptr, f_prev: F64Ptr, factor: Float64, dst: F64Ptr):
    """dst = f * (1 - factor) + f_prev * factor   (pyscf.scf.hf.damping)."""
    vlincomb(dst, n2, 1.0 - factor, f, factor, f_prev)


def level_shift(
    blas: Blas,
    nao: Int,
    s: F64Ptr,
    dm: F64Ptr,
    f: F64Ptr,
    factor: Float64,
    dst: F64Ptr,
    dm_scale: Float64 = 0.5,
) raises:
    """dst = f + factor * (s - s (dm_scale * dm) s)   (pyscf.scf.hf.level_shift).

    pyscf passes ``dm * 0.5`` for RHF (``dm_scale = 0.5``) and the plain spin
    density for UHF (``dm_scale = 1``).
    """
    var n2 = nao * nao
    var sd = List[Float64](length=n2, fill=0.0)
    var sds = List[Float64](length=n2, fill=0.0)
    var psd = list_ptr(sd)
    var psds = list_ptr(sds)
    blas.gemm(False, False, nao, nao, nao, dm_scale, s, dm, 0.0, psd)
    blas.gemm(False, False, nao, nao, nao, 1.0, psd, s, 0.0, psds)
    # dst = f + factor*s - factor*sds
    vlincomb(dst, n2, 1.0, f, factor, s)
    vaxpy(dst, n2, -factor, psds)
    _ = sd^
    _ = sds^


def diis_errvec(
    blas: Blas,
    nao: Int,
    nmo: Int,
    s: F64Ptr,
    dm: F64Ptr,
    f: F64Ptr,
    has_x: Bool,
    x: F64Ptr,
    err: F64Ptr,
) raises -> Int:
    """CDIIS error vector (SDF)^T - SDF, optionally in the orthogonal basis.

    With ``has_x`` the matrix is first projected as X^T (S D F) X where X is the
    (nao x nmo) orthogonaliser, matching ``pyscf.scf.diis.get_err_vec_orth``.
    Returns the length of ``err``.
    """
    var n2 = nao * nao
    var sd = List[Float64](length=n2, fill=0.0)
    var sdf = List[Float64](length=n2, fill=0.0)
    var psd = list_ptr(sd)
    var psdf = list_ptr(sdf)
    blas.gemm(False, False, nao, nao, nao, 1.0, s, dm, 0.0, psd)
    blas.gemm(False, False, nao, nao, nao, 1.0, psd, f, 0.0, psdf)
    if has_x:
        var t = List[Float64](length=nmo * nao, fill=0.0)
        var u = List[Float64](length=nmo * nmo, fill=0.0)
        var pt = list_ptr(t)
        var pu = list_ptr(u)
        blas.gemm(True, False, nmo, nao, nao, 1.0, x, psdf, 0.0, pt)
        blas.gemm(False, False, nmo, nmo, nao, 1.0, pt, x, 0.0, pu)
        for i in range(nmo):
            for j in range(nmo):
                err[unsafe_offset=i * nmo + j] = pu[unsafe_offset=j * nmo + i] - pu[unsafe_offset=i * nmo + j]
        _ = t^
        _ = u^
        _ = sd^
        _ = sdf^
        return nmo * nmo
    for i in range(nao):
        for j in range(nao):
            err[unsafe_offset=i * nao + j] = psdf[unsafe_offset=j * nao + i] - psdf[unsafe_offset=i * nao + j]
    _ = sd^
    _ = sdf^
    return n2


def eigh_fock(
    blas: Blas,
    nao: Int,
    nmo: Int,
    f: F64Ptr,
    s: F64Ptr,
    has_x: Bool,
    x: F64Ptr,
    mo_energy: F64Ptr,
    mo_coeff: F64Ptr,
) raises:
    """Diagonalise the Fock matrix: F C = S C E, or X^T F X C' = C' E, C = X C'.

    Mirrors ``pyscf.scf.hf.SCF._eigh`` including the phase convention.
    """
    if not has_x:
        blas.eigh_gen(nao, f, s, mo_energy, mo_coeff)
        return
    var t = List[Float64](length=nmo * nao, fill=0.0)
    var hp = List[Float64](length=nmo * nmo, fill=0.0)
    var cp = List[Float64](length=nmo * nmo, fill=0.0)
    var pt = list_ptr(t)
    var php = list_ptr(hp)
    var pcp = list_ptr(cp)
    blas.gemm(True, False, nmo, nao, nao, 1.0, x, f, 0.0, pt)
    blas.gemm(False, False, nmo, nmo, nao, 1.0, pt, x, 0.0, php)
    blas.eigh(nmo, php, mo_energy, pcp)
    blas.gemm(False, False, nao, nmo, nmo, 1.0, x, pcp, 0.0, mo_coeff)
    adjust_phase(mo_coeff, nao, nmo)
    _ = t^
    _ = hp^
    _ = cp^


def jk_dense(nao: Int, eri: F64Ptr, dm: F64Ptr, vj: F64Ptr, vk: F64Ptr):
    """Coulomb and exchange matrices from a full (nao^4) ERI tensor.

    vj[k,l] = sum_ij eri[i,j,k,l] dm[j,i]  and  vk[i,l] = sum_jk eri[i,j,k,l] dm[j,k],
    exactly as ``pyscf.scf.hf.dot_eri_dm`` does for unpacked integrals.
    """
    var n = nao
    var n2 = n * n
    var n3 = n2 * n
    var dmt = List[Float64](length=n2, fill=0.0)
    var pdmt = list_ptr(dmt)
    transpose(pdmt, dm, n, n)
    vfill(vj, n2, 0.0)
    vfill(vk, n2, 0.0)

    # Exchange: row i of vk only touches eri[i, :, :, :].
    def krow(i: Int) {imm eri, imm dm, imm vk, imm n, imm n2, imm n3}:
        var dst = vk.unsafe_offset(i * n)
        for j in range(n):
            for k in range(n):
                var d = dm[unsafe_offset=j * n + k]
                if d != 0.0:
                    vaxpy(dst, n, d, eri.unsafe_offset(i * n3 + j * n2 + k * n))

    parallelize(krow, n)

    # Coulomb: vj (as a flat n2 vector) = sum_ij dm^T[ij] * eri[ij, :]; split the
    # output range across workers so no reduction is needed.
    var nchunks = min(n2, 64)
    var chunk = (n2 + nchunks - 1) // nchunks

    def jchunk(c: Int) {imm eri, imm pdmt, imm vj, imm n2, imm chunk}:
        var start = c * chunk
        var stop = min(start + chunk, n2)
        if stop <= start:
            return
        var dst = vj.unsafe_offset(start)
        var length = stop - start
        for ij in range(n2):
            var d = pdmt[unsafe_offset=ij]
            if d != 0.0:
                vaxpy(dst, length, d, eri.unsafe_offset(ij * n2 + start))

    parallelize(jchunk, nchunks)
    _ = dmt^
