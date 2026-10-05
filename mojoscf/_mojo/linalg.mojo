"""Dense linear-algebra primitives for mojoscf.

Every matrix handled here is a row-major, C-contiguous ``float64`` buffer that
is addressed through a raw pointer (normally borrowed from a NumPy array).
The expensive operations, GEMM and the symmetric eigensolvers, are dispatched
to a BLAS/LAPACK shared library that is opened at runtime through ``dlopen``.
When no library is available the portable Mojo fallbacks below are used, so
the package always works, just more slowly for large basis sets.
"""
from std.ffi import OwnedDLHandle, c_int, c_char
from std.memory import Pointer
from std.math import sqrt
from std.sys import simd_width_of
from max.algorithm import parallelize

comptime W = simd_width_of[DType.float64]()
comptime F64Ptr = Pointer[Float64, MutAnyOrigin]

# Vectors shorter than this are reduced on the calling thread (a parallel
# dispatch costs about 10 us, i.e. roughly the serial time for 2^17 elements).
comptime PARALLEL_MIN = 1 << 18


def anyptr[T: AnyType](ref x: T) -> Pointer[T, MutAnyOrigin]:
    """Address of a local value as an origin-erased pointer (for C calls)."""
    return Pointer[T, MutAnyOrigin](unsafe_from_address=Int(Pointer(to=x)))


def list_ptr(ref buf: List[Float64]) -> F64Ptr:
    """Origin-erased pointer to the storage of a ``List[Float64]``."""
    return F64Ptr(unsafe_from_address=Int(buf.unsafe_ptr()))


# --------------------------------------------------------------------------
# Elementwise vector kernels (manually SIMD-unrolled, no closures involved)
# --------------------------------------------------------------------------


def vcopy(dst: F64Ptr, src: F64Ptr, n: Int):
    var i = 0
    while i + W <= n:
        dst.unsafe_store(i, src.unsafe_load[width=W](i))
        i += W
    while i < n:
        dst[unsafe_offset=i] = src[unsafe_offset=i]
        i += 1


def vfill(dst: F64Ptr, n: Int, value: Float64):
    var v = SIMD[DType.float64, W](value)
    var i = 0
    while i + W <= n:
        dst.unsafe_store(i, v)
        i += W
    while i < n:
        dst[unsafe_offset=i] = value
        i += 1


def vscale(x: F64Ptr, n: Int, a: Float64):
    """x *= a."""
    var i = 0
    while i + W <= n:
        x.unsafe_store(i, x.unsafe_load[width=W](i) * a)
        i += W
    while i < n:
        x[unsafe_offset=i] = x[unsafe_offset=i] * a
        i += 1


def vaxpy(y: F64Ptr, n: Int, a: Float64, x: F64Ptr):
    """y += a * x."""
    var i = 0
    while i + W <= n:
        y.unsafe_store(i, y.unsafe_load[width=W](i) + x.unsafe_load[width=W](i) * a)
        i += W
    while i < n:
        y[unsafe_offset=i] = y[unsafe_offset=i] + a * x[unsafe_offset=i]
        i += 1


def vlincomb(dst: F64Ptr, n: Int, a: Float64, x: F64Ptr, b: Float64, y: F64Ptr):
    """dst = a * x + b * y  (``dst`` may alias ``x`` or ``y``)."""
    var i = 0
    while i + W <= n:
        dst.unsafe_store(i, x.unsafe_load[width=W](i) * a + y.unsafe_load[width=W](i) * b)
        i += W
    while i < n:
        dst[unsafe_offset=i] = a * x[unsafe_offset=i] + b * y[unsafe_offset=i]
        i += 1


def vdot_serial(x: F64Ptr, y: F64Ptr, n: Int) -> Float64:
    var acc = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= n:
        acc += x.unsafe_load[width=W](i) * y.unsafe_load[width=W](i)
        i += W
    var s = acc.reduce_add()
    while i < n:
        s += x[unsafe_offset=i] * y[unsafe_offset=i]
        i += 1
    return s


def vdot(x: F64Ptr, y: F64Ptr, n: Int) -> Float64:
    """Dot product, parallelised over chunks for long vectors."""
    if n < PARALLEL_MIN:
        return vdot_serial(x, y, n)
    var nchunks = min(64, (n + PARALLEL_MIN - 1) // PARALLEL_MIN)
    var chunk = (n + nchunks - 1) // nchunks
    var partial = List[Float64](length=nchunks, fill=0.0)
    var pp = list_ptr(partial)

    def work(c: Int) {imm x, imm y, imm n, imm chunk, imm pp}:
        var start = c * chunk
        var stop = min(start + chunk, n)
        if stop > start:
            pp[unsafe_offset=c] = vdot_serial(
                x.unsafe_offset(start), y.unsafe_offset(start), stop - start
            )

    parallelize(work, nchunks)
    var s = 0.0
    for c in range(nchunks):
        s += partial[c]
    return s


def vnorm2diff(x: F64Ptr, y: F64Ptr, n: Int) -> Float64:
    """Sum of squared differences, sum((x - y)**2)."""
    var acc = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= n:
        var d = x.unsafe_load[width=W](i) - y.unsafe_load[width=W](i)
        acc += d * d
        i += W
    var s = acc.reduce_add()
    while i < n:
        var d = x[unsafe_offset=i] - y[unsafe_offset=i]
        s += d * d
        i += 1
    return s


def transpose(dst: F64Ptr, src: F64Ptr, nrow: Int, ncol: Int):
    """dst (ncol x nrow) = src (nrow x ncol)^T, cache-blocked."""
    comptime B = 32
    var i0 = 0
    while i0 < nrow:
        var i1 = min(i0 + B, nrow)
        var j0 = 0
        while j0 < ncol:
            var j1 = min(j0 + B, ncol)
            for i in range(i0, i1):
                for j in range(j0, j1):
                    dst[unsafe_offset=j * nrow + i] = src[unsafe_offset=i * ncol + j]
            j0 = j1
        i0 = i1


def _trace_prod_rows(a: F64Ptr, b: F64Ptr, n: Int, i0: Int, i1: Int) -> Float64:
    """sum over rows i0 <= i < i1 of sum_j a[i, j] * b[j, i], cache-blocked in j."""
    comptime JB = 64
    var acc = SIMD[DType.float64, W](0.0)
    var tail = 0.0
    var j0 = 0
    while j0 < n:
        var j1 = min(j0 + JB, n)
        for i in range(i0, i1):
            var arow = a.unsafe_offset(i * n)
            var bcol = b.unsafe_offset(i)
            var j = j0
            while j + W <= j1:
                acc += arow.unsafe_load[width=W](j) * bcol.unsafe_offset(j * n).unsafe_strided_load[width=W](n)
                j += W
            while j < j1:
                tail += arow[unsafe_offset=j] * bcol[unsafe_offset=j * n]
                j += 1
        j0 = j1
    return acc.reduce_add() + tail


def trace_prod(a: F64Ptr, b: F64Ptr, n: Int) -> Float64:
    """sum_ij a[i, j] * b[j, i]  (numpy.einsum('ij,ji->', a, b))."""
    if n * n < PARALLEL_MIN:
        return _trace_prod_rows(a, b, n, 0, n)
    var nchunks = min(n, 16)
    var rows = (n + nchunks - 1) // nchunks
    var partial = List[Float64](length=nchunks, fill=0.0)
    var pp = list_ptr(partial)

    def work(c: Int) {imm a, imm b, imm n, imm rows, imm pp}:
        var i0 = c * rows
        var i1 = min(i0 + rows, n)
        if i1 > i0:
            pp[unsafe_offset=c] = _trace_prod_rows(a, b, n, i0, i1)

    parallelize(work, nchunks)
    var total = 0.0
    for c in range(nchunks):
        total += partial[c]
    return total


def symmetrize_upper(c: F64Ptr, n: Int):
    """Copy the row-major upper triangle (i <= j) of c onto the lower one."""
    for i in range(n):
        for j in range(i):
            c[unsafe_offset=i * n + j] = c[unsafe_offset=j * n + i]


def adjust_phase(c: F64Ptr, nrow: Int, ncol: Int):
    """Make the largest-magnitude component of every column positive.

    Mirrors ``pyscf.scf.hf._adjust_phase_`` for real orbitals.
    """
    for k in range(ncol):
        var best = 0
        var bestval = abs(c[unsafe_offset=k])
        for i in range(1, nrow):
            var v = abs(c[unsafe_offset=i * ncol + k])
            if v > bestval:
                bestval = v
                best = i
        if c[unsafe_offset=best * ncol + k] < 0.0:
            for i in range(nrow):
                c[unsafe_offset=i * ncol + k] = -c[unsafe_offset=i * ncol + k]


# --------------------------------------------------------------------------
# Native fallbacks: GEMM, Jacobi eigensolver, Cholesky
# --------------------------------------------------------------------------


def gemm_native(
    transa: Bool,
    transb: Bool,
    m: Int,
    n: Int,
    k: Int,
    alpha: Float64,
    a: F64Ptr,
    b: F64Ptr,
    beta: Float64,
    c: F64Ptr,
):
    """C(m x n) = alpha * op(A) * op(B) + beta * C, all row-major."""
    var ta = List[Float64](length=1, fill=0.0)
    var tb = List[Float64](length=1, fill=0.0)
    var pa = a
    var pb = b
    if transa:
        ta = List[Float64](length=m * k, fill=0.0)
        pa = list_ptr(ta)
        transpose(pa, a, k, m)
    if transb:
        tb = List[Float64](length=k * n, fill=0.0)
        pb = list_ptr(tb)
        transpose(pb, b, n, k)

    def row(i: Int) {imm pa, imm pb, imm c, imm n, imm k, imm alpha, imm beta}:
        var crow = c.unsafe_offset(i * n)
        if beta == 0.0:
            vfill(crow, n, 0.0)
        elif beta != 1.0:
            vscale(crow, n, beta)
        var arow = pa.unsafe_offset(i * k)
        for kk in range(k):
            var aik = arow[unsafe_offset=kk] * alpha
            if aik != 0.0:
                vaxpy(crow, n, aik, pb.unsafe_offset(kk * n))

    parallelize(row, m)
    _ = ta^
    _ = tb^


def jacobi_eigh(a: F64Ptr, n: Int, w: F64Ptr, v: F64Ptr):
    """Cyclic Jacobi diagonalisation of the symmetric matrix ``a`` (destroyed).

    Eigenvalues are returned ascending in ``w``; the columns of ``v`` (n x n,
    row-major) are the matching eigenvectors.
    """
    for i in range(n):
        for j in range(n):
            v[unsafe_offset=i * n + j] = 1.0 if i == j else 0.0
    if n <= 1:
        if n == 1:
            w[unsafe_offset=0] = a[unsafe_offset=0]
        return

    for _sweep in range(100):
        var off = 0.0
        var total = 0.0
        for p in range(n):
            total += a[unsafe_offset=p * n + p] * a[unsafe_offset=p * n + p]
            for q in range(p + 1, n):
                off += a[unsafe_offset=p * n + q] * a[unsafe_offset=p * n + q]
        if off == 0.0 or off <= 1e-30 * (total + off):
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                var apq = a[unsafe_offset=p * n + q]
                if apq == 0.0:
                    continue
                var app = a[unsafe_offset=p * n + p]
                var aqq = a[unsafe_offset=q * n + q]
                var theta = (aqq - app) / (2.0 * apq)
                var sign = 1.0 if theta >= 0.0 else -1.0
                var t = sign / (abs(theta) + sqrt(theta * theta + 1.0))
                var cs = 1.0 / sqrt(t * t + 1.0)
                var sn = t * cs
                for kk in range(n):
                    var akp = a[unsafe_offset=kk * n + p]
                    var akq = a[unsafe_offset=kk * n + q]
                    a[unsafe_offset=kk * n + p] = cs * akp - sn * akq
                    a[unsafe_offset=kk * n + q] = sn * akp + cs * akq
                for kk in range(n):
                    var apk = a[unsafe_offset=p * n + kk]
                    var aqk = a[unsafe_offset=q * n + kk]
                    a[unsafe_offset=p * n + kk] = cs * apk - sn * aqk
                    a[unsafe_offset=q * n + kk] = sn * apk + cs * aqk
                for kk in range(n):
                    var vkp = v[unsafe_offset=kk * n + p]
                    var vkq = v[unsafe_offset=kk * n + q]
                    v[unsafe_offset=kk * n + p] = cs * vkp - sn * vkq
                    v[unsafe_offset=kk * n + q] = sn * vkp + cs * vkq

    for i in range(n):
        w[unsafe_offset=i] = a[unsafe_offset=i * n + i]
    # Selection sort of eigenpairs, ascending.
    for i in range(n - 1):
        var best = i
        for j in range(i + 1, n):
            if w[unsafe_offset=j] < w[unsafe_offset=best]:
                best = j
        if best != i:
            var tmp = w[unsafe_offset=i]
            w[unsafe_offset=i] = w[unsafe_offset=best]
            w[unsafe_offset=best] = tmp
            for kk in range(n):
                var t2 = v[unsafe_offset=kk * n + i]
                v[unsafe_offset=kk * n + i] = v[unsafe_offset=kk * n + best]
                v[unsafe_offset=kk * n + best] = t2


def cholesky_lower(s: F64Ptr, n: Int, l: F64Ptr) raises:
    """l (lower triangular, row-major) with s = l l^T."""
    vfill(l, n * n, 0.0)
    for i in range(n):
        for j in range(i + 1):
            var acc = s[unsafe_offset=i * n + j]
            for kk in range(j):
                acc -= l[unsafe_offset=i * n + kk] * l[unsafe_offset=j * n + kk]
            if i == j:
                if acc <= 0.0:
                    raise Error("overlap matrix is not positive definite")
                l[unsafe_offset=i * n + i] = sqrt(acc)
            else:
                l[unsafe_offset=i * n + j] = acc / l[unsafe_offset=j * n + j]


def _forward_subst_rows(l: F64Ptr, n: Int, x: F64Ptr):
    """Solve L X = X in place for the row-major (n x n) right-hand side X."""
    for i in range(n):
        var xi = x.unsafe_offset(i * n)
        for j in range(i):
            vaxpy(xi, n, -l[unsafe_offset=i * n + j], x.unsafe_offset(j * n))
        vscale(xi, n, 1.0 / l[unsafe_offset=i * n + i])


def _backward_subst_rows_t(l: F64Ptr, n: Int, x: F64Ptr):
    """Solve L^T X = X in place (L lower triangular, row-major)."""
    var i = n - 1
    while i >= 0:
        var xi = x.unsafe_offset(i * n)
        for j in range(i + 1, n):
            vaxpy(xi, n, -l[unsafe_offset=j * n + i], x.unsafe_offset(j * n))
        vscale(xi, n, 1.0 / l[unsafe_offset=i * n + i])
        i -= 1


def eigh_gen_native(n: Int, h: F64Ptr, s: F64Ptr, w: F64Ptr, c: F64Ptr) raises:
    """H C = S C diag(w) through Cholesky reduction and Jacobi (h, s destroyed)."""
    var lbuf = List[Float64](length=n * n, fill=0.0)
    var l = list_ptr(lbuf)
    cholesky_lower(s, n, l)
    # Y = L^-1 H
    _forward_subst_rows(l, n, h)
    # H' = (L^-1 Y^T)^T = L^-1 H L^-T  (symmetric)
    var tbuf = List[Float64](length=n * n, fill=0.0)
    var t = list_ptr(tbuf)
    transpose(t, h, n, n)
    _forward_subst_rows(l, n, t)
    transpose(h, t, n, n)
    jacobi_eigh(h, n, w, c)
    # C = L^-T V
    _backward_subst_rows_t(l, n, c)
    _ = lbuf^
    _ = tbuf^


# --------------------------------------------------------------------------
# BLAS / LAPACK dispatch
# --------------------------------------------------------------------------


struct Blas(Movable):
    """Optional handle to a BLAS/LAPACK shared library.

    ``path`` may be empty to force the native Mojo fallbacks.  ``prefix`` is
    prepended to the Fortran symbol names (e.g. ``"scipy_"`` for the OpenBLAS
    bundled with SciPy).  Only the LP64 (32-bit integer) interface is used.

    Opening an already-loaded library is a cheap ``dlopen`` (a few
    microseconds), so a ``Blas`` is created per kernel call; the symbol check
    (``verify``) is only done when probing a library for the first time.
    """

    var handle: Optional[OwnedDLHandle]
    var prefix: String
    var path: String

    def __init__(out self, path: String, prefix: String, verify: Bool = False):
        self.path = path
        self.prefix = prefix
        self.handle = None
        if path.byte_length() > 0:
            try:
                var h = OwnedDLHandle(path)
                if not verify or (
                    h.check_symbol(prefix + "dgemm_")
                    and h.check_symbol(prefix + "dsyrk_")
                    and h.check_symbol(prefix + "dsygvd_")
                    and h.check_symbol(prefix + "dsyevd_")
                ):
                    self.handle = h^
            except:
                pass

    def available(self) -> Bool:
        return Bool(self.handle)

    def gemm(
        self,
        transa: Bool,
        transb: Bool,
        m: Int,
        n: Int,
        k: Int,
        alpha: Float64,
        a: F64Ptr,
        b: F64Ptr,
        beta: Float64,
        c: F64Ptr,
    ) raises:
        """C(m x n) = alpha * op(A) * op(B) + beta * C with row-major storage.

        ``op(A)`` is ``m x k`` (stored ``k x m`` when ``transa``), ``op(B)`` is
        ``k x n`` (stored ``n x k`` when ``transb``).
        """
        if m == 0 or n == 0:
            return
        if k == 0:
            if beta == 0.0:
                vfill(c, m * n, 0.0)
            elif beta != 1.0:
                vscale(c, m * n, beta)
            return
        if not self.handle:
            gemm_native(transa, transb, m, n, k, alpha, a, b, beta, c)
            return
        # Row-major C = op(A) op(B)  <=>  column-major C^T = op(B)^T op(A)^T.
        var ta = c_char(ord("T")) if transa else c_char(ord("N"))
        var tb = c_char(ord("T")) if transb else c_char(ord("N"))
        var mm = c_int(m)
        var nn = c_int(n)
        var kk = c_int(k)
        var lda = c_int(m) if transa else c_int(k)
        var ldb = c_int(k) if transb else c_int(n)
        var ldc = c_int(n)
        var al = alpha
        var be = beta
        var f = self.handle.value().get_function[NoneType](self.prefix + "dgemm_")
        f(
            anyptr(tb), anyptr(ta), anyptr(nn), anyptr(mm), anyptr(kk),
            anyptr(al), b, anyptr(ldb), a, anyptr(lda), anyptr(be), c, anyptr(ldc),
        )

    def syrk_upper(self, n: Int, k: Int, alpha: Float64, a: F64Ptr, beta: Float64, c: F64Ptr) raises:
        """Symmetric rank-k update C (n x n) = alpha * A^T A + beta * C, A (k x n) row-major.

        Only the row-major *upper* triangle (i <= j) of C is guaranteed to be
        written (BLAS does half the work of a GEMM); mirror it with
        ``symmetrize_upper`` once all updates are accumulated.
        """
        if n == 0:
            return
        if not self.handle:
            gemm_native(True, False, n, n, k, alpha, a, a, beta, c)
            return
        # Column-major view: the buffer of A is A^T (n x k), so C = (A^T)(A^T)^T
        # with trans = 'N'; the column-major lower triangle is the row-major upper one.
        var uplo = c_char(ord("L"))
        var trans = c_char(ord("N"))
        var nn = c_int(n)
        var kk = c_int(k)
        var al = alpha
        var be = beta
        var f = self.handle.value().get_function[NoneType](self.prefix + "dsyrk_")
        f(anyptr(uplo), anyptr(trans), anyptr(nn), anyptr(kk), anyptr(al), a, anyptr(nn), anyptr(be), c, anyptr(nn))

    def syr2k_lower(self, n: Int, k: Int, alpha: Float64, a: F64Ptr, b: F64Ptr, beta: Float64, c: F64Ptr) raises:
        """Symmetric rank-2k update C (n x n) = alpha (A B^T + B A^T) + beta C, A and B (n x k) row-major.

        Only the row-major *lower* triangle (i >= j) of C is guaranteed to be
        written (half the work of the two GEMMs).
        """
        if n == 0:
            return
        if not self.handle:
            gemm_native(False, True, n, n, k, alpha, a, b, beta, c)
            gemm_native(False, True, n, n, k, alpha, b, a, 1.0, c)
            return
        # Column-major view: the buffers are A^T, B^T (k x n), so C = A'^T B' + B'^T A'
        # with trans = 'T'; the column-major upper triangle is the row-major lower one.
        var uplo = c_char(ord("U"))
        var trans = c_char(ord("T"))
        var nn = c_int(n)
        var kk = c_int(k)
        var al = alpha
        var be = beta
        var f = self.handle.value().get_function[NoneType](self.prefix + "dsyr2k_")
        f(
            anyptr(uplo), anyptr(trans), anyptr(nn), anyptr(kk), anyptr(al), a, anyptr(kk), b, anyptr(kk),
            anyptr(be), c, anyptr(nn),
        )

    def eigh(self, n: Int, h: F64Ptr, w: F64Ptr, c: F64Ptr) raises:
        """Standard symmetric eigenproblem H C = C diag(w); H is preserved.

        Eigenvalues ascending; eigenvectors in the columns of the row-major
        ``c`` with pyscf's phase convention.
        """
        var abuf = List[Float64](length=n * n, fill=0.0)
        var a = list_ptr(abuf)
        vcopy(a, h, n * n)
        if self.handle:
            var jobz = c_char(ord("V"))
            var uplo = c_char(ord("L"))
            var nn = c_int(n)
            var lwork = c_int(1 + 6 * n + 2 * n * n)
            var liwork = c_int(3 + 5 * n)
            var work = List[Float64](length=Int(lwork), fill=0.0)
            var iwork = List[c_int](length=Int(liwork), fill=0)
            var info = c_int(0)
            var f = self.handle.value().get_function[NoneType](self.prefix + "dsyevd_")
            f(
                anyptr(jobz), anyptr(uplo), anyptr(nn), a, anyptr(nn), w,
                work.unsafe_ptr(), anyptr(lwork), iwork.unsafe_ptr(), anyptr(liwork), anyptr(info),
            )
            _ = work^
            _ = iwork^
            if Int(info) != 0:
                raise Error("LAPACK dsyevd failed with info=" + String(Int(info)))
            # LAPACK leaves eigenvectors column-major in a; expose them row-major.
            transpose(c, a, n, n)
        else:
            jacobi_eigh(a, n, w, c)
        adjust_phase(c, n, n)
        _ = abuf^

    def eigh_gen(self, n: Int, h: F64Ptr, s: F64Ptr, w: F64Ptr, c: F64Ptr) raises:
        """Generalised symmetric eigenproblem H C = S C diag(w); inputs preserved."""
        var abuf = List[Float64](length=n * n, fill=0.0)
        var bbuf = List[Float64](length=n * n, fill=0.0)
        var a = list_ptr(abuf)
        var b = list_ptr(bbuf)
        vcopy(a, h, n * n)
        vcopy(b, s, n * n)
        if self.handle:
            var itype = c_int(1)
            var jobz = c_char(ord("V"))
            var uplo = c_char(ord("L"))
            var nn = c_int(n)
            var lwork = c_int(1 + 6 * n + 2 * n * n)
            var liwork = c_int(3 + 5 * n)
            var work = List[Float64](length=Int(lwork), fill=0.0)
            var iwork = List[c_int](length=Int(liwork), fill=0)
            var info = c_int(0)
            var f = self.handle.value().get_function[NoneType](self.prefix + "dsygvd_")
            f(
                anyptr(itype), anyptr(jobz), anyptr(uplo), anyptr(nn), a, anyptr(nn), b, anyptr(nn), w,
                work.unsafe_ptr(), anyptr(lwork), iwork.unsafe_ptr(), anyptr(liwork), anyptr(info),
            )
            _ = work^
            _ = iwork^
            if Int(info) != 0:
                raise Error("LAPACK dsygvd failed with info=" + String(Int(info)))
            transpose(c, a, n, n)
        else:
            eigh_gen_native(n, a, b, w, c)
        adjust_phase(c, n, n)
        _ = abuf^
        _ = bbuf^
