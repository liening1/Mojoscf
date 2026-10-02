"""Pulay DIIS (CDIIS) with pyscf's exact bookkeeping.

The algorithm reproduces ``pyscf.lib.diis.DIIS.update`` / ``extrapolate`` for
the case where an explicit error vector is supplied on every call (which is
how ``pyscf.scf.diis.CDIIS`` uses it):

* vectors and error vectors live in ring buffers of ``space`` slots,
* the (space+1)^2 matrix ``H`` keeps the error overlaps with a border of ones,
* the coefficients come from ``solve(H[:nd+1,:nd+1], e_0)`` unless the
  eigenvalues of that block show a linear dependence, in which case the
  pseudo-inverse built from the eigen-decomposition is used (threshold 1e-14).

The state is kept in caller-provided buffers so the same code serves both
the native SCF driver (``DIIS`` struct) and the NumPy-backed Python class.
"""
from std.memory import Pointer
from std.math import sqrt
from _mojo.linalg import F64Ptr, list_ptr, vcopy, vdot, vaxpy, vfill, jacobi_eigh

comptime LINDEP_THRESHOLD = 1e-14


def solve_diis_coefficients(nd1: Int, h: F64Ptr, c: F64Ptr):
    """c = H^-1 e_0 for the (nd1 x nd1) bordered matrix H, pyscf style.

    ``h`` is destroyed.
    """
    # Eigen-decomposition to detect (near) linear dependence.
    var hcopy = List[Float64](length=nd1 * nd1, fill=0.0)
    var w = List[Float64](length=nd1, fill=0.0)
    var v = List[Float64](length=nd1 * nd1, fill=0.0)
    var ph = list_ptr(hcopy)
    var pw = list_ptr(w)
    var pv = list_ptr(v)
    vcopy(ph, h, nd1 * nd1)
    jacobi_eigh(ph, nd1, pw, pv)
    var singular = False
    for i in range(nd1):
        if abs(w[i]) < LINDEP_THRESHOLD:
            singular = True
    vfill(c, nd1, 0.0)
    if singular:
        # c = V diag(1/w) V^T e_0 restricted to |w| > threshold; (V^T e_0)_i = V[0, i].
        for i in range(nd1):
            if abs(w[i]) >= LINDEP_THRESHOLD:
                var coef = pv[unsafe_offset=i] / w[i]
                for r in range(nd1):
                    c[unsafe_offset=r] = c[unsafe_offset=r] + coef * pv[unsafe_offset=r * nd1 + i]
        _ = hcopy^
        _ = v^
        return
    _ = hcopy^
    _ = v^
    # Gaussian elimination with partial pivoting on [h | e_0].
    var rhs = List[Float64](length=nd1, fill=0.0)
    rhs[0] = 1.0
    for col in range(nd1):
        var piv = col
        var best = abs(h[unsafe_offset=col * nd1 + col])
        for r in range(col + 1, nd1):
            var val = abs(h[unsafe_offset=r * nd1 + col])
            if val > best:
                best = val
                piv = r
        if piv != col:
            for k in range(nd1):
                var t = h[unsafe_offset=col * nd1 + k]
                h[unsafe_offset=col * nd1 + k] = h[unsafe_offset=piv * nd1 + k]
                h[unsafe_offset=piv * nd1 + k] = t
            var tr = rhs[col]
            rhs[col] = rhs[piv]
            rhs[piv] = tr
        var d = h[unsafe_offset=col * nd1 + col]
        for r in range(col + 1, nd1):
            var fct = h[unsafe_offset=r * nd1 + col] / d
            if fct != 0.0:
                for k in range(col, nd1):
                    h[unsafe_offset=r * nd1 + k] = h[unsafe_offset=r * nd1 + k] - fct * h[unsafe_offset=col * nd1 + k]
                rhs[r] = rhs[r] - fct * rhs[col]
    var r = nd1 - 1
    while r >= 0:
        var acc = rhs[r]
        for k in range(r + 1, nd1):
            acc -= h[unsafe_offset=r * nd1 + k] * c[unsafe_offset=k]
        c[unsafe_offset=r] = acc / h[unsafe_offset=r * nd1 + r]
        r -= 1


def diis_init_hmat(space: Int, hmat: F64Ptr):
    """Zero H and set its border of ones (H[0, 1:] = H[1:, 0] = 1)."""
    var sp1 = space + 1
    vfill(hmat, sp1 * sp1, 0.0)
    for i in range(1, sp1):
        hmat[unsafe_offset=i] = 1.0
        hmat[unsafe_offset=i * sp1] = 1.0


def diis_update_buffers(
    space: Int,
    min_space: Int,
    vlen: Int,
    elen: Int,
    bufx: F64Ptr,
    bufe: F64Ptr,
    hmat: F64Ptr,
    state: Pointer[Int64, MutAnyOrigin],
    x: F64Ptr,
    xerr: F64Ptr,
    dst: F64Ptr,
) -> Int:
    """One ``DIIS.update(x, xerr)`` step on external buffers.

    ``state[0]`` is the ring-buffer head and ``state[1]`` the number of stored
    vectors.  ``dst`` receives the extrapolated vector (it may alias ``x``).
    Returns the number of vectors in the subspace after the push.
    """
    var head = Int(state[unsafe_offset=0])
    var count = Int(state[unsafe_offset=1])
    var sp1 = space + 1
    # push_err_vec
    if head >= space:
        head = 0
    vcopy(bufe.unsafe_offset(head * elen), xerr, elen)
    # push_vec (error vector was supplied)
    vcopy(bufx.unsafe_offset(head * vlen), x, vlen)
    head += 1
    count = min(count + 1, space)
    state[unsafe_offset=0] = Int64(head)
    state[unsafe_offset=1] = Int64(count)
    var nd = count
    if nd < min_space:
        if Int(dst) != Int(x):
            vcopy(dst, x, vlen)
        return nd
    # Overlaps of the newest error vector (slot head-1 <-> H row head).
    var dt = bufe.unsafe_offset((head - 1) * elen)
    for i in range(nd):
        var tmp = vdot(dt, bufe.unsafe_offset(i * elen), elen)
        hmat[unsafe_offset=head * sp1 + i + 1] = tmp
        hmat[unsafe_offset=(i + 1) * sp1 + head] = tmp
    # Extrapolate.
    var nd1 = nd + 1
    var h = List[Float64](length=nd1 * nd1, fill=0.0)
    var c = List[Float64](length=nd1, fill=0.0)
    var ph = list_ptr(h)
    var pc = list_ptr(c)
    for i in range(nd1):
        for j in range(nd1):
            ph[unsafe_offset=i * nd1 + j] = hmat[unsafe_offset=i * sp1 + j]
    solve_diis_coefficients(nd1, ph, pc)
    vfill(dst, vlen, 0.0)
    for i in range(nd):
        vaxpy(dst, vlen, c[i + 1], bufx.unsafe_offset(i * vlen))
    _ = h^
    return nd


struct DIIS(Movable):
    """Owning CDIIS state used by the native SCF driver."""

    var space: Int
    var min_space: Int
    var vlen: Int
    var elen: Int
    var bufx: List[Float64]
    var bufe: List[Float64]
    var hmat: List[Float64]
    var state: List[Int64]

    def __init__(out self, space: Int, min_space: Int, vlen: Int, elen: Int):
        self.space = space
        self.min_space = min_space
        self.vlen = vlen
        self.elen = elen
        self.bufx = List[Float64](length=space * vlen, fill=0.0)
        self.bufe = List[Float64](length=space * elen, fill=0.0)
        self.hmat = List[Float64](length=(space + 1) * (space + 1), fill=0.0)
        self.state = List[Int64](length=2, fill=0)
        diis_init_hmat(space, list_ptr(self.hmat))

    def update(mut self, x: F64Ptr, xerr: F64Ptr, dst: F64Ptr) -> Int:
        var pstate = Pointer[Int64, MutAnyOrigin](unsafe_from_address=Int(self.state.unsafe_ptr()))
        return diis_update_buffers(
            self.space, self.min_space, self.vlen, self.elen,
            list_ptr(self.bufx), list_ptr(self.bufe), list_ptr(self.hmat), pstate, x, xerr, dst,
        )
