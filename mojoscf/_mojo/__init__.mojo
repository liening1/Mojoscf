"""Python extension module ``mojoscf._mojoscf``.

Thin bindings that unpack NumPy arrays into raw pointers and call the Mojo
kernels.  All array arguments must be C-contiguous float64 arrays of the
documented shapes; the Python wrappers in ``mojoscf`` take care of that.
"""
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.memory import Pointer
from std.math import sqrt
from std.os import abort
from std.runtime import initialize_runtime, parallelism_level
from std.sys import simd_width_of
from _mojo.linalg import F64Ptr, Blas, vnorm2diff, trace_prod, vdot
from _mojo.kernels import (
    make_rdm1,
    get_occ,
    get_grad,
    damping,
    level_shift,
    diis_errvec,
    eigh_fock,
    jk_dense,
)
from _mojo.diis import diis_update_buffers, diis_init_hmat
from _mojo.driver import scf_kernel, f64ptr

comptime VERSION = "0.2.0"


@export
def PyInit__mojoscf() abi("C") -> PythonObject:
    try:
        initialize_runtime()
        var m = PythonModuleBuilder("_mojoscf")
        m.def_function[py_version]("version", docstring="Version of the Mojo kernels.")
        m.def_function[py_runtime_info]("runtime_info", docstring="(parallelism level, float64 SIMD width).")
        m.def_function[py_blas_probe]("blas_probe", docstring="blas_probe(path, prefix) -> bool: can dgemm_/dsygvd_/dsyevd_ be loaded?")
        m.def_function[py_gemm]("gemm", docstring="gemm(a, b, c, transa, transb, alpha, beta, path, prefix): row-major C = alpha op(A) op(B) + beta C.")
        m.def_function[py_eigh]("eigh", docstring="eigh(h, s_or_None, x_or_None, w_out, c_out, path, prefix): symmetric (generalised) eigensolver with pyscf phases.")
        m.def_function[py_make_rdm1]("make_rdm1", docstring="make_rdm1(mo_coeff, mo_occ, dm_out, path, prefix).")
        m.def_function[py_trace_prod]("trace_prod", docstring="trace_prod(a, b) -> sum_ij a[i,j] b[j,i].")
        m.def_function[py_energy_elec]("energy_elec", docstring="energy_elec(h1e, vhf, dm) -> (e1, e2) with e2 = tr(vhf dm)/2.")
        m.def_function[py_get_occ]("get_occ", docstring="get_occ(mo_energy, nocc, mo_occ_out, occ_value) -> (homo, lumo) or None.")
        m.def_function[py_get_grad]("get_grad", docstring="get_grad(mo_coeff, mo_occ, fock, g_out, prefactor, path, prefix) -> gradient length.")
        m.def_function[py_damping]("damping", docstring="damping(f, f_prev, factor, dst).")
        m.def_function[py_level_shift]("level_shift", docstring="level_shift(s, dm, f, factor, dst, dm_scale, path, prefix): dst = f + factor (s - s (dm_scale dm) s).")
        m.def_function[py_diis_errvec]("diis_errvec", docstring="diis_errvec(s, dm, f, x_or_None, err_out, path, prefix) -> length.")
        m.def_function[py_diis_init]("diis_init", docstring="diis_init(hmat, state): reset DIIS buffers.")
        m.def_function[py_diis_update]("diis_update", docstring="diis_update(x, xerr, bufx, bufe, hmat, state, space, min_space, dst) -> nd.")
        m.def_function[py_norm_diff]("norm_diff", docstring="norm_diff(a, b) -> ||a - b||_F.")
        m.def_function[py_jk_dense]("jk_dense", docstring="jk_dense(eri, dm, vj_out, vk_out) from a full (n,n,n,n) ERI tensor.")
        m.def_function[scf_kernel]("scf_kernel", docstring="Native RHF/UHF SCF driver; see mojoscf.scf.kernel.")
        return m.finalize()
    except e:
        abort(String("error creating the mojoscf._mojoscf module: ", e))


def _blas(path: PythonObject, prefix: PythonObject) raises -> Blas:
    return Blas(String(path), String(prefix))


def py_version() raises -> PythonObject:
    return PythonObject(VERSION)


def py_runtime_info() raises -> PythonObject:
    return Python.tuple(PythonObject(parallelism_level()), PythonObject(simd_width_of[DType.float64]()))


def py_blas_probe(path: PythonObject, prefix: PythonObject) raises -> PythonObject:
    var b = Blas(String(path), String(prefix), verify=True)
    return PythonObject(b.available())


def py_gemm(
    a: PythonObject,
    b: PythonObject,
    c: PythonObject,
    transa: PythonObject,
    transb: PythonObject,
    alpha: PythonObject,
    beta: PythonObject,
    path: PythonObject,
    prefix: PythonObject,
) raises -> PythonObject:
    var ta = Bool(py=transa)
    var tb = Bool(py=transb)
    var m = Int(py=c.shape[0])
    var n = Int(py=c.shape[1])
    var k = Int(py=a.shape[0]) if ta else Int(py=a.shape[1])
    var kb = Int(py=b.shape[1]) if tb else Int(py=b.shape[0])
    if k != kb:
        raise Error("gemm: inner dimensions do not match")
    var blas = _blas(path, prefix)
    blas.gemm(ta, tb, m, n, k, Float64(py=alpha), f64ptr(a), f64ptr(b), Float64(py=beta), f64ptr(c))
    return PythonObject(None)


def py_eigh(
    h: PythonObject,
    s: PythonObject,
    x: PythonObject,
    w: PythonObject,
    c: PythonObject,
    path: PythonObject,
    prefix: PythonObject,
) raises -> PythonObject:
    var blas = _blas(path, prefix)
    var nao = Int(py=h.shape[0])
    if not (x is None):
        var nmo = Int(py=x.shape[1])
        eigh_fock(blas, nao, nmo, f64ptr(h), f64ptr(h), True, f64ptr(x), f64ptr(w), f64ptr(c))
    elif s is None:
        blas.eigh(nao, f64ptr(h), f64ptr(w), f64ptr(c))
    else:
        blas.eigh_gen(nao, f64ptr(h), f64ptr(s), f64ptr(w), f64ptr(c))
    return PythonObject(None)


def py_make_rdm1(
    mo_coeff: PythonObject, mo_occ: PythonObject, dm: PythonObject, path: PythonObject, prefix: PythonObject
) raises -> PythonObject:
    var blas = _blas(path, prefix)
    var nao = Int(py=mo_coeff.shape[0])
    var nmo = Int(py=mo_coeff.shape[1])
    make_rdm1(blas, nao, nmo, f64ptr(mo_coeff), f64ptr(mo_occ), f64ptr(dm))
    return PythonObject(None)


def py_trace_prod(a: PythonObject, b: PythonObject) raises -> PythonObject:
    var n = Int(py=a.shape[0])
    return PythonObject(trace_prod(f64ptr(a), f64ptr(b), n))


def py_energy_elec(h1e: PythonObject, vhf: PythonObject, dm: PythonObject) raises -> PythonObject:
    var n = Int(py=dm.shape[0])
    var pdm = f64ptr(dm)
    var e1 = trace_prod(f64ptr(h1e), pdm, n)
    var e2 = 0.5 * trace_prod(f64ptr(vhf), pdm, n)
    return Python.tuple(PythonObject(e1), PythonObject(e2))


def py_get_occ(
    mo_energy: PythonObject, nocc: PythonObject, mo_occ: PythonObject, occ_value: PythonObject
) raises -> PythonObject:
    var nmo = Int(py=mo_energy.shape[0])
    var hl = List[Float64](length=2, fill=0.0)
    var phl = F64Ptr(unsafe_from_address=Int(hl.unsafe_ptr()))
    var ok = get_occ(nmo, f64ptr(mo_energy), Int(py=nocc), f64ptr(mo_occ), phl, Float64(py=occ_value))
    var result = PythonObject(None)
    if ok:
        result = Python.tuple(PythonObject(hl[0]), PythonObject(hl[1]))
    _ = hl^
    return result


def py_get_grad(
    mo_coeff: PythonObject,
    mo_occ: PythonObject,
    fock: PythonObject,
    g: PythonObject,
    prefactor: PythonObject,
    path: PythonObject,
    prefix: PythonObject,
) raises -> PythonObject:
    var blas = _blas(path, prefix)
    var nao = Int(py=mo_coeff.shape[0])
    var nmo = Int(py=mo_coeff.shape[1])
    var ng = get_grad(
        blas, nao, nmo, f64ptr(mo_coeff), f64ptr(mo_occ), f64ptr(fock), f64ptr(g), Float64(py=prefactor)
    )
    return PythonObject(ng)


def py_damping(f: PythonObject, f_prev: PythonObject, factor: PythonObject, dst: PythonObject) raises -> PythonObject:
    var n2 = Int(py=f.size)
    damping(n2, f64ptr(f), f64ptr(f_prev), Float64(py=factor), f64ptr(dst))
    return PythonObject(None)


def py_level_shift(
    s: PythonObject,
    dm: PythonObject,
    f: PythonObject,
    factor: PythonObject,
    dst: PythonObject,
    dm_scale: PythonObject,
    path: PythonObject,
    prefix: PythonObject,
) raises -> PythonObject:
    var blas = _blas(path, prefix)
    var nao = Int(py=s.shape[0])
    level_shift(blas, nao, f64ptr(s), f64ptr(dm), f64ptr(f), Float64(py=factor), f64ptr(dst), Float64(py=dm_scale))
    return PythonObject(None)


def py_diis_errvec(
    s: PythonObject,
    dm: PythonObject,
    f: PythonObject,
    x: PythonObject,
    err: PythonObject,
    path: PythonObject,
    prefix: PythonObject,
) raises -> PythonObject:
    var blas = _blas(path, prefix)
    var nao = Int(py=s.shape[0])
    var has_x = not (x is None)
    var nmo = nao
    var px = f64ptr(s)
    if has_x:
        nmo = Int(py=x.shape[1])
        px = f64ptr(x)
    var n = diis_errvec(blas, nao, nmo, f64ptr(s), f64ptr(dm), f64ptr(f), has_x, px, f64ptr(err))
    return PythonObject(n)


def py_diis_init(hmat: PythonObject, state: PythonObject) raises -> PythonObject:
    var sp1 = Int(py=hmat.shape[0])
    diis_init_hmat(sp1 - 1, f64ptr(hmat))
    var pstate = Pointer[Int64, MutAnyOrigin](unsafe_from_address=Int(py=state.ctypes.data))
    pstate[unsafe_offset=0] = 0
    pstate[unsafe_offset=1] = 0
    return PythonObject(None)


def py_diis_update(
    x: PythonObject,
    xerr: PythonObject,
    bufx: PythonObject,
    bufe: PythonObject,
    hmat: PythonObject,
    state: PythonObject,
    space: PythonObject,
    min_space: PythonObject,
    dst: PythonObject,
) raises -> PythonObject:
    var vlen = Int(py=x.size)
    var elen = Int(py=xerr.size)
    var pstate = Pointer[Int64, MutAnyOrigin](unsafe_from_address=Int(py=state.ctypes.data))
    var nd = diis_update_buffers(
        Int(py=space), Int(py=min_space), vlen, elen,
        f64ptr(bufx), f64ptr(bufe), f64ptr(hmat), pstate, f64ptr(x), f64ptr(xerr), f64ptr(dst),
    )
    return PythonObject(nd)


def py_norm_diff(a: PythonObject, b: PythonObject) raises -> PythonObject:
    var n = Int(py=a.size)
    return PythonObject(sqrt(vnorm2diff(f64ptr(a), f64ptr(b), n)))


def py_jk_dense(eri: PythonObject, dm: PythonObject, vj: PythonObject, vk: PythonObject) raises -> PythonObject:
    var nao = Int(py=dm.shape[0])
    jk_dense(nao, f64ptr(eri), f64ptr(dm), f64ptr(vj), f64ptr(vk))
    return PythonObject(None)
