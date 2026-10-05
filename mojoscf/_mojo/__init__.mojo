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
from _mojo.linalg import F64Ptr, Blas, list_ptr, vnorm2diff, trace_prod, vdot
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
from _mojo.dfjk import df_jk_core, factorize_density, block_size
from _mojo.erijk import jk_s8_core
from _mojo.driver import scf_kernel, f64ptr
from _mojo.integrals import Basis, BoysTable, int1e_core, int1e_ip_core, eri_s8_core, int3c2e_core, int2c2e_core
from _mojo.directjk import DirectJK, basis_from_py, jk_ip1_core
from _mojo.gradients import grad2e_core, grad2c_core, df_grad_rhs, grad_df3c_core

comptime VERSION = "0.7.0"


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
        m.def_function[py_df_jk]("df_jk", docstring="df_jk(cderi, dms, orbs_or_None, ms_or_None, signs_or_None, vj, vk, with_j, with_k, block_mb, fact_tol, path, prefix, seq_path, seq_prefix).")
        m.def_function[py_jk_s8]("jk_s8", docstring="jk_s8(eri_s8, dms, vj, vk, with_j, with_k): J/K from 8-fold packed ERIs.")
        m.def_function[py_factorize_density]("factorize_density", docstring="factorize_density(dm, orb_out, sign_out, rel_tol, path, prefix) -> m.")
        m.def_function[scf_kernel]("scf_kernel", docstring="Native RHF/UHF SCF driver; see mojoscf.scf.kernel.")
        m.def_function[py_boys_table]("boys_table", docstring="boys_table(out): fill out (721 * 40 float64) with the Boys-function table.")
        m.def_function[py_int1e]("int1e", docstring="int1e(basis, s_out, t_out, v_out, table): overlap, kinetic and nuclear attraction matrices.")
        m.def_function[py_int2e_s8]("int2e_s8", docstring="int2e_s8(basis, eri_out, schwarz_tol, table): 8-fold packed electron repulsion integrals.")
        m.def_function[py_int3c2e]("int3c2e", docstring="int3c2e(basis, auxbasis, out, table): (ab|P) as a (naux, npair) array.")
        m.def_function[py_int2c2e]("int2c2e", docstring="int2c2e(auxbasis, out, table): (P|Q) as a dense (naux, naux) array.")
        m.def_function[py_int1e_ip]("int1e_ip", docstring="int1e_ip(basis, table, centers, charges, want_st, s_out, t_out, v_out): <nabla i|j>, <nabla i|T|j>, <nabla i|sum q/r|j>.")
        m.def_function[py_grad2e]("grad2e", docstring="grad2e(basis, table, dmj, dmk, jfac, kfac, tol, de): two-electron energy gradient (natm, 3).")
        m.def_function[py_int3c2e_block]("int3c2e_block", docstring="int3c2e_block(basis, auxbasis, table, s0, s1, out): (ab|P) for the auxiliary shells [s0, s1), (np, npair).")
        m.def_function[py_df_grad_rhs]("df_grad_rhs", docstring="df_grad_rhs(basis, auxbasis, table, dm_tril, orbs, blk, rho, q, seq_path, seq_prefix): fit right-hand sides of the DF gradient.")
        m.def_function[py_grad_df3c]("grad_df3c", docstring="grad_df3c(basis, auxbasis, table, coef, dpack, jfac, kfac, xs, cns, blk, tol, de, seq_path, seq_prefix): three-centre term of the DF gradient.")
        m.def_function[py_grad2c]("grad2c", docstring="grad2c(auxbasis, table, w, de): d/dR of -1/2 sum (P|Q) W_PQ.")
        m.def_function[py_jk_ip1]("jk_ip1", docstring="jk_ip1(basis, table, dms, vj, vk, with_j, with_k, tol): sum_kl (nabla i j|kl) D_lk and sum_jk (nabla i j|kl) D_jk.")
        m.def_function[py_direct_jk]("direct_jk", docstring="direct_jk(basis, table, dms, vj, vk, with_j, with_k, tol): integral-direct J/K of symmetric densities.")
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


def py_factorize_density(
    dm: PythonObject, orb: PythonObject, sign: PythonObject, rel_tol: PythonObject, path: PythonObject, prefix: PythonObject
) raises -> PythonObject:
    var blas = _blas(path, prefix)
    var nao = Int(py=dm.shape[0])
    var m = factorize_density(blas, nao, f64ptr(dm), f64ptr(orb), f64ptr(sign), Float64(py=rel_tol))
    return PythonObject(m)


def py_df_jk(
    cderi: PythonObject,
    dms: PythonObject,
    orbs: PythonObject,
    ms: PythonObject,
    signs: PythonObject,
    vj: PythonObject,
    vk: PythonObject,
    with_j: PythonObject,
    with_k: PythonObject,
    block_mb: PythonObject,
    fact_tol: PythonObject,
    path: PythonObject,
    prefix: PythonObject,
    seq_path: PythonObject,
    seq_prefix: PythonObject,
) raises -> PythonObject:
    """J and K of every density, with pyscf semantics: vj[s] = J(dms[s]), vk[s] = K(dms[s]).

    ``orbs`` (nset, nao, nao; column k of set s = weighted orbital, zero padded),
    ``ms`` (nset, int64) and ``signs`` (nset, nao) describe the factorisation of
    each density; pass None to let the kernel diagonalise the densities.
    """
    var blas = _blas(path, prefix)
    var blas_seq = _blas(seq_path, seq_prefix)
    var naux = Int(py=cderi.shape[0])
    var nset = Int(py=dms.shape[0])
    var nao = Int(py=dms.shape[1])
    var n2 = nao * nao
    var pdm = f64ptr(dms)
    var nj = nset if Bool(py=with_j) else 0
    var nk = nset if Bool(py=with_k) else 0
    var orb = List[Float64](length=max(nk, 1) * n2, fill=0.0)
    var sgn = List[Float64](length=max(nk, 1) * nao, fill=0.0)
    var mlist = List[Int64](length=max(nk, 1), fill=0)
    var porb = list_ptr(orb)
    var psgn = list_ptr(sgn)
    var pms = Pointer[Int64, MutAnyOrigin](unsafe_from_address=Int(mlist.unsafe_ptr()))
    var mmax = 0
    if nk > 0:
        if orbs is None:
            for s in range(nk):
                var m = factorize_density(
                    blas, nao, pdm.unsafe_offset(s * n2), porb.unsafe_offset(s * n2),
                    psgn.unsafe_offset(s * nao), Float64(py=fact_tol),
                )
                pms[unsafe_offset=s] = Int64(m)
                if m > mmax:
                    mmax = m
        else:
            var po = f64ptr(orbs)
            var psg = f64ptr(signs)
            var pm = Pointer[Int64, MutAnyOrigin](unsafe_from_address=Int(py=ms.__array_interface__["data"][0]))
            for s in range(nk):
                var m = Int(pm[unsafe_offset=s])
                pms[unsafe_offset=s] = Int64(m)
                if m > mmax:
                    mmax = m
                for i in range(nao):
                    for k in range(m):
                        porb[unsafe_offset=s * n2 + i * nao + k] = po[unsafe_offset=s * n2 + i * nao + k]
                for k in range(m):
                    psgn[unsafe_offset=s * nao + k] = psg[unsafe_offset=s * nao + k]
    var blk = block_size(naux, nao, nk, mmax, Int(py=block_mb) * 1024 * 1024)
    df_jk_core(blas, blas_seq, f64ptr(cderi), naux, nao, blk, nj, pdm, f64ptr(vj), nk, porb, n2, pms, psgn, f64ptr(vk))
    _ = orb^
    _ = sgn^
    _ = mlist^
    return PythonObject(None)


def py_jk_s8(
    eri: PythonObject, dms: PythonObject, vj: PythonObject, vk: PythonObject, with_j: PythonObject, with_k: PythonObject
) raises -> PythonObject:
    var nset = Int(py=dms.shape[0])
    var nao = Int(py=dms.shape[1])
    var nj = nset if Bool(py=with_j) else 0
    var nk = nset if Bool(py=with_k) else 0
    jk_s8_core(f64ptr(eri), nao, nj, f64ptr(dms), f64ptr(vj), nk, f64ptr(dms), f64ptr(vk))
    return PythonObject(None)


# --------------------------------------------------------------------------
# Integral engine
# --------------------------------------------------------------------------


def i64ptr(arr: PythonObject) raises -> Pointer[Int64, MutAnyOrigin]:
    return Pointer[Int64, MutAnyOrigin](unsafe_from_address=Int(py=arr.__array_interface__["data"][0]))


def _basis(b: PythonObject) raises -> Basis:
    """``b`` is the tuple (atm, bas, env, nf, c2s) prepared by mojoscf.integrals."""
    return basis_from_py(b)


def _boys(table: PythonObject) raises -> BoysTable:
    """The Boys table passed from Python (built once by ``boys_table``), or a fresh one for None."""
    if table is None:
        return BoysTable()
    return BoysTable(f64ptr(table))


def py_boys_table(dst: PythonObject) raises -> PythonObject:
    var boys = BoysTable()
    boys.write(f64ptr(dst))
    _ = boys^
    return PythonObject(None)


def py_int1e(basis: PythonObject, s: PythonObject, t: PythonObject, v: PythonObject, table: PythonObject) raises -> PythonObject:
    var bs = _basis(basis)
    var boys = _boys(table)
    int1e_core(bs, boys, f64ptr(s), f64ptr(t), f64ptr(v))
    _ = bs^
    _ = boys^
    return PythonObject(None)


def py_int2e_s8(basis: PythonObject, eri: PythonObject, schwarz_tol: PythonObject, table: PythonObject) raises -> PythonObject:
    var bs = _basis(basis)
    var boys = _boys(table)
    eri_s8_core(bs, boys, f64ptr(eri), Float64(py=schwarz_tol))
    _ = bs^
    _ = boys^
    return PythonObject(None)


def py_int3c2e(basis: PythonObject, auxbasis: PythonObject, dst: PythonObject, table: PythonObject) raises -> PythonObject:
    var bs = _basis(basis)
    var aux = _basis(auxbasis)
    var boys = _boys(table)
    int3c2e_core(bs, aux, boys, f64ptr(dst))
    _ = bs^
    _ = aux^
    _ = boys^
    return PythonObject(None)


def py_int2c2e(auxbasis: PythonObject, dst: PythonObject, table: PythonObject) raises -> PythonObject:
    var aux = _basis(auxbasis)
    var boys = _boys(table)
    int2c2e_core(aux, boys, f64ptr(dst))
    _ = aux^
    _ = boys^
    return PythonObject(None)


def py_direct_jk(
    basis: PythonObject, table: PythonObject, dms: PythonObject, vj: PythonObject, vk: PythonObject,
    with_j: PythonObject, with_k: PythonObject, tol: PythonObject,
) raises -> PythonObject:
    """vj[s] = J[dms[s]], vk[s] = K[dms[s]] for a stack of symmetric densities (outputs overwritten)."""
    var jk = DirectJK(_basis(basis), _boys(table))
    var nset = Int(py=dms.shape[0])
    var nao = Int(py=dms.shape[1])
    var n2 = nao * nao
    var pd = f64ptr(dms)
    var pj = f64ptr(vj)
    var pk = f64ptr(vk)
    var wj = Bool(py=with_j)
    var wk = Bool(py=with_k)
    var t = Float64(py=tol)
    for s in range(nset):
        jk.jk(
            1 if wj else 0, pd.unsafe_offset(s * n2), pj.unsafe_offset(s * n2),
            1 if wk else 0, pd.unsafe_offset(s * n2), pk.unsafe_offset(s * n2), t,
        )
    _ = jk^
    return PythonObject(None)


def py_int1e_ip(
    basis: PythonObject, table: PythonObject, centers: PythonObject, charges: PythonObject, want_st: PythonObject,
    s_out: PythonObject, t_out: PythonObject, v_out: PythonObject,
) raises -> PythonObject:
    var bs = _basis(basis)
    var boys = _boys(table)
    int1e_ip_core(
        bs, boys, Int(py=charges.shape[0]), f64ptr(centers), f64ptr(charges), Bool(py=want_st),
        f64ptr(s_out), f64ptr(t_out), f64ptr(v_out),
    )
    _ = bs^
    _ = boys^
    return PythonObject(None)


def py_jk_ip1(
    basis: PythonObject, table: PythonObject, dms: PythonObject, vj: PythonObject, vk: PythonObject,
    with_j: PythonObject, with_k: PythonObject, tol: PythonObject,
) raises -> PythonObject:
    jk_ip1_core(
        _basis(basis), _boys(table), Int(py=dms.shape[0]), f64ptr(dms), f64ptr(vj), f64ptr(vk),
        Bool(py=with_j), Bool(py=with_k), Float64(py=tol),
    )
    return PythonObject(None)


def py_grad2e(
    basis: PythonObject, table: PythonObject, dmj: PythonObject, dmk: PythonObject, jfac: PythonObject,
    kfac: PythonObject, tol: PythonObject, de: PythonObject,
) raises -> PythonObject:
    grad2e_core(
        _basis(basis), _boys(table), f64ptr(dmj), Int(py=dmk.shape[0]), f64ptr(dmk), Float64(py=jfac),
        Float64(py=kfac), Float64(py=tol), f64ptr(de),
    )
    return PythonObject(None)


def py_int3c2e_block(
    basis: PythonObject, auxbasis: PythonObject, table: PythonObject, s0: PythonObject, s1: PythonObject,
    dst: PythonObject,
) raises -> PythonObject:
    var bs = _basis(basis)
    var aux = _basis(auxbasis)
    var boys = _boys(table)
    int3c2e_core(bs, aux, boys, f64ptr(dst), Int(py=s0), Int(py=s1))
    _ = bs^
    _ = aux^
    _ = boys^
    return PythonObject(None)


def py_df_grad_rhs(
    basis: PythonObject, auxbasis: PythonObject, table: PythonObject, dm_tril: PythonObject, orbs: PythonObject,
    blk: PythonObject, rho: PythonObject, q: PythonObject, seq_path: PythonObject, seq_prefix: PythonObject,
) raises -> PythonObject:
    df_grad_rhs(
        _blas(seq_path, seq_prefix), _basis(basis), _basis(auxbasis), _boys(table), f64ptr(dm_tril),
        Int(py=orbs.shape[0]), Int(py=orbs.shape[2]), f64ptr(orbs), Int(py=blk), f64ptr(rho), f64ptr(q),
    )
    return PythonObject(None)


def py_grad_df3c(
    basis: PythonObject, auxbasis: PythonObject, table: PythonObject, coef: PythonObject, dpack: PythonObject,
    jfac: PythonObject, kfac: PythonObject, xs: PythonObject, cns: PythonObject, blk: PythonObject,
    tol: PythonObject, de: PythonObject, seq_path: PythonObject, seq_prefix: PythonObject,
) raises -> PythonObject:
    grad_df3c_core(
        _blas(seq_path, seq_prefix), _basis(basis), _basis(auxbasis), _boys(table), f64ptr(coef), f64ptr(dpack),
        Float64(py=jfac), Float64(py=kfac), Int(py=cns.shape[0]), Int(py=cns.shape[2]), f64ptr(xs), f64ptr(cns),
        Int(py=blk), Float64(py=tol), f64ptr(de),
    )
    return PythonObject(None)


def py_grad2c(auxbasis: PythonObject, table: PythonObject, w: PythonObject, de: PythonObject) raises -> PythonObject:
    grad2c_core(_basis(auxbasis), _boys(table), f64ptr(w), Int(py=de.shape[0]), f64ptr(de))
    return PythonObject(None)
