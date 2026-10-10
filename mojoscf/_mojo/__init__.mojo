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
from _mojo.dfmo import df_mo_core, df_sandwich_core, cphf_k_core
from _mojo.erijk import jk_s8_core
from _mojo.driver import scf_kernel, f64ptr
from _mojo.integrals import Basis, BoysTable, int1e_core, int1e_ip_core, int1e_iprinv_dm_core, eri_s8_core, int3c2e_core, int2c2e_core
from _mojo.directjk import DirectJK, basis_from_py, jk_ip1_core, h1_2e_core
from _mojo.gradients import grad2e_core, grad2e_pairs_core, hess2e_core, grad2c_core, df_grad_rhs, grad_df3c_core, int3c2e_ip1_core, hess_df3c_core
from _mojo.qmmm import mm_potential_core, mm_grad_core, mm_esp_core
from _mojo.pcm import pcm_ds_core, pcm_pair_core
from _mojo.numint import eval_ao_core, xc_rho_core, xc_vmat_core, xc_grad_core, xc_grad_dm_core, xc_fxc_core, xc_hess_core, xc_h1_core
from _mojo.pdft import ontop_pi_core, ontop_grad_core, ontop_paaa_core, ontop_density_core
from _mojo.grids import becke_response_core

comptime VERSION = "0.12.0"


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
        m.def_function[py_df_mo]("df_mo", docstring="df_mo(cderi, cl, cr, out, seq_path, seq_prefix): out[Q] = cl^T E_Q cr for the packed DF tensor (naux, npair), out (naux, nl, nr).")
        m.def_function[py_df_sandwich]("df_sandwich", docstring="df_sandwich(a, x, b, nvec, alpha, r, seq_path, seq_prefix, a_qstride, b_qstride): r (m*nvec, p) += alpha sum_Q reshape(a[Q] x, (m*nvec, k2)) b[Q] for a (nq, m, k1), x (k1, nvec*k2), b (nq, k2, p); a[Q], b[Q] start a_qstride, b_qstride elements apart.")
        m.def_function[py_cphf_k]("cphf_k", docstring="cphf_k(lfull, lmo, loo, xs, xts, alpha, r, seq_path, seq_prefix): r (nmo, nset, nocc) += alpha sum_Q L_Q (x (oo|Q) + E_o x^T (po|Q)) with xs (nmo, nset, nocc), the MO-basis exchange of the orbital-Hessian response.")
        m.def_function[py_jk_s8]("jk_s8", docstring="jk_s8(eri_s8, dms, vj, vk, with_j, with_k, nanti): J/K from 8-fold packed ERIs (the last nanti densities antisymmetric: K only).")
        m.def_function[py_factorize_density]("factorize_density", docstring="factorize_density(dm, orb_out, sign_out, rel_tol, path, prefix) -> m.")
        m.def_function[scf_kernel]("scf_kernel", docstring="Native RHF/UHF SCF driver; see mojoscf.scf.kernel.")
        m.def_function[py_boys_table]("boys_table", docstring="boys_table(out): fill out (721 * 40 float64) with the Boys-function table.")
        m.def_function[py_int1e]("int1e", docstring="int1e(basis, s_out, t_out, v_out, table): overlap, kinetic and nuclear attraction matrices.")
        m.def_function[py_int2e_s8]("int2e_s8", docstring="int2e_s8(basis, eri_out, schwarz_tol, table, omega): 8-fold packed electron repulsion integrals (omega > 0: erf(omega r) / r).")
        m.def_function[py_int3c2e]("int3c2e", docstring="int3c2e(basis, auxbasis, out, table, omega): (ab|P) as a (naux, npair) array; omega > 0: erf(omega r)/r.")
        m.def_function[py_int2c2e]("int2c2e", docstring="int2c2e(auxbasis, out, table, omega): (P|Q) as a dense (naux, naux) array; omega > 0: erf(omega r)/r.")
        m.def_function[py_int1e_ip]("int1e_ip", docstring="int1e_ip(basis, table, centers, charges, want_st, s_out, t_out, v_out): <nabla i|j>, <nabla i|T|j>, <nabla i|sum q/r|j>.")
        m.def_function[py_grad2e_pairs]("grad2e_pairs", docstring="grad2e_pairs(basis, table, jl, jr, jc, kl, kr, kc, tol, de, omega): d/dR of sum_p jc_p sum (ij|kl) L_ij R_kl + sum_q kc_q sum (ij|kl) A_jk B_il (natm, 3).")
        m.def_function[py_hess2e]("hess2e", docstring="hess2e(basis, table, dmj, dmk, jfac, kfac, tol, hess, omega): second derivatives (natm, natm, 3, 3) of the two-electron energy at fixed densities.")
        m.def_function[py_grad2e]("grad2e", docstring="grad2e(basis, table, dmj, dmk, jfac, kfac, tol, de, omega): two-electron energy gradient (natm, 3) (omega > 0: erf(omega r) / r).")
        m.def_function[py_int3c2e_block]("int3c2e_block", docstring="int3c2e_block(basis, auxbasis, table, s0, s1, out): (ab|P) for the auxiliary shells [s0, s1), (np, npair).")
        m.def_function[py_int3c2e_cols]("int3c2e_cols", docstring="int3c2e_cols(basis, auxbasis, table, a0, a1, out, omega): (ab|P) for the AO pairs (i, j <= i) with i in the AO shells [a0, a1): the pack_tril columns [c0, c1) as (naux, c1 - c0).")
        m.def_function[py_df_grad_rhs]("df_grad_rhs", docstring="df_grad_rhs(basis, auxbasis, table, dm_tril, orbs, blk, rho, q, seq_path, seq_prefix, omega): fit right-hand sides of the DF gradient.")
        m.def_function[py_grad_df3c]("grad_df3c", docstring="grad_df3c(basis, auxbasis, table, coef, dpack, jfac, kfac, xs, cns, blk, tol, de, seq_path, seq_prefix, omega): three-centre term of the DF gradient.")
        m.def_function[py_grad2c]("grad2c", docstring="grad2c(auxbasis, table, w, de, omega): d/dR of -1/2 sum (P|Q) W_PQ.")
        m.def_function[py_int3c2e_ip1]("int3c2e_ip1", docstring="int3c2e_ip1(basis, auxbasis, table, ps0, ps1, tol, dst, omega): (nabla mu nu|P) for auxiliary shells [ps0, ps1) as (3, np, nao, nao); omega > 0: erf(omega r)/r.")
        m.def_function[py_hess_df3c]("hess_df3c", docstring="hess_df3c(basis, auxbasis, table, coef, dpack, jfac, kfac, xs, cns, blk, tol, hess, seq_path, seq_prefix, omega): second-derivative three-centre term of the DF Hessian.")
        m.def_function[py_eval_ao]("eval_ao", docstring="eval_ao(basis, coords, deriv, out): AO values and derivatives (deriv <= 3) on the points, out (ncomp, ngrid, nao).")
        m.def_function[py_xc_rho]("xc_rho", docstring="xc_rho(basis, coords, kind, dms, orbs, occs, rho, seq_path, seq_prefix): densities (kind 0), with gradients (1), and tau (2: meta-GGA) of symmetric dms (nset, nao, nao), optionally also given as orbitals orbs (nset, nao, norb) with occupations occs (nset, norb; norb may be 0), into rho (nset, ncomp, ngrid).")
        m.def_function[py_xc_vmat]("xc_vmat", docstring="xc_vmat(basis, coords, kind, wv, vmat, seq_path, seq_prefix): sum_p phi(p) (sum_c wv_c(p) phi_c(p))^T (+ the tau term for kind 2) into vmat (nset, nao, nao).")
        m.def_function[py_xc_fxc]("xc_fxc", docstring="xc_fxc(basis, coords, weights, kind, fxc, dms, lfac, rfac, project, vmat, seq_path, seq_prefix): XC kernel fxc ((nspin nvar)^2, ngrid) contracted with the densities of the symmetric dms (nspin, nset, nao, nao), optionally also given as L R^T factors lfac (nspin, nset, nao, rank) and rfac (nspin, nao, rank), into vmat (nspin, nset, nao, nao), not symmetrised; with project, vmat (nspin, nset, nao, rank) = (V + V^T) R.")
        m.def_function[py_xc_hess]("xc_hess", docstring="xc_hess(basis, coords, weights, kind, dms, vxc, fxc, aoatm, de2, seq_path, seq_prefix): XC part of the nuclear Hessian at fixed density (pyscf's _get_vxc_diag + _get_vxc_deriv2 contracted), LDA/GGA/meta-GGA (kind 0/1/2), dms (nspin, nao, nao), de2 (natm, natm, 3, 3).")
        m.def_function[py_xc_h1]("xc_h1", docstring="xc_h1(basis, coords, weights, kind, dms, fxc, aoatm, cmo, nocc, h1, seq_path, seq_prefix): kernel part C^T (F + F^T) C_o of the XC first-derivative Fock matrices of the Hessian, h1 (nspin, natm, 3, nmo, nocc).")
        m.def_function[py_xc_grad]("xc_grad", docstring="xc_grad(basis, coords, kind, wv, vmat, seq_path, seq_prefix): XC gradient matrices (nset, 3, nao, nao) of pyscf's grad.rks.get_vxc (LDA 0, GGA 1, meta-GGA 2), before its sign flip.")
        m.def_function[py_xc_grad_dm]("xc_grad_dm", docstring="xc_grad_dm(basis, coords, kind, wv, dms, orbs, occs, de, seq_path, seq_prefix): XC term of the nuclear gradient (natm, 3), the XC gradient matrices contracted with the densities (optionally also given as orbitals, norb may be 0).")
        m.def_function[py_ontop_pi]("ontop_pi", docstring="ontop_pi(basis, coords, deriv, mo_cas, lt, pi, seq_path, seq_prefix): cumulant part of the MC-PDFT on-top pair density (deriv 0) and its gradient (deriv 1) into pi (1 or 4, ngrid), active orbitals mo_cas (nao, ncas), lt (ncas^2, ncas^2) the transposed 2-RDM cumulant.")
        m.def_function[py_ontop_grad]("ontop_grad", docstring="ontop_grad(basis, coords, wpi, mo_cas, lmat, de, seq_path, seq_prefix): on-top pair-density term of the MC-PDFT nuclear gradient at fixed grids (natm, 3) for weighted v_Pi wpi (ngrid), active orbitals mo_cas (nao, ncas), lmat (ncas^2, ncas^2) the 2-RDM cumulant.")
        m.def_function[py_ontop_paaa]("ontop_paaa", docstring="ontop_paaa(basis, coords, wpi, mo_cas, paaa, seq_path, seq_prefix): paaa (nao, ncas^3) = sum_p phi_mu wpi psi_u psi_v psi_w, the AO-active-active-active block of the MC-PDFT on-top potential.")
        m.def_function[py_ontop_density]("ontop_density", docstring="ontop_density(basis, coords, deriv, has_core, dm_core, mo_cas, casdm1s, lt, pi_deriv, rho, pi, rho_core, seq_path, seq_prefix): spin densities rho (2, ncomp, ngrid) of Dc/2 + C casdm1s[s] C^T, the on-top pair density pi (1 or 4, ngrid) and the core density rho_core (ngrid), in one pass.")
        m.def_function[py_becke_response]("becke_response", docstring="becke_response(atm, adj, use_adj, scheme, coords, vol, owner, eot, mode, w0, de): Becke weights w0 of the grid of atom owner and, for mode 1, de (natm, 3) = sum_r eot(r) dw(r)/dR, as pyscf's grids_response_becke (scheme 0 original Becke, 1 Stratmann; adj the radii adjustment when use_adj).")
        m.def_function[py_mm_potential]("mm_potential", docstring="mm_potential(basis, table, coords, weights, zetas, point, out): sum_k w_k (ij|k) for point or unit Gaussian charges (nao, nao).")
        m.def_function[py_pcm_ds]("pcm_ds", docstring="pcm_ds(table, coords, zeta, switch, norm, rvdw, with_d, s, d): pyscf's PCM S (and D) matrices (n, n).")
        m.def_function[py_pcm_pair]("pcm_pair", docstring="pcm_pair(table, coords, zeta, norm, kind, a, b, g): G_p = a_p sum_j dX_pj b_j - b_p sum_i a_i dX_ip for X = S (kind 0) or D (1), g (n, 3).")
        m.def_function[py_mm_esp]("mm_esp", docstring="mm_esp(basis, table, coords, zetas, point, dms, out): sum_ij D_s,ij (ij|k) for symmetric dms (nset, nao, nao) at point or unit Gaussian charges, out (nset, nch).")
        m.def_function[py_mm_grad]("mm_grad", docstring="mm_grad(basis, table, coords, weights, zetas, point, dm, mat, forces, atoms): sum_k w_k (nabla i j|k) (3, nao, nao), sum_ij D_ij w_k (ij|nabla k) (nch, 3), 2 sum_{i on A} D_ij sum_k w_k (nabla i j|k) (natm, 3); an empty output is skipped.")
        m.def_function[py_int1e_iprinv_dm]("int1e_iprinv_dm", docstring="int1e_iprinv_dm(basis, table, centers, dm, out): sum_ij D_ij <nabla i|1/|r-R_c||j> per centre (ncenter, 3).")
        m.def_function[py_jk_ip1]("jk_ip1", docstring="jk_ip1(basis, table, dms, vj, vk, with_j, with_k, tol, omega): sum_kl (nabla i j|kl) D_lk and sum_jk (nabla i j|kl) D_jk (omega > 0: erf(omega r) / r).")
        m.def_function[py_h1_2e]("h1_2e", docstring="h1_2e(basis, table, dmj, dmk, tol, out, omega): d/dR_Ax J[Dj_s] and K[Dk_s] for every atom and direction, (natm, 3, nj + nk, nao, nao).")
        m.def_function[py_direct_jk]("direct_jk", docstring="direct_jk(basis, table, dms, vj, vk, with_j, with_k, tol, nanti, omega): integral-direct J/K in one pass (the last nanti densities antisymmetric: K only; omega > 0: erf(omega r) / r).")
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


def py_df_mo(
    cderi: PythonObject, cl: PythonObject, cr: PythonObject, dst: PythonObject, seq_path: PythonObject,
    seq_prefix: PythonObject,
) raises -> PythonObject:
    var blas_seq = _blas(seq_path, seq_prefix)
    df_mo_core(
        blas_seq, f64ptr(cderi), Int(py=cderi.shape[0]), Int(py=cl.shape[0]), f64ptr(cl), Int(py=cl.shape[1]),
        f64ptr(cr), Int(py=cr.shape[1]), f64ptr(dst),
    )
    return PythonObject(None)


def py_df_sandwich(
    a: PythonObject, x: PythonObject, b: PythonObject, nvec: PythonObject, alpha: PythonObject, r: PythonObject,
    seq_path: PythonObject, seq_prefix: PythonObject, a_qstride: PythonObject, b_qstride: PythonObject,
) raises -> PythonObject:
    var blas_seq = _blas(seq_path, seq_prefix)
    var nv = Int(py=nvec)
    df_sandwich_core(
        blas_seq, Int(py=a.shape[0]), f64ptr(a), Int(py=a.shape[1]), Int(py=a.shape[2]), f64ptr(x), nv,
        Int(py=b.shape[1]), f64ptr(b), Int(py=b.shape[2]), Float64(py=alpha), f64ptr(r),
        Int(py=a_qstride), Int(py=b_qstride),
    )
    return PythonObject(None)


def py_cphf_k(
    lfull: PythonObject, lmo: PythonObject, loo: PythonObject, xs: PythonObject, xts: PythonObject,
    alpha: PythonObject, r: PythonObject, seq_path: PythonObject, seq_prefix: PythonObject,
) raises -> PythonObject:
    var blas_seq = _blas(seq_path, seq_prefix)
    cphf_k_core(
        blas_seq, Int(py=lfull.shape[0]), Int(py=lfull.shape[1]), Int(py=loo.shape[1]), Int(py=xs.shape[1]),
        f64ptr(lfull), f64ptr(lmo), f64ptr(loo), f64ptr(xs), f64ptr(xts), Float64(py=alpha), f64ptr(r),
    )
    return PythonObject(None)


def py_jk_s8(
    eri: PythonObject, dms: PythonObject, vj: PythonObject, vk: PythonObject, with_j: PythonObject, with_k: PythonObject,
    nanti: PythonObject,
) raises -> PythonObject:
    """J of the first nset - nanti densities, K of all (the last nanti antisymmetric)."""
    var nset = Int(py=dms.shape[0])
    var nao = Int(py=dms.shape[1])
    var na = Int(py=nanti)
    var nj = nset - na if Bool(py=with_j) else 0
    var nk = nset if Bool(py=with_k) else 0
    jk_s8_core(f64ptr(eri), nao, nj, f64ptr(dms), f64ptr(vj), nk, f64ptr(dms), f64ptr(vk), na if nk > 0 else 0)
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


def py_int2e_s8(
    basis: PythonObject, eri: PythonObject, schwarz_tol: PythonObject, table: PythonObject, omega: PythonObject
) raises -> PythonObject:
    var bs = _basis(basis)
    var boys = _boys(table)
    eri_s8_core(bs, boys, f64ptr(eri), Float64(py=schwarz_tol), Float64(py=omega))
    _ = bs^
    _ = boys^
    return PythonObject(None)


def py_int3c2e(
    basis: PythonObject, auxbasis: PythonObject, dst: PythonObject, table: PythonObject, omega: PythonObject
) raises -> PythonObject:
    var bs = _basis(basis)
    var aux = _basis(auxbasis)
    var boys = _boys(table)
    int3c2e_core(bs, aux, boys, f64ptr(dst), 0, -1, Float64(py=omega))
    _ = bs^
    _ = aux^
    _ = boys^
    return PythonObject(None)


def py_int2c2e(auxbasis: PythonObject, dst: PythonObject, table: PythonObject, omega: PythonObject) raises -> PythonObject:
    var aux = _basis(auxbasis)
    var boys = _boys(table)
    int2c2e_core(aux, boys, f64ptr(dst), Float64(py=omega))
    _ = aux^
    _ = boys^
    return PythonObject(None)


def py_h1_2e(
    basis: PythonObject, table: PythonObject, dmj: PythonObject, dmk: PythonObject, tol: PythonObject,
    dst: PythonObject, omega: PythonObject,
) raises -> PythonObject:
    h1_2e_core(
        _basis(basis), _boys(table), Int(py=dmj.shape[0]), f64ptr(dmj), Int(py=dmk.shape[0]), f64ptr(dmk),
        Float64(py=tol), f64ptr(dst), Float64(py=omega),
    )
    return PythonObject(None)


def py_direct_jk(
    basis: PythonObject, table: PythonObject, dms: PythonObject, vj: PythonObject, vk: PythonObject,
    with_j: PythonObject, with_k: PythonObject, tol: PythonObject, nanti: PythonObject, omega: PythonObject,
) raises -> PythonObject:
    """vj[s] = J[dms[s]] for the first nset - nanti (symmetric) densities, vk[s] = K[dms[s]] for all (the last
    nanti antisymmetric), in one pass over the integrals (outputs overwritten); omega > 0: erf(omega r) / r."""
    var jk = DirectJK(_basis(basis), _boys(table), Float64(py=omega))
    var nset = Int(py=dms.shape[0])
    var na = Int(py=nanti)
    var nj = nset - na if Bool(py=with_j) else 0
    var nk = nset if Bool(py=with_k) else 0
    jk.jk(nj, f64ptr(dms), f64ptr(vj), nk, f64ptr(dms), f64ptr(vk), Float64(py=tol), na if nk > 0 else 0)
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
    with_j: PythonObject, with_k: PythonObject, tol: PythonObject, omega: PythonObject,
) raises -> PythonObject:
    jk_ip1_core(
        _basis(basis), _boys(table), Int(py=dms.shape[0]), f64ptr(dms), f64ptr(vj), f64ptr(vk),
        Bool(py=with_j), Bool(py=with_k), Float64(py=tol), Float64(py=omega),
    )
    return PythonObject(None)


def py_grad2e_pairs(
    basis: PythonObject, table: PythonObject, jl: PythonObject, jr: PythonObject, jc: PythonObject,
    kl: PythonObject, kr: PythonObject, kc: PythonObject, tol: PythonObject, de: PythonObject, omega: PythonObject,
) raises -> PythonObject:
    grad2e_pairs_core(
        _basis(basis), _boys(table), Int(py=jc.shape[0]), f64ptr(jl), f64ptr(jr), f64ptr(jc),
        Int(py=kc.shape[0]), f64ptr(kl), f64ptr(kr), f64ptr(kc), Float64(py=tol), f64ptr(de), Float64(py=omega),
    )
    return PythonObject(None)


def py_hess2e(
    basis: PythonObject, table: PythonObject, dmj: PythonObject, dmk: PythonObject, jfac: PythonObject,
    kfac: PythonObject, tol: PythonObject, hess: PythonObject, omega: PythonObject,
) raises -> PythonObject:
    hess2e_core(
        _basis(basis), _boys(table), f64ptr(dmj), Int(py=dmk.shape[0]), f64ptr(dmk), Float64(py=jfac),
        Float64(py=kfac), Float64(py=tol), f64ptr(hess), Float64(py=omega),
    )
    return PythonObject(None)


def py_grad2e(
    basis: PythonObject, table: PythonObject, dmj: PythonObject, dmk: PythonObject, jfac: PythonObject,
    kfac: PythonObject, tol: PythonObject, de: PythonObject, omega: PythonObject,
) raises -> PythonObject:
    grad2e_core(
        _basis(basis), _boys(table), f64ptr(dmj), Int(py=dmk.shape[0]), f64ptr(dmk), Float64(py=jfac),
        Float64(py=kfac), Float64(py=tol), f64ptr(de), Float64(py=omega),
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


def py_int3c2e_cols(
    basis: PythonObject, auxbasis: PythonObject, table: PythonObject, a0: PythonObject, a1: PythonObject,
    dst: PythonObject, omega: PythonObject,
) raises -> PythonObject:
    var bs = _basis(basis)
    var aux = _basis(auxbasis)
    var boys = _boys(table)
    int3c2e_core(bs, aux, boys, f64ptr(dst), 0, -1, Float64(py=omega), Int(py=a0), Int(py=a1))
    _ = bs^
    _ = aux^
    _ = boys^
    return PythonObject(None)


def py_df_grad_rhs(
    basis: PythonObject, auxbasis: PythonObject, table: PythonObject, dm_tril: PythonObject, orbs: PythonObject,
    blk: PythonObject, rho: PythonObject, q: PythonObject, seq_path: PythonObject, seq_prefix: PythonObject,
    omega: PythonObject,
) raises -> PythonObject:
    df_grad_rhs(
        _blas(seq_path, seq_prefix), _basis(basis), _basis(auxbasis), _boys(table), f64ptr(dm_tril),
        Int(py=orbs.shape[0]), Int(py=orbs.shape[2]), f64ptr(orbs), Int(py=blk), f64ptr(rho), f64ptr(q),
        Float64(py=omega),
    )
    return PythonObject(None)


def py_grad_df3c(
    basis: PythonObject, auxbasis: PythonObject, table: PythonObject, coef: PythonObject, dpack: PythonObject,
    jfac: PythonObject, kfac: PythonObject, xs: PythonObject, cns: PythonObject, blk: PythonObject,
    tol: PythonObject, de: PythonObject, seq_path: PythonObject, seq_prefix: PythonObject, omega: PythonObject,
) raises -> PythonObject:
    grad_df3c_core(
        _blas(seq_path, seq_prefix), _basis(basis), _basis(auxbasis), _boys(table), f64ptr(coef), f64ptr(dpack),
        Float64(py=jfac), Float64(py=kfac), Int(py=cns.shape[0]), Int(py=cns.shape[2]), f64ptr(xs), f64ptr(cns),
        Int(py=blk), Float64(py=tol), f64ptr(de), Float64(py=omega),
    )
    return PythonObject(None)


def py_int3c2e_ip1(
    basis: PythonObject, auxbasis: PythonObject, table: PythonObject, ps0: PythonObject, ps1: PythonObject,
    tol: PythonObject, dst: PythonObject, omega: PythonObject,
) raises -> PythonObject:
    int3c2e_ip1_core(
        _basis(basis), _basis(auxbasis), _boys(table), Int(py=ps0), Int(py=ps1), Float64(py=tol), f64ptr(dst),
        Float64(py=omega),
    )
    return PythonObject(None)


def py_hess_df3c(
    basis: PythonObject, auxbasis: PythonObject, table: PythonObject, coef: PythonObject, dpack: PythonObject,
    jfac: PythonObject, kfac: PythonObject, xs: PythonObject, cns: PythonObject, blk: PythonObject,
    tol: PythonObject, hess: PythonObject, seq_path: PythonObject, seq_prefix: PythonObject, omega: PythonObject,
) raises -> PythonObject:
    hess_df3c_core(
        _blas(seq_path, seq_prefix), _basis(basis), _basis(auxbasis), _boys(table), f64ptr(coef), f64ptr(dpack),
        Float64(py=jfac), Float64(py=kfac), Int(py=cns.shape[0]), Int(py=cns.shape[2]), f64ptr(xs), f64ptr(cns),
        Int(py=blk), Float64(py=tol), f64ptr(hess), Float64(py=omega),
    )
    return PythonObject(None)


def py_grad2c(
    auxbasis: PythonObject, table: PythonObject, w: PythonObject, de: PythonObject, omega: PythonObject
) raises -> PythonObject:
    grad2c_core(_basis(auxbasis), _boys(table), f64ptr(w), Int(py=de.shape[0]), f64ptr(de), Float64(py=omega))
    return PythonObject(None)


def py_int1e_iprinv_dm(
    basis: PythonObject, table: PythonObject, centers: PythonObject, dm: PythonObject, dst: PythonObject
) raises -> PythonObject:
    var bs = _basis(basis)
    var boys = _boys(table)
    int1e_iprinv_dm_core(bs, boys, Int(py=centers.shape[0]), f64ptr(centers), f64ptr(dm), f64ptr(dst))
    _ = bs^
    _ = boys^
    return PythonObject(None)


def py_mm_potential(
    basis: PythonObject, table: PythonObject, coords: PythonObject, weights: PythonObject, zetas: PythonObject,
    point: PythonObject, v_out: PythonObject,
) raises -> PythonObject:
    var bs = _basis(basis)
    var boys = _boys(table)
    mm_potential_core(
        bs, boys, Int(py=weights.shape[0]), f64ptr(coords), f64ptr(weights), f64ptr(zetas), Bool(py=point), f64ptr(v_out)
    )
    _ = bs^
    _ = boys^
    return PythonObject(None)


def py_mm_esp(
    basis: PythonObject, table: PythonObject, coords: PythonObject, zetas: PythonObject, point: PythonObject,
    dms: PythonObject, esp: PythonObject,
) raises -> PythonObject:
    var bs = _basis(basis)
    var boys = _boys(table)
    mm_esp_core(
        bs, boys, Int(py=coords.shape[0]), f64ptr(coords), f64ptr(zetas), Bool(py=point), Int(py=dms.shape[0]),
        f64ptr(dms), f64ptr(esp),
    )
    _ = bs^
    _ = boys^
    return PythonObject(None)


def py_mm_grad(
    basis: PythonObject, table: PythonObject, coords: PythonObject, weights: PythonObject, zetas: PythonObject,
    point: PythonObject, dm: PythonObject, mat: PythonObject, forces: PythonObject, atoms: PythonObject,
) raises -> PythonObject:
    var bs = _basis(basis)
    var boys = _boys(table)
    mm_grad_core(
        bs, boys, Int(py=weights.shape[0]), f64ptr(coords), f64ptr(weights), f64ptr(zetas), Bool(py=point),
        f64ptr(dm), Int(py=mat.size) > 0, f64ptr(mat), Int(py=forces.size) > 0, f64ptr(forces),
        Int(py=atoms.size) > 0, f64ptr(atoms),
    )
    _ = bs^
    _ = boys^
    return PythonObject(None)


def py_eval_ao(basis: PythonObject, coords: PythonObject, deriv: PythonObject, ao_out: PythonObject) raises -> PythonObject:
    var bs = _basis(basis)
    eval_ao_core(bs, Int(py=coords.shape[0]), f64ptr(coords), Int(py=deriv), f64ptr(ao_out))
    _ = bs^
    return PythonObject(None)


def py_xc_rho(
    basis: PythonObject, coords: PythonObject, deriv: PythonObject, dms: PythonObject, orbs: PythonObject,
    occs: PythonObject, rho: PythonObject, seq_path: PythonObject, seq_prefix: PythonObject,
) raises -> PythonObject:
    var bs = _basis(basis)
    xc_rho_core(
        _blas(seq_path, seq_prefix), bs, Int(py=coords.shape[0]), f64ptr(coords), Int(py=deriv),
        Int(py=dms.shape[0]), f64ptr(dms), Int(py=orbs.shape[2]), f64ptr(orbs), f64ptr(occs), f64ptr(rho),
    )
    _ = bs^
    return PythonObject(None)


def py_xc_vmat(
    basis: PythonObject, coords: PythonObject, deriv: PythonObject, wv: PythonObject, vmat: PythonObject,
    seq_path: PythonObject, seq_prefix: PythonObject,
) raises -> PythonObject:
    var bs = _basis(basis)
    xc_vmat_core(
        _blas(seq_path, seq_prefix), bs, Int(py=coords.shape[0]), f64ptr(coords), Int(py=deriv),
        Int(py=wv.shape[0]), f64ptr(wv), f64ptr(vmat),
    )
    _ = bs^
    return PythonObject(None)


def py_xc_fxc(
    basis: PythonObject, coords: PythonObject, weights: PythonObject, kind: PythonObject, fxc: PythonObject,
    dms: PythonObject, lfac: PythonObject, rfac: PythonObject, project: PythonObject, vmat: PythonObject,
    seq_path: PythonObject, seq_prefix: PythonObject,
) raises -> PythonObject:
    var bs = _basis(basis)
    var rank = Int(py=lfac.shape[3])
    xc_fxc_core(
        _blas(seq_path, seq_prefix), bs, Int(py=coords.shape[0]), f64ptr(coords), f64ptr(weights), Int(py=kind),
        Int(py=dms.shape[0]), f64ptr(fxc), Int(py=dms.shape[1]), f64ptr(dms), rank,
        f64ptr(lfac), f64ptr(rfac), Bool(py=project) and rank > 0, f64ptr(vmat),
    )
    _ = bs^
    return PythonObject(None)


def py_xc_hess(
    basis: PythonObject, coords: PythonObject, weights: PythonObject, kind: PythonObject, dms: PythonObject,
    vxc: PythonObject, fxc: PythonObject, aoatm: PythonObject, de2: PythonObject, seq_path: PythonObject,
    seq_prefix: PythonObject,
) raises -> PythonObject:
    var bs = _basis(basis)
    var pat = Pointer[Int64, MutAnyOrigin](unsafe_from_address=Int(py=aoatm.__array_interface__["data"][0]))
    xc_hess_core(
        _blas(seq_path, seq_prefix), bs, Int(py=coords.shape[0]), f64ptr(coords), f64ptr(weights), Int(py=kind),
        Int(py=dms.shape[0]), f64ptr(dms), f64ptr(vxc), f64ptr(fxc), pat, Int(py=de2.shape[0]), f64ptr(de2),
    )
    _ = bs^
    return PythonObject(None)


def py_xc_h1(
    basis: PythonObject, coords: PythonObject, weights: PythonObject, kind: PythonObject, dms: PythonObject,
    fxc: PythonObject, aoatm: PythonObject, cmo: PythonObject, nocc: PythonObject, h1: PythonObject,
    seq_path: PythonObject, seq_prefix: PythonObject,
) raises -> PythonObject:
    var bs = _basis(basis)
    var pat = Pointer[Int64, MutAnyOrigin](unsafe_from_address=Int(py=aoatm.__array_interface__["data"][0]))
    xc_h1_core(
        _blas(seq_path, seq_prefix), bs, Int(py=coords.shape[0]), f64ptr(coords), f64ptr(weights), Int(py=kind),
        Int(py=dms.shape[0]), f64ptr(dms), f64ptr(fxc), pat, Int(py=h1.shape[1]), f64ptr(cmo), Int(py=cmo.shape[2]),
        Int(py=nocc), f64ptr(h1),
    )
    _ = bs^
    return PythonObject(None)


def py_ontop_pi(
    basis: PythonObject, coords: PythonObject, deriv: PythonObject, mo_cas: PythonObject, lt: PythonObject,
    pi: PythonObject, seq_path: PythonObject, seq_prefix: PythonObject,
) raises -> PythonObject:
    var bs = _basis(basis)
    ontop_pi_core(
        _blas(seq_path, seq_prefix), bs, Int(py=coords.shape[0]), f64ptr(coords), Int(py=deriv),
        Int(py=mo_cas.shape[1]), f64ptr(mo_cas), f64ptr(lt), f64ptr(pi),
    )
    _ = bs^
    return PythonObject(None)


def py_ontop_grad(
    basis: PythonObject, coords: PythonObject, wpi: PythonObject, mo_cas: PythonObject, lmat: PythonObject,
    de: PythonObject, seq_path: PythonObject, seq_prefix: PythonObject,
) raises -> PythonObject:
    var bs = _basis(basis)
    ontop_grad_core(
        _blas(seq_path, seq_prefix), bs, Int(py=coords.shape[0]), f64ptr(coords), f64ptr(wpi),
        Int(py=mo_cas.shape[1]), f64ptr(mo_cas), f64ptr(lmat), f64ptr(de),
    )
    _ = bs^
    return PythonObject(None)


def py_ontop_paaa(
    basis: PythonObject, coords: PythonObject, wpi: PythonObject, mo_cas: PythonObject, paaa: PythonObject,
    seq_path: PythonObject, seq_prefix: PythonObject,
) raises -> PythonObject:
    var bs = _basis(basis)
    ontop_paaa_core(
        _blas(seq_path, seq_prefix), bs, Int(py=coords.shape[0]), f64ptr(coords), f64ptr(wpi),
        Int(py=mo_cas.shape[1]), f64ptr(mo_cas), f64ptr(paaa),
    )
    _ = bs^
    return PythonObject(None)


def py_becke_response(
    atm: PythonObject, adj: PythonObject, use_adj: PythonObject, scheme: PythonObject, coords: PythonObject,
    vol: PythonObject, owner: PythonObject, eot: PythonObject, mode: PythonObject, w0: PythonObject,
    de: PythonObject,
) raises -> PythonObject:
    becke_response_core(
        Int(py=atm.shape[0]), f64ptr(atm), f64ptr(adj), Int(py=use_adj), Int(py=scheme), Int(py=coords.shape[0]),
        f64ptr(coords), f64ptr(vol), Int(py=owner), f64ptr(eot), Int(py=mode), f64ptr(w0), f64ptr(de),
    )
    return PythonObject(None)


def py_ontop_density(
    basis: PythonObject, coords: PythonObject, deriv: PythonObject, has_core: PythonObject, dm_core: PythonObject,
    mo_cas: PythonObject, casdm1s: PythonObject, lt: PythonObject, pi_deriv: PythonObject, rho: PythonObject,
    pi: PythonObject, rho_core: PythonObject, seq_path: PythonObject, seq_prefix: PythonObject,
) raises -> PythonObject:
    var bs = _basis(basis)
    ontop_density_core(
        _blas(seq_path, seq_prefix), bs, Int(py=coords.shape[0]), f64ptr(coords), Int(py=deriv), Int(py=has_core),
        f64ptr(dm_core), Int(py=mo_cas.shape[1]), f64ptr(mo_cas), f64ptr(casdm1s), f64ptr(lt), Int(py=pi_deriv),
        f64ptr(rho), f64ptr(pi), f64ptr(rho_core),
    )
    _ = bs^
    return PythonObject(None)


def py_xc_grad(
    basis: PythonObject, coords: PythonObject, kind: PythonObject, wv: PythonObject, vmat: PythonObject,
    seq_path: PythonObject, seq_prefix: PythonObject,
) raises -> PythonObject:
    var bs = _basis(basis)
    xc_grad_core(
        _blas(seq_path, seq_prefix), bs, Int(py=coords.shape[0]), f64ptr(coords), Int(py=kind),
        Int(py=wv.shape[0]), f64ptr(wv), f64ptr(vmat),
    )
    _ = bs^
    return PythonObject(None)


def py_xc_grad_dm(
    basis: PythonObject, coords: PythonObject, gga: PythonObject, wv: PythonObject, dms: PythonObject,
    orbs: PythonObject, occs: PythonObject, de: PythonObject, seq_path: PythonObject, seq_prefix: PythonObject,
) raises -> PythonObject:
    var bs = _basis(basis)
    xc_grad_dm_core(
        _blas(seq_path, seq_prefix), bs, Int(py=coords.shape[0]), f64ptr(coords), Int(py=gga),
        Int(py=wv.shape[0]), f64ptr(wv), f64ptr(dms), Int(py=orbs.shape[2]), f64ptr(orbs), f64ptr(occs), f64ptr(de),
    )
    _ = bs^
    return PythonObject(None)



def py_pcm_ds(
    table: PythonObject, coords: PythonObject, zeta: PythonObject, switch: PythonObject, norm: PythonObject,
    rvdw: PythonObject, with_d: PythonObject, s: PythonObject, d: PythonObject,
) raises -> PythonObject:
    var boys = _boys(table)
    pcm_ds_core(
        boys, Int(py=coords.shape[0]), f64ptr(coords), f64ptr(zeta), f64ptr(switch), f64ptr(norm), f64ptr(rvdw),
        Bool(py=with_d), f64ptr(s), f64ptr(d),
    )
    _ = boys^
    return PythonObject(None)


def py_pcm_pair(
    table: PythonObject, coords: PythonObject, zeta: PythonObject, norm: PythonObject, kind: PythonObject,
    a: PythonObject, b: PythonObject, g: PythonObject,
) raises -> PythonObject:
    var boys = _boys(table)
    pcm_pair_core(
        boys, Int(py=coords.shape[0]), f64ptr(coords), f64ptr(zeta), f64ptr(norm), Int(py=kind), f64ptr(a), f64ptr(b),
        f64ptr(g),
    )
    _ = boys^
    return PythonObject(None)
