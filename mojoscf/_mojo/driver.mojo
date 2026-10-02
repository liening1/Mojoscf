"""The SCF iteration loop (RHF and UHF), implemented natively.

``scf_kernel`` is a line-by-line port of ``pyscf.scf.hf.kernel`` for closed-shell
RHF (one spin channel) and UHF (two spin channels).  Everything pyscf does in
Python/NumPy between two ``get_veff`` calls (Fock assembly, damping, DIIS, level
shift, diagonalisation, occupations, density, energies, convergence tests) runs
here; the only Python call per iteration is the ``get_veff`` callback which
builds the two-electron contribution with pyscf's C integral code.

Array layout: for RHF every matrix is ``(nao, nao)``, orbitals ``(nao, nmo)``,
energies and occupations ``(nmo,)``.  For UHF a leading spin axis of length 2 is
added to all of them.  ``nmo`` is below ``nao`` only when linear dependencies in
the basis were projected out with the orthogonaliser ``x_orth``.
"""
from std.python import Python, PythonObject
from std.memory import Pointer
from std.math import sqrt
from _mojo.linalg import F64Ptr, Blas, list_ptr, vcopy, vlincomb, vnorm2diff, trace_prod
from _mojo.kernels import make_rdm1, get_occ, grad_sumsq, level_shift, diis_errvec, eigh_fock
from _mojo.diis import DIIS


def f64ptr(arr: PythonObject) raises -> F64Ptr:
    """Data pointer of a C-contiguous float64 NumPy array.

    ``__array_interface__`` is noticeably cheaper to query than ``ctypes.data``.
    """
    return F64Ptr(unsafe_from_address=Int(py=arr.__array_interface__["data"][0]))


def _opt_float(opts: PythonObject, key: String, default: Float64) raises -> Float64:
    if key in opts:
        return Float64(py=opts[key])
    return default


def _opt_int(opts: PythonObject, key: String, default: Int) raises -> Int:
    if key in opts:
        return Int(py=opts[key])
    return default


def _opt_bool(opts: PythonObject, key: String, default: Bool) raises -> Bool:
    if key in opts:
        return Bool(py=opts[key])
    return default


def _matrices(np: PythonObject, ns: Int, d1: Int, d2: Int) raises -> PythonObject:
    """Uninitialised (d1, d2) array, with a leading spin axis when ns == 2."""
    if ns == 2:
        return np.empty(Python.tuple(2, d1, d2))
    return np.empty(Python.tuple(d1, d2))


def _vectors(np: PythonObject, ns: Int, n: Int) raises -> PythonObject:
    """Zero-filled (n,) array, with a leading spin axis when ns == 2."""
    if ns == 2:
        return np.zeros(Python.tuple(2, n))
    return np.zeros(n)


def scf_kernel(
    h1e: PythonObject,
    s1e: PythonObject,
    dm0: PythonObject,
    nspin_py: PythonObject,
    nocc_a_py: PythonObject,
    nocc_b_py: PythonObject,
    e_nuc_py: PythonObject,
    get_veff: PythonObject,
    x_orth: PythonObject,
    opts: PythonObject,
    log: PythonObject,
    callback: PythonObject,
    blas_path: PythonObject,
    blas_prefix: PythonObject,
) raises -> PythonObject:
    """Run the SCF iterations; returns a dict with the pyscf result fields.

    Arguments
    ---------
    h1e, s1e  : (nao, nao) float64 C-contiguous arrays (core Hamiltonian, overlap)
    dm0       : initial density matrix, (nao, nao) for RHF or (2, nao, nao) for UHF
    nspin_py  : 1 (RHF, mo_occ in {0, 2}) or 2 (UHF, one spin channel each, mo_occ in {0, 1})
    nocc_a_py, nocc_b_py : occupied orbitals per channel (RHF uses ``nocc_a_py`` only)
    e_nuc_py  : nuclear repulsion energy
    get_veff  : callable(dm, dm_last, vhf_last, mo_coeff, mo_occ) -> vhf, float64 and
                shaped like ``dm``; the orbitals let the caller tag ``dm`` like
                pyscf's make_rdm1 does
    x_orth    : None or (nao, nmo) orthogonaliser used when S is ill-conditioned
    opts      : dict of SCF options (see ``mojoscf.scf``)
    log       : None or callable(cycle, e_tot, delta_e, norm_g, norm_ddm)
    callback  : None or callable(dict) invoked after every cycle
    blas_path, blas_prefix : BLAS/LAPACK library (empty path -> native kernels)
    """
    var np = Python.import_module("numpy")
    var blas = Blas(String(blas_path), String(blas_prefix))

    var ns = Int(py=nspin_py)
    var ph = f64ptr(h1e)
    var ps = f64ptr(s1e)
    var nao = Int(py=h1e.shape[0])
    var nmo = nao
    var has_x = not (x_orth is None)
    # Pointer is non-nullable: alias the overlap when no orthogonaliser is used
    # (the kernels never dereference ``px`` unless ``has_x``).
    var px = ps
    if has_x:
        nmo = Int(py=x_orth.shape[1])
        px = f64ptr(x_orth)
    var nocc_a = Int(py=nocc_a_py)
    var nocc_b = Int(py=nocc_b_py)
    var e_nuc = Float64(py=e_nuc_py)
    var n2 = nao * nao
    var nmo2 = nao * nmo
    var elen = nmo * nmo if has_x else n2
    # Channel-dependent constants: closed shell keeps each orbital doubly occupied
    # and halves the density in the level shift; UHF does neither.
    var occ_value = 2.0 if ns == 1 else 1.0
    var grad_prefactor = 2.0 if ns == 1 else 1.0
    var dm_scale = 0.5 if ns == 1 else 1.0

    var conv_tol = _opt_float(opts, "conv_tol", 1e-9)
    var conv_tol_grad = _opt_float(opts, "conv_tol_grad", sqrt(conv_tol))
    var max_cycle = _opt_int(opts, "max_cycle", 50)
    var use_diis = _opt_bool(opts, "diis", True)
    var diis_space = _opt_int(opts, "diis_space", 8)
    var diis_min_space = _opt_int(opts, "diis_min_space", 1)
    var diis_start_cycle = _opt_int(opts, "diis_start_cycle", 1)
    var diis_damp = _opt_float(opts, "diis_damp", 0.0)
    var damp = _opt_float(opts, "damp", 0.0)
    var damp_b = _opt_float(opts, "damp_b", 0.0)
    var shift_a = _opt_float(opts, "level_shift", 0.0)
    var shift_b = _opt_float(opts, "level_shift_b", 0.0)
    var conv_check = _opt_bool(opts, "conv_check", True)
    var has_log = not (log is None)
    var has_callback = not (callback is None)

    var dm = np.array(dm0, dtype=np.float64, order="C", copy=True)
    var pdm = f64ptr(dm)
    var vhf = np.ascontiguousarray(get_veff(dm, None, None, None, None), dtype=np.float64)
    var pvhf = f64ptr(vhf)

    var e1 = 0.0
    var e2 = 0.0
    for sp in range(ns):
        e1 += trace_prod(ph, pdm.unsafe_offset(sp * n2), nao)
        e2 += 0.5 * trace_prod(pvhf.unsafe_offset(sp * n2), pdm.unsafe_offset(sp * n2), nao)
    var e_tot = e1 + e2 + e_nuc
    if has_log:
        _ = log(PythonObject(-1), PythonObject(e_tot), PythonObject(0.0), PythonObject(0.0), PythonObject(0.0))

    var mo_energy = _vectors(np, ns, nmo)
    var mo_coeff = _matrices(np, ns, nao, nmo)
    var mo_occ = _vectors(np, ns, nmo)
    var pe = f64ptr(mo_energy)
    var pc = f64ptr(mo_coeff)
    var po = f64ptr(mo_occ)

    var fock = _matrices(np, ns, nao, nao)         # Fock matrix fed to the eigensolver
    var fock_plain = _matrices(np, ns, nao, nao)   # h1e + vhf of the current density
    var fock_last = _matrices(np, ns, nao, nao)    # extrapolated Fock matrix of the previous cycle
    var pf = f64ptr(fock)
    var pfp = f64ptr(fock_plain)
    var pfl = f64ptr(fock_last)
    var have_fock_last = False

    var errbuf = List[Float64](length=ns * elen, fill=0.0)
    var perr = list_ptr(errbuf)
    var homo_lumo = List[Float64](length=2, fill=0.0)
    var phl = list_ptr(homo_lumo)
    var diis = DIIS(diis_space, diis_min_space, ns * n2, ns * elen)

    var scf_conv = False
    var last_e = e_tot
    var norm_gorb = 0.0
    var norm_ddm = 0.0
    var has_gap = False
    var dm_last = dm
    var cycle = 0

    # Skip SCF iterations: only the energy of the initial density is wanted.
    if max_cycle <= 0:
        for sp in range(ns):
            vlincomb(pf.unsafe_offset(sp * n2), n2, 1.0, ph, 1.0, pvhf.unsafe_offset(sp * n2))
            eigh_fock(blas, nao, nmo, pf.unsafe_offset(sp * n2), ps, has_x, px, pe.unsafe_offset(sp * nmo), pc.unsafe_offset(sp * nmo2))
            var nocc_s = nocc_a if sp == 0 else nocc_b
            var gap0 = get_occ(nmo, pe.unsafe_offset(sp * nmo), nocc_s, po.unsafe_offset(sp * nmo), phl, occ_value)
            if sp == 0:
                has_gap = gap0
        var res0 = Python.dict()
        res0["converged"] = PythonObject(False)
        res0["e_tot"] = PythonObject(e_tot)
        res0["e1"] = PythonObject(e1)
        res0["e2"] = PythonObject(e2)
        res0["mo_energy"] = mo_energy
        res0["mo_coeff"] = mo_coeff
        res0["mo_occ"] = mo_occ
        res0["dm"] = dm
        res0["vhf"] = vhf
        res0["fock"] = fock
        res0["cycles"] = PythonObject(0)
        res0["norm_gorb"] = PythonObject(0.0)
        res0["norm_ddm"] = PythonObject(0.0)
        if has_gap and ns == 1:
            res0["homo"] = PythonObject(homo_lumo[0])
            res0["lumo"] = PythonObject(homo_lumo[1])
        return res0

    while cycle < max_cycle:
        dm_last = dm
        last_e = e_tot

        # --- get_fock: F = h1e + vhf, then damping / DIIS / level shift.
        # ``fock_last`` is the Fock matrix that was diagonalised in the previous
        # cycle (after damping, DIIS and level shift), exactly as in pyscf.
        for sp in range(ns):
            vlincomb(pf.unsafe_offset(sp * n2), n2, 1.0, ph, 1.0, pvhf.unsafe_offset(sp * n2))
        if cycle < diis_start_cycle - 1 and abs(damp) + abs(damp_b) > 1e-4 and have_fock_last:
            # pyscf damps both spin channels with the alpha factor.
            vlincomb(pf, ns * n2, 1.0 - damp, pf, damp, pfl)
        if use_diis and cycle >= diis_start_cycle:
            for sp in range(ns):
                _ = diis_errvec(
                    blas, nao, nmo, ps, pdm.unsafe_offset(sp * n2), pf.unsafe_offset(sp * n2),
                    has_x, px, perr.unsafe_offset(sp * elen),
                )
            if abs(diis_damp) >= 1e-6 and have_fock_last:
                vlincomb(pf, ns * n2, 1.0 - diis_damp, pf, diis_damp, pfl)
            _ = diis.update(pf, perr, pf)
        if abs(shift_a) + abs(shift_b) > 1e-4:
            for sp in range(ns):
                var factor = shift_a if sp == 0 else shift_b
                level_shift(
                    blas, nao, ps, pdm.unsafe_offset(sp * n2), pf.unsafe_offset(sp * n2), factor,
                    pf.unsafe_offset(sp * n2), dm_scale,
                )
        vcopy(pfl, pf, ns * n2)
        have_fock_last = True

        # --- diagonalise, occupy, new density ---
        mo_energy = _vectors(np, ns, nmo)
        mo_coeff = _matrices(np, ns, nao, nmo)
        mo_occ = _vectors(np, ns, nmo)
        pe = f64ptr(mo_energy)
        pc = f64ptr(mo_coeff)
        po = f64ptr(mo_occ)
        dm = _matrices(np, ns, nao, nao)
        pdm = f64ptr(dm)
        for sp in range(ns):
            eigh_fock(blas, nao, nmo, pf.unsafe_offset(sp * n2), ps, has_x, px, pe.unsafe_offset(sp * nmo), pc.unsafe_offset(sp * nmo2))
            var nocc_s = nocc_a if sp == 0 else nocc_b
            var gap_s = get_occ(nmo, pe.unsafe_offset(sp * nmo), nocc_s, po.unsafe_offset(sp * nmo), phl, occ_value)
            if sp == 0:
                has_gap = gap_s
            make_rdm1(blas, nao, nmo, pc.unsafe_offset(sp * nmo2), po.unsafe_offset(sp * nmo), pdm.unsafe_offset(sp * n2))

        # --- two-electron part (pyscf C code) and energy ---
        vhf = np.ascontiguousarray(get_veff(dm, dm_last, vhf, mo_coeff, mo_occ), dtype=np.float64)
        pvhf = f64ptr(vhf)
        e1 = 0.0
        e2 = 0.0
        for sp in range(ns):
            e1 += trace_prod(ph, pdm.unsafe_offset(sp * n2), nao)
            e2 += 0.5 * trace_prod(pvhf.unsafe_offset(sp * n2), pdm.unsafe_offset(sp * n2), nao)
        e_tot = e1 + e2 + e_nuc

        # --- convergence measures on the un-extrapolated Fock matrix ---
        var gsq = 0.0
        for sp in range(ns):
            vlincomb(pfp.unsafe_offset(sp * n2), n2, 1.0, ph, 1.0, pvhf.unsafe_offset(sp * n2))
            gsq += grad_sumsq(
                blas, nao, nmo, pc.unsafe_offset(sp * nmo2), po.unsafe_offset(sp * nmo),
                pfp.unsafe_offset(sp * n2), grad_prefactor,
            )
        norm_gorb = sqrt(gsq)
        norm_ddm = sqrt(vnorm2diff(pdm, f64ptr(dm_last), ns * n2))
        if has_log:
            _ = log(
                PythonObject(cycle), PythonObject(e_tot), PythonObject(e_tot - last_e),
                PythonObject(norm_gorb), PythonObject(norm_ddm),
            )
        scf_conv = abs(e_tot - last_e) < conv_tol and norm_gorb < conv_tol_grad

        if has_callback:
            var env = Python.dict()
            env["cycle"] = PythonObject(cycle)
            env["e_tot"] = PythonObject(e_tot)
            env["last_hf_e"] = PythonObject(last_e)
            env["dm"] = dm
            env["dm_last"] = dm_last
            env["vhf"] = vhf
            env["fock"] = fock_plain
            env["mo_energy"] = mo_energy
            env["mo_coeff"] = mo_coeff
            env["mo_occ"] = mo_occ
            env["norm_gorb"] = PythonObject(norm_gorb)
            env["norm_ddm"] = PythonObject(norm_ddm)
            env["scf_conv"] = PythonObject(scf_conv)
            env["h1e"] = h1e
            env["s1e"] = s1e
            _ = callback(env)

        cycle += 1
        if scf_conv:
            break

    var cycles = cycle

    if scf_conv and conv_check:
        # An extra diagonalisation of the plain Fock matrix to remove any level shift.
        mo_energy = _vectors(np, ns, nmo)
        mo_coeff = _matrices(np, ns, nao, nmo)
        mo_occ = _vectors(np, ns, nmo)
        pe = f64ptr(mo_energy)
        pc = f64ptr(mo_coeff)
        po = f64ptr(mo_occ)
        dm_last = dm
        dm = _matrices(np, ns, nao, nao)
        pdm = f64ptr(dm)
        for sp in range(ns):
            eigh_fock(blas, nao, nmo, pfp.unsafe_offset(sp * n2), ps, has_x, px, pe.unsafe_offset(sp * nmo), pc.unsafe_offset(sp * nmo2))
            var nocc_s = nocc_a if sp == 0 else nocc_b
            var gap_s = get_occ(nmo, pe.unsafe_offset(sp * nmo), nocc_s, po.unsafe_offset(sp * nmo), phl, occ_value)
            if sp == 0:
                has_gap = gap_s
            make_rdm1(blas, nao, nmo, pc.unsafe_offset(sp * nmo2), po.unsafe_offset(sp * nmo), pdm.unsafe_offset(sp * n2))
        vhf = np.ascontiguousarray(get_veff(dm, dm_last, vhf, mo_coeff, mo_occ), dtype=np.float64)
        pvhf = f64ptr(vhf)
        last_e = e_tot
        e1 = 0.0
        e2 = 0.0
        for sp in range(ns):
            e1 += trace_prod(ph, pdm.unsafe_offset(sp * n2), nao)
            e2 += 0.5 * trace_prod(pvhf.unsafe_offset(sp * n2), pdm.unsafe_offset(sp * n2), nao)
        e_tot = e1 + e2 + e_nuc
        var gsq2 = 0.0
        for sp in range(ns):
            vlincomb(pfp.unsafe_offset(sp * n2), n2, 1.0, ph, 1.0, pvhf.unsafe_offset(sp * n2))
            gsq2 += grad_sumsq(
                blas, nao, nmo, pc.unsafe_offset(sp * nmo2), po.unsafe_offset(sp * nmo),
                pfp.unsafe_offset(sp * n2), grad_prefactor,
            )
        norm_gorb = sqrt(gsq2)
        norm_ddm = sqrt(vnorm2diff(pdm, f64ptr(dm_last), ns * n2))
        conv_tol = conv_tol * 10.0
        conv_tol_grad = conv_tol_grad * 3.0
        scf_conv = abs(e_tot - last_e) < conv_tol or norm_gorb < conv_tol_grad
        if has_log:
            _ = log(
                PythonObject(-2), PythonObject(e_tot), PythonObject(e_tot - last_e),
                PythonObject(norm_gorb), PythonObject(norm_ddm),
            )

    var res = Python.dict()
    res["converged"] = PythonObject(scf_conv)
    res["e_tot"] = PythonObject(e_tot)
    res["e1"] = PythonObject(e1)
    res["e2"] = PythonObject(e2)
    res["mo_energy"] = mo_energy
    res["mo_coeff"] = mo_coeff
    res["mo_occ"] = mo_occ
    res["dm"] = dm
    res["vhf"] = vhf
    res["fock"] = fock_plain
    res["cycles"] = PythonObject(cycles)
    res["norm_gorb"] = PythonObject(norm_gorb)
    res["norm_ddm"] = PythonObject(norm_ddm)
    if has_gap and ns == 1:
        res["homo"] = PythonObject(homo_lumo[0])
        res["lumo"] = PythonObject(homo_lumo[1])
    _ = errbuf^
    _ = homo_lumo^
    return res
