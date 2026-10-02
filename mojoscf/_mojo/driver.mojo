"""The RHF SCF iteration loop, implemented natively.

``rhf_kernel`` is a line-by-line port of ``pyscf.scf.hf.kernel`` specialised to
closed-shell RHF.  Everything pyscf does in Python/NumPy between two
``get_veff`` calls (Fock assembly, damping, DIIS, level shift,
diagonalisation, occupations, density, energies, convergence tests) runs
here; the only Python call per iteration is the ``get_veff`` callback which
builds the two-electron contribution with pyscf's C integral code.
"""
from std.python import Python, PythonObject
from std.memory import Pointer
from std.math import sqrt
from _mojo.linalg import F64Ptr, Blas, list_ptr, vcopy, vlincomb, vnorm2diff, trace_prod
from _mojo.kernels import make_rdm1, get_occ, grad_norm, level_shift, diis_errvec, eigh_fock
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


def rhf_kernel(
    h1e: PythonObject,
    s1e: PythonObject,
    dm0: PythonObject,
    nocc_py: PythonObject,
    e_nuc_py: PythonObject,
    get_veff: PythonObject,
    x_orth: PythonObject,
    opts: PythonObject,
    log: PythonObject,
    callback: PythonObject,
    blas_path: PythonObject,
    blas_prefix: PythonObject,
) raises -> PythonObject:
    """Run the RHF SCF iterations; returns a dict with the pyscf result fields.

    Arguments
    ---------
    h1e, s1e : (nao, nao) float64 C-contiguous arrays (core Hamiltonian, overlap)
    dm0      : (nao, nao) initial density matrix (copied)
    nocc_py  : number of doubly occupied orbitals
    e_nuc_py : nuclear repulsion energy
    get_veff : callable(dm, dm_last, vhf_last) -> vhf (nao, nao) float64
    x_orth   : None or (nao, nmo) orthogonaliser used when S is ill-conditioned
    opts     : dict of SCF options (see ``mojoscf.scf``)
    log      : None or callable(cycle, e_tot, delta_e, norm_g, norm_ddm)
    callback : None or callable(dict) invoked after every cycle
    blas_path, blas_prefix : BLAS/LAPACK library (empty path -> native kernels)
    """
    var np = Python.import_module("numpy")
    var blas = Blas(String(blas_path), String(blas_prefix))

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
    var nocc = Int(py=nocc_py)
    var e_nuc = Float64(py=e_nuc_py)
    var n2 = nao * nao
    var elen = nmo * nmo if has_x else n2

    var conv_tol = _opt_float(opts, "conv_tol", 1e-9)
    var conv_tol_grad = _opt_float(opts, "conv_tol_grad", sqrt(conv_tol))
    var max_cycle = _opt_int(opts, "max_cycle", 50)
    var use_diis = _opt_bool(opts, "diis", True)
    var diis_space = _opt_int(opts, "diis_space", 8)
    var diis_min_space = _opt_int(opts, "diis_min_space", 1)
    var diis_start_cycle = _opt_int(opts, "diis_start_cycle", 1)
    var diis_damp = _opt_float(opts, "diis_damp", 0.0)
    var damp = _opt_float(opts, "damp", 0.0)
    var shift = _opt_float(opts, "level_shift", 0.0)
    var conv_check = _opt_bool(opts, "conv_check", True)
    var has_log = not (log is None)
    var has_callback = not (callback is None)

    var dm = np.array(dm0, dtype=np.float64, order="C", copy=True)
    var pdm = f64ptr(dm)
    var vhf = np.ascontiguousarray(get_veff(dm, None, None), dtype=np.float64)
    var pvhf = f64ptr(vhf)

    var e1 = trace_prod(ph, pdm, nao)
    var e2 = 0.5 * trace_prod(pvhf, pdm, nao)
    var e_tot = e1 + e2 + e_nuc
    if has_log:
        _ = log(PythonObject(-1), PythonObject(e_tot), PythonObject(0.0), PythonObject(0.0), PythonObject(0.0))

    var mo_energy = np.empty(nmo)
    var mo_coeff = np.empty(Python.tuple(nao, nmo))
    var mo_occ = np.zeros(nmo)
    var pe = f64ptr(mo_energy)
    var pc = f64ptr(mo_coeff)
    var po = f64ptr(mo_occ)

    var fock = np.empty(Python.tuple(nao, nao))      # Fock matrix fed to the eigensolver
    var fock_plain = np.empty(Python.tuple(nao, nao))  # h1e + vhf of the current density
    var fock_last = np.empty(Python.tuple(nao, nao))   # extrapolated Fock matrix of the previous cycle
    var pf = f64ptr(fock)
    var pfp = f64ptr(fock_plain)
    var pfl = f64ptr(fock_last)
    var have_fock_last = False

    var errbuf = List[Float64](length=elen, fill=0.0)
    var perr = list_ptr(errbuf)
    var homo_lumo = List[Float64](length=2, fill=0.0)
    var phl = list_ptr(homo_lumo)
    var diis = DIIS(diis_space, diis_min_space, n2, elen)

    var scf_conv = False
    var last_e = e_tot
    var norm_gorb = 0.0
    var norm_ddm = 0.0
    var has_gap = False
    var dm_last = dm
    var cycle = 0

    # Skip SCF iterations: only the energy of the initial density is wanted.
    if max_cycle <= 0:
        vlincomb(pf, n2, 1.0, ph, 1.0, pvhf)
        eigh_fock(blas, nao, nmo, pf, ps, has_x, px, pe, pc)
        has_gap = get_occ(nmo, pe, nocc, po, phl)
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
        if has_gap:
            res0["homo"] = PythonObject(homo_lumo[0])
            res0["lumo"] = PythonObject(homo_lumo[1])
        return res0

    while cycle < max_cycle:
        dm_last = dm
        last_e = e_tot

        # --- get_fock: F = h1e + vhf, then damping / DIIS / level shift.
        # ``fock_last`` is the Fock matrix that was diagonalised in the previous
        # cycle (after damping, DIIS and level shift), exactly as in pyscf.
        vlincomb(pf, n2, 1.0, ph, 1.0, pvhf)
        if cycle < diis_start_cycle - 1 and abs(damp) > 1e-4 and have_fock_last:
            vlincomb(pf, n2, 1.0 - damp, pf, damp, pfl)
        if use_diis and cycle >= diis_start_cycle:
            _ = diis_errvec(blas, nao, nmo, ps, pdm, pf, has_x, px, perr)
            if abs(diis_damp) >= 1e-6 and have_fock_last:
                vlincomb(pf, n2, 1.0 - diis_damp, pf, diis_damp, pfl)
            _ = diis.update(pf, perr, pf)
        if abs(shift) > 1e-4:
            level_shift(blas, nao, ps, pdm, pf, shift, pf)
        vcopy(pfl, pf, n2)
        have_fock_last = True

        # --- diagonalise, occupy, new density ---
        mo_energy = np.empty(nmo)
        mo_coeff = np.empty(Python.tuple(nao, nmo))
        mo_occ = np.zeros(nmo)
        pe = f64ptr(mo_energy)
        pc = f64ptr(mo_coeff)
        po = f64ptr(mo_occ)
        eigh_fock(blas, nao, nmo, pf, ps, has_x, px, pe, pc)
        has_gap = get_occ(nmo, pe, nocc, po, phl)
        dm = np.empty(Python.tuple(nao, nao))
        pdm = f64ptr(dm)
        make_rdm1(blas, nao, nmo, pc, po, pdm)

        # --- two-electron part (pyscf C code) and energy ---
        vhf = np.ascontiguousarray(get_veff(dm, dm_last, vhf), dtype=np.float64)
        pvhf = f64ptr(vhf)
        e1 = trace_prod(ph, pdm, nao)
        e2 = 0.5 * trace_prod(pvhf, pdm, nao)
        e_tot = e1 + e2 + e_nuc

        # --- convergence measures on the un-extrapolated Fock matrix ---
        vlincomb(pfp, n2, 1.0, ph, 1.0, pvhf)
        norm_gorb = grad_norm(blas, nao, nmo, pc, po, pfp)
        norm_ddm = sqrt(vnorm2diff(pdm, f64ptr(dm_last), n2))
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
        mo_energy = np.empty(nmo)
        mo_coeff = np.empty(Python.tuple(nao, nmo))
        mo_occ = np.zeros(nmo)
        pe = f64ptr(mo_energy)
        pc = f64ptr(mo_coeff)
        po = f64ptr(mo_occ)
        eigh_fock(blas, nao, nmo, pfp, ps, has_x, px, pe, pc)
        has_gap = get_occ(nmo, pe, nocc, po, phl)
        dm_last = dm
        dm = np.empty(Python.tuple(nao, nao))
        pdm = f64ptr(dm)
        make_rdm1(blas, nao, nmo, pc, po, pdm)
        vhf = np.ascontiguousarray(get_veff(dm, dm_last, vhf), dtype=np.float64)
        pvhf = f64ptr(vhf)
        last_e = e_tot
        e1 = trace_prod(ph, pdm, nao)
        e2 = 0.5 * trace_prod(pvhf, pdm, nao)
        e_tot = e1 + e2 + e_nuc
        vlincomb(pfp, n2, 1.0, ph, 1.0, pvhf)
        norm_gorb = grad_norm(blas, nao, nmo, pc, po, pfp)
        norm_ddm = sqrt(vnorm2diff(pdm, f64ptr(dm_last), n2))
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
    if has_gap:
        res["homo"] = PythonObject(homo_lumo[0])
        res["lumo"] = PythonObject(homo_lumo[1])
    _ = errbuf^
    _ = homo_lumo^
    return res
