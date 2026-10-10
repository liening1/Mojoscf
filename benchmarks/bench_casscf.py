"""State-averaged DF-CASSCF of transition-metal complexes: pyscf versus mojoscf, each in its own process.

For each (system, driver) pair the script converges a density-fitted ROHF
reference (conv_tol 1e-9; the same pyscf object, accelerated with
``mojoscf.accelerate`` for mojoscf), takes the metal 3d orbitals as the
active space with AVAS (singly occupied orbitals kept active,
``openshell_option=3``) and runs pyscf's CASSCF (``mcscf.CASSCF``,
conv_tol 1e-8) averaged over all states of that space with equal weights:
the ligand-field states of the high-spin ion.  With an accelerated
reference, pyscf's driver builds its integrals with :mod:`mojoscf.casscf`
(which also runs the J/K of the orbital Hessian steps) and its other J/K
with the Mojo DF kernel (:class:`mojoscf.dft.MojoDF`).  Reported: the CASSCF
and SCF times, the CASSCF macro iterations and orbital Hessian steps (J/K
builds) of each driver and the energy difference.  ``--tuned-pyscf`` also
runs pyscf with ``OPENBLAS_THREAD_TIMEOUT=16``, the OpenBLAS spin-wait that
mojoscf sets when it is imported.  ``--nevpt2`` adds pyscf's
strongly-contracted NEVPT2 (``mrpt.NEVPT``) of the three lowest states, a
CASCI on the state-averaged orbitals (DF integrals from
:func:`mojoscf.casscf.nevpt2_eris` for mojoscf).  ``--grad`` instead times
the nuclear gradient of the state-specific CASSCF of the lowest state
(``mc.nuc_grad_method()``: :class:`mojoscf.casscf.Gradients` for mojoscf);
``--sa-grad`` adds the state-averaged gradient of the lowest state after
the state-averaged CASSCF.  Usage:

    python benchmarks/bench_casscf.py [--cases a,b,...] [--heavy] [--tuned-pyscf] [--nevpt2] [--sa-grad] [--grad]
                                      [--list]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

FE6 = 'octahedral_atoms("Fe", "H2O", 2.12)'
NI6 = 'octahedral_atoms("Ni", "H2O", 2.06)'
CU4 = 'square_planar_atoms("Cu", ["NH3"] * 4, [2.03] * 4)'

# key: (label, atoms, basis, charge, spin, AVAS AO labels, states averaged)
CASES = {
    "fe6-svp": ("[Fe(H2O)6]2+ quintet / def2-SVP", FE6, "def2-svp", 2, 4, ["Fe 3d"], 5),
    "ni6-svp": ("[Ni(H2O)6]2+ triplet / def2-SVP", NI6, "def2-svp", 2, 2, ["Ni 3d"], 10),
    "cu4-svp": ("[Cu(NH3)4]2+ doublet / def2-SVP", CU4, "def2-svp", 2, 1, ["Cu 3d"], 5),
    "fe6-tzvp": ("[Fe(H2O)6]2+ quintet / def2-TZVP", FE6, "def2-tzvp", 2, 4, ["Fe 3d"], 5),
    "cu4-tzvp": ("[Cu(NH3)4]2+ doublet / def2-TZVP", CU4, "def2-tzvp", 2, 1, ["Cu 3d"], 5),
}
HEAVY: set[str] = set()     # cases left out unless --heavy

WORKER = r'''
import json, sys, time
sys.path.insert(0, %(bench_dir)r)
from pyscf import gto, mcscf, scf
from pyscf.mcscf import avas
from systems import *
mol = gto.M(atom=atoms_to_str(%(atoms)s), basis=%(basis)r, charge=%(charge)d, spin=%(spin)d,
            verbose=0, max_memory=12000)
driver = %(driver)r
if driver == "mojoscf":
    import mojoscf
    mojoscf.UHF(gto.M(atom="H 0 0 0; H 0 0 1", basis="sto-3g", verbose=0)).run()  # start the runtime
mf = scf.ROHF(mol).density_fit()
mf.conv_tol = 1e-9; mf.max_cycle = 100
if driver == "mojoscf":
    mojoscf.accelerate(mf)
t0 = time.perf_counter(); mf.kernel(); tscf = time.perf_counter() - t0
ncas, nelecas, mo = avas.avas(mf, %(ao_labels)r, canonicalize=False, openshell_option=3, verbose=0)
if %(grad)r:
    mc = mcscf.CASSCF(mf, ncas, nelecas).run(mo, conv_tol=1e-10)
    t0 = time.perf_counter(); de = mc.nuc_grad_method().kernel(); tgrad = time.perf_counter() - t0
    print(json.dumps(dict(tscf=tscf, tgrad=tgrad, de=de.tolist(), ecas=mc.e_tot, ncas=int(ncas), nelecas=int(nelecas),
                          nao=mol.nao_nr(), conv=bool(mc.converged))))
    sys.exit()
nstates = %(nstates)d
mc = mcscf.CASSCF(mf, ncas, nelecas)
if nstates > 1:
    mc = mc.state_average_([1.0 / nstates] * nstates)
mc.conv_tol = 1e-8
stats = {}
mc.callback = lambda envs: stats.update(macro=envs["imacro"], jk=envs["totinner"])
t0 = time.perf_counter(); mc.kernel(mo); tcas = time.perf_counter() - t0
tsag, dsag = None, []
if %(sa_grad)r:
    t0 = time.perf_counter(); dsag = mc.nuc_grad_method().kernel(state=0).tolist(); tsag = time.perf_counter() - t0
enev, tnev = [], None
if %(nevpt2)r:
    from pyscf import mrpt
    mc2 = mcscf.CASCI(mf, ncas, nelecas)
    mc2.fcisolver.nroots = 3
    mc2.kernel(mc.mo_coeff)
    t0 = time.perf_counter()
    enev = [float(mrpt.NEVPT(mc2, root=i).kernel()) for i in range(3)]
    tnev = time.perf_counter() - t0
print(json.dumps(dict(tscf=tscf, tcas=tcas, ecas=mc.e_tot, ncas=int(ncas), nelecas=int(nelecas), nao=mol.nao_nr(),
                      conv=bool(mc.converged), scf_conv=bool(mf.converged), tnev=tnev, enev=enev, tsag=tsag,
                      dsag=dsag, **stats)))
'''


def run(case, driver, bench_dir, env=None, nevpt2=False, grad=False, sa_grad=False):
    _, atoms, basis, charge, spin, ao_labels, nstates = CASES[case]
    code = WORKER % dict(bench_dir=bench_dir, atoms=atoms, basis=basis, charge=charge, spin=spin,
                         ao_labels=ao_labels, nstates=nstates, driver=driver, nevpt2=nevpt2, grad=grad,
                         sa_grad=sa_grad)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env=None if env is None else {**os.environ, **env})
    if out.returncode != 0:
        raise RuntimeError(f"{case} / {driver} failed:\n{out.stderr[-3000:]}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=None)
    ap.add_argument("--heavy", action="store_true", help="also run the cases marked heavy")
    ap.add_argument("--tuned-pyscf", action="store_true",
                    help="also time pyscf with OPENBLAS_THREAD_TIMEOUT=16 (mojoscf's OpenBLAS spin-wait)")
    ap.add_argument("--nevpt2", action="store_true", help="also time NEVPT2 of the three lowest states")
    ap.add_argument("--grad", action="store_true", help="time the state-specific CASSCF gradient instead")
    ap.add_argument("--sa-grad", action="store_true", help="also time the state-averaged gradient of state 0")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        print("\n".join(f"{k:9s} {v[0]}, SA({v[6]}) over the {' '.join(v[5])} space{' (heavy)' if k in HEAVY else ''}"
                        for k, v in CASES.items()))
        return
    bench_dir = os.path.dirname(os.path.abspath(__file__))
    tuned = f" {'tuned':>7s}" if args.tuned_pyscf else ""
    keys = args.cases.split(",") if args.cases else [k for k in CASES if args.heavy or k not in HEAVY]
    if args.grad:
        print(f"{'system':38s} {'nao':>4s} {'CAS':>7s} | {'grad pyscf':>10s}{tuned} {'mojoscf':>8s} {'x':>5s} | "
              f"{'max|dg|':>8s} {'|g|':>7s}")
        for key in keys:
            ref = run(key, "pyscf", bench_dir, grad=True)
            tun = run(key, "pyscf", bench_dir, {"OPENBLAS_THREAD_TIMEOUT": "16"}, grad=True) if args.tuned_pyscf else None
            moj = run(key, "mojoscf", bench_dir, grad=True)
            dg = max(abs(a - b) for ra, rb in zip(ref["de"], moj["de"]) for a, b in zip(ra, rb))
            gmax = max(abs(a) for ra in ref["de"] for a in ra)
            tcol = f" {tun['tgrad']:7.1f}" if tun else ""
            cas = f"({ref['nelecas']},{ref['ncas']})"
            print(f"{CASES[key][0]:38s} {ref['nao']:4d} {cas:>7s} | {ref['tgrad']:10.1f}{tcol} {moj['tgrad']:8.1f} "
                  f"{ref['tgrad'] / moj['tgrad']:4.1f}x | {dg:8.1e} {gmax:7.1e}", flush=True)
        return
    nev = f" | {'NEVPT2 pyscf':>12s}{tuned} {'mojoscf':>8s} {'x':>5s} {'|dE2|':>7s}" if args.nevpt2 else ""
    nev += f" | {'SA grad pyscf':>13s}{tuned} {'mojoscf':>8s} {'x':>5s} {'max|dg|':>8s}" if args.sa_grad else ""
    print(f"{'system':38s} {'nao':>4s} {'CAS':>7s} {'SA':>3s} {'macro':>5s} {'AH J/K':>7s} | {'CASSCF pyscf':>12s}{tuned} "
          f"{'mojoscf':>8s} {'x':>5s} | {'SCF pyscf':>9s} {'mojoscf':>8s} | {'|dE|':>7s}{nev}")
    for key in keys:
        name, nstates = CASES[key][0], CASES[key][6]
        opts = dict(nevpt2=args.nevpt2, sa_grad=args.sa_grad)
        ref = run(key, "pyscf", bench_dir, **opts)
        tun = run(key, "pyscf", bench_dir, {"OPENBLAS_THREAD_TIMEOUT": "16"}, **opts) if args.tuned_pyscf else None
        moj = run(key, "mojoscf", bench_dir, **opts)
        cas = f"({ref['nelecas']},{ref['ncas']})"
        flag = "" if ref["conv"] and moj["conv"] else "  NOT CONVERGED"
        tcol = f" {tun['tcas']:7.1f}" if tun else ""
        nev = ""
        if args.nevpt2:
            de2 = max(abs(a - b) for a, b in zip(ref["enev"], moj["enev"]))
            tnev = f" {tun['tnev']:7.1f}" if tun else ""
            nev = f" | {ref['tnev']:12.1f}{tnev} {moj['tnev']:8.1f} {ref['tnev'] / moj['tnev']:4.1f}x {de2:7.1e}"
        if args.sa_grad:
            dg = max(abs(a - b) for ra, rb in zip(ref["dsag"], moj["dsag"]) for a, b in zip(ra, rb))
            tsag = f" {tun['tsag']:7.1f}" if tun else ""
            nev += f" | {ref['tsag']:13.1f}{tsag} {moj['tsag']:8.1f} {ref['tsag'] / moj['tsag']:4.1f}x {dg:8.1e}"
        print(f"{name:38s} {ref['nao']:4d} {cas:>7s} {nstates:3d} {ref['macro']:2d}/{moj['macro']:<2d} "
              f"{ref['jk']:3d}/{moj['jk']:<3d} | {ref['tcas']:12.1f}{tcol} {moj['tcas']:8.1f} "
              f"{ref['tcas'] / moj['tcas']:4.1f}x | {ref['tscf']:9.1f} {moj['tscf']:8.1f} | "
              f"{abs(ref['ecas'] - moj['ecas']):7.1e}{flag}{nev}", flush=True)


if __name__ == "__main__":
    main()
