"""Linear-response TDDFT / TDA: pyscf versus mojoscf (dft.accelerate + mojoscf.tdscf), each in its own process.

For each (system, driver) pair the script converges the SCF (conv_tol 1e-9,
pyscf's default grids, density fitting or exact integrals as the case says;
accelerated with ``mojoscf.dft.accelerate`` for mojoscf, whose
``mf.TDA()``/``mf.TDDFT()`` then create the mojoscf.tdscf classes) and solves
for the lowest excited states with pyscf's Davidson solvers (conv_tol 1e-5).
Reported: the TD time, the SCF time and the largest difference of the
excitation energies; with ``--grad`` also the nuclear gradient of the first
excited state (``td.nuc_grad_method()``, exact-integral cases) and the largest
difference of the gradients.  Usage:

    python benchmarks/bench_tddft.py [--cases a,b,...] [--grad] [--list]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

CU4 = 'square_planar_atoms("Cu", ["NH3"] * 4, [2.03] * 4)'

# key: (label, atoms, basis, charge, spin, xc, method, nstates, density fitting)
CASES = {
    "bz-b3lyp-tddft": ("benzene / def2-SVP B3LYP, TDDFT", "BENZENE_ATOMS", "def2-svp", 0, 0, "b3lyp", "TDDFT", 10, True),
    "fc-pbe0-tda": ("ferrocene / def2-SVP PBE0, TDA", "ferrocene_atoms()", "def2-svp", 0, 0, "pbe0", "TDA", 10, True),
    "fc-pbe0-tddft": ("ferrocene / def2-SVP PBE0, TDDFT", "ferrocene_atoms()", "def2-svp", 0, 0, "pbe0", "TDDFT", 10, True),
    "fc-pbe-casida": ("ferrocene / def2-SVP PBE, TDDFT (Casida)", "ferrocene_atoms()", "def2-svp", 0, 0, "pbe", "TDDFT", 10, True),
    "fc-camb3lyp-tda": ("ferrocene / def2-SVP CAM-B3LYP, TDA", "ferrocene_atoms()", "def2-svp", 0, 0, "camb3lyp", "TDA", 10, True),
    "cu-b3lyp-tda": ("[Cu(NH3)4]2+ doublet / def2-TZVP B3LYP, TDA (UKS)", CU4, "def2-tzvp", 2, 1, "b3lyp", "TDA", 8, True),
    "fep-b3lyp-tda": ("Fe(II) porphine triplet / def2-SVP B3LYP, TDA (UKS)", 'porphyrin_atoms("Fe")', "def2-svp", 0, 2,
                      "b3lyp", "TDA", 5, True),
    # exact integrals (pyscf's default)
    "bz-b3lyp-tddft-x": ("benzene / def2-SVP B3LYP, TDDFT", "BENZENE_ATOMS", "def2-svp", 0, 0, "b3lyp", "TDDFT", 10, False),
    "bz-camb3lyp-tda-x": ("benzene / def2-SVP CAM-B3LYP, TDA", "BENZENE_ATOMS", "def2-svp", 0, 0, "camb3lyp", "TDA", 10, False),
    "fc-pbe0-tda-x": ("ferrocene / def2-SVP PBE0, TDA", "ferrocene_atoms()", "def2-svp", 0, 0, "pbe0", "TDA", 10, False),
    "fc-pbe-casida-x": ("ferrocene / def2-SVP PBE, TDDFT (Casida)", "ferrocene_atoms()", "def2-svp", 0, 0, "pbe", "TDDFT", 10, False),
    "cu-b3lyp-tda-x": ("[Cu(NH3)4]2+ doublet / def2-SVP B3LYP, TDA (UKS)", CU4, "def2-svp", 2, 1, "b3lyp", "TDA", 8, False),
}

WORKER = r'''
import json, sys, time
import numpy as np
sys.path.insert(0, %(bench_dir)r)
from pyscf import dft, gto
from systems import *
from bench_scf import BENZENE
BENZENE_ATOMS = [(a[0], tuple(a[1])) for a in gto.format_atom(BENZENE, unit=1.0)]
mol = gto.M(atom=atoms_to_str(%(atoms)s), basis=%(basis)r, charge=%(charge)d, spin=%(spin)d,
            verbose=0, max_memory=12000)
driver = %(driver)r
if driver == "mojoscf":
    import mojoscf
    mojoscf.UHF(gto.M(atom="H 0 0 0; H 0 0 1", basis="sto-3g", verbose=0)).run()  # start the runtime
mf = (dft.UKS if %(spin)d else dft.RKS)(mol, xc=%(xc)r)
if %(df)r:
    mf = mf.density_fit()
mf.verbose = 0; mf.conv_tol = 1e-9; mf.max_cycle = 100
if driver == "mojoscf":
    mojoscf.dft.accelerate(mf)
t0 = time.perf_counter(); mf.kernel(); tscf = time.perf_counter() - t0
td = getattr(mf, %(method)r)()
td.nstates = %(nstates)d
t0 = time.perf_counter(); e = td.kernel()[0]; ttd = time.perf_counter() - t0
tgrad, grad = None, None
if %(grad)r:
    t0 = time.perf_counter(); grad = td.nuc_grad_method().kernel(state=1); tgrad = time.perf_counter() - t0
    grad = np.asarray(grad).tolist()
print(json.dumps(dict(tscf=tscf, ttd=ttd, e=np.asarray(e).tolist(), conv=bool(np.all(td.converged)) and bool(mf.converged),
                      nao=mol.nao_nr(), tdclass=type(td).__module__ + "." + type(td).__name__, tgrad=tgrad, grad=grad)))
'''


def run(case, driver, bench_dir, grad=False):
    _, atoms, basis, charge, spin, xc, method, nstates, df = CASES[case]
    code = WORKER % dict(bench_dir=bench_dir, atoms=atoms, basis=basis, charge=charge, spin=spin, xc=xc,
                         method=method, nstates=nstates, driver=driver, df=df, grad=grad and not df)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def main():
    import os

    import numpy as np

    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--grad", action="store_true", help="also the gradient of the first excited state (exact integrals)")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        print("\n".join(f"{k:18s} {v[0]}{'' if v[8] else ', exact integrals'}" for k, v in CASES.items()))
        return
    bench_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"{'system':50s} {'ints':>5s} {'nao':>4s} {'nst':>4s} | {'TD pyscf':>8s} {'mojoscf':>8s} {'x':>5s} | "
          f"{'SCF pyscf':>9s} {'mojoscf':>8s} | {'max|dE| eV':>10s}"
          + (f" | {'grad pyscf':>10s} {'mojoscf':>8s} {'x':>5s} {'max|dg|':>8s}" if args.grad else ""))
    for key in args.cases.split(","):
        name, nstates, df = CASES[key][0], CASES[key][7], CASES[key][8]
        ref = run(key, "pyscf", bench_dir, args.grad)
        moj = run(key, "mojoscf", bench_dir, args.grad)
        flag = "" if ref["conv"] and moj["conv"] else "  NOT CONVERGED"
        n = min(len(ref["e"]), len(moj["e"]))
        de = abs(np.array(ref["e"][:n]) - np.array(moj["e"][:n])).max() * 27.211386
        line = (f"{name:50s} {'DF' if df else 'exact':>5s} {ref['nao']:4d} {nstates:4d} | {ref['ttd']:8.1f} {moj['ttd']:8.1f} "
                f"{ref['ttd'] / moj['ttd']:4.1f}x | {ref['tscf']:9.1f} {moj['tscf']:8.1f} | {de:10.1e}")
        if args.grad and ref["tgrad"] is not None:
            dg = abs(np.array(ref["grad"]) - np.array(moj["grad"])).max()
            line += f" | {ref['tgrad']:10.1f} {moj['tgrad']:8.1f} {ref['tgrad'] / moj['tgrad']:4.1f}x {dg:8.1e}"
        print(line + flag, flush=True)


if __name__ == "__main__":
    main()
