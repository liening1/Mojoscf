"""Implicit solvation (pyscf.solvent PCM family, SMD): SCF and gradient, pyscf versus mojoscf, each in its own process.

Kohn-Sham cases are accelerated with ``mojoscf.dft.accelerate`` (XC, J/K,
gradients and the solvent kernels), Hartree-Fock cases with
``mojoscf.solvent.attach`` only (the native loop does not take solvent
objects).  Water as the solvent, conv_tol 1e-9, density fitting.  Usage:

    python benchmarks/bench_solvent.py [--cases a,b,...] [--list]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

CU4 = 'square_planar_atoms("Cu", ["NH3"] * 4, [2.03] * 4)'

# key: (label, atoms, basis, charge, spin, method (xc or "hf"), model)
CASES = {
    "bz-b3lyp-cpcm": ("benzene / def2-SVP B3LYP, C-PCM (DF)", "BENZENE_ATOMS", "def2-svp", 0, 0, "b3lyp", "C-PCM"),
    "fc-pbe-iefpcm": ("ferrocene / def2-SVP PBE, IEF-PCM (DF)", "ferrocene_atoms()", "def2-svp", 0, 0, "pbe", "IEF-PCM"),
    "cu-b3lyp-smd": ("[Cu(NH3)4]2+ doublet / def2-SVP B3LYP, SMD (DF, UKS)", CU4, "def2-svp", 2, 1, "b3lyp", "SMD"),
    "w5-hf-iefpcm": ("(H2O)5 / def2-TZVP HF, IEF-PCM (DF; attach)", "water_cluster_atoms(5)", "def2-tzvp", 0, 0, "hf", "IEF-PCM"),
}

WORKER = r'''
import json, sys, time, warnings
warnings.simplefilter("ignore")
import numpy as np
sys.path.insert(0, %(bench_dir)r)
from pyscf import dft, gto, scf
from systems import *
from bench_scf import BENZENE
BENZENE_ATOMS = [(a[0], tuple(a[1])) for a in gto.format_atom(BENZENE, unit=1.0)]
mol = gto.M(atom=atoms_to_str(%(atoms)s), basis=%(basis)r, charge=%(charge)d, spin=%(spin)d,
            verbose=0, max_memory=12000)
driver = %(driver)r
if driver == "mojoscf":
    import mojoscf
    mojoscf.UHF(gto.M(atom="H 0 0 0; H 0 0 1", basis="sto-3g", verbose=0)).run()  # start the runtime
method = %(method)r
if method == "hf":
    mf = (scf.UHF if %(spin)d else scf.RHF)(mol)
else:
    mf = (dft.UKS if %(spin)d else dft.RKS)(mol, xc=method)
mf = mf.density_fit()
if %(model)r == "SMD":
    mf = mf.SMD()
    mf.with_solvent.solvent = "water"
else:
    mf = mf.PCM()
    mf.with_solvent.method = %(model)r
mf.verbose = 0; mf.conv_tol = 1e-9; mf.max_cycle = 100
if driver == "mojoscf":
    if method == "hf":
        mojoscf.solvent.attach(mf)
    else:
        mojoscf.dft.accelerate(mf)
t0 = time.perf_counter(); mf.kernel(); tscf = time.perf_counter() - t0
g = mf.nuc_grad_method()
t0 = time.perf_counter(); de = g.kernel(); tgrad = time.perf_counter() - t0
print(json.dumps(dict(tscf=tscf, tgrad=tgrad, e=mf.e_tot, cycles=mf.cycles, conv=bool(mf.converged), de=de.tolist(),
                      nao=mol.nao_nr(), nsurf=int(mf.with_solvent.surface["grid_coords"].shape[0]))))
'''


def run(case, driver, bench_dir):
    _, atoms, basis, charge, spin, method, model = CASES[case]
    code = WORKER % dict(bench_dir=bench_dir, atoms=atoms, basis=basis, charge=charge, spin=spin, method=method,
                         model=model, driver=driver)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def main():
    import os

    import numpy as np

    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        print("\n".join(f"{k:14s} {v[0]}" for k, v in CASES.items()))
        return
    bench_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"{'system':54s} {'nao':>4s} {'surf':>5s} {'cyc':>6s} | {'SCF pyscf':>9s} {'mojoscf':>8s} {'x':>5s} | "
          f"{'grad pyscf':>10s} {'mojoscf':>8s} {'x':>5s} | {'|dE|':>7s} {'max|dg|':>8s}")
    for key in args.cases.split(","):
        name = CASES[key][0]
        ref = run(key, "pyscf", bench_dir)
        moj = run(key, "mojoscf", bench_dir)
        flag = "" if ref["conv"] and moj["conv"] else "  NOT CONVERGED"
        dg = abs(np.array(ref["de"]) - np.array(moj["de"])).max()
        print(f"{name:54s} {ref['nao']:4d} {ref['nsurf']:5d} {ref['cycles']:2d}/{moj['cycles']:<3d}| {ref['tscf']:9.1f} "
              f"{moj['tscf']:8.1f} {ref['tscf'] / moj['tscf']:4.1f}x | {ref['tgrad']:10.1f} {moj['tgrad']:8.1f} "
              f"{ref['tgrad'] / moj['tgrad']:4.1f}x | {abs(ref['e'] - moj['e']):7.1e} {dg:8.1e}{flag}", flush=True)


if __name__ == "__main__":
    main()
