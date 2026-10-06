"""Kohn-Sham DFT: SCF and nuclear gradient, pyscf versus mojoscf.dft.accelerate, each run in its own process.

For each (system, driver) pair the script converges the SCF (conv_tol 1e-9,
pyscf's default grids; the same pyscf object, accelerated with
``mojoscf.dft.accelerate`` for mojoscf) and then computes the gradient with
``nuc_grad_method()``.  "DF" runs use pyscf's default fitting basis,
"direct" runs ``max_memory=1`` (the 4-index integrals are recomputed every
cycle), "in-core" stores them.  Usage:

    python benchmarks/bench_dft.py [--cases a,b,...] [--list]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

FE6 = 'octahedral_atoms("Fe", "H2O", 2.12)'
CU4 = 'square_planar_atoms("Cu", ["NH3"] * 4, [2.03] * 4)'

# key: (label, atoms, basis, charge, spin, xc, mode)
CASES = {
    "bz-b3lyp-df": ("benzene / def2-SVP B3LYP (DF)", "BENZENE_ATOMS", "def2-svp", 0, 0, "b3lyp", "df"),
    "bzp-b3lyp-df": ("benzene+ / def2-SVP B3LYP (DF, UKS)", "BENZENE_ATOMS", "def2-svp", 1, 1, "b3lyp", "df"),
    "c8-pbe-df": ("C8H18 / def2-TZVP PBE (DF)", "alkane_atoms(8)", "def2-tzvp", 0, 0, "pbe", "df"),
    "fc-pbe-df": ("ferrocene / def2-SVP PBE (DF)", "ferrocene_atoms()", "def2-svp", 0, 0, "pbe", "df"),
    "fc-b3lyp-df": ("ferrocene / def2-SVP B3LYP (DF)", "ferrocene_atoms()", "def2-svp", 0, 0, "b3lyp", "df"),
    "fc-pbe-tz-df": ("ferrocene / def2-TZVP PBE (DF)", "ferrocene_atoms()", "def2-tzvp", 0, 0, "pbe", "df"),
    "fe6-pbe0-df": ("[Fe(H2O)6]2+ quintet / def2-TZVP PBE0 (DF, UKS)", FE6, "def2-tzvp", 2, 4, "pbe0", "df"),
    "cu-b3lyp-df": ("[Cu(NH3)4]2+ doublet / def2-TZVP B3LYP (DF, UKS)", CU4, "def2-tzvp", 2, 1, "b3lyp", "df"),
    "fc-r2scan-df": ("ferrocene / def2-SVP r2SCAN (DF)", "ferrocene_atoms()", "def2-svp", 0, 0, "r2scan", "df"),
    "cu-r2scan-df": ("[Cu(NH3)4]2+ doublet / def2-TZVP r2SCAN (DF, UKS)", CU4, "def2-tzvp", 2, 1, "r2scan", "df"),
    "w5-pbe-ic": ("(H2O)5 / def2-TZVP PBE (in-core)", "water_cluster_atoms(5)", "def2-tzvp", 0, 0, "pbe", "incore"),
    "c8-b3lyp-d": ("C8H18 / 6-31G* B3LYP (direct)", "alkane_atoms(8)", "6-31g*", 0, 0, "b3lyp", "direct"),
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
if %(mode)r == "df":
    mf = mf.density_fit()
elif %(mode)r == "direct":
    mf.max_memory = 1
mf.verbose = 0; mf.conv_tol = 1e-9; mf.max_cycle = 100
if driver == "mojoscf":
    mojoscf.dft.accelerate(mf)
t0 = time.perf_counter(); mf.kernel(); tscf = time.perf_counter() - t0
g = mf.nuc_grad_method()
t0 = time.perf_counter(); de = g.kernel(); tgrad = time.perf_counter() - t0
print(json.dumps(dict(tscf=tscf, tgrad=tgrad, e=mf.e_tot, cycles=mf.cycles, conv=bool(mf.converged), de=de.tolist(),
                      nao=mol.nao_nr(), ngrid=int(mf.grids.weights.size), gclass=type(g).__name__)))
'''


def run(case, driver, bench_dir):
    _, atoms, basis, charge, spin, xc, mode = CASES[case]
    code = WORKER % dict(bench_dir=bench_dir, atoms=atoms, basis=basis, charge=charge, spin=spin, xc=xc, mode=mode,
                         driver=driver)
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
    print(f"{'system':50s} {'nao':>4s} {'grid':>7s} {'cyc':>6s} | {'SCF pyscf':>9s} {'mojoscf':>8s} {'x':>5s} | "
          f"{'grad pyscf':>10s} {'mojoscf':>8s} {'x':>5s} | {'|dE|':>7s} {'max|dg|':>8s}")
    for key in args.cases.split(","):
        name = CASES[key][0]
        ref = run(key, "pyscf", bench_dir)
        moj = run(key, "mojoscf", bench_dir)
        flag = "" if ref["conv"] and moj["conv"] else "  NOT CONVERGED"
        dg = abs(np.array(ref["de"]) - np.array(moj["de"])).max()
        print(f"{name:50s} {ref['nao']:4d} {ref['ngrid']:7d} {ref['cycles']:2d}/{moj['cycles']:<3d}| {ref['tscf']:9.1f} "
              f"{moj['tscf']:8.1f} {ref['tscf'] / moj['tscf']:4.1f}x | {ref['tgrad']:10.1f} {moj['tgrad']:8.1f} "
              f"{ref['tgrad'] / moj['tgrad']:4.1f}x | {abs(ref['e'] - moj['e']):7.1e} {dg:8.1e}{flag}", flush=True)


if __name__ == "__main__":
    main()
