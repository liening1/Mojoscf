"""Analytical Hessians: pyscf versus mojoscf (dft.accelerate + mojoscf.hessian), each in its own process.

For each system the script converges the Kohn-Sham SCF (density fitting or
exact integrals as the case says; conv_tol 1e-10, pyscf's default grids;
accelerated with ``mojoscf.dft.accelerate`` for mojoscf, whose
``mf.Hessian()`` then carries the Mojo kernels) and computes the nuclear
Hessian with pyscf's driver (auxiliary-basis response included, pyscf's
default for DF Hessians).
Reported: the Hessian time, the SCF time, the largest difference of the
Hessian elements and of the harmonic frequencies.  Usage:

    python benchmarks/bench_hessian.py [--cases a,b,...] [--list] [--cache FILE] [--drivers pyscf,mojoscf]

``--cache`` keeps pyscf's results in a JSON file and reuses them (its
Hessians of the larger systems take tens of minutes; mojoscf always runs);
``--drivers pyscf`` only fills it.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

CUCL4 = 'square_planar_atoms("Cu", ["Cl"] * 4, [2.25] * 4)'

# key: (label, atoms, basis, charge, spin, xc, density fitting)
CASES = {
    "bz-b3lyp": ("benzene / def2-SVP B3LYP", "BENZENE_ATOMS", "def2-svp", 0, 0, "b3lyp", True),
    "fc-pbe0": ("ferrocene / def2-SVP PBE0", "ferrocene_atoms()", "def2-svp", 0, 0, "pbe0", True),
    "fc-pbe": ("ferrocene / def2-SVP PBE", "ferrocene_atoms()", "def2-svp", 0, 0, "pbe", True),
    "fc-camb3lyp": ("ferrocene / def2-SVP CAM-B3LYP", "ferrocene_atoms()", "def2-svp", 0, 0, "camb3lyp", True),
    "cucl4-b3lyp": ("[CuCl4]2- doublet / def2-SVP B3LYP (UKS)", CUCL4, "def2-svp", -2, 1, "b3lyp", True),
    # exact integrals (pyscf's default)
    "bz-b3lyp-x": ("benzene / def2-SVP B3LYP", "BENZENE_ATOMS", "def2-svp", 0, 0, "b3lyp", False),
    "bz-camb3lyp-x": ("benzene / def2-SVP CAM-B3LYP", "BENZENE_ATOMS", "def2-svp", 0, 0, "camb3lyp", False),
    "cucl4-b3lyp-x": ("[CuCl4]2- doublet / def2-SVP B3LYP (UKS)", CUCL4, "def2-svp", -2, 1, "b3lyp", False),
    "fc-pbe0-x": ("ferrocene / def2-SVP PBE0", "ferrocene_atoms()", "def2-svp", 0, 0, "pbe0", False),
}

WORKER = r'''
import json, sys, time
import numpy as np
sys.path.insert(0, %(bench_dir)r)
from pyscf import dft, gto
from pyscf.hessian import thermo
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
mf.verbose = 0; mf.conv_tol = 1e-10; mf.max_cycle = 100
if driver == "mojoscf":
    mojoscf.dft.accelerate(mf)
t0 = time.perf_counter(); mf.kernel(); tscf = time.perf_counter() - t0
h = mf.Hessian()
t0 = time.perf_counter(); hess = h.kernel(); thess = time.perf_counter() - t0
freq = thermo.harmonic_analysis(mol, hess)["freq_wavenumber"]
print(json.dumps(dict(tscf=tscf, thess=thess, hess=np.asarray(hess).ravel().tolist(),
                      freq=np.real(np.asarray(freq)).tolist(), conv=bool(mf.converged), nao=mol.nao_nr(),
                      hclass=type(h).__name__)))
'''


def run(case, driver, bench_dir):
    _, atoms, basis, charge, spin, xc, df = CASES[case]
    code = WORKER % dict(bench_dir=bench_dir, atoms=atoms, basis=basis, charge=charge, spin=spin, xc=xc,
                         driver=driver, df=df)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def main():
    import os

    import numpy as np

    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--drivers", default="pyscf,mojoscf")
    args = ap.parse_args()
    cache = {}
    if args.cache and os.path.exists(args.cache):
        with open(args.cache) as f:
            cache = json.load(f)

    def result(key, driver):
        if driver != "pyscf":
            return run(key, driver, bench_dir)
        tag = f"{key}/{driver}"
        if tag not in cache:
            cache[tag] = run(key, driver, bench_dir)
            if args.cache:
                with open(args.cache, "w") as f:
                    json.dump(cache, f)
        return cache[tag]

    if args.list:
        print("\n".join(f"{k:14s} {v[0]}{'' if v[6] else ', exact integrals'}" for k, v in CASES.items()))
        return
    bench_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"{'system':42s} {'ints':>5s} {'nao':>4s} | {'Hess pyscf':>10s} {'mojoscf':>8s} {'x':>5s} | "
          f"{'SCF pyscf':>9s} {'mojoscf':>8s} | {'max|dH|':>8s} {'max|dfreq|':>10s}")
    drivers = args.drivers.split(",")
    for key in args.cases.split(","):
        name, ints = CASES[key][0], "DF" if CASES[key][6] else "exact"
        ref = result(key, "pyscf")
        if "mojoscf" not in drivers:
            print(f"{name:42s} {ints:>5s} {ref['nao']:4d} | {ref['thess']:10.1f}", flush=True)
            continue
        moj = result(key, "mojoscf")
        flag = "" if ref["conv"] and moj["conv"] else "  NOT CONVERGED"
        dh = abs(np.array(ref["hess"]) - np.array(moj["hess"])).max()
        n = min(len(ref["freq"]), len(moj["freq"]))
        df = abs(np.array(ref["freq"][:n]) - np.array(moj["freq"][:n])).max()
        print(f"{name:42s} {ints:>5s} {ref['nao']:4d} | {ref['thess']:10.1f} {moj['thess']:8.1f} "
              f"{ref['thess'] / moj['thess']:4.1f}x | {ref['tscf']:9.1f} {moj['tscf']:8.1f} | "
              f"{dh:8.1e} {df:8.2f}cm-1{flag}", flush=True)


if __name__ == "__main__":
    main()
