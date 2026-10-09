"""SCF stability analysis (``mf.stability()``): pyscf versus mojoscf, each in its own process.

For each (system, driver) pair the script converges the SCF (conv_tol 1e-9,
pyscf's default grids; ``mojoscf.dft.accelerate`` for mojoscf, whose
``mf.stability()`` is :mod:`mojoscf.stability`) and runs pyscf's default
internal analysis plus, for restricted references, the external one (real ->
complex and RHF -> UHF).  Reported: the time of ``mf.stability``, the SCF
time and the lowest eigenvalue of each analysis from the log.  Usage:

    python benchmarks/bench_stability.py [--cases a,b,...] [--list]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

CU4 = 'square_planar_atoms("Cu", ["NH3"] * 4, [2.03] * 4)'

# key: (label, atoms, basis, charge, spin, xc, density fitting)
CASES = {
    "fc-pbe0": ("ferrocene / def2-SVP PBE0 (RKS)", "ferrocene_atoms()", "def2-svp", 0, 0, "pbe0", True),
    "fc-pbe0-x": ("ferrocene / def2-SVP PBE0 (RKS)", "ferrocene_atoms()", "def2-svp", 0, 0, "pbe0", False),
    "cu-b3lyp": ("[Cu(NH3)4]2+ doublet / def2-TZVP B3LYP (UKS)", CU4, "def2-tzvp", 2, 1, "b3lyp", True),
    "fe-b3lyp": ("[Fe(H2O)6]2+ quintet / def2-SVP B3LYP (UKS)", 'octahedral_atoms("Fe", "H2O", 2.12)',
                 "def2-svp", 2, 4, "b3lyp", True),
}

WORKER = r'''
import json, re, sys, time
import numpy as np
sys.path.insert(0, %(bench_dir)r)
from pyscf import dft, gto, lib
from systems import *
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
import io
buf = io.StringIO()
mf.stdout = buf
mf.verbose = 4
t0 = time.perf_counter()
out = mf.stability(internal=True, external=not %(spin)d, return_status=True)
tstab = time.perf_counter() - t0
eigs = {m.group(1): [float(v) for v in m.group(2).replace("[", " ").replace("]", " ").split()][:1]
        for m in re.finditer(r"(\w+): lowest eigs of H = (\[[^\]]*\]|\S+)", buf.getvalue())}
print(json.dumps(dict(tscf=tscf, tstab=tstab, stable=[bool(s) if s is not None else None for s in out[2:]],
                      eigs=eigs, nao=mol.nao_nr(), conv=bool(mf.converged))))
'''


def run(case, driver, bench_dir):
    _, atoms, basis, charge, spin, xc, df = CASES[case]
    code = WORKER % dict(bench_dir=bench_dir, atoms=atoms, basis=basis, charge=charge, spin=spin, xc=xc, df=df,
                         driver=driver)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"{case} / {driver} failed:\n{out.stderr[-3000:]}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def main():
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        print("\n".join(f"{k:12s} {v[0]}{'' if v[6] else ', exact integrals'}" for k, v in CASES.items()))
        return
    bench_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"{'system':46s} {'ints':>5s} {'nao':>4s} | {'stab pyscf':>10s} {'mojoscf':>8s} {'x':>5s} | "
          f"{'SCF pyscf':>9s} {'mojoscf':>8s} | lowest eigenvalues (pyscf / mojoscf)")
    for key in args.cases.split(","):
        name, df = CASES[key][0], CASES[key][6]
        ref = run(key, "pyscf", bench_dir)
        moj = run(key, "mojoscf", bench_dir)
        eigs = "  ".join(f"{k} {ref['eigs'].get(k, ['-'])[0]:.6g} / {moj['eigs'].get(k, ['-'])[0]:.6g}"
                         for k in sorted(set(ref["eigs"]) | set(moj["eigs"])))
        flag = "" if ref["stable"] == moj["stable"] else f"  STABILITY DIFFERS {ref['stable']} {moj['stable']}"
        print(f"{name:46s} {'DF' if df else 'exact':>5s} {ref['nao']:4d} | {ref['tstab']:10.1f} {moj['tstab']:8.1f} "
              f"{ref['tstab'] / moj['tstab']:4.1f}x | {ref['tscf']:9.1f} {moj['tscf']:8.1f} | {eigs}{flag}", flush=True)


if __name__ == "__main__":
    main()
