"""Large systems: pyscf driver versus mojoscf driver, each run in its own process.

Density-fitted RHF/UHF, in-core non-DF cases and direct-SCF cases (run with
``max_memory=1`` so that neither code keeps the 4-index integrals: pyscf
recomputes them with libcint/libcvhf every cycle, mojoscf with its own engine).  Running every (system, driver)
pair in a fresh process keeps the timings independent of one another (BLAS thread
pools, memory, caches).  The script reports the total wall time, the time inside
``mf.get_veff`` for pyscf (for mojoscf the two-electron part is built natively and
is included in "glue"), and the agreement of the energies.  Usage:

    python benchmarks/bench_large.py [--cases a,b,...] [--list]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

CASES = {
    "C10H22": ("C10H22 / 6-31G* (DF)", "alkane_atoms(10)", "6-31g*", 0, 0, "rhf", True),
    "C20H42": ("C20H42 / 6-31G (DF)", "alkane_atoms(20)", "6-31g", 0, 0, "rhf", True),
    "w10": ("(H2O)10 / cc-pVDZ (DF)", "water_cluster_atoms(10)", "cc-pvdz", 0, 0, "rhf", True),
    "C60": ("C60 / STO-3G (DF)", "c60_atoms()", "sto-3g", 0, 0, "rhf", True),
    "w20": ("(H2O)20 / cc-pVDZ (DF)", "water_cluster_atoms(20)", "cc-pvdz", 0, 0, "rhf", True),
    "C20H41": ("C20H41 radical / 6-31G (DF, UHF)", "alkane_atoms(20)[:-1]", "6-31g", 0, 1, "uhf", True),
    "w10+": ("(H2O)10 cation / cc-pVDZ (DF, UHF)", "water_cluster_atoms(10)", "cc-pvdz", 1, 1, "uhf", True),
    "bz": ("benzene / cc-pVDZ (in-core)", "BENZENE", "cc-pvdz", 0, 0, "rhf", False),
    "bz+": ("benzene cation / cc-pVDZ (UHF)", "BENZENE", "cc-pvdz", 1, 1, "uhf", False),
    "w5": ("(H2O)5 / cc-pVDZ (in-core)", "water_cluster_atoms(5)", "cc-pvdz", 0, 0, "rhf", False),
    "bz-d": ("benzene / cc-pVDZ (direct)", "BENZENE", "cc-pvdz", 0, 0, "rhf", "direct"),
    "C8-d": ("C8H18 / cc-pVDZ (direct)", "alkane_atoms(8)", "cc-pvdz", 0, 0, "rhf", "direct"),
    "w10-d": ("(H2O)10 / cc-pVDZ (direct)", "water_cluster_atoms(10)", "cc-pvdz", 0, 0, "rhf", "direct"),
    "w5+-d": ("(H2O)5 cation / cc-pVDZ (direct, UHF)", "water_cluster_atoms(5)", "cc-pvdz", 1, 1, "uhf", "direct"),
    "bzt-d": ("benzene / def2-TZVP (direct)", "BENZENE", "def2-tzvp", 0, 0, "rhf", "direct"),
}

WORKER = r'''
import json, sys, time
sys.path.insert(0, %(bench_dir)r)
from pyscf import gto, lib, scf
from systems import *
from bench_scf import BENZENE
atoms = %(atoms)s
atom = atoms if isinstance(atoms, str) else atoms_to_str(atoms)
mol = gto.M(atom=atom, basis=%(basis)r, charge=%(charge)d, spin=%(spin)d, verbose=0, max_memory=12000)
driver = %(driver)r
if driver == "mojoscf":
    import mojoscf
    mojoscf.UHF(gto.M(atom="H 0 0 0; H 0 0 1", basis="sto-3g", verbose=0)).run()  # start the runtime
cls = scf.RHF if %(kind)r == "rhf" else scf.UHF
mf = cls(mol)
if %(df)r == "direct":
    mf.max_memory = 1          # the 4-index integrals are recomputed every cycle
elif %(df)r:
    mf = mf.density_fit()
mf.verbose = 0; mf.conv_tol = 1e-9; mf.max_cycle = 100
veff = [0.0]
if driver == "mojoscf":
    mojoscf.accelerate(mf)   # J/K built natively: nothing to time separately
else:
    orig = mf.get_veff
    def timed(*a, **k):
        t = time.perf_counter(); r = orig(*a, **k); veff[0] += time.perf_counter() - t; return r
    mf.get_veff = timed      # (an instance-level get_veff would disable mojoscf's native J/K)
t0 = time.perf_counter(); mf.kernel(); total = time.perf_counter() - t0
print(json.dumps(dict(total=total, veff=veff[0], e=mf.e_tot, cycles=mf.cycles, conv=bool(mf.converged),
                      nao=mol.nao_nr(), mode=mf.scf_summary.get("mojoscf_veff_mode", -1))))
'''


def run(case, driver, bench_dir):
    name, atoms, basis, charge, spin, kind, df = CASES[case]
    code = WORKER % dict(bench_dir=bench_dir, atoms=atoms, basis=basis, charge=charge, spin=spin,
                         driver=driver, kind=kind, df=df)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def main():
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        print("\n".join(f"{k:8s} {v[0]}" for k, v in CASES.items()))
        return
    bench_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"{'system':36s} {'nao':>4s} {'cyc':>6s} | {'pyscf [s]':>9s} {'(veff)':>7s} | {'mojoscf [s]':>11s} {'mode':>4s} | {'speedup':>7s} {'|dE| [Eh]':>9s}")
    for key in args.cases.split(","):
        name = CASES[key][0]
        ref = run(key, "pyscf", bench_dir)
        moj = run(key, "mojoscf", bench_dir)
        flag = "" if ref["conv"] and moj["conv"] else "  NOT CONVERGED"
        print(f"{name:36s} {ref['nao']:4d} {ref['cycles']:2d}/{moj['cycles']:<3d}| {ref['total']:9.1f} {ref['veff']:7.1f} | "
              f"{moj['total']:11.1f} {moj['mode']:4d} | {ref['total'] / moj['total']:6.2f}x {abs(ref['e'] - moj['e']):9.1e}{flag}",
              flush=True)


if __name__ == "__main__":
    main()
