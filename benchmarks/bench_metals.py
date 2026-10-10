"""Transition-metal complexes: SCF and nuclear gradient, pyscf versus mojoscf, each run in its own process.

The systems are idealised geometries (``systems.py``): ferrocene, high-spin
[Fe(H2O)6]2+ (alone and with a second shell of 12 waters), [Cu(NH3)4]2+,
Ni(CO)4 and cisplatin (Pt with the def2 effective core potential), with
def2 basis sets (an f shell on the metal already in def2-SVP).  "DF" runs use
pyscf's default def2 JK-fitting basis ("DF on disk": with pyscf's default
``max_memory`` of 4000 MB, too little for the tensor, which pyscf then keeps
in a file), "direct" runs ``max_memory=1`` (the 4-index integrals are
recomputed every cycle), "in-core" stores them.  For
each (system, driver) pair the script converges the SCF (conv_tol 1e-9; the
same pyscf object, accelerated with ``mojoscf.accelerate`` for mojoscf) and
then computes the gradient with ``nuc_grad_method()``.  Usage:

    python benchmarks/bench_metals.py [--cases a,b,...] [--list]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

FE6 = 'octahedral_atoms("Fe", "H2O", 2.12)'
FE18 = 'solvated_ion_atoms("Fe", 12)'
CU4 = 'square_planar_atoms("Cu", ["NH3"] * 4, [2.03] * 4)'
NI4 = 'tetrahedral_atoms("Ni", "CO", 1.83)'
CPT = 'square_planar_atoms("Pt", ["NH3", "NH3", "Cl", "Cl"], [2.05, 2.05, 2.32, 2.32])'

# key: (label, atoms, basis, ecp element, charge, spin, kind, mode)
CASES = {
    "fc-svp-df": ("ferrocene / def2-SVP (DF)", "ferrocene_atoms()", "def2-svp", None, 0, 0, "rhf", "df"),
    "fc-tz-df": ("ferrocene / def2-TZVP (DF)", "ferrocene_atoms()", "def2-tzvp", None, 0, 0, "rhf", "df"),
    "fe6-tz-df": ("[Fe(H2O)6]2+ quintet / def2-TZVP (DF, UHF)", FE6, "def2-tzvp", None, 2, 4, "uhf", "df"),
    "fe18-svp-df": ("[Fe(H2O)6]2+ 12H2O quintet / def2-SVP (DF, UHF)", FE18, "def2-svp", None, 2, 4, "uhf", "df"),
    "cu-tz-df": ("[Cu(NH3)4]2+ doublet / def2-TZVP (DF, UHF)", CU4, "def2-tzvp", None, 2, 1, "uhf", "df"),
    "ni-tz-df": ("Ni(CO)4 / def2-TZVP (DF)", NI4, "def2-tzvp", None, 0, 0, "rhf", "df"),
    "cpt-tz-df": ("cisplatin / def2-TZVP (DF, Pt ECP)", CPT, "def2-tzvp", "Pt", 0, 0, "rhf", "df"),
    "fc-svp-d": ("ferrocene / def2-SVP (direct)", "ferrocene_atoms()", "def2-svp", None, 0, 0, "rhf", "direct"),
    "fe6-svp-d": ("[Fe(H2O)6]2+ quintet / def2-SVP (direct, UHF)", FE6, "def2-svp", None, 2, 4, "uhf", "direct"),
    "cpt-svp-d": ("cisplatin / def2-SVP (direct, Pt ECP)", CPT, "def2-svp", "Pt", 0, 0, "rhf", "direct"),
    "ni-svp-ic": ("Ni(CO)4 / def2-SVP (in-core)", NI4, "def2-svp", None, 0, 0, "rhf", "incore"),
    "fep-svp-df": ("Fe(II) porphine triplet / def2-SVP (DF, UHF)", 'porphyrin_atoms("Fe")', "def2-svp", None, 0, 2,
                   "uhf", "df"),
    # the DF tensor (about 5 GB) does not fit in pyscf's default max_memory of 4000 MB: kept on disk
    "fep-tz-disk": ("Fe(II) porphine triplet / def2-TZVP (DF on disk, UHF)", 'porphyrin_atoms("Fe")', "def2-tzvp",
                    None, 0, 2, "uhf", "df-disk"),
}

WORKER = r'''
import json, sys, time
import numpy as np
sys.path.insert(0, %(bench_dir)r)
from pyscf import gto, scf
from systems import *
ecp = {%(ecp)r: %(basis)r} if %(ecp)r else None
mol = gto.M(atom=atoms_to_str(%(atoms)s), basis=%(basis)r, ecp=ecp, charge=%(charge)d, spin=%(spin)d,
            verbose=0, max_memory=12000)
driver = %(driver)r
if driver == "mojoscf":
    import mojoscf
    mojoscf.UHF(gto.M(atom="H 0 0 0; H 0 0 1", basis="sto-3g", verbose=0)).run()  # start the runtime
cls = scf.RHF if %(kind)r == "rhf" else scf.UHF
mf = cls(mol)
if %(mode)r in ("df", "df-disk"):
    mf = mf.density_fit()
    if %(mode)r == "df-disk":
        mf.with_df.max_memory = 4000
elif %(mode)r == "direct":
    mf.max_memory = 1
mf.verbose = 0; mf.conv_tol = 1e-9; mf.max_cycle = 100
if driver == "mojoscf":
    mojoscf.accelerate(mf)
t0 = time.perf_counter(); mf.kernel(); tscf = time.perf_counter() - t0
g = mf.nuc_grad_method()
t0 = time.perf_counter(); de = g.kernel(); tgrad = time.perf_counter() - t0
print(json.dumps(dict(tscf=tscf, tgrad=tgrad, e=mf.e_tot, cycles=mf.cycles, conv=bool(mf.converged), de=de.tolist(),
                      nao=mol.nao_nr(), mode=mf.scf_summary.get("mojoscf_veff_mode", -1), gclass=type(g).__name__)))
'''


def run(case, driver, bench_dir):
    _, atoms, basis, ecp, charge, spin, kind, mode = CASES[case]
    code = WORKER % dict(bench_dir=bench_dir, atoms=atoms, basis=basis, ecp=ecp, charge=charge, spin=spin,
                         kind=kind, mode=mode, driver=driver)
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
        print("\n".join(f"{k:12s} {v[0]}" for k, v in CASES.items()))
        return
    bench_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"{'system':48s} {'nao':>4s} {'cyc':>6s} | {'SCF pyscf':>9s} {'mojoscf':>8s} {'x':>5s} | "
          f"{'grad pyscf':>10s} {'mojoscf':>8s} {'x':>5s} | {'|dE|':>7s} {'max|dg|':>8s}")
    for key in args.cases.split(","):
        name = CASES[key][0]
        ref = run(key, "pyscf", bench_dir)
        moj = run(key, "mojoscf", bench_dir)
        flag = "" if ref["conv"] and moj["conv"] else "  NOT CONVERGED"
        dg = abs(np.array(ref["de"]) - np.array(moj["de"])).max()
        print(f"{name:48s} {ref['nao']:4d} {ref['cycles']:2d}/{moj['cycles']:<3d}| {ref['tscf']:9.1f} {moj['tscf']:8.1f} "
              f"{ref['tscf'] / moj['tscf']:4.1f}x | {ref['tgrad']:10.1f} {moj['tgrad']:8.1f} {ref['tgrad'] / moj['tgrad']:4.1f}x | "
              f"{abs(ref['e'] - moj['e']):7.1e} {dg:8.1e}{flag}", flush=True)


if __name__ == "__main__":
    main()
