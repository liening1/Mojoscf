"""Nuclear gradients: pyscf's ``grad.rhf``/``grad.uhf``/``df.grad`` versus ``mojoscf.grad``, each in its own process.

Both processes converge the same pyscf SCF object (conv_tol 1e-11) and time
only the gradient, ``mf.nuc_grad_method().kernel()`` for pyscf and
``mojoscf.grad.Gradients(mf).kernel()`` (``UGradients`` for UHF,
``DFGradients``/``DFUGradients`` for the density-fitted "(DF)" cases) for
mojoscf, so the comparison isolates the derivative integrals and their
contraction.
The script reports both wall times, the time of the two-electron part alone
and the largest difference between the two gradients.  Usage:

    python benchmarks/bench_grad.py [--cases a,b,...] [--list]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

CASES = {
    "w-tz": ("H2O / cc-pVTZ", "H2O", "cc-pvtz", 0, 0, "rhf", False),
    "bz": ("benzene / cc-pVDZ", "BENZENE", "cc-pvdz", 0, 0, "rhf", False),
    "bz+": ("benzene cation / cc-pVDZ (UHF)", "BENZENE", "cc-pvdz", 1, 1, "uhf", False),
    "C8": ("C8H18 / cc-pVDZ", "atoms_to_str(alkane_atoms(8))", "cc-pvdz", 0, 0, "rhf", False),
    "w5": ("(H2O)5 / aug-cc-pVDZ", "atoms_to_str(water_cluster_atoms(5))", "aug-cc-pvdz", 0, 0, "rhf", False),
    "w10": ("(H2O)10 / cc-pVDZ", "atoms_to_str(water_cluster_atoms(10))", "cc-pvdz", 0, 0, "rhf", False),
    "bzt": ("benzene / def2-TZVP", "BENZENE", "def2-tzvp", 0, 0, "rhf", False),
    "bz-df": ("benzene / cc-pVDZ (DF)", "BENZENE", "cc-pvdz", 0, 0, "rhf", True),
    "C8-df": ("C8H18 / cc-pVDZ (DF)", "atoms_to_str(alkane_atoms(8))", "cc-pvdz", 0, 0, "rhf", True),
    "w10-df": ("(H2O)10 / cc-pVDZ (DF)", "atoms_to_str(water_cluster_atoms(10))", "cc-pvdz", 0, 0, "rhf", True),
    "w10+df": ("(H2O)10 cation / cc-pVDZ (DF, UHF)", "atoms_to_str(water_cluster_atoms(10))", "cc-pvdz", 1, 1, "uhf", True),
    "bzt-df": ("benzene / def2-TZVP (DF)", "BENZENE", "def2-tzvp", 0, 0, "rhf", True),
    "C20-df": ("C20H42 / 6-31G (DF)", "atoms_to_str(alkane_atoms(20))", "6-31g", 0, 0, "rhf", True),
}

WORKER = r'''
import json, sys, time
import numpy as np
sys.path.insert(0, %(bench_dir)r)
from pyscf import gto, scf
from systems import *
from bench_scf import BENZENE
H2O = "O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587"
mol = gto.M(atom=%(atoms)s, basis=%(basis)r, charge=%(charge)d, spin=%(spin)d, verbose=0, max_memory=12000)
driver = %(driver)r
cls = scf.RHF if %(kind)r == "rhf" else scf.UHF
mf = cls(mol)
if %(df)r:
    mf = mf.density_fit()
mf.conv_tol = 1e-11
mf.kernel()
if driver == "mojoscf":
    import mojoscf
    small = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mojoscf.grad.Gradients(scf.RHF(small).run()).kernel()   # start the runtime
    if %(df)r:
        g = (mojoscf.grad.DFUGradients if %(kind)r == "uhf" else mojoscf.grad.DFGradients)(mf)
    else:
        g = (mojoscf.grad.UGradients if %(kind)r == "uhf" else mojoscf.grad.Gradients)(mf)
else:
    g = mf.nuc_grad_method()
dm = mf.make_rdm1()
t0 = time.perf_counter(); de = g.kernel(); total = time.perf_counter() - t0
t0 = time.perf_counter()
if driver == "mojoscf":
    g.grad_2e(dm)
else:
    g.get_veff(mol, dm)     # includes the auxiliary-basis response for DF
t2e = time.perf_counter() - t0
print(json.dumps(dict(total=total, t2e=t2e, de=de.tolist(), nao=mol.nao_nr())))
'''


def run(case, driver, bench_dir):
    _, atoms, basis, charge, spin, kind, df = CASES[case]
    code = WORKER % dict(bench_dir=bench_dir, atoms=atoms, basis=basis, charge=charge, spin=spin,
                         driver=driver, kind=kind, df=df)
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
        print("\n".join(f"{k:6s} {v[0]}" for k, v in CASES.items()))
        return
    bench_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"{'system':36s} {'nao':>4s} | {'pyscf [s]':>9s} {'(2e)':>6s} | {'mojoscf [s]':>11s} {'(2e)':>6s} | "
          f"{'speedup':>7s} {'max|dg|':>8s}")
    for key in args.cases.split(","):
        name = CASES[key][0]
        ref = run(key, "pyscf", bench_dir)
        moj = run(key, "mojoscf", bench_dir)
        dg = abs(np.array(ref["de"]) - np.array(moj["de"])).max()
        print(f"{name:36s} {ref['nao']:4d} | {ref['total']:9.2f} {ref['t2e']:6.2f} | {moj['total']:11.2f} {moj['t2e']:6.2f} | "
              f"{ref['total'] / moj['total']:6.2f}x {dg:8.1e}", flush=True)


if __name__ == "__main__":
    main()
