"""QM/MM electrostatic embedding (pyscf.qmmm): pyscf versus mojoscf, each run in its own process.

The QM solute sits in a sphere of TIP3P point-charge waters (``systems.mm_water_charges``,
about liquid density, waters closer than 2.6 A to a QM atom left out).  For
each case the script converges the QM/MM SCF (``qmmm.mm_charge``, conv_tol
1e-9; the same pyscf object accelerated with ``mojoscf.accelerate`` for
mojoscf), then times the nuclear gradient of the QM atoms
(``nuc_grad_method().kernel()``) followed by the forces on the MM charges
(``grad_hcore_mm(dm) + grad_nuc_mm()``), the two together being one MD
step's worth of forces (mojoscf obtains the MM forces in the gradient's
pass over the charge integrals).  The Kohn-Sham cases run pyscf's DFT with
only the MM-charge terms from mojoscf (``mojoscf.qmmm.attach``).  Usage:

    python benchmarks/bench_qmmm.py [--cases a,b,...] [--list]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

# key: (label, QM atoms, basis, MM radius [A], charge, spin, kind, mode, Gaussian radius [A] or None)
CASES = {
    "bz-15": ("benzene / cc-pVDZ, 15 A (DF)", "BENZENE_ATOMS", "cc-pvdz", 15.0, 0, 0, "rhf", "df", None),
    "bz-25": ("benzene / cc-pVDZ, 25 A (DF)", "BENZENE_ATOMS", "cc-pvdz", 25.0, 0, 0, "rhf", "df", None),
    "bz-25g": ("benzene / cc-pVDZ, 25 A Gaussian charges (DF)", "BENZENE_ATOMS", "cc-pvdz", 25.0, 0, 0, "rhf", "df", 0.8),
    "w5-20": ("(H2O)5 / aug-cc-pVDZ, 20 A (in-core)", "water_cluster_atoms(5)", "aug-cc-pvdz", 20.0, 0, 0, "rhf", "incore", None),
    "c8-20": ("C8H18 / 6-31G*, 20 A (direct)", "alkane_atoms(8)", "6-31g*", 20.0, 0, 0, "rhf", "direct", None),
    "fc-25": ("ferrocene / def2-SVP, 25 A (DF)", "ferrocene_atoms()", "def2-svp", 25.0, 0, 0, "rhf", "df", None),
    "w10p-20": ("(H2O)10+ / cc-pVDZ, 20 A (DF, UHF)", "water_cluster_atoms(10)", "cc-pvdz", 20.0, 1, 1, "uhf", "df", None),
    # Kohn-Sham: pyscf's SCF and QM gradient, MM-charge terms from mojoscf.qmmm.attach
    "bz-25-ks": ("benzene / def2-SVP B3LYP, 25 A (DF)", "BENZENE_ATOMS", "def2-svp", 25.0, 0, 0, "b3lyp", "df", None),
    "fc-25-ks": ("ferrocene / def2-SVP PBE, 25 A (DF)", "ferrocene_atoms()", "def2-svp", 25.0, 0, 0, "pbe", "df", None),
    "bzp-25-ks": ("benzene+ / def2-SVP B3LYP, 25 A (DF, UKS)", "BENZENE_ATOMS", "def2-svp", 25.0, 1, 1, "ub3lyp", "df", None),
}

WORKER = r'''
import json, sys, time
import numpy as np
sys.path.insert(0, %(bench_dir)r)
from pyscf import dft, gto, scf, qmmm
from systems import *
from bench_scf import BENZENE
BENZENE_ATOMS = [(a[0], tuple(a[1])) for a in gto.format_atom(BENZENE, unit=1.0)]
qm = %(atoms)s
coords, charges = mm_water_charges(qm, %(radius)r)
radii = None if %(gauss)r is None else np.full(len(charges), %(gauss)r)
mol = gto.M(atom=atoms_to_str(qm), basis=%(basis)r, charge=%(charge)d, spin=%(spin)d, verbose=0, max_memory=12000)
driver = %(driver)r
if driver == "mojoscf":
    import mojoscf
    mojoscf.UHF(gto.M(atom="H 0 0 0; H 0 0 1", basis="sto-3g", verbose=0)).run()  # start the runtime
kind = %(kind)r
if kind in ("rhf", "uhf"):
    mf = scf.RHF(mol) if kind == "rhf" else scf.UHF(mol)
else:
    mf = dft.UKS(mol, xc=kind[1:]) if kind.startswith("u") else dft.RKS(mol, xc=kind)
if %(mode)r == "df":
    mf = mf.density_fit()
elif %(mode)r == "direct":
    mf.max_memory = 1
mf = qmmm.mm_charge(mf, coords, charges, radii=radii)
mf.verbose = 0; mf.conv_tol = 1e-9; mf.max_cycle = 100
if driver == "mojoscf":
    if kind in ("rhf", "uhf"):
        mojoscf.accelerate(mf)          # native SCF loop, J/K, gradients and MM terms
    else:
        mojoscf.qmmm.attach(mf)         # MM-charge terms only
t0 = time.perf_counter(); mf.kernel(); tscf = time.perf_counter() - t0
g = mf.nuc_grad_method()
t0 = time.perf_counter(); de = g.kernel(); tgrad = time.perf_counter() - t0
dm = mf.make_rdm1()
if dm.ndim == 3:
    dm = dm[0] + dm[1]
t0 = time.perf_counter(); fmm = g.grad_hcore_mm(dm) + g.grad_nuc_mm(); tmm = time.perf_counter() - t0
print(json.dumps(dict(tscf=tscf, tgrad=tgrad, tmm=tmm, e=mf.e_tot, cycles=mf.cycles, conv=bool(mf.converged),
                      de=de.tolist(), fmm=fmm.tolist(), nao=mol.nao_nr(), nmm=len(charges))))
'''


def run(case, driver, bench_dir):
    _, atoms, basis, radius, charge, spin, kind, mode, gauss = CASES[case]
    code = WORKER % dict(bench_dir=bench_dir, atoms=atoms, basis=basis, radius=radius, charge=charge, spin=spin,
                         kind=kind, mode=mode, gauss=gauss, driver=driver)
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
        print("\n".join(f"{k:8s} {v[0]}" for k, v in CASES.items()))
        return
    bench_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"{'system':46s} {'nao':>4s} {'MM q':>6s} | {'SCF pyscf':>9s} {'mojoscf':>8s} {'x':>5s} | {'grad pyscf':>10s} "
          f"{'(MM f)':>6s} {'mojoscf':>8s} {'x':>5s} | {'|dE|':>7s} {'max|dg|':>8s} {'max|dF|':>8s}")
    for key in args.cases.split(","):
        name = CASES[key][0]
        ref = run(key, "pyscf", bench_dir)
        moj = run(key, "mojoscf", bench_dir)
        flag = "" if ref["conv"] and moj["conv"] else "  NOT CONVERGED"
        dg = abs(np.array(ref["de"]) - np.array(moj["de"])).max()
        df = abs(np.array(ref["fmm"]) - np.array(moj["fmm"])).max()
        tr = ref["tgrad"] + ref["tmm"]
        tm = moj["tgrad"] + moj["tmm"]
        print(f"{name:46s} {ref['nao']:4d} {ref['nmm']:6d} | {ref['tscf']:9.1f} {moj['tscf']:8.1f} {ref['tscf'] / moj['tscf']:4.1f}x | "
              f"{tr:10.2f} {ref['tmm']:6.2f} {tm:8.2f} {tr / tm:4.1f}x | "
              f"{abs(ref['e'] - moj['e']):7.1e} {dg:8.1e} {df:8.1e}{flag}", flush=True)


if __name__ == "__main__":
    main()
