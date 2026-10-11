"""Strongly contracted NEVPT2 with larger active spaces: pyscf versus mojoscf, each in its own process.

For each case the script converges a density-fitted ROHF/RHF reference
(conv_tol 1e-9; the same pyscf object, accelerated with ``mojoscf.accelerate``
for mojoscf), runs a CASCI with the ``ncas`` orbitals around the Fermi level
and pyscf's NEVPT2 (``mrpt.NEVPT``) of its ground state.  With ten or more
active orbitals pyscf's 3-particle density matrix and the 4-particle
contractions of NEVPT2 (C kernels over the FCI strings) dominate; with
mojoscf imported they come from :mod:`mojoscf.fcirdm` and the DF integrals
from :func:`mojoscf.casscf.nevpt2_eris`.  Reported: the CASCI and NEVPT2
times and the NEVPT2 correlation energies.  Usage:

    python benchmarks/bench_nevpt2.py [--cases a,b,...] [--list]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

FE6 = 'octahedral_atoms("Fe", "H2O", 2.12)'

# key: (label, atoms, basis, charge, spin, ncas, nelecas)
CASES = {
    "fc-10": ("ferrocene singlet / def2-SVP, CAS(10,10)", "ferrocene_atoms()", "def2-svp", 0, 0, 10, 10),
    "fe6-12": ("[Fe(H2O)6]2+ quintet / def2-SVP, CAS(12,12)", FE6, "def2-svp", 2, 4, 12, 12),
    "fc-12": ("ferrocene singlet / def2-SVP, CAS(12,12)", "ferrocene_atoms()", "def2-svp", 0, 0, 12, 12),
}

WORKER = r'''
import json, sys, time
sys.path.insert(0, %(bench_dir)r)
from pyscf import gto, mcscf, mrpt, scf
from systems import *
mol = gto.M(atom=atoms_to_str(%(atoms)s), basis=%(basis)r, charge=%(charge)d, spin=%(spin)d,
            verbose=0, max_memory=12000)
if %(driver)r == "mojoscf":
    import mojoscf
else:   # the reference must be plain pyscf: mojoscf patches some pyscf functions when imported
    assert "mojoscf" not in sys.modules
mf = (scf.ROHF(mol) if mol.spin else scf.RHF(mol)).density_fit()
mf.conv_tol = 1e-9; mf.max_cycle = 100
if %(driver)r == "mojoscf":
    mojoscf.accelerate(mf)
mf.kernel()
mc = mcscf.CASCI(mf, %(ncas)d, %(nelecas)d)
mc.fcisolver.max_memory = 12000
t0 = time.perf_counter(); mc.kernel(); tcas = time.perf_counter() - t0
t0 = time.perf_counter(); e2 = float(mrpt.NEVPT(mc).kernel()); tnev = time.perf_counter() - t0
print(json.dumps(dict(tcas=tcas, tnev=tnev, ecas=float(mc.e_tot), e2=e2, nao=mol.nao_nr())))
'''


def run(case, driver, bench_dir):
    _, atoms, basis, charge, spin, ncas, nelecas = CASES[case]
    code = WORKER % dict(bench_dir=bench_dir, atoms=atoms, basis=basis, charge=charge, spin=spin, ncas=ncas,
                         nelecas=nelecas, driver=driver)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"{case} / {driver} failed:\n{out.stderr[-3000:]}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        for k, v in CASES.items():
            print(f"{k:8s} {v[0]}")
        return
    cases = args.cases.split(",") if args.cases else list(CASES)
    bench_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"{'system':46s} {'nao':>4s} | {'CASCI pyscf':>11s} {'mojoscf':>8s} | {'NEVPT2 pyscf':>12s} {'mojoscf':>8s}"
          f" {'x':>6s} | {'|dE2|':>8s}", flush=True)
    for case in cases:
        ref = run(case, "pyscf", bench_dir)
        new = run(case, "mojoscf", bench_dir)
        print(f"{CASES[case][0]:46s} {new['nao']:4d} | {ref['tcas']:11.1f} {new['tcas']:8.1f} | {ref['tnev']:12.1f}"
              f" {new['tnev']:8.1f} {ref['tnev'] / new['tnev']:5.1f}x | {abs(ref['e2'] - new['e2']):8.1e}", flush=True)


if __name__ == "__main__":
    main()
