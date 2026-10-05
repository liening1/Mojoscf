"""The Mojo integral engine against libcint (pyscf's ``mol.intor``).

Usage: python benchmarks/bench_integrals.py [--repeat N] [--cases i,j,...] [--scf]

For every molecule the one-electron matrices, the 8-fold packed electron
repulsion integrals and the density-fitting tensors are computed with both
engines; the maximum deviation and the best wall time of ``--repeat`` runs
are printed.  Both engines use all available cores (libcint through OpenMP,
the Mojo engine through its runtime).  With ``--scf`` the RHF energy and time
with either set of integrals is shown as well.

Large output arrays are allocated and touched once before the timed runs: on
virtual machines the first touch of a few GB of memory can cost seconds and
would otherwise be charged to whichever engine happens to run first.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
from pyscf import df, gto, lib

import mojoscf
from mojoscf import integrals as mi

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from systems import alkane_atoms, atoms_to_str, water_cluster_atoms  # noqa: E402

WATER = "O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587"
BENZENE = """
C  0.000  1.396  0; C  1.209  0.698  0; C  1.209 -0.698  0
C  0.000 -1.396  0; C -1.209 -0.698  0; C -1.209  0.698  0
H  0.000  2.479  0; H  2.147  1.240  0; H  2.147 -1.240  0
H  0.000 -2.479  0; H -2.147 -1.240  0; H -2.147  1.240  0
"""

CASES = [
    ("H2O / cc-pVDZ", WATER, "cc-pvdz", "cc-pvdz-jkfit"),
    ("H2O / aug-cc-pVTZ", WATER, "aug-cc-pvtz", "cc-pvtz-jkfit"),
    ("H2O / cc-pVQZ", WATER, "cc-pvqz", "cc-pvqz-jkfit"),
    ("benzene / cc-pVDZ", BENZENE, "cc-pvdz", "cc-pvdz-jkfit"),
    ("benzene / def2-TZVP", BENZENE, "def2-tzvp", "def2-universal-jkfit"),
    ("(H2O)5 / cc-pVDZ", atoms_to_str(water_cluster_atoms(5)), "cc-pvdz", "cc-pvdz-jkfit"),
    ("C8H18 / cc-pVDZ", atoms_to_str(alkane_atoms(8)), "cc-pvdz", "cc-pvdz-jkfit"),
]


def best(fn, repeat):
    t = []
    out = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        t.append(time.perf_counter() - t0)
    return out, min(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--cases", default=None, help="comma-separated case indices")
    ap.add_argument("--scf", action="store_true", help="also compare RHF timings with either integral engine")
    ap.add_argument("--no-eri", action="store_true", help="skip the four-index integrals (large bases)")
    args = ap.parse_args()
    cases = CASES if args.cases is None else [CASES[int(i)] for i in args.cases.split(",")]
    print(f"threads: pyscf/OpenMP {lib.num_threads()}, mojo {mojoscf.backend_info()['parallelism_level']}")
    print(f"{'system':22s} {'nao':>5s} {'naux':>5s} | {'1e mojo':>8s} {'1e cint':>8s} | {'eri mojo':>9s} {'eri cint':>9s} {'ratio':>5s} | "
          f"{'df mojo':>8s} {'df cint':>8s} {'ratio':>5s} | max|diff|")
    for label, atoms, basis, auxbasis in cases:
        mol = gto.M(atom=atoms, basis=basis, verbose=0)
        auxmol = df.addons.make_auxmol(mol, auxbasis)
        nao = mol.nao_nr()
        (s, t, v), t1m = best(lambda: mi.int1e(mol), args.repeat)
        ref1, t1c = best(lambda: (mol.intor("int1e_ovlp"), mol.intor("int1e_kin"), mol.intor("int1e_nuc")), args.repeat)
        d1 = max(abs(a - b).max() for a, b in zip((s, t, v), ref1))
        if args.no_eri or nao > 400:
            terim = teric = float("nan")
            d2 = float("nan")
        else:
            npair = nao * (nao + 1) // 2
            warm = np.empty(2 * (npair * (npair + 1) // 2))
            warm.fill(0.0)
            del warm
            eri, terim = best(lambda: mi.int2e_s8(mol), args.repeat)
            ref, teric = best(lambda: mol.intor("int2e", aosym="s8"), args.repeat)
            d2 = abs(eri - ref).max()
            del eri, ref
        cd, tdfm = best(lambda: mi.cholesky_eri(mol, auxmol=auxmol), args.repeat)
        cref, tdfc = best(lambda: df.incore.cholesky_eri(mol, auxmol=auxmol), args.repeat)
        d3 = abs(cd - cref).max()
        print(f"{label:22s} {nao:5d} {auxmol.nao_nr():5d} | {t1m*1e3:7.1f}ms {t1c*1e3:7.1f}ms | {terim:8.3f}s {teric:8.3f}s {terim/teric:5.2f} | "
              f"{tdfm:7.3f}s {tdfc:7.3f}s {tdfm/tdfc:5.2f} | 1e {d1:.0e} eri {d2:.0e} cderi {d3:.0e}", flush=True)
        if args.scf and not (args.no_eri or nao > 400):
            del cd, cref
            engine = mi.engine()
            mi.set_engine("libcint")     # the driver otherwise takes its ERIs from the Mojo engine
            mf = mojoscf.RHF(mol)
            mf.conv_tol = 1e-10
            t0 = time.perf_counter()
            mf.kernel()
            tl = time.perf_counter() - t0
            mi.set_engine(engine)
            e_l = mf.e_tot
            del mf               # release the ERI tensor before the second run
            mf2 = mojoscf.RHF(mol)
            mf2.conv_tol = 1e-10
            t0 = time.perf_counter()
            mi.attach(mf2)       # integrals from the Mojo engine, inside the timed region
            mf2.kernel()
            tm = time.perf_counter() - t0
            print(f"    RHF: libcint integrals {tl:7.2f}s  mojo integrals {tm:7.2f}s  "
                  f"E = {e_l:.10f} / {mf2.e_tot:.10f}  (|dE| = {abs(e_l - mf2.e_tot):.1e})", flush=True)
            del mf2


if __name__ == "__main__":
    main()
