"""Wall-clock comparison of pyscf's RHF driver and the mojoscf driver.

Usage: python benchmarks/bench_scf.py [--repeat N] [--threads N]

The two-electron integrals are computed by the same pyscf C code in both
cases, so the difference is the Python/NumPy glue that mojoscf moves to Mojo.
Both drivers start from the same initial guess and use identical settings.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
from pyscf import gto, lib, scf

import mojoscf

WATER = "O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587"
BENZENE = """
C  0.000  1.396  0; C  1.209  0.698  0; C  1.209 -0.698  0
C  0.000 -1.396  0; C -1.209 -0.698  0; C -1.209  0.698  0
H  0.000  2.479  0; H  2.147  1.240  0; H  2.147 -1.240  0
H  0.000 -2.479  0; H -2.147 -1.240  0; H -2.147  1.240  0
"""
WATER3 = """
O -1.551 -0.114 0.000; H -1.934  0.762 0.000; H -0.599  0.040 0.000
O  1.350 -0.111 0.000; H  1.680  0.373 -0.758; H  1.680  0.373  0.758
O  0.000  2.300 0.000; H  0.000  2.900  0.758; H  0.000  2.900 -0.758
"""

CASES = [
    ("H2O / sto-3g", WATER, "sto-3g"),
    ("H2O / cc-pVDZ", WATER, "cc-pvdz"),
    ("H2O / cc-pVTZ", WATER, "cc-pvtz"),
    ("benzene / sto-3g", BENZENE, "sto-3g"),
    ("benzene / cc-pVDZ", BENZENE, "cc-pvdz"),
    ("(H2O)3 / cc-pVDZ", WATER3, "cc-pvdz"),
]


def run(factory, mol, repeat):
    best = float("inf")
    for _ in range(repeat):
        mf = factory(mol)
        mf.verbose = 0
        mf.conv_tol = 1e-10
        t0 = time.perf_counter()
        mf.kernel()
        best = min(best, time.perf_counter() - t0)
    return best, mf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--cases", type=str, default=None, help="comma separated indices into CASES")
    args = ap.parse_args()
    if args.threads:
        lib.num_threads(args.threads)
    info = mojoscf.backend_info()
    print(f"mojoscf {mojoscf.__version__}; BLAS: {info['blas_library'] or 'native Mojo fallback'}; "
          f"threads: pyscf={lib.num_threads()} mojo={info['parallelism_level']}")
    cases = CASES if args.cases is None else [CASES[int(i)] for i in args.cases.split(",")]
    print(f"{'system':22s} {'nao':>5s} {'cycles':>6s} {'pyscf [s]':>10s} {'mojoscf [s]':>12s} {'speedup':>8s} {'|dE|':>9s}")
    for name, atom, basis in cases:
        mol = gto.M(atom=atom, basis=basis, verbose=0)
        t_ref, mf_ref = run(scf.RHF, mol, args.repeat)
        t_mojo, mf_mojo = run(mojoscf.RHF, mol, args.repeat)
        de = abs(mf_ref.e_tot - mf_mojo.e_tot)
        print(f"{name:22s} {mol.nao_nr():5d} {mf_ref.cycles:3d}/{mf_mojo.cycles:<3d} {t_ref:10.3f} {t_mojo:12.3f} "
              f"{t_ref / t_mojo:7.2f}x {de:9.1e}")


if __name__ == "__main__":
    main()
