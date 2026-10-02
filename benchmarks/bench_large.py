"""Large systems: density-fitted RHF/UHF, pyscf driver versus mojoscf driver.

Density fitting keeps the two-electron part cheap enough that the SCF glue
(DIIS, diagonalisation, density, ...) remains visible; the integral/J/K code is
identical in both drivers.  Besides the total time the script reports the "glue"
time, i.e. total minus the time spent inside ``mf.get_veff``, which is the part
mojoscf replaces and which is not affected by run-to-run noise of the shared
J/K code.  Usage:

    python benchmarks/bench_large.py [--cases name,name,...] [--list]
"""
from __future__ import annotations

import argparse
import time

from pyscf import gto, lib, scf

import mojoscf
from systems import alkane_atoms, atoms_to_str, c60_atoms, water_cluster_atoms

CASES = {
    "C10H22": ("C10H22 / 6-31G*", lambda: alkane_atoms(10), "6-31g*", 0, 0, "rhf"),
    "C20H42": ("C20H42 / 6-31G", lambda: alkane_atoms(20), "6-31g", 0, 0, "rhf"),
    "w10": ("(H2O)10 / cc-pVDZ", lambda: water_cluster_atoms(10), "cc-pvdz", 0, 0, "rhf"),
    "C60": ("C60 / STO-3G", c60_atoms, "sto-3g", 0, 0, "rhf"),
    "w20": ("(H2O)20 / cc-pVDZ", lambda: water_cluster_atoms(20), "cc-pvdz", 0, 0, "rhf"),
    "C20H41": ("C20H41 radical / 6-31G (UHF)", lambda: alkane_atoms(20)[:-1], "6-31g", 0, 1, "uhf"),
    "w10+": ("(H2O)10 cation / cc-pVDZ (UHF)", lambda: water_cluster_atoms(10), "cc-pvdz", 1, 1, "uhf"),
}


def run(factory, mol, accelerate):
    mf = factory(mol).density_fit()
    mf.verbose = 0
    mf.conv_tol = 1e-9
    mf.max_cycle = 100
    if accelerate:
        mojoscf.accelerate(mf)
    veff = [0.0]
    orig = mf.get_veff

    def timed(*args, **kwargs):
        t = time.perf_counter()
        out = orig(*args, **kwargs)
        veff[0] += time.perf_counter() - t
        return out

    mf.get_veff = timed
    t0 = time.perf_counter()
    mf.kernel()
    total = time.perf_counter() - t0
    return total, total - veff[0], mf


def warm_up():
    """Start the Mojo runtime and load the BLAS library before anything is timed."""
    mol = gto.M(atom="H 0 0 0; H 0 0 1", basis="sto-3g", verbose=0)
    mojoscf.UHF(mol).run()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    warm_up()
    if args.list:
        print("\n".join(f"{k:8s} {v[0]}" for k, v in CASES.items()))
        return
    info = mojoscf.backend_info()
    print(f"threads: pyscf={lib.num_threads()} mojo={info['parallelism_level']}; BLAS small/large: "
          f"{(info['blas_library'] or 'native').split('/')[-1]} / {(info['blas_library_large'] or 'native').split('/')[-1]} "
          f"(threaded from n >= {info['threaded_min']})")
    print(f"{'system':32s} {'nao':>5s} {'cycles':>7s} | {'total [s]':>17s} {'speedup':>8s} | {'glue [s]':>17s} {'speedup':>8s} | {'|dE| [Eh]':>9s}")
    print(f"{'':32s} {'':>5s} {'':>7s} | {'pyscf':>8s} {'mojoscf':>8s} {'':>8s} | {'pyscf':>8s} {'mojoscf':>8s} {'':>8s} |")
    for key in args.cases.split(","):
        name, atoms, basis, charge, spin, kind = CASES[key]
        mol = gto.M(atom=atoms_to_str(atoms()), basis=basis, charge=charge, spin=spin, verbose=0, max_memory=12000)
        cls = scf.RHF if kind == "rhf" else scf.UHF
        t_ref, g_ref, ref = run(cls, mol, False)
        t_moj, g_moj, moj = run(cls, mol, True)
        print(f"{name:32s} {mol.nao_nr():5d} {ref.cycles:3d}/{moj.cycles:<3d} | {t_ref:8.1f} {t_moj:8.1f} {t_ref / t_moj:7.2f}x | "
              f"{g_ref:8.2f} {g_moj:8.2f} {g_ref / g_moj:7.2f}x | {abs(ref.e_tot - moj.e_tot):9.1e}"
              f"{'' if ref.converged and moj.converged else '  NOT CONVERGED'}", flush=True)


if __name__ == "__main__":
    main()
