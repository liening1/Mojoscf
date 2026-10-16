"""Second-order SCF (``mf.newton()``): pyscf versus mojoscf, each in its own process.

For each (system, driver) pair the script runs pyscf's co-iterative
augmented-Hessian SCF (conv_tol 1e-9, pyscf's default grids;
``mojoscf.dft.accelerate`` for mojoscf, whose ``newton()`` objects take the
orbital Hessian from :mod:`mojoscf.soscf`), from pyscf's default initial
guess or, for the cases with a start geometry, as the step of a geometry
optimisation does: from the density of the converged SCF at that geometry
(pyscf's DIIS, computed once in a plain pyscf process and shared by both
drivers).  From the minao guess the solver passes the transition-metal
complexes through regions of negative orbital-Hessian curvature, where
products that differ in the last bits lead to different SCF solutions
(ferrocene: pyscf -1649.494 Eh, mojoscf the ground state -1649.700 Eh), so
those runs would not compare the same work.  Reported: the time of
``kernel()``, the macro iterations, the orbital-Hessian products (``h_op``
calls), pyscf's count of J/K builds (products and Fock builds, from its log)
and the energies.  Usage:

    python benchmarks/bench_soscf.py [--cases a,b,...] [--list] [--cache FILE]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

CU4 = 'square_planar_atoms("Cu", ["NH3"] * 4, [2.03] * 4)'

# key: (label, atoms, basis, charge, spin, xc, density fitting, restricted open shell, start geometry or None)
CASES = {
    "bz-b3lyp-x": ("benzene / def2-SVP B3LYP (RKS)", "BENZENE_ATOMS", "def2-svp", 0, 0, "b3lyp", False, False, None),
    "cu-b3lyp": ("[Cu(NH3)4]2+ doublet / def2-SVP B3LYP (UKS)", CU4, "def2-svp", 2, 1, "b3lyp", True, False, None),
    "cu-b3lyp-ro": ("[Cu(NH3)4]2+ doublet / def2-SVP B3LYP (ROKS)", CU4, "def2-svp", 2, 1, "b3lyp", True, True,
                    None),
    "fc-pbe0": ("ferrocene / def2-SVP PBE0 (RKS), Fe-Cp 1.56 -> 1.66 A", "ferrocene_atoms()", "def2-svp", 0, 0,
                "pbe0", True, False, "ferrocene_atoms(fe_ring=1.56)"),
    "ni-b3lyp": ("[Ni(H2O)6]2+ triplet / def2-SVP B3LYP (UKS), Ni-O 1.95 -> 2.05 A",
                 'octahedral_atoms("Ni", "H2O", 2.05)', "def2-svp", 2, 2, "b3lyp", True, False,
                 'octahedral_atoms("Ni", "H2O", 1.95)'),
}

START = r'''
import sys
import numpy as np
sys.path.insert(0, %(bench_dir)r)
from pyscf import dft, gto
from systems import *
assert "mojoscf" not in sys.modules
mol = gto.M(atom=atoms_to_str(%(start)s), basis=%(basis)r, charge=%(charge)d, spin=%(spin)d,
            verbose=0, max_memory=12000)
mf = (dft.ROKS if %(ro)r else dft.UKS if %(spin)d else dft.RKS)(mol, xc=%(xc)r)
if %(df)r:
    mf = mf.density_fit()
mf.conv_tol = 1e-9
mf.max_cycle = 100
mf.kernel()
assert mf.converged
np.save(%(path)r, np.asarray(mf.make_rdm1()))
'''

WORKER = r'''
import io, json, re, sys, time, types
import numpy as np
sys.path.insert(0, %(bench_dir)r)
from pyscf import dft, gto
from systems import *
from bench_scf import BENZENE
BENZENE_ATOMS = [(a[0], tuple(a[1])) for a in gto.format_atom(BENZENE, unit=1.0)]
mol = gto.M(atom=atoms_to_str(%(atoms)s), basis=%(basis)r, charge=%(charge)d, spin=%(spin)d,
            verbose=0, max_memory=12000)
driver = %(driver)r
if driver == "mojoscf":
    import mojoscf
    mojoscf.UHF(gto.M(atom="H 0 0 0; H 0 0 1", basis="sto-3g", verbose=0)).run()  # start the runtime
else:   # the reference must be plain pyscf: mojoscf patches some pyscf functions when imported
    assert "mojoscf" not in sys.modules
mf = (dft.ROKS if %(ro)r else dft.UKS if %(spin)d else dft.RKS)(mol, xc=%(xc)r)
if %(df)r:
    mf = mf.density_fit()
if driver == "mojoscf":
    mojoscf.dft.accelerate(mf)
mf = mf.newton()
mf.conv_tol = 1e-9
mf.stdout = buf = io.StringIO()
mf.verbose = 4
nhop = [0]
gen_g_hop = type(mf).gen_g_hop


def counted(self, *args, **kwargs):
    g, h_op, h_diag = gen_g_hop(self, *args, **kwargs)

    def h(x):
        nhop[0] += 1
        return h_op(x)

    return g, h, h_diag


mf.gen_g_hop = types.MethodType(counted, mf)
dm0 = np.load(%(start_path)r) if %(start_path)r else None
t0 = time.perf_counter()
mf.kernel(dm0=dm0)
t = time.perf_counter() - t0
m = re.search(r"macro X = (\d+)\s+E=\S+\s+\|g\|= \S+\s+total (\d+) KF (\d+) JK", buf.getvalue())
print(json.dumps(dict(t=t, e=mf.e_tot, conv=bool(mf.converged), nao=mol.nao_nr(), hop=nhop[0],
                      macro=int(m.group(1)) if m else -1, jk=int(m.group(3)) if m else -1)))
'''


def start_density(case, bench_dir, workdir):
    """Path of the start density of ``case`` (computed once with plain pyscf), or None for pyscf's guess."""
    _, atoms, basis, charge, spin, xc, df, ro, start = CASES[case]
    if start is None:
        return None
    path = os.path.join(workdir, f"soscf_start_{case}.npy")
    if not os.path.exists(path):
        code = START % dict(bench_dir=bench_dir, start=start, basis=basis, charge=charge, spin=spin, xc=xc, df=df,
                            ro=ro, path=path)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        if out.returncode != 0:
            raise RuntimeError(f"{case} start density failed:\n{out.stderr[-3000:]}")
    return path


def run(case, driver, bench_dir, workdir):
    _, atoms, basis, charge, spin, xc, df, ro, _ = CASES[case]
    code = WORKER % dict(bench_dir=bench_dir, atoms=atoms, basis=basis, charge=charge, spin=spin, xc=xc, df=df,
                         ro=ro, driver=driver, start_path=start_density(case, bench_dir, workdir) or "")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"{case} / {driver} failed:\n{out.stderr[-3000:]}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def main():
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--cache", default=None, help="JSON file keeping pyscf's results between runs")
    args = ap.parse_args()
    workdir = os.path.dirname(os.path.abspath(args.cache)) if args.cache else tempfile.mkdtemp()
    cache = {}
    if args.cache and os.path.exists(args.cache):
        with open(args.cache) as f:
            cache = json.load(f)

    def result(key, driver):
        tag = f"{key}/{driver}"
        if driver != "pyscf" or tag not in cache:
            cache[tag] = run(key, driver, bench_dir, workdir)
            if args.cache and driver == "pyscf":
                with open(args.cache, "w") as f:
                    json.dump(cache, f)
        return cache[tag]

    if args.list:
        print("\n".join(f"{k:12s} {v[0]}{'' if v[6] else ', exact integrals'}" for k, v in CASES.items()))
        return
    bench_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"{'system':64s} {'ints':>5s} {'nao':>4s} | {'pyscf [s]':>9s} {'mojoscf':>8s} {'x':>6s} | "
          f"{'macro':>9s} {'h_op':>9s} {'JK':>9s} | E pyscf / mojoscf [Eh]")
    for key in args.cases.split(","):
        name, df = CASES[key][0], CASES[key][6]
        ref = result(key, "pyscf")
        moj = result(key, "mojoscf")
        flag = "" if ref["conv"] and moj["conv"] else f"  NOT CONVERGED {ref['conv']} {moj['conv']}"
        print(f"{name:64s} {'DF' if df else 'exact':>5s} {ref['nao']:4d} | {ref['t']:9.1f} {moj['t']:8.1f} "
              f"{ref['t'] / moj['t']:5.1f}x | {ref['macro']:4d}/{moj['macro']:<4d} {ref['hop']:4d}/{moj['hop']:<4d} "
              f"{ref['jk']:4d}/{moj['jk']:<4d} | {ref['e']:.10f} / {moj['e']:.10f} ({moj['e'] - ref['e']:+.1e}){flag}",
              flush=True)


if __name__ == "__main__":
    main()
