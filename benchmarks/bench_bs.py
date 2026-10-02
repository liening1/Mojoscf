"""Broken-symmetry UHF: pyscf driver versus mojoscf driver from identical start densities.

    python benchmarks/bench_bs.py            # small systems
    python benchmarks/bench_bs.py --heavy    # adds Cu2Cl6(2-) and [Fe2S2(SH)4](2-)

Each system is started from the same spin-polarised density (see ``mojoscf.guess``),
so both drivers follow the same SCF path; the table shows that they reach the same
broken-symmetry state with the same number of cycles.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from pyscf import gto, scf

import mojoscf
from mojoscf.guess import afm_guess_by_atom, atom_ao_slices, flip_spin_on_atoms, mix_homo_lumo_guess
from systems import atoms_to_str


def solve(cls, mol, dm0, df=False, **opts):
    mf = cls(mol)
    mf.verbose = 0
    mf.conv_tol = opts.pop("conv_tol", 1e-10)
    mf.max_cycle = 300
    if df:
        mf = mf.density_fit()
    for key, val in opts.items():
        setattr(mf, key, val)
    t0 = time.perf_counter()
    mf.kernel(dm0=dm0)
    return mf, time.perf_counter() - t0


def report(name, mol, dm0, df=False, **opts):
    ref, t_ref = solve(scf.UHF, mol, dm0, df, **dict(opts))
    moj, t_moj = solve(mojoscf.UHF, mol, dm0, df, **dict(opts))
    rhf = scf.RHF(mol) if mol.spin == 0 else None
    gain = ""
    if rhf is not None and mol.nao_nr() < 120:
        rhf.verbose = 0
        rhf.kernel()
        gain = f"{1e3 * (rhf.e_tot - ref.e_tot):8.1f}"
    print(f"{name:34s} {mol.nao_nr():4d} {gain:>9s} {ref.spin_square()[0]:7.3f} {ref.cycles:3d}/{moj.cycles:<3d} "
          f"{t_ref:8.2f} {t_moj:8.2f} {t_ref / t_moj:6.2f}x {abs(ref.e_tot - moj.e_tot):9.1e}"
          f"{'' if ref.converged and moj.converged else '  NOT CONVERGED'}", flush=True)
    return ref, moj


def spin_populations(mf, mol, atoms):
    dm = mf.make_rdm1()
    s = mf.get_ovlp()
    sl = atom_ao_slices(mol)
    return [np.einsum("ij,ji->", (dm[0] - dm[1])[sl[i][1]:sl[i][2]], s[:, sl[i][1]:sl[i][2]]) for i in atoms]


def small():
    for r in (2.0, 3.0):
        mol = gto.M(atom=f"H 0 0 0; H 0 0 {r}", basis="cc-pvdz", verbose=0)
        report(f"H2 R={r} A / cc-pVDZ", mol, mix_homo_lumo_guess(mol))
    for n in (10, 20, 30, 40):
        mol = gto.M(atom="; ".join(f"H 0 0 {1.8 * i}" for i in range(n)), basis="6-31g", verbose=0)
        report(f"H{n} chain 1.8 A / 6-31G (AFM)", mol, afm_guess_by_atom(mol, set(range(0, n, 2))))
    mol = gto.M(atom="N 0 0 0; N 0 0 2.2", basis="cc-pvdz", verbose=0)
    report("N2 R=2.2 A / cc-pVDZ", mol, mix_homo_lumo_guess(mol))
    mol = gto.M(atom="F 0 0 0; F 0 0 2.6", basis="cc-pvdz", verbose=0)
    report("F2 R=2.6 A / cc-pVDZ", mol, mix_homo_lumo_guess(mol))
    ethy = "C 0 0 0.667; C 0 0 -0.667; H 0.92 0 1.236; H -0.92 0 1.236; H 0 0.92 -1.236; H 0 -0.92 -1.236"
    mol = gto.M(atom=ethy, basis="cc-pvdz", verbose=0)
    report("twisted C2H4 (90 deg) / cc-pVDZ", mol, mix_homo_lumo_guess(mol))


def metal_dimer(name, atoms, charge, spin_hs, flip, basis, **opts):
    common = dict(atom=atoms_to_str(atoms), basis=basis, charge=charge, verbose=0, max_memory=12000)
    hs, bs = gto.M(spin=spin_hs, **common), gto.M(spin=0, **common)
    ref_hs = scf.UHF(hs).density_fit()
    ref_hs.verbose = 0
    ref_hs.conv_tol = 1e-8
    ref_hs.max_cycle = 200
    ref_hs.level_shift = opts.get("level_shift", 0.0)
    ref_hs.kernel()
    ref, moj = report(name, bs, flip_spin_on_atoms(ref_hs.make_rdm1(), bs, {flip}), df=True, conv_tol=1e-8, **opts)
    pops = spin_populations(ref, bs, (0, 1))
    print(f"    E(BS) - E(HS) = {1e3 * (ref.e_tot - ref_hs.e_tot):.2f} mEh;  Mulliken spin on the metal centres: "
          f"{pops[0]:+.3f} / {pops[1]:+.3f}", flush=True)


def heavy():
    cu = [("Cu", (-1.65, 0, 0)), ("Cu", (1.65, 0, 0)), ("Cl", (0, 1.6, 0)), ("Cl", (0, -1.6, 0))]
    cu += [("Cl", (sx * 3.136, sy * 1.622, 0)) for sx in (-1, 1) for sy in (-1, 1)]
    metal_dimer("[Cu2Cl6]2- AFM / def2-SVP (DF)", cu, -2, 2, 1, "def2-svp", level_shift=0.2)
    fe = [("Fe", (-1.35, 0, 0)), ("Fe", (1.35, 0, 0)), ("S", (0, 1.74, 0)), ("S", (0, -1.74, 0))]
    for sx in (-1, 1):
        for sz in (-1, 1):
            x, z = sx * 2.55, sz * 1.96
            fe += [("S", (x, 0, z)), ("H", (x + sx * 0.6, 0, z + sz * 1.2))]
    metal_dimer("[Fe2S2(SH)4]2- AFM / def2-SVP (DF)", fe, -2, 10, 1, "def2-svp", level_shift=0.3)


def warm_up():
    """Start the Mojo runtime and load the BLAS library before anything is timed."""
    mol = gto.M(atom="H 0 0 0; H 0 0 1", basis="sto-3g", verbose=0)
    mojoscf.UHF(mol).run()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heavy", action="store_true", help="also run the transition-metal dimers (minutes)")
    args = ap.parse_args()
    warm_up()
    print(f"{'system':34s} {'nao':>4s} {'E_RHF-E_BS':>9s} {'<S^2>':>7s} {'cycles':>7s} {'pyscf':>8s} {'mojoscf':>8s} {'speedup':>7s} {'|dE|':>9s}")
    print(f"{'':34s} {'':>4s} {'[mEh]':>9s} {'':>7s} {'':>7s} {'[s]':>8s} {'[s]':>8s}")
    small()
    if args.heavy:
        heavy()


if __name__ == "__main__":
    main()
