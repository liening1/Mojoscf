# mojoscf

**Mojo replacements for the Python glue in pyscf's SCF driver.**

[pyscf](https://pyscf.org) evaluates integrals in C (libcint), but the
self-consistent-field loop that stitches everything together is Python + NumPy:
Fock assembly, damping, DIIS extrapolation, the generalised eigenproblem,
occupations, density matrices, energies, convergence tests and logging, plus
the orchestration of the two-electron (J/K) build.  `mojoscf` re-implements the
whole SCF iteration in [Mojo](https://www.modular.com/mojo): the glue, the J/K
build (from the in-core density-fitting tensor, from in-core 8-fold packed ERIs,
or integral-direct when the ERIs do not fit in memory) and the two-electron
integrals themselves, with its own Gaussian integral engine
(`mojoscf.integrals`), which is faster than libcint for these integrals.  An
iteration then runs without touching Python or pyscf's C code at all.

* **Drop-in**: `mojoscf.RHF(mol)` and `mojoscf.UHF(mol)` are subclasses of
  `pyscf.scf.hf.RHF` / `pyscf.scf.uhf.UHF`; results (energies, orbitals,
  iteration counts, `scf_summary`) agree with pyscf to round-off with the same
  integrals, and to about 1e-12 Eh with the default Mojo integrals, because the
  driver is a port of `pyscf.scf.hf.kernel`, not a reimplementation of SCF.
  UHF includes open-shell and broken-symmetry (BS) calculations.
* **Whole loop in Mojo**: one native call runs all iterations.  The
  Coulomb/exchange matrices are built by Mojo kernels too: from the DF tensor
  or in-core ERIs (`mojoscf.kernels.df_jk`, `jk_s8`), or integral-direct for
  direct SCF (`mojoscf.integrals.get_jk`, 2.6x faster than pyscf's direct
  SCF), so a cycle makes no Python call.
* **Mojo integral engine** (`mojoscf.integrals`): a McMurchie-Davidson
  implementation of the one-electron, electron-repulsion and density-fitting
  integrals over contracted Gaussians, agreeing with libcint to about 1e-13
  for s to g functions, spherical or Cartesian.  The 4-index ERI tensor is
  built in 0.47 to 0.97x of libcint's time and the 3-index DF integrals in
  about 0.75x.  The SCF driver uses it by default for the two-electron
  integrals, also for molecules with effective core potentials or finite
  nuclei (those only change the one-electron Hamiltonian, which stays
  pyscf's), and falls back to libcint only for range-separated `mol.omega`
  and l > 8; `MOJOSCF_INTEGRALS=libcint` or
  `mojoscf.integrals.set_engine("libcint")` switches it off.
* **Nuclear gradients** (`mojoscf.grad`): `nuc_grad_method()` of `mojoscf.RHF`
  and `UHF`, with exact or density-fitted integrals, returns pyscf's gradient
  classes with the derivative integrals from the Mojo engine; the
  two-electron term is evaluated directly from the derivative integrals,
  3.3 to 7.1x faster than `pyscf.grad` (exact) and 2.9 to 6.0x faster than
  `pyscf.df.grad` (DF) with the same gradients to about 1e-13.
* **Individual kernels** are also exposed (`mojoscf.kernels`) and a
  Mojo-backed `CDIIS` class can be dropped into any pyscf SCF object.
* **BLAS/LAPACK** (OpenBLAS bundled with pyscf and SciPy) is called from Mojo
  through `dlopen`, choosing a sequential or a threaded library by matrix
  size; portable Mojo fallbacks (SIMD GEMM, Jacobi eigensolver) keep
  everything working when no library is found.

## Results

Hartree-Fock on 4 cores (2.1 GHz Xeon), pyscf 2.14, Mojo 1.1.0.  Both drivers
keep the ERIs in core for these molecules (pyscf: libcint integrals and its C
contraction; mojoscf: Mojo integrals and its Mojo kernel), start from the
same initial guess with `conv_tol = 1e-10`; best of 3 runs, including the
one-time integral evaluation.

| system             | nao | cycles | pyscf [s] | mojoscf [s] | speed-up | &#124;ΔE&#124; [Eh] |
|--------------------|----:|-------:|----------:|------------:|---------:|--------:|
| H2O / STO-3G       |   7 |    7/7 |     0.039 |       0.033 |    1.2x  | 0 |
| H2O / cc-pVDZ      |  24 |    9/9 |     0.179 |       0.062 |    2.9x  | 1e-13 |
| H2O / cc-pVTZ      |  58 |    9/9 |     0.202 |       0.130 |    1.6x  | 1e-13 |
| benzene / STO-3G   |  36 |    7/7 |     0.204 |       0.170 |    1.2x  | 3e-13 |
| benzene / cc-pVDZ  | 114 |    8/8 |     3.011 |       0.617 |    4.9x  | 7e-13 |
| (H2O)3 / cc-pVDZ   |  72 |  10/10 |     0.610 |       0.226 |    2.7x  | 6e-14 |

(`python benchmarks/bench_scf.py`.)  Below about 0.2 s the totals are dominated
by fixed costs (molecule set-up, initial guess, one-electron integrals) and by
timer noise.  The energy differences come from the integrals themselves,
which agree with libcint's to about 1e-14.
With the same integrals (`MOJOSCF_INTEGRALS=libcint`) the per-cycle logs are
identical to pyscf's to all printed digits, including with damping, level
shifting and DIIS damping switched on.

Stand-alone kernels (`python benchmarks/bench_kernels.py`, same machine) show
where the time goes: the DIIS update and the Fock diagonalisation are the
expensive glue steps and are 1.5x to more than 10x faster in Mojo
(`CDIIS.update` 199 → 46 µs at nao = 24, 32 ms → 3.4 ms at nao = 300).  The
small O(n²) kernels (`make_rdm1`, `get_grad`, `energy_elec`) win inside the
native loop but, called one at a time from Python on tiny matrices, pay a
fixed ~10 µs for the NumPy-to-pointer conversion and the `dlopen` of the BLAS
library and can then be slower than NumPy.  Use `mojoscf.RHF` /
`mojoscf.accelerate` rather than the per-kernel API when speed matters.

### Larger systems

Every (system, driver) pair below ran in its own process on an otherwise idle
4-core machine (2.1 GHz Xeon, `python benchmarks/bench_large.py`), so BLAS
thread pools and memory of one run cannot affect another.  For pyscf the time
inside `mf.get_veff` (which includes the one-time integral evaluation) is
listed separately; for mojoscf the integrals and the J/K build are part of
the native driver (J/K mode 1 = density fitting, 2 = in-core ERIs,
3 = integral-direct).  "Direct" cases run with `max_memory=1`, so both codes
recompute the 4-index integrals in every cycle.

| system | nao | cycles | pyscf [s] | of which get_veff [s] | mojoscf [s] | J/K mode | speed-up | &#124;ΔE&#124; [Eh] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C10H22 / 6-31G* (DF) | 184 | 9/9 | 12.4 | 10.7 | 4.5 | 1 | 2.79x | 5.7e-13 |
| C20H42 / 6-31G (DF) | 264 | 8/8 | 32.2 | 30.0 | 17.9 | 1 | 1.80x | 8.2e-12 |
| (H2O)10 / cc-pVDZ (DF) | 240 | 10/10 | 16.9 | 14.9 | 4.4 | 1 | 3.85x | 1.8e-12 |
| C60 / STO-3G (DF) | 300 | 8/8 | 104.8 | 102.8 | 56.7 | 1 | 1.85x | 5.8e-11 |
| (H2O)20 / cc-pVDZ (DF) | 480 | 10/10 | 91.2 | 88.2 | 46.5 | 1 | 1.96x | 1.5e-11 |
| C20H41 radical / 6-31G (DF, UHF) | 262 | 13/13 | 56.5 | 51.3 | 31.6 | 1 | 1.79x | 1.7e-11 |
| (H2O)10 cation / cc-pVDZ (DF, UHF) | 240 | 19/19 | 31.1 | 24.5 | 11.7 | 1 | 2.66x | 6.8e-12 |
| benzene / cc-pVDZ (in-core) | 114 | 8/8 | 2.8 | 1.2 | 0.6 | 2 | 5.03x | 6.3e-13 |
| benzene cation / cc-pVDZ (UHF) | 114 | 12/12 | 6.1 | 2.4 | 0.8 | 2 | 7.50x | 2.8e-13 |
| (H2O)5 / cc-pVDZ (in-core) | 120 | 10/10 | 3.6 | 2.0 | 0.6 | 2 | 6.48x | 3.4e-13 |
| benzene / cc-pVDZ (direct) | 114 | 8/8 | 9.7 | 8.4 | 1.7 | 3 | 5.67x | 1.7e-13 |
| C8H18 / cc-pVDZ (direct) | 202 | 9/9 | 23.2 | 21.5 | 9.3 | 3 | 2.51x | 1.7e-12 |
| (H2O)10 / cc-pVDZ (direct) | 240 | 10/10 | 22.2 | 20.0 | 10.2 | 3 | 2.17x | 2.3e-12 |
| (H2O)5 cation / cc-pVDZ (direct, UHF) | 120 | 19/19 | 19.1 | 12.8 | 3.1 | 3 | 6.20x | 1.5e-12 |
| benzene / def2-TZVP (direct) | 222 | 8/8 | 28.8 | 27.2 | 16.0 | 3 | 1.80x | 1.6e-12 |

What the numbers mean:

* **Direct SCF** (integrals recomputed every cycle): mojoscf evaluates the
  quartets with its own integral engine and folds them into J and K in Mojo,
  with pyscf's screening and incremental update, so the SCF is 1.8 to 6x
  faster than with libcint + libcvhf.  The largest gains are for the smaller
  and the open-shell systems (more cycles, cheaper quartets); the smallest
  for def2-TZVP, whose f shells take the single-quartet path.
* **In-core ERIs** (non-DF, up to about 250 orbitals with pyscf's default
  memory limit): the tensor comes from the Mojo engine (about 2x faster than
  libcint) and is contracted by a Mojo kernel that is 1.6 to 1.8x faster than
  pyscf's, on top of the glue savings: 5 to 8x end to end.  For UHF pyscf
  recomputes the integrals in every cycle, so the in-core build is a large win
  there.
* **Density fitting**: the 3-index integrals come from the Mojo engine
  (0.55-0.65x of libcint's time) and the triangular solve runs in place; the
  exchange build is a GEMM-bound operation (`sum_Q (Q|mu i)(Q|nu i)`) that
  pyscf already runs near the machine's GEMM speed, so the Mojo kernel's
  gain per iteration on it is modest (a streaming J and a `dsyrk` update).
  End to end: 1.8 to 3.9x here.
* **What is still C**: the one-electron matrices (`get_hcore`, `get_ovlp`,
  milliseconds) come from libcint unless `attach(mf)` is used, and the BLAS
  and LAPACK calls (GEMM, eigensolvers, Cholesky) are OpenBLAS.  Molecules the
  integral engine does not support use libcint for everything.
* Earlier versions of this table were measured on a 2.8 GHz machine and with
  libcint integrals throughout; these numbers replace them.  On this virtual
  machine the same run can vary by 20-40% between sessions (pyscf's C10H22
  DF run took 9.1 s in an earlier session, 12.4 s here), so compare within a
  row rather than across tables.

### Broken-symmetry UHF

Starting from identical spin-polarised densities (`mojoscf.guess`), pyscf and
mojoscf follow the same SCF path: same number of cycles, same energies to
1e-13 Eh for the small systems and 1e-11 Eh for the transition-metal dimers.
`E_RHF - E_BS` shows how far the broken-symmetry solution lies below the
restricted one, `<S^2>` how spin-contaminated it is (`python
benchmarks/bench_bs.py [--heavy]`, 4 cores, direct SCF unless marked DF).

| system                              | nao | E_RHF-E_BS [mEh] | <S^2>  | cycles | pyscf [s] | mojoscf [s] | speed-up | &#124;ΔE&#124; [Eh] |
|-------------------------------------|----:|-----------------:|-------:|-------:|----------:|------------:|---------:|-------:|
| H2, R = 2.0 Å / cc-pVDZ             |  10 |             80.9 |  0.904 |    7/7 |      0.02 |        0.01 |    1.9x  | 1e-15 |
| H2, R = 3.0 Å / cc-pVDZ             |  10 |            172.3 |  0.995 |    6/6 |      0.02 |        0.01 |    2.0x  | 2e-15 |
| H10 chain, AFM / 6-31G              |  20 |            298.3 |  3.693 |    8/8 |      0.03 |        0.02 |    1.8x  | 4e-14 |
| H20 chain, AFM / 6-31G              |  40 |            595.3 |  7.266 |    8/8 |      0.05 |        0.03 |    1.7x  | 4e-14 |
| H30 chain, AFM / 6-31G              |  60 |            892.3 | 10.840 |    8/8 |      0.12 |        0.09 |    1.3x  | 9e-14 |
| H40 chain, AFM / 6-31G              |  80 |           1189.3 | 14.413 |    8/8 |      1.13 |        0.28 |    4.0x  | 1e-13 |
| N2, R = 2.2 Å / cc-pVDZ             |  28 |            346.1 |  1.993 |  15/15 |      0.07 |        0.03 |    2.6x  | 1e-14 |
| F2, R = 2.6 Å / cc-pVDZ             |  28 |            304.6 |  1.001 |    8/8 |      0.04 |        0.02 |    2.1x  | 6e-14 |
| twisted C2H4 (90°) / cc-pVDZ        |  48 |             98.3 |  1.035 |  10/10 |      0.08 |        0.06 |    1.5x  | 1e-13 |
| [Cu2Cl6]2-, AFM / def2-SVP (DF)     | 170 |                  |  1.009 |  10/10 |     18.09 |       16.01 |    1.1x  | 3e-11 |
| [Fe2S2(SH)4]2-, AFM / def2-SVP (DF) | 190 |                  |  4.988 |  39/39 |     68.35 |       47.28 |    1.4x  | 1e-11 |

The two metal dimers are antiferromagnetically coupled singlets prepared by
flipping the spin of one metal centre in the converged high-spin density
(`flip_spin_on_atoms`): Cu(II)/Cu(II) (Mulliken spin +0.842 / -0.842, BS 0.18 mEh
above the triplet) and Fe(III)/Fe(III) with five unpaired electrons per iron
(Mulliken spin +3.88 / -4.54, BS 16.5 mEh below the S = 5 state), the latter
needing 39 cycles with a 0.3 Eh level shift.  The N2 row is one of two
broken-symmetry states the mix guess can reach (on the previous machine both
codes reached the other one, 190.6 mEh, <S^2> = 1.018), and the RHF reference
of twisted ethylene has degenerate pi orbitals; see the degenerate-shell caveat
below.  Measured on the 2.1 GHz machine with the default Mojo integrals.

### Transition-metal complexes

`benchmarks/bench_metals.py` runs the SCF and the nuclear gradient for
idealised complexes with def2 basis sets (which carry an f shell on the metal
already in def2-SVP; Pt with its def2 effective core potential): pyscf
versus the same object accelerated with `mojoscf.accelerate`, each in its own
process, 4 cores of a 2.1 GHz Xeon, conv_tol 1e-9.  "DF" uses pyscf's
default def2 JK-fitting basis, "direct" `max_memory=1`.

| system                                         | nao | cycles | SCF pyscf [s] | mojoscf [s] | x | grad pyscf [s] | mojoscf [s] | x | \|dE\| [Eh] |
|------------------------------------------------|----:|------:|------:|------:|-----:|------:|------:|-----:|--------:|
| ferrocene / def2-SVP (DF)                      | 221 | 14/14 |   17.7 |    5.4 | 3.3x |    8.9 |   1.9 | 4.6x | 7.3e-12 |
| ferrocene / def2-TZVP (DF)                     | 415 | 16/16 |   34.6 |   16.8 | 2.1x |   27.2 |   7.8 | 3.5x | 1.1e-11 |
| [Fe(H2O)6]2+ quintet / def2-TZVP (DF, UHF)     | 303 | 45/45 |   68.6 |   29.9 | 2.3x |   16.3 |   3.3 | 5.0x | 4.5e-13 |
| [Fe(H2O)6]2+ 12H2O quintet / def2-SVP (DF, UHF) | 463 | 56/56 |  547.6 |  348.2 | 1.6x |  178.7 |  23.3 | 7.7x | 5.0e-11 |
| [Cu(NH3)4]2+ doublet / def2-TZVP (DF, UHF)     | 241 | 15/15 |   19.8 |    5.9 | 3.3x |    7.3 |   1.9 | 3.7x | 5.7e-11 |
| Ni(CO)4 / def2-TZVP (DF)                       | 293 | 15/15 |   16.1 |    6.0 | 2.7x |    8.9 |   2.3 | 3.9x | 3.2e-11 |
| cisplatin / def2-TZVP (DF, Pt ECP)             | 212 | 12/12 |    7.6 |    2.9 | 2.6x |    5.4 |   2.2 | 2.5x | 1.8e-12 |
| ferrocene / def2-SVP (direct)                  | 221 | 14/14 |   59.6 |   27.4 | 2.2x |   35.2 |   9.7 | 3.6x | 5.0e-12 |
| [Fe(H2O)6]2+ quintet / def2-SVP (direct, UHF)  | 175 | 28/28 |   50.5 |   20.7 | 2.4x |   12.9 |   3.1 | 4.2x | 8.6e-12 |
| cisplatin / def2-SVP (direct, Pt ECP)          | 126 | 12/12 |   11.9 |    2.7 | 4.3x |    5.1 |   1.8 | 2.9x | 1.4e-12 |
| Ni(CO)4 / def2-SVP (in-core)                   | 143 | 14/14 |    6.2 |    1.3 | 5.0x |    6.2 |   1.6 | 3.8x | 1.2e-11 |

Gradients agree to 1e-11 or better except for the direct UHF Fe(II) case
(3e-8), where the two independently converged SCF solutions differ at that
level (on the same SCF object they agree to 1e-12, `tests/test_grad.py`).

* **Effective core potentials** only change the one-electron Hamiltonian,
  so molecules with ECPs (4d/5d metals with def2 or similar basis sets) use
  the Mojo engine for all two-electron work: in-core ERIs, the DF tensor,
  direct J/K and the gradient's two-electron term; in the gradients only the
  ECP derivative integrals come from pyscf.  Before this, cisplatin ran on
  libcint and pyscf's code throughout.
* **Large density-fitted SCF** (the 463-AO Fe(II) cluster, 56 UHF cycles) is
  dominated by the DF exchange build, `K = sum_Q (C^T E_Q)^T (C^T E_Q)`.
  mojoscf and pyscf do the same GEMMs; with ~100 occupied orbitals as the
  M dimension these reach ~35-40 GFLOPS per thread here (about 65% of this
  machine's single-thread DGEMM peak), so per cycle the two codes are close
  and the SCF gains come from the DF tensor build (Mojo integrals, in-place
  triangular solve), J and the native loop.  The gradients of the same
  systems are 3.5-8x faster.
* **Segmented basis sets** (def2) consist largely of single-primitive shells,
  where vectorising over the primitive quartets of one shell quartet leaves
  most SIMD lanes empty.  The drivers therefore batch kets as SIMD lanes
  (see *SIMD over kets* below): one direct J/K build for ferrocene / def2-SVP
  went from 2.11 s to 1.60 s (pyscf 4.2-4.7 s), for def2-TZVP from 18.5 s to
  16.4 s (pyscf 36.6 s), and for [Fe(H2O)6]2+ / def2-SVP from 0.78 s to
  0.57 s.  The f shells of def2-TZVP give pairs beyond the lane kernels'
  Hermite degree, which keep the single-quartet path.

### QM/MM

`benchmarks/bench_qmmm.py` embeds a QM solute in a sphere of TIP3P
point-charge waters (`pyscf.qmmm.mm_charge`; liquid density, waters within
2.6 Å of a QM atom left out) and times the SCF, the gradient on the QM atoms
(`nuc_grad_method().kernel()`) and the forces on the MM charges
(`grad_hcore_mm(dm) + grad_nuc_mm()`): pyscf versus the same object
accelerated with `mojoscf.accelerate`, each in its own process, 4 cores of a
2.1 GHz Xeon, conv_tol 1e-9.

| system                                          | nao | MM charges | SCF pyscf [s] | mojoscf [s] | x | grad pyscf [s] | mojoscf [s] | x | MM forces pyscf [s] | mojoscf [s] | x |
|-------------------------------------------------|----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|------:|-----:|------:|
| benzene / cc-pVDZ, 15 Å (DF)                    | 114 | 1353 |  4.3 |  0.7 | 5.9x |  1.5 |  0.4 | 3.5x |  0.72 | 0.10 |  7.2x |
| benzene / cc-pVDZ, 25 Å (DF)                    | 114 | 6486 |  3.8 |  0.8 | 4.8x |  2.7 |  0.7 | 3.9x |  3.73 | 0.42 |  8.9x |
| benzene / cc-pVDZ, 25 Å, Gaussian charges (DF)  | 114 | 6486 |  4.7 |  0.8 | 5.6x |  3.9 |  0.7 | 5.3x |  3.37 | 0.43 |  7.8x |
| (H2O)5 / aug-cc-pVDZ, 20 Å (in-core)            | 205 | 3366 | 14.1 |  4.0 | 3.5x | 18.2 |  5.4 | 3.4x |  6.29 | 0.70 |  9.0x |
| C8H18 / 6-31G*, 20 Å (direct)                   | 148 | 3330 | 16.7 |  4.5 | 3.7x | 11.4 |  2.5 | 4.6x |  5.15 | 0.30 | 17.2x |
| ferrocene / def2-SVP, 25 Å (DF)                 | 221 | 6456 | 23.4 |  8.2 | 2.9x | 13.9 |  3.2 | 4.4x | 10.91 | 1.44 |  7.6x |
| (H2O)10+ / cc-pVDZ, 20 Å (DF, UHF)              | 240 | 3330 | 31.9 | 11.8 | 2.7x | 19.0 |  3.2 | 6.0x |  7.44 | 0.58 | 12.9x |

Energies agree to 4e-11 Eh or better, QM gradients to 9e-12 and MM forces
to 5e-14 Eh/Bohr.  pyscf builds one integral matrix per block of 200
charges (`int1e_grids` for the Hamiltonian, `int1e_grids_ip` and
`int3c2e_ip2` with charges of exponent 1e16 for the gradients) and
contracts it with NumPy.  `_mojo/qmmm.mojo` treats each charge as the
s-type ket of the batched ERI kernel instead (a point charge as exponent
1e30, exact to double precision; Gaussian charges with their own
exponent), with up to 64 charges as the SIMD lanes against one AO shell
pair, and contracts on the fly: the potential in one pass, and the
derivative matrix and the forces on all charges from the six-component pair
table (nabla a, nabla b) in another, the charge's derivative following from
translational invariance.  The MM terms stay a sizeable part for a small
QM region (benzene with 6486 charges: 0.12 s for the potential, 0.42 s for
the derivative matrix inside the 0.7 s gradient, 0.42 s for the forces);
the rest of the speed-up is the SCF loop, J/K and the QM gradient described
above.

## Mojo integral engine

`mojoscf.integrals` evaluates the integrals of a pyscf `Mole` in Mojo and
returns exactly what `mol.intor` returns (same conventions: pyscf's
`_atm`/`_bas`/`_env` tables, `gto_norm`-normalised primitives, libcint's
Cartesian component order, `cart2sph` spherical functions or `mol.cart`
Cartesian ones, `aosym="s8"` packing, pyscf's DF tensor layout).  For the
four-index and three-centre integrals it is faster than libcint.

| function                          | pyscf equivalent                                   |
|-----------------------------------|----------------------------------------------------|
| `int1e(mol)` / `get_ovlp`, `get_kin`, `get_nuc`, `get_hcore` | `mol.intor("int1e_ovlp" / "int1e_kin" / "int1e_nuc")`, `scf.hf.get_hcore` |
| `int2e_s8(mol)`, `int2e(mol)`     | `mol.intor("int2e", aosym="s8")`, `mol.intor("int2e")` |
| `int3c2e(mol, auxmol)`, `int2c2e(auxmol)` | `df.incore.aux_e2(..., aosym="s2ij").T`, `auxmol.intor("int2c2e")` |
| `cholesky_eri(mol, auxbasis)`     | `df.incore.cholesky_eri`                           |
| `attach(mf)`                      | makes an RHF/UHF object (plain or density-fitted) use all of the above |
| `int1e_ip(mol)`, `int1e_iprinv(mol, atom)` | `mol.intor("int1e_ipovlp" / "int1e_ipkin" / "int1e_ipnuc")`, `int1e_iprinv` at a nucleus |
| `get_jk_ip1(mol, dm)`             | `pyscf.grad.rhf.get_jk` (derivative J/K matrices)  |
| `grad2e(mol, dm_j, dm_k, j_factor, k_factor)` | the two-electron term of `pyscf.grad.rhf/uhf.grad_elec` |
| `grad2e_df(mol, auxmol, dm_j, orbs, occs, j_factor, k_factor)` | the two-electron term of `pyscf.df.grad.rhf/uhf` (with the auxiliary-basis response) |
| `int1e_grids_sum(mol, coords, w, zetas=None)` | `einsum('kij,k->ij', mol.intor("int1e_grids", grids=coords), w)`; with `zetas`, Gaussian charges (`int3c2e` with `fakemol_for_charges`) |
| `int1e_grids_ip_sum(mol, coords, w, zetas=None)` | `einsum('xkij,k->xij', mol.intor("int1e_grids_ip", grids=coords), w)` |
| `mm_charge_forces(mol, dm, coords, w, zetas=None)` | `pyscf.qmmm` `QMMMGrad.grad_hcore_mm(dm)` (`int3c2e_ip2` contracted with dm) |

**Method.** McMurchie-Davidson: Hermite expansion coefficients `E_t^{ij}`,
Hermite Coulomb integrals `R_{tuv}` from the Boys function (an 8-term Taylor
table on a 0.05 grid up to T = 36, the asymptotic form beyond), and

    (ab|cd) = 2 pi^{5/2} / (p q sqrt(p+q)) sum_{tuv} E^{ab}_{tuv} sum_{TUV} (-1)^{T+U+V} E^{cd}_{TUV} R_{t+T,u+U,v+V}

The implementation is built around a few decisions, each measured:

* **Shell-pair table.**  Every shell pair is set up once, in parallel: for
  each primitive pair it stores p, 1/p, P and a matrix `E[h][j]` over the
  Hermite indices h and the pair's *final* basis functions j, with the
  contraction coefficients and the Cartesian-to-spherical transformation
  already applied.  The integrals are linear in these matrices, so a quartet
  comes out directly in spherical functions with no transformation step
  (which had cost 38% of the time for d shells), and a d-d pair carries 25
  columns instead of 36.
* **Two dense products per quartet.**  For an outer pair O and an inner pair
  I, `T[h_o][i] += sum_{h_i} R[h_o + h_i] E_I[h_i][i]` accumulates over the
  inner primitives and `(o|i) += sum_{h_o} (-1)^{|h_o|} E_O[h_o][o] T[h_o][i]`
  runs once per outer primitive.  `eri_quartet` makes the pair with more
  primitives (by operation count) the inner one, so for contracted shells the
  outer transform is amortised.
* **Compile-time kernels.**  For Hermite degrees up to 4 per pair (and up to
  6 against a pair of degree <= 2) the Boys function, the Hermite recursion
  and both transforms are unrolled with `comptime` indices
  (`tools/gen_eri_kernel.py` writes the register-blocked code).  The unrolled
  L=4 recursion takes 11 ns instead of over 100 ns with run-time loops.
* **SIMD over primitive quartets.**  Eight inner primitive pairs are processed
  together: prefactors, the Boys function (table rows gathered per lane,
  branch-free blend with the asymptotic form) and the Hermite recursion run as
  8-wide vectors, the transforms keep eight independent FMA chains in
  registers and take R as an embedded broadcast operand, and all lanes of a
  batch accumulate before one store.  Single primitive quartets use a scalar
  version of the same code (or, in the drivers, the batches below).  Vector
  operands are 64-byte aligned (cache-line splits had cost 1.45x).
* **SIMD over kets.**  Segmented basis sets (def2, Pople) are dominated by
  quartets whose pairs have one or a few primitive pairs, where vectorising
  over the primitives of one quartet leaves most lanes empty.  `eri_batch`
  evaluates one bra pair against many ket pairs at once: every primitive
  pair of the kets (up to 32; kets of one class, i.e. the same Hermite degree
  and number of functions) is one SIMD lane of `eri_kernel_lanes`.
  Prefactors, the Boys function and the Hermite recursion run across the
  lanes; the bra transform
  `U[o][h_k] += sum_{h_b} (-1)^{|h_b|} E_B[h_b][o] R[h_b + h_k]` accumulates
  over the bra's primitive pairs with `E_B` broadcast, and each ket's own
  transform `(o|k) = sum_{h_k} U[o][h_k] E_K[h_k][k]` runs once per ket
  primitive pair afterwards.  The direct J/K, the in-core ERIs, the
  three-centre integrals and the DF gradient queue each bra's kets by class
  and flush full batches.  `lanes_preferred` keeps a quartet on the
  single-quartet path where per-class timings showed that to be faster
  (kets with five or more primitive pairs against a less contracted bra,
  contracted kets against bras of Hermite degree above 2).  The four-centre
  derivative integrals of the gradient keep the single-quartet path: there
  the bra carries six derivative components and batching measured no gain.
* Higher degrees use a generic kernel with run-time loops and
  register-tiled SIMD products.  Primitive pairs with `mu |AB|^2 > 60` are
  dropped (libcint's `EXPCUTOFF`), shell quartets are Schwarz-screened
  (`schwarz_tol`, default 1e-14), and worker threads pull bra pairs from a
  shared counter and write the 8-fold packed elements directly.  Three- and
  two-centre integrals reuse the kernels with a dummy s shell of exponent 0;
  a three-centre task owns one bra shell, whose AO pairs form a contiguous
  column block of the `(naux, npair)` output, so it writes whole row segments.
  The Cholesky factorisation and triangular solve of `cholesky_eri` are LAPACK
  calls through SciPy, as in pyscf.

**Accuracy.** Against libcint on H2O, Ne-H, Ne, benzene, octane and a
30-water cluster with STO-3G up to cc-pVQZ (s to g shells, general
contractions, spherical and Cartesian): overlap to 1e-15, kinetic energy to
1e-14, nuclear attraction to 2e-13, every four-index integral to 1e-13, three-
and two-centre integrals to 2e-13 and 1e-11 (values of order 1e3).  SCF
energies with either set of integrals agree to 1e-12 Eh.

**Speed** (`benchmarks/bench_integrals.py --scf --repeat 3`, 4 cores of a
2.1 GHz Xeon for both: libcint through pyscf's OpenMP, the Mojo engine through
its runtime; best of 3, output memory touched beforehand):

| system                 | nao | naux | ERIs (s8) Mojo | libcint | ratio | DF tensor Mojo | libcint | ratio |
|------------------------|----:|-----:|---------------:|--------:|------:|---------------:|--------:|------:|
| H2O / cc-pVDZ          |  24 |  116 |        0.003 s | 0.002 s | 1.20  |        0.004 s | 0.006 s | 0.69  |
| H2O / aug-cc-pVTZ      |  92 |  139 |        0.051 s | 0.055 s | 0.93  |        0.009 s | 0.026 s | 0.34  |
| H2O / cc-pVQZ          | 115 |  208 |        0.126 s | 0.151 s | 0.84  |        0.019 s | 0.027 s | 0.68  |
| benzene / cc-pVDZ      | 114 |  558 |        0.147 s | 0.370 s | 0.40  |        0.073 s | 0.124 s | 0.59  |
| benzene / def2-TZVP    | 222 |  558 |        2.03 s  | 3.38 s  | 0.60  |        0.193 s | 0.317 s | 0.61  |
| (H2O)5 / cc-pVDZ       | 120 |  580 |        0.163 s | 0.314 s | 0.52  |        0.066 s | 0.107 s | 0.62  |
| C8H18 / cc-pVDZ        | 202 |  974 |        1.05 s  | 2.73 s  | 0.38  |        0.301 s | 0.481 s | 0.63  |

The DF tensor column includes the Cholesky solve, which is the same SciPy
call in both; the three-centre integrals alone take 0.09 s against libcint's
0.17 s for octane (0.61 s against 0.80 s for ferrocene / def2-TZVP).  The one-electron
matrices take 0.6 to 14 ms (libcint 0.3 to 11 ms).  libcint's times for the
4-index tensor vary by up to 2x between runs (memory traffic for a tensor of
up to 2.5 GB); the table keeps the best run of each.

End to end (integrals plus the Mojo RHF loop with in-core ERIs, one run each,
the previous ERI tensor released first):

| system                 | RHF with libcint integrals | RHF with Mojo integrals | energies agree to |
|------------------------|---------------------------:|------------------------:|------------------:|
| H2O / cc-pVDZ          |                     0.19 s |                  0.05 s |           1e-14 Eh |
| H2O / aug-cc-pVTZ      |                     0.25 s |                  0.21 s |           1e-13 Eh |
| H2O / cc-pVQZ          |                     0.44 s |                  0.49 s |           2e-13 Eh |
| benzene / cc-pVDZ      |                     0.82 s |                  0.60 s |           2e-12 Eh |
| benzene / def2-TZVP    |                     6.21 s |                  4.93 s |           8e-13 Eh |
| (H2O)5 / cc-pVDZ       |                     0.88 s |                  0.73 s |                0 Eh |
| C8H18 / cc-pVDZ        |                     4.61 s |                  3.94 s |           1e-12 Eh |

Per shell class (benzene / cc-pVDZ, 4-index tensor restricted to the listed
shells): s-only 0.50x of libcint's time, p-only 0.75x, s+p 0.44x, s+d 0.53x,
p+d 0.60x, all 0.47x.  For uncontracted high angular momentum
McMurchie-Davidson needs more operations than libcint's Rys quadrature, as
there are no primitives to amortise the transforms over; batching kets as
SIMD lanes keeps single-primitive d shells ahead (30 centres with one
uncontracted d shell each: 0.70x libcint's time, 0.78x without batching) and
single-primitive f shells, which keep the single-quartet path, near parity
(0.8-1.1x between runs).  Calls are
lightweight (0.13 ms for H2; the Boys table is built once per process and
tiny jobs stay on the calling thread).

**In the SCF driver.**  `mojoscf.RHF`/`UHF` (and objects upgraded with
`accelerate`) take their two-electron integrals from the engine by default:
the in-core ERIs, the in-core DF tensor (built under the same memory rule as
`pyscf.df.DF.build`) and, for direct SCF, the integral-direct J/K of
`_mojo/directjk.mojo`.  That builds the shell-pair table and the Schwarz
bounds once, then in every cycle recomputes the significant quartets of the
density change (pyscf's incremental `direct_scf` update and its
`direct_scf_tol` test, Schwarz bound times the largest density element of the
six shell blocks involved) and folds them into per-thread J/K accumulators
with the 8-fold symmetry weights.  Each unique quartet is evaluated by the
task of its pair with more primitive pairs, whose kets are batched as above;
with the batched integrals the J/K fold (scalar code over the quartet's
functions) is about a quarter of the time for ferrocene/def2-SVP.  Molecules with effective core potentials or finite
nuclei use the engine too (only their one-electron Hamiltonian differs, and
it comes from pyscf); range-separated `mol.omega` uses libcint, and
`MOJOSCF_INTEGRALS=libcint` switches the engine off.  One-electron matrices
still come from pyscf unless `attach(mf)` is used.  The DF tensor
`L^-1 (P|mu nu)` is solved in place (`dtrsm` on the transposed view of the
C-ordered integrals), so no copy of it is made at any point.

### Nuclear gradients

`nuc_grad_method()` (and `Gradients()`) of `mojoscf.RHF`/`UHF` and of
`accelerate`d objects returns `mojoscf.grad.Gradients`/`UGradients`:
pyscf's `grad.rhf`/`grad.uhf` classes with the integral work done by the
engine.  The classes also take plain pyscf objects,
`mojoscf.grad.Gradients(scf.RHF(mol).run()).kernel()`.

* **Derivative integrals.**  A derivative pair table stores, like the plain
  one, coefficient-weighted Hermite matrices, built from
  `d/dx x^i e^{-a x^2} = i x^{i-1} e^{-a x^2} - 2a x^{i+1} e^{-a x^2}`
  (Hermite degree la + lb + 1, three components per differentiated
  function), so the same kernels produce derivative integrals.  The
  one-electron derivatives (`int1e_ipovlp`, `ipkin`, `ipnuc`, `iprinv`) come
  from the same expansion.
* **Two-electron term from the energy, not from matrices.**  pyscf builds
  `sum_kl (nabla i j|kl) D_lk` and `sum_jk (nabla i j|kl) D_jk` as
  (3, nao, nao) matrices, which can only use the 4-fold symmetry of
  `(nabla i j|kl)`.  `mojoscf.grad` differentiates
  `E2 = 1/2 sum (ij|kl) G_ijkl` instead (`integrals.grad2e`, with
  `G = D_ij D_kl - (D_ik D_jl + D_il D_jk)/4` for RHF and the spin-resolved
  form for UHF): every unique quartet a >= b, c >= d, ab >= cd is evaluated
  once with its eight permutations folded into a weight, as two kernel calls
  (a six-component (nabla a, nabla b) bra against the plain ket, the ket's
  nabla c against the plain bra), the derivative on d follows from
  translational invariance, and each block is contracted with the block of G
  as it is produced.  Quartets with
  `max(q'_ab q_cd, q_ab q'_cd) max|G| < 1e-14` are skipped (q' the Schwarz
  bound of the derivative pair).  This is 1.6 to 2.2x faster than the
  J/K-matrix formulation in the same engine, which is still provided
  (`get_jk_ip1`, `Gradients.get_jk`) for code that asks for the matrices.
  Contracting G into the half-transformed Hermite intermediate, so that the
  six derivative components share one outer transform, was tried as well:
  for benzene / cc-pVDZ its integral kernels alone took 0.58 s against
  0.70 s for this whole contraction, leaving no room for a gain.  It fixes
  the derivative pair as the outer one, so the kernel can no longer put the
  pair with more primitives on the inner, vectorised side.
* **Same assembly as pyscf.**  The one-electron and overlap terms
  (`hcore_generator`, `make_rdm1e`) are put together exactly as in
  `pyscf.grad.rhf.grad_elec`; scanners, `atmlst`, TDHF gradients on top of
  a Mojo SCF and pyscf code calling `get_jk` (also with the non-symmetric
  densities of TDHF, which go to pyscf) keep working.  With effective core
  potentials only the ECP derivative integrals (`ECPscalar_ipnuc`,
  `ECPscalar_iprinv`) come from pyscf; X2C objects take all one-electron
  pieces from pyscf, and range-separated operators use pyscf's gradient code.
* **Density fitting.**  For DF objects (`mojoscf.RHF(mol).density_fit()`,
  or `accelerate`d DF objects) `nuc_grad_method()` returns
  `DFGradients`/`DFUGradients`, pyscf's `df.grad` classes with the same
  treatment.  The derivative of the DF energy at fixed densities is
  `sum (mu nu|P)' Gamma_P,mu nu - 1/2 sum (P|Q)' W_PQ`, with
  `c = V^-1 rho`, `X = V^-1 (P|ij)`,
  `Gamma_P = c_P D - k (C n) X_P (C n)^T` and
  `W = c c^T - k sum n_i n_j X_Pij X_Qij` (`integrals.grad2e_df`).  Mojo
  computes, per block of auxiliary functions sized by `max_memory`, the
  three-centre integrals and from them `rho` and `(P|ij)` (worker threads
  unpack one `(P|mu nu)` at a time and transform it with the sequential
  BLAS, as the DF exchange build does), then the rows `Gamma_P` (a GEMM and
  a rank-2k update each) and contracts them with the six-component
  `(nabla a, nabla b|P)` integrals as they are produced, the auxiliary
  centre taking minus their sum; the metric term uses `(nabla P|Q)`.  The
  metric solves and W run on the packed `i >= j` columns (`X` is
  symmetric), W as a rank-k update.  The response of the auxiliary basis is
  included, as with pyscf's default `auxbasis_response = True`;
  `auxbasis_response = False`, `only_dfj` and DF objects other than
  `pyscf.df.DF` use pyscf's code.  `density_fit()` of the Mojo classes puts
  a small class in front of pyscf's `_DFHF` (whose `nuc_grad_method` would
  otherwise come first); `undo_df()` removes the DF part as usual.

Timings (`benchmarks/bench_grad.py`, 4 cores of a 2.1 GHz Xeon; the same
converged pyscf SCF in both processes, only the gradient timed; "2e" is the
two-electron term alone, `get_veff` for pyscf (for DF including the
auxiliary-basis response) and `grad_2e` for mojoscf; "(DF)" rows compare
pyscf's `df.grad` with `DFGradients`):

| system                             | nao | pyscf [s] | (2e)  | mojoscf [s] | (2e)  | speedup | max \|dg\| |
|------------------------------------|----:|----------:|------:|------------:|------:|--------:|----------:|
| H2O / cc-pVTZ                      |  58 |      0.49 |  0.39 |        0.07 |  0.05 |   7.50x |   2.6e-14 |
| benzene / cc-pVDZ                  | 114 |      4.03 |  3.88 |        0.96 |  0.96 |   4.19x |   6.6e-13 |
| benzene cation / cc-pVDZ (UHF)     | 114 |      4.23 |  4.30 |        0.91 |  0.81 |   4.64x |   1.4e-11 |
| C8H18 / cc-pVDZ                    | 202 |     18.25 | 18.22 |        4.67 |  4.40 |   3.91x |   9.0e-12 |
| (H2O)5 / aug-cc-pVDZ               | 205 |     14.99 | 14.53 |        4.81 |  4.52 |   3.12x |   4.4e-08 |
| (H2O)10 / cc-pVDZ                  | 240 |     17.45 | 17.70 |        3.96 |  3.60 |   4.40x |   9.7e-12 |
| benzene / def2-TZVP                | 222 |     28.29 | 26.65 |        9.93 |  9.29 |   2.85x |   2.9e-12 |
| benzene / cc-pVDZ (DF)             | 114 |      1.36 |  0.95 |        0.37 |  0.26 |   3.64x |   4.8e-13 |
| C8H18 / cc-pVDZ (DF)               | 202 |      4.82 |  4.29 |        1.06 |  0.79 |   4.57x |   2.2e-12 |
| (H2O)10 / cc-pVDZ (DF)             | 240 |     10.10 |  7.40 |        1.68 |  1.38 |   6.00x |   5.7e-12 |
| (H2O)10 cation / cc-pVDZ (DF, UHF) | 240 |     13.42 | 13.66 |        2.25 |  1.75 |   5.95x |   5.8e-12 |
| benzene / def2-TZVP (DF)           | 222 |      2.74 |  2.13 |        0.87 |  0.68 |   3.17x |   5.1e-13 |
| C20H42 / 6-31G (DF)                | 264 |     29.18 | 29.13 |        5.39 |  5.18 |   5.41x |   5.6e-12 |

Each process converges its own SCF, so `max |dg|` (Eh/Bohr) includes the
SCF convergence (1e-11 Eh; for the aug-cc-pVDZ water cluster the two
processes' SCF solutions moved the gradient by 4e-8 in this run, while on one
shared SCF object the two gradients agree to 5e-12); on the same SCF object
the gradients agree to about 1e-13 (`tests/test_grad.py`).  This run is about
25% slower for both codes than the previous one on the same type of
machine; the ratios are what carries over.

## Installation

```bash
pip install pyscf numpy scipy
pip install modular            # the Mojo compiler (needed to build the kernels)
pip install -e .
python -m mojoscf.build        # compiles mojoscf/_mojo -> mojoscf/_mojoscf.so
python -m pytest               # optional
```

The extension is also compiled automatically on first import when the sources
are newer than the build.  Set `MOJOSCF_SKIP_BUILD=1` to disable that.

## Usage

```python
from pyscf import gto
import mojoscf

mol = gto.M(atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587", basis="cc-pvdz")
mf = mojoscf.RHF(mol)        # any pyscf RHF option works: mf.level_shift = 0.2, ...
mf.kernel()
print(mf.e_tot, mf.cycles, mf.scf_summary)

# Upgrade an existing RHF object in place (density fitting, X2C, ... are kept)
from pyscf import scf
mf = scf.RHF(mol).density_fit()
mojoscf.accelerate(mf)
mf.kernel()

# Open shell and broken symmetry (UHF)
from mojoscf.guess import mix_homo_lumo_guess, afm_guess_by_atom, flip_spin_on_atoms
mol = gto.M(atom="H 0 0 0; H 0 0 3.0", basis="cc-pvdz")
mf = mojoscf.UHF(mol)
mf.kernel(dm0=mix_homo_lumo_guess(mol))      # spin-polarised start -> BS solution
print(mf.e_tot, mf.spin_square())

# Integrals from the Mojo engine instead of libcint
from mojoscf import integrals
S = integrals.get_ovlp(mol)                     # == mol.intor("int1e_ovlp")
eri = integrals.int2e_s8(mol)                   # == mol.intor("int2e", aosym="s8")
cderi = integrals.cholesky_eri(mol, "cc-pvdz-jkfit")   # == pyscf.df.incore.cholesky_eri
mf = integrals.attach(mojoscf.RHF(mol))         # hcore, overlap and ERIs from Mojo
mf.kernel()

# Nuclear gradients with derivative integrals from the Mojo engine
mf = mojoscf.RHF(mol).run()
g = mf.nuc_grad_method().kernel()               # mojoscf.grad.Gradients, == pyscf's to ~1e-13
g = mojoscf.RHF(mol).density_fit().run().nuc_grad_method().kernel()   # DFGradients
g = mojoscf.grad.Gradients(scf.RHF(mol).run()).kernel()   # also for pyscf objects

# QM/MM electrostatic embedding (pyscf.qmmm): the MM-charge terms come from the Mojo engine
from pyscf import qmmm
mf = mojoscf.accelerate(qmmm.mm_charge(scf.RHF(mol).density_fit(), mm_coords, mm_charges))
mf.kernel()                                     # (also: qmmm.mm_charge(mojoscf.RHF(mol), ...))
g = mf.nuc_grad_method()
de_qm = g.kernel()                              # forces on the QM atoms
de_mm = g.grad_hcore_mm(mf.make_rdm1()) + g.grad_nuc_mm()   # forces on the MM charges

# Use the individual kernels
from mojoscf import kernels
dm = kernels.make_rdm1(mf.mo_coeff, mf.mo_occ)
w, c = kernels.eigh(fock, s)                    # pyscf's phase convention
diis = mojoscf.CDIIS()                          # pyscf.scf.diis.CDIIS replacement
```

`mojoscf.backend_info()` reports which BLAS library and how many threads are
used.  Environment variables: `MOJOSCF_BLAS=/path/lib.so[:symbol_prefix]`,
`MOJOSCF_NATIVE=1` (pure-Mojo fallbacks), `MOJOSCF_INTEGRALS=libcint` (use
pyscf's integrals instead of the Mojo engine), `MOJOSCF_SKIP_BUILD=1`,
`MOJOSCF_MOJO=/path/to/mojo`.

### BLAS/LAPACK backend

GEMM and the symmetric eigensolvers (`dsygvd`/`dsyevd`) are called through
`dlopen`, so no link-time dependency exists.  Two libraries are used:

* matrices smaller than `MOJOSCF_THREADED_MIN` (default 200 orbitals): the
  OpenBLAS bundled with pyscf wheels, a *sequential* build.  Inside an SCF of a
  small molecule a threaded BLAS competes with pyscf's OpenMP integral code for
  the cores and costs up to 40% of the run time.
* larger matrices: SciPy's bundled OpenBLAS (LP64, threaded, `scipy_` symbol
  prefix).  Measured on 4 cores, the glue (diagonalisation, DIIS error vector,
  density) is 1.6x faster than pyscf's at 300 orbitals and 2.7x at 1000 with the
  threaded library, but 1.4 to 2.3x *slower* with the sequential one.
* the density-fitting exchange build uses both: Mojo worker threads each call
  the *sequential* library for their per-auxiliary-function GEMM (as pyscf's C
  transform does with OpenMP), and the threaded library closes every block
  with one `dsyrk` rank-k update.

`MOJOSCF_BLAS=/path/lib.so[:prefix]` forces one library for all sizes, and
`mojoscf.set_blas(small, prefix, large, prefix)` selects them from Python.  A
system `libopenblas`/`liblapack` is the last resort.  Without any library the
pure-Mojo fallbacks (SIMD GEMM, Jacobi eigensolver) are used; they are correct
but slow for more than a few dozen orbitals.

## What runs where

| step in `pyscf.scf.hf.kernel`               | pyscf                     | mojoscf                              |
|---------------------------------------------|---------------------------|--------------------------------------|
| `h1e + vhf`, damping, level shift (per spin for UHF) | NumPy            | Mojo SIMD kernels                    |
| CDIIS error vector, overlaps, extrapolation | NumPy + `scipy.linalg`    | Mojo (`_mojo/diis.mojo`)             |
| `eig` (`x^T F x`, `dsyevd`, back-transform) | NumPy + LAPACK            | Mojo + LAPACK via `dlopen`           |
| `get_occ`, `make_rdm1`, `energy_elec`, `get_grad`, norms | NumPy          | Mojo (+ BLAS `dgemm`)                |
| convergence test, bookkeeping, logging       | Python                    | Mojo (log lines via one callback)    |
| J/K, density fitting (`df_jk.get_jk`)       | Python loop over blocks, C transform, NumPy matmul | Mojo (`_mojo/dfjk.mojo`): streaming J passes, per-Q sequential GEMM, threaded `dsyrk` |
| J/K, in-core 8-fold ERIs (`_vhf.incore`)    | C (`libcvhf`, OpenMP)     | Mojo (`_mojo/erijk.mojo`), 1.6-1.8x faster |
| J/K, direct SCF (integrals every cycle)     | C (libcint + `libcvhf`)   | Mojo (`_mojo/directjk.mojo`): Mojo integrals, libcvhf's screening, incremental build |
| two-electron integrals (3-index DF tensor, 4-index ERIs), once | C (libcint) | Mojo engine (`_mojo/integrals.mojo`); libcint for unsupported molecules |
| one-electron integrals (`get_hcore`, `get_ovlp`) | C (libcint)          | unchanged (`attach(mf)` uses the Mojo engine) |
| nuclear gradients (`nuc_grad_method().kernel()`), exact or DF | C (libcint derivative integrals, `libcvhf` J/K, `libao2mo`) + NumPy/SciPy | Mojo derivative integrals and contractions (`_mojo/gradients.mojo`, `int1e_ip_core`); DF metric solves in SciPy; terms assembled as in pyscf |
| QM/MM charges (`pyscf.qmmm`): potential, its derivative, forces on the MM charges | C (libcint `int1e_grids`, `int1e_grids_ip`, `int3c2e_ip2`, one integral matrix per block of 200 charges) + NumPy | Mojo (`_mojo/qmmm.mojo`): one pass over the shell pairs with the charges as SIMD lanes, contracted on the fly; nucleus-charge terms NumPy as in pyscf |

Source layout:

```
mojoscf/
  _mojo/linalg.mojo    vector kernels, GEMM/eigensolver dispatch, native fallbacks
  _mojo/kernels.mojo   make_rdm1, get_occ, get_grad, damping, level_shift, DIIS errvec, dense J/K
  _mojo/dfjk.mojo      density-fitted J/K from pyscf's (naux, npair) tensor; density factorisation
  _mojo/erijk.mojo     J/K from 8-fold packed ERIs
  _mojo/diis.mojo      pyscf-compatible CDIIS bookkeeping and extrapolation
  _mojo/driver.mojo    the RHF/UHF SCF loop (port of pyscf.scf.hf.kernel) with native J/K modes
  _mojo/integrals.mojo Gaussian integral engine (Boys function, Hermite recursions, S/T/V, ERIs, 3c2e/2c2e)
  _mojo/directjk.mojo  integral-direct J/K (screening, 8-fold digestion) for direct SCF; derivative J/K matrices
  _mojo/gradients.mojo two-electron gradient terms, exact and density-fitted
  _mojo/qmmm.mojo      potential of MM point/Gaussian charges, its derivative, forces on the charges
  _mojo/__init__.mojo  Python bindings (module mojoscf._mojoscf)
  _backend.py          build/load the extension, discover BLAS/LAPACK
  kernels.py           NumPy-facing wrappers
  integrals.py         pyscf-compatible integral functions and attach()
  diis.py              CDIIS drop-in class
  scf.py               RHF/UHF classes, kernel(), accelerate()
  grad.py              RHF/UHF nuclear gradient classes, exact and DF (nuc_grad_method)
  qmmm.py              QM/MM (pyscf.qmmm) hooks: MM-charge Hamiltonian and gradient terms from the engine
  guess.py             broken-symmetry start densities (HOMO/LUMO mix, AFM atoms, spin flip)
tests/                 kernels vs NumPy/pyscf references; full SCF vs pyscf; integrals vs libcint; gradients vs pyscf
benchmarks/            bench_scf.py, bench_kernels.py, bench_large.py, bench_bs.py, bench_integrals.py, bench_grad.py, bench_metals.py, bench_qmmm.py
tools/gen_eri_kernel.py  generates the register-blocked ERI kernels (single quartet, batched kets) in _mojo/integrals.mojo
```

## Scope and limitations

* Closed-shell **RHF** and **UHF** with real orbitals, with or without density
  fitting, X2C or other decorations that only change `get_jk`/`get_hcore`.
  ROHF, GHF, Kohn-Sham DFT, symmetry-adapted and second-order (Newton) SCF
  objects are rejected by `accelerate` and are not provided as classes yet.
* The native J/K build covers plain `pyscf.df.DF` objects with the tensor in
  core, the in-core 8-fold ERI path (used when `mol.incore_anyway` or pyscf's
  own memory check allows it) and integral-direct J/K otherwise (pyscf's
  direct SCF, with the same incremental update and `direct_scf_tol`
  screening).  With the Mojo integral engine (the default) energies agree
  with pyscf's to about 1e-12 Eh instead of round-off, because the integrals
  themselves differ at the 1e-14 level.  Range separation, `only_dfj`, DF
  tensors on disk and overridden `get_jk`/`get_veff` fall back to calling
  `mf.get_veff`, as does direct SCF for molecules the engine does not support.
  `mf.scf_summary["mojoscf_veff_mode"]` reports which path ran (1 = DF,
  2 = in-core ERIs, 3 = integral-direct, 0 = pyscf callback).
* Only CDIIS is native.  EDIIS/ADIIS, DIIS objects assigned to `mf.diis`,
  `diis_space_rollback`, `diis_file`, a custom `check_convergence` and
  dispersion corrections make the driver fall back to pyscf's loop (with the
  Mojo glue functions still in use), so results stay correct.
* The native loop bypasses `get_fock`, `eig`, `get_occ`, `make_rdm1`,
  `energy_elec`, `energy_tot` and `get_grad`.  If any of them is overridden
  (Fermi smearing, constrained UHF, a user-supplied Fock build, ...),
  `accelerate` raises `TypeError` and `kernel()` falls back to pyscf's loop
  rather than silently ignoring the override.
* The checkpoint file is written once at the end of the SCF instead of every
  cycle, and the per-cycle HOMO/LUMO lines of pyscf's log are printed once, for
  the final orbitals.
* The integral engine handles contracted Gaussians up to l = 8 (spherical or
  Cartesian); it provides the overlap, kinetic, point-charge
  nuclear-attraction, four-index and 3-/2-centre Coulomb integrals and the
  first derivatives needed for HF gradients (no ECP integrals, second
  derivatives, multipoles or range-separated operators).  Molecules with
  ECPs or finite nuclei use it for everything but the ECP / finite-nucleus
  terms.  `unsupported_reason(mol, two_electron=..., allow_ecp=...)` says why
  a molecule is rejected for a given use.
* Nuclear gradients are native for RHF and UHF, with exact or
  density-fitted (in-core `pyscf.df.DF`, auxiliary-basis response included)
  two-electron integrals.
* QM/MM (`pyscf.qmmm.mm_charge`, point or Gaussian MM charges) runs on the
  native loop with the MM-charge terms of the Hamiltonian and of the
  gradients (QM atoms and MM charges) from the engine, for QM shells up to
  g; the periodic interface (`qmmm.pbc`) is not covered.  The charge sums are not screened (all charges interact with all
  shell pairs), as in pyscf.
* The first call in a process starts the Mojo runtime and loads BLAS
  (about 50 ms); time a second run when benchmarking tiny systems.
* Only the LP64 (32-bit integer) BLAS/LAPACK interface is supported.

### Reproducibility caveat: partially filled degenerate shells

When an open-shell state has a partially filled shell of *exactly degenerate*
orbitals (OH radical, stretched N2 or Be2, ...), which orbital of the pair
receives the electron is decided by eigensolver rounding noise.  The resulting
solutions can differ in energy (Be2 / 6-31G* converges to -28.6445 Eh or
-28.5930 Eh), and **pyscf itself changes solution when only the LAPACK driver
used by `scipy.linalg.eigh` is changed**, so such runs are not reproducible
between programs, BLAS builds or even repeated runs.  mojoscf follows pyscf's
trajectory exactly whenever the problem is well posed (all other examples in this
repository), and the tests compare only rotation-invariant quantities for the
degenerate cases.

## Development

```bash
make build      # python -m mojoscf.build --force
make test       # build + pytest
make bench      # SCF benchmark
```

Mojo sources follow Mojo 1.1 (`def`-only, `std.` namespaced imports,
closures with explicit capture lists).  The register-blocked ERI kernel in
`_mojo/integrals.mojo` is generated: edit `tools/gen_eri_kernel.py` and run
`python tools/gen_eri_kernel.py` to rewrite it.  The GitHub Actions workflow in
`.github/workflows/ci.yml` builds the kernels and runs the test suite.

## License

Apache License 2.0, the same license as pyscf.
