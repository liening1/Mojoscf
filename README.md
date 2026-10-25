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
  direct SCF (`mojoscf.integrals.get_jk`, 3-5x faster than pyscf's direct
  J/K for transition-metal complexes), so a cycle makes no Python call.
* **Mojo integral engine** (`mojoscf.integrals`): a McMurchie-Davidson
  implementation of the one-electron, electron-repulsion and density-fitting
  integrals over contracted Gaussians, agreeing with libcint to about 1e-13
  for s to g functions, spherical or Cartesian.  The 4-index ERI tensor is
  built in 0.47 to 0.97x of libcint's time and the 3-index DF integrals in
  about 0.75x.  The SCF driver uses it by default for the two-electron
  integrals, also for molecules with effective core potentials or finite
  nuclei (those only change the one-electron Hamiltonian, which stays
  pyscf's), and falls back to libcint only for a range-separated `mol.omega`
  on the whole molecule (the long-range integrals of range-separated
  functionals, four-, three- and two-centre, come from the engine) and
  l > 8; `MOJOSCF_INTEGRALS=libcint` or
  `mojoscf.integrals.set_engine("libcint")` switches it off.
* **Nuclear gradients** (`mojoscf.grad`): `nuc_grad_method()` of `mojoscf.RHF`
  and `UHF` (and of accelerated Kohn-Sham objects), with exact or
  density-fitted integrals, returns pyscf's gradient classes with the
  derivative integrals from the Mojo engine; the
  two-electron term is evaluated directly from the derivative integrals,
  3.3 to 7.1x faster than `pyscf.grad` (exact) and 2.9 to 6.0x faster than
  `pyscf.df.grad` (DF) with the same gradients to about 1e-13.  With
  spin-free X2C the derivative of the X2C Hamiltonian, which pyscf forms
  atom by atom, is contracted with the density in one backward pass
  (Fe(II) porphine: 4 s instead of about two minutes).
* **Kohn-Sham DFT** (`mojoscf.dft`): `mojoscf.dft.accelerate(mf)` gives a
  pyscf RKS/UKS object the Mojo exchange-correlation integration (LDA, GGA
  and meta-GGA such as r2SCAN), J/K (including the long-range exchange of
  range-separated hybrids such as ωB97X), eigensolver and nuclear gradients
  while pyscf's loop drives the SCF: 3.5 to 5.7x faster SCF cycles and 4.5
  to 14x faster gradients than pyscf (9 to 12x with the response of the
  grid), with energies agreeing to 1e-10 Eh or better; the grids of
  molecules with 60 to 120 atoms build 6 to 10x faster (the Becke weights
  from the cell functions near each point).
* **Excited states** (`mojoscf.tdscf`, `mojoscf.tdgrad`): `mf.TDA()`/`mf.TDDFT()`
  of accelerated objects build the linear response in the occupied-virtual
  space (Coulomb and exchange through MO-basis DF tensors or exact
  integrals, the XC kernel in one fused pass): 3.3 to 8.6x faster than
  pyscf's operators for benzene, ferrocene and Cu(II) complexes, with the
  same excitation energies (15 to 16x in implicit solvent, PCM and SMD, as
  re-measured with the current XC response kernel);
  excited-state gradients 3.9 to 4.5x faster,
  the SCF stability analysis on the same operators 8 to 24x, and
  second-order SCF (`mf.newton()`, also ROKS and with solvent models)
  2.6 to 4.5x.
* **Analytical Hessians** (`mojoscf.hessian`): `mf.Hessian()` of
  accelerated Kohn-Sham objects (LDA, GGA, meta-GGA) and of the mojoscf
  Hartree-Fock classes keeps pyscf's driver and computes the XC terms, the
  Coulomb/exchange second derivatives (DF or exact integrals) and the
  coupled-perturbed equations with Mojo kernels in the MO basis:
  13 to 21x faster than pyscf for benzene, ferrocene and Cu(II) complexes
  (17 to 32x with VV10 or a solvent model), with the same Hessians to about
  1e-8.
* **DF-CASSCF** (`mojoscf.casscf`): pyscf's `mcscf.CASSCF` of an
  accelerated density-fitted reference builds its integrals and the J/K of
  its orbital-Hessian steps with Mojo kernels in the MO basis:
  state-averaged ligand-field CASSCF of Fe, Ni and Cu complexes 15 to 26x
  faster than pyscf (10 to 23x against pyscf with mojoscf's OpenBLAS
  setting), with the same energies, the NEVPT2 that follows 4 to 5.5x
  (5 to 7.4x with ten and twelve active orbitals),
  state-specific CASSCF gradients 8 to 10x and state-averaged gradients 13
  to 19x.
* **MC-PDFT** (`mojoscf.mcpdft`): pyscf's multiconfiguration pair-density
  functional theory (`pyscf.mcpdft`) takes its on-top energies, effective
  potentials and nuclear-gradient grid terms from Mojo kernels (the
  on-top pair density of all states in one pass over the grid, the Becke
  weight derivatives point by point): on the state-averaged CASSCF of the
  same complexes the MC-PDFT energies 13 to 23x and the gradients 14 to
  17x faster than pyscf.
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
| H2, R = 2.0 Å / cc-pVDZ             |  10 |             80.9 |  0.904 |    7/7 |      0.03 |        0.01 |    2.4x  | 2e-15 |
| H2, R = 3.0 Å / cc-pVDZ             |  10 |            172.3 |  0.995 |    6/6 |      0.02 |        0.01 |    2.5x  | 9e-16 |
| H10 chain, AFM / 6-31G              |  20 |            298.3 |  3.693 |    8/8 |      0.03 |        0.02 |    2.2x  | 3e-14 |
| H20 chain, AFM / 6-31G              |  40 |            595.3 |  7.266 |    8/8 |      0.04 |        0.03 |    1.4x  | 5e-14 |
| H30 chain, AFM / 6-31G              |  60 |            892.3 | 10.840 |    8/8 |      0.11 |        0.07 |    1.5x  | 1e-13 |
| H40 chain, AFM / 6-31G              |  80 |           1189.3 | 14.413 |    8/8 |      1.12 |        0.23 |    4.8x  | 1e-13 |
| N2, R = 2.2 Å / cc-pVDZ             |  28 |            346.1 |  1.993 |  13/13 |      0.05 |        0.02 |    2.5x  | 1e-13 |
| F2, R = 2.6 Å / cc-pVDZ             |  28 |            304.6 |  1.001 |    8/8 |      0.04 |        0.03 |    1.4x  | 0     |
| twisted C2H4 (90°) / cc-pVDZ        |  48 |             98.3 |  1.035 |  10/10 |      0.09 |        0.05 |    1.7x  | 2e-13 |
| [Cu2Cl6]2-, AFM / def2-SVP (DF)     | 170 |                  |  1.009 |  10/10 |     16.79 |        3.95 |    4.3x  | 2e-11 |
| [Fe2S2(SH)4]2-, AFM / def2-SVP (DF) | 190 |                  |  4.988 |  38/38 |     71.53 |       12.70 |    5.6x  | 3e-12 |

The two metal dimers are antiferromagnetically coupled singlets prepared by
flipping the spin of one metal centre in the converged high-spin density
(`flip_spin_on_atoms`): Cu(II)/Cu(II) (Mulliken spin +0.842 / -0.842, BS 0.18 mEh
above the triplet) and Fe(III)/Fe(III) with five unpaired electrons per iron
(Mulliken spin +3.88 / -4.54, BS 16.5 mEh below the S = 5 state), the latter
needing 38 cycles with a 0.3 Eh level shift.  The two dimers gained most from
the per-worker DF exchange (*The exchange update of small and medium
molecules* below): 1.7x and 2.0x before.  The N2 row is one of two
broken-symmetry states the mix guess can reach (on another machine both
codes reached the other one, 190.6 mEh, <S^2> = 1.018), and the RHF reference
of twisted ethylene has degenerate pi orbitals; see the degenerate-shell caveat
below.  Measured on the 2.1 GHz machine with the default Mojo integrals.  On a
2.8 GHz Cascade Lake VM the level-shifted [Fe2S2] iteration took 88 cycles in
pyscf and 108 in mojoscf, with energies 2e-7 Eh apart at conv_tol 1e-8: that
SCF is sensitive to the rounding of the CPU's BLAS kernels (pyscf alone went
from 38 to 88 cycles); [Cu2Cl6]2- took 12.8 s against 4.2 s there.

### Transition-metal complexes

`benchmarks/bench_metals.py` runs the SCF and the nuclear gradient for
idealised complexes with def2 basis sets (which carry an f shell on the metal
already in def2-SVP; Pt with its def2 effective core potential): pyscf
versus the same object accelerated with `mojoscf.accelerate`, each in its own
process, 4 cores of a 2.1 GHz Xeon, conv_tol 1e-9.  "DF" uses pyscf's
default def2 JK-fitting basis, "direct" `max_memory=1`; "X2C" adds the
spin-free one-electron X2C Hamiltonian (`mf.x2c()`, all-electron: cisplatin
there with Pt in dyall-v2z and def2-SVP on the ligands).

| system                                         | nao | cycles | SCF pyscf [s] | mojoscf [s] | x | grad pyscf [s] | mojoscf [s] | x | \|dE\| [Eh] |
|------------------------------------------------|----:|------:|------:|------:|-----:|------:|------:|-----:|--------:|
| ferrocene / def2-SVP (DF)                      | 221 | 14/14 |   13.9 |    3.6 | 3.8x |    8.4 |   1.8 | 4.7x | 5.5e-12 |
| ferrocene / def2-TZVP (DF)                     | 415 | 16/16 |   33.1 |    9.4 | 3.5x |   24.6 |   7.3 | 3.3x | 1.8e-11 |
| [Fe(H2O)6]2+ quintet / def2-TZVP (DF, UHF)     | 303 | 45/45 |   74.1 |   16.7 | 4.4x |   14.1 |   2.3 | 6.0x | 3.2e-12 |
| [Fe(H2O)6]2+ 12H2O quintet / def2-SVP (DF, UHF) | 463 | 56/56 |  651.9 |  141.0 | 4.6x |  155.4 |  14.2 | 10.9x | 6.0e-11 |
| Fe(II) porphine triplet / def2-SVP (DF, UHF)   | 427 | 32/32 |  303.6 |   68.7 | 4.4x |  131.8 |  12.4 | 10.6x | 4.4e-10 |
| [Cu(NH3)4]2+ doublet / def2-TZVP (DF, UHF)     | 241 | 15/15 |   20.4 |    4.1 | 5.0x |    7.9 |   1.4 | 5.8x | 6.0e-11 |
| Ni(CO)4 / def2-TZVP (DF)                       | 293 | 15/15 |   16.5 |    4.6 | 3.6x |    8.7 |   2.1 | 4.1x | 3.0e-11 |
| cisplatin / def2-TZVP (DF, Pt ECP)             | 212 | 12/12 |    8.4 |    2.0 | 4.2x |    4.9 |   1.9 | 2.5x | 1.8e-12 |
| ferrocene / def2-SVP (direct)                  | 221 | 14/14 |   71.0 |   28.4 | 2.5x |   41.0 |   9.5 | 4.3x | 4.5e-13 |
| [Fe(H2O)6]2+ quintet / def2-SVP (direct, UHF)  | 175 | 28/28 |   58.9 |   19.0 | 3.1x |   14.8 |   3.3 | 4.4x | 1.8e-11 |
| cisplatin / def2-SVP (direct, Pt ECP)          | 126 | 12/12 |   12.5 |    2.7 | 4.6x |    5.7 |   1.6 | 3.5x | 1.4e-12 |
| Ni(CO)4 / def2-SVP (in-core)                   | 143 | 14/14 |    6.1 |    1.1 | 5.5x |    6.5 |   1.4 | 4.6x | 1.5e-11 |
| ferrocene / def2-TZVP (DF, X2C)                | 415 | 16/16 |   31.7 |    8.9 | 3.5x |   63.7 |   7.8 | 8.2x | 1.0e-11 |
| Fe(II) porphine triplet / def2-SVP (DF, UHF, X2C) | 427 | 35/34 | 277.6 |   72.7 | 3.8x |  207.6 |  14.9 | 13.9x | 9.4e-10 |
| cisplatin / Pt dyall-v2z (DF, X2C)             | 298 | 12/12 |   19.3 |    4.8 | 4.0x |   22.5 |   2.8 | 8.1x | 3.6e-12 |

Gradients agree to 1e-11 or better except for the direct UHF Fe(II) case
(8e-9) and Fe(II) porphine (2e-6), where the two independently converged SCF
solutions differ at that level (on the same SCF object they agree to 1e-12,
`tests/test_grad.py`).  The same benchmark on a 2.8 GHz Cascade Lake VM (an
older core, where the AVX-512 kernels of mojoscf gain less than pyscf's C
code from the higher clock) gave 2.0-4.4x for the SCF and 2.1-6.7x for the
gradients.

* **Effective core potentials** only change the one-electron Hamiltonian,
  so molecules with ECPs (4d/5d metals with def2 or similar basis sets) use
  the Mojo engine for all two-electron work: in-core ERIs, the DF tensor,
  direct J/K and the gradient's two-electron term; in the gradients only the
  ECP derivative integrals come from pyscf.  Before this, cisplatin ran on
  libcint and pyscf's code throughout.
* **Scalar-relativistic X2C** (`mf.x2c()`, spin-free, all-electron).  The
  gradient's one-electron term is then the derivative of the X2C Hamiltonian
  `h_FW = c_FW^T h c_FW`, `c_FW = [R; X R]`, formed in the uncontracted
  basis.  pyscf (`x2c.sfx2c1e_grad`) builds it as a matrix for every atom
  and coordinate from the first-order solution of the 2n x 2n modified Dirac
  equation (`c1 = (C^T h1 C - e C^T s1 C) / (e_q - e_p)`), of `X` and of `R`
  (JCP 135, 084114 (2011)): dozens of (2n)^3 products per coordinate, 3.3 s
  per atom for Fe(II) porphine / def2-SVP (733 uncontracted functions), about
  two minutes for the molecule, more than the rest of its gradient.  A
  gradient only needs `sum_ij D_ij dh_ij`, and every step from the derivative
  integrals to `dh` is linear in them, so `mojoscf.x2c` takes the density
  backwards through the chain once (each product transposed, the adjoint)
  and contracts the effective densities that come out with the derivative
  integrals of all atoms: the basis-function derivatives (`int1e_ipovlp`,
  `ipkin`, `ipnuc`, `ippnucp`) once, the nuclear-attraction operator
  derivatives in one Mojo pass and `int1e_ipprinvp` per nucleus.  For Fe(II)
  porphine the term takes 4.0 s instead of about 125 s and agrees with
  pyscf's to 6e-12 (`tests/test_x2c.py` also covers the `atom1e`
  approximation, `xuncontract=False` and a projection basis); the
  ground-state and DF-CASSCF gradients use it (`grad.hcore_deriv_dm`).
  The X2C rows of the table: whole gradients 8.2x (ferrocene / def2-TZVP,
  where the X2C term was about 39 of pyscf's 63.7 s), 13.9x (Fe(II)
  porphine) and 8.1x (all-electron cisplatin) faster than pyscf's.
  pyscf's X2C Hessian (`sfx2c1e_hess`, a second-order response per pair of
  atoms) and TDDFT gradients still form the matrices.
* **Large density-fitted SCF** (the 463-AO Fe(II) cluster, 56 UHF cycles) is
  dominated by the DF exchange build, `K = sum_Q (C^T E_Q)^T (C^T E_Q)`.
  mojoscf and pyscf do the same GEMMs, but pyscf's C transform calls the
  OpenBLAS bundled with pyscf (0.3.3, no AVX-512 kernels: 13 GFLOPS per
  thread for these shapes on this machine), while mojoscf's worker threads
  call NumPy's newer OpenBLAS pinned to one thread each (45 GFLOPS); that
  took this SCF from 1.6x to 3.5x faster than pyscf, and the per-worker
  exchange update below to 4.6x.  The gradients of the same systems are
  2.5-11x faster.
* **The exchange update of small and medium molecules.**  After the
  transforms, `K += U^T U` over a block of auxiliary functions is a
  tall-skinny product (nao x nao from naux_block x nocc rows) that a
  threaded BLAS splits poorly: for nao = 170, 2x on 4 threads.  Up to about
  2000 basis functions on 4 cores every worker now transforms a contiguous
  range of auxiliary functions and accumulates its own K with one BLAS
  thread, the partial matrices summed at the end: one UHF J/K build of the
  [Cu2Cl6]2- dimer below went from 0.55 to 0.26 s (pyscf 0.74 s), its
  broken-symmetry SCF from 7.9 to 3.7 s (2.8 GHz machine).  Transforming both
  spins in one GEMM per auxiliary function (a larger M) measured no gain:
  these shapes run at about 54 GFLOPS per core in OpenBLAS whether 94, 186 or
  372 orbitals are transformed.
* **DF tensors on disk.**  pyscf keeps the DF tensor in an HDF5 file when it
  does not fit in 90% of `max_memory` (4000 MB by default; Fe(II)
  porphine/def2-TZVP needs about 7 GB) or when a file is named in
  `with_df._cderi_to_save`, and then reads it block by block in every J/K
  build.  mojoscf builds that file with the Mojo integrals (blocks of AO
  shells, `integrals.int3c2e_cols`, written as the columns of one contiguous
  dataset, the layout of pyscf's `_compatible_format`) and maps it into
  memory when it fits in the memory the system has available: the page
  cache then holds it (pyscf's `max_memory` does not count the page cache),
  and the native loop and the response, stability and Hessian kernels read
  it as an in-memory tensor.  Larger files, and pyscf's own block-format
  files, are streamed through the J/K kernel (`kernels.df_jk_blocks`, each
  block read while the previous one is processed).  Ferrocene/def2-TZVP with
  `max_memory=1000`: building the file 5.1 s (pyscf) against 2.9 s, one J/K
  2.05 s against 0.50 s, the SCF 33.0 s against 7.4 s (4.5x, as with the
  tensor in memory; streaming instead of mapping: 21.8 s).  Fe(II)
  porphine/def2-TZVP UHF (861 basis functions, a 6.8 GB tensor;
  `bench_metals.py --heavy`): 100 SCF cycles 3241 s against 653 s (5.0x;
  neither run converges from the default guess within 100 cycles), the
  gradient 319 s against 46 s (7.0x).
* **The DF object.**  The `with_df` of accelerated objects (and of
  `mojoscf.RHF(mol).density_fit()`) is a `mojoscf.dft.MojoDF`, pyscf's
  `df.DF` whose `get_jk` runs the Mojo kernel (in memory, mapped or
  streamed), so pyscf code that calls the DF object directly, such as
  DF-CASSCF (`mcscf.CASSCF` of a density-fitted reference) for its core
  potentials, gets it too (its integrals and orbital-Hessian steps: see
  *DF-CASSCF* below).
* **Segmented basis sets** (def2) consist largely of single-primitive shells,
  where vectorising over the primitive quartets of one shell quartet leaves
  most SIMD lanes empty.  The drivers therefore batch kets as SIMD lanes
  (see *SIMD over kets* below): one direct J/K build for ferrocene / def2-SVP
  went from 2.11 s to 1.60 s (pyscf 4.2-4.7 s), for def2-TZVP from 18.5 s to
  16.4 s (pyscf 36.6 s), and for [Fe(H2O)6]2+ / def2-SVP from 0.78 s to
  0.57 s.  The f shells of def2-TZVP give pairs beyond the lane kernels'
  Hermite degree, which keep the single-quartet path.
* **Integral-direct J/K, second pass.**  A profile of one direct J/K build of
  ferrocene / def2-SVP (9.4 s of thread time on 4 cores) split it into the
  lane kernel (3.2 s, two thirds of it the Boys-table gathers: see *SIMD over
  primitive quartets* below), the J/K digestion (2.9 s, mostly per-quartet
  overhead: clearing scratch, short loops of varying length), the per-ket
  transform (1.8 s) and the rest.  The table rows are now loaded whole and
  transposed in registers, the digestion is unrolled over the ket's
  functions and adds straight into the matrices, kets with one primitive
  pair stay one per SIMD lane through the J/K contractions, and the bras with
  the most primitive pairs (the most kets) are taken first.  One J/K build
  from the minao density, 4 cores of a 2.8 GHz Xeon, before -> after (pyscf):
  ferrocene / def2-SVP 2.14 -> 1.54 s (4.65 s), [Fe(H2O)6]2+ / def2-SVP
  0.76 -> 0.54 s (1.64 s), cisplatin / def2-SVP 0.23 -> 0.16 s (0.75 s),
  C8H18 / cc-pVDZ 1.13 -> 0.75 s (2.67 s).  The direct SCF of the three
  "direct" rows of the table (measured before this change) on that machine:
  34.5 -> 23.7 s (pyscf 73.0 s), 25.8 -> 18.1 s (60.1 s) and 3.4 -> 2.6 s
  (12.9 s), i.e. 3.1x, 3.3x and 5.0x pyscf's speed (2.1-3.8x before), with
  the same energies.  Their gradients, after the derivative integrals were
  moved to the ket side and batched as well (*Gradient two-electron term*
  below): 12.2 -> 9.3 s (pyscf 41.9 s), 3.8 -> 2.7 s (17.5 s) and
  1.6 -> 1.2 s (7.3 s), i.e. 4.5x, 6.5x and 6.1x.
* **Larger clusters.**  One direct J/K build from the minao density,
  pyscf against mojoscf: benzene / cc-pVDZ 0.67 s against 0.11 s (5.9x),
  (H2O)20 / cc-pVDZ 15.1 s against 7.5 s (2.0x), C40H82 / 6-31G 18.8 s
  against 9.3 s (2.0x), [Fe(H2O)6]2+ 12H2O / def2-SVP 13.0 s against 7.5 s
  (1.7x).  Stack samples of the (H2O)20 build put about 40% of the
  workers' time in the single-primitive ket batches (`flush_batch1`: the
  lane kernel and the per-lane gathers and scattered additions into rows
  of nao elements) and 20% in the bra's ket loop.  Two changes measured no
  gain and were dropped: the kets of each primitive-pair class sorted by
  Schwarz bound, so that a bra's loop stops at its first ket below
  `tol / (q_ab min(1, 4 max|D|))` (10-20% slower with the minao density,
  the digestion losing the shell order of the kets; within 8% either way
  with a density scaled by 1e-5, as in late incremental cycles), and the
  Boys function from the asymptotic form alone when no lane of a vector is
  within the table (no change).
* **Exact-integral Hessians.**  The two-electron second-derivative term
  (`hess2e_core`) was over half of mojoscf's Hartree-Fock Hessian of
  Ni(CO)4 / def2-SVP (14.5 of 26.2 s on the 2.8 GHz machine).  With
  translational invariance, the derivative components contracted inside
  the kernel and the quartets of a derivative pair summed in one call
  (*Hessian two-electron term* below) it takes 2.8 s.  That left the CPHF
  J/K over the in-core integrals in front (3N first-order densities per
  iteration, twice for UHF): with three or more densities the in-core J/K
  now runs them as SIMD lanes (*In-core J/K of many densities* below), 27
  densities of Ni(CO)4 in 0.12 s instead of 0.50 s (pyscf's `dot_eri_dm`
  1.17 s).  The first-order Fock matrices then moved to the gradient
  driver's integral scheme with the blocks folded per atom (*First-order
  Fock matrices of Hessians* below; [Fe(H2O)6]2+ 20.9 -> 19.8 s), the UHF
  CPHF operator takes J of the spin sums only, with K per spin, in the same
  call (`dft.exact_jk_sets`; it had J of every spin density and added them:
  18.7 s), and the CPHF equations are solved by `mojoscf.cphf`, pyscf's
  Krylov solver with its steps as matrix products: `lib.krylov` runs the
  Gram-Schmidt projections, the QR of each block of new vectors and the
  subspace matrix as Python loops over single vectors (a `dot` and a
  full-size temporary per pair; 2.7 s of the [Fe(H2O)6]2+ Hessian, 513
  trial vectors of 14700 elements, growing quadratically with the system).
  Hessians from the same SCF, 4 cores of a 2.3 GHz Xeon:

  | system                       | before | `hess2e` | + J/K  | + `h1`, CPHF | pyscf   |
  |------------------------------|-------:|---------:|-------:|-------------:|--------:|
  | RHF Ni(CO)4 / def2-SVP       | 15.7 s |   8.5 s  |  5.3 s |        5.0 s |  72.7 s |
  | UHF [Fe(H2O)6]2+ / def2-SVP  | 69.8 s |  53.5 s  | 20.9 s |       16.4 s | 240.7 s |

  i.e. 4.6x -> 14.5x and 3.4x -> 14.7x pyscf's speed; the Hessians agree
  with each other to 9e-10 and with pyscf's (from its own SCF) to 1.4e-8.
  Without the ERIs in core (integral-direct CPHF J/K, *Integral-direct J/K
  of many densities* below) the UHF Hessian of [Fe(H2O)6]2+ took 81.4 s
  and takes 25.4 s before the Krylov solver.

### DF-CASSCF of metal complexes

`benchmarks/bench_casscf.py` runs pyscf's CASSCF (`mcscf.CASSCF`, conv_tol
1e-8) on a density-fitted ROHF reference, averaged with equal weights over
all states of the metal 3d active space from AVAS (singly occupied orbitals
kept active): the ligand-field states of the high-spin ion.  pyscf versus
the same objects with the reference accelerated by `mojoscf.accelerate`,
each in its own process, 4 cores of a 2.1 GHz Xeon.  "tuned" is pyscf with
`OPENBLAS_THREAD_TIMEOUT=16`, the OpenBLAS spin-wait mojoscf sets when it is
imported (see *BLAS/LAPACK backend*); macro iterations and orbital-Hessian
steps are given as pyscf/mojoscf.

| system                           | nao | CAS   | states | macro | AH steps | CASSCF pyscf [s] | tuned [s] | mojoscf [s] | x (tuned)     | SCF pyscf [s] | mojoscf [s] | \|dE\| [Eh] |
|----------------------------------|----:|------:|-------:|------:|---------:|-----------------:|----------:|------------:|--------------:|--------------:|------------:|------------:|
| [Fe(H2O)6]2+ quintet / def2-SVP  | 175 | (6,5) |      5 |   3/3 |    48/49 |             72.3 |      49.5 |         4.7 | 15.3x (10.5x) |          21.5 |         3.7 |     5.2e-12 |
| [Ni(H2O)6]2+ triplet / def2-SVP  | 175 | (8,5) |     10 |   4/4 |    72/64 |            100.8 |      70.2 |         5.0 | 20.1x (14.0x) |          15.6 |         2.8 |     2.7e-12 |
| [Cu(NH3)4]2+ doublet / def2-SVP  | 147 | (9,5) |      5 |   4/4 |    82/84 |             77.3 |      43.3 |         4.2 | 18.5x (10.3x) |           9.4 |         1.5 |     3.4e-11 |
| [Fe(H2O)6]2+ quintet / def2-TZVP | 303 | (6,5) |      5 |   3/3 |    65/61 |            307.0 |     274.7 |        11.9 | 25.7x (23.1x) |          34.9 |         8.8 |     3.8e-11 |
| [Cu(NH3)4]2+ doublet / def2-TZVP | 241 | (9,5) |      5 |   4/4 |    91/89 |            211.1 |     168.8 |         8.2 | 25.7x (20.6x) |          16.8 |         3.5 |     3.0e-11 |

`--nevpt2` adds pyscf's strongly contracted NEVPT2 (`mrpt.NEVPT`) of the
three lowest states, a CASCI on each run's state-averaged orbitals:

| system                           | NEVPT2 pyscf [s] | tuned [s] | mojoscf [s] | x (tuned)   | max \|dE\| [Eh] |
|----------------------------------|-----------------:|----------:|------------:|------------:|----------------:|
| [Fe(H2O)6]2+ quintet / def2-SVP  |             15.7 |      14.9 |         3.7 | 4.3x (4.0x) |         6.1e-08 |
| [Ni(H2O)6]2+ triplet / def2-SVP  |             15.3 |      15.8 |         3.8 | 4.0x (4.2x) |         4.9e-08 |
| [Cu(NH3)4]2+ doublet / def2-SVP  |             10.9 |       6.4 |         2.0 | 5.5x (3.2x) |         5.2e-07 |
| [Fe(H2O)6]2+ quintet / def2-TZVP |             61.8 |      56.4 |        12.9 | 4.8x (4.4x) |         7.0e-08 |
| [Cu(NH3)4]2+ doublet / def2-TZVP |             26.9 |      26.3 |         5.0 | 5.3x (5.3x) |         9.1e-06 |

The NEVPT2 energies inherit the difference of the two runs' CASSCF orbitals
(each converged to conv_tol 1e-8; NEVPT2 is not stationary in them): on the
same orbitals pyscf's and mojoscf's integrals give the same energies to
1e-10.  The CASSCF energies agree to 4e-11 Eh, in the same number of macro
iterations.

pyscf's driver, FCI solver and orbital optimiser run unchanged; with an
accelerated reference two of its parts come from `mojoscf.casscf`:

* **The integrals of each macro iteration** (`pyscf.mcscf.df._ERIS`).  pyscf
  transforms the DF tensor to (Q|pq) for all orbital pairs, writes it to a
  temporary HDF5 file and reads it back to form ppaa = (pq|uv) and
  papa = (pu|qv) with the active orbitals u, v: 3.6 s per macro iteration
  for [Fe(H2O)6]2+/def2-SVP, a sixth of its time.  mojoscf transforms once
  with the Mojo kernel and keeps (Q|pq) in memory for the macro iteration
  (naux nmo^2 doubles, 0.23 GB here) when it fits in `max_memory`.  The
  same pass (`kernels.casscf_mo`) forms `j_pc`, `k_pc` and the core
  potential (2J - K of the core density, in the MO basis; K as a rank-ncore
  SYRK of each B_Q while it is in cache), `ppaa` and `papa` come from the
  kept tensor: 0.34 s.  Within a CASSCF `kernel` each macro iteration's
  tensor reuses the memory of the previous one: pyscf drops its integrals
  before it builds the next, but under Python < 3.13 the `locals()`
  snapshots it hands to its schedulers keep them alive, so a second tensor
  would be allocated (twice the memory, and on the 3.3 GB tensor of iron
  porphyrin/def2-SVP the fresh pages cost 1 to 2 s per macro iteration).
* **The J/K of the orbital-Hessian steps** (`update_jk_in_ah`), two thirds
  of pyscf's time.  pyscf builds Coulomb and exchange matrices in the AO
  basis for two densities of rank 2 ncore and 2 ncore + 2 ncas (1.1 s per
  step with `df_jk.get_jk`, 0.2 s with the Mojo DF kernel) and keeps only
  the active rows of the first and the core-noncore block of the second.
  mojoscf computes just these rows in the MO basis from the kept tensor,
  in one pass over it (`kernels.casscf_ah`).  The first density has only
  the core-noncore blocks of the rotation R, so B_Q[occ, :] X3 is two thin
  GEMMs with R (4 nocc ncore nvir flops instead of 2 nocc nmo^2 for a dense
  X), the exchange rows follow with one GEMM per four auxiliary functions
  (their B_Q[:, x] are rows of one strided matrix), the active-core block
  from the symmetry of B X3 B, and J from rho_Q = sum_rs B_Q,rs X_rs with
  the same B_Q: 34 ms per step (62 ms with two dense sandwich products; on
  iron porphyrin 0.72 s instead of 1.36 s).

**NEVPT2** builds its integrals with `mojoscf.casscf.nevpt2_eris`.  pyscf transforms the DF
tensor four times (once over all orbital pairs, for `ppaa`) and stores
(cv|cv), (ncore nvir)^2 doubles: 0.8 GB for [Fe(H2O)6]2+/def2-TZVP, growing
with the fourth power of the system and moved to disk when it does not fit.
mojoscf transforms to (Q|up) and (Q|cv) only, forms (pq|uv) from the ncas^2
AO matrices sum_Q E_Q (Q|uv), and evaluates the one subspace that needs
(cv|cv) (Sijrs) from the DF factors, one core orbital at a time and for
j <= i only; the three-index `einsum` contractions of the Srsi and Srs
subspaces run as GEMMs.  The other subspaces are pyscf's, with BLAS
contraction paths for their some 230 `einsum` calls over the active-space
2- and 3-particle density matrices (5 to 10x on those for ten active
orbitals).

With ten or more active orbitals the reduced density matrices dominate:
pyscf evaluates the 3-particle matrix <p+q r+s t+u> and the two 4-particle
contractions of strongly contracted NEVPT2, sum_xyz (fx|yz) <E_ab E_cd E_ex E_yz>
and sum_xyz (ex|yz) <E_ab E_cd E_xf E_yz>, with C kernels over the FCI
strings (68 s and 2 x 89 s for a CAS(12,12) vector here, against 0.3 s for
the 2-particle matrix).  `mojoscf.fcirdm` (installed when mojoscf is
imported, for any NEVPT2) forms the vectors E_tu|ket> and E_qp|bra> over all
determinants; every single excitation of an alpha string then adds one GEMM
over the beta strings (of a beta string, over the alpha strings) to the
p >= r >= t part of the 3-particle matrix, which pyscf's `_complete_dm3_`
completes.  The 4-particle contractions are 3-particle matrices with the
third vector O_ef|ci> = sum_x E_ex W_fx|ci>, W_fx = sum_yz (fx|yz) E_yz, (or
sum_x E_xf W_ex|ci>), one more GEMM: 9 s and 2 x 16 s.
`benchmarks/bench_nevpt2.py` runs a CASCI with the orbitals around the Fermi
level and pyscf's NEVPT2 of its ground state on a density-fitted reference,
pyscf versus mojoscf in separate processes:

| system                                       | nao | CASCI pyscf [s] | mojoscf [s] | NEVPT2 pyscf [s] | mojoscf [s] | x    | \|dE2\| [Eh] |
|----------------------------------------------|----:|----------------:|------------:|-----------------:|------------:|-----:|-------------:|
| ferrocene singlet / def2-SVP, CAS(10,10)     | 221 |             3.3 |         0.6 |             21.3 |         4.2 | 5.0x |      5.0e-11 |
| [Fe(H2O)6]2+ quintet / def2-SVP, CAS(12,12)  | 175 |             2.8 |         0.7 |             87.0 |        11.8 | 7.4x |      2.2e-13 |
| ferrocene singlet / def2-SVP, CAS(12,12)     | 221 |             5.6 |         3.0 |            272.7 |        39.1 | 7.0x |      4.0e-09 |

The two work vectors take 2 ndet ncas^2 doubles (2 GB for CAS(12,12)); when
they do not fit (within pyscf's `MAX_MEMORY` or half the available memory)
pyscf's kernels run.  On the same CASCI vector the matrices agree with
pyscf's to 1e-15 relative (tests); the energy differences above come from
the two runs' separately converged references.

**Nuclear gradients** of the state-specific CASSCF (`mc.nuc_grad_method()`,
`bench_casscf.py --grad`: the lowest state, conv_tol 1e-10) come from
`mojoscf.casscf.Gradients`.  pyscf evaluates the DF gradient J/K of the core
and active densities and the active two-particle terms in separate passes
over libcint's three-centre derivative integrals.  The two-electron part is
the derivative of E2(Dc + Da) - E2(Da) + 1/2 sum (uv|wx) G_uvwx (E2(D) =
1/2 Tr D (J - K/2)[D], Dc and Da the core and active densities, G the
2-RDM); in the basis of the core and the active natural orbitals all its
Coulomb, exchange and 2-RDM terms become one matrix per auxiliary function
and the metric terms one W, so one pass of the Mojo DF gradient kernels
(`integrals.grad2e_df_casscf`) gives it, auxiliary response included.

| system                           | nao | CAS   | grad pyscf [s] | tuned [s] | mojoscf [s] | x (tuned)    | max \|dg\| [Eh/bohr] |
|----------------------------------|----:|------:|---------------:|----------:|------------:|-------------:|---------------------:|
| [Fe(H2O)6]2+ quintet / def2-SVP  | 175 | (6,5) |            9.5 |       8.6 |         1.0 | 9.8x (8.6x)  |              7.8e-09 |
| [Ni(H2O)6]2+ triplet / def2-SVP  | 175 | (8,5) |            9.4 |       7.5 |         1.0 | 9.3x (7.5x)  |              2.1e-07 |
| [Cu(NH3)4]2+ doublet / def2-SVP  | 147 | (9,5) |            5.6 |       4.1 |         0.7 | 8.2x (5.9x)  |              7.0e-13 |
| [Fe(H2O)6]2+ quintet / def2-TZVP | 303 | (6,5) |           27.2 |      30.9 |         3.4 | 7.9x (9.1x)  |              7.1e-09 |
| [Cu(NH3)4]2+ doublet / def2-TZVP | 241 | (9,5) |           18.7 |      14.5 |         1.8 | 10.5x (8.1x) |              1.2e-08 |

The gradients of the two runs differ by what their separately converged
orbitals do; on the same CASSCF solution they agree to 1e-12 (tests).

**State-averaged gradients** (`mc.nuc_grad_method().kernel(state=i)` after
`mc.state_average_`; `bench_casscf.py --sa-grad`: the lowest state, after
the state-averaged CASSCF of the first table) stay pyscf's
`df.grad.sacasscf.Gradients`, whose parts now run on mojoscf: the response
equations for the Lagrange multipliers (pyscf's conjugate-gradient solver)
on the accelerated integrals and orbital-Hessian J/K, the state's own
gradient from `mojoscf.casscf.Gradients`, and the two Lagrange terms.  pyscf
evaluates each Lagrange term with a DF gradient J/K of four (orbital term)
or two (CI term) densities and one or two further passes over the
derivative integrals for the active two-particle terms.  With the
multipliers L, the orbital term is the derivative of
Tr(DL (J - K/2)[Dc]) + Tr(DLc (J - K/2)[Da]) + sum (uv|w'x) G_uvwx (DL the
densities with one orbital rotated by L, w' a rotated index), in the basis
[C_core, C_act, C_core L, C_act L], and the CI term that of
Tr(DT (J - K/2)[Dc]) + 1/2 sum (uv|wx) T_uvwx with the symmetrised
transition densities DT and T.  mojoscf evaluates both terms together
(`casscf.lagrange_terms`, in place of pyscf's `get_LdotJnuc`): one J/K build
for their five densities, one loop over the atoms' one-electron derivative
integrals and one pass of the Mojo DF gradient kernels for both two-electron
parts, the CI term written in the orbital term's basis
(`integrals.grad2e_df_terms`, auxiliary response included).

| system                           | nao | CAS   | states | SA grad pyscf [s] | tuned [s] | mojoscf [s] | x (tuned)     | max \|dg\| [Eh/bohr] |
|----------------------------------|----:|------:|-------:|------------------:|----------:|------------:|--------------:|---------------------:|
| [Fe(H2O)6]2+ quintet / def2-SVP  | 175 | (6,5) |      5 |              65.4 |      57.7 |         5.0 | 13.1x (11.5x) |              8.2e-09 |
| [Ni(H2O)6]2+ triplet / def2-SVP  | 175 | (8,5) |     10 |              66.1 |      55.4 |         4.8 | 13.7x (11.5x) |              3.8e-09 |
| [Cu(NH3)4]2+ doublet / def2-SVP  | 147 | (9,5) |      5 |              42.8 |      27.7 |         2.7 | 15.8x (10.3x) |              1.2e-08 |
| [Fe(H2O)6]2+ quintet / def2-TZVP | 303 | (6,5) |      5 |             205.7 |     197.4 |        10.8 | 19.1x (18.3x) |              9.8e-09 |
| [Cu(NH3)4]2+ doublet / def2-TZVP | 241 | (9,5) |      5 |             104.6 |      85.4 |         5.8 | 17.9x (14.7x) |              1.2e-08 |

On the same CASSCF solution the Lagrange terms agree with pyscf's to
better than 1e-10 for arbitrary multipliers and the state-averaged
gradients to better than 1e-9 (tests).

When the MO-basis tensor does not fit, the integrals are built block by
block and the orbital-Hessian J/K runs on the Mojo DF kernel in the AO
basis.  For open-shell references the AVAS option matters: pyscf's default
(`openshell_option=2`) rotates the singly occupied orbitals together with
the doubly occupied ones, which here moved part of them into the core (CAS
reference energies 69 to 185 mEh above ROHF; single-state CASSCF of
[Fe(H2O)6]2+ then did not converge within 50 macro iterations in either
code); `openshell_option=3` keeps them active.

### MC-PDFT of metal complexes

Multiconfiguration pair-density functional theory (`pyscf.mcpdft`, e.g.
`mcpdft.CASSCF(mf, "tPBE", ncas, nelecas)`) runs a CASSCF and then
evaluates, for every state, the on-top energy E_ot[rho, Pi] on the DFT
grid, with the on-top pair density

    Pi = rho_a rho_b + 1/2 sum_uvxy psi_u psi_v psi_x psi_y L_uvxy

(psi the active orbitals on the grid, L the cumulant of the active-space
2-RDM) and a translated Kohn-Sham functional.  Its nuclear gradient adds the
functional's effective potentials, response equations for the CASSCF
parameters and the derivative of the grid integral with respect to the
orbitals, the grid points and the Becke weights.  With mojoscf imported, the
CASSCF, the response equations and the Lagrange terms run on
`mojoscf.casscf` (for an accelerated density-fitted reference, as above),
and `mojoscf.mcpdft` replaces pyscf's grid code:

* **The on-top energies** (`otfnal.energy_ot`, called by
  `compute_pdft_energy_` once per state).  pyscf loops over blocks of grid
  points: AO values, two spin densities, the active orbitals on the grid,
  their pair products contracted with L by NumPy `tensordot`, libxc per
  block, and it rebuilds the grid for every state.  mojoscf computes the
  spin densities and Pi of all states in one Mojo pass (per block of 128
  points one GEMM for the core density and one for the active orbitals
  psi = C^T phi, shared by the states; per state the active densities and
  t = L^T q, q_kl = psi_k psi_l) before pyscf's per-state calls, which take
  the results, and evaluates pyscf's translated functional once per state
  on the whole grid.
* **The effective potentials of the gradient** (`pdft_veff.kernel` with
  `paaa_only`, as the MC-PDFT and MS-PDFT gradients call it): the same
  density pass, the one-body potentials with the XC potential-matrix kernel
  and the paaa block sum_p phi_mu v_Pi psi_u psi_v psi_w in one more pass.
* **The grid part of the Hellmann-Feynman gradient**
  (`grad.mcpdft.mcpdft_HellmanFeynman_grad`).  Per atom's grid the orbital
  terms come from the XC gradient kernel with the kernel v_rho + v_Pi rho/2
  and from one pass for the cumulant term, the grid-point term of an atom
  is minus the sum of its points' orbital terms, and the Becke weight
  derivatives, which pyscf forms as (natm, natm, 3) NumPy arrays per point,
  are contracted with the energy density point by point in a Mojo kernel;
  the Coulomb term comes from the Mojo DF derivative integrals (pyscf built
  a DF gradient J/K for it).

`bench_casscf.py --pdft` runs tPBE MC-PDFT (`mcpdft.CASSCF`, default grids)
on the state-averaged CASSCF of the first table: the MC-PDFT energies of
all states and the nuclear gradient of the lowest (the CASSCF itself took
70 to 285 s with pyscf and 3.9 to 11.9 s with mojoscf in this run, 17 to
25x):

| system                           | nao | CAS   | states | energies pyscf [s] | tuned [s] | mojoscf [s] | x (tuned)     | gradient pyscf [s] | tuned [s] | mojoscf [s] | x (tuned)     | max \|dE\| [Eh] | max \|dg\| [Eh/bohr] |
|----------------------------------|----:|------:|-------:|-------------------:|----------:|------------:|--------------:|-------------------:|----------:|------------:|--------------:|-----------------:|---------------------:|
| [Cu(NH3)4]2+ doublet / def2-SVP  | 147 | (9,5) |      5 |                6.7 |       5.5 |        0.51 | 13.2x (10.8x) |               61.7 |      43.2 |         3.9 | 15.7x (11.1x) |          3.1e-06 |              2.2e-07 |
| [Fe(H2O)6]2+ quintet / def2-SVP  | 175 | (6,5) |      5 |               10.0 |       7.3 |        0.57 | 17.4x (12.7x) |               88.0 |      73.5 |         6.1 | 14.4x (12.0x) |          1.0e-07 |              2.7e-08 |
| [Ni(H2O)6]2+ triplet / def2-SVP  | 175 | (8,5) |     10 |               20.4 |      14.6 |        0.88 | 23.1x (16.5x) |               89.4 |      75.9 |         6.3 | 14.2x (12.0x) |          1.1e-07 |              3.8e-08 |
| [Fe(H2O)6]2+ quintet / def2-TZVP | 303 | (6,5) |      5 |               14.7 |      13.0 |        0.81 | 18.2x (16.1x) |              234.2 |     207.1 |        13.6 | 17.2x (15.2x) |          8.2e-08 |              7.0e-08 |
| [Cu(NH3)4]2+ doublet / def2-TZVP | 241 | (9,5) |      5 |               10.5 |       9.3 |        0.60 | 17.4x (15.4x) |              121.3 |     107.3 |         7.1 | 17.0x (15.1x) |          7.7e-06 |              2.1e-07 |

Of the [Fe(H2O)6]2+/def2-SVP gradient (6.8 s in a profiled run) the
state-averaged CASSCF response takes most: 1.5 s the conjugate-gradient
solve on the orbital-Hessian J/K and 2.3 s the two Lagrange terms (one DF
gradient pass for both); the Hellmann-Feynman part takes 2.0 s and the
effective potentials 0.7 s.  As for NEVPT2, the energies and gradients of the two runs
differ through their separately converged CASSCF orbitals (MC-PDFT is not
stationary in them; most for the Cu doublets); on the same CASSCF solution
the on-top energies agree with pyscf's to 1e-9 Eh or better (translated and
fully translated GGAs, a hybrid and a meta-GGA) and the state-averaged
MC-PDFT gradients to 1e-8 (tests).

### Kohn-Sham DFT

`mojoscf.dft.accelerate(mf)` gives a pyscf RKS/UKS object (with or without
density fitting, also QM/MM) the Mojo kernels while pyscf's SCF loop stays in
charge: the exchange-correlation integration (`mojoscf.dft.NumInt`; LDA,
GGA and meta-GGA functionals), J/K (from the DF tensor, from in-core ERIs
built by the engine, or integral-direct), the eigensolver and DIIS, and the
nuclear gradients (`mojoscf.grad.KSGradients` and its UKS and DF variants).
DFT+U objects (`dft.RKSpU`, `dft.UKSpU`) keep pyscf's Hubbard potential and
energy on top of these, and their gradients are the Mojo ones plus pyscf's
Hubbard term (`grad.KSpUGradients`, `UKSpUGradients`).
`benchmarks/bench_dft.py`: pyscf versus the same object accelerated, each in
its own process, pyscf's default grids (level 3), conv_tol 1e-9, 4 cores of
a 2.1 GHz Xeon.

| system                                          | nao | grid points | cycles | SCF pyscf [s] | mojoscf [s] | x | grad pyscf [s] | mojoscf [s] | x | \|dE\| [Eh] |
|-------------------------------------------------|----:|-------:|------:|------:|------:|-----:|------:|-----:|-----:|--------:|
| benzene / def2-SVP B3LYP (DF)                   | 114 | 143560 |   7/7 |   8.7 |   2.3 | 3.8x |   3.3 |  0.5 | 6.6x | 6.8e-13 |
| benzene+ / def2-SVP B3LYP (DF, UKS)             | 114 | 143560 |   9/9 |  16.1 |   3.9 | 4.1x |   5.1 |  0.6 | 9.0x | 5.1e-13 |
| C8H18 / def2-TZVP PBE (DF)                      | 356 | 289488 | 10/10 |  55.2 |  12.3 | 4.5x |  23.3 |  2.8 | 8.4x | 1.8e-12 |
| ferrocene / def2-SVP PBE (DF)                   | 221 | 260168 | 20/20 |  60.5 |  13.9 | 4.4x |  11.6 |  1.6 | 7.1x | 1.1e-10 |
| ferrocene / def2-SVP B3LYP (DF)                 | 221 | 260168 | 72/41 | 233.6 |  32.1 | 7.3x |  17.0 |  2.7 | 6.3x | 2.1e-09 |
| ferrocene / def2-TZVP PBE (DF)                  | 415 | 260168 | 20/20 | 138.0 |  28.1 | 4.9x |  27.4 |  5.1 | 5.4x | 1.1e-11 |
| [Fe(H2O)6]2+ quintet / def2-TZVP PBE0 (DF, UKS) | 303 | 223096 | 41/41 | 253.5 |  48.8 | 5.2x |  32.5 |  3.4 | 9.6x | 4.5e-12 |
| [Cu(NH3)4]2+ doublet / def2-TZVP B3LYP (DF, UKS) | 241 | 193944 | 12/12 |  60.4 |  11.9 | 5.1x |  21.1 |  2.1 | 10.2x | 4.5e-13 |
| ferrocene / def2-SVP r2SCAN (DF)                | 221 | 260168 | 42/85 | 211.5 |  79.1 | 2.7x |  18.4 |  2.0 | 9.3x | 5.8e-09 |
| [Cu(NH3)4]2+ doublet / def2-TZVP r2SCAN (DF, UKS) | 241 | 193944 | 12/12 |  96.5 |  16.8 | 5.7x |  26.0 |  1.8 | 14.3x | 3.2e-12 |
| ferrocene / def2-SVP ωB97X (DF)                 | 221 | 260168 | 38/38 | 130.0 |  31.9 | 4.1x |  22.8 |  3.7 | 6.1x | 4.3e-11 |
| [Cu(NH3)4]2+ doublet / def2-TZVP CAM-B3LYP (DF, UKS) | 241 | 193944 | 12/12 |  61.4 |  13.5 | 4.6x |  26.1 |  3.0 | 8.8x | 9.1e-13 |
| [Cu(NH3)4]2+ doublet / def2-TZVP ωB97M-V (DF, UKS) | 241 | 193944 | 12/12 | 758.9 |  80.9 | 9.4x |  79.3 |  7.5 | 10.5x | 2.7e-12 |
| (H2O)5 / def2-TZVP PBE (in-core)                | 215 | 168496 | 12/12 |  25.0 |   7.1 | 3.5x |  16.3 |  3.6 | 4.5x | 3.4e-13 |
| C8H18 / 6-31G* B3LYP (direct)                   | 148 | 289488 |   9/9 |  32.4 |   9.1 | 3.6x |  15.1 |  2.7 | 5.7x | 2.3e-13 |
| Fe(II) porphine triplet / def2-SVP B3LYP (DF, UKS) | 427 | 476568 | 40/42 | 1044.0 | 178.1 | 5.9x | 196.9 | 15.5 | 12.7x | 9.9e-10 |

Energies agree to 1e-10 Eh or better, except for ferrocene with B3LYP and
r2SCAN, whose SCFs converge slowly and along different DIIS paths in the two
runs (72 and 41, 42 and 85 cycles; per cycle mojoscf is 4.1x and 5.4x
faster) to energies 2e-9 and 6e-9 Eh apart.  The gradients of the
independently converged SCFs differ by up to 6e-6 Eh/Bohr for these slowly
converging ferrocene runs and Fe(II) porphine, and by up to 8e-7 for the
others (conv_tol 1e-9
leaves the density uncertain at about 3e-5, and the runs take different DIIS
paths); on the same SCF solution pyscf's and mojoscf's gradients agree to
9e-12 and the energies to 4e-12 for ferrocene/PBE, ferrocene/B3LYP and
[Fe(H2O)6]2+/PBE0.  On the 2.8 GHz Cascade Lake VM the same benchmark ran
2.3-3.4x (SCF cycles) and 3.4-7.3x (gradients) faster than pyscf.

* **XC integration.**  pyscf's `NumInt` loops in Python over blocks of grid
  points: basis functions (C), density and potential matrix (GEMMs with
  sparsity masks), libxc on each block.  `mojoscf.dft.NumInt` makes two Mojo
  passes over the whole grid around a single libxc call: for each block of
  128 points it evaluates only the shells that reach a point of the block
  (per-shell cutoff radius at 1e-16, tested against the block's bounding
  sphere and, where that is not conclusive, its points), values and
  gradients SIMD over the points with an exponential accurate to one ulp and
  the spherical transforms by their nonzero coefficients, then (1) the density
  and its gradient and (2) `V^T += Z phi^T` with `Z = sum_c w_c phi_c`, one
  GEMM per block into per-thread matrices, against a transposed copy of the
  block's values so that it is row-major `A B` (OpenBLAS's small-matrix
  kernels, see the XC kernel item of the TDDFT section).  Fe(II) porphine
  triplet / def2-SVP, UKS B3LYP: 1.79 -> 1.26 s per `nr_uks` with these
  changes; testing the points drops a quarter of the shells the bounding
  sphere keeps, functions all below the cut on the block, and halves the
  basis-function evaluation with the sparse transforms.  When the density carries
  pyscf's `mo_coeff`/`mo_occ` tags (every SCF density does), blocks where it
  is cheaper use the occupied orbitals, `psi = C^T phi`,
  `rho = sum_k n_k psi_k^2`, `grad rho = 2 sum (grad phi)(C n psi)`, as
  pyscf's `eval_rho2`: for ferrocene/def2-TZVP that halves the density pass.
  Results agree with pyscf's to 1e-10 or better.
* **Gradients.**  The XC term needs the second derivatives of the basis
  functions; instead of pyscf's (3, nao, nao) matrices
  (`grad.rks.get_vxc`), the kernel contracts them with the density inside
  the pass (`-2 sum_{mu on A} D_mu nu V^x_mu nu` via `D phi` and `D Z`, two
  GEMMs per block, again through the orbitals where cheaper).  The Coulomb
  and exact-exchange part is `grad2e_df`/`grad2e` with the functional's
  exchange fraction; a pure functional needs no orbital transformation at
  all.  pyscf's own `grad.rks.get_vxc`/`grad.uks.get_vxc` also run on the
  Mojo kernels when called for a `mojoscf.dft.NumInt`.
* **Range-separated hybrids** (CAM-B3LYP, ωB97X, HSE06, ...) need the
  exchange of the long-range operator erf(ωr)/r.  Its integrals differ from
  the Coulomb ones only in the Hermite R integrals (α' with
  1/α' = 1/p + 1/q + 1/ω² instead of α) and a factor sqrt(α'/α); the engine
  gets both by replacing one side's primitive-pair exponent p by
  pω²/(p + ω²) and scaling its prefactor (`integrals.attenuate`), so every
  kernel computes them unchanged.  The long-range DF tensor of pyscf's
  `with_df.range_coulomb(ω)` is built that way (agreeing with libcint to
  1e-13) and its exchange runs in the Mojo DF kernel; the gradient adds the
  long-range term from the attenuated derivative integrals.  Without density
  fitting the in-core and integral-direct four-centre kernels and the
  derivative kernels take the same attenuated table on one side of every
  quartet (the bra; the Schwarz bounds of the full operator still bound
  the integrals), so the long-range exchange of the SCF, its gradient and
  the TDDFT terms are native too (in-core ERIs, direct J/K and derivative
  J/K agree with libcint to 1e-11).  Same-SCF
  gradients agree with pyscf to about 1e-9, the noise level of the nearly
  singular long-range metric (reciprocal condition number ~1e-22 for water
  with ωB97X), confirmed against finite differences; see the ωB97X and
  CAM-B3LYP rows of the table for timings.  (pyscf's
  `range_coulomb(ω)` is a context manager; until this was handled, the
  long-range exchange silently fell back to pyscf's code.)
* **Meta-GGA** functionals add the kinetic-energy density
  `tau = 1/2 sum_k n_k |grad psi_k|^2` to the density pass (through the
  orbitals: `C^T grad phi`, three more GEMMs), `sum_c d_c phi (w_tau d_c phi)^T`
  to the potential matrix and the corresponding second-derivative term to
  the gradient kernel; r2SCAN, SCAN, TPSS and M06-L agree with pyscf to
  about 1e-14.  The response kernels, the Hessian terms and the
  excited-state gradient contractions carry tau as well.  Functionals that
  need the laplacian of the density (BR89, ...) are rejected by pyscf as
  well.
* **VV10 non-local correlation** (ωB97M-V, ωB97X-V, B97M-V, or `mf.nlc =
  "vv10"`) is a double sum over the NLC grid, which pyscf builds as fine as
  the XC grid (level 3): for every pair of points with R2 = |r_i - r_j|^2
  the kernel 1/(g g' (g + g')), g = R2 W0_i + K_i, g' = R2 W0_j + K_j, and
  two more terms for the potential.  pyscf's `_vv10nlc` runs it in C,
  point by point over all ordered pairs, in every SCF cycle and once more
  for the gradient: 44 s per call for the 194k points of
  [Cu(NH3)4]2+/def2-TZVP.  `mojoscf.dft.vv10nlc` takes the same per-point
  terms and sums the pairs in Mojo (`_mojo/nlc.mojo`): SIMD over tiles of
  points that stay in cache, each unordered pair once when one grid serves
  as both sets (the denominator is symmetric, so one reciprocal gives the
  terms of both points), and one division per four vectors of pairs
  (1/(d1 d2 d3 d4) times the other three denominators; the division
  otherwise bounds the loop): 4.0-4.3 s per call (2.1 GHz machine), the
  same values to 5e-14.  pyscf's `numint._vv10nlc` is replaced at import, so pyscf's own
  NLC code uses it too.  The density and the potential matrix on the NLC
  grid come from the XC kernels, and the gradient adds the VV10 term to
  the native Coulomb/exchange and XC terms (`dft.grad_nlc`: the pair sums
  once, then a GGA-type contraction with the AO derivatives) instead of
  handing the whole two-electron term back to pyscf's `get_veff`; same-SCF
  gradients agree with pyscf to 1e-13 (exact integrals) and 1e-9 (DF, the
  long-range metric).  In the ωB97M-V rows the pair sums are most of
  mojoscf's time.
* **Grids of large molecules.**  `Grids.build` (once per geometry) cuts the
  space into atomic cells with Becke's partition: the weight of a point of
  atom A's grid is vol P_A / sum_B P_B, every cell function P_B a product
  of smoothed steps s(mu_BC) over all other atoms C.  pyscf's C loop
  (`VXCgen_grid`) evaluates all natm^2 / 2 atom pairs at every point, 11.8 s
  of the 1.37 million points of C40H82 (122 atoms).  `mojoscf.dft` replaces
  `gen_grid.get_partition` with a Mojo kernel.  Below 40 atoms it runs the
  same pair loop SIMD over the points (Stratmann's scheme included, which
  pyscf loops over in NumPy).  From 40 atoms only the cell functions that
  matter at a point are formed, about ten for these molecules: point by
  point, SIMD over the atoms, the row of steps of an atom C gives P_C in
  full and, from the same polynomials, one factor of every other cell
  function, so the running products bound them.  Rows are taken for the
  owner and then for the atom of the largest bound, until every bound is
  below 1e-20 times the largest cell function (11 rows per point for
  C40H82, where the pair loop does the work of 61).  The weights are
  pyscf's to rounding (<= 8e-15 of the largest weight).  Grouping the
  points into boxes (`arg_group_grids`) sorts one integer key instead of
  calling `numpy.unique` on the (n, 3) box indices.

  | `Grids.build`, level 3   | atoms | grid points | pyscf [s] | mojoscf [s] |     x |
  |--------------------------|------:|------------:|----------:|------------:|------:|
  | Fe(II) porphine          |    37 |      476568 |      0.89 |        0.30 |  2.9x |
  | C20H42                   |    62 |      694296 |      2.46 |        0.40 |  6.1x |
  | C40H82                   |   122 |     1368976 |     12.98 |        1.28 | 10.1x |
  | (H2O)40                  |   120 |     1347920 |     12.75 |        1.37 |  9.3x |

  The partition alone for C40H82: 11.8 s in pyscf, 2.2 s for the Mojo pair
  loop, 0.51 s screened.  Of mojoscf's `Grids.build` that is now under half:
  pyscf's shell screening mask (`make_screen_index`) takes 0.35 s and the
  grouping 0.33 s.  In a C40H82/6-31G* B3LYP DF calculation (724 basis
  and 4686 auxiliary functions) the grids are now 1.8 s against 110 s for
  the SCF, most of it the density-fitted exchange: 7.8 s per J/K build,
  about 1.6e12 flops at the rate of `dgemm`, where pyscf's `df_jk.get_jk`
  takes 26 s (3.3x; the DF tensor itself 46 s against pyscf's 76 s).
  Exploiting the sparsity of the exchange with localized factors of the
  density (pivoted Cholesky, blocks of columns with the basis functions they
  reach) does not pay at this size: the B3LYP density matrix of C20H42
  still has elements of 3e-3 between atoms 20 Bohr apart and 4e-5 at 40
  Bohr, the length of the chain, so every block of factor columns reaches
  all basis functions above 1e-8 and a block-sparse K would do the dense
  work.
* **Grid response.**  With `grid_response = True` the XC gradient adds the
  derivative of the grid itself: the points move with their atom and the
  Becke weights depend on every nucleus.  pyscf's `get_vxc_full_response`
  takes, for each atom's grid, the weights and their derivatives as
  (natm, natm, 3) NumPy arrays per point (`grids_response_cc`) and the AO
  second derivatives block by block.  `mojoscf.dft.grad_xc_grid_response`
  takes each grid's fixed-point term from the XC gradient kernel, the motion
  of its points from the same per-atom sums (minus their total, on the
  grid's atom), and the weight term from the Becke kernel, contracted with
  the energy density point by point.  PBE/def2-SVP (DF), each code on its
  own SCF:

  | gradient with `grid_response`      | atoms | pyscf [s] | mojoscf [s] |     x | of which grid term pyscf [s] | mojoscf [s] | max \|dg\| [Eh/Bohr] |
  |------------------------------------|------:|----------:|------------:|------:|-----------------------------:|------------:|----------------------:|
  | ferrocene                          |    21 |      15.6 |         1.7 |  9.3x |                          6.3 |        0.40 |               4.0e-07 |
  | Zn(II) porphine                    |    37 |      98.5 |         8.0 | 12.3x |                         53.8 |        2.39 |               4.0e-11 |

  (the grid term is the difference to the same gradient without it; it
  shifts the gradient by 8e-5 and 9e-5 Eh/Bohr, the same in both codes; the
  ferrocene SCFs agree to 3e-10 Eh, so their gradients only to 4e-7).
* **Thread pools.**  With pyscf's loop around the kernels, NumPy calls
  (DIIS, `make_rdm1`) alternate with Mojo passes, and every threaded OpenBLAS
  call leaves its worker threads spinning for about 0.13 s.  mojoscf now uses
  NumPy's own OpenBLAS (ILP64) as its threaded library, so both share one
  pool, and avoids threaded NumPy BLAS between its passes (the energy sum;
  `mojoscf.CDIIS` replaces pyscf's default CDIIS, same iterates): the
  ferrocene/def2-SVP SCF went from 17.6 s to 13.5 s with that alone.

### Implicit solvation (PCM, SMD)

pyscf's PCM family (C-PCM, COSMO, IEF-PCM, SS(V)PE) and SMD represent the
solvent by Gaussian charges on the molecular surface.  Each SCF cycle needs
the potential of the electron density at the surface points and the matrix
of the induced charges, which pyscf builds from three-centre integrals over
the surface (`int3c2e` with a "fakemol", contracted with `einsum`), plus two
dense solves of the PCM equations; the gradient forms the (3, n, n)
geometry derivatives of the surface matrices.  `mojoscf.solvent.attach(mf)`
(done by `mojoscf.dft.accelerate`) gives the solvent object

* the potential at the points from one density-contracted pass (each shell
  pair's Hermite matrices contracted with the density first, the points as
  SIMD lanes of the ERI kernel) and the charge matrix from the QM/MM
  potential kernel;
* an LU factorisation of the PCM matrix K, kept with pyscf's intermediates;
* the surface matrices S and D, and in the gradient their geometry
  derivatives already contracted with the PCM vectors (no n x n x 3
  arrays), with erf from the engine's Boys function (std's vector erf is
  accurate to only ~1e-8);
* the integral term of the gradient (QM atoms and surface charges) from one
  density-contracted pass of the QM/MM gradient kernel.

`benchmarks/bench_solvent.py` (water, conv_tol 1e-9, density fitting; the
Hartree-Fock case uses `attach` alone, as the native loop does not take
solvent objects):

| system                                             | nao | surface points | cycles | SCF pyscf [s] | mojoscf [s] | x | grad pyscf [s] | mojoscf [s] | x | \|dE\| [Eh] |
|----------------------------------------------------|----:|-----:|------:|------:|-----:|-----:|-----:|-----:|------:|--------:|
| benzene / def2-SVP B3LYP, C-PCM (DF)               | 114 | 1538 |   7/7 |  14.8 |  3.8 | 3.9x |  4.8 |  0.7 |  7.0x | 1.7e-12 |
| ferrocene / def2-SVP PBE, IEF-PCM (DF)             | 221 | 2404 | 23/23 | 159.2 | 28.3 | 5.6x | 21.9 |  3.3 |  6.7x | 1.1e-09 |
| [Cu(NH3)4]2+ doublet / def2-SVP B3LYP, SMD (DF, UKS) | 147 | 4182 | 12/12 |  92.6 | 15.4 | 6.0x | 33.4 |  2.7 | 12.5x | 7.3e-12 |
| (H2O)5 / def2-TZVP HF, IEF-PCM (DF; `attach` only) | 215 | 2191 | 10/10 |  43.4 |  8.8 | 4.9x | 14.1 |  2.9 |  4.9x | 4.5e-13 |

On the same SCF solution energies and gradients agree with pyscf to about
1e-13 for all five models (`tests/test_solvent.py`; the ferrocene runs above
converge independently to energies 1e-9 Eh apart).  In pyscf the solvent
terms dominate these runs (ferrocene: 98 s of the SCF in `int3c2e`, 18 s in
`einsum`, 5 s for the derivative arrays of the gradient); with the kernels
the potential at the points and the charge matrix cost 0.14 s and 0.25 s
per cycle for ferrocene's 2404 points.

Analytical Hessians of solvated molecules (`mf.Hessian()` of an accelerated
object) take the native Hessian paths with the solvent terms from these and
further kernels: see the solvent item of the Analytical Hessians section.

### Excited states (TDDFT, TDA)

pyscf's linear-response solvers (`pyscf.tdscf`) need, in every Davidson
iteration, the product of the response matrices with a batch of trial
vectors.  pyscf forms it in the AO basis: every vector becomes a transition
density `C_v z^T C_o^T`, which goes through `get_jk` and the XC kernel, and
the result is projected back.  With density fitting the transition densities
are not symmetric, so `get_jk` cannot use its occupied-orbital shortcut and
contracts the whole AO tensor for each of them; in ferrocene/def2-SVP PBE0
TDA that took 330 of 680 s, and the XC kernel (`nr_rks_fxc`, a Python loop
over blocks per density) most of the rest.  `mojoscf.tdscf`, whose classes
`mf.TDA()`, `mf.TDDFT()` and `mf.CasidaTDDFT()` of accelerated objects create
(or explicitly, for any RHF/UHF/RKS/UKS object), keeps pyscf's solvers,
initial guesses and analysis and replaces the operator:

* **Coulomb and exact exchange in the occupied-virtual space.**  The DF
  tensor is transformed once to `(ov|Q)`, `(oo|Q)` and `(vv|Q)`
  (`kernels.df_mo`); then `J = sum_Q (ia|Q) rho_Q`, the A-type exchange
  `sum_Q (oo|Q) x (vv|Q)` and the B-type `sum_Q (ov|Q) y^T (ov|Q)` of all
  trial vectors come from one pass over Q with two GEMMs per auxiliary
  function in worker threads (`kernels.df_sandwich`); range-separated
  hybrids add the same with the long-range tensor.
* **XC kernel in one fused pass.**  `xc_fxc_core` evaluates a block's basis
  functions once, forms the response densities from the factors `C_v z^T`
  and `C_o`, contracts them with the kernel on the spot and returns `V C_o`
  rather than V.  With `A = (C_v z^T)^T phi` per vector and
  `B_c = C_o^T d_c phi` shared by all vectors, the density is
  `sum_k A_k B_0k` and its gradient `sum_i d_c phi_i X_i + sum_k A_k B_ck`
  with `X = (C_v z^T) B_0`: two GEMMs of nocc x nrow per vector (meta-GGA:
  the four `A_c`, whose products also give tau) where a dense density needs
  one of nrow x nrow.  The potential is `phi U + Z B_0^T` with
  `U = sum_c w_c B_c^T`, two more, so the nrow x nrow GEMMs and the nao x nao
  matrices disappear.  The `B_c^T` are kept as transposed copies so that
  every product is row-major `A B`: OpenBLAS runs `A B^T` of these shapes in
  its blocked path, at 45-55 GFlop/s per thread here, but `A B` of up to 1e6
  multiply-adds in small-matrix kernels without packing, at 75-85 GFlop/s,
  and larger per-block products are issued as blocks of rows (or of K)
  within that size (`Blas.gemm_small`; with OpenBLAS 0.3.17 or later on the
  SkylakeX-family cores that have those kernels).  For ferrocene/def2-SVP
  PBE0 a trial vector costs 0.088 s in a batch of 40 (0.32 s alone), against
  0.157 s (0.37 s) with three density GEMMs per vector and `A B^T`
  potential products.  `cache_xc_kernel`, `nr_rks_fxc`, `nr_rks_fxc_st` and
  `nr_uks_fxc` of `mojoscf.dft.NumInt` run on the same pass (CPHF,
  polarisabilities, ...).
* **Exact integrals** (pyscf's default, no density fitting) go the same
  way.  ERIs pyscf keeps in core are transformed once to `(ia|jb)`,
  `(ij|ab)` and `(ib|ja)`, each an (ov) x (ov) matrix, after which every
  product is a GEMM for all trial vectors.  Integral-direct references (or
  MO matrices beyond `max_memory`) get J and K of the AO transition
  densities of all vectors from one call of the Mojo kernels; as the
  transition densities are not symmetric, the kernels take their symmetric
  parts with J and K and their antisymmetric parts with K only, through the
  same 8-fold digestion (K = A - A^T for those), in core or
  integral-direct.  Range-separated hybrids add the same with the
  long-range four-centre integrals of the engine.
* **The eigensolver's BLAS.**  pyscf's TDDFT solver alternates
  `numpy.linalg` and `scipy.linalg` calls on its subspace matrices (several
  hundred rows by the end of a run): `lu`, `inv`, `cholesky`, `eigh` in every
  iteration.  NumPy and SciPy each bundle an OpenBLAS whose threads keep
  spinning for about 0.1 s after a call, so each call waited for the other
  pool (52 ms for an `inv` + `cholesky` pair of 400 x 400 that takes 9 ms
  otherwise: 13 of the 61 s of a benzene TDDFT run).  The TD classes and
  the stability analysis below run with SciPy's OpenBLAS on one thread
  (`_backend.serial_scipy_blas`).
* **Excited-state gradients** (`td.nuc_grad_method()`, `mojoscf.tdgrad`):
  pyscf's `tdrhf`/`tdrks`/`tduks` gradient driver.  pyscf contracts the
  derivative J/K matrices of four densities (one of them antisymmetric)
  with the same densities; that sum is the nuclear derivative of one
  two-electron energy of density pairs, which here comes from a single
  eight-fold-symmetric pass over the derivative integrals
  (`grad2e_pairs_core`, a second pass per long-range exchange term)
  without forming the matrices.  The XC-kernel contractions
  (`_contract_xc_kernel`: second and third functional derivatives with the
  transition and relaxed densities on the grid) run in Mojo passes and the
  Z-vector equations through the Mojo J/K and XC kernels (meta-GGA
  included).  pyscf has no density-fitted TDDFT gradients.

`benchmarks/bench_tddft.py`: pyscf versus `mojoscf.dft.accelerate`, each in
its own process, density fitting, pyscf's default grids, conv_tol 1e-9 for
the SCF and pyscf's default 1e-5 for the excited states, on the 2.1 GHz
machine of the SCF tables (re-measured after the changes to the XC kernels,
XC item above):

| system                                              | nao | states | TD pyscf [s] | mojoscf [s] | x | SCF pyscf [s] | mojoscf [s] | max \|dE\| [eV] |
|-----------------------------------------------------|----:|---:|-------:|------:|------:|------:|-----:|--------:|
| benzene / def2-SVP B3LYP, TDDFT                     | 114 | 10 |   84.7 |   7.5 | 11.3x |   7.8 |  1.2 | 1.5e-11 |
| ferrocene / def2-SVP PBE0, TDA                      | 221 | 10 |  773.9 |  52.2 | 14.8x | 243.0 | 23.0 | 3.0e-05 |
| ferrocene / def2-SVP PBE0, TDDFT                    | 221 | 10 | 1201.5 | 108.4 | 11.1x | 178.5 | 21.4 | 3.2e-05 |
| ferrocene / def2-SVP PBE, TDDFT (Casida)            | 221 | 10 |  197.2 |  17.3 | 11.4x |  53.0 |  8.8 | 4.2e-05 |
| ferrocene / def2-SVP CAM-B3LYP, TDA                 | 221 | 10 | 1037.7 |  65.0 | 16.0x | 245.7 | 21.3 | 4.6e-05 |
| [Cu(NH3)4]2+ doublet / def2-TZVP B3LYP, TDA (UKS)   | 241 |  8 |  703.8 |  38.5 | 18.3x |  51.5 |  8.3 | 2.3e-11 |

The ferrocene excitation energies differ by up to 5e-5 eV because the two
independently converged SCFs do (conv_tol 1e-9; their orbital-energy gaps
differ by about 1e-4 eV, the densities by 2e-5); on the same SCF solution
the ferrocene operator products agree to 1e-13.  The operator products agree
with pyscf's to about 1e-14 (relative) for
RHF/UHF/RKS/UKS, TDA, full TDDFT and the Casida form, singlets and triplets,
global and range-separated hybrids and meta-GGAs, with and without frozen
orbitals, density fitting and exact integrals (in-core MO matrices, AO
densities with in-core or integral-direct J/K; `tests/test_tdscf.py`), and
the excitation energies to 1e-12 Hartree; the excited-state gradients agree
with pyscf's on the same TD solution to 1e-11 (`tests/test_tdgrad.py`).
`wfnsym` restrictions, short-range-only hybrids and DF tensors that do not
fit in `max_memory` run pyscf's operator (Kohn-Sham objects still with the
Mojo XC kernels).

**Solvent models.**  `mf.TDA()`/`mf.TDDFT()` of a solvated object (`mf.PCM()`,
`mf.SMD()`, ...) is pyscf's `TDSCFWithSolvent`: the gas-phase response plus
the solvent's `_B_dot_x` of the transition density, by default from a copy
of the solvent with the optical dielectric constant 1.78 (non-equilibrium
solvation; `equilibrium_solvation=True` keeps the ground state's).  The
accelerated objects build that wrapper around the mojoscf TD classes, whose
operator adds the solvent's response wherever the Coulomb term enters: to
the symmetric part of the transition densities of all trial vectors at
once (x + y, none for triplets), with the Mojo PCM kernels, projected on
both blocks.  The products agree with pyscf's to about 1e-14 (relative) for
TDA, full TDDFT and the Casida form, RKS singlets and triplets, UKS,
RHF/UHF, PCM and SMD, equilibrium and non-equilibrium solvation, DF or
exact integrals (`tests/test_tdscf.py`).  `bench_tddft.py` in water (same
machine and settings as the table above):

| system                                                   | nao | states | TD pyscf [s] | mojoscf [s] | x | SCF pyscf [s] | mojoscf [s] | max \|dE\| [eV] |
|----------------------------------------------------------|----:|---:|-------:|------:|-----:|------:|-----:|--------:|
| ferrocene / def2-SVP PBE0, TDA, C-PCM                    | 221 | 10 |  856.3 |  52.4 | 16.3x | 210.3 | 18.7 | 2.6e-05 |
| ferrocene / def2-SVP PBE0, TDDFT, SMD                    | 221 | 10 | 1917.1 | 120.3 | 15.9x | 304.2 | 22.4 | 2.1e-07 |
| [Cu(NH3)4]2+ doublet / def2-SVP B3LYP, TDA (UKS), C-PCM  | 147 |  8 |  265.1 |  17.9 | 14.8x |  57.0 |  5.5 | 3.2e-11 |

pyscf evaluates the solvent's response to every trial vector from
three-centre integrals over the surface (`int3c2e`); here the potential at
the surface points and the potential matrices of all vectors of an
iteration come from one pass each of the Mojo PCM kernels (13 s for the
2404 surface points of the ferrocene TDA; the rest is the gas-phase
response).  The rows were re-measured after the changes to the XC kernels
(XC item above), against the same pyscf runs.  The ferrocene energies
differ by up to 3e-5 eV for the reason given above (independently converged
SCFs and Davidson iterations).

With exact integrals (pyscf's default) and the nuclear gradient of the first
excited state (`td.nuc_grad_method()`, `bench_tddft.py --grad`), on the
2.1 GHz machine of the SCF tables:

| system                                              | nao | states | TD pyscf [s] | mojoscf [s] | x | gradient pyscf [s] | mojoscf [s] | x | SCF pyscf [s] | mojoscf [s] | max \|dE\| [eV] |
|-----------------------------------------------------|----:|---:|------:|-----:|-----:|------:|-----:|-----:|------:|-----:|--------:|
| benzene / def2-SVP B3LYP, TDDFT                     | 114 | 10 |  64.3 |  9.0 | 7.2x |  13.8 |  3.4 | 4.1x |   7.0 |  2.2 | 5.5e-12 |
| benzene / def2-SVP CAM-B3LYP, TDA                   | 114 | 10 |  53.5 |  6.2 | 8.6x |  22.2 |  5.0 | 4.5x |  11.2 |  3.0 | 3.3e-12 |
| ferrocene / def2-SVP PBE0, TDA                      | 221 | 10 | 536.4 | 78.7 | 6.8x | 101.9 | 22.6 | 4.5x | 109.5 | 31.0 | 1.0e-04 |
| [Cu(NH3)4]2+ doublet / def2-SVP B3LYP, TDA (UKS)    | 147 |  8 | 183.3 | 26.2 | 7.0x |  58.5 | 14.9 | 3.9x |  23.4 |  6.5 | 6.5e-12 |

The gradients agree with pyscf's to 5e-10 Eh/bohr for benzene and the Cu(II)
complex.  The first excited state of (eclipsed, D5h) ferrocene is one of a
degenerate pair (2.32 eV, as are the next ones), so its gradient depends on
the combination of the pair each Davidson run ends with: the two codes'
gradients differ by up to 9e-4 Eh/bohr here, while on the same TD solution
they agree to 1e-11 (`tests/test_tdgrad.py`).

### SCF stability analysis

`mf.stability()` (`pyscf.scf.stability`) looks for the lowest eigenvalues of
the orbital Hessian with Davidson's method; pyscf forms one Hessian-vector
product per call, each an AO first-order density through `gen_response`.
These Hessians are the linear-response matrices of the previous section:
internal stability of real orbitals needs `(A + B) x` (the TDDFT top block
with y = x), RHF -> UHF the triplet `A + B`, real -> complex `A - B` without
Coulomb and XC terms.  `mojoscf.stability`, which `mf.stability()` of the
mojoscf classes runs, takes them from the `mojoscf.tdscf` operators (DF
tensors, MO-basis ERIs or integral-direct J/K, the projected XC kernel)
for all vectors of an iteration at once (`lib.davidson1` with a batched
operator in place of pyscf's per-vector `lib.davidson`), with pyscf's Fock
blocks, preconditioner, thresholds and orbital rotation; the products agree
with pyscf's to 1e-9 (relative) for RHF/RKS/UHF/UKS with global and
range-separated hybrids, DF or exact (`tests/test_stability.py`).  With a
solvent model the analysis lets the solvent respond, as pyscf's
`_Solvation.stability` does (equilibrium solvation unless the solvent is
frozen): its `_B_dot_x` of the first-order densities enters the products
of the internal and the RHF -> UHF analyses, the latter with the Fock
matrix and the solvent term of pyscf's own external operator.  The
starting space adds unit vectors on the lowest diagonal elements to pyscf's
single vector, and Davidson follows as many roots as there are starting
vectors (the lowest `nroots` are reported).  From pyscf's vector alone
Davidson reaches the symmetry sectors it does not span only through
rounding noise: for the octahedral [Fe(H2O)6]2+ quintet (UKS B3LYP) pyscf's
analysis stopped at the second root (0.0294) in one of two runs on the same
SCF solution, the lowest being 0.0148.  Following only the lowest `nroots`
Ritz pairs, on the other hand, lets the pairs of the unit vectors converge
to higher roots before pyscf's vector has developed the lowest one: for the
[Cu(NH3)4]2+ doublet an earlier version reported 0.1989 where pyscf finds
0.1876.

`benchmarks/bench_stability.py`: pyscf's default analysis (internal; for
benzene also the external ones) of the converged SCF of each code, pyscf's
default grids, conv_tol 1e-9, 4 cores of the 2.1 GHz Xeon.

| system                                        | ints  | nao | stability pyscf [s] | mojoscf [s] | x | SCF pyscf [s] | mojoscf [s] | lowest eigenvalue (pyscf / mojoscf) |
|-----------------------------------------------|-------|----:|------:|-----:|------:|------:|-----:|-----------------------------------|
| benzene / def2-SVP B3LYP (RKS), with external | exact | 114 |  80.1 |  9.8 |  8.2x |   7.3 |  2.6 | internal 0.777848 / 0.777837, real -> complex 0.19589 / 0.19589, RHF -> UHF 0.092476 / 0.092434 |
| ferrocene / def2-SVP PBE0 (RKS)               | DF    | 221 | 400.3 | 16.6 | 24.2x | 232.9 | 35.7 | 0.334362 / 0.334367 |
| [Cu(NH3)4]2+ doublet / def2-SVP B3LYP (UKS)   | DF    | 147 | 213.1 | 10.7 | 19.9x |  35.5 |  7.1 | 0.187447 / 0.187456 |
| [Fe(H2O)6]2+ quintet / def2-SVP B3LYP (UKS)   | DF    | 175 | 349.3 | 27.4 | 12.8x |  85.2 | 17.1 | 0.029346 / 0.014790 |

The eigenvalues agree to the solver's tolerance (1e-4 on the eigenvalue),
except for [Fe(H2O)6]2+, where pyscf's single starting vector converged to
the second root, as described above.

### Second-order SCF

`mf.newton()` (`pyscf.soscf.newton_ah`) minimises the energy with pyscf's
co-iterative augmented-Hessian solver, which applies the orbital Hessian to
one vector per micro iteration (`h_op` of `gen_g_hop`), forming each product
as the stability analysis does: an AO first-order density through
`gen_response` (J/K and the XC kernel in the AO basis) and a projection
back.  The Hessian is the one of the internal stability analysis at the
current orbitals, so `mojoscf.soscf` (installed on pyscf's
`_SecondOrderRHF`, `_SecondOrderUHF` and `_SecondOrderROHF` at import) sets
up the operators of `mojoscf.stability` once per macro iteration for its
orbitals, Fock matrix and density and returns pyscf's gradient, Hessian
products and diagonal: Coulomb and exchange from the MO-basis DF tensors, or
with exact integrals from the J/K of the first-order AO density (with a
handful of products per macro iteration, cheaper than transforming the ERIs
to the MO basis each time), and the XC response from the factors of the
first-order density.  ROHF/ROKS objects take pyscf's route through the UHF
Hessian of the alpha (occupation > 0) and beta (occupation 2) orbitals,
folded onto the ROHF rotations.  Solvated objects (`mf.PCM()`, SMD, ...)
get the products of pyscf's solvated `gen_response`: the gas-phase ones,
plus with `with_solvent.equilibrium_solvation` the solvent's `_B_dot_x` of
the first-order densities (the Mojo PCM kernels when the solvent has them).
The solver, its step control and keyframes and the orbital update stay
pyscf's; the products agree with pyscf's to about 1e-13 (relative) on
orbitals away from convergence for RHF/RKS,
UHF/UKS and ROHF/ROKS with global and range-separated hybrids and
meta-GGAs, DF or exact (`tests/test_soscf.py`).  Each product still needs a
grid pass of the XC kernel for its single vector, which cannot share the
basis-function values with other vectors as a Davidson iteration does; for
one GGA vector the kernel forms the projection with one GEMM instead of the
three that a batch of vectors shares (0.49 -> 0.38 s per product for
[Ni(H2O)6]2+, 223k grid points).  The kernel set-up of every macro
iteration (and of TDDFT, the stability analysis and CPHF) takes the second
derivatives of the functional by the chain rule written out instead of
pyscf's general tensor transform (`eval_xc_eff`: 0.89 -> 0.32 s for a
spin-polarised B3LYP kernel on 223k points, libxc included).  For
restricted objects it uses the kernel of the spin-unpolarised functional,
as pyscf's orbital Hessian does (`gen_response(singlet=None)`), rather than
TDDFT's spin-resolved one.

`benchmarks/bench_soscf.py`: `mf.newton().kernel()` to conv_tol 1e-9 with
pyscf's default grids, 4 cores of a 2.8 GHz Xeon.  Benzene and the Cu(II)
complex start from pyscf's initial guess (minao).  Ferrocene and
[Ni(H2O)6]2+ start as in a geometry optimisation step, from the density of
the converged SCF at a nearby geometry (pyscf's DIIS; the same density for
both codes).

| system | ints | nao | pyscf [s] | mojoscf [s] | x | macro iterations | Hessian products | J/K builds | E [Eh] (difference between the codes) |
|---------------------------------------------------------------|-------|----:|------:|-----:|-----:|---:|---:|---:|----------------------------|
| benzene / def2-SVP B3LYP (RKS)                                 | exact | 114 |  11.9 |  4.5 | 2.7x |  3 | 11 |  21 | -232.0842901184 (3e-13)    |
| [Cu(NH3)4]2+ doublet / def2-SVP B3LYP (UKS)                    | DF    | 147 | 102.6 | 22.8 | 4.5x |  5 | 34 |  52 | -1865.9003816618 (2e-12)   |
| [Cu(NH3)4]2+ doublet / def2-SVP B3LYP (ROKS)                   | DF    | 147 |  86.3 | 22.6 | 3.8x |  5 | 33 |  50 | -1865.8992676570 (7e-12)   |
| ferrocene / def2-SVP PBE0 (RKS), Fe-Cp 1.56 -> 1.66 Å          | DF    | 221 |  50.7 | 19.6 | 2.6x |  3 | 13 |  23 | -1649.6998717659 (5e-13)   |
| [Ni(H2O)6]2+ triplet / def2-SVP B3LYP (UKS), Ni-O 1.95 -> 2.05 Å | DF  | 175 |  40.9 | 12.7 | 3.2x |  2 |  9 |  16 | -1965.9219335802 (2e-12)   |

The macro iterations, Hessian products and J/K builds (pyscf's count:
products and Fock builds) are the same in both codes.  From the minao
guess, the solver takes ferrocene and [Ni(H2O)6]2+ through regions of
negative orbital-Hessian curvature.  There, products that differ only in
the last bits can send it to a different SCF solution:

* ferrocene: pyscf ended at -1649.49418 Eh and mojoscf at the ground state,
  -1649.69987 Eh;
* [Ni(H2O)6]2+: two mojoscf builds whose products agree to 3.5e-16
  (relative) ended at -1965.92193 Eh (pyscf's solution) and at
  -1965.87805 Eh; a start from 8 DIIS cycles ended at -1965.89002 Eh, the
  solution DIIS itself converges to.

For a like-for-like comparison, the table restarts these two from a nearby
geometry, where the solver stays in the ground state's basin.

### Analytical Hessians

pyscf's Kohn-Sham Hessian (`pyscf.hessian.rks`/`uks`, with density fitting
`pyscf.df.hessian`) spends its time in three places: the exchange-correlation
terms (`_get_vxc_diag`, `_get_vxc_deriv2`, `_get_vxc_deriv1`: Python loops
over grid blocks with AO values up to third derivatives and one nao x nao
matrix per atom and direction), the density-fitted Coulomb/exchange
second-derivative terms (`_partial_hess_ejk`, `_gen_jk`: per-atom `einsum`s
over six kinds of libcint derivative integrals, intermediates in temporary
HDF5 files) and the coupled-perturbed equations for the 3N perturbations
(AO first-order densities through `get_jk` and `nr_rks_fxc` in every
iteration).  `mojoscf.hessian`, which `mf.Hessian()` of accelerated Kohn-Sham
objects and of the mojoscf Hartree-Fock classes returns
(`mojoscf.hessian.accelerate(h)` for any RHF/UHF/RKS/UKS Hessian object),
keeps pyscf's driver, nuclear term and CPHF solver and replaces the rest:

* **XC partial Hessian in one pass** (`xc_hess_core`): AO values up to third
  derivatives per block, the second-derivative term per AO row, the
  atom-pair terms as GEMMs over the rows of each atom and the kernel term,
  contracted with the density on the spot: no nao x nao matrices.  LDA, GGA
  and meta-GGA (the kinetic-energy density in the response densities, its
  potential with the third and second AO derivatives).
* **First-order Fock matrices in the MO basis.**  Only `C^T F^(A,x) C_o`
  enters the CPHF and the Hessian; the XC kernel part is projected inside the
  grid pass (`xc_h1_core`) and the Coulomb/exchange part comes from the
  density-fitting pass below, so the 3N nao x nao matrices are never formed.
* **One-electron second derivatives**: the nuclear attraction for all pairs
  of atoms (basis-function and nuclear centres) from one pass of the charge
  kernel of the QM/MM section with the nuclei as point charges, and the
  core potentials (ECP) from one pair of integrals per ECP centre, the
  derivatives with respect to the centre by translational invariance.
  pyscf's `hcore_generator` evaluates integrals with the origin at a
  nucleus for every pair of atoms and the summed core-potential integrals
  once more (cisplatin with the def2 Pt ECP: 3.3 s -> 1.3 s, 278 -> 12
  libcint calls; a 32-atom alkane: 4.5 s -> 0.5 s).
* **Coupled-perturbed equations in the MO basis** with pyscf's solver: the
  operator uses the DF tensor transformed once to the MO basis (Coulomb
  through `(pi|Q)`, exchange as one `nmo x nmo x (nset nocc)` GEMM per
  auxiliary function for all perturbations, `kernels.cphf_k`) and the fused
  XC response kernel of the TDDFT section.
* **Density-fitted Coulomb/exchange second derivatives** (`df_jk_terms`):
  the second-derivative three-centre integrals (21 derivative components per
  shell pair) are contracted with the fitted densities as they are produced
  (`hess_df3c_core`); the first-derivative integrals come in blocks of
  auxiliary functions (`int3c2e_ip1_core`), from which the Coulomb and
  exchange response vectors `dB_a - V_a c` are built per perturbation
  (Mojo kernels with the auxiliary functions in parallel), whitened with
  the Cholesky factor of the metric and contracted with BLAS, the
  auxiliary-centre derivatives following from translational invariance.
  Range-separated hybrids add a long-range pass with every integral of
  erf(omega r)/r.
* **Exact integrals** (pyscf's default) follow the same plan with the
  four-centre derivative integrals of the engine: the Coulomb/exchange
  partial Hessian from the unique shell quartets, the second-derivative
  integrals contracted with the density products inside the kernel and
  never formed (`hess2e_core`, see *Hessian two-electron term* in the
  engine section), the first-order Fock matrices of all 3N
  perturbations from one pass with the first-derivative integrals of all
  four centres digested into derivative J/K matrices (`h1_2e_core`, in
  place of pyscf's per-atom `_get_jk` calls in `make_h1`), and the CPHF
  operator from J/K of the first-order densities of all perturbations in
  one call of the integral-direct kernels (or the in-core ERIs).
  Range-separated hybrids add one long-range pass of each.
* **VV10 functionals** (ωB97M-V, ωB97X-V, B97M-V, `mf.nlc`; RKS, as pyscf
  has no UKS Hessian for them).  pyscf's analytical NLC terms
  (`_get_enlc_deriv2`, `_get_vnlc_deriv1` with its grid response, and
  `get_vnlc_resp` in every CPHF iteration) evaluate double sums over the NLC
  grid in C: per-point sums U, W, A, B, C, E, the kernel applied to the trial
  densities of all perturbations, grid-response and weight-derivative sums.
  Each loops over all ordered pairs of points one at a time with three
  divisions per pair, and the sums over trial densities and nuclear
  displacements evaluate the pair terms again for every six trials or every
  displacement.  `mojoscf.nlchess` answers these calls with Mojo kernels
  (pyscf's `libdft` in `pyscf.hessian.rks` is wrapped at import): the
  symmetric sums take each unordered pair once, the trial and displacement
  sums are GEMMs of tiles of pair coefficients (one division per four
  vectors of pairs) with the trial vectors, and the grid-response sums run
  over the points of one atom at a time (all four agree with pyscf's C to
  1e-12).  In the accelerated Hessian the CPHF response is native
  (`nlchess.Response`: the densities and potential matrices of all
  first-order densities from the XC kernels, the ground-state sums once)
  and the Coulomb/exchange and semi-local XC terms take the native paths
  above; the NLC energy and Fock derivatives are pyscf's functions on the
  Mojo sums.  Same-SCF Hessians agree with pyscf to 2e-12 (exact integrals),
  1e-11 with density fitting, and 1e-8 for ωB97M-V with density fitting (the
  long-range metric, 7e-9 without the VV10 term).
* **Solvent models** (PCM family, SMD; `pyscf.solvent.hessian.pcm`, RKS and
  UKS).  pyscf's solvent Hessian adds the solvent's response to the
  first-order densities in every CPHF iteration (their potential at the
  surface points, the PCM equations, the potential matrices of the induced
  charges, one `int3c2e` pass per set of charges), the solvent term of the
  first-order Fock matrices (`analytical_grad_vmat`: an `int3c2e` pass per
  atom and direction for the relaxed charges) and the solvent's second
  derivatives: integral terms (`analytical_hess_qv`: `int3c2e_ipip1`,
  `ipvip1`, `ip1ip2`, `ipip2` and a `grad_qv` per atom and direction) and
  surface terms (`analytical_hess_solver`, `get_dqsym_dx`: the first and
  second geometry derivatives of the surface matrices S and D as dense
  (n, n, 3) and (3, 3, n, n) arrays, of the switching function as
  (natm, natm, 3, 3, n) arrays, and a fresh solve with K in each
  contraction).  With the solvent kernels of `mojoscf.solvent` the Hessian
  keeps the native J/K, XC and CPHF paths and adds the solvent response to
  the total first-order density in the MO basis (pyscf's
  `vind_with_solvent`); the potential matrices of all charge sets come from
  one pass of the charge-lane kernel, the potential derivatives at the
  surface for every atom from one table (for `get_dvgrids` and all the
  `grad_qv` calls), and the second derivatives over basis-function and
  charge centres from one pass with the 21-component pair table contracted
  with the density (`_mojo/qmmm.mojo`).  The surface terms keep pyscf's
  assembly with stand-ins for its dense arrays whose contractions are
  evaluated on the fly (`_mojo/pcm.mojo`; the switching function per block
  of points over the atoms it depends on) and the cached LU of K, so their
  memory grows as n natm instead of n^2 (n surface points: 9 n^2 words for
  each of d2S and d2D in pyscf).  Same-SCF Hessians of water agree with
  pyscf to 2e-12 (exact integrals) and 2e-11 (density fitting; C-PCM,
  IEF-PCM, SMD, RKS and UKS), the individual terms to 1e-11 or better.

`benchmarks/bench_hessian.py`: pyscf versus `mojoscf.dft.accelerate`, each in
its own process, density fitting (auxiliary-basis response included, pyscf's
default), pyscf's default grids, SCF conv_tol 1e-10, on the 4-core machine of
the TDDFT table:

| system                                      | nao | Hessian pyscf [s] | mojoscf [s] | x | SCF pyscf [s] | mojoscf [s] | max \|dH\| | max \|dfreq\| [cm^-1] |
|---------------------------------------------|----:|-------:|------:|-----:|------:|-----:|--------:|-----:|
| benzene / def2-SVP B3LYP                    | 114 |  143.6 |   8.5 | 17.0x |   8.2 |  1.4 | 1.4e-08 | 0.00 |
| ferrocene / def2-SVP PBE0                   | 221 | 1525.5 |  71.6 | 21.3x | 238.5 | 32.2 | 4.8e-06 | 0.01 |
| ferrocene / def2-SVP PBE                    | 221 |  896.7 |  46.1 | 19.5x |  51.9 |  7.7 | 6.7e-06 | 0.01 |
| ferrocene / def2-SVP CAM-B3LYP              | 221 | 2054.1 | 123.0 | 16.7x | 119.7 | 22.8 | 1.6e-06 | 0.01 |
| [CuCl4]2- doublet / def2-SVP B3LYP (UKS)    | 103 |  130.5 |  10.4 | 12.6x |  19.1 |  1.8 | 2.3e-08 | 0.00 |
| ethylene / def2-SVP ωB97M-V                 |  48 | 1537.4 |  74.8 | 20.5x |  78.7 |  5.6 | 2.8e-09 | 0.00 |
| [ZnCl4]2- / def2-SVP ωB97M-V                | 103 | 3293.3 | 188.4 | 17.5x | 190.2 | 13.4 | 1.1e-07 | 0.00 |
| Ni(CO)4 / def2-SVP B3LYP, C-PCM (water)     | 143 |  485.0 |  15.1 | 32.1x |  54.7 |  3.6 | 2.0e-07 | 0.00 |
| [CuCl4]2- doublet / def2-SVP B3LYP, SMD (water, UKS) | 103 | 278.4 | 13.6 | 20.5x | 32.8 | 2.3 | 2.4e-08 | 0.00 |

The rows were re-measured after the changes to the XC kernels (XC kernel
item of the TDDFT section), each mojoscf run against pyscf's on the same
machine (pyscf's own times vary by up to a third between runs on this
virtual machine).  In the solvent rows the solvent terms take about 5 s of mojoscf's Ni(CO)4
Hessian (1460 surface points: the solvent response in the CPHF iterations
2.4 s, the solvent Fock derivatives 1.5 s, the solvent's second derivatives
under 1 s); the rest is the Kohn-Sham Hessian of the solute.

In the ωB97M-V rows (VV10 on pyscf's default level-3 NLC grid, 67k points
for ethylene) most of mojoscf's time is the kernel sums over trial densities
(`f_t`, 8 n_trial flops per ordered pair of grid points, as GEMMs: in every
CPHF iteration and for the displacement terms), then pyscf's NumPy code of
the NLC Fock derivatives; pyscf spends most of its time in the scalar C
pair loops.

With exact integrals (pyscf's default), on the 2.1 GHz machine of the SCF
tables:

| system                                      | nao | Hessian pyscf [s] | mojoscf [s] | x | SCF pyscf [s] | mojoscf [s] | max \|dH\| | max \|dfreq\| [cm^-1] |
|---------------------------------------------|----:|-------:|------:|-----:|------:|-----:|--------:|-----:|
| benzene / def2-SVP B3LYP                    | 114 |  125.1 |   6.9 | 18.2x |   7.2 |  1.4 | 8.6e-12 | 0.00 |
| benzene / def2-SVP CAM-B3LYP                | 114 |  187.2 |  10.6 | 17.6x |  14.1 |  1.9 | 9.9e-12 | 0.00 |
| [CuCl4]2- doublet / def2-SVP B3LYP (UKS)    | 103 |  101.4 |   7.5 | 13.5x |  14.0 |  1.7 | 4.1e-09 | 0.00 |

max |dH| is in Eh/bohr^2.  The ferrocene Hessians differ by up to 7e-6
because the two independently converged SCFs do (conv_tol 1e-10); on the same SCF
solution the ferrocene PBE0 Hessians agree to 1.6e-7 (the coupled-perturbed
equations of both are converged to pyscf's tolerance, `conv_tol_cpscf` per
atom), and the harmonic frequencies agree to 0.04 cm^-1 or better in all
cases.  On the same SCF
solution the Coulomb/exchange partial Hessian terms agree with pyscf's to
1e-11 (relative) for water and 1e-9 for an open-shell Cu(II) complex, the
projected Fock derivatives to 1e-13 and whole Hessians to 1e-8 or better,
also for a Pd complex with def2 effective core potentials
(`tests/test_hessian.py`); range-separated functionals to about 1e-9 (their
long-range metric is numerically singular).  In the ferrocene PBE0 run the
largest part of mojoscf's time is the XC response in the CPHF iterations,
which, like the other XC kernels, is bound by the throughput of its
per-block GEMMs.  Stacking the response vectors of a block into larger
GEMMs makes it slower: products beyond 1e6 multiply-adds leave OpenBLAS's
small-matrix kernels (XC kernel item of the TDDFT section).

`auxbasis_response < 2`, objects with a `get_jk` of their own and
short-range operators with omega < 0 run pyscf's code for the
Coulomb/exchange part.

### QM/MM

`benchmarks/bench_qmmm.py` embeds a QM solute in a sphere of TIP3P
point-charge waters (`pyscf.qmmm.mm_charge`; liquid density, waters within
2.6 Å of a QM atom left out) and times the SCF, then the gradient on the QM
atoms (`nuc_grad_method().kernel()`) followed by the forces on the MM charges
(`grad_hcore_mm(dm) + grad_nuc_mm()`), the two together being the forces of
one QM/MM MD step: pyscf versus the same object accelerated with
`mojoscf.accelerate`, each in its own process, 4 cores of a 2.1 GHz Xeon,
conv_tol 1e-9.

| system                                          | nao | MM charges | SCF pyscf [s] | mojoscf [s] | x | grad + MM forces pyscf [s] | (MM forces) | mojoscf [s] | x |
|-------------------------------------------------|----:|-----:|-----:|-----:|-----:|------:|------:|-----:|------:|
| benzene / cc-pVDZ, 15 Å (DF)                    | 114 | 1353 |  3.9 |  0.7 | 5.8x |  2.16 |  0.69 | 0.35 |  6.2x |
| benzene / cc-pVDZ, 25 Å (DF)                    | 114 | 6486 |  4.2 |  0.7 | 5.9x |  5.54 |  3.23 | 0.41 | 13.5x |
| benzene / cc-pVDZ, 25 Å, Gaussian charges (DF)  | 114 | 6486 |  3.9 |  0.7 | 5.5x |  6.80 |  3.24 | 0.45 | 15.0x |
| (H2O)5 / aug-cc-pVDZ, 20 Å (in-core)            | 205 | 3366 |  9.3 |  3.4 | 2.7x | 22.84 |  5.46 | 4.12 |  5.5x |
| C8H18 / 6-31G*, 20 Å (direct)                   | 148 | 3330 | 15.9 |  3.6 | 4.4x | 14.30 |  3.51 | 1.99 |  7.2x |
| ferrocene / def2-SVP, 25 Å (DF)                 | 221 | 6456 | 22.5 |  7.7 | 2.9x | 23.31 | 11.98 | 2.26 | 10.3x |
| (H2O)10+ / cc-pVDZ, 20 Å (DF, UHF)              | 240 | 3330 | 30.3 | 10.1 | 3.0x | 20.45 |  6.63 | 2.28 |  9.0x |

Energies agree to 4e-11 Eh or better, QM gradients to 9e-12 and MM forces
to 5e-14 Eh/Bohr.  pyscf builds one integral matrix per block of 200
charges (`int1e_grids` for the Hamiltonian, `int1e_grids_ip` and
`int3c2e_ip2` with charges of exponent 1e16 for the gradients) and
contracts it with NumPy; its MM forces alone ("(MM forces)") often cost
more than the QM gradient.  `_mojo/qmmm.mojo` treats each charge as the
s-type ket of the batched ERI kernel instead (a point charge as exponent
1e30, exact to double precision; Gaussian charges with their own exponent),
with up to 64 charges as the SIMD lanes against one AO shell pair, and
contracts on the fly.  The SCF needs the potential matrix, one pass.  The
gradient needs only density-contracted derivatives: the derivative Hermite
matrices of each shell pair (nabla a, nabla b) are contracted with the
density before the charge loop, so the kernel transforms six components
instead of 6 n_a n_b, and a single pass gives the charge term of the
QM-atom gradient and, by translational invariance, the force on every
charge; `grad_hcore_mm(dm)` returns those forces when called for the
gradient's density, geometry and charges.  For benzene with 6486 charges
the potential takes 0.12 s and the gradient pass 0.25 s.

**Kohn-Sham and other methods.**  QM/MM is mostly run with DFT.
`mojoscf.dft.accelerate(mf)` on a `pyscf.qmmm` Kohn-Sham object adds the
Mojo MM-charge terms to the other Kohn-Sham kernels (see *Kohn-Sham DFT*
above); for any other method (ROHF, ...) `mojoscf.qmmm.mm_charge(method,
coords, charges)` (or `mojoscf.qmmm.attach(mf)` on an existing `pyscf.qmmm`
object) gives only the MM-charge terms.  Same benchmark, Kohn-Sham with
`mojoscf.dft.accelerate`:

| system                                         | nao | MM charges | SCF pyscf [s] | mojoscf [s] | x | grad + MM forces pyscf [s] | (MM forces) | mojoscf [s] | x |
|------------------------------------------------|----:|-----:|-----:|-----:|-----:|------:|------:|-----:|------:|
| benzene / def2-SVP B3LYP, 25 Å (DF)            | 114 | 6486 |  9.7 |  2.8 | 3.5x |  6.66 |  2.85 | 0.58 | 11.5x |
| benzene+ / def2-SVP B3LYP, 25 Å (DF, UKS)      | 114 | 6486 | 37.4 |  9.4 | 4.0x |  8.42 |  3.08 | 0.63 | 13.3x |
| ferrocene / def2-SVP PBE, 25 Å (DF)            | 221 | 6456 | 67.9 | 14.4 | 4.7x | 26.53 | 12.15 | 1.93 | 13.8x |

Energies agree to 1e-12 Eh, gradients to 3e-10 and MM forces to 5e-12
Eh/Bohr (independently converged SCFs).  With only the MM-charge terms
(`attach`, the previous version of this table) the gradient step was 1.9 to
2.5x faster and the SCF unchanged.

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
| `mm_grad_terms(mol, dm, coords, w, zetas=None)` | the charge term of the QM-atom gradient and `grad_hcore_mm(dm)` together, from one pass |

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
  together: prefactors, the Boys function (each lane's table row loaded as
  whole vectors and transposed 8 x 8 in registers, branch-free blend with the
  asymptotic form) and the Hermite recursion run as 8-wide vectors, the
  transforms keep eight independent FMA chains in registers and take R as an
  embedded broadcast operand, and all lanes of a batch accumulate before one
  store.  Single primitive quartets use a scalar version of the same code
  (or, in the drivers, the batches below).  Vector operands are 64-byte
  aligned (cache-line splits had cost 1.45x).  The table rows were first read
  with hardware gathers, which cost about 20 cycles each on Xeons with the
  GDS ("Downfall") microcode mitigation: two thirds of an (ss|ss) lane-kernel call
  (74 ns per eight primitive quartets, 25 ns with the transposes) and 20% of
  a (pp|pp) one.
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
  and flush full batches.  In the direct J/K, kets with one primitive pair
  and two shells of l_c + l_d <= 3 (2l + 1 functions each; 71% of the
  batched kets of ferrocene / def2-SVP) are queued by (l_c, l_d) instead and
  stay one ket per lane to the end: `(o|kl) = sum_h U[o][h] E[h][kl]` and the
  six J/K contractions are vector FMAs over the kets, with the density
  elements of each lane gathered before and its sums added to the matrices
  after (`lanes_digest1`; half the cost per ket of transforming and
  digesting the kets one at a time).  The other quartets are digested by
  `digest_cd`, unrolled over the ket's functions at compile time.
  `lanes_preferred` keeps a quartet on the single-quartet path where
  per-class timings showed that to be faster (kets with five or more
  primitive pairs against a less contracted bra, contracted kets against
  bras of Hermite degree above 2).
* **Gradient two-electron term** (`grad2e_core`).  Each unique quartet needs
  the derivatives at three centres (the fourth from translational
  invariance).  They are taken on the ket side, the less contracted pair:
  (ab|nabla c d) and (ab|c nabla d) from a six-component ket table, nabla a
  from a three-component bra table, and the bra pass is dropped when a and b
  sit on the same atom (only their sum is needed).  Quartets whose ket is
  the higher pair use the mirror split (nabla a, nabla b on the bra, nabla c
  unless c and d share an atom).  Kets up to degree 3 are batched as SIMD
  lanes like the direct J/K's; for the six-component ket pass the integrals
  are never formed: `V[h] = sum_ij U[ij][h] G[ij]` is contracted with the
  ket's Hermite matrices directly, a sixth of the products of transforming
  first.  This replaced one six-component bra pass and one three-component
  ket pass per quartet on the single-quartet path, where single-primitive
  quartets ran in scalar code.
* **Hessian two-electron term** (`hess2e_core`).  A quartet's second
  derivatives at its four centres have 78 components (21 within each pair,
  36 between them); translational invariance leaves 45, those in a, c and
  d: the 21 components of the ket pair, nabla a times nabla c and nabla d
  (18) and nabla a nabla a (6), and the blocks of b follow.  A pair whose
  shells sit on one atom needs only their sum, so one pass with the other
  pair's 21 components gives every block, and quartets on one atom are
  skipped.  Each pass has its derivatives on one pair (or one first
  derivative on each) and runs through `eri_fused_raw`: G is first
  contracted with the other pair's Hermite matrices (a small GEMM), and the
  kernel contracts each derivative component with the result instead of
  transforming it.  Profiling then showed the passes bound by memory: the
  21-component Hermite matrices of a contracted d-p pair take 1.7 MB and
  were read for every quartet.  What a pass adds to the Hessian depends on
  the quartet only through the derivative pair and the atoms of the other
  pair's centres it involves (b for the ket pass, a and b for the mixed
  one, none for nabla a nabla a), so the quartets of a derivative pair that
  share them are summed in one kernel call, the other pairs' primitive
  pairs joined as its inner side, which also fills the SIMD lanes (quartets
  here average 5-10 primitive quartets).  The ket and mixed passes run over
  blocks of four kets, so that each bra's matrices are read once per block.
  From the minao density, 4 cores of a 2.8 GHz Xeon: Ni(CO)4 / def2-SVP
  14.5 -> 2.8 s, [Fe(H2O)6]2+ / def2-SVP 31.2 -> 6.5 s (invariance alone
  gave 7.5 and 16.1 s, the fused contraction then 5.3 s for Ni(CO)4).
* **In-core J/K of many densities** (`jk_s8_core`).  With the 8-fold
  packed integrals, the J/K of several densities looped over the densities
  inside every row segment of the integrals: short dot products and AXPYs,
  each with its overheads, and the K rows of all densities streamed through
  the cache per row.  From three densities on (CPHF: three per atom) they
  are interleaved as the SIMD lanes, D[a][b][s] for eight densities s at a
  time, so that every integral updates all of them with vector FMAs, and
  each row of the integrals goes to all groups of eight (up to 64
  densities) while it is in cache (a pass per group read the 0.9 GB of
  [Fe(H2O)6]2+ / def2-SVP 15 times for its 114 CPHF densities).  J
  takes eight consecutive rows side by side (each element of D + D^T and J'
  loaded once per eight rows; per row, as first written, J was bound by that
  traffic); K takes the segments of four consecutive k of a row together,
  so that K'_il and K'_jl receive their four contributions in one update and
  the D_jl, D_il loads serve all four (single-thread, in cache: 9-10 cycles
  per integral for eight densities one segment at a time, 6-7 grouped).
  27 densities of Ni(CO)4 / def2-SVP (143 AOs): 0.50 -> 0.12 s, 114 of
  [Fe(H2O)6]2+ (175 AOs) 0.9 s, pyscf's `dot_eri_dm` 1.17 s for the 27;
  one or two densities keep the previous code.
* **First-order Fock matrices of Hessians** (`h1_2e_core`).  The
  derivative J/K matrices of every displacement took, per unique quartet,
  a six-component pass (nabla a, nabla b) and a three-component one
  (nabla c) through `eri_quartet`, and twelve digestions of the blocks
  (four centres, three directions); the three-component pass cost as much
  as the other, single-primitive kets running in scalar code.  The passes
  now follow `grad2e_core`'s driver: bras by primitive pairs, nabla a with
  the ket's six components or the mirror split, a single pass when a pair
  sits on one atom (only the sum of its derivatives is needed), and the
  bra's kets of degree <= 3 as SIMD lanes of `eri_batch`.  The blocks are
  summed per atom and folded together (`digest_h1`): the density elements
  are read once for all of them, and the atom of the missing centre
  receives minus the others' sums (translational invariance), so its block
  is never formed: 3 (k - 1) blocks for k atoms, 6.6 on average for
  Ni(CO)4, instead of 12.  One J and one K density from the minao guess,
  4 cores of a 2.3 GHz Xeon: Ni(CO)4 / def2-SVP 1.61 -> 1.15 s,
  [Fe(H2O)6]2+ / def2-SVP 3.94 -> 2.86 s, Fe-porphine / def2-SVP (37
  atoms, 427 AOs) 84.2 -> 60.8 s.  Two variants of the digestion did not
  pay off: one single-primitive ket per SIMD lane (gathering the lanes'
  Hermite matrices cost more than the vector contractions saved), and the
  three directions of an atom as one vector (on medium blocks the scalar
  code already runs at about two FMAs per cycle).
* **Integral-direct J/K of many densities** (`DirectJK.jk`).  The direct
  J/K evaluates every quartet once for all densities but digested its
  block once per density: 114 CPHF densities of [Fe(H2O)6]2+ / def2-SVP
  took 11.7 s against 0.44 s for one, every CPHF iteration of a Hessian
  whose ERIs do not fit in core paying for 100+ digestions per quartet.
  From three densities on they are now SIMD lanes, as in the in-core J/K:
  groups of eight interleaved as D[a][b][s], each integral broadcast into
  vector FMAs for the eight (`digest_lanes`), and the kets of one primitive
  pair batched with the others.  A bra's integral blocks are collected and
  digested group by group (`QuartetBuffer`), so that its rows of one
  group's arrays stay in cache for all its kets (digesting every group per
  quartet missed the caches on rows megabytes apart: 20% slower); the
  Coulomb ket-side rows, different for every quartet, still miss, so J
  costs more than K.  A pass over the integrals takes as many groups as
  keep the workers' accumulators within 1.5 GB.  4 cores of a 2.3 GHz
  Xeon: 114 densities of [Fe(H2O)6]2+ 11.7 -> 2.7 s, 11 densities 1.04 ->
  0.67 s; Fe-porphine / def2-SVP (427 AOs) 8 densities 13.6 -> 8.4 s, 111
  in 53.5 s (about 1.1 s per density before).  The UHF Hessian of
  [Fe(H2O)6]2+ with integral-direct J/K (ERIs not in core) takes 28.7 s
  instead of 81.4 s, its CPHF J/K 16.0 s instead of 68.8 s; J of the spin
  sums only (`direct_jk_sets`: separate J and K densities in one pass)
  then 25.4 s and 12.4 s.  The exact-integral TD operator of UKS uses the
  same pair of calls.
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
it comes from pyscf); a range-separated `mol.omega` on the whole molecule
uses libcint (the long-range integrals of range-separated functionals come
from the engine, see *Kohn-Sham DFT*), and
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
  `ECPscalar_iprinv`) come from pyscf; for spin-free X2C objects the
  derivative of the X2C Hamiltonian is contracted with the density in one
  backward pass (`mojoscf.x2c`, see *Transition-metal complexes*), and
  short-range-only operators (omega < 0) use pyscf's
  gradient code (the long-range exchange of range-separated functionals is
  native, with or without density fitting).
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

# ROHF, point-group symmetry, solvent models, newton(): pyscf's loop with the Mojo kernels
mf = mojoscf.accelerate(scf.ROHF(mol).density_fit())
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
de_mm = g.grad_hcore_mm(mf.make_rdm1()) + g.grad_nuc_mm()   # forces on the MM charges (from the same pass)
from pyscf import dft                           # any other method (ROHF, ...): only the MM-charge terms
mf = mojoscf.qmmm.mm_charge(scf.ROHF(mol), mm_coords, mm_charges)

# Implicit solvation (pyscf.solvent PCM family, SMD): the surface-charge terms from the Mojo kernels
mf = mojoscf.dft.accelerate(dft.RKS(mol, xc="pbe").density_fit().PCM())   # Kohn-Sham: everything
mf = mojoscf.solvent.attach(scf.RHF(mol).PCM())          # any SCF object: the solvent terms

# Kohn-Sham DFT: XC integration, J/K, eigensolver, DIIS and gradients from mojoscf, pyscf's SCF loop
mf = mojoscf.dft.accelerate(dft.RKS(mol, xc="b3lyp").density_fit())   # also UKS, without DF, QM/MM
mf.kernel()
g = mf.nuc_grad_method().kernel()               # mojoscf.grad.DFKSGradients
mf = dft.RKS(mol, xc="pbe"); mf._numint = mojoscf.dft.NumInt()   # only the XC integration

# Excited states (pyscf.tdscf): the response built in the occupied-virtual space
mf = mojoscf.dft.accelerate(dft.RKS(mol, xc="pbe0").density_fit()).run()   # or exact integrals
td = mf.TDA(); td.nstates = 10; td.kernel()     # also mf.TDDFT(), mf.CasidaTDDFT(), UKS
td = mojoscf.tdscf.TDDFT(scf.RHF(mol).density_fit().run())   # TDHF of any RHF/UHF object
mf = mojoscf.dft.accelerate(dft.RKS(mol, xc="camb3lyp")).run()
td = mf.TDDFT().run()
de = td.nuc_grad_method().kernel(state=1)        # excited-state gradient (mojoscf.tdgrad)

# SCF stability analysis (pyscf.scf.stability) with the same response operators
mo_i, mo_e, stable_i, stable_e = mf.stability(external=True, return_status=True)

# Second-order SCF (pyscf.soscf.newton_ah): the orbital Hessian from the same operators (also ROKS, HF)
mf = mojoscf.dft.accelerate(dft.UKS(mol, xc="b3lyp").density_fit()).newton().run()

# Analytical Hessians (pyscf.hessian): XC, Coulomb/exchange (DF or exact) and CPHF terms from the Mojo kernels
hess = mf.Hessian().kernel()                    # (natm, natm, 3, 3), pyscf's driver and conventions
hess = mojoscf.RHF(mol).run().Hessian().kernel()   # Hartree-Fock too (mf.TDA(), mf.TDHF() likewise)

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
`MOJOSCF_MOJO=/path/to/mojo`, `MOJOSCF_OPENBLAS_TIMEOUT=n` (spin-wait of
NumPy's and SciPy's OpenBLAS threads, 2^n cycles, default 16; `0` keeps
OpenBLAS's own, see below).

### BLAS/LAPACK backend

GEMM and the symmetric eigensolvers (`dsygvd`/`dsyevd`) are called through
`dlopen`, so no link-time dependency exists.  Two libraries are used:

* matrices smaller than `MOJOSCF_THREADED_MIN` (default 200 orbitals): the
  OpenBLAS bundled with pyscf wheels, a *sequential* build.  Inside an SCF of a
  small molecule a threaded BLAS competes with pyscf's OpenMP integral code for
  the cores and costs up to 40% of the run time.
* larger matrices: NumPy's bundled OpenBLAS (`scipy_openblas64`: ILP64,
  threaded, symbols `scipy_dgemm_64_`; SciPy's LP64 copy if NumPy has none).
  Measured on 4 cores, the glue (diagonalisation, DIIS error vector,
  density) is 1.6x faster than pyscf's at 300 orbitals and 2.7x at 1000 with the
  threaded library, but 1.4 to 2.3x *slower* with the sequential one.
  NumPy's copy rather than SciPy's because OpenBLAS worker threads keep
  spinning for about 0.13 s (2^28 cycles) after every threaded call: pyscf's
  own NumPy products run on NumPy's pool, and with a second pool for mojoscf
  the two sets of spinning threads competed for the cores.  Inside a
  Kohn-Sham SCF (pyscf's loop around mojoscf's kernels) `eigh` took 140 ms
  instead of 5 ms that way, and the whole ferrocene/def2-SVP SCF ran 25%
  longer.
* the density-fitting exchange build uses both: Mojo worker threads each call
  the threaded library pinned to one thread for their per-auxiliary-function
  GEMM (as pyscf's C transform does with OpenMP), and the threaded library
  closes every block with one `dsyrk` rank-k update.  The other Mojo kernels
  that run GEMMs in worker threads (DF gradients, XC integration) do the same
  (`MOJOSCF_SEQ_BLAS=1`: the sequential library instead).
* the spin-wait itself: everything that runs in parallel right after a
  threaded BLAS call shares the cores with the spinning threads: pyscf's
  OpenMP code (`lib.dot` splits small products over threads, libcint, the
  FCI solver) and the worker threads of the Mojo kernels.  In pyscf's
  DF-CASSCF a 5 x 175 x 25 `lib.dot` took 0.4 ms instead of 25 us.  When
  mojoscf is imported it restarts the thread pools of NumPy's and SciPy's
  bundled OpenBLAS with a spin of 2^16 cycles (`OPENBLAS_THREAD_TIMEOUT`,
  which OpenBLAS reads when a pool starts; `MOJOSCF_OPENBLAS_TIMEOUT=n` for
  2^n, `0` to keep OpenBLAS's setting; a timeout set in the environment is
  left alone).  That is long enough for the back-to-back calls inside LAPACK
  (`numpy.linalg.eigh` of 400 x 400 kept its 15 ms; with the shortest
  spin, 2^4 cycles, it took 27 ms) and returns the cores to the code that
  follows.  Measured on 4 cores: ferrocene/def2-SVP SCF (DF) 3.7 s -> 2.6 s,
  direct 8.7 s -> 7.2 s, [Fe(H2O)6]2+ UKS/B3LYP 15.4 s -> 13.8 s, a
  state-averaged DF-CASSCF of [Fe(H2O)6]2+ 29 s -> 20 s (before its
  MO-basis orbital Hessian).  pyscf without mojoscf gains from the same
  setting: its ROHF SCF of [Fe(H2O)6]2+ went from 24.1 s to 12.0 s and the
  DF-CASSCF from 110 s to 88 s with `OPENBLAS_THREAD_TIMEOUT=4`.  The
  benchmark tables compare with pyscf as installed (the DF-CASSCF table also
  with that setting); the tables measured before mojoscf set it (all but
  DF-CASSCF) are conservative for mojoscf.

Both the LP64 (`dgemm_`) and the ILP64 interface with suffixed symbols
(`dgemm_64_`) are recognised when a library is opened.
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
| J/K, density fitting (`df_jk.get_jk`)       | Python loop over blocks, C transform, NumPy matmul | Mojo (`_mojo/dfjk.mojo`): streaming J passes, per-Q GEMMs in worker threads (one BLAS thread each), `dsyrk` updates; a tensor on disk memory-mapped or streamed block by block |
| J/K, in-core 8-fold ERIs (`_vhf.incore`)    | C (`libcvhf`, OpenMP)     | Mojo (`_mojo/erijk.mojo`), 1.6-1.8x faster; three or more densities (CPHF, response) as SIMD lanes, 10x for 27 |
| J/K, direct SCF (integrals every cycle)     | C (libcint + `libcvhf`)   | Mojo (`_mojo/directjk.mojo`): Mojo integrals, libcvhf's screening, incremental build |
| J/K outside the SCF loop (`mf.get_jk`/`get_j`/`get_k`, hence `get_veff`, `get_fock`: CASSCF core Fock, `pyscf.prop` response equations, `newton()`) | as in the loop, pyscf's | the same Mojo kernels as in the loop (DF tensor, in-core ERIs, integral-direct) for the objects mojoscf accelerates (also ROHF/ROKS and symmetry-adapted ones); a `get_jk` of a subclass is kept |
| two-electron integrals (3-index DF tensor in memory or on disk, 4-index ERIs), once | C (libcint) | Mojo engine (`_mojo/integrals.mojo`); libcint for unsupported molecules |
| one-electron integrals (`get_hcore`, `get_ovlp`) | C (libcint)          | unchanged (`attach(mf)` uses the Mojo engine) |
| nuclear gradients (`nuc_grad_method().kernel()`), exact or DF | C (libcint derivative integrals, `libcvhf` J/K, `libao2mo`) + NumPy/SciPy | Mojo derivative integrals and contractions (`_mojo/gradients.mojo`, `int1e_ip_core`); DF metric solves in SciPy; terms assembled as in pyscf |
| Kohn-Sham XC (`NumInt.nr_rks`/`nr_uks`, LDA/GGA/meta-GGA), with `mojoscf.dft.accelerate` | Python loop over blocks: C AO values, NumPy/C GEMMs, libxc per block | Mojo (`_mojo/numint.mojo`): two passes over the grid (densities; potential matrix) around one libxc call; screened shells per block of 128 points, SIMD AO values, per-block GEMMs; pyscf's grids and libxc |
| Kohn-Sham gradients (`grad.rks`/`uks`, DF or not) | pyscf `get_vxc` (C AO second derivatives, NumPy contractions) + J/K derivative matrices | Mojo: XC term contracted with the density inside one pass over AO second derivatives; Coulomb/exact-exchange term from `grad2e`/`grad2e_df` |
| Kohn-Sham grids (`Grids.build`: Becke partition `get_partition`, `arg_group_grids`) and the grid response of the gradients (`grid_response = True`) | C loop over all atom pairs at every point (`VXCgen_grid`; a NumPy loop for Stratmann's scheme), `numpy.unique` on the box indices; for the response per atom's grid the Becke weight derivatives as (natm, natm, 3) NumPy arrays per point and AO second derivatives per block | Mojo (`_mojo/grids.mojo`): from 40 atoms only the cell functions near each point (about ten), SIMD over the atoms, below that the pair loop SIMD over the points; grouping by one sorted integer key; for the response the weight derivatives contracted with the energy density point by point and the orbital terms from the XC gradient kernel |
| spin-free X2C one-electron gradient term (`x2c.sfx2c1e_grad.hcore_grad_generator`) | NumPy: per atom and coordinate the first-order X2C response in the uncontracted basis (2n x 2n eigenvector response, `X1`, `R1`, `h_FW1`) | NumPy (`mojoscf/x2c.py`): the density taken backwards through that chain once, the effective densities contracted with the derivative integrals of all atoms (nuclear-attraction operator term in one Mojo pass) |
| VV10 non-local correlation (`numint._vv10nlc` in `nr_nlc_vxc` and the NLC gradient) | C double loop over all ordered pairs of grid points (OpenMP), scalar; densities and potential by the Python block loop | Mojo (`_mojo/nlc.mojo`): SIMD over cached tiles, each unordered pair once (symmetric denominators), one division per four vectors of pairs; densities, potential matrix and gradient contraction from the XC kernels |
| VV10 terms of the analytical Hessian (`pyscf.hessian.rks`: `_get_enlc_deriv2`, `_get_vnlc_deriv1`, `get_vnlc_resp` in every CPHF iteration) | C double loops over all ordered pairs of NLC grid points, three divisions per pair, the pair terms evaluated again for every six trial densities (`f_t`) or every grid-weight derivative | Mojo (`_mojo/nlc.mojo`, `mojoscf.nlchess`): the symmetric sums once per unordered pair, the trial and weight-derivative sums as GEMMs of tiles of pair coefficients, the grid-response sums per atom; the CPHF response from the XC kernels with the ground-state sums once |
| XC response kernels (`cache_xc_kernel`, `nr_rks_fxc`, `nr_uks_fxc`: TDDFT, CPHF) | Python loop over blocks per density: C AO values, NumPy GEMMs and `einsum` | Mojo (`xc_fxc_core`): one fused pass, response densities, kernel contraction and potential per block; for transition densities from their occupied-virtual factors |
| Analytical Hessians (`pyscf.hessian.rks`/`uks`, DF or not): XC second-derivative terms, first-order Fock matrices, CPHF | Python loops over grid blocks (AO values up to third derivatives, one nao x nao matrix per atom and direction), per-atom `einsum`s over libcint derivative integrals with HDF5 intermediates (DF), per-atom `_get_jk` calls over libcint derivative integrals (exact), AO first-order densities through `get_jk` and `nr_rks_fxc` per CPHF iteration | Mojo (`xc_hess_core`, `xc_h1_core`, `hess_df3c_core`, `int3c2e_ip1_core`, `hess_df_ip1_k_core`, `hess_dfk_rk_core`, `cphf_k_core`; exact: `hess2e_core`, `h1_2e_core`) + BLAS: XC terms contracted inside one grid pass, Fock derivatives and CPHF in the MO basis, second-derivative integrals contracted as produced, DF response vectors whitened with the metric's Cholesky factor, exact first-order Fock matrices of all atoms from one 8-fold pass; pyscf's driver and CPHF solver |
| TDA/TDDFT/TDHF response (`pyscf.tdscf` `gen_vind`), DF | AO transition densities through `get_jk` (DF with non-symmetric densities: Python loop, `einsum`) and the XC kernel; with a solvent model `_B_dot_x` per call (`int3c2e` over the surface) | Mojo (`_mojo/dfmo.mojo`, `mojoscf.tdscf`): MO-basis DF tensors once, Coulomb and exchange of all trial vectors in the occupied-virtual space (per-Q GEMMs), XC response projected in the kernel, the solvent response of all vectors from the Mojo PCM kernels; pyscf's Davidson solvers |
| TDA/TDDFT/TDHF response, exact integrals | AO transition densities through `get_jk` with `hermi=0` (`_vhf.incore` or libcint + `libcvhf` direct) and the XC kernel | in-core ERIs transformed once to (ov) x (ov) matrices (GEMMs per iteration), or J/K of all transition densities in one integral-direct pass (antisymmetric parts K only); long-range exchange from erf-attenuated Mojo integrals |
| excited-state gradients (`pyscf.grad.tdrks`/`tduks`/`tdrhf`), exact integrals | derivative J/K of four densities (libcint `int2e_ip1`, `libcvhf`), `_contract_xc_kernel` (Python loops, AO second/third derivatives), Z-vector through `gen_response` | Mojo: the two-electron term as one 8-fold pass over the derivative integrals contracted with density pairs (`grad2e_pairs_core`, also long-range), the XC-kernel contractions in grid passes, Z-vector through the Mojo J/K and XC kernels; pyscf's driver |
| SCF stability analysis (`mf.stability()`) | orbital Hessian one vector per Davidson call, AO densities through `gen_response` | the TDDFT operators above for all vectors of an iteration (`lib.davidson1`); pyscf's Fock blocks, preconditioner and rotation |
| Second-order SCF (`mf.newton()`): orbital Hessian products (`gen_g_hop`) | one AO first-order density per product through `gen_response` (J/K and the XC kernel in the AO basis) | the operators of the stability analysis, set up once per macro iteration for its orbitals and density (MO-basis DF tensors, or with exact integrals the Mojo J/K of the first-order AO density; the XC response from the density factors; with a solvent model its `_B_dot_x`); pyscf's solver, step control and orbital update |
| DF-CASSCF integrals and orbital Hessian (`mcscf.df._ERIS`, `update_jk_in_ah`) | per macro iteration (Q|pq) for all MO pairs written to a temporary HDF5 file and read back for `ppaa`/`papa`; two AO J/K builds per orbital-Hessian step (rank 2 ncore densities, `df_jk.get_jk`) | Mojo transform with the core terms in the same pass (`casscf_mo`: `j_pc`, `k_pc`, the core potential with a rank-ncore SYRK per auxiliary function), (Q|pq) kept in memory for the macro iteration (its memory reused by the next one within a `kernel`) for `ppaa`, `papa`; the orbital-Hessian J/K only for the rows pyscf uses, in the MO basis in one pass (`casscf_ah`: the core-noncore structure of the rotation, the exchange rows from one GEMM per four auxiliary functions); pyscf's driver, FCI solver and orbital optimiser |
| DF-NEVPT2 integrals (`mrpt.dfnevpt2._ERIS`) and its Sijrs, Srsi, Srs subspaces | four `with_df.ao2mo` transforms (the full (Q|pq) for `ppaa`), (cv|cv) as an (ncore nvir)^2 matrix in memory or on disk; three-index `einsum` contractions | two partial Mojo transforms ((Q|up), (Q|cv)); (pq|uv) from the AO matrices sum_Q E_Q (Q|uv), one per active pair; Sijrs from the DF factors one core orbital at a time (no (cv|cv) matrix; pairs j <= i only); the Srsi and Srs contractions as GEMMs |
| FCI 3-particle density matrix (`fci.rdm.make_dm123`) and NEVPT2's 4-particle contractions (`mrpt.nevpt2._contract4pdm`) | C kernels over the FCI strings for p >= r >= t (4-particle: one free index of (pq|rs) contracted with a 4-particle matrix) | Mojo: the vectors E_tu|ket>, E_qp|bra> over all determinants, one GEMM per single excitation of each string for the p >= r >= t part (pyscf's `_complete_dm3_` completes it); the 4-particle contractions as the same matrices with the third vector sum_x E_ex W_fx|ci> (W_fx = sum_yz (fx|yz) E_yz) |
| DF-CASSCF nuclear gradient (`mc.nuc_grad_method()`, state-specific) | the DF gradient J/K of the core and active densities (libcint three-centre derivative integrals, auxiliary response) and the active 2-RDM terms (`solve_df_rdm2`, `grad_elec_dferi`, auxiliary response) in separate passes | one pass of the Mojo DF gradient kernels: in the basis of the core and active natural orbitals the Coulomb, exchange and 2-RDM terms of E2(Dc + Da) - E2(Da) + E_aa form one matrix per auxiliary function and the metric terms one W; Mojo one-electron derivative integrals; pyscf's generalized Fock matrix |
| SA-CASSCF nuclear gradient (`mc.nuc_grad_method().kernel(state=i)` after `mc.state_average_`): the Lagrange terms (`df.grad.sacasscf.Lorb_dot_dgorb_dx`, `Lci_dot_dgci_dx`) | per term a DF gradient J/K of four (orbital term) or two (CI term) densities and one or two further passes for the active 2-RDM terms (`solve_df_rdm2`, `grad_elec_dferi`), each with its auxiliary response | both terms in one pass of the Mojo DF gradient kernels (`integrals.grad2e_df_terms`): Coulomb, exchange and 2-RDM terms with one rotated orbital in the basis [C_core, C_act, C_core L, C_act L] (orbital term) and with the symmetrised transition densities in the same basis (CI term), one J/K build for their densities; pyscf's generalized Fock matrices and response solver |
| MC-PDFT (`pyscf.mcpdft`): on-top energies, effective potentials, Hellmann-Feynman gradient | Python loops over grid blocks: AO values, spin, core and active densities (`make_rho`), the active orbitals on the grid and the on-top pair density by NumPy products and `tensordot` with the cumulant, libxc per block, the grid rebuilt for every state; for the gradient per atom grid AO second derivatives and NumPy contractions, the Becke weight derivatives as (natm, natm, 3) arrays per point, a DF gradient J/K (K unused) for the Coulomb term | Mojo (`_mojo/pdft.mojo`, `_mojo/grids.mojo`): spin densities and on-top pair densities of all states in one pass, the functional once per state on the whole grid (pyscf's `eval_ot`), the grid reused; the potentials with the XC potential-matrix kernel and one pass for the paaa block; the gradient's orbital terms with the XC gradient kernel (kernel v_rho + v_Pi rho/2) and one pass for the cumulant term, grid-point terms from the same per-atom sums, Becke weight derivatives contracted with the energy density point by point, the Coulomb term from the Mojo DF derivative integrals |
| PCM/SMD solvation (`pyscf.solvent`): potential at the surface points, surface-charge matrix, S/D matrices, gradient | C (libcint `int3c2e`, `int3c2e_ip1/ip2` with the surface fakemol) + NumPy (einsum, (3, n, n) derivative arrays, two dense solves per cycle) | Mojo (`_mojo/qmmm.mojo`, `_mojo/pcm.mojo`): density-contracted potential pass, charge-lane potential matrix, S/D and their contracted derivatives (Boys-function erf); LU factorisation of K kept per build |
| PCM/SMD Hessian (`pyscf.solvent.hessian.pcm`): solvent response in the CPHF, solvent Fock derivatives, `analytical_hess_qv`, `analytical_hess_solver` | C (libcint `int3c2e` per set of charges and per atom and direction, `int3c2e_ip1/ip2/ipip1/ipvip1/ip1ip2/ipip2`) + NumPy (dense (n, n, 3), (3, 3, n, n) and (natm, natm, 3, 3, n) surface derivative arrays, a dense solve with K per contraction), AO first-order densities | Mojo (`_mojo/qmmm.mojo`: potential matrices of many charge sets in one pass, per-atom potential derivatives at the surface, the second-derivative pass; `_mojo/pcm.mojo`: S/D first and second derivatives contracted on the fly) with pyscf's assembly, the cached LU of K, the solvent response in the MO-basis CPHF |
| QM/MM charges (`pyscf.qmmm`): potential, its derivative, forces on the MM charges | C (libcint `int1e_grids`, `int1e_grids_ip`, `int3c2e_ip2`, one integral matrix per block of 200 charges) + NumPy | Mojo (`_mojo/qmmm.mojo`): one pass over the shell pairs with the charges as SIMD lanes, contracted on the fly (the gradient pass with density-contracted Hermite matrices gives the QM-atom term and all charge forces at once); nucleus-charge terms NumPy as in pyscf |

Source layout:

```
mojoscf/
  _mojo/linalg.mojo    vector kernels, GEMM/eigensolver dispatch, native fallbacks
  _mojo/kernels.mojo   make_rdm1, get_occ, get_grad, damping, level_shift, DIIS errvec, dense J/K
  _mojo/dfjk.mojo      density-fitted J/K from pyscf's (naux, npair) tensor; density factorisation
  _mojo/dfmo.mojo      DF tensor in an orbital basis; exchange contractions of the linear-response operators
  _mojo/erijk.mojo     J/K from 8-fold packed ERIs
  _mojo/diis.mojo      pyscf-compatible CDIIS bookkeeping and extrapolation
  _mojo/driver.mojo    the RHF/UHF SCF loop (port of pyscf.scf.hf.kernel) with native J/K modes
  _mojo/integrals.mojo Gaussian integral engine (Boys function, Hermite recursions, S/T/V, ERIs, 3c2e/2c2e)
  _mojo/directjk.mojo  integral-direct J/K (screening, 8-fold digestion) for direct SCF; derivative J/K matrices (gradients, Hessian first-order Fock)
  _mojo/gradients.mojo two-electron gradient and exact Hessian terms, exact and density-fitted
  _mojo/qmmm.mojo      potential of MM point/Gaussian charges, its derivatives, forces on the charges; PCM Hessian integral terms
  _mojo/numint.mojo    XC integration on pyscf's grids: AO values and derivatives, densities, XC matrices, XC gradient, response kernels
  _mojo/nlc.mojo       VV10 non-local correlation: the pair sums over the NLC grid (energy, potential, Hessian terms)
  _mojo/grids.mojo     Becke partition of pyscf's grids and its nuclear derivatives (weight response)
  _mojo/pdft.mojo      MC-PDFT: on-top pair density and its potential and gradient terms
  _mojo/fcirdm.mojo    FCI 1-, 2- and 3-particle density matrices; NEVPT2's 4-particle contractions
  _mojo/pcm.mojo       PCM surface matrices S, D and their contracted first and second geometry derivatives
  _mojo/__init__.mojo  Python bindings (module mojoscf._mojoscf)
  _backend.py          build/load the extension, discover BLAS/LAPACK
  kernels.py           NumPy-facing wrappers
  integrals.py         pyscf-compatible integral functions and attach()
  diis.py              CDIIS drop-in class
  scf.py               RHF/UHF classes, kernel(), accelerate()
  grad.py              RHF/UHF and RKS/UKS nuclear gradient classes, exact and DF (nuc_grad_method)
  x2c.py               spin-free X2C: the gradient of the one-electron Hamiltonian contracted with densities
  dft.py               Kohn-Sham: NumInt (XC integration), XC and VV10 gradient terms, accelerate() for pyscf RKS/UKS objects
  nlchess.py           the VV10 pair sums of pyscf's analytical Hessian and the native VV10 CPHF response
  casscf.py            DF-CASSCF integrals, orbital Hessian, gradients and DF-NEVPT2 integrals (pyscf.mcscf, pyscf.mrpt)
  mcpdft.py            MC-PDFT (pyscf.mcpdft) on the Mojo XC kernels
  fcirdm.py            FCI 3-particle density matrices and NEVPT2's 4-particle contractions
  solvent.py           PCM/SMD (pyscf.solvent) with the Mojo kernels: attach(); SCF, gradient and Hessian terms
  tdscf.py             TDA/TDDFT/TDHF (pyscf.tdscf) with the response in the occupied-virtual space
  tdgrad.py            excited-state gradients (pyscf.grad.tdrhf/tdrks/tduks) with the Mojo kernels
  stability.py         SCF stability analysis (pyscf.scf.stability) with the occupied-virtual response
  soscf.py             second-order SCF (pyscf.soscf.newton_ah) with the occupied-virtual orbital Hessian
  hessian.py           analytical Hessians (pyscf.hessian): XC terms, DF and exact Coulomb/exchange terms, MO-basis CPHF
  cphf.py              pyscf's CPHF solvers (scf.cphf, scf.ucphf) on a Krylov solver in matrix products
  qmmm.py              QM/MM (pyscf.qmmm) hooks: MM-charge Hamiltonian and gradient terms from the engine
  guess.py             broken-symmetry start densities (HOMO/LUMO mix, AFM atoms, spin flip)
tests/                 kernels vs NumPy/pyscf references; full SCF vs pyscf; integrals vs libcint; gradients vs pyscf
benchmarks/            bench_scf.py, bench_kernels.py, bench_large.py, bench_bs.py, bench_integrals.py, bench_grad.py, bench_metals.py, bench_qmmm.py, bench_dft.py, bench_solvent.py, bench_tddft.py, bench_hessian.py, bench_stability.py, bench_soscf.py, bench_casscf.py, bench_nevpt2.py
tools/gen_eri_kernel.py  generates the register-blocked ERI kernels (single quartet, batched kets) in _mojo/integrals.mojo
```

## Scope and limitations

* Closed-shell **RHF** and **UHF** with real orbitals, with or without density
  fitting, X2C or other decorations that only change `get_jk`/`get_hcore`,
  run in the native loop.  Other Hartree-Fock objects (ROHF, point-group
  symmetry, solvent models, second-order SCF, smearing, constrained UHF and
  other overridden glue methods) keep pyscf's SCF loop when accelerated,
  with the Mojo J/K (DF tensor, in-core ERIs or integral-direct), eigensolver
  and DIIS (pyscf's DIIS with point-group symmetry, whose error vectors it
  masks by irrep) and the Mojo gradients, stability analysis and excited
  states where pyscf's plain methods would run; a solvent model's TD
  wrapper and stability set-up are rebuilt around them, other decorations'
  own methods are kept.  `accelerate`
  hands Kohn-Sham objects to `mojoscf.dft.accelerate` and rejects GHF and
  Dirac-HF; `mojoscf.is_supported(mf)` tells whether the native loop runs.
* **Kohn-Sham** RKS/UKS objects get the Mojo kernels with
  `mojoscf.dft.accelerate` but keep pyscf's SCF loop.  The XC integration
  covers LDA, GGA and meta-GGA functionals (not those using the laplacian
  of the density) with symmetric real densities, and their response
  kernels, and VV10 non-local correlation (`nlc`) in the SCF and the
  gradient, also with the response of the grid (`grid_response = True`:
  `dft.grad_xc_grid_response`); with VV10 the grid response keeps pyscf's
  code (with the Mojo VV10 pair sums), as do short-range-only
  (`omega < 0`) exchange and DF objects other than pyscf's plain `df.DF`
  (in memory or on disk).
* **Excited states**: `mojoscf.tdscf` (what `mf.TDA()`, `mf.TDDFT()` and
  `mf.CasidaTDDFT()` of accelerated objects create) builds the response of
  RHF/UHF/RKS/UKS references with in-core density fitting or exact integrals
  in the occupied-virtual space, also with a solvent model (pyscf's
  `TDSCFWithSolvent` around them); point-group restricted states (`wfnsym`),
  short-range-only hybrids, the NLC response, DF tensors that do not fit in
  `max_memory`, ddCOSMO/PolEmbed TD wrappers and objects with a `get_jk` of
  their own keep pyscf's operator (with the Mojo XC response kernels for
  Kohn-Sham).
  Excited-state gradients (exact integrals; pyscf has no DF version) run
  pyscf's driver with the Mojo kernels (`mojoscf.tdgrad`).  ROKS/GHF and
  spin-flip TDDFT stay pyscf's.
* **Stability analysis**: `mf.stability()` of the mojoscf classes runs the
  internal (RHF/RKS, UHF/UKS) and external RHF/RKS analyses with the
  occupied-virtual operators, also with a solvent model; ROHF, GHF, the
  UHF -> GHF analysis and point-group symmetry labels use pyscf's.
* **Second-order SCF**: `newton()` objects of RHF/RKS, UHF/UKS and ROHF/ROKS
  objects accelerated by mojoscf take the orbital Hessian from
  `mojoscf.soscf`, also with a solvent model.  These keep pyscf's
  `gen_g_hop`: point-group symmetry (whose forbidden rotations pyscf
  masks), ROHF/ROKS with equilibrium solvation, the dual-basis Newton
  solver, GHF, and the cases `mojoscf.tdscf` leaves to pyscf.
* **DF-CASSCF**: pyscf's `mcscf.CASSCF` (and `mc.state_average_`,
  `mc2step`) of a reference whose `with_df` is a `mojoscf.dft.MojoDF` with
  the tensor in memory or memory-mapped gets its integrals from
  `mojoscf.casscf` (installed into `pyscf.mcscf.df` when such a DF object is
  created).  The MO-basis tensor (naux nmo^2 doubles) is kept when it fits
  in `mc.max_memory`; otherwise the integrals are built block by block and
  the orbital-Hessian J/K runs on the DF object.  UCASSCF, `approx_hessian`,
  other DF classes and tensors that cannot be mapped keep pyscf's code; the
  CASCI active-space integrals (`with_df.ao2mo`) stay pyscf's.  pyscf's
  NEVPT2 (`mrpt.NEVPT`) of a CASCI on such a reference gets its integrals
  from `mojoscf.casscf.nevpt2_eris` when they fit in `max_memory`, without
  the (cv|cv) matrix (its Sijrs subspace from the DF factors; Srsi and Srs
  with GEMMs); its other subspaces run pyscf's code (their `einsum` calls on
  BLAS contraction paths).  With mojoscf imported, pyscf's spin-traced
  3-particle density matrix (`fci.rdm.make_dm123`, also behind
  `direct_spin1.make_rdm123`) and NEVPT2's 4-particle contractions
  (`mrpt.nevpt2._contract4pdm`) come from `mojoscf.fcirdm` for real CI
  vectors whose work arrays fit in memory, whatever the integrals.  `mc.nuc_grad_method()` of a state-specific DF-CASSCF on
  such a reference is `mojoscf.casscf.Gradients`; that of a state-averaged
  one stays pyscf's `df.grad.sacasscf.Gradients`, with the state's own term
  from `mojoscf.casscf.Gradients` and both Lagrange terms in one pass from
  `mojoscf.casscf.lagrange_terms` (its `get_LdotJnuc`; the separate
  `Lorb_dot_dgorb_dx`/`Lci_dot_dgci_dx` from `lorb_dot_dgorb_dx` and
  `lci_dot_dgci_dx`; without the auxiliary-basis response,
  `auxbasis_response=False`, pyscf's).  Frozen
  orbitals keep pyscf's code.
* **MC-PDFT**: with mojoscf imported, pyscf's MC-PDFT (`pyscf.mcpdft`:
  `mcpdft.CASSCF`/`CASCI`, state-averaged and multi-state variants) takes
  its on-top energies from `mojoscf.mcpdft` (translated and fully
  translated LDA, GGA and meta-GGA functionals, hybrids) for molecules the
  Mojo basis code handles, whatever the reference; the effective potentials
  with `paaa_only`/`aaaa_only` and the Hellmann-Feynman gradient terms run
  on the Mojo kernels for translated LDA and GGA functionals.  Meta-GGA and
  fully translated gradients, L-PDFT's full potentials (`get_pdft_veff`
  without `paaa_only`), the second-order potentials (`get_pdft_feff`:
  L-PDFT and CMS-PDFT gradients) and the weight derivatives of the LKO grid
  partition keep pyscf's code.  `Grids.build` everywhere groups the points
  with `mojoscf.dft.group_grids` (pyscf's order, without `numpy.unique`) and
  takes the Becke partition from `mojoscf.dft.get_partition` (original and
  Stratmann schemes with or without the Treutler and Becke radii
  adjustments; LKO and other schemes keep pyscf's code).
* The native J/K build covers plain `pyscf.df.DF` objects with the tensor in
  core, the in-core 8-fold ERI path (used when `mol.incore_anyway` or pyscf's
  own memory check allows it) and integral-direct J/K otherwise (pyscf's
  direct SCF, with the same incremental update and `direct_scf_tol`
  screening).  With the Mojo integral engine (the default) energies agree
  with pyscf's to about 1e-12 Eh instead of round-off, because the integrals
  themselves differ at the 1e-14 level.  A DF tensor on disk is used
  memory-mapped where it fits in the available memory.  Range separation,
  `only_dfj`, DF tensors on disk that are streamed and overridden
  `get_jk`/`get_veff` fall back to calling `mf.get_veff` (still with the Mojo
  J/K where pyscf's own `get_jk` would run), as does direct SCF for molecules
  the engine does not support.
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
  nuclear-attraction, four-index and 3-/2-centre Coulomb integrals, the
  first derivatives needed for HF gradients and the second derivatives of
  the four- and three-centre integrals for Hessians (no ECP integrals,
  other second derivatives or multipoles; of the range-separated operators
  the long-range erf(ωr)/r four-, three- and two-centre integrals and their
  derivatives).  Molecules with
  ECPs or finite nuclei use it for everything but the ECP / finite-nucleus
  terms.  `unsupported_reason(mol, two_electron=..., allow_ecp=..., allow_omega=...)` says why
  a molecule is rejected for a given use.
* Nuclear gradients are native for RHF and UHF, for RKS/UKS objects
  accelerated with `mojoscf.dft.accelerate` (LDA, GGA, meta-GGA, global and
  range-separated hybrids), and for ROHF/ROKS (`mojoscf.accelerate`,
  `mojoscf.dft.accelerate`: pyscf's ROHF/ROKS gradients are the UHF/UKS
  ones with the ROHF energy-weighted density, and so are mojoscf's; a
  [Ni(H2O)6]2+ triplet ROKS B3LYP gradient takes 2.8 s instead of pyscf's
  8.1 s), with exact or density-fitted (pyscf's `df.DF`, auxiliary-basis
  response included; the gradient recomputes its integrals, so the tensor
  may be on disk) two-electron integrals.  With ECPs the core-potential
  derivative integrals are pyscf's, evaluated once per ECP centre.
* Analytical Hessians: `mf.Hessian()` of accelerated RKS/UKS objects and of
  the mojoscf RHF/UHF classes keeps pyscf's driver; the XC terms are native
  for LDA, GGA and meta-GGA, the Coulomb/exchange terms and the CPHF
  operator for density-fitted references (pyscf's `df.DF` with the tensor
  in memory or memory-mapped, `auxbasis_response = 2`) and exact integrals (in-core or
  integral-direct), global and range-separated hybrids, and VV10 (RKS; the
  NLC energy and Fock derivatives are pyscf's functions on the Mojo pair
  sums of `mojoscf.nlchess`, the CPHF response native) and PCM/SMD solvent
  models (the solvent terms from the solvent kernels below); objects with
  their own `get_jk` and short-range operators with omega < 0 run pyscf's
  code for the parts they touch.
* QM/MM (`pyscf.qmmm.mm_charge`, point or Gaussian MM charges) runs on the
  native loop with the MM-charge terms of the Hamiltonian and of the
  gradients (QM atoms and MM charges) from the engine, for QM shells up to
  g; Kohn-Sham objects get them with the other Kohn-Sham kernels from
  `mojoscf.dft.accelerate`, other methods (ROHF, ...) only the MM-charge
  terms (`mojoscf.qmmm.mm_charge` / `attach`); the periodic interface (`qmmm.pbc`)
  is not covered.  The charge sums are not screened (all charges interact with all
  shell pairs), as in pyscf.
* Implicit solvation: pyscf's PCM family (C-PCM, COSMO, IEF-PCM, SS(V)PE)
  and SMD get the Mojo kernels with `mojoscf.solvent.attach` (done by
  `mojoscf.dft.accelerate`); the native Hartree-Fock loop does not take
  solvent objects (they override `get_fock`), so HF in solvent runs pyscf's
  loop with the Mojo solvent terms.  Analytical Hessians include the
  solvent terms (the second-derivative integral pass needs shells up to f;
  with g shells `analytical_hess_qv` is pyscf's).  ddCOSMO/ddPCM keep
  pyscf's code.
* The first call in a process starts the Mojo runtime and loads BLAS
  (about 50 ms); time a second run when benchmarking tiny systems.
* BLAS/LAPACK: the LP64 interface and the ILP64 one with `_64_`-suffixed
  symbols (OpenBLAS `INTERFACE64` builds such as NumPy's); integer arguments
  are passed in 64-bit slots, which assumes a little-endian machine.

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

Hard SCF problems with several nearby solutions are sensitive to the
summation order of the parallel kernels as well: an Fe(II) porphine triplet
(UKS B3LYP) converged to two states 9 mEh apart in two runs whose XC
integration had distributed the grid blocks to the threads differently.
The XC kernels therefore assign the blocks round-robin and the
density-fitted exchange gives each thread a fixed range of auxiliary
functions, so these results are bitwise reproducible for a given number of
threads; the integral-direct J/K and the gradient kernels schedule shell
pairs dynamically for load balance (as pyscf's libcvhf does with OpenMP).

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
