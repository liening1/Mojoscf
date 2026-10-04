# mojoscf

**Mojo replacements for the Python glue in pyscf's SCF driver.**

[pyscf](https://pyscf.org) evaluates integrals in C (libcint), but the
self-consistent-field loop that stitches everything together is Python + NumPy:
Fock assembly, damping, DIIS extrapolation, the generalised eigenproblem,
occupations, density matrices, energies, convergence tests and logging, plus
the orchestration of the two-electron (J/K) build.  `mojoscf` re-implements the
whole SCF iteration in [Mojo](https://www.modular.com/mojo): the glue, and the
J/K build itself whenever the integrals are in core (density-fitting tensor or
8-fold packed ERIs).  An iteration then runs without touching Python at all.
pyscf still evaluates the integrals (once) and runs direct-SCF J/K builds when
the integrals do not fit in memory.

* **Drop-in**: `mojoscf.RHF(mol)` and `mojoscf.UHF(mol)` are subclasses of
  `pyscf.scf.hf.RHF` / `pyscf.scf.uhf.UHF`; results (energies, orbitals,
  iteration counts, `scf_summary`) agree with pyscf to round-off because the
  driver is a port of `pyscf.scf.hf.kernel`, not a reimplementation of SCF.
  UHF includes open-shell and broken-symmetry (BS) calculations.
* **Whole loop in Mojo**: one native call runs all iterations.  With density
  fitting or in-core ERIs the Coulomb/exchange matrices are built by Mojo
  kernels too (`mojoscf.kernels.df_jk`, `jk_s8`), so a cycle makes no Python
  call; only direct SCF still calls back into pyscf's C integral code.
* **Individual kernels** are also exposed (`mojoscf.kernels`) and a
  Mojo-backed `CDIIS` class can be dropped into any pyscf SCF object.
* **BLAS/LAPACK** (OpenBLAS bundled with pyscf and SciPy) is called from Mojo
  through `dlopen`, choosing a sequential or a threaded library by matrix
  size; portable Mojo fallbacks (SIMD GEMM, Jacobi eigensolver) keep
  everything working when no library is found.

## Results

Hartree-Fock on 4 cores, pyscf 2.14, Mojo 1.1.0.  Both drivers keep the ERIs
in core for these molecules (pyscf's C contraction versus mojoscf's Mojo
kernel), start from the same initial guess with `conv_tol = 1e-10`; best of 3
runs, including the one-time integral evaluation.

| system             | nao | cycles | pyscf [s] | mojoscf [s] | speed-up | &#124;ΔE&#124; [Eh] |
|--------------------|----:|-------:|----------:|------------:|---------:|--------:|
| H2O / STO-3G       |   7 |    7/7 |     0.037 |       0.035 |    1.1x  | 3e-14 |
| H2O / cc-pVDZ      |  24 |    9/9 |     0.141 |       0.068 |    2.1x  | 1e-13 |
| H2O / cc-pVTZ      |  58 |    9/9 |     0.164 |       0.148 |    1.1x  | 1e-13 |
| benzene / STO-3G   |  36 |    7/7 |     0.201 |       0.184 |    1.1x  | 0 |
| benzene / cc-pVDZ  | 114 |    8/8 |     2.435 |       0.835 |    2.9x  | 5e-13 |
| (H2O)3 / cc-pVDZ   |  72 |  10/10 |     0.607 |       0.285 |    2.1x  | 7e-13 |

(`python benchmarks/bench_scf.py`.)  Below about 0.2 s the totals are dominated
by the integral evaluation (libcint, identical in both) and by timer noise.
Per-cycle logs are identical to pyscf's to all printed digits, including with
damping, level shifting and DIIS damping switched on.

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
4-core machine (`python benchmarks/bench_large.py`), so BLAS thread pools and
memory of one run cannot affect another.  For pyscf the time inside
`mf.get_veff` is listed separately; for mojoscf the J/K build is part of the
native loop (mode 1 = density fitting, 2 = in-core ERIs).

| system | nao | cycles | pyscf [s] | of which get_veff [s] | mojoscf [s] | J/K mode | speed-up | &#124;ΔE&#124; [Eh] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C10H22 / 6-31G* (DF) | 184 | 9/9 | 5.9 | 4.4 | 2.7 | 1 | 2.20x | 0.0e+00 |
| C20H42 / 6-31G (DF) | 264 | 8/8 | 19.3 | 17.7 | 12.4 | 1 | 1.55x | 8.2e-12 |
| (H2O)10 / cc-pVDZ (DF) | 240 | 10/10 | 7.6 | 5.8 | 4.8 | 1 | 1.59x | 1.6e-12 |
| C60 / STO-3G (DF) | 300 | 8/8 | 57.3 | 55.6 | 46.0 | 1 | 1.24x | 1.1e-10 |
| (H2O)20 / cc-pVDZ (DF) | 480 | 10/10 | 47.7 | 45.6 | 41.1 | 1 | 1.16x | 2.1e-11 |
| C20H41 radical / 6-31G (DF, UHF) | 262 | 13/13 | 32.9 | 28.5 | 23.6 | 1 | 1.40x | 1.2e-11 |
| (H2O)10 cation / cc-pVDZ (DF, UHF) | 240 | 19/19 | 22.4 | 16.3 | 8.7 | 1 | 2.56x | 7.7e-12 |
| benzene / cc-pVDZ (in-core) | 114 | 8/8 | 2.8 | 1.4 | 0.9 | 2 | 3.25x | 9.1e-13 |
| benzene cation / cc-pVDZ (UHF) | 114 | 12/12 | 5.7 | 2.2 | 1.0 | 2 | 5.64x | 1.0e-12 |
| (H2O)5 / cc-pVDZ (in-core) | 120 | 10/10 | 3.2 | 1.7 | 0.8 | 2 | 4.09x | 8.0e-13 |

What the numbers mean:

* **In-core ERIs** (non-DF, up to about 250 orbitals with pyscf's default
  memory limit): pyscf's C contraction is replaced by a Mojo kernel that is
  1.6 to 1.8x faster, on top of the glue savings.  For UHF pyscf never uses
  this path (it recomputes the integrals in every cycle), so the in-core
  build is a large win there.
* **Density fitting**: the exchange build is a GEMM-bound operation
  (`sum_Q (Q|mu i)(Q|nu i)`, 2.1e11 flops for 20 waters) that pyscf already
  runs at about 100 GFlop/s on this machine; the Mojo kernel reaches 123
  GFlop/s on the whole J+K build by streaming J and using a `dsyrk` update, a
  1.2x gain per iteration (K alone takes the same 1.6 s in both).  The first
  cycle, whose density has no orbitals, is handled by diagonalising the
  density into weighted orbitals instead of pyscf's O(naux nao^3) path.  The
  one-time 3-index integral build (libcint, about 20 s for 20 waters) is
  unchanged, so the end-to-end gain for the largest DF systems is bounded by
  the integral code: 1.16x for 20 waters, 1.24x for C60, 1.4 to 2.6x when
  more cycles are needed (UHF).
* **What is still C**: libcint evaluates all integrals (one-electron matrices,
  the 3-index DF tensor, the 4-index ERIs) once per molecule, and direct SCF
  (integrals recomputed every cycle because they do not fit in memory) runs
  pyscf's `libcvhf`.  A Mojo integral engine competitive with libcint would be
  a project of its own; on this hardware the remaining large-system cost is
  compute-bound BLAS and integral work, not Python glue.
* The earlier version of this table (v0.2.0) was measured while a leftover
  background benchmark was competing for the CPU, which roughly doubled the
  pyscf reference times; these numbers replace it.

### Broken-symmetry UHF

Starting from identical spin-polarised densities (`mojoscf.guess`), pyscf and
mojoscf follow the same SCF path: same number of cycles, same energies to
1e-13 Eh for the small systems and 1e-11 Eh for the transition-metal dimers.
`E_RHF - E_BS` shows how far the broken-symmetry solution lies below the
restricted one, `<S^2>` how spin-contaminated it is (`python
benchmarks/bench_bs.py [--heavy]`, 4 cores, direct SCF unless marked DF).

| system                              | nao | E_RHF-E_BS [mEh] | <S^2>  | cycles | pyscf [s] | mojoscf [s] | speed-up | &#124;ΔE&#124; [Eh] |
|-------------------------------------|----:|-----------------:|-------:|-------:|----------:|------------:|---------:|-------:|
| H2, R = 2.0 Å / cc-pVDZ             |  10 |             80.9 |  0.904 |    7/7 |      0.02 |        0.01 |    3.8x  | 0 |
| H2, R = 3.0 Å / cc-pVDZ             |  10 |            172.3 |  0.995 |    6/6 |      0.02 |        0.01 |    3.4x  | 4e-16 |
| H10 chain, AFM / 6-31G              |  20 |            298.3 |  3.693 |    8/8 |      0.03 |        0.01 |    2.3x  | 9e-15 |
| H20 chain, AFM / 6-31G              |  40 |            595.3 |  7.266 |    8/8 |      0.05 |        0.03 |    1.7x  | 1e-14 |
| H30 chain, AFM / 6-31G              |  60 |            892.3 | 10.840 |    8/8 |      0.12 |        0.09 |    1.3x  | 3e-14 |
| H40 chain, AFM / 6-31G              |  80 |           1189.3 | 14.413 |    8/8 |      1.03 |        0.32 |    3.2x  | 1e-14 |
| N2, R = 2.2 Å / cc-pVDZ             |  28 |            190.6 |  1.018 |  17/17 |      0.06 |        0.02 |    3.2x  | 1e-13 |
| F2, R = 2.6 Å / cc-pVDZ             |  28 |            304.6 |  1.001 |    8/8 |      0.03 |        0.01 |    2.6x  | 6e-14 |
| twisted C2H4 (90°) / cc-pVDZ        |  48 |            129.1 |  1.035 |  10/10 |      0.09 |        0.06 |    1.5x  | 0 |
| [Cu2Cl6]2-, AFM / def2-SVP (DF)     | 170 |                  |  1.009 |  10/10 |     17.69 |       15.51 |    1.1x  | 3e-11 |
| [Fe2S2(SH)4]2-, AFM / def2-SVP (DF) | 190 |                  |  4.988 |  38/38 |     66.96 |       47.59 |    1.4x  | 9e-13 |

The two metal dimers are antiferromagnetically coupled singlets prepared by
flipping the spin of one metal centre in the converged high-spin density
(`flip_spin_on_atoms`): Cu(II)/Cu(II) (Mulliken spin +0.842 / -0.842, BS 0.18 mEh
above the triplet) and Fe(III)/Fe(III) with five unpaired electrons per iron
(Mulliken spin +3.88 / -4.54, BS 16.5 mEh below the S = 5 state), the latter
needing 38 cycles with a 0.3 Eh level shift.  The N2 row is one of two
broken-symmetry states the mix guess can reach, see the degenerate-shell caveat
below.

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

# Use the individual kernels
from mojoscf import kernels
dm = kernels.make_rdm1(mf.mo_coeff, mf.mo_occ)
w, c = kernels.eigh(fock, s)                    # pyscf's phase convention
diis = mojoscf.CDIIS()                          # pyscf.scf.diis.CDIIS replacement
```

`mojoscf.backend_info()` reports which BLAS library and how many threads are
used.  Environment variables: `MOJOSCF_BLAS=/path/lib.so[:symbol_prefix]`,
`MOJOSCF_NATIVE=1` (pure-Mojo fallbacks), `MOJOSCF_SKIP_BUILD=1`,
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
| J/K, direct SCF (integrals every cycle)     | C (libcint + `libcvhf`)   | unchanged, called once per cycle     |
| integral evaluation (1e, 3-index DF tensor, 4-index ERIs) | C (libcint), once | unchanged                      |

Source layout:

```
mojoscf/
  _mojo/linalg.mojo    vector kernels, GEMM/eigensolver dispatch, native fallbacks
  _mojo/kernels.mojo   make_rdm1, get_occ, get_grad, damping, level_shift, DIIS errvec, dense J/K
  _mojo/dfjk.mojo      density-fitted J/K from pyscf's (naux, npair) tensor; density factorisation
  _mojo/erijk.mojo     J/K from 8-fold packed ERIs
  _mojo/diis.mojo      pyscf-compatible CDIIS bookkeeping and extrapolation
  _mojo/driver.mojo    the RHF/UHF SCF loop (port of pyscf.scf.hf.kernel) with native J/K modes
  _mojo/__init__.mojo  Python bindings (module mojoscf._mojoscf)
  _backend.py          build/load the extension, discover BLAS/LAPACK
  kernels.py           NumPy-facing wrappers
  diis.py              CDIIS drop-in class
  scf.py               RHF/UHF classes, kernel(), accelerate()
  guess.py             broken-symmetry start densities (HOMO/LUMO mix, AFM atoms, spin flip)
tests/                 kernels vs NumPy/pyscf references; full SCF vs pyscf
benchmarks/            bench_scf.py, bench_kernels.py
```

## Scope and limitations

* Closed-shell **RHF** and **UHF** with real orbitals, with or without density
  fitting, X2C or other decorations that only change `get_jk`/`get_hcore`.
  ROHF, GHF, Kohn-Sham DFT, symmetry-adapted and second-order (Newton) SCF
  objects are rejected by `accelerate` and are not provided as classes yet.
* The native J/K build covers plain `pyscf.df.DF` objects with the tensor in
  core and the in-core 8-fold ERI path (used when `mol.incore_anyway` or
  pyscf's own memory check allows it; for UHF this replaces pyscf's direct SCF
  with an in-core build of the same integrals, so energies agree to the
  direct-SCF screening threshold of 1e-13 rather than to round-off).  Range
  separation, `only_dfj`, DF tensors on disk and overridden `get_jk`/`get_veff`
  fall back to calling `mf.get_veff`.  `mf.scf_summary["mojoscf_veff_mode"]`
  reports which path ran (1 = DF, 2 = in-core ERIs, 0 = pyscf callback).
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
closures with explicit capture lists).  The GitHub Actions workflow in
`.github/workflows/ci.yml` builds the kernels and runs the test suite.

## License

Apache License 2.0, the same license as pyscf.
