# mojoscf

**Mojo replacements for the Python glue in pyscf's SCF driver.**

[pyscf](https://pyscf.org) does its heavy lifting (integrals, J/K builds) in C,
but the self-consistent-field loop that stitches those pieces together is
Python + NumPy: Fock assembly, damping, DIIS extrapolation, the generalised
eigenproblem, occupations, density matrices, energies, convergence tests and
logging.  For small and medium molecules that glue is a large fraction of the
wall time.  `mojoscf` re-implements exactly that layer in
[Mojo](https://www.modular.com/mojo) and leaves everything else to pyscf.

* **Drop-in**: `mojoscf.RHF(mol)` and `mojoscf.UHF(mol)` are subclasses of
  `pyscf.scf.hf.RHF` / `pyscf.scf.uhf.UHF`; results (energies, orbitals,
  iteration counts, `scf_summary`) agree with pyscf to round-off because the
  driver is a port of `pyscf.scf.hf.kernel`, not a reimplementation of SCF.
  UHF includes open-shell and broken-symmetry (BS) calculations.
* **Whole loop in Mojo**: one native call runs all iterations; the only Python
  call per cycle is `mf.get_veff`, which is pyscf's compiled two-electron code.
* **Individual kernels** are also exposed (`mojoscf.kernels`) and a
  Mojo-backed `CDIIS` class can be dropped into any pyscf SCF object.
* **BLAS/LAPACK** (OpenBLAS bundled with pyscf and SciPy) is called from Mojo
  through `dlopen`, choosing a sequential or a threaded library by matrix
  size; portable Mojo fallbacks (SIMD GEMM, Jacobi eigensolver) keep
  everything working when no library is found.

## Results

Hartree-Fock on 4 cores, pyscf 2.14, Mojo 1.1.0, default backend (pyscf's
bundled OpenBLAS).  Both drivers use pyscf's direct-SCF `get_veff`, the same
initial guess and `conv_tol = 1e-10`; best of 3 runs.

| system             | nao | cycles | pyscf [s] | mojoscf [s] | speed-up | &#124;ΔE&#124; [Eh] |
|--------------------|----:|-------:|----------:|------------:|---------:|--------:|
| H2O / STO-3G       |   7 |    7/7 |     0.032 |       0.019 |    1.7x  | 1e-14 |
| H2O / cc-pVDZ      |  24 |    9/9 |     0.169 |       0.050 |    3.4x  | 6e-14 |
| H2O / cc-pVTZ      |  58 |    9/9 |     0.191 |       0.095 |    2.0x  | 3e-14 |
| benzene / STO-3G   |  36 |    7/7 |     0.215 |       0.180 |    1.2x  | 2e-13 |
| benzene / cc-pVDZ  | 114 |    8/8 |     2.760 |       0.874 |    3.2x  | 9e-13 |
| (H2O)3 / cc-pVDZ   |  72 |  10/10 |     0.478 |       0.310 |    1.5x  | 7e-13 |

(`python benchmarks/bench_scf.py`.)  The remaining time is the integral code
both drivers share, so the speed-up shrinks as `get_veff` grows relative to
the glue.  Per-cycle logs are identical to pyscf's to all printed digits,
including with damping, level shifting and DIIS damping switched on.

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

With density fitting the two-electron part is cheap enough to see the glue
(`python benchmarks/bench_large.py`, 4 cores).  "Glue" is total time minus the
time inside `mf.get_veff`, i.e. exactly the part mojoscf replaces; it is not
affected by the run-to-run noise (about 10%) of the J/K code both drivers share.

| system                           | nao | cycles | total pyscf [s] | total mojoscf [s] | speed-up | glue pyscf [s] | glue mojoscf [s] | glue speed-up | &#124;ΔE&#124; [Eh] |
|----------------------------------|----:|-------:|----------:|----------:|--------:|--------:|--------:|--------:|--------:|
| C10H22 / 6-31G*                  | 184 |    9/9 |       8.4 |       6.6 |   1.27x |    1.78 |    0.36 |   4.95x | 5e-13 |
| C20H42 / 6-31G                   | 264 |    8/8 |      27.5 |      25.7 |   1.07x |    1.79 |    0.52 |   3.43x | 1e-11 |
| (H2O)10 / cc-pVDZ                | 240 |  10/10 |      13.3 |      11.5 |   1.16x |    2.23 |    0.54 |   4.11x | 2e-12 |
| C60 / STO-3G                     | 300 |    8/8 |      94.2 |      92.8 |   1.01x |    1.81 |    0.69 |   2.65x | 2e-10 |
| (H2O)20 / cc-pVDZ                | 480 |  10/10 |      81.2 |      82.0 |   0.99x |    2.79 |    0.88 |   3.18x | 2e-11 |
| C20H41 radical / 6-31G, UHF      | 262 |  13/13 |      60.2 |      55.3 |   1.09x |    4.74 |    0.76 |   6.26x | 2e-11 |
| (H2O)10 cation / cc-pVDZ, UHF    | 240 |  19/19 |      33.2 |      28.3 |   1.18x |    6.84 |    0.91 |   7.48x | 7e-12 |

The glue is 2.7 to 7.5 times faster at every size, but it is only 2 to 15% of
these runs, so the total improves by 1.0 to 1.3 times: once the integral code
dominates (C60, 20 waters) the end-to-end gain disappears into the noise.
Speed-ups of 2 to 3 times overall need the glue to be a large fraction of the
run, as in the small and medium systems above or with cheaper two-electron
methods.

Two things in this table were found by running these systems, not the small
ones, and are fixed: the density handed to `get_veff` must carry pyscf's
`mo_coeff`/`mo_occ` tags (density fitting then uses a much cheaper exchange
build; without them the first call was 2x and the SCF up to 6x slower), and the
BLAS library has to be chosen by matrix size (below).

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
| `get_veff` (J/K)                            | pyscf C (`libcvhf`)       | unchanged, called once per cycle     |

Source layout:

```
mojoscf/
  _mojo/linalg.mojo    vector kernels, GEMM/eigensolver dispatch, native fallbacks
  _mojo/kernels.mojo   make_rdm1, get_occ, get_grad, damping, level_shift, DIIS errvec, J/K
  _mojo/diis.mojo      pyscf-compatible CDIIS bookkeeping and extrapolation
  _mojo/driver.mojo    the RHF SCF loop (port of pyscf.scf.hf.kernel)
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
