# mojoscf

**Mojo replacements for the Python glue in pyscf's SCF driver.**

[pyscf](https://pyscf.org) does its heavy lifting (integrals, J/K builds) in C,
but the self-consistent-field loop that stitches those pieces together is
Python + NumPy: Fock assembly, damping, DIIS extrapolation, the generalised
eigenproblem, occupations, density matrices, energies, convergence tests and
logging.  For small and medium molecules that glue is a large fraction of the
wall time.  `mojoscf` re-implements exactly that layer in
[Mojo](https://www.modular.com/mojo) and leaves everything else to pyscf.

* **Drop-in**: `mojoscf.RHF(mol)` is a `pyscf.scf.hf.RHF` subclass; results
  (energies, orbitals, iteration counts, `scf_summary`) agree with pyscf to
  round-off because the driver is a port of `pyscf.scf.hf.kernel`, not a
  reimplementation of SCF.
* **Whole loop in Mojo**: one native call runs all iterations; the only Python
  call per cycle is `mf.get_veff`, which is pyscf's compiled two-electron code.
* **Individual kernels** are also exposed (`mojoscf.kernels`) and a
  Mojo-backed `CDIIS` class can be dropped into any pyscf SCF object.
* **BLAS/LAPACK** (the OpenBLAS bundled with pyscf by default) is called from
  Mojo through `dlopen`; portable Mojo fallbacks (SIMD GEMM, Jacobi
  eigensolver) keep everything working when no library is found.

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
`dlopen`, so no link-time dependency exists.  Libraries are tried in this
order: the OpenBLAS bundled with pyscf wheels (a *sequential* build, which is
deliberately the default: inside the SCF loop the glue matrices are small and
a threaded BLAS whose workers spin between calls competes with pyscf's OpenMP
integral code, costing up to 40% of the SCF time in our measurements), then
SciPy's bundled OpenBLAS (LP64, 4 threads, `scipy_` symbol prefix; the better
choice for large stand-alone kernel calls), then a system `libopenblas` /
`liblapack`.  Without any library the pure-Mojo fallbacks (SIMD GEMM, Jacobi
eigensolver) are used; they are correct but about 20% slower than pyscf for
the systems above because the Jacobi solver is O(n³) per sweep.

## What runs where

| step in `pyscf.scf.hf.kernel`               | pyscf                     | mojoscf                              |
|---------------------------------------------|---------------------------|--------------------------------------|
| `h1e + vhf`, damping, level shift           | NumPy                     | Mojo SIMD kernels                    |
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
  scf.py               RHF class, kernel(), accelerate()
tests/                 kernels vs NumPy/pyscf references; full SCF vs pyscf
benchmarks/            bench_scf.py, bench_kernels.py
```

## Scope and limitations

* Closed-shell **RHF** with real orbitals.  ROHF, UHF, GHF, Kohn-Sham DFT,
  symmetry-adapted and second-order (Newton) SCF objects are rejected by
  `accelerate` and are not provided as classes yet.
* Only CDIIS is native.  EDIIS/ADIIS, DIIS objects assigned to `mf.diis`,
  `diis_space_rollback`, `diis_file`, a custom `check_convergence` and
  dispersion corrections make the driver fall back to pyscf's loop (with the
  Mojo glue functions still in use), so results stay correct.
* The checkpoint file is written once at the end of the SCF instead of every
  cycle.
* Only the LP64 (32-bit integer) BLAS/LAPACK interface is supported.

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
