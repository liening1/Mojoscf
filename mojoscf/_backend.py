"""Loading of the Mojo extension and discovery of a BLAS/LAPACK library.

The compiled extension lives next to this file as ``_mojoscf.so``.  It is
(re)built from the sources in ``_mojo/`` with the ``mojo`` compiler whenever
the sources changed, unless ``MOJOSCF_SKIP_BUILD=1`` is set.

Environment variables
---------------------
MOJOSCF_BLAS        ``/path/to/libblas.so[:symbol_prefix]`` to force one library for all sizes.
MOJOSCF_THREADED_MIN  matrix dimension from which the multi-threaded BLAS is used
                    (default 200); smaller matrices use the sequential library.
MOJOSCF_DF_BLOCK_MB work-buffer budget of the density-fitting J/K build (default 512).
MOJOSCF_NATIVE      ``1`` to use the pure-Mojo fallbacks instead of BLAS/LAPACK.
MOJOSCF_SKIP_BUILD  ``1`` to never invoke the Mojo compiler.
MOJOSCF_MOJO        Path of the ``mojo`` executable (default: ``mojo`` on PATH).
MOJOSCF_OPENBLAS_TIMEOUT  busy-wait of NumPy's and SciPy's OpenBLAS worker threads after a
                    call, as log2 of CPU cycles (default 16; ``0`` keeps OpenBLAS's own
                    setting), see :func:`tune_openblas_spin`.
"""
from __future__ import annotations

import ctypes
import glob
import hashlib
import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

__all__ = [
    "build_extension",
    "get_extension",
    "blas_args",
    "df_block_mb",
    "set_blas",
    "use_native",
    "backend_info",
    "BackendError",
]

_PKG_DIR = Path(__file__).resolve().parent
_MOJO_SRC = _PKG_DIR / "_mojo"
_EXT_NAME = "_mojoscf"
_EXT_PATH = _PKG_DIR / f"{_EXT_NAME}.so"
_STAMP_PATH = _PKG_DIR / f"{_EXT_NAME}.hash"

_ext = None
# (small-matrix library, large-matrix library), each ``(path, symbol_prefix)``.
_blas: tuple[tuple[str, str], tuple[str, str]] | None = None
THREADED_MIN_DEFAULT = 200


class BackendError(RuntimeError):
    """The Mojo extension could not be built or loaded."""


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def source_hash() -> str:
    """Hash of all Mojo sources, used to detect stale builds."""
    h = hashlib.sha256()
    for path in sorted(_MOJO_SRC.rglob("*.mojo")):
        h.update(str(path.relative_to(_MOJO_SRC)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


def mojo_executable() -> str | None:
    exe = os.environ.get("MOJOSCF_MOJO")
    if exe:
        return exe
    exe = shutil.which("mojo")
    if exe:
        return exe
    candidate = Path(sys.executable).with_name("mojo")
    if candidate.exists():
        return str(candidate)
    return None


def build_extension(force: bool = False, verbose: bool = False) -> Path:
    """Compile ``_mojo/`` into ``_mojoscf.so`` if it is missing or stale."""
    current = source_hash()
    if (
        not force
        and _EXT_PATH.exists()
        and _STAMP_PATH.exists()
        and _STAMP_PATH.read_text().strip() == current
    ):
        return _EXT_PATH
    exe = mojo_executable()
    if exe is None:
        raise BackendError(
            "The mojoscf extension needs to be compiled but no `mojo` compiler was "
            "found. Install it with `pip install modular` (or set MOJOSCF_MOJO)."
        )
    tmp = _EXT_PATH.with_suffix(".building.so")
    cmd = [exe, "build", str(_MOJO_SRC / "__init__.mojo"), "--emit", "shared-lib", "-o", str(tmp)]
    if verbose:
        print("[mojoscf] " + " ".join(cmd), file=sys.stderr)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise BackendError(
            "Compiling the mojoscf Mojo kernels failed:\n" + proc.stdout + proc.stderr
        )
    os.replace(tmp, _EXT_PATH)
    _STAMP_PATH.write_text(current + "\n")
    return _EXT_PATH


def get_extension():
    """Import (building first if necessary) the ``mojoscf._mojoscf`` module."""
    global _ext
    if _ext is not None:
        return _ext
    if not _env_flag("MOJOSCF_SKIP_BUILD"):
        try:
            build_extension()
        except BackendError:
            if not _EXT_PATH.exists():
                raise
    elif not _EXT_PATH.exists():
        raise BackendError(
            f"{_EXT_PATH} does not exist and MOJOSCF_SKIP_BUILD is set; "
            "run `python -m mojoscf.build` first."
        )
    _ext = importlib.import_module(f"{__package__}.{_EXT_NAME}")
    return _ext


# ---------------------------------------------------------------------------
# BLAS / LAPACK discovery
# ---------------------------------------------------------------------------

_PRELOADED: set[str] = set()


def _preload_dependencies(libdir: str) -> None:
    """Load the Fortran runtime libraries shipped next to a bundled OpenBLAS.

    Wheels (pyscf, scipy) ship ``libgfortran``/``libquadmath``/``libgomp`` with
    mangled names that the dynamic loader cannot find by itself when we
    ``dlopen`` the BLAS library directly, so they are loaded first, globally.
    """
    for pattern in ("libquadmath*", "libgfortran*", "libgomp*"):
        for path in sorted(glob.glob(os.path.join(libdir, pattern))):
            if path in _PRELOADED:
                continue
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                _PRELOADED.add(path)
            except OSError:
                pass


def _bundled_libraries() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Bundled/system BLAS+LAPACK libraries, as ``(sequential, threaded)`` lists.

    The OpenBLAS shipped inside pyscf wheels is a *sequential* build.  For the
    small matrices of a small-molecule SCF it is the better choice: a threaded
    BLAS then competes with pyscf's OpenMP integral code for the cores
    (measured: up to 40% slower SCF runs).  From a few hundred orbitals on,
    a threaded OpenBLAS makes the glue 2-4x faster than the sequential one.
    NumPy's own copy (``scipy_openblas64``: ILP64, symbols ``scipy_dgemm_64_``)
    comes first: pyscf's Python code (DIIS, ``make_rdm1``, ...) runs its
    matrix products on it, and OpenBLAS worker threads keep spinning for about
    0.1 s after each call, so a second thread pool (SciPy's LP64 copy, the
    fallback) called in between would compete with them for the cores
    (measured: 25% of a Kohn-Sham SCF loop).
    """
    seq: list[tuple[str, str]] = []
    thr: list[tuple[str, str]] = []
    try:
        import pyscf  # noqa: F401

        libdir = os.path.join(os.path.dirname(pyscf.__file__), "lib")
        for path in sorted(glob.glob(os.path.join(libdir, "libopenblas*.so*"))):
            seq.append((path, ""))
    except ImportError:
        pass
    try:
        import numpy

        libdir = os.path.join(os.path.dirname(os.path.dirname(numpy.__file__)), "numpy.libs")
        for path in sorted(glob.glob(os.path.join(libdir, "libscipy_openblas64_*.so*"))):
            thr.append((path, "scipy_"))
    except ImportError:  # pragma: no cover
        pass
    try:
        import scipy  # noqa: F401

        libdir = os.path.join(os.path.dirname(os.path.dirname(scipy.__file__)), "scipy.libs")
        for path in sorted(glob.glob(os.path.join(libdir, "libscipy_openblas-*.so*"))):
            thr.append((path, "scipy_"))
    except ImportError:
        pass
    system = [(name, "") for name in ("libopenblas.so.0", "libopenblas.so", "liblapack.so.3", "liblapack.so")]
    return seq + system, thr + system


def _probe(ext, path: str, prefix: str) -> bool:
    try:
        libdir = os.path.dirname(path)
        if libdir:
            _preload_dependencies(libdir)
        return bool(ext.blas_probe(path, prefix))
    except Exception:  # pragma: no cover - defensive
        return False


def _first_working(ext, candidates) -> tuple[str, str] | None:
    for path, prefix in candidates:
        if _probe(ext, path, prefix):
            return (path, prefix)
    return None


def _discover_blas() -> tuple[tuple[str, str], tuple[str, str]]:
    native = ("", "")
    if _env_flag("MOJOSCF_NATIVE"):
        return (native, native)
    ext = get_extension()
    spec = os.environ.get("MOJOSCF_BLAS")
    if spec:
        path, _, prefix = spec.partition(":")
        if _probe(ext, path, prefix):
            return ((path, prefix), (path, prefix))
        raise BackendError(f"MOJOSCF_BLAS={spec!r} does not provide dgemm_/dsygvd_/dsyevd_")
    seq_c, thr_c = _bundled_libraries()
    small = _first_working(ext, seq_c)
    large = _first_working(ext, thr_c)
    if small is None:
        small = large
    if large is None:
        large = small
    if small is None:
        return (native, native)
    return (small, large)


def df_block_mb() -> int:
    """Work-buffer budget (MB) for the density-fitting exchange build."""
    try:
        return max(16, int(os.environ.get("MOJOSCF_DF_BLOCK_MB", 512)))
    except ValueError:
        return 512


def threaded_min() -> int:
    """Matrix dimension from which the large-matrix (threaded) library is used."""
    try:
        return int(os.environ.get("MOJOSCF_THREADED_MIN", THREADED_MIN_DEFAULT))
    except ValueError:
        return THREADED_MIN_DEFAULT


def blas_config() -> tuple[tuple[str, str], tuple[str, str]]:
    """``((small_path, prefix), (large_path, prefix))``; empty path = native Mojo."""
    global _blas
    if _blas is None:
        _blas = _discover_blas()
    return _blas


_thread_ctl: dict = {}


def _openblas_thread_control(path: str, prefix: str):
    """(get_num_threads, set_num_threads) of an OpenBLAS library, or None."""
    key = (path, prefix)
    if key not in _thread_ctl:
        ctl = None
        if path:
            try:
                import ctypes

                lib = ctypes.CDLL(path)
                # ILP64 builds with suffixed symbols name them openblas_get_num_threads64_
                sfx = "" if hasattr(lib, prefix + "openblas_get_num_threads") else "64_"
                get = getattr(lib, prefix + "openblas_get_num_threads" + sfx)
                set_ = getattr(lib, prefix + "openblas_set_num_threads" + sfx)
                get.restype = ctypes.c_int
                set_.argtypes = [ctypes.c_int]
                ctl = (get, set_)
            except (OSError, AttributeError):
                ctl = None
        _thread_ctl[key] = ctl
    return _thread_ctl[key]


def _scipy_blas_control():
    """(get_num_threads, set_num_threads) of SciPy's bundled OpenBLAS, or None."""
    try:
        import scipy
    except ImportError:  # pragma: no cover
        return None
    libdir = os.path.join(os.path.dirname(os.path.dirname(scipy.__file__)), "scipy.libs")
    for path in sorted(glob.glob(os.path.join(libdir, "libscipy_openblas-*.so*"))):
        ctl = _openblas_thread_control(path, "scipy_")
        if ctl is not None:
            return ctl
    return None


class serial_scipy_blas:
    """Context manager: SciPy's own OpenBLAS (``scipy.linalg``) on one thread inside the block.

    SciPy wheels bundle an OpenBLAS separate from NumPy's, with its own thread
    pool; both pools keep their threads spinning for about 0.1 s after each
    call, so alternating ``numpy.linalg`` and ``scipy.linalg`` calls on
    matrices of a few hundred rows (pyscf's TDDFT subspace solver: ``lu``,
    ``inv``, ``cholesky``, ``eigh`` every iteration) makes each wait for the
    other pool's spinning threads: 52 ms for an ``inv`` + ``cholesky`` pair of
    400 x 400 that takes 9 ms with SciPy's calls on the calling thread.
    Nests; restores the previous thread count.
    """

    def __enter__(self):
        self._ctl = _scipy_blas_control()
        self._n = None
        if self._ctl is not None and not _env_flag("MOJOSCF_KEEP_SCIPY_THREADS"):
            get, set_ = self._ctl
            self._n = get()
            if self._n > 1:
                set_(1)
        return self

    def __exit__(self, *exc):
        if self._n is not None and self._n > 1:
            self._ctl[1](self._n)
        return False


OPENBLAS_TIMEOUT_DEFAULT = 16
_spin_tuned = False


def tune_openblas_spin() -> int:
    """Shorten the busy-wait of the OpenBLAS thread pools of NumPy and SciPy; returns how many were changed.

    After each multi-threaded call, OpenBLAS keeps its worker threads
    spinning for 2**28 CPU cycles (about 0.1 s) in case another call follows.
    Everything else that runs in parallel in between then shares the cores
    with them: pyscf's OpenMP code (``lib.dot`` splits small products over
    threads, libcint, the FCI solver) and the worker threads of the Mojo
    kernels.  In pyscf's DF-CASSCF a 5 x 175 x 25 ``lib.dot`` took 0.4 ms
    instead of 25 us; whole SCF runs were 10-30% slower.  This restarts the
    pools of the bundled libraries already loaded with a spin of
    2**``MOJOSCF_OPENBLAS_TIMEOUT`` cycles (``OPENBLAS_THREAD_TIMEOUT``, read
    when a pool starts): long enough for the back-to-back calls inside
    LAPACK (``eigh`` keeps its speed), short enough to release the cores for
    the code that follows.  Nothing is changed when ``OPENBLAS_THREAD_TIMEOUT``
    is set in the environment (OpenBLAS then already uses it) or
    ``MOJOSCF_OPENBLAS_TIMEOUT`` is 0.  Runs once, when mojoscf is imported.
    """
    global _spin_tuned
    if _spin_tuned or "OPENBLAS_THREAD_TIMEOUT" in os.environ:
        return 0
    _spin_tuned = True
    try:
        n = int(os.environ.get("MOJOSCF_OPENBLAS_TIMEOUT", OPENBLAS_TIMEOUT_DEFAULT))
    except ValueError:
        n = OPENBLAS_TIMEOUT_DEFAULT
    if n <= 0:
        return 0
    paths = []
    for pkg, pattern in (("numpy", "libscipy_openblas64_*.so*"), ("scipy", "libscipy_openblas-*.so*")):
        try:
            mod = importlib.import_module(pkg)
        except ImportError:  # pragma: no cover
            continue
        libdir = os.path.join(os.path.dirname(os.path.dirname(mod.__file__)), pkg + ".libs")
        paths += sorted(glob.glob(os.path.join(libdir, pattern)))
    changed = 0
    os.environ["OPENBLAS_THREAD_TIMEOUT"] = str(min(n, 30))
    try:
        for path in paths:
            try:
                # only libraries the process has loaded; the pool restarts at the next threaded call
                lib = ctypes.CDLL(path, mode=getattr(os, "RTLD_NOLOAD", 4))
                read_env, shutdown = lib.openblas_read_env, lib.blas_thread_shutdown_
            except (OSError, AttributeError):
                continue
            read_env()
            shutdown()
            changed += 1
    finally:
        del os.environ["OPENBLAS_THREAD_TIMEOUT"]
    return changed


def worker_blas() -> tuple[str, str]:
    """``(path, prefix)`` of the BLAS for GEMMs issued concurrently from Mojo worker threads.

    The bundled sequential library is old (pyscf's OpenBLAS 0.3.3, no AVX-512
    kernels: 13 GFlop/s per thread for a 200 x 128 x 200 GEMM here, against
    45 for scipy's).  When the large-matrix library is an OpenBLAS whose
    thread count can be set, the Mojo kernels use it, pinned to one thread
    around their parallel regions (``Blas.serial_begin``); otherwise the
    sequential library.
    """
    small, large = blas_config()
    if large != small and _openblas_thread_control(*large) is not None and not _env_flag("MOJOSCF_SEQ_BLAS"):
        return large
    return small


def blas_args(n: int = 0) -> tuple[str, str]:
    """``(library_path, symbol_prefix)`` to use for matrices of dimension ``n``.

    An empty path selects the native Mojo fallbacks.
    """
    small, large = blas_config()
    return large if n >= threaded_min() else small


def set_blas(path: str, prefix: str = "", large_path: str | None = None, large_prefix: str = "") -> None:
    """Force BLAS/LAPACK shared libraries (empty path = native Mojo kernels).

    With one argument the library is used for all sizes; ``large_path`` selects
    a different (threaded) library for matrices at or above ``threaded_min()``.
    """
    global _blas
    ext = get_extension() if (path or large_path) else None
    for p, pre in ((path, prefix), (large_path or "", large_prefix)):
        if p and not _probe(ext, p, pre):
            raise BackendError(f"{p!r} does not provide dgemm_/dsygvd_/dsyevd_ (prefix {pre!r})")
    small = (path, prefix)
    large = small if large_path is None else (large_path, large_prefix)
    _blas = (small, large)


def use_native() -> None:
    """Use the pure-Mojo GEMM and eigensolvers (slow for large basis sets)."""
    set_blas("", "")


def backend_info() -> dict:
    ext = get_extension()
    (path, prefix), (lpath, lprefix) = blas_config()
    level, width = ext.runtime_info()
    return {
        "extension": str(_EXT_PATH),
        "kernels_version": ext.version(),
        "blas_library": path or None,
        "blas_symbol_prefix": prefix,
        "blas_library_large": lpath or None,
        "blas_symbol_prefix_large": lprefix,
        "threaded_min": threaded_min(),
        "native_fallback": not path,
        "parallelism_level": int(level),
        "simd_width_f64": int(width),
    }
