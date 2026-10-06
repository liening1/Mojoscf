import numpy as np
import pytest
from pyscf import gto

import mojoscf


@pytest.fixture(scope="session")
def ext():
    return mojoscf._backend.get_extension()


def _bundled(kind):
    """NumPy's ILP64 or SciPy's LP64 OpenBLAS as (path, prefix), if installed and loadable."""
    import glob
    import os

    import scipy

    root = os.path.dirname(os.path.dirname(np.__file__ if kind == "numpy" else scipy.__file__))
    pattern = "numpy.libs/libscipy_openblas64_*.so*" if kind == "numpy" else "scipy.libs/libscipy_openblas-*.so*"
    for path in sorted(glob.glob(os.path.join(root, pattern))):
        if mojoscf._backend._probe(mojoscf._backend.get_extension(), path, "scipy_"):
            return (path, "scipy_")
    return None


@pytest.fixture(scope="session", params=["blas", "native", "numpy-ilp64", "scipy-lp64"])
def blas(request, ext):
    """The BLAS/LAPACK-backed and the pure-Mojo code paths; NumPy's ILP64 and SciPy's LP64
    OpenBLAS each for all matrix sizes."""
    if request.param == "blas":
        small, large = mojoscf.blas_config()
        if not small[0]:
            pytest.skip("no BLAS/LAPACK library found; only the native path is available")
        return (small, large)
    if request.param != "native":
        lib = _bundled(request.param.split("-")[0])
        if lib is None:
            pytest.skip(f"{request.param} OpenBLAS not installed")
        return (lib, lib)
    return (("", ""), ("", ""))


@pytest.fixture
def use_blas(blas):
    """Temporarily select the given backend for the NumPy-facing wrappers."""
    saved = mojoscf.blas_config()
    mojoscf._backend._blas = blas
    try:
        yield blas
    finally:
        mojoscf._backend._blas = saved


@pytest.fixture(scope="session")
def h2o():
    return gto.M(atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587", basis="sto-3g", verbose=0)


@pytest.fixture(scope="session")
def h2o_dz():
    return gto.M(atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587", basis="cc-pvdz", verbose=0)


@pytest.fixture
def rng():
    return np.random.default_rng(1234)
