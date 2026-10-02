import numpy as np
import pytest
from pyscf import gto

import mojoscf


@pytest.fixture(scope="session")
def ext():
    return mojoscf._backend.get_extension()


@pytest.fixture(scope="session", params=["blas", "native"])
def blas(request, ext):
    """Both the BLAS/LAPACK-backed and the pure-Mojo code paths."""
    if request.param == "blas":
        small, large = mojoscf.blas_config()
        if not small[0]:
            pytest.skip("no BLAS/LAPACK library found; only the native path is available")
        return (small, large)
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
