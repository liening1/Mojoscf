"""mojoscf: Mojo replacements for the Python glue in pyscf's SCF driver.

>>> import mojoscf
>>> mf = mojoscf.RHF(mol).run()

See :mod:`mojoscf.scf` for details, :mod:`mojoscf.kernels` for the
individual NumPy-facing kernels and :mod:`mojoscf.integrals` for the Mojo
integral engine (``mojoscf.integrals.attach(mf)`` makes an SCF object use it).
"""
from ._backend import BackendError, backend_info, blas_args, blas_config, build_extension, set_blas, use_native
from . import guess, integrals, kernels
from .diis import CDIIS
from .scf import RHF, UHF, accelerate, is_supported, kernel, native_veff

__version__ = "0.5.0"

__all__ = [
    "RHF",
    "UHF",
    "CDIIS",
    "kernel",
    "native_veff",
    "accelerate",
    "is_supported",
    "kernels",
    "integrals",
    "guess",
    "backend_info",
    "blas_args",
    "blas_config",
    "build_extension",
    "set_blas",
    "use_native",
    "BackendError",
    "__version__",
]
