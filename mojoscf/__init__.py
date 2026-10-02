"""mojoscf: Mojo replacements for the Python glue in pyscf's SCF driver.

>>> import mojoscf
>>> mf = mojoscf.RHF(mol).run()

See :mod:`mojoscf.scf` for details and :mod:`mojoscf.kernels` for the
individual NumPy-facing kernels.
"""
from ._backend import BackendError, backend_info, blas_args, blas_config, build_extension, set_blas, use_native
from . import guess, kernels
from .diis import CDIIS
from .scf import RHF, UHF, accelerate, is_supported, kernel

__version__ = "0.2.0"

__all__ = [
    "RHF",
    "UHF",
    "CDIIS",
    "kernel",
    "accelerate",
    "is_supported",
    "kernels",
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
