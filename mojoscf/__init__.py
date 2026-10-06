"""mojoscf: Mojo replacements for the Python glue in pyscf's SCF driver.

>>> import mojoscf
>>> mf = mojoscf.RHF(mol).run()

See :mod:`mojoscf.scf` for details, :mod:`mojoscf.kernels` for the
individual NumPy-facing kernels and :mod:`mojoscf.integrals` for the Mojo
integral engine (``mojoscf.integrals.attach(mf)`` makes an SCF object use it);
:mod:`mojoscf.grad` has the nuclear gradients (``mf.nuc_grad_method()``) and
:mod:`mojoscf.qmmm` the QM/MM terms used for ``pyscf.qmmm`` objects;
:mod:`mojoscf.dft` the exchange-correlation integration for pyscf's Kohn-Sham objects.
"""
from ._backend import BackendError, backend_info, blas_args, blas_config, build_extension, set_blas, use_native
from . import dft, grad, guess, integrals, kernels, qmmm
from .diis import CDIIS
from .scf import RHF, UHF, accelerate, is_supported, kernel, native_veff

__version__ = "0.9.0"

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
    "dft",
    "grad",
    "guess",
    "qmmm",
    "backend_info",
    "blas_args",
    "blas_config",
    "build_extension",
    "set_blas",
    "use_native",
    "BackendError",
    "__version__",
]
