"""mojoscf: Mojo replacements for the Python glue in pyscf's SCF driver.

>>> import mojoscf
>>> mf = mojoscf.RHF(mol).run()

See :mod:`mojoscf.scf` for details, :mod:`mojoscf.kernels` for the
individual NumPy-facing kernels and :mod:`mojoscf.integrals` for the Mojo
integral engine (``mojoscf.integrals.attach(mf)`` makes an SCF object use it);
:mod:`mojoscf.grad` has the nuclear gradients (``mf.nuc_grad_method()``) and
:mod:`mojoscf.qmmm` the QM/MM terms used for ``pyscf.qmmm`` objects;
:mod:`mojoscf.dft` the exchange-correlation integration for pyscf's Kohn-Sham objects and
:mod:`mojoscf.mcpdft` the on-top functional terms of pyscf's MC-PDFT.
"""
from ._backend import BackendError, backend_info, blas_args, blas_config, build_extension, set_blas, use_native
from ._backend import tune_openblas_spin as _tune_openblas_spin
from . import dft, grad, guess, integrals, kernels, mcpdft, qmmm, solvent
from .diis import CDIIS
from .scf import RHF, UHF, accelerate, is_supported, kernel, native_veff

_tune_openblas_spin()
mcpdft.install_on_import()

__version__ = "0.12.0"

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
    "mcpdft",
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
