"""mojoscf: Mojo replacements for the Python glue in pyscf's SCF driver.

>>> import mojoscf
>>> mf = mojoscf.RHF(mol).run()

See :mod:`mojoscf.scf` for details, :mod:`mojoscf.kernels` for the
individual NumPy-facing kernels and :mod:`mojoscf.integrals` for the Mojo
integral engine (``mojoscf.integrals.attach(mf)`` makes an SCF object use it);
:mod:`mojoscf.grad` has the nuclear gradients (``mf.nuc_grad_method()``) and
:mod:`mojoscf.qmmm` the QM/MM terms used for ``pyscf.qmmm`` objects;
:mod:`mojoscf.dft` the exchange-correlation integration for pyscf's Kohn-Sham objects,
:mod:`mojoscf.mcpdft` the on-top functional terms of pyscf's MC-PDFT,
:mod:`mojoscf.fcirdm` the FCI 3-particle density matrices and NEVPT2 4-particle contractions,
:mod:`mojoscf.nlchess` the VV10 pair sums of pyscf's analytical Hessian, and
:mod:`mojoscf.soscf` the orbital Hessian of pyscf's second-order SCF (``mf.newton()``).
"""
from ._backend import BackendError, backend_info, blas_args, blas_config, build_extension, set_blas, use_native
from ._backend import tune_openblas_spin as _tune_openblas_spin
from . import dft, fcirdm, grad, guess, integrals, kernels, mcpdft, nlchess, qmmm, soscf, solvent
from .diis import CDIIS
from .scf import RHF, UHF, accelerate, is_supported, kernel, native_veff

_tune_openblas_spin()
mcpdft.install_on_import()
fcirdm.install()
dft.install_nlc()
nlchess.install()
soscf.install()

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
    "fcirdm",
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
