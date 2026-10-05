"""QM/MM electrostatic embedding (``pyscf.qmmm``) with the MM-charge integrals from the Mojo engine.

pyscf's QM/MM objects (``qmmm.mm_charge(mf, coords, charges, radii=None)``)
add the potential of MM point or Gaussian charges to the core Hamiltonian
and, in the gradients, its derivative for the QM atoms and the forces on the
MM charges.  pyscf evaluates these with one integral matrix per block of 200
charges (``int1e_grids``, ``int1e_grids_ip``, ``int3c2e_ip2``).  When such an
object runs on the Mojo driver (``mojoscf.accelerate``, or a mojoscf object
decorated with ``qmmm.mm_charge``), the classes here take over:

* ``_MojoQMMMHook`` (in front of the SCF class): ``get_hcore`` = the
  Hamiltonian without the charges plus
  ``-sum_k q_k <i|1/|r - R_k||j>`` from :func:`mojoscf.integrals.int1e_grids_sum`.
* ``_MojoQMMMGrad`` (a subclass of pyscf's ``QMMMGrad`` wrapped around
  mojoscf's own gradient object): the MM part of ``get_hcore`` from
  :func:`~mojoscf.integrals.int1e_grids_ip_sum` and ``grad_hcore_mm`` (the
  electronic force on every MM charge) from
  :func:`~mojoscf.integrals.mm_charge_forces`; the nucleus-charge terms
  (``energy_nuc``, ``grad_nuc``, ``grad_nuc_mm``) stay pyscf's, they are
  O(natm x ncharge) NumPy.

Everything else (screening-free sums over all charges, units, the Gaussian
charge model ``radii``) follows pyscf.  Molecules the Mojo one-electron code
does not support, shells beyond g, and ``MOJOSCF_INTEGRALS=libcint`` keep
pyscf's implementation.
"""
from __future__ import annotations

import numpy as np
from pyscf import lib
from pyscf.qmmm import itrf

from . import integrals


def _mm_data(mm_mol):
    """(coords in Bohr, charges, zetas or None for point charges) of pyscf's MM ``Mole``."""
    zetas = mm_mol.get_zetas() if mm_mol.charge_model == "gaussian" else None
    return mm_mol.atom_coords(), mm_mol.atom_charges(), zetas


def mojo_ok(mol, mm_mol) -> bool:
    """True if the MM-charge terms of ``mol`` in the field of ``mm_mol`` come from the Mojo engine."""
    return (
        integrals.engine() == "mojo"
        and getattr(mm_mol, "charge_model", None) in ("point", "gaussian")
        and integrals.mm_supported(mol)
    )


class _MojoQMMMHook:
    """Placed in front of a QM/MM SCF object run by the Mojo driver (see the module docstring)."""

    __name_mixin__ = "Mojo"

    def get_hcore(self, mol=None):
        mol = self.mol if mol is None else mol
        if not isinstance(self, itrf.QMMMSCF) or not mojo_ok(mol, self.mm_mol):
            return super().get_hcore(mol)
        h1e = super(itrf.QMMMSCF, self).get_hcore(mol)      # without the MM charges
        coords, charges, zetas = _mm_data(self.mm_mol)
        return h1e - integrals.int1e_grids_sum(mol, coords, charges, zetas)

    def nuc_grad_method(self):
        """Nuclear gradients with Mojo integrals, QM/MM terms included (:class:`_MojoQMMMGrad`)."""
        from .scf import _mojo_grad_method

        return _mojo_grad_method(self, _MojoQMMMHook)

    Gradients = nuc_grad_method


class _MojoQMMMGrad(itrf.QMMMGrad):
    """pyscf's ``QMMMGrad`` with the MM-charge derivative integrals from the Mojo engine."""

    def get_hcore(self, mol=None):
        """(QM one-electron derivative) + sum_k q_k <nabla i|1/|r - R_k||j>, as pyscf."""
        mol = self.mol if mol is None else mol
        mm_mol = self.base.mm_mol
        if not mojo_ok(mol, mm_mol):
            return super().get_hcore(mol)
        g_qm = super(itrf.QMMMGrad, self).get_hcore(mol)    # without the MM charges
        coords, charges, zetas = _mm_data(mm_mol)
        return g_qm + integrals.int1e_grids_ip_sum(mol, coords, charges, zetas)

    def grad_hcore_mm(self, dm, mol=None):
        """Electronic part of the gradient with respect to the MM charge positions, shape (ncharge, 3)."""
        mol = self.mol if mol is None else mol
        mm_mol = self.base.mm_mol
        dm = np.asarray(dm)
        if not mojo_ok(mol, mm_mol):
            return super().grad_hcore_mm(dm, mol)
        if dm.ndim == 3:        # (alpha, beta) densities
            dm = dm[0] + dm[1]
        coords, charges, zetas = _mm_data(mm_mol)
        # the integrals are symmetric in the AO pair, so only the symmetric part of dm contributes
        return integrals.mm_charge_forces(mol, 0.5 * (dm + dm.T), coords, charges, zetas)

    contract_hcore_mm = grad_hcore_mm


def qmmm_grad_for_scf(scf_grad):
    """``scf_grad`` (a gradient object of a QM/MM SCF) with the Mojo QM/MM terms, as pyscf's ``qmmm_grad_for_scf``."""
    if isinstance(scf_grad, _MojoQMMMGrad):
        return scf_grad
    if isinstance(scf_grad, itrf.QMMMGrad):
        # already decorated by pyscf: put the Mojo version in front
        return scf_grad.view(lib.make_class((_MojoQMMMGrad, type(scf_grad))))
    itrf.qmmm_grad_for_scf(scf_grad)        # pyscf's checks (X2C, the base being a QM/MM SCF)
    return scf_grad.view(lib.make_class((_MojoQMMMGrad, type(scf_grad))))
