"""QM/MM electrostatic embedding (``pyscf.qmmm``) with the MM-charge integrals from the Mojo engine.

pyscf's QM/MM objects (``qmmm.mm_charge(mf, coords, charges, radii=None)``)
add the potential of MM point or Gaussian charges to the core Hamiltonian
and, in the gradients, its derivative for the QM atoms and the forces on the
MM charges.  pyscf evaluates these with one integral matrix per block of 200
charges (``int1e_grids``, ``int1e_grids_ip``, ``int3c2e_ip2``).  When such an
object runs on the Mojo driver (``mojoscf.accelerate``, or a mojoscf object
decorated with ``qmmm.mm_charge``), or for any other pyscf SCF method
(Kohn-Sham DFT, ROHF, ...) through :func:`mm_charge` / :func:`attach` here,
the classes below take over:

* ``_MojoQMMMHook`` (in front of the SCF class): ``get_hcore`` = the
  Hamiltonian without the charges plus
  ``-sum_k q_k <i|1/|r - R_k||j>`` from :func:`mojoscf.integrals.int1e_grids_sum`.
* ``_MojoQMMMGrad`` (a subclass of pyscf's ``QMMMGrad`` wrapped around the
  method's gradient object, mojoscf's own for RHF/UHF): ``grad_elec`` adds
  the charge term of the QM-atom gradient and computes the forces on the
  charges in one pass (:func:`~mojoscf.integrals.mm_grad_terms`), which
  ``grad_hcore_mm`` then returns; ``get_hcore`` (the derivative matrix) uses
  :func:`~mojoscf.integrals.int1e_grids_ip_sum`.  The nucleus-charge terms
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


def _total_density(dm):
    """The symmetric part of the total density (the charge integrals are symmetric in the AO pair)."""
    dm = np.asarray(dm, dtype=np.float64)
    if dm.ndim == 3:            # (alpha, beta)
        dm = dm[0] + dm[1]
    return 0.5 * (dm + dm.T)


def _mm_key(mol, mm_mol, dm):
    coords, charges, zetas = _mm_data(mm_mol)
    return (mol.atom_coords(), mol._env.copy(), coords, charges, zetas, dm)


def _same_key(a, b):
    return all(
        (x is None and y is None) or (x is not None and y is not None and np.array_equal(x, y)) for x, y in zip(a, b)
    )


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
    """pyscf's ``QMMMGrad`` with the MM-charge derivative integrals from the Mojo engine.

    ``grad_elec`` adds the charge term of the QM-atom gradient from one
    density-contracted pass (:func:`mojoscf.integrals.mm_grad_terms`), which
    also yields the forces on the charges; ``grad_hcore_mm`` returns those
    when called for the same density, geometry and charges, so a gradient
    step with MM forces evaluates the charge integrals once.
    """

    _mm_in_hcore = True         # get_hcore includes the charges (not while grad_elec adds them itself)
    _mm_force_cache = None

    def get_hcore(self, mol=None):
        """(QM one-electron derivative) + sum_k q_k <nabla i|1/|r - R_k||j>, as pyscf."""
        mol = self.mol if mol is None else mol
        mm_mol = self.base.mm_mol
        if not mojo_ok(mol, mm_mol):
            return super().get_hcore(mol)
        g_qm = super(itrf.QMMMGrad, self).get_hcore(mol)    # without the MM charges
        if not self._mm_in_hcore:
            return g_qm
        coords, charges, zetas = _mm_data(mm_mol)
        return g_qm + integrals.int1e_grids_ip_sum(mol, coords, charges, zetas)

    def grad_elec(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        mol = self.mol
        mm_mol = self.base.mm_mol
        if not mojo_ok(mol, mm_mol):
            return super().grad_elec(mo_energy, mo_coeff, mo_occ, atmlst)
        mf = self.base
        dm = _total_density(mf.make_rdm1(mf.mo_coeff if mo_coeff is None else mo_coeff,
                                         mf.mo_occ if mo_occ is None else mo_occ))
        self._mm_in_hcore = False
        try:
            de = super().grad_elec(mo_energy, mo_coeff, mo_occ, atmlst)
        finally:
            del self._mm_in_hcore
        coords, charges, zetas = _mm_data(mm_mol)
        g_atoms, g_charges = integrals.mm_grad_terms(mol, dm, coords, charges, zetas)
        self._mm_force_cache = (_mm_key(mol, mm_mol, dm), g_charges)
        if atmlst is None:
            atmlst = range(mol.natm)
        return de + g_atoms[list(atmlst)]

    def grad_hcore_mm(self, dm, mol=None):
        """Electronic part of the gradient with respect to the MM charge positions, shape (ncharge, 3)."""
        mol = self.mol if mol is None else mol
        mm_mol = self.base.mm_mol
        if not mojo_ok(mol, mm_mol):
            return super().grad_hcore_mm(dm, mol)
        dm = _total_density(dm)
        cache = self._mm_force_cache
        if cache is not None and _same_key(cache[0], _mm_key(mol, mm_mol, dm)):
            return cache[1].copy()
        coords, charges, zetas = _mm_data(mm_mol)
        return integrals.mm_charge_forces(mol, dm, coords, charges, zetas)

    contract_hcore_mm = grad_hcore_mm


def attach(mf):
    """Make a QM/MM SCF object (``pyscf.qmmm``) take its MM-charge terms from the Mojo engine, in place.

    Works for any pyscf SCF method, including Kohn-Sham DFT and ROHF, which
    the native mojoscf loop does not run: their SCF and the QM part of their
    gradients stay pyscf's; the potential of the charges, its derivative for
    the QM atoms and the forces on the charges come from :mod:`mojoscf.integrals`.
    (``mojoscf.accelerate`` does this as well for the RHF/UHF it supports.)
    Returns ``mf``.
    """
    if not isinstance(mf, itrf.QMMMSCF):
        raise TypeError(f"{type(mf).__name__} is not a pyscf.qmmm QM/MM SCF object")
    if not isinstance(mf, _MojoQMMMHook):
        lib.set_class(mf, (_MojoQMMMHook, type(mf)))
    return mf


def mm_charge(method, atoms_or_coords, charges, radii=None, unit=None):
    """``pyscf.qmmm.mm_charge`` with the MM-charge terms from the Mojo engine (see :func:`attach`).

    >>> mf = mojoscf.qmmm.mm_charge(dft.RKS(mol, xc="b3lyp"), coords, charges)
    >>> mf.kernel(); g = mf.nuc_grad_method(); de = g.kernel()
    >>> f_mm = g.grad_hcore_mm(mf.make_rdm1()) + g.grad_nuc_mm()
    """
    return attach(itrf.mm_charge(method, atoms_or_coords, charges, radii=radii, unit=unit))


def qmmm_grad_for_scf(scf_grad):
    """``scf_grad`` (a gradient object of a QM/MM SCF) with the Mojo QM/MM terms, as pyscf's ``qmmm_grad_for_scf``."""
    if isinstance(scf_grad, _MojoQMMMGrad):
        return scf_grad
    if isinstance(scf_grad, itrf.QMMMGrad):
        # already decorated by pyscf: put the Mojo version in front
        return scf_grad.view(lib.make_class((_MojoQMMMGrad, type(scf_grad))))
    itrf.qmmm_grad_for_scf(scf_grad)        # pyscf's checks (X2C, the base being a QM/MM SCF)
    return scf_grad.view(lib.make_class((_MojoQMMMGrad, type(scf_grad))))
