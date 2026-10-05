"""Nuclear gradients of RHF/UHF with derivative integrals from the Mojo engine.

``Gradients`` (RHF) and ``UGradients`` (UHF) are pyscf's gradient classes with
the integral work done by ``mojoscf.integrals``:

* ``grad_elec``: the two-electron part is the derivative of
  ``1/2 sum (ij|kl) G_ijkl`` (``G`` the two-particle density of the
  determinant) evaluated directly from the derivative integrals of the
  unique shell quartets (``mojoscf.integrals.grad2e``), instead of building
  pyscf's (3, nao, nao) derivative J/K matrices from 4-fold symmetric
  integrals; the one-electron and overlap terms are assembled as in pyscf
* ``get_hcore``: ``-(int1e_ipkin + int1e_ipnuc)``
* ``get_ovlp``: ``-int1e_ipovlp``
* ``hcore_generator``: the per-nucleus ``int1e_iprinv`` terms
* ``get_jk``/``get_veff``: ``-sum_kl (nabla i j|kl) D_lk`` and
  ``-sum_jk (nabla i j|kl) D_jk`` (integral-direct, for code that asks for
  the matrices)

Results agree with ``pyscf.grad`` to the precision of the integrals.
Molecules the engine does not support, density-fitted and X2C objects,
range-separated operators and subclasses that override ``get_veff`` or
``get_jk`` use pyscf's own implementation.

>>> mf = mojoscf.RHF(mol).run()
>>> g = mf.nuc_grad_method().kernel()
"""
from __future__ import annotations

import numpy as np
from pyscf.grad import rhf as rhf_grad
from pyscf.grad import uhf as uhf_grad
from pyscf.lib import logger

from . import integrals

__all__ = ["Gradients", "UGradients", "grad_tol"]

#: Screening threshold of the derivative integrals (``grad2e`` and the derivative J/K matrices).
grad_tol = 1e-14


class _MojoGradMixin:
    """Integral pieces of pyscf's gradient classes from the Mojo engine."""

    def _mojo_ok(self, mol=None, omega=None):
        mol = self.mol if mol is None else mol
        base = self.base
        return (
            integrals.available(mol)
            and not getattr(mol, "_pseudo", None)
            and getattr(base, "with_df", None) is None
            and getattr(base, "with_x2c", None) is None
            and not omega
        )

    _mojo_ip_cache = None

    def _ip_ints(self, mol):
        """(ipovlp, ipkin, ipnuc), computed once per molecule and geometry.

        ``get_hcore``, ``get_ovlp`` and ``hcore_generator`` each need them;
        the key holds the molecule itself and copies of its tables so that a
        scanner moving the atoms (in place or with a new object) recomputes.
        """
        cache = self._mojo_ip_cache
        if (
            cache is None
            or cache[0] is not mol
            or not np.array_equal(cache[1], mol._env)
            or not np.array_equal(cache[2], mol._bas)
            or not np.array_equal(cache[3], mol._atm)
        ):
            ints = integrals.int1e_ip(mol)
            cache = (mol, mol._env.copy(), mol._bas.copy(), mol._atm.copy(), ints)
            self._mojo_ip_cache = cache
        return cache[4]

    def get_hcore(self, mol=None):
        mol = self.mol if mol is None else mol
        if not self._mojo_ok(mol):
            return super().get_hcore(mol)
        _, t, v = self._ip_ints(mol)
        return -(t + v)

    def get_ovlp(self, mol=None):
        mol = self.mol if mol is None else mol
        if not self._mojo_ok(mol):
            return super().get_ovlp(mol)
        return -self._ip_ints(mol)[0]

    def hcore_generator(self, mol=None):
        mol = self.mol if mol is None else mol
        if not self._mojo_ok(mol):
            return super().hcore_generator(mol)
        charges = -mol.atom_charges().astype(np.float64)
        aoslices = mol.aoslice_by_atom()
        h1 = self.get_hcore(mol)

        def hcore_deriv(atm_id):
            p0, p1 = aoslices[atm_id, 2:]
            vrinv = integrals.int1e_iprinv(mol, atm_id) * charges[atm_id]
            vrinv[:, p0:p1] += h1[:, p0:p1]
            return vrinv + vrinv.transpose(0, 2, 1)

        return hcore_deriv

    def _mojo_jk_ok(self, mol, dm, omega):
        """The Mojo derivative J/K needs real symmetric densities (pyscf also passes others, e.g. in TDHF)."""
        if not self._mojo_ok(mol, omega):
            return False
        dm = np.asarray(dm)
        if dm.ndim < 2 or not np.isrealobj(dm):
            return False
        return np.allclose(dm, dm.swapaxes(-1, -2), rtol=0.0, atol=1e-12)

    def get_jk(self, mol=None, dm=None, hermi=0, omega=None):
        mol = self.mol if mol is None else mol
        if dm is None:
            dm = self.base.make_rdm1()
        if not self._mojo_jk_ok(mol, dm, omega):
            return super().get_jk(mol, dm, hermi, omega)
        cpu0 = (logger.process_clock(), logger.perf_counter())
        vj, vk = integrals.get_jk_ip1(mol, np.asarray(dm), tol=grad_tol)
        logger.timer(self, "vj and vk (Mojo)", *cpu0)
        return vj, vk

    def get_j(self, mol=None, dm=None, hermi=0, omega=None):
        mol = self.mol if mol is None else mol
        if dm is None:
            dm = self.base.make_rdm1()
        if not self._mojo_jk_ok(mol, dm, omega):
            return super().get_j(mol, dm, hermi, omega)
        return integrals.get_jk_ip1(mol, np.asarray(dm), with_k=False, tol=grad_tol)[0]

    def get_k(self, mol=None, dm=None, hermi=0, omega=None):
        mol = self.mol if mol is None else mol
        if dm is None:
            dm = self.base.make_rdm1()
        if not self._mojo_jk_ok(mol, dm, omega):
            return super().get_k(mol, dm, hermi, omega)
        return integrals.get_jk_ip1(mol, np.asarray(dm), with_j=False, tol=grad_tol)[1]


    def _direct_2e(self):
        """True if the two-electron term can bypass ``get_veff`` (it is not overridden)."""
        cls = type(self)
        pyscf_cls = uhf_grad.Gradients if self._unrestricted else rhf_grad.Gradients
        return (
            cls.get_veff is pyscf_cls.get_veff
            and cls.get_jk is _MojoGradMixin.get_jk
            and not ({"get_veff", "get_jk"} & self.__dict__.keys())
        )

    def grad_2e(self, dm0, mol=None):
        """d/dR of the two-electron energy at fixed density (natm, 3); ``dm0`` as from ``base.make_rdm1()``."""
        mol = self.mol if mol is None else mol
        dm0 = np.asarray(dm0)
        if self._unrestricted:
            return integrals.grad2e(mol, dm0[0] + dm0[1], dm0, 1.0, 1.0, tol=grad_tol)
        return integrals.grad2e(mol, dm0, dm0, 1.0, 0.5, tol=grad_tol)

    def grad_elec(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        """Electronic gradient, as pyscf's ``grad_elec`` with the two-electron term from ``grad_2e``."""
        if not (self._mojo_ok() and self._direct_2e()):
            return super().grad_elec(mo_energy, mo_coeff, mo_occ, atmlst)
        mf = self.base
        mol = self.mol
        if mo_energy is None:
            mo_energy = mf.mo_energy
        if mo_occ is None:
            mo_occ = mf.mo_occ
        if mo_coeff is None:
            mo_coeff = mf.mo_coeff
        log = logger.Logger(self.stdout, self.verbose)

        hcore_deriv = self.hcore_generator(mol)
        s1 = self.get_ovlp(mol)
        dm0 = np.asarray(mf.make_rdm1(mo_coeff, mo_occ))
        dme0 = np.asarray(self.make_rdm1e(mo_energy, mo_coeff, mo_occ))
        if self._unrestricted:
            dm0_sf = dm0[0] + dm0[1]
            dme0_sf = dme0[0] + dme0[1]
        else:
            dm0_sf, dme0_sf = dm0, dme0

        t0 = (logger.process_clock(), logger.perf_counter())
        log.debug("Computing Gradients of the Coulomb repulsion (Mojo)")
        de2 = self.grad_2e(dm0, mol)
        log.timer("gradients of 2e part", *t0)

        if atmlst is None:
            atmlst = range(mol.natm)
        aoslices = mol.aoslice_by_atom()
        de = np.zeros((len(atmlst), 3))
        for k, ia in enumerate(atmlst):
            p0, p1 = aoslices[ia, 2:]
            h1ao = hcore_deriv(ia)
            de[k] += np.einsum("xij,ij->x", h1ao, dm0_sf)
            de[k] += de2[ia]
            de[k] -= np.einsum("xij,ij->x", s1[:, p0:p1], dme0_sf[p0:p1]) * 2
            de[k] += self.extra_force(ia, locals())

        if log.verbose >= logger.DEBUG:
            log.debug("gradients of electronic part")
            rhf_grad._write(log, mol, de, atmlst)
        return de


class Gradients(_MojoGradMixin, rhf_grad.Gradients):
    """RHF nuclear gradients with Mojo derivative integrals (see the module docstring)."""

    _unrestricted = False


class UGradients(_MojoGradMixin, uhf_grad.Gradients):
    """UHF nuclear gradients with Mojo derivative integrals (see the module docstring)."""

    _unrestricted = True
