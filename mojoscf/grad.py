"""Nuclear gradients of RHF/UHF and RKS/UKS with derivative integrals from the Mojo engine.

``Gradients`` (RHF) and ``UGradients`` (UHF) are pyscf's gradient classes with
the integral work done by ``mojoscf.integrals``; ``DFGradients`` and
``DFUGradients`` are pyscf's density-fitted ones (``pyscf.df.grad``) likewise.
``KSGradients``, ``UKSGradients``, ``DFKSGradients`` and ``DFUKSGradients``
do the same for Kohn-Sham objects (``mojoscf.dft.accelerate``), adding the
exact exchange scaled by the functional's fraction and the XC term at fixed
grids (``mojoscf.dft.grad_xc``) to ``grad_2e``:

* ``grad_elec``: the two-electron part is the derivative of the two-electron
  energy evaluated directly from derivative integrals, contracted with the
  density factors as they are produced (``grad_2e``), instead of building
  pyscf's (3, nao, nao) derivative J/K matrices.  With exact integrals that
  is ``1/2 sum (ij|kl) G_ijkl`` over the unique shell quartets
  (``mojoscf.integrals.grad2e``); with density fitting it is
  ``sum (mu nu|P) Gamma_P,mu nu - 1/2 sum (P|Q) W_PQ`` including the response
  of the auxiliary basis (``mojoscf.integrals.grad2e_df``).  The one-electron
  and overlap terms are assembled as in pyscf.
* ``get_hcore``: ``-(int1e_ipkin + int1e_ipnuc)``
* ``get_ovlp``: ``-int1e_ipovlp``
* ``hcore_generator``: the per-nucleus ``int1e_iprinv`` terms
* ``get_jk`` (exact-integral classes): ``-sum_kl (nabla i j|kl) D_lk`` and
  ``-sum_jk (nabla i j|kl) D_jk`` (integral-direct, for code that asks for
  the matrices); the DF classes keep pyscf's DF ``get_jk``.

Results agree with ``pyscf.grad`` / ``pyscf.df.grad`` to the precision of the
integrals.  With effective core potentials only the ECP derivative integrals
come from pyscf; with X2C or finite nuclei all one-electron pieces do, the
two-electron part still from the engine.  Range-separated operators and
functionals, non-local correlation, ``grid_response``, ``only_dfj``,
``auxbasis_response = False`` and subclasses that override ``get_veff`` or
``get_jk`` use pyscf's own implementation.

>>> mf = mojoscf.RHF(mol).run()                  # or .density_fit().run()
>>> g = mf.nuc_grad_method().kernel()
"""
from __future__ import annotations

import numpy as np
from pyscf import lib
from pyscf.df.grad import rhf as df_rhf_grad
from pyscf.df.grad import rks as df_rks_grad
from pyscf.df.grad import uhf as df_uhf_grad
from pyscf.df.grad import uks as df_uks_grad
from pyscf.grad import rhf as rhf_grad
from pyscf.grad import rks as rks_grad
from pyscf.grad import uhf as uhf_grad
from pyscf.grad import uks as uks_grad
from pyscf.lib import logger

from . import integrals

__all__ = [
    "Gradients", "UGradients", "DFGradients", "DFUGradients",
    "KSGradients", "UKSGradients", "DFKSGradients", "DFUKSGradients", "grad_tol",
]

#: Screening threshold of the derivative integrals (``grad2e``, ``grad2e_df`` and the derivative J/K matrices).
grad_tol = 1e-14


def _ecp_atoms(mol):
    """Indices of the atoms that carry an effective core potential."""
    from pyscf import gto

    return set(mol._ecpbas[:, gto.ATOM_OF].tolist()) if mol.has_ecp() else set()


class _MojoGrad1eMixin:
    """One-electron derivative integrals from the Mojo engine and the assembly of ``grad_elec``.

    Subclasses provide ``_direct_2e()`` (whether ``grad_2e`` may replace the
    ``get_veff`` route) and ``grad_2e``.
    """

    _unrestricted = False
    _pyscf_grad = rhf_grad.Gradients      # the pyscf class whose get_veff/get_jk/extra_force grad_2e stands in for

    def _mojo_ok(self, mol=None, omega=None):
        """The one-electron pieces can come from the Mojo engine (no X2C or finite nuclei).

        With effective core potentials the ECP derivative integrals
        (``ECPscalar_ipnuc``, ``ECPscalar_iprinv``) are added from pyscf.
        """
        mol = self.mol if mol is None else mol
        return (
            integrals.available(mol, allow_ecp=True)
            and not getattr(mol, "_pseudo", None)
            and getattr(self.base, "with_x2c", None) is None
            and not omega
        )

    def _mojo_2e_ok(self, mol=None, omega=None):
        """The two-electron pieces can come from the Mojo engine (ECPs, X2C and finite nuclei allowed)."""
        mol = self.mol if mol is None else mol
        return integrals.available(mol, two_electron=True) and not getattr(mol, "_pseudo", None) and not omega

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
        h = t + v
        if mol.has_ecp():
            h = h + mol.intor("ECPscalar_ipnuc", comp=3)
        return -h

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
        ecp_atoms = _ecp_atoms(mol)

        def hcore_deriv(atm_id):
            p0, p1 = aoslices[atm_id, 2:]
            vrinv = integrals.int1e_iprinv(mol, atm_id) * charges[atm_id]
            if atm_id in ecp_atoms:
                with mol.with_rinv_at_nucleus(atm_id):
                    vrinv += mol.intor("ECPscalar_iprinv", comp=3)
            vrinv[:, p0:p1] += h1[:, p0:p1]
            return vrinv + vrinv.transpose(0, 2, 1)

        return hcore_deriv

    def _direct_2e(self):
        return False

    def _extra_force(self, atom_id, envs):
        return self.extra_force(atom_id, envs)

    def grad_elec(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        """Electronic gradient, as pyscf's ``grad_elec`` with the two-electron term from ``grad_2e``."""
        if not (self._mojo_2e_ok() and self._direct_2e()):
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

        # sum_ij D_ij hcore_deriv(A)_ij for all atoms from one pass over the integrals when
        # hcore_generator is ours: 2 (sum_{i in A} h1_ij D_ij - Z_A sum_ij D_ij <nabla i|1/r_A|j>)
        fast_h1 = type(self).hcore_generator is _MojoGrad1eMixin.hcore_generator and self._mojo_ok()
        if not fast_h1:
            hcore_deriv = self.hcore_generator(mol)
        s1 = self.get_ovlp(mol)
        dm0 = lib.tag_array(np.asarray(mf.make_rdm1(mo_coeff, mo_occ)), mo_coeff=mo_coeff, mo_occ=mo_occ)
        dme0 = np.asarray(self.make_rdm1e(mo_energy, mo_coeff, mo_occ))
        if self._unrestricted:
            dm0_sf = dm0[0] + dm0[1]
            dme0_sf = dme0[0] + dme0[1]
        else:
            dm0_sf, dme0_sf = dm0, dme0

        if fast_h1:
            h1 = self.get_hcore(mol)
            rinv = integrals.int1e_iprinv_dm(mol, dm0_sf)
            charges = mol.atom_charges()
            ecp_atoms = _ecp_atoms(mol)

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
            if fast_h1:
                de[k] += 2 * (np.einsum("xij,ij->x", h1[:, p0:p1], dm0_sf[p0:p1]) - charges[ia] * rinv[ia])
                if ia in ecp_atoms:
                    with mol.with_rinv_at_nucleus(ia):
                        de[k] += 2 * np.einsum("xij,ij->x", mol.intor("ECPscalar_iprinv", comp=3), dm0_sf)
            else:
                de[k] += np.einsum("xij,ij->x", hcore_deriv(ia), dm0_sf)
            de[k] += de2[ia]
            de[k] -= np.einsum("xij,ij->x", s1[:, p0:p1], dme0_sf[p0:p1]) * 2
            de[k] += self._extra_force(ia, locals())

        if log.verbose >= logger.DEBUG:
            log.debug("gradients of electronic part")
            rhf_grad._write(log, mol, de, atmlst)
        return de


class _MojoGradMixin(_MojoGrad1eMixin):
    """Exact two-electron integrals: ``grad2e`` and the derivative J/K matrices from the Mojo engine."""

    def _mojo_jk_ok(self, mol, dm, omega):
        """The Mojo derivative J/K needs real symmetric densities (pyscf also passes others, e.g. in TDHF)."""
        if not self._mojo_2e_ok(mol, omega):
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
        return (
            cls.get_veff is self._pyscf_grad.get_veff
            and cls.get_jk is _MojoGradMixin.get_jk
            and not ({"get_veff", "get_jk"} & self.__dict__.keys())
        )

    def grad_2e(self, dm0, mol=None):
        """d/dR of the two-electron energy at fixed density (natm, 3); ``dm0`` as from ``base.make_rdm1()``."""
        return self._grad_jk(dm0, mol, 1.0)

    def _grad_jk(self, dm0, mol, k_scale):
        """Coulomb and (``k_scale`` times) exchange part of ``grad_2e``."""
        mol = self.mol if mol is None else mol
        dm0 = np.asarray(dm0)
        if self._unrestricted:
            return integrals.grad2e(mol, dm0[0] + dm0[1], dm0, 1.0, k_scale, tol=grad_tol)
        return integrals.grad2e(mol, dm0, dm0, 1.0, 0.5 * k_scale, tol=grad_tol)


class _MojoDFGradMixin(_MojoGrad1eMixin):
    """Density fitting: the DF two-electron gradient (``grad2e_df``) from the Mojo engine."""

    def _auxmol(self):
        with_df = self.base.with_df
        auxmol = with_df.auxmol
        if auxmol is None:
            from pyscf.df import addons

            auxmol = addons.make_auxmol(with_df.mol, with_df.auxbasis)
        return auxmol

    def _direct_2e(self):
        """True for a plain pyscf DF object with the default options and no overridden J/K."""
        from pyscf import df

        base = self.base
        with_df = getattr(base, "with_df", None)
        cls = type(self)
        pyscf_cls = self._pyscf_grad
        return (
            type(with_df) is df.DF
            and not getattr(base, "only_dfj", False)
            and self.auxbasis_response
            and not getattr(with_df, "omega", None)
            and cls.get_veff is pyscf_cls.get_veff
            and cls.get_jk is pyscf_cls.get_jk
            and not ({"get_veff", "get_jk"} & self.__dict__.keys())
            and integrals.available(self._auxmol(), two_electron=True)
        )

    def _extra_force(self, atom_id, envs):
        # pyscf's DF classes add the auxiliary-basis response (and the KS grid response, not used
        # with grad_2e) here; grad_2e includes it
        if type(self).extra_force is self._pyscf_grad.extra_force:
            return 0
        return self.extra_force(atom_id, envs)

    def grad_2e(self, dm0, mol=None):
        """d/dR of the density-fitted two-electron energy at fixed density (natm, 3).

        ``dm0`` as from ``base.make_rdm1()``; its occupied orbitals come from
        the ``mo_coeff``/``mo_occ`` tags (or a decomposition of the density),
        as in pyscf's DF gradient.
        """
        return self._grad_jk(dm0, mol, 1.0)

    def _grad_jk(self, dm0, mol, k_scale):
        """Coulomb and (``k_scale`` times) exchange part of ``grad_2e``; no orbitals are needed without exchange."""
        mol = self.mol if mol is None else mol
        if k_scale != 0:
            orbol, orbor = df_rhf_grad._decompose_rdm1(self, mol, dm0)
            occs = [np.einsum("pi,pi->i", r, o) / np.einsum("pi,pi->i", o, o) for o, r in zip(orbol, orbor)]
        else:
            orbol, occs = [], []
        dm0 = np.asarray(dm0)
        max_memory = max(1000, self.max_memory - lib.current_memory()[0])
        auxmol = self._auxmol()
        if self._unrestricted:
            dm_j, k_factor = dm0[0] + dm0[1], 1.0
        else:
            dm_j, k_factor = dm0, 0.5
        return integrals.grad2e_df(
            mol, auxmol, dm_j, orbol, occs, 1.0, k_factor * k_scale, max_memory=max_memory, tol=grad_tol
        )


class _MojoKSGradMixin:
    """Kohn-Sham: ``grad_2e`` is the Coulomb and scaled exact-exchange term plus the XC term.

    The XC term (at fixed grids) is :func:`mojoscf.dft.grad_xc`: the Mojo
    kernels for ``mojoscf.dft.NumInt`` with LDA/GGA functionals, pyscf's
    ``get_vxc`` otherwise.  Range-separated functionals, non-local
    correlation (``nlc``) and ``grid_response`` keep pyscf's ``get_veff``.
    """

    def _hybrid(self):
        """(supported, exact-exchange fraction) of the functional."""
        mf = self.base
        if self.grid_response or mf.do_nlc():
            return False, 0.0
        omega, _, hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=self.mol.spin)
        return omega == 0, hyb

    def _direct_2e(self):
        return self._hybrid()[0] and super()._direct_2e()

    def grad_2e(self, dm0, mol=None):
        """d/dR of the Coulomb, exact-exchange and XC energies at fixed density (natm, 3)."""
        from . import dft

        mol = self.mol if mol is None else mol
        mf = self.base
        de = self._grad_jk(dm0, mol, self._hybrid()[1])
        grids = rks_grad._initialize_grids(self)[0]
        return de + dft.grad_xc(mf._numint, mol, grids, mf.xc, dm0, spin=int(self._unrestricted))


class Gradients(_MojoGradMixin, rhf_grad.Gradients):
    """RHF nuclear gradients with Mojo derivative integrals (see the module docstring)."""

    _unrestricted = False
    _pyscf_grad = rhf_grad.Gradients


class UGradients(_MojoGradMixin, uhf_grad.Gradients):
    """UHF nuclear gradients with Mojo derivative integrals (see the module docstring)."""

    _unrestricted = True
    _pyscf_grad = uhf_grad.Gradients


class DFGradients(_MojoDFGradMixin, df_rhf_grad.Gradients):
    """Density-fitted RHF nuclear gradients with Mojo derivative integrals (see the module docstring)."""

    _unrestricted = False
    _pyscf_grad = df_rhf_grad.Gradients


class DFUGradients(_MojoDFGradMixin, df_uhf_grad.Gradients):
    """Density-fitted UHF nuclear gradients with Mojo derivative integrals (see the module docstring)."""

    _unrestricted = True
    _pyscf_grad = df_uhf_grad.Gradients


class KSGradients(_MojoKSGradMixin, _MojoGradMixin, rks_grad.Gradients):
    """RKS nuclear gradients with Mojo derivative integrals and XC kernels (see the module docstring)."""

    _unrestricted = False
    _pyscf_grad = rks_grad.Gradients


class UKSGradients(_MojoKSGradMixin, _MojoGradMixin, uks_grad.Gradients):
    """UKS nuclear gradients with Mojo derivative integrals and XC kernels (see the module docstring)."""

    _unrestricted = True
    _pyscf_grad = uks_grad.Gradients


class DFKSGradients(_MojoKSGradMixin, _MojoDFGradMixin, df_rks_grad.Gradients):
    """Density-fitted RKS nuclear gradients with Mojo derivative integrals and XC kernels."""

    _unrestricted = False
    _pyscf_grad = df_rks_grad.Gradients


class DFUKSGradients(_MojoKSGradMixin, _MojoDFGradMixin, df_uks_grad.Gradients):
    """Density-fitted UKS nuclear gradients with Mojo derivative integrals and XC kernels."""

    _unrestricted = True
    _pyscf_grad = df_uks_grad.Gradients


_KS_GRADIENTS = {
    rks_grad.Gradients: KSGradients,
    uks_grad.Gradients: UKSGradients,
    df_rks_grad.Gradients: DFKSGradients,
    df_uks_grad.Gradients: DFUKSGradients,
}


def ks_gradients(g):
    """The Mojo counterpart of a pyscf Kohn-Sham gradient object ``g`` of exactly one of pyscf's RKS/UKS
    classes (plain or density-fitted), else ``g`` itself."""
    cls = _KS_GRADIENTS.get(type(g))
    return g if cls is None else cls(g.base)
