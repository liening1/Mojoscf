"""Analytical nuclear Hessians (pyscf.hessian) with the Mojo kernels.

pyscf's RKS/UKS Hessian spends most of its time in Python loops over grid
blocks for the exchange-correlation terms (``_get_vxc_diag``,
``_get_vxc_deriv2``, ``_get_vxc_deriv1``), in the coupled-perturbed
equations and in the second-derivative integrals.  This module replaces
those pieces while keeping pyscf's driver and conventions.
"""
from __future__ import annotations

import numpy as np

from . import integrals
from ._backend import get_extension, worker_blas


def _ao_atoms(mol):
    """Atom index of every AO (int64)."""
    out = np.empty(mol.nao_nr(), dtype=np.int64)
    for ia, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        out[p0:p1] = ia
    return out


def xc_partial_hess(ni, mol, grids, xc_code, mo_coeff, mo_occ):
    """XC part of the partial Hessian at fixed orbitals, (natm, natm, 3, 3), or None for other functionals.

    The sum pyscf's ``partial_hess_elec`` forms from ``_get_vxc_diag`` and
    ``_get_vxc_deriv2`` (contracted with the density, no grid response),
    computed in one pass over the grid (``xc_hess_core``) for LDA and GGA,
    restricted (``mo_coeff`` (nao, nmo)) or unrestricted ((2, nao, nmo)).
    """
    from . import dft

    kind = dft._kind(ni, xc_code)
    if kind not in (0, 1) or not dft._fxc_ok(ni, mol, xc_code):
        return None
    mo_coeff = np.asarray(mo_coeff)
    coords, weights = dft._grid(grids)
    if mo_coeff.ndim == 2:
        occ = np.asarray(mo_occ)
        rho = dft._rho_orbitals(mol, coords, kind, [mo_coeff], [occ])[0]
        vxc, fxc = ni.eval_xc_eff(xc_code, rho[0] if kind == 0 else rho, deriv=2, xctype=ni._xc_type(xc_code))[1:3]
        c = mo_coeff[:, occ > 0]
        dms = ((c * occ[occ > 0]) @ c.T)[None]
        nspin = 1
    else:
        occs = [np.asarray(o) for o in mo_occ]
        rho = dft._rho_orbitals(mol, coords, kind, list(mo_coeff), occs)
        r = (rho[0, 0], rho[1, 0]) if kind == 0 else (rho[0], rho[1])
        vxc, fxc = ni.eval_xc_eff(xc_code, r, deriv=2, xctype=ni._xc_type(xc_code), spin=1)[1:3]
        dms = np.array([(c[:, o > 0] * o[o > 0]) @ c[:, o > 0].T for c, o in zip(mo_coeff, occs)])
        nspin = 2
    nvar = (1, 4)[kind]
    ngrid = coords.shape[0]
    vxc = np.ascontiguousarray(np.asarray(vxc, dtype=np.float64).reshape(nspin * nvar, ngrid))
    fxc = np.ascontiguousarray(np.asarray(fxc, dtype=np.float64).reshape(nspin * nvar, nspin * nvar, ngrid))
    de2 = np.empty((mol.natm, mol.natm, 3, 3))
    path, prefix = worker_blas()
    get_extension().xc_hess(integrals.basis_tables(mol), coords, np.ascontiguousarray(weights, dtype=np.float64),
                            int(kind), np.ascontiguousarray(dms, dtype=np.float64), vxc, fxc, _ao_atoms(mol), de2,
                            path, prefix)
    return de2


class _ZeroXC:
    """Within the block, pyscf's ``_get_vxc_diag``/``_get_vxc_deriv2`` (RKS or UKS) return zeros,
    so that pyscf's ``partial_hess_elec`` forms everything but the XC term (the long-range
    exchange it adds to those arrays included)."""

    def __init__(self, mol, unrestricted):
        from pyscf.hessian import rks as rks_hess
        from pyscf.hessian import uks as uks_hess

        self.module = uks_hess if unrestricted else rks_hess
        self.mol = mol
        self.unrestricted = unrestricted

    def __enter__(self):
        nao, natm = self.mol.nao_nr(), self.mol.natm
        two = self.unrestricted

        def diag(*args, **kwargs):
            z = np.zeros((3, 3, nao, nao))
            return (z, np.zeros_like(z)) if two else z

        def deriv2(*args, **kwargs):
            z = np.zeros((natm, 3, 3, nao, nao))     # untouched zero pages cost no memory
            return (z, np.zeros_like(z)) if two else z

        self.saved = (self.module._get_vxc_diag, self.module._get_vxc_deriv2)
        self.module._get_vxc_diag = diag
        self.module._get_vxc_deriv2 = deriv2
        return self

    def __exit__(self, *exc):
        self.module._get_vxc_diag, self.module._get_vxc_deriv2 = self.saved
        return False


class _MojoHessMixin:
    """In front of pyscf's RKS/UKS Hessian classes (DF or not): the XC terms from the Mojo kernels."""

    __name_mixin__ = "Mojo"

    def partial_hess_elec(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None, max_memory=4000,
                          verbose=None):
        mf = self.base
        mol = self.mol
        if mo_coeff is None:
            mo_coeff = mf.mo_coeff
        if mo_occ is None:
            mo_occ = mf.mo_occ
        xc = None
        if not getattr(self, "grid_response", False):
            grids = self.grids if getattr(self, "grids", None) is not None else mf.grids
            xc = xc_partial_hess(mf._numint, mol, grids, mf.xc, mo_coeff, mo_occ)
        if xc is None:
            return super().partial_hess_elec(mo_energy, mo_coeff, mo_occ, atmlst, max_memory, verbose)
        with _ZeroXC(mol, np.asarray(mo_coeff).ndim == 3):
            de2 = super().partial_hess_elec(mo_energy, mo_coeff, mo_occ, atmlst, max_memory, verbose)
        atm = list(range(mol.natm)) if atmlst is None else list(atmlst)
        return de2 + xc[np.ix_(atm, atm)]


def accelerate(h):
    """Give a pyscf RKS/UKS Hessian object (``mf.Hessian()``, DF or not) the Mojo kernels, in place; returns it."""
    if not isinstance(h, _MojoHessMixin):
        from pyscf import lib

        lib.set_class(h, (_MojoHessMixin, type(h)))
    return h
