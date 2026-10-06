"""Kohn-Sham DFT with the Mojo kernels: exchange-correlation integration, J/K and nuclear gradients.

pyscf's :class:`~pyscf.dft.numint.NumInt` evaluates the basis functions on
blocks of grid points, the density, the functional (libxc) and the XC
potential matrix.  :class:`NumInt` here keeps pyscf's grids and libxc and
replaces the rest for LDA and GGA functionals (``_mojo/numint.mojo``): one
Mojo pass computes the densities (and gradients) on all grid points, pyscf's
``eval_xc_eff`` evaluates the functional on the whole grid at once, and a
second Mojo pass assembles the potential matrix.  Both passes evaluate only
the shells significant on each block of 128 points and do the contractions
as per-block GEMMs.  The XC part of the nuclear gradient (pyscf's
``grad.rks.get_vxc``/``grad.uks.get_vxc``, used by every RKS/UKS gradient
class) runs the same way for :class:`NumInt` objects: one density pass and
one pass over the basis-function second derivatives; the gradient classes of
:mod:`mojoscf.grad` contract it with the density inside that pass
(:func:`grad_xc`).  Meta-GGA, non-symmetric densities, response kernels
(``nr_rks_fxc`` ...) and the grid response of the gradient keep pyscf's code.

:func:`accelerate` gives a pyscf RKS/UKS object this ``NumInt``, J/K from
mojoscf (density fitting, in-core ERIs or integral-direct), the Mojo
gradients and, for QM/MM, the Mojo MM-charge terms; the SCF loop stays
pyscf's.

>>> from pyscf import dft
>>> mf = mojoscf.dft.accelerate(dft.RKS(mol, xc="b3lyp"))   # or mf._numint = mojoscf.dft.NumInt()
>>> mf.kernel()
>>> g = mf.nuc_grad_method().kernel()
"""
from __future__ import annotations

import numpy as np
from pyscf.dft import numint as pyscf_numint

from . import integrals
from ._backend import get_extension, worker_blas


def supported(mol) -> bool:
    """True if the Mojo basis-function code handles ``mol`` (any molecule the integral engine reads)."""
    return mol.nbas > 0 and integrals.available(mol, allow_ecp=True)


def _passes_ok(ni, mol, xc_code, dms, hermi) -> bool:
    return (
        ni._xc_type(xc_code) in ("LDA", "GGA")
        and hermi == 1
        and np.isrealobj(dms)
        and integrals.engine() == "mojo"
        and supported(mol)
    )


def eval_ao(mol, coords, deriv=0):
    """AO values (deriv 0, shape (ngrid, nao)) or values and derivatives up to order deriv (1 or 2:
    (4, ngrid, nao) or (10, ngrid, nao)), as ``mol.eval_gto``."""
    if deriv not in (0, 1, 2):
        raise ValueError(f"deriv {deriv}: only 0, 1 and 2")
    coords = np.ascontiguousarray(coords, dtype=np.float64).reshape(-1, 3)
    out = np.empty(((1, 4, 10)[deriv], coords.shape[0], mol.nao_nr()))
    get_extension().eval_ao(integrals.basis_tables(mol), coords, int(deriv), out)
    return out if deriv else out[0]


def _rho(mol, coords, deriv, dms):
    """(nset, ncomp, ngrid) densities (and gradients) of the symmetric ``dms`` (nset, nao, nao)."""
    rho = np.empty((dms.shape[0], 4 if deriv else 1, coords.shape[0]))
    path, prefix = worker_blas()
    get_extension().xc_rho(integrals.basis_tables(mol), coords, int(deriv), dms, rho, path, prefix)
    return rho


def _vmat(mol, coords, deriv, wv):
    """(nset, nao, nao) sum_p phi(p) (sum_c wv_c(p) phi_c(p))^T, symmetrised (the caller halves wv_0)."""
    nao = mol.nao_nr()
    v = np.empty((wv.shape[0], nao, nao))
    path, prefix = worker_blas()
    get_extension().xc_vmat(integrals.basis_tables(mol), coords, int(deriv), np.ascontiguousarray(wv), v,
                            path, prefix)
    return v + v.transpose(0, 2, 1)


def _xc_grad(mol, coords, gga, wv):
    """(nset, 3, nao, nao) XC gradient matrices of pyscf's ``grad.rks.get_vxc`` before its sign flip."""
    nao = mol.nao_nr()
    v = np.empty((wv.shape[0], 3, nao, nao))
    path, prefix = worker_blas()
    get_extension().xc_grad(integrals.basis_tables(mol), coords, int(gga), np.ascontiguousarray(wv), v,
                            path, prefix)
    return v


def _grid(grids):
    if grids.coords is None:
        grids.build()
    return np.ascontiguousarray(grids.coords, dtype=np.float64), np.asarray(grids.weights, dtype=np.float64)


class NumInt(pyscf_numint.NumInt):
    """pyscf's ``NumInt`` with ``nr_rks``/``nr_uks`` for LDA and GGA functionals from the Mojo kernels."""

    def nr_rks(self, mol, grids, xc_code, dms, relativity=0, hermi=1, max_memory=2000, verbose=None):
        if not _passes_ok(self, mol, xc_code, dms, hermi):
            return super().nr_rks(mol, grids, xc_code, dms, relativity, hermi, max_memory, verbose)
        xctype = self._xc_type(xc_code)
        deriv = 0 if xctype == "LDA" else 1
        nao = mol.nao_nr()
        dm = np.asarray(dms, dtype=np.float64)
        single = dm.ndim == 2
        dm = np.ascontiguousarray(dm.reshape(-1, nao, nao))
        nset = dm.shape[0]
        coords, weights = _grid(grids)
        rho = _rho(mol, coords, deriv, dm)
        nelec = np.zeros(nset)
        excsum = np.zeros(nset)
        wv = np.empty_like(rho)
        for i in range(nset):
            r = rho[i, 0] if deriv == 0 else rho[i]
            exc, vxc = self.eval_xc_eff(xc_code, r, deriv=1, xctype=xctype, spin=0)[:2]
            den = rho[i, 0] * weights
            nelec[i] = den.sum()
            excsum[i] = np.dot(den, exc)
            wv[i] = weights * np.asarray(vxc).reshape(-1, weights.size)
            wv[i, 0] *= 0.5                     # V + V^T below
        vmat = _vmat(mol, coords, deriv, wv)
        if single:
            return nelec[0], excsum[0], vmat[0]
        return nelec, excsum, vmat

    def nr_uks(self, mol, grids, xc_code, dms, relativity=0, hermi=1, max_memory=2000, verbose=None):
        if not _passes_ok(self, mol, xc_code, dms, hermi):
            return super().nr_uks(mol, grids, xc_code, dms, relativity, hermi, max_memory, verbose)
        xctype = self._xc_type(xc_code)
        deriv = 0 if xctype == "LDA" else 1
        nao = mol.nao_nr()
        dma, dmb = pyscf_numint._format_uks_dm(dms)
        single = np.asarray(dma).ndim == 2
        dma = np.asarray(dma, dtype=np.float64).reshape(-1, nao, nao)
        dmb = np.asarray(dmb, dtype=np.float64).reshape(-1, nao, nao)
        nset = dma.shape[0]
        coords, weights = _grid(grids)
        rho = _rho(mol, coords, deriv, np.ascontiguousarray(np.concatenate([dma, dmb])))
        nelec = np.zeros((2, nset))
        excsum = np.zeros(nset)
        wv = np.empty_like(rho)
        for i in range(nset):
            ra, rb = rho[i], rho[nset + i]
            r = (ra[0], rb[0]) if deriv == 0 else (ra, rb)
            exc, vxc = self.eval_xc_eff(xc_code, r, deriv=1, xctype=xctype, spin=1)[:2]
            den_a = ra[0] * weights
            den_b = rb[0] * weights
            nelec[0, i] = den_a.sum()
            nelec[1, i] = den_b.sum()
            excsum[i] = np.dot(den_a, exc) + np.dot(den_b, exc)
            vxc = np.asarray(vxc).reshape(2, -1, weights.size)
            wv[i] = weights * vxc[0]
            wv[nset + i] = weights * vxc[1]
            wv[i, 0] *= 0.5
            wv[nset + i, 0] *= 0.5
        vmat = _vmat(mol, coords, deriv, wv).reshape(2, nset, nao, nao)
        if single:
            return nelec[:, 0], excsum[0], vmat[:, 0]
        return nelec, excsum, vmat


def _grad_weights(ni, xc_code, rho, weights, spin):
    """Weighted XC potential on the grid in pyscf's gradient convention (GGA: w_0 halved).

    ``rho`` (nset * (1 + spin), ncomp, ngrid); returns the same shape.
    """
    xctype = ni._xc_type(xc_code)
    wv = np.empty_like(rho)
    nset = rho.shape[0] // (1 + spin)
    for i in range(nset):
        if spin:
            ra, rb = rho[i], rho[nset + i]
            r = (ra[0], rb[0]) if xctype == "LDA" else (ra, rb)
        else:
            r = rho[i, 0] if xctype == "LDA" else rho[i]
        vxc = np.asarray(ni.eval_xc_eff(xc_code, r, deriv=1, xctype=xctype, spin=spin)[1])
        vxc = vxc.reshape(1 + spin, -1, weights.size)
        for k in range(1 + spin):
            j = k * nset + i
            wv[j] = weights * vxc[k]
            if xctype == "GGA":
                wv[j, 0] *= 0.5
    return wv


def grad_rks_vxc(ni, mol, grids, xc_code, dms, relativity=0, hermi=1, max_memory=2000, verbose=None):
    """``pyscf.grad.rks.get_vxc`` (no grid response) for LDA and GGA, from the Mojo kernels: (None, -vmat)."""
    xctype = ni._xc_type(xc_code)
    gga = xctype == "GGA"
    nao = mol.nao_nr()
    dm = np.ascontiguousarray(np.asarray(dms, dtype=np.float64).reshape(-1, nao, nao))
    coords, weights = _grid(grids)
    rho = _rho(mol, coords, int(gga), dm)
    vmat = _xc_grad(mol, coords, gga, _grad_weights(ni, xc_code, rho, weights, 0))
    if vmat.shape[0] == 1:
        vmat = vmat[0]
    return None, -vmat


def grad_uks_vxc(ni, mol, grids, xc_code, dms, relativity=0, hermi=1, max_memory=2000, verbose=None):
    """``pyscf.grad.uks.get_vxc`` (no grid response) for LDA and GGA, from the Mojo kernels: (None, -vmat)."""
    xctype = ni._xc_type(xc_code)
    gga = xctype == "GGA"
    nao = mol.nao_nr()
    dm = np.ascontiguousarray(np.asarray(dms, dtype=np.float64).reshape(2, nao, nao))
    coords, weights = _grid(grids)
    rho = _rho(mol, coords, int(gga), dm)
    vmat = _xc_grad(mol, coords, gga, _grad_weights(ni, xc_code, rho, weights, 1))
    return None, -vmat


def grad_xc(ni, mol, grids, xc_code, dms, spin=0):
    """XC term of the nuclear gradient at fixed grids, shape (natm, 3).

    What pyscf's gradient adds from the XC part of ``get_veff``:
    ``2 sum_{mu on A} (vxc[:, mu] * D[mu]).sum()`` with ``vxc`` from
    ``grad.rks.get_vxc`` (``spin=0``, ``dms`` the total density) or
    ``grad.uks.get_vxc`` (``spin=1``, ``dms`` (2, nao, nao), summed over
    spins).  For :class:`NumInt` objects with LDA/GGA functionals the Mojo
    kernels contract the derivative matrices with the density as they are
    produced; otherwise pyscf's ``get_vxc`` provides the matrices.
    """
    from pyscf.grad import rks as rks_grad
    from pyscf.grad import uks as uks_grad

    nao = mol.nao_nr()
    dm = np.asarray(dms, dtype=np.float64).reshape(-1, nao, nao)
    if isinstance(ni, NumInt) and _passes_ok(ni, mol, xc_code, dm, 1):
        gga = ni._xc_type(xc_code) == "GGA"
        coords, weights = _grid(grids)
        dm = np.ascontiguousarray(dm)
        rho = _rho(mol, coords, int(gga), dm)
        wv = _grad_weights(ni, xc_code, rho, weights, spin)
        de = np.empty((mol.natm, 3))
        path, prefix = worker_blas()
        get_extension().xc_grad_dm(integrals.basis_tables(mol), coords, int(gga), np.ascontiguousarray(wv), dm, de,
                                   path, prefix)
        return de
    vxc = (uks_grad if spin else rks_grad).get_vxc(ni, mol, grids, xc_code, dms)[1]
    vxc = np.asarray(vxc).reshape(-1, 3, nao, nao)
    de = np.zeros((mol.natm, 3))
    for ia, (p0, p1) in enumerate(mol.aoslice_by_atom()[:, 2:]):
        de[ia] = 2 * np.einsum("sxij,sij->x", vxc[:, :, p0:p1], dm[:, p0:p1])
    return de


def _install_grad_vxc():
    """Route pyscf's XC gradient (``grad.rks.get_vxc``, ``grad.uks.get_vxc``) to the Mojo kernels for
    :class:`NumInt` objects; pyscf's code runs for every other ``NumInt`` and for unsupported cases.

    pyscf's ``get_veff`` (also the density-fitted ones) looks these functions
    up in their modules at call time, so wrapping them there is enough.
    """
    from pyscf.grad import rks as rks_grad
    from pyscf.grad import uks as uks_grad

    def wrap(module, mojo_fn, nset_ok):
        orig = module.get_vxc
        if getattr(orig, "_mojoscf_orig", None) is not None:
            return

        def get_vxc(ni, mol, grids, xc_code, dms, relativity=0, hermi=1, max_memory=2000, verbose=None):
            if isinstance(ni, NumInt) and relativity == 0 and nset_ok(mol, dms) and _passes_ok(ni, mol, xc_code, dms, hermi):
                return mojo_fn(ni, mol, grids, xc_code, dms, relativity, hermi, max_memory, verbose)
            return orig(ni, mol, grids, xc_code, dms, relativity, hermi, max_memory, verbose)

        get_vxc.__doc__ = orig.__doc__
        get_vxc._mojoscf_orig = orig
        module.get_vxc = get_vxc

    nao_ok = lambda mol, dms: np.asarray(dms).shape[-1] == mol.nao_nr()
    wrap(rks_grad, grad_rks_vxc, nao_ok)
    wrap(uks_grad, grad_uks_vxc, lambda mol, dms: nao_ok(mol, dms) and np.asarray(dms).size == 2 * mol.nao_nr() ** 2)


_install_grad_vxc()


def _incore_cderi(mf, mol, dm, hermi, omega):
    """pyscf's in-core DF tensor of ``mf`` for :func:`mojoscf.kernels.df_jk`, or None to use pyscf's get_jk.

    A missing tensor is built with the Mojo integrals when pyscf would keep
    it in core (``integrals.build_df``).
    """
    from pyscf.df import df as pyscf_df
    from pyscf.df import df_jk

    if not isinstance(mf, df_jk._DFHF) or (mol is not None and mol is not mf.mol):
        return None
    if omega or hermi != 1 or not np.isrealobj(dm) or getattr(mf, "only_dfj", False):
        return None
    with_df = mf.with_df
    if type(with_df) is not pyscf_df.DF:
        return None
    if with_df._cderi is None and not integrals.build_df(with_df):
        with_df.build()
    cderi = with_df._cderi
    nao = mf.mol.nao_nr()
    if isinstance(cderi, np.ndarray) and cderi.ndim == 2 and cderi.dtype == np.float64 and cderi.shape[1] == nao * (nao + 1) // 2:
        return cderi
    return None


def _exact_jk(mf, mol, dm, hermi, with_j, with_k, omega):
    """J/K with exact integrals from mojoscf for a non-DF object whose ``get_jk`` is pyscf's ``SCF.get_jk``:

    in-core 8-fold ERIs (built with the Mojo engine when pyscf would keep them
    in core) contracted by :func:`mojoscf.kernels.jk_s8`, else integral-direct
    J/K (:func:`mojoscf.integrals.get_jk`).  None when that does not apply.
    """
    from pyscf.df import df_jk

    if isinstance(mf, df_jk._DFHF) or (mol is not None and mol is not mf.mol):
        return None
    if omega or hermi != 1 or not np.isrealobj(dm) or "get_jk" in vars(mf):
        return None
    mro = type(mf).__mro__
    nxt = next((c for c in mro[mro.index(_MojoKSHook) + 1:] if "get_jk" in c.__dict__), None)
    if nxt is None or nxt.__module__ not in ("pyscf.scf.hf", "pyscf.scf.uhf"):
        return None
    mol = mf.mol
    if integrals.engine() != "mojo" or not integrals.available(mol, two_electron=True):
        return None
    from . import kernels

    nao = mol.nao_nr()
    npair = nao * (nao + 1) // 2
    dm = np.asarray(dm, dtype=np.float64)
    eri = getattr(mf, "_eri", None)
    if eri is None and (mol.incore_anyway or mf._is_mem_enough()):
        eri = mf._eri = integrals.int2e_s8(mol)
    if eri is not None:
        if isinstance(eri, np.ndarray) and eri.dtype == np.float64 and eri.size == npair * (npair + 1) // 2:
            return kernels.jk_s8(eri.reshape(-1), dm, with_j, with_k)
        return None
    return integrals.get_jk(mol, dm, with_j, with_k, direct_scf_tol=mf.direct_scf_tol)


class _MojoKSHook:
    """In front of a pyscf Kohn-Sham class (:func:`accelerate`): J/K and the eigensolver from mojoscf."""

    __name_mixin__ = "Mojo"

    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True, omega=None):
        if dm is None:
            dm = self.make_rdm1()
        cderi = _incore_cderi(self, mol, dm, hermi, omega)
        if cderi is None:
            jk = _exact_jk(self, mol, dm, hermi, with_j, with_k, omega)
            if jk is not None:
                return jk
            return super().get_jk(mol, dm, hermi, with_j, with_k, omega)
        from . import kernels

        mo_coeff = getattr(dm, "mo_coeff", None)
        mo_occ = getattr(dm, "mo_occ", None)
        if mo_coeff is None or mo_occ is None or np.asarray(mo_coeff).ndim != np.asarray(dm).ndim:
            mo_coeff = mo_occ = None
        return kernels.df_jk(cderi, np.asarray(dm), mo_coeff, mo_occ, with_j, with_k)

    def get_j(self, mol=None, dm=None, hermi=1, omega=None):
        return self.get_jk(mol, dm, hermi, True, False, omega)[0]

    def get_k(self, mol=None, dm=None, hermi=1, omega=None):
        return self.get_jk(mol, dm, hermi, False, True, omega)[1]

    def nuc_grad_method(self):
        """Nuclear gradients with Mojo derivative integrals and XC kernels (``mojoscf.grad.KSGradients`` ...)."""
        from .scf import _mojo_grad_method

        return _mojo_grad_method(self, _MojoKSHook)

    Gradients = nuc_grad_method

    def _eigh(self, h, s, overwrite=False, x=None):
        from . import kernels
        from .scf import _is_real

        if not _is_real(h, s, x):
            return super()._eigh(h, s, overwrite, x)
        if x is None:
            return kernels.eigh(h, s)
        return kernels.eigh(h, x=x)


def accelerate(mf):
    """Give a pyscf Kohn-Sham object the Mojo kernels, in place; returns ``mf``.

    * ``mf._numint``: :class:`NumInt` (the settings of the old one, ``omega``,
      ``libxc`` and cutoffs, are kept);
    * density-fitted objects: J/K from :func:`mojoscf.kernels.df_jk` on the
      in-core DF tensor, which is built with the Mojo integrals; others: J/K
      from in-core ERIs built by the Mojo engine (:func:`mojoscf.kernels.jk_s8`)
      or integral-direct (:func:`mojoscf.integrals.get_jk`);
    * nuclear gradients: :class:`mojoscf.grad.KSGradients` and its UKS and
      density-fitted variants (Mojo derivative integrals and XC kernels);
    * the eigensolver: :func:`mojoscf.kernels.eigh`;
    * QM/MM objects: the Mojo MM-charge terms (:func:`mojoscf.qmmm.attach`).

    The SCF loop itself stays pyscf's.
    """
    old = getattr(mf, "_numint", None)
    if old is None:
        raise TypeError(f"{type(mf).__name__} has no _numint: not a Kohn-Sham object")
    if not isinstance(old, NumInt):
        if type(old) is not pyscf_numint.NumInt:
            raise TypeError(f"mojoscf.dft.accelerate supports pyscf's NumInt, not {type(old).__name__}")
        ni = NumInt()
        ni.__dict__.update(old.__dict__)
        mf._numint = ni
    if not isinstance(mf, _MojoKSHook):
        from pyscf import lib

        lib.set_class(mf, (_MojoKSHook, type(mf)))
    from pyscf.qmmm import itrf

    if isinstance(mf, itrf.QMMMSCF):
        from . import qmmm

        qmmm.attach(mf)
    return mf
