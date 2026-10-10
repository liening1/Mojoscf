"""Kohn-Sham DFT with the Mojo kernels: exchange-correlation integration, J/K and nuclear gradients.

pyscf's :class:`~pyscf.dft.numint.NumInt` evaluates the basis functions on
blocks of grid points, the density, the functional (libxc) and the XC
potential matrix.  :class:`NumInt` here keeps pyscf's grids and libxc and
replaces the rest for LDA, GGA and meta-GGA functionals (``_mojo/numint.mojo``;
not those using the laplacian): one Mojo pass computes the densities (with
gradients and kinetic-energy densities) on all grid points, pyscf's
``eval_xc1`` (libxc) evaluates the functional on the whole grid at once, and
a second Mojo pass assembles the potential matrix.  Both passes evaluate only
the shells significant on each block of 128 points and do the contractions
as per-block GEMMs.  The XC part of the nuclear gradient (pyscf's
``grad.rks.get_vxc``/``grad.uks.get_vxc``, used by every RKS/UKS gradient
class) runs the same way for :class:`NumInt` objects: one density pass and
one pass over the basis-function second derivatives (LDA and GGA; meta-GGA
keeps pyscf's ``get_vxc``); the gradient classes of :mod:`mojoscf.grad`
contract it with the density inside that pass (:func:`grad_xc`, meta-GGA
included).  The second-order kernels of linear response (``cache_xc_kernel``,
``nr_rks_fxc``, ``nr_rks_fxc_st``, ``nr_uks_fxc``: TDDFT, CPHF) take the
densities from the same pass and contract the kernel in one fused pass
(:func:`fxc_matrices`; for transition densities from their occupied-virtual
factors, see :mod:`mojoscf.tdscf`).  Laplacian meta-GGAs, non-symmetric
densities in the SCF and the grid response of the gradient keep pyscf's code.

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
from pyscf.df import df as pyscf_df
from pyscf.dft import numint as pyscf_numint

from . import integrals
from ._backend import get_extension, worker_blas


def supported(mol) -> bool:
    """True if the Mojo basis-function code handles ``mol`` (any molecule the integral engine reads)."""
    return mol.nbas > 0 and integrals.available(mol, allow_ecp=True)


_KINDS = {"LDA": 0, "GGA": 1, "MGGA": 2}


def _kind(ni, xc_code):
    """0 (LDA), 1 (GGA), 2 (meta-GGA without the laplacian) or None (other functionals)."""
    kind = _KINDS.get(ni._xc_type(xc_code))
    if kind == 2 and any(x in xc_code.upper() for x in ("CC06", "CS", "BR89", "MK00")):
        return None                         # laplacian meta-GGAs (pyscf rejects them too)
    return kind


def _passes_ok(ni, mol, xc_code, dms, hermi, max_kind=2) -> bool:
    kind = _kind(ni, xc_code)
    return (
        kind is not None
        and kind <= max_kind
        and hermi == 1
        and np.isrealobj(dms)
        and integrals.engine() == "mojo"
        and supported(mol)
    )


def eval_ao(mol, coords, deriv=0):
    """AO values (deriv 0, shape (ngrid, nao)) or values and derivatives up to order deriv (1, 2 or 3:
    (4, ngrid, nao), (10, ngrid, nao) or (20, ngrid, nao)), as ``mol.eval_gto``."""
    if deriv not in (0, 1, 2, 3):
        raise ValueError(f"deriv {deriv}: only 0 to 3")
    coords = np.ascontiguousarray(coords, dtype=np.float64).reshape(-1, 3)
    out = np.empty(((1, 4, 10, 20)[deriv], coords.shape[0], mol.nao_nr()))
    get_extension().eval_ao(integrals.basis_tables(mol), coords, int(deriv), out)
    return out if deriv else out[0]


def _orbitals(dm, nset, nao):
    """Occupied orbitals of the density from pyscf's ``mo_coeff``/``mo_occ`` tags, as
    ((nset, nao, norb), (nset, norb)) zero-padded arrays, or None without (consistent) tags."""
    mo_coeff = getattr(dm, "mo_coeff", None)
    mo_occ = getattr(dm, "mo_occ", None)
    if mo_coeff is None or mo_occ is None:
        return None
    c = np.asarray(mo_coeff)
    n = np.asarray(mo_occ)
    if not (np.isrealobj(c) and np.isrealobj(n)):
        return None
    c = c.reshape(-1, *c.shape[-2:])
    n = n.reshape(-1, n.shape[-1])
    if c.shape[0] != nset or n.shape[0] != nset or c.shape[1] != nao or c.shape[2] != n.shape[1]:
        return None
    keep = [np.flatnonzero(n[i]) for i in range(nset)]
    norb = max(len(k) for k in keep)
    orbs = np.zeros((nset, nao, norb))
    occs = np.zeros((nset, norb))
    for i, k in enumerate(keep):
        orbs[i, :, : len(k)] = c[i][:, k]
        occs[i, : len(k)] = n[i][k]
    return orbs, occs


def _rho(mol, coords, kind, dms, orbitals=None):
    """(nset, ncomp, ngrid) densities of the symmetric ``dms`` (nset, nao, nao): rho (kind 0), with its
    gradient (1, GGA) and tau (2, meta-GGA; pyscf's 1/2 sum |grad psi|^2).

    ``orbitals`` (from :func:`_orbitals`) describe the same densities; the
    kernel uses them where that is cheaper.
    """
    nset, nao = dms.shape[0], dms.shape[-1]
    rho = np.empty((nset, (1, 4, 5)[kind], coords.shape[0]))
    orbs, occs = orbitals if orbitals is not None else (np.zeros((nset, nao, 0)), np.zeros((nset, 0)))
    path, prefix = worker_blas()
    get_extension().xc_rho(integrals.basis_tables(mol), coords, int(kind), dms, np.ascontiguousarray(orbs),
                           np.ascontiguousarray(occs), rho, path, prefix)
    return rho


def _vmat(mol, coords, kind, wv):
    """(nset, nao, nao) sum_p phi(p) (sum_c wv_c(p) phi_c(p))^T (meta-GGA: + sum_c d_c phi (wv_4 d_c phi)^T),
    symmetrised (the caller halves wv_0 and quarters wv_4)."""
    nao = mol.nao_nr()
    v = np.empty((wv.shape[0], nao, nao))
    path, prefix = worker_blas()
    get_extension().xc_vmat(integrals.basis_tables(mol), coords, int(kind), np.ascontiguousarray(wv), v,
                            path, prefix)
    return v + v.transpose(0, 2, 1)


def _xc_grad(mol, coords, kind, wv):
    """(nset, 3, nao, nao) XC gradient matrices of pyscf's ``grad.rks.get_vxc`` before its sign flip
    (``kind`` 0 LDA, 1 GGA, 2 meta-GGA; ``wv`` in pyscf's convention, see :func:`_grad_weights`)."""
    nao = mol.nao_nr()
    v = np.empty((wv.shape[0], 3, nao, nao))
    path, prefix = worker_blas()
    get_extension().xc_grad(integrals.basis_tables(mol), coords, int(kind), np.ascontiguousarray(wv), v,
                            path, prefix)
    return v


def _wsum(a, b) -> float:
    """sum_p a_p b_p without BLAS: a threaded ``ddot`` over the grid would leave OpenBLAS threads
    spinning (~0.1 s) while the next Mojo pass runs."""
    return float(np.einsum("p,p->", a, np.asarray(b).ravel()))


def _halve(w):
    """pyscf's weights for the symmetrised matrix V + V^T (in place): w_0 / 2, and w_tau / 4 (the
    tau term sum_c d_c phi (w_tau d_c phi)^T / 2 is symmetric already)."""
    w[0] *= 0.5
    if w.shape[0] == 5:
        w[4] *= 0.25


def _stock_eval_xc_eff(ni) -> bool:
    """True if ``ni.eval_xc_eff`` is pyscf's (``eval_xc1`` and the derivative transform), which
    :func:`_potential_weights` reproduces to first order."""
    return (type(ni).eval_xc_eff is pyscf_numint.LibXCMixin.eval_xc_eff
            and "eval_xc_eff" not in getattr(ni, "__dict__", {}))


def _potential_weights(ni, xc_code, xctype, kind, rho, weights, spin):
    """The functional on the whole grid and the weights of the potential matrix: ``(exc, wv)``.

    ``rho`` is (nvar, ngrid) for ``spin`` 0 and (2, nvar, ngrid) for 1 (LDA:
    (ngrid,) and (2, ngrid)); ``wv`` = weights * v, v pyscf's first-order
    derivative tensor (``eval_xc_eff``; (nvar, ngrid) or (2, nvar, ngrid)),
    with w_0 halved and w_tau quartered (:func:`_halve`).  pyscf's
    ``eval_xc_eff`` evaluates ``eval_xc1`` and then transforms the derivatives
    with respect to sigma by a general tensor routine; the first-order terms
    are formed here directly (2 v_sigma grad rho, or 2 v_aa grad rho_a +
    v_ab grad rho_b), which saves passes over the grid and copies of the
    densities.
    """
    ngrid = weights.size
    nvar = (1, 4, 5)[kind]
    if not _stock_eval_xc_eff(ni):
        exc, vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype, spin=spin)[:2]
        wv = weights * np.asarray(vxc).reshape((nvar, ngrid) if spin == 0 else (2, nvar, ngrid))
        for w in [wv] if spin == 0 else wv:
            _halve(w)
        return exc, wv
    out = ni.eval_xc1(xc_code, rho, spin, 1, ni.omega)
    if spin == 0:
        wv = np.empty((nvar, ngrid))
        np.multiply(out[1], weights, out=wv[0])
        wv[0] *= 0.5
        if kind > 0:
            g = out[2] * weights
            g *= 2.0
            np.multiply(rho[1:4], g, out=wv[1:4])
        if kind == 2:
            np.multiply(out[3], weights, out=wv[4])
            wv[4] *= 0.25
        return out[0], wv
    wv = np.empty((2, nvar, ngrid))
    for s in range(2):
        np.multiply(out[1 + s], weights, out=wv[s, 0])
        wv[s, 0] *= 0.5
    if kind > 0:
        ga, gb = rho[0, 1:4], rho[1, 1:4]
        waa = out[3] * weights
        waa *= 2.0
        wab = out[4] * weights
        wbb = out[5] * weights
        wbb *= 2.0
        np.multiply(ga, waa, out=wv[0, 1:4])
        wv[0, 1:4] += gb * wab
        np.multiply(gb, wbb, out=wv[1, 1:4])
        wv[1, 1:4] += ga * wab
    if kind == 2:
        for s in range(2):
            np.multiply(out[6 + s], weights, out=wv[s, 4])
            wv[s, 4] *= 0.25
    return out[0], wv


def _grid(grids):
    if grids.coords is None:
        grids.build()
    return np.ascontiguousarray(grids.coords, dtype=np.float64), np.asarray(grids.weights, dtype=np.float64)


class NumInt(pyscf_numint.NumInt):
    """pyscf's ``NumInt`` with ``nr_rks``/``nr_uks`` for LDA, GGA and meta-GGA functionals from the Mojo kernels."""

    def nr_rks(self, mol, grids, xc_code, dms, relativity=0, hermi=1, max_memory=2000, verbose=None):
        if not _passes_ok(self, mol, xc_code, dms, hermi):
            return super().nr_rks(mol, grids, xc_code, dms, relativity, hermi, max_memory, verbose)
        xctype = self._xc_type(xc_code)
        kind = _kind(self, xc_code)
        nao = mol.nao_nr()
        dm = np.asarray(dms, dtype=np.float64)
        single = dm.ndim == 2
        dm = np.ascontiguousarray(dm.reshape(-1, nao, nao))
        nset = dm.shape[0]
        coords, weights = _grid(grids)
        rho = _rho(mol, coords, kind, dm, _orbitals(dms, nset, nao))
        nelec = np.zeros(nset)
        excsum = np.zeros(nset)
        wvs = []
        for i in range(nset):
            exc, w = _potential_weights(self, xc_code, xctype, kind, rho[i, 0] if kind == 0 else rho[i], weights, 0)
            den = rho[i, 0] * weights
            nelec[i] = den.sum()
            excsum[i] = _wsum(den, exc)
            wvs.append(w)
        vmat = _vmat(mol, coords, kind, wvs[0][np.newaxis] if nset == 1 else np.array(wvs))
        if single:
            return nelec[0], excsum[0], vmat[0]
        return nelec, excsum, vmat

    def nr_uks(self, mol, grids, xc_code, dms, relativity=0, hermi=1, max_memory=2000, verbose=None):
        if not _passes_ok(self, mol, xc_code, dms, hermi):
            return super().nr_uks(mol, grids, xc_code, dms, relativity, hermi, max_memory, verbose)
        xctype = self._xc_type(xc_code)
        kind = _kind(self, xc_code)
        nao = mol.nao_nr()
        dma, dmb = pyscf_numint._format_uks_dm(dms)
        single = np.asarray(dma).ndim == 2
        dma = np.asarray(dma, dtype=np.float64).reshape(-1, nao, nao)
        dmb = np.asarray(dmb, dtype=np.float64).reshape(-1, nao, nao)
        nset = dma.shape[0]
        coords, weights = _grid(grids)
        orbitals = _orbitals(dms, 2, nao) if nset == 1 else None
        rho = _rho(mol, coords, kind, np.ascontiguousarray(np.concatenate([dma, dmb])), orbitals)
        nelec = np.zeros((2, nset))
        excsum = np.zeros(nset)
        wvs = []
        for i in range(nset):
            r = rho if nset == 1 else rho[[i, nset + i]]           # (2, nvar, ngrid): alpha, beta
            exc, w = _potential_weights(self, xc_code, xctype, kind, r[:, 0] if kind == 0 else r, weights, 1)
            den_a = r[0, 0] * weights
            den_b = r[1, 0] * weights
            nelec[0, i] = den_a.sum()
            nelec[1, i] = den_b.sum()
            excsum[i] = _wsum(den_a + den_b, exc)
            wvs.append(w)
        wv = wvs[0] if nset == 1 else np.array(wvs).transpose(1, 0, 2, 3).reshape(2 * nset, -1, weights.size)
        vmat = _vmat(mol, coords, kind, wv).reshape(2, nset, nao, nao)
        if single:
            return nelec[:, 0], excsum[0], vmat[:, 0]
        return nelec, excsum, vmat

    # ---- second-order kernels (TDDFT, CPHF, Hessians) ----

    def cache_xc_kernel(self, mol, grids, xc_code, mo_coeff, mo_occ, spin=0, max_memory=2000):
        """pyscf's ``cache_xc_kernel`` (rho0, vxc, fxc on the grid) with the densities from the Mojo pass."""
        mo_coeff = np.asarray(mo_coeff)
        if not (_fxc_ok(self, mol, xc_code) and np.isrealobj(mo_coeff)):
            return super().cache_xc_kernel(mol, grids, xc_code, mo_coeff, mo_occ, spin, max_memory)
        kind = _kind(self, xc_code)
        coords, _ = _grid(grids)
        if mo_coeff.ndim == 2:          # RKS
            rho = _rho_orbitals(mol, coords, kind, [mo_coeff], [np.asarray(mo_occ)])[0]
            if kind == 0:
                rho = rho[0]
            if spin == 1:               # RKS with nr_rks_fxc_st
                rho = np.repeat((rho * 0.5)[np.newaxis], 2, axis=0)
        else:
            assert spin == 1
            rho = _rho_orbitals(mol, coords, kind, list(mo_coeff), [np.asarray(o) for o in mo_occ])
            rho = (rho[0, 0], rho[1, 0]) if kind == 0 else (rho[0], rho[1])
        vxc, fxc = self.eval_xc_eff(xc_code, rho, deriv=2, xctype=self._xc_type(xc_code), spin=spin)[1:3]
        return rho, vxc, fxc

    def cache_xc_kernel1(self, mol, grids, xc_code, dm, spin=0, max_memory=2000):
        """pyscf's ``cache_xc_kernel1`` (from a density matrix) with the densities from the Mojo pass."""
        dm = np.asarray(dm)
        if not (_fxc_ok(self, mol, xc_code) and np.isrealobj(dm)):
            return super().cache_xc_kernel1(mol, grids, xc_code, dm, spin, max_memory)
        kind = _kind(self, xc_code)
        coords, _ = _grid(grids)
        nao = mol.nao_nr()
        if spin == 0:
            d = 0.5 * (dm.reshape(nao, nao) + dm.reshape(nao, nao).T)
            rho = _rho(mol, coords, kind, np.ascontiguousarray(d[None]))[0]
            rho = rho[0] if kind == 0 else rho
        else:
            d = dm.reshape(-1, nao, nao)
            if d.shape[0] == 1:     # an RKS density for the spin-resolved kernel
                d = np.array([d[0] * 0.5, d[0] * 0.5])
            d = 0.5 * (d + d.transpose(0, 2, 1))
            r = _rho(mol, coords, kind, np.ascontiguousarray(d))
            rho = (r[0, 0], r[1, 0]) if kind == 0 else (r[0], r[1])
        vxc, fxc = self.eval_xc_eff(xc_code, rho, deriv=2, xctype=self._xc_type(xc_code), spin=spin)[1:3]
        return rho, vxc, fxc

    def nr_rks_fxc(self, mol, grids, xc_code, dm0=None, dms=None, relativity=0, hermi=0, rho0=None, vxc=None,
                   fxc=None, max_memory=2000, verbose=None):
        """pyscf's ``nr_rks_fxc``: the XC kernel contracted with the (response) densities ``dms``.

        One fused Mojo pass (:func:`fxc_matrices`) forms the response
        densities of all ``dms`` (their symmetric part: the only one that
        enters) block by block, contracts them with ``fxc`` and builds the
        matrices.
        """
        dms_arr = np.asarray(dms)
        if not (_fxc_ok(self, mol, xc_code) and np.isrealobj(dms_arr) and relativity == 0):
            return super().nr_rks_fxc(mol, grids, xc_code, dm0, dms, relativity, hermi, rho0, vxc, fxc,
                                      max_memory, verbose)
        kind = _kind(self, xc_code)
        if fxc is None:
            fxc = self.cache_xc_kernel1(mol, grids, xc_code, dm0, spin=0, max_memory=max_memory)[2]
        nao = mol.nao_nr()
        d = dms_arr.reshape(1, -1, nao, nao)
        vmat = fxc_matrices(mol, grids, kind, fxc, d)[0]
        return vmat[0] if dms_arr.ndim == 2 else vmat.reshape(dms_arr.shape)

    def nr_uks_fxc(self, mol, grids, xc_code, dm0=None, dms=None, relativity=0, hermi=0, rho0=None, vxc=None,
                   fxc=None, max_memory=2000, verbose=None):
        """pyscf's ``nr_uks_fxc`` (spin-resolved kernel, response densities (2, [nset,] nao, nao))."""
        dms_arr = np.asarray(dms)
        if not (_fxc_ok(self, mol, xc_code) and np.isrealobj(dms_arr) and relativity == 0):
            return super().nr_uks_fxc(mol, grids, xc_code, dm0, dms, relativity, hermi, rho0, vxc, fxc,
                                      max_memory, verbose)
        kind = _kind(self, xc_code)
        if fxc is None:
            fxc = self.cache_xc_kernel1(mol, grids, xc_code, dm0, spin=1, max_memory=max_memory)[2]
        nao = mol.nao_nr()
        d = dms_arr.reshape(2, -1, nao, nao)
        vmat = fxc_matrices(mol, grids, kind, fxc, d)
        return vmat.reshape(dms_arr.shape)


def fxc_matrices(mol, grids, kind, fxc, dms=None, factors=None, project=False):
    """Symmetrised response potentials (nspin, nset, nao, nao) of pyscf's ``nr_rks_fxc``/``nr_uks_fxc``.

    ``fxc`` is the kernel on the grid in pyscf's layout ((nvar, nvar, ngrid)
    for one spin channel, (2, nvar, 2, nvar, ngrid) for two); ``kind`` 0, 1
    or 2 (LDA, GGA, meta-GGA).  The first-order densities are those of the
    symmetric parts of ``dms`` (nspin, nset, nao, nao), or of ``L R^T`` for
    ``factors`` = [(L (nset, nao, r_a), R (nao, r_a)) per spin] (transition
    densities, whose rank is the number of occupied orbitals): the kernel
    then forms them from the factors on the blocks where that is cheaper.
    With ``project`` (factors required) only ``V R`` is formed, shape
    (nspin, nset, nao, max r_a) (columns past r_a zero): all a
    linear-response operator needs, at a fraction of the cost.  The sets are
    processed in chunks that keep the per-thread sums within about 512 MB.
    """
    import os

    coords, weights = _grid(grids)
    nao = mol.nao_nr()
    if factors is not None:
        nspin = len(factors)
        nset = factors[0][0].shape[0]
        rank = max(r.shape[1] for _, r in factors)
    else:
        if project:
            raise ValueError("project needs the factors")
        dms = np.asarray(dms, dtype=np.float64)
        nspin, nset = dms.shape[0], dms.shape[1]
        rank = 0
    project = bool(project) and rank > 0
    ncol = rank if project else nao
    nvar = (1, 4, 5)[kind]
    fxc = np.ascontiguousarray(np.asarray(fxc, dtype=np.float64).reshape(nspin * nvar, nspin * nvar, -1))
    nthreads = os.cpu_count() or 1
    chunk = max(1, int(512e6 / (8.0 * nthreads * nspin * nao * max(ncol, 1))))
    out = np.empty((nspin, nset, nao, ncol))
    basis = integrals.basis_tables(mol)
    path, prefix = worker_blas()
    for s0 in range(0, nset, chunk):
        s1 = min(s0 + chunk, nset)
        n = s1 - s0
        if factors is not None:
            lfac = np.zeros((nspin, n, nao, rank))
            rfac = np.zeros((nspin, nao, rank))
            d = np.empty((nspin, n, nao, nao))
            for a, (lf, rf) in enumerate(factors):
                r = rf.shape[1]
                lfac[a, :, :, :r] = lf[s0:s1]
                rfac[a, :, :r] = rf
                d[a] = lf[s0:s1] @ rf.T
            d += d.transpose(0, 1, 3, 2)
            d *= 0.5
        else:
            d = 0.5 * (dms[:, s0:s1] + dms[:, s0:s1].transpose(0, 1, 3, 2))
            lfac = np.zeros((nspin, n, nao, 0))
            rfac = np.zeros((nspin, nao, 0))
        v = np.empty((nspin, n, nao, ncol))
        get_extension().xc_fxc(basis, coords, weights, int(kind), fxc, np.ascontiguousarray(d), lfac, rfac,
                               project, v, path, prefix)
        out[:, s0:s1] = v if project else v + v.transpose(0, 1, 3, 2)
    return out


def _fxc_ok(ni, mol, xc_code) -> bool:
    """The Mojo second-order kernels apply (LDA, GGA, meta-GGA without laplacian; engine enabled)."""
    return _kind(ni, xc_code) is not None and integrals.engine() == "mojo" and supported(mol)


def _rho_orbitals(mol, coords, kind, mo_coeffs, mo_occs):
    """(nspin, ncomp, ngrid) densities of the occupied orbitals of each spin channel (pyscf's eval_rho2)."""
    nao = mol.nao_nr()
    dms, orbs, occs = [], [], []
    for c, n in zip(mo_coeffs, mo_occs):
        c = np.asarray(c, dtype=np.float64)
        n = np.asarray(n, dtype=np.float64)
        keep = np.flatnonzero(n)
        dms.append((c[:, keep] * n[keep]) @ c[:, keep].T)
        orbs.append(c[:, keep])
        occs.append(n[keep])
    norb = max(o.shape[1] for o in orbs)
    po = np.zeros((len(orbs), nao, norb))
    pn = np.zeros((len(orbs), norb))
    for i, (o, n) in enumerate(zip(orbs, occs)):
        po[i, :, : o.shape[1]] = o
        pn[i, : n.size] = n
    return _rho(mol, coords, kind, np.ascontiguousarray(np.array(dms)), (po, pn))


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
            if xctype in ("GGA", "MGGA"):
                wv[j, 0] *= 0.5
            if xctype == "MGGA":
                wv[j, 4] *= 0.5                 # the 1/2 of tau
    return wv


def grad_rks_vxc(ni, mol, grids, xc_code, dms, relativity=0, hermi=1, max_memory=2000, verbose=None):
    """``pyscf.grad.rks.get_vxc`` (no grid response) for LDA, GGA and meta-GGA, from the Mojo kernels:
    (None, -vmat)."""
    kind = _kind(ni, xc_code)
    nao = mol.nao_nr()
    dm = np.ascontiguousarray(np.asarray(dms, dtype=np.float64).reshape(-1, nao, nao))
    coords, weights = _grid(grids)
    rho = _rho(mol, coords, kind, dm, _orbitals(dms, dm.shape[0], nao))
    vmat = _xc_grad(mol, coords, kind, _grad_weights(ni, xc_code, rho, weights, 0))
    if vmat.shape[0] == 1:
        vmat = vmat[0]
    return None, -vmat


def grad_uks_vxc(ni, mol, grids, xc_code, dms, relativity=0, hermi=1, max_memory=2000, verbose=None):
    """``pyscf.grad.uks.get_vxc`` (no grid response) for LDA, GGA and meta-GGA, from the Mojo kernels:
    (None, -vmat)."""
    kind = _kind(ni, xc_code)
    nao = mol.nao_nr()
    dm = np.ascontiguousarray(np.asarray(dms, dtype=np.float64).reshape(2, nao, nao))
    coords, weights = _grid(grids)
    rho = _rho(mol, coords, kind, dm, _orbitals(dms, 2, nao))
    vmat = _xc_grad(mol, coords, kind, _grad_weights(ni, xc_code, rho, weights, 1))
    return None, -vmat


def grad_xc(ni, mol, grids, xc_code, dms, spin=0):
    """XC term of the nuclear gradient at fixed grids, shape (natm, 3).

    What pyscf's gradient adds from the XC part of ``get_veff``:
    ``2 sum_{mu on A} (vxc[:, mu] * D[mu]).sum()`` with ``vxc`` from
    ``grad.rks.get_vxc`` (``spin=0``, ``dms`` the total density) or
    ``grad.uks.get_vxc`` (``spin=1``, ``dms`` (2, nao, nao), summed over
    spins).  For :class:`NumInt` objects with LDA, GGA and meta-GGA functionals the Mojo
    kernels contract the derivative matrices with the density as they are
    produced; otherwise pyscf's ``get_vxc`` provides the matrices.
    """
    from pyscf.grad import rks as rks_grad
    from pyscf.grad import uks as uks_grad

    nao = mol.nao_nr()
    dm = np.asarray(dms, dtype=np.float64).reshape(-1, nao, nao)
    if isinstance(ni, NumInt) and _passes_ok(ni, mol, xc_code, dm, 1):
        kind = _kind(ni, xc_code)
        coords, weights = _grid(grids)
        dm = np.ascontiguousarray(dm)
        orbitals = _orbitals(dms, dm.shape[0], nao)
        rho = _rho(mol, coords, kind, dm, orbitals)
        wv = _grad_weights(ni, xc_code, rho, weights, spin)
        de = np.empty((mol.natm, 3))
        if orbitals is None:
            orbitals = (np.zeros((dm.shape[0], nao, 0)), np.zeros((dm.shape[0], 0)))
        path, prefix = worker_blas()
        get_extension().xc_grad_dm(integrals.basis_tables(mol), coords, int(kind), np.ascontiguousarray(wv), dm,
                                   np.ascontiguousarray(orbitals[0]), np.ascontiguousarray(orbitals[1]), de,
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
            if (isinstance(ni, NumInt) and relativity == 0 and nset_ok(mol, dms)
                    and _passes_ok(ni, mol, xc_code, dms, hermi)):
                return mojo_fn(ni, mol, grids, xc_code, dms, relativity, hermi, max_memory, verbose)
            return orig(ni, mol, grids, xc_code, dms, relativity, hermi, max_memory, verbose)

        get_vxc.__doc__ = orig.__doc__
        get_vxc._mojoscf_orig = orig
        module.get_vxc = get_vxc

    nao_ok = lambda mol, dms: np.asarray(dms).shape[-1] == mol.nao_nr()
    wrap(rks_grad, grad_rks_vxc, nao_ok)
    wrap(uks_grad, grad_uks_vxc, lambda mol, dms: nao_ok(mol, dms) and np.asarray(dms).size == 2 * mol.nao_nr() ** 2)


_install_grad_vxc()


class _MojoDFObject:
    """In front of pyscf's ``df.DF`` (:data:`MojoDF`, the ``with_df`` of the objects mojoscf accelerates):
    ``get_jk`` from the Mojo DF kernel for real symmetric densities, the tensor in memory or on disk
    (:func:`df_object_jk`).  pyscf code that calls the DF object directly, such as the orbital-Hessian
    steps of DF-CASSCF, then runs on the same kernel as the SCF."""

    __name_mixin__ = "Mojo"

    def get_jk(self, dm, hermi=1, with_j=True, with_k=True, direct_scf_tol=1e-13, omega=None):
        jk = df_object_jk(self, dm, hermi, with_j, with_k, omega)
        if jk is not None:
            return jk
        return super().get_jk(dm, hermi, with_j, with_k, direct_scf_tol, omega)


MojoDF = type("MojoDF", (_MojoDFObject, pyscf_df.DF), {"__module__": __name__})


def mojo_df(with_df):
    """Put :class:`_MojoDFObject` in front of a plain pyscf ``df.DF`` object (in place); returns it.

    Also lets pyscf's DF-CASSCF build its integrals for such objects with
    :mod:`mojoscf.casscf`.
    """
    if type(with_df) is pyscf_df.DF:
        with_df.__class__ = MojoDF
    if isinstance(with_df, MojoDF):
        from . import casscf

        casscf.install()
    return with_df


def plain_df(with_df) -> bool:
    """True for pyscf's own ``df.DF`` objects (with :class:`_MojoDFObject` in front or not), whose tensor
    layout and J/K the Mojo kernels reproduce; False for other DF classes."""
    return type(with_df) is pyscf_df.DF or type(with_df) is MojoDF


def _available_memory() -> int:
    """Bytes of memory the system has available (Linux ``MemAvailable``, else free pages; 0 if unknown)."""
    import os

    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return 0


def ondisk_tensor(with_df):
    """The DF tensor of ``with_df`` that pyscf keeps on disk, memory-mapped read-only, or None.

    Applies to one contiguous, uncompressed ``(naux, npair)`` HDF5 dataset
    (what :func:`mojoscf.integrals.cholesky_eri_h5` and pyscf's
    ``_compatible_format`` write) when the file fits in the memory the system
    has available: the page cache then holds the file, which pyscf's
    ``max_memory`` does not count, and every kernel reads the tensor as if it
    were in memory.  pyscf's block format and larger files are streamed
    instead (:func:`df_object_jk`).  The map is kept on the DF object until
    its ``_cderi`` changes.
    """
    import h5py

    cderi = with_df._cderi
    if cderi is None or isinstance(cderi, np.ndarray):
        return None
    cached = getattr(with_df, "_mojo_mmap", None)
    if cached is not None and cached[0] is cderi:
        return cached[1]
    name = cderi if isinstance(cderi, str) else getattr(cderi, "name", None)
    if not isinstance(name, str) or not h5py.is_hdf5(name):
        return None
    with h5py.File(name, "r") as f:
        dset = f.get(with_df._dataname)
        if (not isinstance(dset, h5py.Dataset) or dset.ndim != 2 or dset.dtype != np.float64
                or dset.chunks is not None or dset.compression is not None):
            return None
        offset = dset.id.get_offset()
        shape = dset.shape
    nao = with_df.mol.nao_nr()
    if offset is None or shape[1] != nao * (nao + 1) // 2 or shape[0] * shape[1] * 8 > 0.8 * _available_memory():
        return None
    arr = np.memmap(name, dtype=np.float64, mode="r", offset=offset, shape=shape)
    with_df._mojo_mmap = (cderi, arr)
    return arr


def _tensor_blocks(with_df):
    """``with_df.loop`` blocks of a tensor on disk, sized to keep two of them (one read ahead) within
    about 30% of the free memory and at least 32 auxiliary functions each."""
    from pyscf import lib as pyscf_lib

    nao = with_df.mol.nao_nr()
    max_memory = with_df.max_memory - pyscf_lib.current_memory()[0]
    return with_df.loop(max(32, int(max_memory * 0.3e6 / 8 / (nao * (nao + 1) // 2))))


def _tags(dm):
    mo_coeff = getattr(dm, "mo_coeff", None)
    mo_occ = getattr(dm, "mo_occ", None)
    if mo_coeff is None or mo_occ is None or np.asarray(mo_coeff).ndim != np.asarray(dm).ndim:
        return None, None
    return mo_coeff, mo_occ


def df_object_jk(with_df, dm, hermi=1, with_j=True, with_k=True, omega=None):
    """pyscf's ``with_df.get_jk(dm)`` from the Mojo DF kernel: the tensor in memory
    (:func:`mojoscf.kernels.df_jk`) or on disk, streamed block by block
    (:func:`mojoscf.kernels.df_jk_blocks`); a missing tensor is built where
    pyscf would build it (:func:`mojoscf.integrals.build_df`).  For plain pyscf
    ``df.DF`` objects and real symmetric densities; None otherwise (``omega``,
    ``hermi=0``, ...)."""
    from . import kernels

    if omega or hermi != 1 or not np.isrealobj(dm) or not plain_df(with_df):
        return None
    if with_df._cderi is None and not integrals.build_df(with_df):
        with_df.build()
    cderi = with_df._cderi
    nao = with_df.mol.nao_nr()
    if cderi is None or np.asarray(dm).shape[-1] != nao:
        return None
    mo_coeff, mo_occ = _tags(dm)
    if not isinstance(cderi, np.ndarray):
        mapped = ondisk_tensor(with_df)
        if mapped is None:
            return kernels.df_jk_blocks(_tensor_blocks(with_df), np.asarray(dm), mo_coeff, mo_occ, with_j, with_k)
        cderi = mapped
    if cderi.ndim != 2 or cderi.dtype != np.float64 or cderi.shape[1] != nao * (nao + 1) // 2:
        return None
    return kernels.df_jk(cderi, np.asarray(dm), mo_coeff, mo_occ, with_j, with_k)


def _df_tensor(mf, omega=0.0):
    """pyscf's in-core DF tensor ``(naux, nao*(nao+1)//2)`` of the density-fitted object ``mf``, or None.

    A missing tensor is built with the Mojo integrals where pyscf would build
    it (``integrals.build_df``).  ``omega`` > 0 (the long-range exchange of
    range-separated functionals) gives the tensor of pyscf's
    ``with_df.range_coulomb(omega)`` (a context manager that sets the
    molecules' ``omega`` while it is open), built with the attenuated
    integrals.  A tensor on disk is memory-mapped where that applies
    (:func:`ondisk_tensor`).  None for other DF classes, tensors on disk that
    cannot be mapped and ``only_dfj`` objects.
    """
    from pyscf.df import df_jk

    if not isinstance(mf, df_jk._DFHF) or getattr(mf, "only_dfj", False):
        return None
    with_df = mf.with_df
    if not plain_df(with_df):
        return None
    nao = mf.mol.nao_nr()

    def tensor(d):
        if not plain_df(d):
            return None
        if d._cderi is None and not integrals.build_df(d):
            d.build()
        cderi = d._cderi
        if not isinstance(cderi, np.ndarray):
            cderi = ondisk_tensor(d)
        if isinstance(cderi, np.ndarray) and cderi.ndim == 2 and cderi.dtype == np.float64 and cderi.shape[1] == nao * (nao + 1) // 2:
            return cderi
        return None

    if omega:
        with with_df.range_coulomb(omega) as rsh_df:
            return tensor(rsh_df)
    return tensor(with_df)


def _incore_cderi(mf, mol, dm, hermi, omega):
    """The DF tensor (:func:`_df_tensor`) for :func:`mojoscf.kernels.df_jk`, or None to use pyscf's get_jk."""
    if mol is not None and mol is not mf.mol:
        return None
    if (omega is not None and omega < 0) or hermi != 1 or not np.isrealobj(dm):
        return None
    return _df_tensor(mf, omega or 0.0)


def _df_jk(cderi, dm, with_j, with_k):
    """:func:`mojoscf.kernels.df_jk` for ``dm``, through its orbitals when it carries pyscf's tags."""
    from . import kernels

    mo_coeff, mo_occ = _tags(dm)
    return kernels.df_jk(cderi, np.asarray(dm), mo_coeff, mo_occ, with_j, with_k)


def _ondisk_df_jk(mf, dm, hermi, with_j, with_k, omega):
    """J/K of the density-fitted object ``mf`` whose DF tensor is kept on disk (pyscf's choice when it does
    not fit in memory, or a file named in ``_cderi_to_save``), streamed (:func:`df_object_jk`); None when
    that does not apply."""
    if getattr(mf, "only_dfj", False) or isinstance(mf.with_df._cderi, np.ndarray):
        return None
    return df_object_jk(mf.with_df, dm, hermi, with_j, with_k, omega)


def hooked_jk(mf, hook, mol, dm, hermi, with_j, with_k, omega):
    """J/K from the Mojo kernels for ``mf.get_jk`` when ``hook`` (a class of ``type(mf).__mro__``) sits in
    front of pyscf's own ``get_jk``; None when that does not apply (pyscf's then runs).

    The implementation behind ``hook`` (the next class in the MRO defining
    ``get_jk``) decides what is computed: pyscf's density-fitting one
    (``_DFHF``) gets :func:`mojoscf.kernels.df_jk` on the in-core DF tensor
    (:func:`_df_tensor`; a tensor on disk is streamed, :func:`_ondisk_df_jk`),
    pyscf's exact one (``RHF``/``UHF``) gets
    :func:`exact_jk` (in-core ERIs or integral-direct); anything else,
    including a ``get_jk`` set on the instance, is left alone.
    """
    from pyscf.df import df_jk

    if "get_jk" in vars(mf) or (mol is not None and mol is not mf.mol) or not np.isrealobj(dm):
        return None
    mro = type(mf).__mro__
    nxt = next((c for c in mro[mro.index(hook) + 1:] if "get_jk" in c.__dict__), None)
    if nxt is df_jk._DFHF:
        cderi = _incore_cderi(mf, mol, dm, hermi, omega)
        if cderi is None:
            return _ondisk_df_jk(mf, dm, hermi, with_j, with_k, omega)
        return _df_jk(cderi, dm, with_j, with_k)
    if nxt is None or nxt.__module__ not in ("pyscf.scf.hf", "pyscf.scf.uhf"):
        return None
    if (omega or 0) < 0 or hermi not in (0, 1):
        return None
    if integrals.engine() != "mojo" or not integrals.available(mf.mol, two_electron=True):
        return None
    return exact_jk(mf, dm, hermi, with_j, with_k, omega)


def _exact_jk(mf, mol, dm, hermi, with_j, with_k, omega):
    """J/K with exact integrals from mojoscf (:func:`exact_jk`) for a non-DF object whose ``get_jk`` behind
    :class:`_MojoKSHook` is pyscf's ``SCF.get_jk``; None when that does not apply."""
    from pyscf.df import df_jk

    if isinstance(mf, df_jk._DFHF) or (mol is not None and mol is not mf.mol):
        return None
    if (omega or 0) < 0 or hermi not in (0, 1) or not np.isrealobj(dm) or "get_jk" in vars(mf):
        return None
    mro = type(mf).__mro__
    nxt = next((c for c in mro[mro.index(_MojoKSHook) + 1:] if "get_jk" in c.__dict__), None)
    if nxt is None or nxt.__module__ not in ("pyscf.scf.hf", "pyscf.scf.uhf"):
        return None
    if integrals.engine() != "mojo" or not integrals.available(mf.mol, two_electron=True):
        return None
    return exact_jk(mf, dm, hermi, with_j, with_k, omega)


def exact_jk_applies(mf) -> bool:
    """Whether ``mf.get_jk`` is pyscf's exact-integral ``SCF.get_jk`` (``RHF``/``UHF``, with at most
    mojoscf's routing hooks in front, which compute the same) and the Mojo engine handles the molecule,
    so that :func:`exact_jk` returns the same matrices."""
    from pyscf.df import df_jk

    if isinstance(mf, df_jk._DFHF) or "get_jk" in vars(mf):
        return False
    nxt = next((c for c in type(mf).__mro__
                if "get_jk" in c.__dict__ and not getattr(c.__dict__["get_jk"], "_mojoscf_routes", False)), None)
    if nxt is None or nxt.__module__ not in ("pyscf.scf.hf", "pyscf.scf.uhf"):
        return False
    if integrals.engine() != "mojo" or not integrals.available(mf.mol, two_electron=True):
        return False
    eri = getattr(mf, "_eri", None)
    nao = mf.mol.nao_nr()
    npair = nao * (nao + 1) // 2
    return eri is None or (isinstance(eri, np.ndarray) and eri.dtype == np.float64
                           and eri.size == npair * (npair + 1) // 2)


def exact_jk(mf, dm, hermi=1, with_j=True, with_k=True, omega=None):
    """J/K of ``dm`` with exact integrals over ``mf.mol``, as pyscf's ``RHF.get_jk`` forms them.

    pyscf's in-core 8-fold ERIs ``mf._eri`` (built with the Mojo engine when
    pyscf would keep them in core) are contracted by
    :func:`mojoscf.kernels.jk_s8`, else the integral-direct kernel runs
    (:func:`mojoscf.integrals.get_jk`, screened with ``mf.direct_scf_tol``).
    ``hermi`` 1: symmetric densities, 0: any.  ``omega`` > 0: the long-range
    operator erf(omega r12) / r12, integral-direct as in pyscf.  None when
    ``mf._eri`` is not an 8-fold ERI array.
    """
    from . import kernels

    mol = mf.mol
    nao = mol.nao_nr()
    npair = nao * (nao + 1) // 2
    dm = np.asarray(dm, dtype=np.float64)
    if omega:
        return integrals.get_jk(mol, dm, with_j, with_k, direct_scf_tol=mf.direct_scf_tol, hermi=hermi,
                                omega=omega)
    eri = getattr(mf, "_eri", None)
    if eri is None and (mol.incore_anyway or mf._is_mem_enough()):
        eri = mf._eri = integrals.int2e_s8(mol)
    if eri is not None:
        if isinstance(eri, np.ndarray) and eri.dtype == np.float64 and eri.size == npair * (npair + 1) // 2:
            return kernels.jk_s8(eri.reshape(-1), dm, with_j, with_k, hermi)
        return None
    return integrals.get_jk(mol, dm, with_j, with_k, direct_scf_tol=mf.direct_scf_tol, hermi=hermi)


class _MojoKSHook:
    """In front of a pyscf Kohn-Sham class (:func:`accelerate`): J/K and the eigensolver from mojoscf."""

    __name_mixin__ = "Mojo"

    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True, omega=None):
        if dm is None:
            dm = self.make_rdm1()
        jk = hooked_jk(self, _MojoKSHook, mol, dm, hermi, with_j, with_k, omega)
        if jk is not None:
            return jk
        return super().get_jk(mol, dm, hermi, with_j, with_k, omega)

    get_jk._mojoscf_routes = True

    def get_j(self, mol=None, dm=None, hermi=1, omega=None):
        return self.get_jk(mol, dm, hermi, True, False, omega)[0]

    def get_k(self, mol=None, dm=None, hermi=1, omega=None):
        return self.get_jk(mol, dm, hermi, False, True, omega)[1]

    @property
    def DIIS(self):
        """mojoscf's CDIIS in place of pyscf's default CDIIS (the same iterates).

        pyscf's CDIIS takes overlaps of error vectors with NumPy, whose
        threaded ``ddot`` leaves OpenBLAS threads spinning while the next Mojo
        pass runs (about 6% of a ferrocene SCF).  A DIIS class set on the
        object, another scheme (EDIIS, ADIIS), ``diis_space_rollback``,
        ``diis_file`` and point-group symmetry (whose error vectors pyscf masks
        by irrep) keep pyscf's.
        """
        from .scf import _hook_diis

        return _hook_diis(self, _MojoKSHook)

    @DIIS.setter
    def DIIS(self, value):
        self.__dict__["DIIS"] = value

    def density_fit(self, auxbasis=None, with_df=None, only_dfj=False):
        """pyscf's ``density_fit()``, the Mojo J/K and gradients in front of its density fitting."""
        from .scf import _hook_density_fit

        return _hook_density_fit(self, _MojoKSHook, auxbasis, with_df, only_dfj)

    def nuc_grad_method(self):
        """Nuclear gradients with Mojo derivative integrals and XC kernels (``mojoscf.grad.KSGradients`` ...)."""
        from .scf import _mojo_grad_method

        return _mojo_grad_method(self, _MojoKSHook)

    Gradients = nuc_grad_method

    def Hessian(self):
        """pyscf's analytical Hessian (DF or not) with the Mojo kernels (:mod:`mojoscf.hessian`): the XC terms,
        and for density fitting the Coulomb/exchange terms and the coupled-perturbed operator."""
        from . import hessian

        return hessian.accelerate(super().Hessian())

    # The response methods below are mojoscf's where pyscf's plain RKS/UKS ones would run; a decoration's
    # own (a solvent model wraps the TD object and sets up the solvent response) is kept.

    def stability(self, *args, **kwargs):
        """pyscf's stability analysis with the orbital Hessian of :mod:`mojoscf.stability`."""
        from .scf import _plain_next

        if not _plain_next(self, _MojoKSHook, "stability"):
            return super().stability(*args, **kwargs)
        from . import stability

        return stability.stability(self, *args, **kwargs)

    def TDA(self, *args, **kwargs):
        """pyscf's TDA with the response in the occupied-virtual space (:mod:`mojoscf.tdscf`)."""
        from .scf import _plain_next

        if not _plain_next(self, _MojoKSHook, "TDA"):
            return super().TDA(*args, **kwargs)
        from . import tdscf

        return tdscf.TDA(self, *args, **kwargs)

    def TDDFT(self, *args, **kwargs):
        """pyscf's ``TDDFT`` (full TDDFT for hybrids, the Casida form otherwise) with :mod:`mojoscf.tdscf`."""
        from .scf import _plain_next

        if not _plain_next(self, _MojoKSHook, "TDDFT"):
            return super().TDDFT(*args, **kwargs)
        from . import tdscf

        return tdscf.TDDFT(self, *args, **kwargs)

    def CasidaTDDFT(self, *args, **kwargs):
        from .scf import _plain_next

        if not _plain_next(self, _MojoKSHook, "CasidaTDDFT"):
            return super().CasidaTDDFT(*args, **kwargs)
        from . import tdscf

        return tdscf.CasidaTDDFT(self, *args, **kwargs)

    TDDFTNoHybrid = CasidaTDDFT

    def _eigh(self, h, s, overwrite=False, x=None):
        from .scf import _hook_eigh

        return _hook_eigh(self, _MojoKSHook, h, s, overwrite, x)


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
    * the eigensolver: :func:`mojoscf.kernels.eigh`; DIIS: :class:`mojoscf.CDIIS`
      (when pyscf's default CDIIS would be used);
    * QM/MM objects: the Mojo MM-charge terms (:func:`mojoscf.qmmm.attach`);
    * PCM/SMD solvents: the Mojo surface-charge kernels (:func:`mojoscf.solvent.attach`).

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
    if getattr(mf, "with_df", None) is not None:
        mojo_df(mf.with_df)
    from pyscf.qmmm import itrf

    if isinstance(mf, itrf.QMMMSCF):
        from . import qmmm

        qmmm.attach(mf)
    from . import solvent

    solvent.attach(mf)
    return mf
