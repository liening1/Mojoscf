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


def _xc_inputs(ni, mol, grids, xc_code, mo_coeff, mo_occ, kind):
    """(coords, weights, dms (nspin, nao, nao), vxc (nspin nvar, ngrid), fxc ((nspin nvar)^2, ngrid))."""
    from . import dft

    mo_coeff = np.asarray(mo_coeff)
    coords, weights = dft._grid(grids)
    if mo_coeff.ndim == 2:
        occ = np.asarray(mo_occ)
        rho = dft._rho_orbitals(mol, coords, kind, [mo_coeff], [occ])[0]
        vxc, fxc = ni.eval_xc_eff(xc_code, rho[0] if kind == 0 else rho, deriv=2, xctype=ni._xc_type(xc_code))[1:3]
        c = mo_coeff[:, occ > 0]
        dms = ((c * occ[occ > 0]) @ c.T)[None]
    else:
        occs = [np.asarray(o) for o in mo_occ]
        rho = dft._rho_orbitals(mol, coords, kind, list(mo_coeff), occs)
        r = (rho[0, 0], rho[1, 0]) if kind == 0 else (rho[0], rho[1])
        vxc, fxc = ni.eval_xc_eff(xc_code, r, deriv=2, xctype=ni._xc_type(xc_code), spin=1)[1:3]
        dms = np.array([(c[:, o > 0] * o[o > 0]) @ c[:, o > 0].T for c, o in zip(mo_coeff, occs)])
    nspin = dms.shape[0]
    nvar = (1, 4)[kind]
    ngrid = coords.shape[0]
    vxc = np.ascontiguousarray(np.asarray(vxc, dtype=np.float64).reshape(nspin * nvar, ngrid))
    fxc = np.ascontiguousarray(np.asarray(fxc, dtype=np.float64).reshape(nspin * nvar, nspin * nvar, ngrid))
    return coords, np.ascontiguousarray(weights, dtype=np.float64), np.ascontiguousarray(dms, dtype=np.float64), vxc, fxc


def xc_h1mo(ni, mol, grids, xc_code, mo_coeff, mo_occ):
    """The XC first-derivative Fock matrices of pyscf's ``_get_vxc_deriv1`` projected to the MO basis.

    Returns ``C_s^T h1[s][A][x] C_o,s`` as one (natm, 3, nmo, nocc_s) array per
    spin (MOs in their original order, occupied ones as in ``mo_occ > 0``), or
    None for functionals the kernels do not cover.  The gradient-matrix part
    comes from the XC gradient kernel (``dft._xc_grad``), the kernel part from
    ``xc_h1_core`` without forming the nao x nao matrices.
    """
    from . import dft

    kind = dft._kind(ni, xc_code)
    if kind not in (0, 1) or not dft._fxc_ok(ni, mol, xc_code):
        return None
    coords, weights, dms, vxc, fxc = _xc_inputs(ni, mol, grids, xc_code, mo_coeff, mo_occ, kind)
    nspin = dms.shape[0]
    nvar = (1, 4)[kind]
    wv = (vxc * weights).reshape(nspin, nvar, -1)
    if kind == 1:
        wv[:, 0] *= 0.5
    vgrad = dft._xc_grad(mol, coords, kind == 1, np.ascontiguousarray(wv))        # (nspin, 3, nao, nao)
    cs = [np.asarray(mo_coeff)] if nspin == 1 else [np.asarray(c) for c in mo_coeff]
    occs = [np.asarray(mo_occ)] if nspin == 1 else [np.asarray(o) for o in mo_occ]
    perms = [np.concatenate((np.flatnonzero(o > 0), np.flatnonzero(o == 0))) for o in occs]
    noccs = [int((o > 0).sum()) for o in occs]
    nao, nmo = cs[0].shape
    nocc = max(max(noccs), 1)
    cmo = np.ascontiguousarray(np.array([c[:, p] for c, p in zip(cs, perms)]))
    h1 = np.empty((nspin, mol.natm, 3, nmo, nocc))
    path, prefix = worker_blas()
    get_extension().xc_h1(integrals.basis_tables(mol), coords, weights, int(kind), dms, fxc, _ao_atoms(mol), cmo, nocc,
                          h1, path, prefix)
    out = []
    for s in range(nspin):
        c = cmo[s]
        co = c[:, : noccs[s]]
        h = -h1[s][..., : noccs[s]]
        for ia, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
            t = vgrad[s][:, p0:p1]                                  # rows of atom A: (3, nA, nao)
            h[ia] -= c[p0:p1].T @ (t @ co) + (t @ c).transpose(0, 2, 1) @ co[p0:p1]
        back = np.empty_like(h)
        back[:, :, perms[s]] = h
        out.append(back)
    return out


class _CPHFChannel:
    """One spin channel of the coupled-perturbed operator: orbitals (occupied first) and MO-basis DF tensors."""

    def __init__(self, mo_coeff, mo_occ):
        occ = np.asarray(mo_occ)
        self.perm = np.concatenate((np.flatnonzero(occ > 0), np.flatnonzero(occ == 0)))
        self.c = np.ascontiguousarray(np.asarray(mo_coeff)[:, self.perm])
        self.nmo = self.c.shape[1]
        self.nocc = int((occ > 0).sum())
        self.co = np.ascontiguousarray(self.c[:, : self.nocc])
        self.lmo = None          # (naux, nmo, nocc) for J
        self.k = []              # [(c, lfull, lmo, loo)] per exchange term


def cphf_operator(mf, mo_coeff=None, mo_occ=None):
    """``fx(mo1) -> v1vo`` of pyscf's Hessian ``gen_vind`` (RHF/RKS or UHF/UKS) in the MO basis, or None.

    ``mo1`` holds first-order orbitals x (nmo x nocc per spin); the response
    of D = C x C_o^T + h.c. (x2 for RKS) projected to C^T V C_o is built from
    the in-core DF tensor transformed once to the MO basis: Coulomb via (pi|Q),
    exchange with :func:`mojoscf.kernels.cphf_k`, and the XC kernel in one
    fused pass from the factors C x and C_o (:func:`mojoscf.dft.fxc_matrices`
    with ``project``).  None (pyscf's operator runs) without an in-core DF
    tensor, for solvent models, NLC, short-range-only hybrids, fractional
    occupations and MO tensors that do not fit in ``max_memory``.
    """
    from pyscf import lib
    from pyscf.scf import hf, rohf, uhf

    from . import dft, kernels, tdscf

    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    if mo_occ is None:
        mo_occ = mf.mo_occ
    if integrals.engine() != "mojo" or not np.isrealobj(mo_coeff):
        return None
    unrestricted = isinstance(mf, uhf.UHF)
    if isinstance(mf, rohf.ROHF) or not (unrestricted or isinstance(mf, hf.RHF)):
        return None
    try:
        from pyscf.solvent._attach_solvent import _Solvation

        if isinstance(mf, _Solvation):
            return None
    except ImportError:  # pragma: no cover
        pass
    ks = isinstance(mf, hf.KohnShamDFT)
    if ks and mf.do_nlc():
        return None
    full = 2.0 if not unrestricted else 1.0
    occs = [np.asarray(mo_occ)] if not unrestricted else [np.asarray(o) for o in mo_occ]
    if any(not np.all((o == 0) | (o == full)) for o in occs):
        return None
    kterms = tdscf._exchange_terms(mf)
    if kterms is None:
        return None
    cderi = dft._df_tensor(mf)
    if cderi is None:
        return None
    tensors = {0.0: cderi}
    for _, omega in kterms:
        if omega not in tensors:
            t = dft._df_tensor(mf, omega)
            if t is None:
                return None
            tensors[omega] = t
    cs = [np.asarray(mo_coeff)] if not unrestricted else [np.asarray(c) for c in mo_coeff]
    chans = [_CPHFChannel(c, o) for c, o in zip(cs, occs)]
    words = 0
    for ch in chans:
        words += cderi.shape[0] * ch.nmo * ch.nocc
        for c, omega in kterms:
            words += tensors[omega].shape[0] * (ch.nmo * ch.nmo + ch.nmo * ch.nocc + ch.nocc * ch.nocc)
    if words * 8e-6 > 0.7 * (mf.max_memory - lib.current_memory()[0]):
        return None
    for ch in chans:
        if ch.nocc == 0:
            continue
        for c, omega in kterms:
            if c == 0:
                continue
            lfull = kernels.df_mo(tensors[omega], ch.c, ch.c)
            lmo = np.ascontiguousarray(lfull[:, :, : ch.nocc])
            loo = np.ascontiguousarray(lfull[:, : ch.nocc, : ch.nocc])
            ch.k.append((c, lfull, lmo, loo))
            if omega == 0 and ch.lmo is None:
                ch.lmo = lmo
        if ch.lmo is None:
            ch.lmo = kernels.df_mo(cderi, ch.c, ch.co)

    xc = None
    if ks and mf._numint._xc_type(mf.xc) != "HF":
        ni = mf._numint
        mol = mf.mol
        rho0, vxc, fxc = ni.cache_xc_kernel(mol, mf.grids, mf.xc, mo_coeff, mo_occ, 1 if unrestricted else 0)
        fast = isinstance(ni, dft.NumInt) and dft._fxc_ok(ni, mol, mf.xc)
        kind = dft._kind(ni, mf.xc) if fast else None
        max_memory = max(2000, mf.max_memory * 0.8 - lib.current_memory()[0])

        def xc(factors):
            if fast:
                w = dft.fxc_matrices(mol, mf.grids, kind, fxc, factors=factors, project=True)
                return [ch.c.T @ w[a][:, :, : ch.nocc] for a, ch in enumerate(chans)]
            if unrestricted:
                dms = np.asarray([lf @ rf.T for lf, rf in factors])
                dms = dms + dms.transpose(0, 1, 3, 2)
                v1 = ni.nr_uks_fxc(mol, mf.grids, mf.xc, None, dms * 0.5, 0, 1, rho0, vxc, fxc,
                                   max_memory=max_memory)
            else:
                lf, rf = factors[0]
                d = lf @ rf.T
                v1 = [ni.nr_rks_fxc(mol, mf.grids, mf.xc, None, 0.5 * (d + d.transpose(0, 2, 1)), 0, 1, rho0, vxc,
                                    fxc, max_memory=max_memory)]
            return [ch.c.T @ v @ ch.co for ch, v in zip(chans, v1)]

    def fx(mo1):
        mo1 = np.asarray(mo1)
        sizes = [ch.nmo * ch.nocc for ch in chans]
        flat = mo1.reshape(-1, sum(sizes))
        nset = flat.shape[0]
        xs, off = [], 0
        for ch, n in zip(chans, sizes):
            x = flat[:, off: off + n].reshape(nset, ch.nmo, ch.nocc)
            xs.append(np.ascontiguousarray(x[:, ch.perm]))
            off += n
        # Coulomb: rho_Q = s sum (pi|Q) x_pi over the spins (s = 4 RKS, 2 UKS)
        rho = 0.0
        for ch, x in zip(chans, xs):
            if ch.nocc:
                rho = rho + ch.lmo.reshape(ch.lmo.shape[0], -1) @ x.reshape(nset, -1).T
        rho = rho * (4.0 if not unrestricted else 2.0)
        vs = []
        for ch, x in zip(chans, xs):
            if ch.nocc == 0:
                vs.append(np.zeros_like(x))
                continue
            v = (rho.T @ ch.lmo.reshape(ch.lmo.shape[0], -1)).reshape(x.shape)
            for c, lfull, lmo, loo in ch.k:
                kernels.cphf_k(lfull, lmo, loo, x, -c, out=v)
            vs.append(v)
        if xc is not None:
            scale = 4.0 if not unrestricted else 2.0
            px = xc([(scale * (ch.c @ x), ch.co) for ch, x in zip(chans, xs)])
            for v, p in zip(vs, px):
                v += p
        out = np.empty_like(flat)
        off = 0
        for ch, v, n in zip(chans, vs, sizes):
            back = np.empty_like(v)
            back[:, ch.perm] = v
            out[:, off: off + n] = back.reshape(nset, -1)
            off += n
        if unrestricted:
            return out                       # as pyscf's: (nset, nmoa nocca + nmob noccb)
        return out.reshape(nset, chans[0].nmo, chans[0].nocc)

    return fx


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


class _ZeroXC1:
    """Within the block, pyscf's ``_get_vxc_deriv1`` (RKS or UKS) returns zeros, so that pyscf's
    ``make_h1`` forms everything but the XC term."""

    def __init__(self, mol, unrestricted):
        from pyscf.hessian import rks as rks_hess
        from pyscf.hessian import uks as uks_hess

        self.module = uks_hess if unrestricted else rks_hess
        self.mol = mol
        self.unrestricted = unrestricted

    def __enter__(self):
        nao, natm = self.mol.nao_nr(), self.mol.natm
        two = self.unrestricted

        def deriv1(*args, **kwargs):
            z = np.zeros((natm, 3, nao, nao))
            return (z, np.zeros_like(z)) if two else z

        self.saved = self.module._get_vxc_deriv1
        self.module._get_vxc_deriv1 = deriv1
        return self

    def __exit__(self, *exc):
        self.module._get_vxc_deriv1 = self.saved
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

    def hess_elec(self, mo_energy=None, mo_coeff=None, mo_occ=None, mo1=None, mo_e1=None, h1ao=None,
                  atmlst=None, max_memory=4000, verbose=None):
        """pyscf's ``hess_elec`` with the first-order Fock matrices and the coupled-perturbed equations in the
        MO basis: the XC part of the Fock derivatives from ``xc_h1mo``, the rest from pyscf's ``make_h1``
        (projected), and the response from :func:`cphf_operator`."""
        from pyscf.lib import logger

        mf = self.base
        mol = self.mol
        if mo_energy is None:
            mo_energy = mf.mo_energy
        if mo_occ is None:
            mo_occ = mf.mo_occ
        if mo_coeff is None:
            mo_coeff = mf.mo_coeff
        xch1 = None
        if mo1 is None and mo_e1 is None and h1ao is None and not getattr(self, "grid_response", False):
            grids = self.grids if getattr(self, "grids", None) is not None else mf.grids
            xch1 = xc_h1mo(mf._numint, mol, grids, mf.xc, mo_coeff, mo_occ)
        if xch1 is None:
            return super().hess_elec(mo_energy, mo_coeff, mo_occ, mo1, mo_e1, h1ao, atmlst, max_memory, verbose)
        log = logger.new_logger(self, verbose)
        t0 = (logger.process_clock(), logger.perf_counter())
        atmlst = list(range(mol.natm)) if atmlst is None else list(atmlst)
        unrestricted = np.asarray(mo_coeff).ndim == 3
        de2 = self.partial_hess_elec(mo_energy, mo_coeff, mo_occ, atmlst, max_memory, log)
        with _ZeroXC1(mol, unrestricted):
            h1ao = self.make_h1(mo_coeff, mo_occ, None, atmlst, log)
        t1 = log.timer_debug1("making H1", *t0)
        cs = [np.asarray(c) for c in mo_coeff] if unrestricted else [np.asarray(mo_coeff)]
        occs = [np.asarray(o) for o in mo_occ] if unrestricted else [np.asarray(mo_occ)]
        es = [np.asarray(e) for e in mo_energy] if unrestricted else [np.asarray(mo_energy)]
        h1s = list(h1ao) if unrestricted else [h1ao]
        nao = cs[0].shape[0]
        s1a = -mol.intor("int1e_ipovlp", comp=3)
        aoslices = mol.aoslice_by_atom()
        hmo, smo = [], []
        for c, o, h1 in zip(cs, occs, h1s):
            co = c[:, o > 0]
            hm = np.zeros((mol.natm, 3, c.shape[1], co.shape[1]))
            sm = np.zeros_like(hm)
            for ia in atmlst:
                p0, p1 = aoslices[ia][2:]
                hm[ia] = c.T @ np.asarray(h1[ia]) @ co
                s1ao = np.zeros((3, nao, nao))
                s1ao[:, p0:p1] += s1a[:, p0:p1]
                s1ao[:, :, p0:p1] += s1a[:, p0:p1].transpose(0, 2, 1)
                sm[ia] = c.T @ s1ao @ co
            hmo.append(hm)
            smo.append(sm)
        for hm, xh in zip(hmo, xch1):
            hm[atmlst] += xh[atmlst]
        mo1s, e1s = self._solve_mo1_mo(mo_energy, mo_coeff, mo_occ, hmo, smo, atmlst, max_memory, log)
        t1 = log.timer_debug1("solving MO1", *t1)
        fac = (4.0, 2.0) if not unrestricted else (2.0, 1.0)
        for s in range(len(cs)):
            eo = es[s][occs[s] > 0]
            occ_rows = np.flatnonzero(occs[s] > 0)
            hm, sm, m1, e1 = hmo[s], smo[s], mo1s[s], e1s[s]
            for i0, ia in enumerate(atmlst):
                s1oo = sm[ia][:, occ_rows]
                for j0, ja in enumerate(atmlst[: i0 + 1]):
                    de2[i0, j0] += fac[0] * np.einsum("xpi,ypi->xy", hm[ia], m1[ja])
                    de2[i0, j0] -= fac[0] * np.einsum("xpi,ypi,i->xy", sm[ia], m1[ja], eo)
                    de2[i0, j0] -= fac[1] * np.einsum("xij,yij->xy", s1oo, e1[ja])
        for i0 in range(len(atmlst)):
            for j0 in range(i0):
                de2[j0, i0] = de2[i0, j0].T
        log.timer("Mojo hessian", *t0)
        return de2

    def _solve_mo1_mo(self, mo_energy, mo_coeff, mo_occ, hmo, smo, atmlst, max_memory, log):
        """pyscf's ``solve_mo1`` with the first-order matrices already in the MO basis; MO-basis results."""
        from pyscf import lib
        from pyscf.hessian import rhf as rhf_hess
        from pyscf.hessian import uhf as uhf_hess
        from pyscf.scf import cphf, ucphf

        mf = self.base
        mol = self.mol
        unrestricted = len(hmo) == 2
        fx = cphf_operator(mf, mo_coeff, mo_occ)
        if fx is None:
            fx = (uhf_hess if unrestricted else rhf_hess).gen_vind(mf, mo_coeff, mo_occ)
        nao = mol.nao_nr()
        nmo, nocc = hmo[0].shape[2], sum(h.shape[3] for h in hmo)
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, max_memory * 0.9 - mem_now)
        blksize = max(2, int(max_memory * 1e6 / 8 / (nmo * nocc * 3 * 6)))
        mo1s = [[None] * mol.natm for _ in hmo]
        e1s = [[None] * mol.natm for _ in hmo]
        for a0, a1 in lib.prange(0, len(atmlst), blksize):
            atoms = atmlst[a0:a1]
            h1vo = [np.vstack([h[ia] for ia in atoms]) for h in hmo]
            s1vo = [np.vstack([sm[ia] for ia in atoms]) for sm in smo]
            tol = mf.conv_tol_cpscf * (a1 - a0)
            if unrestricted:
                mo1, e1 = ucphf.solve(fx, mo_energy, mo_occ, tuple(h1vo), tuple(s1vo), max_cycle=self.max_cycle,
                                      level_shift=self.level_shift, tol=tol)
            else:
                mo1, e1 = cphf.solve(fx, mo_energy, mo_occ, h1vo[0], s1vo[0], max_cycle=self.max_cycle,
                                     level_shift=self.level_shift, tol=tol)
                mo1, e1 = (mo1,), (e1,)
            for s in range(len(hmo)):
                m = np.asarray(mo1[s]).reshape(len(atoms), 3, nmo, -1)
                e = np.asarray(e1[s]).reshape(len(atoms), 3, m.shape[3], m.shape[3])
                for k, ia in enumerate(atoms):
                    mo1s[s][ia] = m[k]
                    e1s[s][ia] = e[k]
        return mo1s, e1s

    def solve_mo1(self, mo_energy, mo_coeff, mo_occ, h1ao_or_chkfile, fx=None, atmlst=None, max_memory=4000,
                  verbose=None):
        if fx is None:
            fx = cphf_operator(self.base, mo_coeff, mo_occ)
        return super().solve_mo1(mo_energy, mo_coeff, mo_occ, h1ao_or_chkfile, fx, atmlst, max_memory, verbose)


def accelerate(h):
    """Give a pyscf RKS/UKS Hessian object (``mf.Hessian()``, DF or not) the Mojo kernels, in place; returns it."""
    if not isinstance(h, _MojoHessMixin):
        from pyscf import lib

        lib.set_class(h, (_MojoHessMixin, type(h)))
    return h
