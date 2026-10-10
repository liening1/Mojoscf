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
    computed in one pass over the grid (``xc_hess_core``) for LDA, GGA and
    meta-GGA (without the laplacian), restricted (``mo_coeff`` (nao, nmo)) or
    unrestricted ((2, nao, nmo)).
    """
    from . import dft

    kind = dft._kind(ni, xc_code)
    if kind is None or not dft._fxc_ok(ni, mol, xc_code):
        return None
    coords, weights, dms, vxc, fxc = _xc_inputs(ni, mol, grids, xc_code, mo_coeff, mo_occ, kind)
    de2 = np.empty((mol.natm, mol.natm, 3, 3))
    path, prefix = worker_blas()
    get_extension().xc_hess(integrals.basis_tables(mol), coords, weights, int(kind), dms, vxc, fxc, _ao_atoms(mol),
                            de2, path, prefix)
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
    nvar = (1, 4, 5)[kind]
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
    if kind is None or not dft._fxc_ok(ni, mol, xc_code):
        return None
    coords, weights, dms, vxc, fxc = _xc_inputs(ni, mol, grids, xc_code, mo_coeff, mo_occ, kind)
    nspin = dms.shape[0]
    nvar = (1, 4, 5)[kind]
    wv = (vxc * weights).reshape(nspin, nvar, -1)
    if kind >= 1:
        wv[:, 0] *= 0.5
    if kind == 2:
        wv[:, 4] *= 0.5                                                          # the 1/2 of tau
    vgrad = dft._xc_grad(mol, coords, kind, np.ascontiguousarray(wv))            # (nspin, 3, nao, nao)
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
    with ``project``).  With exact integrals the Coulomb and exchange parts
    are J/K of the (symmetric) first-order densities of all vectors from one
    call of the Mojo kernels (:func:`mojoscf.dft.exact_jk`, long-range ones
    included), projected to C^T V C_o.  None (pyscf's operator runs) without
    an in-core DF tensor or exact integrals with pyscf's ``get_jk``, for
    solvent models, NLC, short-range-only hybrids, fractional occupations and
    MO tensors that do not fit in ``max_memory``.
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
    cs = [np.asarray(mo_coeff)] if not unrestricted else [np.asarray(c) for c in mo_coeff]
    chans = [_CPHFChannel(c, o) for c, o in zip(cs, occs)]
    cderi = dft._df_tensor(mf)
    exact = cderi is None
    if exact:
        if not dft.exact_jk_applies(mf):
            return None
    else:
        tensors = {0.0: cderi}
        for _, omega in kterms:
            if omega not in tensors:
                t = dft._df_tensor(mf, omega)
                if t is None:
                    return None
                tensors[omega] = t
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
    kfull = sum(c for c, omega in kterms if omega == 0)
    klr = [(c, omega) for c, omega in kterms if omega != 0 and c != 0]

    def exact_jk_terms(xs):
        """J - K of the first-order densities (exact integrals), projected to C^T V C_o per channel."""
        nset = len(xs[0])
        dms = []
        for ch, x in zip(chans, xs):
            d = ch.c @ x @ ch.co.T
            d = d + d.transpose(0, 2, 1)
            dms.append(d if unrestricted else 2.0 * d)
        ksc = 1.0 if unrestricted else 0.5
        dall = np.concatenate(dms)
        if kfull != 0:
            vj, vk = dft.exact_jk(mf, dall, 1, True, True)
            vk = vk * (ksc * kfull)
        else:
            vj, vk = dft.exact_jk(mf, dall, 1, True, False)[0], np.zeros_like(dall)
        for c, omega in klr:
            vk = vk + dft.exact_jk(mf, dall, 1, False, True, omega)[1] * (ksc * c)
        vj = vj.reshape(len(chans), nset, *vj.shape[1:]).sum(axis=0)
        vs = []
        for s, ch in enumerate(chans):
            v = vj - vk[s * nset:(s + 1) * nset]
            vs.append(ch.c.T @ v @ ch.co)
        return vs

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
        if exact:
            vs = exact_jk_terms(xs)
        else:
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


class _Metric:
    """The Coulomb metric V of the auxiliary basis, factorised as pyscf's ``_gen_metric_solver`` does:
    Cholesky V = L L^T, or (not positive definite) the eigenvectors above ``lindep``, V^+ = U U^T.

    ``whiten(r)`` gives L^-1 r (U^T r), so that r_a^T V^-1 r_b = whiten(r_a) . whiten(r_b);
    ``solve(r)`` gives V^-1 r.  ``r`` is (naux, ...) and is overwritten.
    """

    def __init__(self, v, lindep):
        import scipy.linalg

        try:
            self.low = scipy.linalg.cholesky(v, lower=True)
            self.u = None
        except scipy.linalg.LinAlgError:
            w, u = scipy.linalg.eigh(v)
            keep = w > lindep
            self.low = None
            self.u = u[:, keep] / np.sqrt(w[keep])

    def whiten(self, r):
        from scipy.linalg import blas

        r2 = np.ascontiguousarray(r).reshape(r.shape[0], -1)
        if self.low is None:
            return (self.u.T @ r2).reshape((-1,) + r.shape[1:])
        # X L^T = r^T on the Fortran-ordered view: X^T = L^-1 r, in place
        out = blas.dtrsm(1.0, self.low, r2.T, side=1, lower=1, trans_a=1, overwrite_b=1)
        return out.T.reshape(r.shape)

    def solve(self, r):
        from scipy.linalg import blas

        r2 = np.ascontiguousarray(r).reshape(r.shape[0], -1)
        if self.low is None:
            return (self.u @ (self.u.T @ r2)).reshape(r.shape)
        out = blas.dtrsm(1.0, self.low, r2.T, side=1, lower=1, trans_a=1, overwrite_b=1)
        out = blas.dtrsm(1.0, self.low, out, side=1, lower=1, trans_a=0, overwrite_b=1)
        return out.T.reshape(r.shape)


def _df_jk_reason(hessobj, mo_coeff, mo_occ):
    """Why :func:`df_jk_terms` cannot handle ``hessobj`` (None if it can)."""
    from pyscf.df import df_jk
    from pyscf.scf import hf, rohf, uhf

    from . import dft

    mf = hessobj.base
    mol = hessobj.mol
    if integrals.engine() != "mojo":
        return "the Mojo integral engine is not selected"
    if not isinstance(mf, df_jk._DFHF) or getattr(mf, "only_dfj", False) or not dft.plain_df(mf.with_df):
        return "not a density-fitted SCF (pyscf's DF class)"
    if getattr(hessobj, "auxbasis_response", 2) != 2:
        return "auxbasis_response below 2"
    if isinstance(mf, rohf.ROHF) or not isinstance(mf, (hf.RHF, uhf.UHF)):
        return "only RHF/RKS and UHF/UKS references"
    try:
        from pyscf.solvent._attach_solvent import _Solvation

        if isinstance(mf, _Solvation):
            return "solvent models"
    except ImportError:  # pragma: no cover
        pass
    if isinstance(mf, hf.KohnShamDFT) and mf.do_nlc():
        return "NLC functionals"
    if _hess_exchange_terms(mf) is None:
        return "short-range operator (omega < 0)"
    if getattr(mol, "omega", 0) or getattr(mf.with_df, "omega", None):
        return "range-separated Coulomb operator"
    reason = integrals.unsupported_reason(mol, two_electron=True)
    if reason is not None:
        return reason
    if not np.isrealobj(mo_coeff):
        return "complex orbitals"
    unrestricted = np.asarray(mo_coeff).ndim == 3
    full = 1.0 if unrestricted else 2.0
    occs = [np.asarray(o) for o in mo_occ] if unrestricted else [np.asarray(mo_occ)]
    if any(not np.all((o == 0) | (o == full)) for o in occs):
        return "fractional occupations"
    return None


def _exact_jk_reason(hessobj, mo_coeff, mo_occ):
    """Why :func:`exact_jk_partial` cannot handle ``hessobj`` (None if it can): exact integrals with pyscf's
    ``get_jk``, closed or high-spin open shells, no solvent model or NLC."""
    from pyscf.scf import hf, rohf, uhf

    from . import dft

    mf = hessobj.base
    if integrals.engine() != "mojo":
        return "the Mojo integral engine is not selected"
    if not dft.exact_jk_applies(mf):
        return "not exact integrals with pyscf's get_jk"
    if isinstance(mf, rohf.ROHF) or not isinstance(mf, (hf.RHF, uhf.UHF)):
        return "only RHF/RKS and UHF/UKS references"
    try:
        from pyscf.solvent._attach_solvent import _Solvation

        if isinstance(mf, _Solvation):
            return "solvent models"
    except ImportError:  # pragma: no cover
        pass
    if isinstance(mf, hf.KohnShamDFT) and mf.do_nlc():
        return "NLC functionals"
    if _hess_exchange_terms(mf) is None:
        return "short-range operator (omega < 0)"
    if not np.isrealobj(mo_coeff):
        return "complex orbitals"
    unrestricted = np.asarray(mo_coeff).ndim == 3
    full = 1.0 if unrestricted else 2.0
    occs = [np.asarray(o) for o in mo_occ] if unrestricted else [np.asarray(mo_occ)]
    if any(not np.all((o == 0) | (o == full)) for o in occs):
        return "fractional occupations"
    return None


def exact_jk_partial(hessobj, mo_coeff, mo_occ, tol=1e-14):
    """Coulomb/exchange part of the partial Hessian with exact integrals (natm, natm, 3, 3), or None.

    The second derivatives of E_J - sum_t c_t E_K(omega_t) at fixed density
    (pyscf's ``ej - hyb ek - (alpha - hyb) ek_lr`` of ``_partial_hess_ejk``)
    from :func:`mojoscf.integrals.hess2e`: one pass over the unique shell
    quartets per exchange operator, the second-derivative integrals contracted
    with the density products as they are produced.
    """
    if _exact_jk_reason(hessobj, mo_coeff, mo_occ) is not None:
        return None
    mf = hessobj.base
    mol = hessobj.mol
    unrestricted = np.asarray(mo_coeff).ndim == 3
    cs = [np.asarray(c) for c in mo_coeff] if unrestricted else [np.asarray(mo_coeff)]
    occs = [np.asarray(o) for o in mo_occ] if unrestricted else [np.asarray(mo_occ)]
    dms = [(c[:, o > 0] * o[o > 0]) @ c[:, o > 0].T for c, o in zip(cs, occs)]
    dmj = dms[0] + dms[1] if unrestricted else dms[0]
    kscale = 1.0 if unrestricted else 0.5
    terms = _hess_exchange_terms(mf)
    kfull = sum(c for c, omega in terms if omega == 0)
    de2 = integrals.hess2e(mol, dmj, np.array(dms), 1.0, kscale * kfull, tol=tol)
    for c, omega in terms:
        if omega:
            de2 += integrals.hess2e(mol, dmj, np.array(dms), 0.0, kscale * c, tol=tol, omega=omega)
    return de2


def exact_h1mo(hessobj, mo_coeff, mo_occ, tol=1e-14):
    """Two-electron part of the first-order Fock matrices with exact integrals, per spin C^T (dJ - c dK) C_o
    (natm, 3, nmo, nocc), or None.

    The derivative J/K matrices of the ground-state densities for every
    nuclear displacement come from one pass over the unique shell quartets
    (:func:`mojoscf.integrals.h1_jk`, a second one per long-range exchange
    term) in place of pyscf's per-atom ``_get_jk`` calls in ``make_h1``.
    """
    if _exact_jk_reason(hessobj, mo_coeff, mo_occ) is not None:
        return None
    mf = hessobj.base
    mol = hessobj.mol
    unrestricted = np.asarray(mo_coeff).ndim == 3
    cs = [np.asarray(c) for c in mo_coeff] if unrestricted else [np.asarray(mo_coeff)]
    occs = [np.asarray(o) for o in mo_occ] if unrestricted else [np.asarray(mo_occ)]
    dms = [(c[:, o > 0] * o[o > 0]) @ c[:, o > 0].T for c, o in zip(cs, occs)]
    dmj = dms[0] + dms[1] if unrestricted else dms[0]
    ksc = 1.0 if unrestricted else 0.5
    terms = _hess_exchange_terms(mf)
    kfull = sum(c for c, omega in terms if omega == 0)
    vj, vk = integrals.h1_jk(mol, [dmj], dms if kfull != 0 else [], tol=tol)
    vs = [vj[:, :, 0] - (ksc * kfull) * vk[:, :, s] if kfull != 0 else vj[:, :, 0].copy() for s in range(len(cs))]
    for c, omega in terms:
        if omega:
            vklr = integrals.h1_jk(mol, [], dms, tol=tol, omega=omega)[1]
            for s in range(len(cs)):
                vs[s] -= (ksc * c) * vklr[:, :, s]
    out = []
    for c, o, v in zip(cs, occs, vs):
        co = c[:, o > 0]
        out.append(np.einsum("mp,axmn,nq->axpq", c, v, co, optimize=True))
    return out


class _KChannel:
    """Occupied orbitals of one spin and their exchange intermediates for :func:`df_jk_terms`."""

    def __init__(self, c, occ):
        self.c = c
        self.occ = occ
        self.co = np.ascontiguousarray(c[:, occ > 0])
        self.nocc = self.co.shape[1]
        self.lmo = None       # (naux, nmo, nocc): C^T (mu nu|P) C_o
        self.ck = None        # (naux, nocc, nocc): V^-1 C_o^T (mu nu|P) C_o
        self.g = None         # (3, nao, naux, nocc): (nabla mu nu|P) C_o
        self.hb = None        # (natm, 3, nao, nocc): sum_P N_P[A]^T C_o[A] ck_P
        self.hd = None        # (natm, 3, nao, nocc): sum_{P on A} N_P^T C_o ck_P


def _hess_exchange_terms(mf):
    """[(coefficient, omega)] of the exact exchange in the energy (pyscf's DF Hessian), or None.

    hyb with the full-range operator and alpha - hyb with erf(omega r) / r
    (range-separated hybrids, short-range ones included as full minus
    long-range, as pyscf's ``partial_hess_elec`` and ``make_h1`` form them);
    None for omega < 0.
    """
    from pyscf.scf import hf

    if not isinstance(mf, hf.KohnShamDFT):
        return [(1.0, 0.0)]
    ni = mf._numint
    if not ni.libxc.is_hybrid_xc(mf.xc):
        return []
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mf.mol.spin)
    if omega < 0:
        return None
    terms = [(hyb, 0.0)] if hyb != 0 else []
    if omega > 0 and alpha - hyb != 0:
        terms.append((alpha - hyb, omega))
    return terms


def df_jk_terms(hessobj, mo_coeff=None, mo_occ=None, with_h1=True, tol=1e-14, verbose=None):
    """Coulomb and exchange terms of a density-fitted RKS/UKS (or RHF/UHF) Hessian, or None.

    Returns ``(de2, h1)``: ``de2`` (natm, natm, 3, 3) the J/K part of pyscf's
    ``partial_hess_elec`` (``ej - hyb ek - (alpha - hyb) ek_lr`` with the full
    auxiliary-basis response, ``auxbasis_response = 2``), and ``h1``
    (``with_h1``) per spin the J/K part of ``make_h1`` projected to the MO
    basis, C^T F^(A,x) C_o as (natm, 3, nmo, nocc_s) (MOs in their original
    order, occupied ones as in ``mo_occ > 0``).  None (pyscf's code runs) for
    the cases :func:`_df_jk_reason` names and when the intermediates do not
    fit in ``max_memory``.

    With B_P = (mu nu|P), V = (P|Q), c = V^-1 (B . D) and, per spin, the
    fitted occupied products c_P = V^-1 C_o^T B C_o, the DF energy
    1/2 rho^T V^-1 rho - kappa/2 sum_s sum_ij b_ij V^-1 b_ij has the second
    derivative (fixed orbitals)

        sum d2B_P Gamma_P - 1/2 sum d2V_PQ W_PQ + r_a^T V^-1 r_b - kappa sum_s rK_a^T V^-1 rK_b

    with Gamma_P = c_P D - kappa sum_s C_o c_P C_o^T, W = c c^T - kappa sum_s c_P . c_Q,
    r_a = (dB_a . D) - V_a c and rK_a = C_o^T dB_a C_o - V_a c (per occupied pair).
    The first term is the Mojo kernel ``hess_df3c`` (second-derivative
    three-centre integrals contracted as they are produced), the second uses
    pyscf's two-centre ``int2c2e_ipip1``.  The derivative integrals
    N_P = (nabla mu nu|P) come from the Mojo kernel ``int3c2e_ip1`` in blocks
    of auxiliary functions; from each block the J pieces (B-derivatives
    contracted with D and with c), the half-transformed N_P C_o and the
    exchange pieces of the Fock derivatives are formed with BLAS, the
    derivatives of the auxiliary centres following from translational
    invariance.  The exchange vectors rK are then built from N C_o in blocks
    of occupied orbitals, whitened with the Cholesky factor of V and
    contracted (the Hessian term) and contracted with V^-1 C^T B C_o (the
    Fock derivative), never as nao x nao matrices per atom.  The long-range
    exchange of range-separated hybrids is a second pass of the same with
    every integral (three-centre, metric and their derivatives) of
    erf(omega r) / r, as pyscf's ``with_df.range_coulomb(omega)``.
    """
    from pyscf.df import addons as df_addons
    from pyscf.lib import logger

    mf = hessobj.base
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    if mo_occ is None:
        mo_occ = mf.mo_occ
    if _df_jk_reason(hessobj, mo_coeff, mo_occ) is not None:
        return None
    terms = _hess_exchange_terms(mf)
    with_df = mf.with_df
    auxmol = with_df.auxmol
    if auxmol is None:
        auxmol = df_addons.make_auxmol(with_df.mol, with_df.auxbasis)
    if integrals.unsupported_reason(auxmol, two_electron=True) is not None:
        return None
    unrestricted = np.asarray(mo_coeff).ndim == 3
    cs = [np.asarray(c, dtype=np.float64) for c in mo_coeff] if unrestricted else [np.asarray(mo_coeff, np.float64)]
    occs = [np.asarray(o) for o in mo_occ] if unrestricted else [np.asarray(mo_occ)]
    log = logger.new_logger(hessobj, verbose)
    full = sum(c for c, omega in terms if omega == 0)
    res = _df_jk_pass(hessobj, auxmol, cs, occs, True, full, 0.0, with_h1, tol, log)
    if res is None:
        return None
    de2, h1 = res
    for c, omega in terms:
        if omega > 0:
            res = _df_jk_pass(hessobj, auxmol, cs, occs, False, c, omega, with_h1, tol, log)
            if res is None:
                return None
            de2 += res[0]
            if with_h1:
                for h, d in zip(h1, res[1]):
                    h += d
    return de2, h1


def _df_jk_pass(hessobj, auxmol, cs, occs, with_j, kcoef, omega, with_h1, tol, log):
    """One operator of :func:`df_jk_terms`: Coulomb (``with_j``) and exchange with coefficient ``kcoef``
    (``omega`` > 0: every integral of erf(omega r) / r); (de2, h1) or None if it does not fit in memory."""
    from pyscf import lib
    from pyscf.df.grad.rhf import LINEAR_DEP_THRESHOLD
    from pyscf.lib import logger

    from . import kernels

    mol = hessobj.mol
    unrestricted = len(cs) == 2
    kappa = kcoef * (1.0 if unrestricted else 2.0)
    chans = [_KChannel(c, o) for c, o in zip(cs, occs)]
    kchans = [ch for ch in chans if ch.nocc > 0] if kcoef != 0 else []
    nao, natm, naux = mol.nao_nr(), mol.natm, auxmol.nao_nr()
    npair = nao * (nao + 1) // 2
    nmo = cs[0].shape[1]
    dm = sum((ch.co * ch.occ[ch.occ > 0]) @ ch.co.T for ch in chans)

    # the intermediates kept in memory (words); pyscf's code goes out of core instead
    words = naux * npair + 4 * naux * naux
    for ch in chans:
        words += 2 * naux * nmo * ch.nocc if with_h1 else 0
    for ch in kchans:
        words += 3 * nao * naux * ch.nocc + 4 * naux * ch.nocc * ch.nocc
    free = hessobj.max_memory - lib.current_memory()[0]
    if words * 8e-6 > 0.7 * free:
        return None
    budget = max(200.0, 0.7 * free - words * 8e-6) * 1e6 / 8      # words for the blocks

    t1 = (logger.process_clock(), logger.perf_counter())
    ext = get_extension()
    boys = integrals._boys_table()
    tables = integrals.basis_tables(mol)
    aux_tables = integrals.basis_tables(auxmol)
    seq_path, seq_prefix = worker_blas()
    aoslices = [tuple(int(v) for v in s[2:]) for s in mol.aoslice_by_atom()]
    auxslices = [tuple(int(v) for v in s[2:]) for s in auxmol.aoslice_by_atom()]

    # fitted densities and the three-centre integrals in the MO basis
    j3c = integrals.int3c2e(mol, auxmol, omega)
    metric = _Metric(integrals.int2c2e(auxmol, omega), LINEAR_DEP_THRESHOLD)
    coef = np.zeros(naux)
    if with_j:
        dm_tril = lib.pack_tril(dm + dm.T)
        diag = np.arange(nao)
        dm_tril[diag * (diag + 1) // 2 + diag] *= 0.5
        coef = metric.solve((j3c @ dm_tril)[:, None])[:, 0]
    for ch in chans:
        if ch.nocc and (ch in kchans or (with_h1 and with_j)):
            lmo = kernels.df_mo(j3c, ch.c, ch.co)
            if ch in kchans:
                ch.ck = metric.solve(np.ascontiguousarray(lmo[:, ch.occ > 0]))
            if with_h1:
                ch.lmo = lmo
    j3c = None
    t1 = log.timer_debug1("DF Hessian: fitted densities, MO three-centre tensors", *t1)

    # second-derivative three-centre term
    de2 = np.empty((natm, natm, 3, 3))
    m = max((ch.nocc for ch in kchans), default=0)
    cns = np.zeros((max(len(kchans), 1), nao, m))
    xs = np.zeros((max(len(kchans), 1), naux, m * (m + 1) // 2))
    for s, ch in enumerate(kchans):
        cns[s, :, : ch.nocc] = ch.co
        pad = np.zeros((naux, m, m))
        pad[:, : ch.nocc, : ch.nocc] = ch.ck
        xs[s] = lib.pack_tril(pad)
    pad = None
    blk = max(1, min(naux, int(budget / 2 / npair)))
    ext.hess_df3c(tables, aux_tables, boys, np.ascontiguousarray(coef), np.ascontiguousarray(lib.pack_tril(dm)),
                  1.0 if with_j else 0.0, float(kappa), xs, cns, blk, float(tol), de2, seq_path, seq_prefix,
                  float(omega))
    cns = xs = None
    t1 = log.timer_debug1("DF Hessian: second-derivative three-centre term", *t1)

    # second-derivative two-centre term: -1/2 sum d2V_PQ W_PQ = C_AB - delta_AB sum_B' C_AB'
    w = np.outer(coef, coef)
    for ch in kchans:
        ck2 = ch.ck.reshape(naux, -1)
        w -= kappa * (ck2 @ ck2.T)
    onehot = np.zeros((naux, natm))
    for ia, (q0, q1) in enumerate(auxslices):
        onehot[q0:q1, ia] = 1.0
    with auxmol.with_range_coulomb(omega):
        wv = auxmol.intor("int2c2e_ipip1", comp=9).reshape(9, naux, naux)
    wv *= w
    w = None
    cab = (onehot.T @ wv @ onehot).reshape(3, 3, natm, natm).transpose(2, 3, 0, 1)
    wv = None
    de2 += cab
    for ia in range(natm):
        de2[ia, ia] -= cab[ia].sum(axis=0)
    t1 = log.timer_debug1("DF Hessian: second-derivative two-centre term", *t1)

    # first-derivative integrals, block by block of auxiliary shells
    with auxmol.with_range_coulomb(omega):
        v1 = auxmol.intor("int2c2e_ip1", comp=3)                 # (nabla P|Q)
    ndm = np.empty((3, naux, nao)) if with_j else None           # sum_nu N_P,mu nu D_mu nu
    nt = np.zeros((natm, 3, nao, nao)) if with_h1 and with_j else None   # sum_{P on A} c_P N_P
    for ch in kchans:
        ch.g = np.empty((3, nao, naux, ch.nocc))
        if with_h1:
            ch.hb = np.zeros((natm, 3, nao, ch.nocc))
            ch.hd = np.zeros((natm, 3, nao, ch.nocc))
    aux_loc = auxmol.ao_loc_nr()
    # blocks of first-derivative integrals: at most 1/4 of the budget and 256 MB (one buffer, reused)
    maxp = max(1, int(min(budget / 4, 32e6) / (3 * nao * nao + 3 * nao * (m + 1))))
    maxf = max(int(aux_loc[i + 1] - aux_loc[i]) for i in range(auxmol.nbas))
    n3buf = np.empty(3 * max(maxp, maxf) * nao * nao)
    sh0 = 0
    while sh0 < auxmol.nbas:
        sh1 = sh0 + 1
        while sh1 < auxmol.nbas and aux_loc[sh1 + 1] - aux_loc[sh0] <= maxp:
            sh1 += 1
        p0, p1 = int(aux_loc[sh0]), int(aux_loc[sh1])
        npf = p1 - p0
        n3 = n3buf[: 3 * npf * nao * nao].reshape(3, npf, nao, nao)
        n3[:] = 0.0
        ext.int3c2e_ip1(tables, aux_tables, boys, sh0, sh1, float(tol), n3, float(omega))
        if with_j:
            ndm[:, p0:p1] = np.einsum("xpmn,mn->xpm", n3, dm)
        segs = [(ia, max(q0, p0) - p0, min(q1, p1) - p0) for ia, (q0, q1) in enumerate(auxslices)
                if max(q0, p0) < min(q1, p1)]
        if with_h1 and with_j:
            for ia, s0, s1 in segs:
                nt[ia] += np.einsum("p,xpmn->xmn", coef[p0 + s0: p0 + s1], n3[:, s0:s1])
        for ch in kchans:
            nocc = ch.nocc
            ch.g[:, :, p0:p1] = (n3.reshape(-1, nao) @ ch.co).reshape(3, npf, nao, nocc).transpose(0, 2, 1, 3)
            if with_h1:
                mk = np.matmul(ch.co, ch.ck[p0:p1])              # (npf, nao, nocc): C_o ck_P
                for x in range(3):
                    nx = n3[x]
                    for ia, (a0, a1) in enumerate(aoslices):
                        if a1 > a0:
                            ch.hb[ia, x] += nx[:, a0:a1].reshape(-1, nao).T @ mk[:, a0:a1].reshape(-1, nocc)
                    for ia, s0, s1 in segs:
                        ch.hd[ia, x] += nx[s0:s1].reshape(-1, nao).T @ mk[s0:s1].reshape(-1, nocc)
        n3 = None
        sh0 = sh1
    n3buf = None
    t1 = log.timer_debug1("DF Hessian: first-derivative integrals and their contractions", *t1)

    h1 = [np.zeros((natm, 3, nmo, ch.nocc)) for ch in chans] if with_h1 else None
    if with_j:
        # Coulomb: r_a = y_a - V_a c,  y_(A,x),P = -2 sum_{mu on A} ndm_P,mu + delta(P on A) 2 sum_mu ndm_P,mu
        tot = ndm.sum(axis=2)
        v1c = v1 @ coef
        rj = np.empty((natm, 3, naux))
        for ia, (a0, a1) in enumerate(aoslices):
            q0, q1 = auxslices[ia]
            r = -2.0 * ndm[:, :, a0:a1].sum(axis=2)
            r[:, q0:q1] += 2.0 * tot[:, q0:q1] + v1c[:, q0:q1]
            r += np.einsum("xqp,q->xp", v1[:, q0:q1], coef[q0:q1])
            rj[ia] = r
        ndm = None
        rj = np.ascontiguousarray(rj.reshape(3 * natm, naux).T)          # (naux, 3 natm)
        vr = metric.solve(rj.copy()) if with_h1 else None
        rt = metric.whiten(rj)
        de2 += (rt.T @ rt).reshape(natm, 3, natm, 3).transpose(0, 2, 1, 3)
        rj = rt = None

        if with_h1:
            ntot = nt.sum(axis=0)
            for ch, h in zip(chans, h1):
                if ch.nocc:
                    h += (ch.lmo.reshape(naux, -1).T @ vr).reshape(nmo, ch.nocc, natm, 3).transpose(2, 3, 0, 1)
                    for ia, (a0, a1) in enumerate(aoslices):
                        for x in range(3):
                            f = nt[ia, x] + nt[ia, x].T
                            f[a0:a1] -= ntot[x, a0:a1]
                            f[:, a0:a1] -= ntot[x, a0:a1].T
                            h[ia, x] += ch.c.T @ (f @ ch.co)
            nt = ntot = vr = None
        t1 = log.timer_debug1("DF Hessian: Coulomb response", *t1)

    # exchange
    for ch in kchans:
        nocc = ch.nocc
        g = ch.g
        v1ck = (v1.reshape(3 * naux, naux) @ ch.ck.reshape(naux, -1)).reshape(3, naux, nocc, nocc)
        if with_h1:
            # sum_P (C^T dB_P C_o) ck_P = C^T [-E_A gam - hb_A + gam_A + hd_A],  gam_A = sum_{P on A} g_P ck_P
            gam = np.zeros((natm, 3, nao, nocc))
            for ia, (q0, q1) in enumerate(auxslices):
                if q1 > q0:
                    ckq = ch.ck[q0:q1].reshape(-1, nocc)
                    for x in range(3):
                        gam[ia, x] = g[x][:, q0:q1].reshape(nao, -1) @ ckq
            gtot = gam.sum(axis=0)
            hk = np.empty((natm, 3, nmo, nocc))
            for ia, (a0, a1) in enumerate(aoslices):
                for x in range(3):
                    t = gam[ia, x] + ch.hd[ia, x] - ch.hb[ia, x]
                    t[a0:a1] -= gtot[x, a0:a1]
                    hk[ia, x] = ch.c.T @ t
            gam = gtot = None
            ch.hb = ch.hd = None
            t1 = log.timer_debug1("DF Hessian: exchange Fock derivatives (integral part)", *t1)
        # rK_a,P[i, j] for a = (A, x), in blocks of rows i, each a contiguous (P, i, j) slab:
        # -(Z_A,P + Z_A,P^T) + delta(P on A) (Z_P + Z_P^T) - (V_a ck)_P with Z_A,P = C_o[A]^T g_P[A]
        # and -(V_a ck)_P = delta(P on A) (V1 ck)_P + sum_{Q on A} V1_QP ck_Q
        # blocks of rows: the response vectors of a block (and their whitened copy) within 1/4 of the
        # budget and 256 MB each
        rowb = max(1, min(nocc, int(min(budget / 4, 32e6) / (naux * nocc * (3 * natm + 4)))))
        gram = np.zeros((3 * natm, 3 * natm))
        hr = np.zeros((3 * natm, nmo, nocc)) if with_h1 else None
        # L^-1 C^T B C_o (nmo, naux', nocc): sum_P lmo_P (V^-1 r)_P = sum_Q (L^-1 lmo)_Q (L^-1 r)_Q
        lt = np.ascontiguousarray(metric.whiten(ch.lmo.copy()).transpose(1, 0, 2)) if with_h1 else None
        rkbuf = np.empty(3 * natm * naux * rowb * nocc)
        rtbuf = np.empty(3 * natm * naux * rowb * nocc)
        tmpbuf = np.empty(naux * rowb * nocc)
        zbuf = np.empty(3 * naux * rowb * nocc)
        for i0 in range(0, nocc, rowb):
            i1 = min(nocc, i0 + rowb)
            nb = i1 - i0
            # rk[P, A, x, i, j]: the auxiliary index outermost, so that L^-1 is one triangular solve
            rk = rkbuf[: 3 * natm * naux * nb * nocc].reshape(naux, natm, 3, nb, nocc)
            tmp = tmpbuf[: naux * nb * nocc].reshape(naux * nb, nocc)
            # Z_P + Z_P^T (all atoms) for the delta(P on A) term
            zall = zbuf[: 3 * naux * nb * nocc].reshape(3, naux, nb, nocc)
            cot = np.ascontiguousarray(ch.co[:, i0:i1].T)
            for x in range(3):
                z1 = (cot @ g[x].reshape(nao, -1)).reshape(nb, naux, nocc)
                np.matmul(np.ascontiguousarray(g[x][:, :, i0:i1]).reshape(nao, -1).T, ch.co, out=tmp)
                np.add(z1.transpose(1, 0, 2), tmp.reshape(naux, nb, nocc), out=zall[x])
            for ia, (a0, a1) in enumerate(aoslices):
                if a1 == a0:
                    rk[:, ia] = 0.0
                    continue
                ca = -ch.co[a0:a1]
                cat = np.ascontiguousarray(ca[:, i0:i1].T)
                for x in range(3):
                    ga = g[x, a0:a1]                                                     # (nA, naux, nocc)
                    z1 = (cat @ ga.reshape(a1 - a0, -1)).reshape(nb, naux, nocc)         # -Z_A,P[i, j] at (i, P)
                    gb = ga if nb == nocc else np.ascontiguousarray(ga[:, :, i0:i1])
                    np.matmul(gb.reshape(a1 - a0, -1).T, ca, out=tmp)                    # -Z_A,P[j, i] at (P, i)
                    np.add(z1.transpose(1, 0, 2), tmp.reshape(naux, nb, nocc), out=rk[:, ia, x])
            z1 = None
            for ia, (q0, q1) in enumerate(auxslices):
                if q1 == q0:
                    continue
                cq = ch.ck[q0:q1, i0:i1].reshape(q1 - q0, -1)
                for x in range(3):
                    r = rk[:, ia, x]
                    r[q0:q1] += zall[x, q0:q1]
                    r[q0:q1] += v1ck[x, q0:q1, i0:i1]
                    r += (v1[x, q0:q1].T @ cq).reshape(naux, nb, nocc)
            tmp = cot = zall = None
            rt = metric.whiten(rk.reshape(naux, -1)).reshape(-1, 3 * natm, nb * nocc)
            rk = None
            nw = rt.shape[0]
            rtt = rtbuf[: 3 * natm * nw * nb * nocc].reshape(3 * natm, nw, nb * nocc)
            rtt[:] = rt.transpose(1, 0, 2)
            rt = None
            r2 = rtt.reshape(3 * natm, -1)
            gram += r2 @ r2.T
            if with_h1:
                ltb = lt[:, :, i0:i1].reshape(nmo, -1)
                for a in range(3 * natm):
                    hr[a] += ltb @ rtt[a].reshape(-1, nocc)
                ltb = None
            rtt = r2 = None
        rtbuf = None
        rkbuf = tmpbuf = zbuf = lt = None
        de2 -= kappa * gram.reshape(natm, 3, natm, 3).transpose(0, 2, 1, 3)
        t1 = log.timer_debug1("DF Hessian: exchange response", *t1)
        if with_h1:
            hk += hr.reshape(natm, 3, nmo, nocc)
            h1[chans.index(ch)] -= kcoef * hk
        ch.g = None
    return de2, h1


def _hess_e1(hessobj, mo_energy, mo_coeff, mo_occ):
    """The one-electron (core Hamiltonian and overlap) part of the partial Hessian, (natm, natm, 3, 3),
    as pyscf's ``_partial_hess_ejk`` forms it."""
    from pyscf.hessian import rhf as rhf_hess

    mol = hessobj.mol
    unrestricted = np.asarray(mo_coeff).ndim == 3
    cs = list(mo_coeff) if unrestricted else [mo_coeff]
    occs = list(mo_occ) if unrestricted else [mo_occ]
    es = list(mo_energy) if unrestricted else [mo_energy]
    dm0 = 0.0
    dme0 = 0.0
    for c, o, e in zip(cs, occs, es):
        c, o, e = np.asarray(c), np.asarray(o), np.asarray(e)
        co = c[:, o > 0]
        dm0 = dm0 + (co * o[o > 0]) @ co.T
        dme0 = dme0 + (co * (o * e)[o > 0]) @ co.T
    s1aa, s1ab, _ = rhf_hess.get_ovlp(mol)
    hcore_deriv = hessobj.hcore_generator(mol)
    aoslices = mol.aoslice_by_atom()
    natm = mol.natm
    e1 = np.zeros((natm, natm, 3, 3))
    for ia in range(natm):
        p0, p1 = aoslices[ia][2:]
        e1[ia, ia] -= np.einsum("xypq,pq->xy", s1aa[:, :, p0:p1], dme0[p0:p1]) * 2
        for ja in range(ia + 1):
            q0, q1 = aoslices[ja][2:]
            e1[ia, ja] -= np.einsum("xypq,pq->xy", s1ab[:, :, p0:p1, q0:q1], dme0[p0:p1, q0:q1]) * 2
            e1[ia, ja] += np.einsum("xypq,pq->xy", hcore_deriv(ia, ja), dm0)
        for ja in range(ia):
            e1[ja, ia] = e1[ia, ja].T
    return e1


def _pyscf_xc_partial(hessobj, mo_coeff, mo_occ):
    """The XC part of the partial Hessian from pyscf's ``_get_vxc_diag``/``_get_vxc_deriv2`` (any functional
    pyscf's Hessian supports), contracted as pyscf's ``partial_hess_elec`` does."""
    from pyscf import lib
    from pyscf.hessian import rks as rks_hess
    from pyscf.hessian import uks as uks_hess

    mf = hessobj.base
    mol = hessobj.mol
    unrestricted = np.asarray(mo_coeff).ndim == 3
    max_memory = max(2000, mf.max_memory * 0.9 - lib.current_memory()[0])
    if unrestricted:
        dms = [(c[:, o > 0]) @ c[:, o > 0].T for c, o in zip(mo_coeff, mo_occ)]
        diag = uks_hess._get_vxc_diag(hessobj, mo_coeff, mo_occ, max_memory)
        deriv2 = uks_hess._get_vxc_deriv2(hessobj, mo_coeff, mo_occ, max_memory)
    else:
        co = mo_coeff[:, mo_occ > 0]
        dms = [2.0 * co @ co.T]
        diag = [rks_hess._get_vxc_diag(hessobj, mo_coeff, mo_occ, max_memory)]
        deriv2 = [rks_hess._get_vxc_deriv2(hessobj, mo_coeff, mo_occ, max_memory)]
    aoslices = mol.aoslice_by_atom()
    natm = mol.natm
    de2 = np.zeros((natm, natm, 3, 3))
    for ia in range(natm):
        p0, p1 = aoslices[ia][2:]
        for d, v, dm in zip(diag, deriv2, dms):
            de2[ia, ia] += np.einsum("xypq,pq->xy", d[:, :, p0:p1], dm[p0:p1]) * 2
            for ja in range(ia + 1):
                q0, q1 = aoslices[ja][2:]
                de2[ia, ja] += np.einsum("xypq,pq->xy", v[ia][:, :, q0:q1], dm[q0:q1]) * 2
        for ja in range(ia):
            de2[ja, ia] = de2[ia, ja].T
    return de2


def _pyscf_xc_h1mo(hessobj, mo_coeff, mo_occ):
    """pyscf's ``_get_vxc_deriv1`` projected as :func:`xc_h1mo` returns it."""
    from pyscf import lib
    from pyscf.hessian import rks as rks_hess
    from pyscf.hessian import uks as uks_hess

    mf = hessobj.base
    unrestricted = np.asarray(mo_coeff).ndim == 3
    max_memory = max(2000, mf.max_memory * 0.9 - lib.current_memory()[0])
    if unrestricted:
        vs = uks_hess._get_vxc_deriv1(hessobj, mo_coeff, mo_occ, max_memory)
        pairs = zip(mo_coeff, mo_occ, vs)
    else:
        pairs = [(mo_coeff, mo_occ, rks_hess._get_vxc_deriv1(hessobj, mo_coeff, mo_occ, max_memory))]
    out = []
    for c, o, v in pairs:
        c = np.asarray(c)
        co = c[:, np.asarray(o) > 0]
        out.append(np.einsum("pi,axpq,qj->axij", c, np.asarray(v), co, optimize=True))
    return out


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


def _is_ks(mf):
    from pyscf.scf import hf

    return isinstance(mf, hf.KohnShamDFT)


class _MojoHessMixin:
    """In front of pyscf's RHF/UHF/RKS/UKS Hessian classes (DF or not): the XC terms from the Mojo kernels,
    the Coulomb/exchange terms from :func:`df_jk_terms` (density fitting) or :func:`exact_jk_partial` and
    :func:`exact_h1mo` (exact integrals), the coupled-perturbed equations in the MO basis."""

    __name_mixin__ = "Mojo"

    def _grids(self):
        return self.grids if getattr(self, "grids", None) is not None else self.base.grids

    def _xc_partial(self, mo_coeff, mo_occ):
        """XC part of the partial Hessian (all atoms): the Mojo kernel, else pyscf's terms; zero for HF."""
        mf = self.base
        if not _is_ks(mf):
            return np.zeros((self.mol.natm, self.mol.natm, 3, 3))
        xc = xc_partial_hess(mf._numint, self.mol, self._grids(), mf.xc, mo_coeff, mo_occ)
        return xc if xc is not None else _pyscf_xc_partial(self, mo_coeff, mo_occ)

    def _xc_h1mo(self, mo_coeff, mo_occ):
        """The XC part of the projected Fock derivatives (:func:`xc_h1mo`; zeros for HF), or None."""
        mf = self.base
        if _is_ks(mf):
            return xc_h1mo(mf._numint, self.mol, self._grids(), mf.xc, mo_coeff, mo_occ)
        unrestricted = np.asarray(mo_coeff).ndim == 3
        cs = list(mo_coeff) if unrestricted else [mo_coeff]
        occs = list(mo_occ) if unrestricted else [mo_occ]
        return [np.zeros((self.mol.natm, 3, np.asarray(c).shape[1], int((np.asarray(o) > 0).sum())))
                for c, o in zip(cs, occs)]

    def partial_hess_elec(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None, max_memory=4000,
                          verbose=None):
        mf = self.base
        mol = self.mol
        if mo_energy is None:
            mo_energy = mf.mo_energy
        if mo_coeff is None:
            mo_coeff = mf.mo_coeff
        if mo_occ is None:
            mo_occ = mf.mo_occ
        atm = list(range(mol.natm)) if atmlst is None else list(atmlst)
        if getattr(self, "grid_response", False):
            return super().partial_hess_elec(mo_energy, mo_coeff, mo_occ, atmlst, max_memory, verbose)
        jk = df_jk_terms(self, mo_coeff, mo_occ, with_h1=False)
        if jk is None:
            jk = exact_jk_partial(self, mo_coeff, mo_occ)
            jk = None if jk is None else (jk,)
        if jk is not None:
            de2 = _hess_e1(self, mo_energy, mo_coeff, mo_occ) + jk[0] + self._xc_partial(mo_coeff, mo_occ)
            return de2[np.ix_(atm, atm)]
        if not _is_ks(mf):
            return super().partial_hess_elec(mo_energy, mo_coeff, mo_occ, atmlst, max_memory, verbose)
        xc = xc_partial_hess(mf._numint, mol, self._grids(), mf.xc, mo_coeff, mo_occ)
        if xc is None:
            return super().partial_hess_elec(mo_energy, mo_coeff, mo_occ, atmlst, max_memory, verbose)
        with _ZeroXC(mol, np.asarray(mo_coeff).ndim == 3):
            de2 = super().partial_hess_elec(mo_energy, mo_coeff, mo_occ, atmlst, max_memory, verbose)
        return de2 + xc[np.ix_(atm, atm)]

    def hess_elec(self, mo_energy=None, mo_coeff=None, mo_occ=None, mo1=None, mo_e1=None, h1ao=None,
                  atmlst=None, max_memory=4000, verbose=None):
        """pyscf's ``hess_elec`` with the first-order Fock matrices and the coupled-perturbed equations in the
        MO basis: the XC part of the Fock derivatives from ``xc_h1mo``, the Coulomb/exchange part from
        :func:`df_jk_terms` (density fitting) or pyscf's ``make_h1`` (projected), and the response from
        :func:`cphf_operator`."""
        from pyscf.lib import logger

        mf = self.base
        mol = self.mol
        if mo_energy is None:
            mo_energy = mf.mo_energy
        if mo_occ is None:
            mo_occ = mf.mo_occ
        if mo_coeff is None:
            mo_coeff = mf.mo_coeff
        if mo1 is not None or mo_e1 is not None or h1ao is not None or getattr(self, "grid_response", False):
            return super().hess_elec(mo_energy, mo_coeff, mo_occ, mo1, mo_e1, h1ao, atmlst, max_memory, verbose)
        log = logger.new_logger(self, verbose)
        t0 = (logger.process_clock(), logger.perf_counter())
        xch1 = self._xc_h1mo(mo_coeff, mo_occ)
        jk = df_jk_terms(self, mo_coeff, mo_occ, with_h1=True)
        if xch1 is None and jk is None:
            return super().hess_elec(mo_energy, mo_coeff, mo_occ, mo1, mo_e1, h1ao, atmlst, max_memory, verbose)
        t1 = log.timer_debug1("J/K and XC Fock derivatives", *t0)
        atmlst = list(range(mol.natm)) if atmlst is None else list(atmlst)
        unrestricted = np.asarray(mo_coeff).ndim == 3
        cs = [np.asarray(c) for c in mo_coeff] if unrestricted else [np.asarray(mo_coeff)]
        occs = [np.asarray(o) for o in mo_occ] if unrestricted else [np.asarray(mo_occ)]
        es = [np.asarray(e) for e in mo_energy] if unrestricted else [np.asarray(mo_energy)]
        if jk is not None:
            de2 = _hess_e1(self, mo_energy, mo_coeff, mo_occ) + jk[0] + self._xc_partial(mo_coeff, mo_occ)
            de2 = de2[np.ix_(atmlst, atmlst)]
            if xch1 is None:
                xch1 = _pyscf_xc_h1mo(self, mo_coeff, mo_occ)
            hcore_deriv = mf.nuc_grad_method().hcore_generator(mol)
            hmo = jk[1]
            for ia in atmlst:
                h1 = hcore_deriv(ia)
                for c, o, hm in zip(cs, occs, hmo):
                    hm[ia] += c.T @ h1 @ c[:, o > 0]
        else:
            de2 = self.partial_hess_elec(mo_energy, mo_coeff, mo_occ, atmlst, max_memory, log)
            hmo = exact_h1mo(self, mo_coeff, mo_occ)
            if hmo is not None:
                hcore_deriv = mf.nuc_grad_method().hcore_generator(mol)
                for ia in atmlst:
                    h1 = hcore_deriv(ia)
                    for c, o, hm in zip(cs, occs, hmo):
                        hm[ia] += c.T @ h1 @ c[:, o > 0]
            else:
                with _ZeroXC1(mol, unrestricted):
                    h1ao = self.make_h1(mo_coeff, mo_occ, None, atmlst, log)
                h1s = list(h1ao) if unrestricted else [h1ao]
                hmo = []
                for c, o, h1 in zip(cs, occs, h1s):
                    hm = np.zeros((mol.natm, 3, c.shape[1], int((o > 0).sum())))
                    for ia in atmlst:
                        hm[ia] = c.T @ np.asarray(h1[ia]) @ c[:, o > 0]
                    hmo.append(hm)
        t1 = log.timer_debug1("partial hessian and H1", *t1)
        nao = cs[0].shape[0]
        s1a = -mol.intor("int1e_ipovlp", comp=3)
        aoslices = mol.aoslice_by_atom()
        smo = []
        for c, o in zip(cs, occs):
            co = c[:, o > 0]
            sm = np.zeros((mol.natm, 3, c.shape[1], co.shape[1]))
            for ia in atmlst:
                p0, p1 = aoslices[ia][2:]
                s1ao = np.zeros((3, nao, nao))
                s1ao[:, p0:p1] += s1a[:, p0:p1]
                s1ao[:, :, p0:p1] += s1a[:, p0:p1].transpose(0, 2, 1)
                sm[ia] = c.T @ s1ao @ co
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
