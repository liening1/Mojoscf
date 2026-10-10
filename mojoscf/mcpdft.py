"""MC-PDFT (:mod:`pyscf.mcpdft`) with the Mojo XC kernels.

Multiconfiguration pair-density functional theory evaluates, after the
CASSCF, the on-top energy E_ot[rho, Pi] of each state on the DFT grid.
pyscf's ``energy_ot`` loops over blocks of grid points: the AO values, the
spin densities, the active orbitals psi_u on the grid and the on-top pair
density

    Pi = rho_a rho_b + 1/2 sum_uvxy psi_u psi_v psi_x psi_y L_uvxy

(L the cumulant of the active-space 2-RDM, with its gradient for the fully
translated functionals), then translates (rho, Pi) to effective spin
densities and calls libxc block by block.  :func:`energy_ot` computes the
spin densities and Pi in one Mojo pass over the whole grid
(:func:`ontop_densities`: per block of 128 points one GEMM for the core
density, one for the active orbitals psi = C^T phi, the active densities
and the cumulant contraction t = L^T q, q_kl = psi_k psi_l, from them;
meta-GGAs take the density pass of :mod:`mojoscf.dft` and ``ontop_pi``),
then evaluates the translated functional with pyscf's ``ot.eval_ot`` once
on the whole grid.  pyscf rebuilds the grid for every state (``ot.reset``);
the last build is reused while the molecule and grid settings are unchanged.

The MC-PDFT nuclear gradient (``mc.nuc_grad_method()``, pyscf's
``grad.mcpdft``/``df.grad.mcpdft``) needs the effective potentials of the
functional (``pdft_veff.kernel``: :func:`pdft_veff_kernel`) and the
Hellmann-Feynman terms, whose grid part differentiates the on-top energy's
integrand with respect to the orbitals, the grid points and the Becke
weights (:func:`ontop_grad_terms`, inside :func:`hellmann_feynman_grad`; the
weight derivatives from the Becke kernel of :mod:`mojoscf.dft`, the Coulomb
term from the Mojo derivative integrals); both run on the Mojo kernels for
translated LDA and GGA functionals.  The response equations and the
Lagrange terms are those of the state-averaged CASSCF gradient
(:mod:`mojoscf.casscf` for density-fitted references).

The replacements are installed when pyscf's modules are first imported
(:func:`install_on_import`, called by ``import mojoscf``), so that mojoscf
does not import :mod:`pyscf.mcpdft` itself.  Molecules the Mojo basis code
does not handle, laplacian meta-GGAs and non-symmetric densities keep
pyscf's code; so do the effective potentials and gradients of meta-GGA and
fully translated functionals.
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import sys

import numpy as np

from . import dft, integrals
from ._backend import get_extension, worker_blas


def ontop_pair_density(mol, coords, mo_cas, cascm2, deriv=0):
    """(1, ngrid) or, for ``deriv`` 1, (4, ngrid): the cumulant part 1/2 sum psi_u psi_v psi_x psi_y L_uvxy
    of the on-top pair density and its gradient, the active orbitals ``mo_cas`` (nao, ncas) and the
    cumulant ``cascm2`` (ncas, ncas, ncas, ncas) as pyscf's ``otpd.get_ontop_pair_density``."""
    if deriv not in (0, 1):
        raise ValueError(f"deriv {deriv}: only 0 and 1")
    coords = np.ascontiguousarray(coords, dtype=np.float64).reshape(-1, 3)
    mo_cas = np.ascontiguousarray(mo_cas, dtype=np.float64)
    ncas = mo_cas.shape[1]
    lt = np.ascontiguousarray(np.asarray(cascm2, dtype=np.float64).reshape(ncas * ncas, ncas * ncas).T)
    out = np.empty((1 if deriv == 0 else 4, coords.shape[0]))
    path, prefix = worker_blas()
    get_extension().ontop_pi(integrals.basis_tables(mol), coords, int(deriv), mo_cas, lt, out, path, prefix)
    return out


def ontop_densities(mol, coords, kind, mo_core, mo_cas, casdm1s, cascm2, pi_deriv=0):
    """In one Mojo pass: the spin densities (2, 1 or 4, ngrid) of mo_core mo_core^T + mo_cas casdm1s[s]
    mo_cas^T (``kind`` 0 LDA, 1 GGA: with gradients), the on-top pair density (1 or 4, ngrid) with the
    cumulant ``cascm2`` (its gradient for ``pi_deriv`` 1, which needs ``kind`` 1) and the core density."""
    if kind not in (0, 1) or (pi_deriv and kind != 1):
        raise ValueError(f"kind {kind}, pi_deriv {pi_deriv}")
    coords = np.ascontiguousarray(coords, dtype=np.float64).reshape(-1, 3)
    mo_cas = np.ascontiguousarray(mo_cas, dtype=np.float64)
    ncas = mo_cas.shape[1]
    ngrid = coords.shape[0]
    nao = mol.nao_nr()
    has_core = mo_core.shape[1] > 0
    dmc = np.ascontiguousarray(mo_core @ mo_core.T * 2) if has_core else np.zeros((nao, nao))
    lt = np.ascontiguousarray(np.asarray(cascm2, dtype=np.float64).reshape(ncas * ncas, ncas * ncas).T)
    rho = np.empty((2, (1, 4)[kind], ngrid))
    pi = np.empty(((1, 4)[pi_deriv], ngrid))
    rhoc = np.empty(ngrid)
    path, prefix = worker_blas()
    get_extension().ontop_density(integrals.basis_tables(mol), coords, int(kind), int(has_core), dmc, mo_cas,
                                  np.ascontiguousarray(casdm1s, dtype=np.float64).reshape(2, ncas, ncas), lt,
                                  int(pi_deriv), rho, pi, rhoc, path, prefix)
    return rho, pi, rhoc


def _mojo_ok(ot, casdm1s, casdm2, mo_coeff, hermi) -> bool:
    kind = dft._KINDS.get(ot.xctype)
    return (
        kind is not None
        and hermi == 1
        and getattr(ot, "Pi_deriv", 0) in (0, 1)
        and all(np.isrealobj(x) for x in (casdm1s, casdm2, mo_coeff))
        and integrals.engine() == "mojo"
        and dft.supported(ot.mol)
    )


def _ot_grid(ot):
    """Points and weights of the on-top functional's grid.  pyscf resets that grid before the energy of
    every state (``ot.reset`` in ``energy_tot``), which rebuilds it each time; the last build is kept on
    ``ot`` and reused while the molecule and the grid settings stay the same."""
    g = ot.grids
    if g.coords is not None:
        return dft._grid(g)
    mol = g.mol
    radii = getattr(g, "atomic_radii", None)
    key = (mol.atom_coords().tobytes(), mol.atom_charges().tobytes(), g.level, repr(g.atom_grid), g.radi_method,
           g.prune, g.radii_adjust, None if radii is None else np.asarray(radii).tobytes(), g.becke_scheme,
           getattr(g, "alignment", None), getattr(g, "symmetry", None))
    cached = getattr(ot, "_mojoscf_grid", None)
    if cached is not None and cached[0] == key:
        return cached[1], cached[2]
    coords, weights = dft._grid(g)
    ot._mojoscf_grid = (key, coords, weights)
    return coords, weights


def energy_ot(ot, casdm1s, casdm2, mo_coeff, ncore, max_memory=2000, hermi=1):
    """pyscf's ``otfnal.energy_ot`` (the on-top energy of one state) with the spin densities and the
    on-top pair density from the Mojo kernels and the translated functional evaluated once on the
    whole grid."""
    from pyscf.mcpdft import _dms

    if ot.xctype == "HF":
        return 0.0
    if not _mojo_ok(ot, casdm1s, casdm2, mo_coeff, hermi):
        return _orig["energy_ot"](ot, casdm1s, casdm2, mo_coeff, ncore, max_memory=max_memory, hermi=hermi)
    mol = ot.mol
    ncas = casdm2.shape[0]
    cascm2 = _dms.dm2_cumulant(casdm2, casdm1s)
    mo_cas = mo_coeff[:, ncore:][:, :ncas]
    coords, weights = _ot_grid(ot)
    kind = dft._KINDS[ot.xctype]
    if kind < 2:
        rho, pi, _ = ontop_densities(mol, coords, kind, mo_coeff[:, :ncore], mo_cas, casdm1s, cascm2, ot.Pi_deriv)
    else:
        dm1s = _dms.casdm1s_to_dm1s(ot, casdm1s, mo_coeff=mo_coeff, ncore=ncore, ncas=ncas)
        rho = dft._rho(mol, coords, kind, np.ascontiguousarray(dm1s, dtype=np.float64))
        pi = ontop_pair_density(mol, coords, mo_cas, cascm2, ot.Pi_deriv)
        pi[0] += rho[0, 0] * rho[1, 0]
        if ot.Pi_deriv:
            pi[1:4] += rho[0, 1:4] * rho[1, 0] + rho[0, 0] * rho[1, 1:4]
    eot = ot.eval_ot(rho, pi, dderiv=0, weights=weights)[0]
    return float(np.dot(eot, weights))


def _grad_ok(ot, mol) -> bool:
    return (
        ot.xctype in ("LDA", "GGA")
        and getattr(ot, "Pi_deriv", 0) == 0
        and integrals.engine() == "mojo"
        and dft.supported(mol)
    )


def _atom_grids(grids):
    """Per atom: (atom, points, weights, w0, setup) of pyscf's ``grids_response_cc`` with the Becke
    weights ``w0`` from the Mojo kernel (points grouped by boxes for the kernels' screening), or pyscf's
    generator (``setup`` None, its weight derivatives ``w1`` in place of the points' quadrature
    weights) for partitions the kernel does not cover."""
    from pyscf.grad import rks as rks_grad

    setup = dft.becke_setup(grids)
    if setup is None:
        for ia, (coords, w0, w1) in enumerate(rks_grad.grids_response_cc(grids)):
            yield ia, np.ascontiguousarray(coords, dtype=np.float64), w1, np.asarray(w0, dtype=np.float64), None
        return
    mol = grids.mol
    tab = grids.gen_atomic_grids(mol, grids.atom_grid, grids.radi_method, grids.level, grids.prune)
    atm = mol.atom_coords()
    for ia in range(mol.natm):
        coords, vol = tab[mol.atom_symbol(ia)]
        coords = coords + atm[ia]
        idx = dft.group_grids(mol, coords)
        coords, vol = np.ascontiguousarray(coords[idx]), np.ascontiguousarray(vol[idx])
        yield ia, coords, vol, dft.becke_response(mol, setup, coords, vol, ia)[0], setup


def ontop_grad_terms(ot, mol, mo_occ, occ, ncas, cascm2):
    """The on-top energy's integrand terms of pyscf's ``mcpdft_HellmanFeynman_grad``, each (natm, 3):
    the derivatives of the orbitals in rho and Pi (``de_xc``), of the grid points (``de_grid``) and of
    the Becke weights (``de_wgt``), for the occupied natural orbitals ``mo_occ`` (core then active,
    occupations ``occ``) and the spin-summed cumulant ``cascm2``.

    For each atom's grid (pyscf's ``grids_response_cc``: its points and Becke weights) the density
    (half per spin, as pyscf) and the on-top pair density come from the Mojo kernels and the functional
    and its derivatives from ``ot.eval_ot``.  The density's orbital terms are pyscf's XC gradient
    (``xc_grad_dm``) with the effective kernel v_rho + v_Pi rho/2, those of the cumulant part of Pi come
    from ``ontop_grad``; a grid point moves with its atom, so the grid term of an atom is minus the sum
    over all atoms of its points' orbital terms.  The weight derivatives contracted with the energy
    density come from the Becke kernel (:func:`mojoscf.dft.becke_response`)."""
    natm, nao = mol.natm, mol.nao_nr()
    ncore = mo_occ.shape[1] - ncas
    dm1 = np.ascontiguousarray(((mo_occ * occ) @ mo_occ.T)[None])
    mo_cas = np.ascontiguousarray(mo_occ[:, ncore:])
    mo_core = mo_occ[:, :ncore] * np.sqrt(np.asarray(occ[:ncore]) * 0.5)
    half = np.diag(np.asarray(occ[ncore:], dtype=np.float64) * 0.5)
    casdm1s = np.stack((half, half))
    lmat = np.ascontiguousarray(np.asarray(cascm2, dtype=np.float64).reshape(ncas * ncas, ncas * ncas))
    kind = dft._KINDS[ot.xctype]
    tabs = integrals.basis_tables(mol)
    ext = get_extension()
    path, prefix = worker_blas()
    no_orbs, no_occs = np.zeros((1, nao, 0)), np.zeros((1, 0))
    de_xc, de_grid, de_wgt = np.zeros((natm, 3)), np.zeros((natm, 3)), np.zeros((natm, 3))
    part, part_pi = np.empty((natm, 3)), np.empty((natm, 3))
    for ia, coords, vol, w0, setup in _atom_grids(ot.grids):
        rho2, pi, _ = ontop_densities(mol, coords, kind, mo_core, mo_cas, casdm1s, cascm2, 0)
        rho = rho2[0]
        eot, (vrho, vpi) = ot.eval_ot(rho2, pi, weights=w0)[:2]
        if setup is None:
            de_wgt += np.tensordot(eot, vol, axes=(0, 2))
        else:
            de_wgt += dft.becke_response(ot.grids.mol, setup, coords, vol, ia, eot)[1]
        wv = np.array(vrho, dtype=np.float64).reshape(-1, w0.size)
        wv[0] += vpi[0] * rho[0]
        wv *= w0
        if kind:
            wv[0] *= 0.5
        ext.xc_grad_dm(tabs, coords, kind, np.ascontiguousarray(wv[None]), dm1, no_orbs, no_occs, part, path, prefix)
        ext.ontop_grad(tabs, coords, np.ascontiguousarray(w0 * vpi[0]), mo_cas, lmat, part_pi, path, prefix)
        part += part_pi
        de_xc += part
        de_grid[ia] -= part.sum(0)
    return de_xc, de_grid, de_wgt


def _coulomb_gradient(mf, mol, dm, auxbasis_response):
    """The gradient (natm, 3) of the Coulomb energy 1/2 Tr(D J[D]) as pyscf's MC-PDFT gradient forms it
    from ``mf_grad.get_jk`` (with the auxiliary-basis response for density fitting), from the Mojo
    derivative integrals: DF (pyscf's ``df.DF``) with the response, or exact integrals; None otherwise."""
    from pyscf.df import df as pyscf_df

    with_df = getattr(mf, "with_df", None)
    if with_df is not None:
        if not (auxbasis_response and isinstance(with_df, pyscf_df.DF) and getattr(with_df, "auxmol", None) is not None
                and integrals.available(mol, two_electron=True)
                and integrals.available(with_df.auxmol, two_electron=True)):
            return None
        return integrals.grad2e_df(mol, with_df.auxmol, dm, [], [], j_factor=1.0, k_factor=0.0)
    if not integrals.available(mol, two_electron=True):
        return None
    return integrals.grad2e(mol, dm, dm, j_factor=1.0, k_factor=0.0)


def hellmann_feynman_grad(mc, ot, veff1, veff2, mo_coeff=None, ci=None, atmlst=None, mf_grad=None, verbose=None,
                          max_memory=None, auxbasis_response=False):
    """pyscf's ``grad.mcpdft.mcpdft_HellmanFeynman_grad`` (the Hellmann-Feynman part of the MC-PDFT
    gradient) with the on-top energy's grid terms from :func:`ontop_grad_terms` and the Coulomb term
    from the Mojo derivative integrals (:func:`_coulomb_gradient`; pyscf computed J and K of the
    gradient for it); everything else, and meta-GGA and fully translated functionals altogether, as
    pyscf's."""
    from pyscf.grad import mcpdft as mcpdft_grad
    from pyscf.lib import logger, tag_array
    from pyscf.mcpdft import _dms
    from pyscf.mcscf.casci import cas_natorb

    orig = _orig["hf_grad"]
    if not _grad_ok(ot, mc.mol):
        return orig(mc, ot, veff1, veff2, mo_coeff=mo_coeff, ci=ci, atmlst=atmlst, mf_grad=mf_grad,
                    verbose=verbose, max_memory=max_memory, auxbasis_response=auxbasis_response)
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    if ci is None:
        ci = mc.ci
    if mf_grad is None:
        mf_grad = mc.get_rhf_base().nuc_grad_method()
    if mc.frozen is not None:
        raise NotImplementedError
    t0 = (logger.process_clock(), logger.perf_counter())
    mol = mc.mol
    ncore, ncas = mc.ncore, mc.ncas
    nocc = ncore + ncas
    nelecas = mc.nelecas
    mo_core = mo_coeff[:, :ncore]
    mo_cas = mo_coeff[:, ncore:nocc]
    casdm1, casdm2 = mc.fcisolver.make_rdm12(ci, ncas, nelecas)
    spin = abs(nelecas[0] - nelecas[1])
    omega, _, hyb = ot._numint.rsh_and_hybrid_coeff(ot.otxc, spin=spin)
    if abs(omega) > 1e-11:
        raise NotImplementedError("range-separated on-top functionals")
    if abs(hyb[0] - hyb[1]) > 1e-11:
        raise NotImplementedError("hybrid on-top functionals with different exchange,correlation components")
    cas_hyb = hyb[0]
    ot_hyb = 1.0 - cas_hyb
    if cas_hyb > 1e-11:
        if auxbasis_response:
            from pyscf.df.grad import casscf as casscf_grad
        else:
            from pyscf.grad import casscf as casscf_grad
        de_cas = cas_hyb * casscf_grad.Gradients(mc).grad_elec(mo_coeff=mo_coeff, ci=ci, atmlst=atmlst,
                                                               verbose=verbose)
    dm_core = 2 * mo_core @ mo_core.T
    dm_cas = mo_cas @ casdm1 @ mo_cas.T
    gfock = mcpdft_grad.gfock_sym(mc, mo_coeff, casdm1, casdm2, ot_hyb * mc.get_hcore() + veff1, veff2)
    dme0 = mo_coeff @ (0.5 * (gfock + gfock.T)) @ mo_coeff.T
    del gfock
    if atmlst is None:
        atmlst = range(mol.natm)
    atmlst = list(atmlst)
    de_aux = np.zeros((len(atmlst), 3))
    mo_coeff, ci, mo_occup = cas_natorb(mc, mo_coeff=mo_coeff, ci=ci)
    mo_occ = mo_coeff[:, :nocc]
    dm1 = dm_core + dm_cas
    de_j = _coulomb_gradient(mc._scf, mol, dm1, auxbasis_response)
    if de_j is None:
        dm1 = tag_array(dm1, mo_coeff=mo_coeff, mo_occ=mo_occup)
        vj = mf_grad.get_jk(dm=dm1)[0]
        if auxbasis_response:
            de_aux += ot_hyb * np.squeeze(vj.aux[:, :, atmlst, :])
    else:
        de_aux += ot_hyb * de_j[atmlst]
    casdm1, casdm2 = mc.fcisolver.make_rdm12(ci, ncas, nelecas)
    cascm2 = _dms.dm2_cumulant(casdm2, casdm1)
    de_xc, de_grid, de_wgt = (x[atmlst] for x in ontop_grad_terms(ot, mol, mo_occ, mo_occup[:nocc], ncas, cascm2))
    t0 = logger.timer(mc, "PDFT HlFn quadrature (mojoscf)", *t0)

    def coul_term(p0, p1):
        return 0.0 if de_j is not None else np.tensordot(vj[:, p0:p1], dm1[p0:p1]) * 2

    de_hcore, de_coul, _, de_nuc, de_renorm = mcpdft_grad.sum_terms(mf_grad, mol, atmlst, dm1, dme0, coul_term,
                                                                     np.zeros((3, mol.nao_nr())))
    de_hcore *= ot_hyb
    de_coul *= ot_hyb
    de = de_nuc + de_hcore + de_coul + de_renorm + de_xc + de_grid + de_wgt
    if auxbasis_response or de_j is not None:
        de += de_aux
    if cas_hyb > 1e-11:
        de += de_cas
    logger.timer(mc, "PDFT HlFn total", *t0)
    return de


def pdft_veff_kernel(ot, dm1s, cascm2, mo_coeff, ncore, ncas, max_memory=2000, hermi=1, paaa_only=False,
                     aaaa_only=False, jk_pc=False):
    """pyscf's ``pdft_veff.kernel`` (the on-top energy and the MC-PDFT effective potentials, ``veff1`` in
    the AO basis and ``veff2`` as a ``pdft_eff._ERIS``) for translated LDA/GGA functionals with
    ``paaa_only`` or ``aaaa_only`` (what the MC-PDFT gradients use), from the Mojo kernels: the total,
    active and core densities in one pass, the on-top pair density in a second, the functional once on
    the whole grid (``ot.eval_ot``), ``veff1`` and the AO matrices of the core and active one-body terms
    of ``veff2`` with the XC potential-matrix kernel, and its paaa block (sum_p phi_mu v_Pi psi_u psi_v
    psi_w) in one more pass (``ontop_paaa``); other cases run pyscf's.  ``veff2.energy_core`` is the
    trace of the core block of ``vhf_c`` without the active term (only L-PDFT, which does not take this
    path, reads it)."""
    from pyscf.mcpdft.pdft_eff import _ERIS

    if not (_grad_ok(ot, ot.mol) and hermi == 1 and (paaa_only or aaaa_only) and not jk_pc
            and np.isrealobj(dm1s) and np.isrealobj(mo_coeff)):
        return _orig["veff"](ot, dm1s, cascm2, mo_coeff, ncore, ncas, max_memory=max_memory, hermi=hermi,
                             paaa_only=paaa_only, aaaa_only=aaaa_only, jk_pc=jk_pc)
    omega, _, hyb = ot._numint.rsh_and_hybrid_coeff(ot.otxc)
    if abs(omega) > 1e-11:
        raise NotImplementedError("range-separated on-top functionals")
    if abs(hyb[0] - hyb[1]) > 1e-11:
        raise NotImplementedError(
            "effective potential for hybrid functionals with different exchange, correlations components")
    mol = ot.mol
    nocc = ncore + ncas
    nmo = mo_coeff.shape[1]
    mo_cas = np.ascontiguousarray(mo_coeff[:, ncore:nocc])
    mo_core = mo_coeff[:, :ncore]
    dm1s = np.asarray(dm1s, dtype=np.float64)
    coords, weights = _ot_grid(ot)
    kind = dft._KINDS[ot.xctype]
    # the active spin 1-RDMs of dm1s (pyscf's core + mo_cas casdm1s mo_cas^T) for the one-pass kernel
    proj = mo_cas.T @ mol.intor_symmetric("int1e_ovlp")
    casdm1s = np.einsum("ui,sij,vj->suv", proj, dm1s, proj)
    rec = (mo_core @ mo_core.T)[None] + np.einsum("iu,suv,jv->sij", mo_cas, casdm1s, mo_cas)
    if abs(rec - dm1s).max() < 1e-10 * max(1.0, abs(dm1s).max()):
        rho, pi, rho_c = ontop_densities(mol, coords, kind, mo_core, mo_cas, casdm1s, cascm2, 0)
        rho_a = rho[0, 0] + rho[1, 0] - rho_c
    else:
        dm_core = mo_core @ mo_core.T
        dms = np.ascontiguousarray(np.stack((dm1s[0], dm1s[1], dm1s[0] + dm1s[1] - 2 * dm_core, 2 * dm_core)))
        rho4 = dft._rho(mol, coords, kind, dms)
        rho, rho_a, rho_c = rho4[:2], rho4[2, 0], rho4[3, 0]
        pi = ontop_pair_density(mol, coords, mo_cas, cascm2, 0)
        pi[0] += rho[0, 0] * rho[1, 0]
    eot, (vrho, vpi) = ot.eval_ot(rho, pi, weights=weights)[:2]
    e_ot = float(np.dot(eot, weights))
    wv = np.array(vrho, dtype=np.float64).reshape(-1, weights.size) * weights
    wv[0] *= 0.5
    veff1 = dft._vmat(mol, coords, kind, wv[None])[0]
    wpi = np.ascontiguousarray(weights * vpi[0])
    # vhf_c and the active term: kernels v_Pi rho_c / 2 and v_Pi rho_a / 2 (LDA-like, w_0 halved)
    wl = np.empty((2, 1, weights.size))
    np.multiply(wpi, rho_c, out=wl[0, 0])
    np.multiply(wpi, rho_a, out=wl[1, 0])
    wl *= 0.25
    vc, va = dft._vmat(mol, coords, 0, wl)
    veff2 = _ERIS(mol, mo_coeff, ncore, ncas, paaa_only=paaa_only, aaaa_only=aaaa_only, jk_pc=jk_pc,
                  verbose=ot.verbose, stdout=ot.stdout)
    veff2.vhf_c = mo_coeff.T @ vc @ mo_coeff
    veff2.energy_core = np.trace(veff2.vhf_c[:ncore, :ncore])
    if paaa_only:
        vhf_a = mo_coeff.T @ va @ mo_coeff
        vhf_a[ncore:nocc, :] = vhf_a[:, ncore:nocc] = 0.0
        veff2.vhf_c += vhf_a
    paaa_ao = np.empty((mol.nao_nr(), ncas ** 3))
    path, prefix = worker_blas()
    get_extension().ontop_paaa(integrals.basis_tables(mol), coords, wpi, mo_cas, paaa_ao, path, prefix)
    if aaaa_only:
        veff2.papa[ncore:nocc, :, ncore:nocc, :] += (mo_cas.T @ paaa_ao).reshape((ncas,) * 4)
    else:
        paaa = (mo_coeff.T @ paaa_ao).reshape(nmo, ncas, ncas, ncas)
        veff2.papa[:, :, ncore:nocc, :] += paaa
        veff2.papa[ncore:nocc, :, :, :] += paaa.transpose(2, 3, 0, 1)
        veff2.papa[ncore:nocc, :, ncore:nocc, :] -= paaa[ncore:nocc, :, :, :]
    veff2._finalize()
    return e_ot, veff1, veff2


_orig = {}


def install():
    """Replace pyscf's ``otfnal.energy_ot``, ``pdft_veff.kernel`` and
    ``grad.mcpdft.mcpdft_HellmanFeynman_grad`` (idempotent); the originals stay as ``_mojoscf_orig``."""
    from pyscf.grad import mcpdft as mcpdft_grad
    from pyscf.mcpdft import otfnal, pdft_veff

    _patch_otfnal(otfnal)
    _patch_veff(pdft_veff)
    _patch_grad(mcpdft_grad)


def _patch_otfnal(otfnal):
    if getattr(otfnal.otfnal.energy_ot, "_mojoscf_orig", None) is not None:
        return
    _orig["energy_ot"] = otfnal.otfnal.energy_ot

    def mojo_energy_ot(ot, casdm1s, casdm2, mo_coeff, ncore, max_memory=2000, hermi=1):
        return energy_ot(ot, casdm1s, casdm2, mo_coeff, ncore, max_memory=max_memory, hermi=hermi)

    mojo_energy_ot._mojoscf_orig = _orig["energy_ot"]
    otfnal.otfnal.energy_ot = mojo_energy_ot


def _patch_veff(pdft_veff):
    if getattr(pdft_veff.kernel, "_mojoscf_orig", None) is not None:
        return
    _orig["veff"] = pdft_veff.kernel

    def kernel(*args, **kwargs):
        return pdft_veff_kernel(*args, **kwargs)

    kernel._mojoscf_orig = _orig["veff"]
    pdft_veff.kernel = kernel


def _patch_grad(mcpdft_grad):
    fn = mcpdft_grad.mcpdft_HellmanFeynman_grad
    if getattr(fn, "_mojoscf_orig", None) is not None:
        return
    _orig["hf_grad"] = fn

    def mcpdft_HellmanFeynman_grad(*args, **kwargs):
        return hellmann_feynman_grad(*args, **kwargs)

    mcpdft_HellmanFeynman_grad._mojoscf_orig = fn
    mcpdft_grad.mcpdft_HellmanFeynman_grad = mcpdft_HellmanFeynman_grad


_TARGETS = {"pyscf.mcpdft.otfnal": _patch_otfnal, "pyscf.mcpdft.pdft_veff": _patch_veff,
            "pyscf.grad.mcpdft": _patch_grad}


class _InstallOnImport(importlib.abc.MetaPathFinder):
    """Patches pyscf's MC-PDFT modules right after their first import, so that importing mojoscf does
    not import them."""

    def find_spec(self, fullname, path=None, target=None):
        patch = _TARGETS.get(fullname)
        if patch is None or fullname in sys.modules:
            return None
        if self in sys.meta_path:
            sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            if any(name not in sys.modules and name != fullname for name in _TARGETS):
                sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return spec
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            patch(module)

        spec.loader.exec_module = exec_and_patch
        return spec


def install_on_import():
    """Patch pyscf's MC-PDFT modules that are imported now and the others when they are."""
    for name, patch in _TARGETS.items():
        if name in sys.modules:
            patch(sys.modules[name])
    if any(name not in sys.modules for name in _TARGETS) and not any(
        isinstance(f, _InstallOnImport) for f in sys.meta_path
    ):
        sys.meta_path.insert(0, _InstallOnImport())
