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
spin densities in one Mojo pass over the whole grid (:func:`mojoscf.dft._rho`)
and the cumulant part of Pi in a second (``ontop_pi``: per block of 128
points psi = C^T phi and t = L^T q with the pair products q_kl = psi_k psi_l,
two GEMMs), then evaluates the translated functional with pyscf's
``ot.eval_ot`` once on the whole grid.

The MC-PDFT nuclear gradient (``mc.nuc_grad_method()``, pyscf's
``grad.mcpdft``/``df.grad.mcpdft``) needs the effective potentials of the
functional (``pdft_veff.kernel``: :func:`pdft_veff_kernel`) and the
Hellmann-Feynman terms, whose grid part differentiates the on-top energy's
integrand with respect to the orbitals, the grid points and the Becke
weights (:func:`ontop_grad_terms`, inside :func:`hellmann_feynman_grad`);
both run on the Mojo kernels for translated LDA and GGA functionals.  The
response equations and the Lagrange terms are those of the state-averaged
CASSCF gradient (:mod:`mojoscf.casscf` for density-fitted references).

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
    dm1s = _dms.casdm1s_to_dm1s(ot, casdm1s, mo_coeff=mo_coeff, ncore=ncore, ncas=ncas)
    mo_cas = mo_coeff[:, ncore:][:, :ncas]
    coords, weights = dft._grid(ot.grids)
    kind = dft._KINDS[ot.xctype]
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


def ontop_grad_terms(ot, mol, mo_occ, occ, ncas, cascm2):
    """The on-top energy's integrand terms of pyscf's ``mcpdft_HellmanFeynman_grad``, each (natm, 3):
    the derivatives of the orbitals in rho and Pi (``de_xc``), of the grid points (``de_grid``) and of
    the Becke weights (``de_wgt``), for the occupied natural orbitals ``mo_occ`` (core then active,
    occupations ``occ``) and the spin-summed cumulant ``cascm2``.

    For each atom's grid (pyscf's ``grids_response_cc``: points, weights and their derivatives) the
    density (half per spin, as pyscf) and the on-top pair density come from the Mojo kernels and the
    functional and its derivatives from ``ot.eval_ot``.  The density's orbital terms are pyscf's XC
    gradient (``xc_grad_dm``) with the effective kernel v_rho + v_Pi rho/2, those of the cumulant part
    of Pi come from ``ontop_grad``; a grid point moves with its atom, so the grid term of an atom is
    minus the sum over all atoms of its points' orbital terms."""
    from pyscf.grad import rks as rks_grad

    natm, nao = mol.natm, mol.nao_nr()
    ncore = mo_occ.shape[1] - ncas
    dm1 = np.ascontiguousarray(((mo_occ * occ) @ mo_occ.T)[None])
    mo_cas = np.ascontiguousarray(mo_occ[:, ncore:])
    lmat = np.ascontiguousarray(np.asarray(cascm2, dtype=np.float64).reshape(ncas * ncas, ncas * ncas))
    kind = dft._KINDS[ot.xctype]
    tabs = integrals.basis_tables(mol)
    ext = get_extension()
    path, prefix = worker_blas()
    no_orbs, no_occs = np.zeros((1, nao, 0)), np.zeros((1, 0))
    de_xc, de_grid, de_wgt = np.zeros((natm, 3)), np.zeros((natm, 3)), np.zeros((natm, 3))
    part, part_pi = np.empty((natm, 3)), np.empty((natm, 3))
    for ia, (coords, w0, w1) in enumerate(rks_grad.grids_response_cc(ot.grids)):
        coords = np.ascontiguousarray(coords, dtype=np.float64)
        w0 = np.asarray(w0, dtype=np.float64)
        rho = dft._rho(mol, coords, kind, dm1)[0] * 0.5
        pi = ontop_pair_density(mol, coords, mo_cas, cascm2, 0)
        pi[0] += rho[0] * rho[0]
        eot, (vrho, vpi) = ot.eval_ot(np.stack((rho, rho)), pi, weights=w0)[:2]
        de_wgt += np.tensordot(eot, w1, axes=(0, 2))
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


def hellmann_feynman_grad(mc, ot, veff1, veff2, mo_coeff=None, ci=None, atmlst=None, mf_grad=None, verbose=None,
                          max_memory=None, auxbasis_response=False):
    """pyscf's ``grad.mcpdft.mcpdft_HellmanFeynman_grad`` (the Hellmann-Feynman part of the MC-PDFT
    gradient) with the on-top energy's grid terms from :func:`ontop_grad_terms`; everything else, and
    meta-GGA and fully translated functionals altogether, as pyscf's."""
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
    dm1 = tag_array(dm1, mo_coeff=mo_coeff, mo_occ=mo_occup)
    vj = mf_grad.get_jk(dm=dm1)[0]
    if auxbasis_response:
        de_aux += ot_hyb * np.squeeze(vj.aux[:, :, atmlst, :])
    casdm1, casdm2 = mc.fcisolver.make_rdm12(ci, ncas, nelecas)
    cascm2 = _dms.dm2_cumulant(casdm2, casdm1)
    de_xc, de_grid, de_wgt = (x[atmlst] for x in ontop_grad_terms(ot, mol, mo_occ, mo_occup[:nocc], ncas, cascm2))
    t0 = logger.timer(mc, "PDFT HlFn quadrature (mojoscf)", *t0)

    def coul_term(p0, p1):
        return np.tensordot(vj[:, p0:p1], dm1[p0:p1]) * 2

    de_hcore, de_coul, _, de_nuc, de_renorm = mcpdft_grad.sum_terms(mf_grad, mol, atmlst, dm1, dme0, coul_term,
                                                                     np.zeros((3, mol.nao_nr())))
    de_hcore *= ot_hyb
    de_coul *= ot_hyb
    de = de_nuc + de_hcore + de_coul + de_renorm + de_xc + de_grid + de_wgt
    if auxbasis_response:
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
    dm_core = mo_coeff[:, :ncore] @ mo_coeff[:, :ncore].T
    dm1s = np.asarray(dm1s, dtype=np.float64)
    dms = np.ascontiguousarray(np.stack((dm1s[0], dm1s[1], dm1s[0] + dm1s[1] - 2 * dm_core, 2 * dm_core)))
    coords, weights = dft._grid(ot.grids)
    kind = dft._KINDS[ot.xctype]
    rho4 = dft._rho(mol, coords, kind, dms)
    rho, rho_a, rho_c = rho4[:2], rho4[2], rho4[3]
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
    np.multiply(wpi, rho_c[0], out=wl[0, 0])
    np.multiply(wpi, rho_a[0], out=wl[1, 0])
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
