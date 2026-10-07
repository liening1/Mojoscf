"""Implicit solvation (pyscf.solvent PCM family and SMD) with the Mojo integral kernels.

pyscf's PCM (C-PCM, COSMO, IEF-PCM, SS(V)PE) and SMD represent the solvent
response by Gaussian charges on the molecular surface.  Every SCF cycle
needs the electrostatic potential of the electron density at the surface
points and the potential matrix of the induced charges; pyscf evaluates both
from three-centre integrals over the surface "fakemol" (``int3c2e``, blocks
of points, contracted with ``einsum``) and solves the dense PCM equations
K q = R v twice.  With :func:`attach` the solvent object uses

* ``_get_v``: :func:`mojoscf.integrals.int1e_grids_dm` (one pass over the
  shell pairs, the pair's Hermite matrices contracted with the density
  first, the surface charges as the SIMD lanes of the ERI kernel);
* ``_get_vmat``: :func:`mojoscf.integrals.int1e_grids_sum` (the QM/MM
  potential kernel with Gaussian charges);
* the PCM equations: an LU factorisation of K, kept with pyscf's other
  intermediates (``build`` resets it), instead of two dense solves per call;
* the gradient's integral term (pyscf's ``grad_qv``): one density-contracted
  pass of :func:`mojoscf.integrals.mm_grad_terms` gives the QM-atom term and
  the forces on the surface charges;
* the surface matrices S and D (``get_D_S`` in ``build``) and, in the
  gradient's solver term (``grad_solver``), their geometry derivatives
  contracted with the PCM vectors as they are evaluated
  (``_mojo/pcm.mojo``), instead of pyscf's (3, n, n) derivative arrays and
  einsums; erf comes from the engine's Boys function.

The surface itself, the switching-function derivatives, the nuclear and the
cavity-dispersion (SMD) terms stay pyscf's.  Results agree with pyscf's to
the precision of the integrals.

>>> mf = dft.RKS(mol, xc="b3lyp").density_fit().PCM()
>>> mojoscf.dft.accelerate(mf)        # attaches the solvent kernels too
>>> mojoscf.solvent.attach(scf.RHF(mol).PCM())   # any SCF object with a PCM/SMD solvent
"""
from __future__ import annotations

import numpy as np
import scipy.linalg
from pyscf import lib

from . import integrals
from ._backend import get_extension

__all__ = ["attach", "supported", "get_D_S", "grad_qv", "grad_solver"]


def supported(pcmobj) -> bool:
    """True if the Mojo kernels handle the solvent object's molecule (engine enabled, shells up to g)."""
    return integrals.engine() == "mojo" and integrals.mm_supported(pcmobj.mol)


def _surface(pcmobj):
    surf = pcmobj.surface
    return np.asarray(surf["grid_coords"], dtype=np.float64), np.asarray(surf["charge_exp"], dtype=np.float64) ** 2


def get_D_S(surface, with_S=True, with_D=False):
    """pyscf's ``solvent.pcm.get_D_S`` (D or None, S) from the Mojo kernel."""
    coords = np.ascontiguousarray(surface["grid_coords"], dtype=np.float64)
    n = coords.shape[0]
    S = np.empty((n, n))
    D = np.empty((n, n)) if with_D else np.empty((1, 1))
    get_extension().pcm_ds(
        integrals._boys_table(), coords, np.ascontiguousarray(surface["charge_exp"], dtype=np.float64),
        np.ascontiguousarray(surface["switch_fun"], dtype=np.float64),
        np.ascontiguousarray(surface["norm_vec"], dtype=np.float64),
        np.ascontiguousarray(np.broadcast_to(surface["R_vdw"], (n,)), dtype=np.float64), bool(with_D), S, D,
    )
    return (D if with_D else None), S


def _pair(surface, kind, a, b):
    """G_p = a_p sum_j dX_pj b_j - b_p sum_i a_i dX_ip for X = S ("S") or D ("D"), shape (n, 3).

    pyscf's einsum('i,xij,j->ix', a, dX, b) - einsum('i,xij,j->jx', a, dX, b)
    with dX from ``get_dD_dS`` (the derivative of X_ij with respect to r_i).
    """
    coords = np.ascontiguousarray(surface["grid_coords"], dtype=np.float64)
    g = np.empty((coords.shape[0], 3))
    get_extension().pcm_pair(
        integrals._boys_table(), coords, np.ascontiguousarray(surface["charge_exp"], dtype=np.float64),
        np.ascontiguousarray(surface["norm_vec"], dtype=np.float64), 0 if kind == "S" else 1,
        np.ascontiguousarray(a, dtype=np.float64), np.ascontiguousarray(b, dtype=np.float64), g,
    )
    return g


def grad_solver(pcmobj, dm, method=None, epsilon=None):
    """pyscf's ``solvent.grad.pcm.grad_solver`` (and SMD's) with the S/D derivative terms from ``_pair``.

    The same formulas as pyscf: dE = 1/2 v^T K^-1 (dR - dK q) with the
    switching-function (dF, dA) terms from pyscf's ``get_dF_dA``; ``method``
    and ``epsilon`` default to the solvent object's.
    """
    from pyscf.solvent.grad import pcm as pcm_grad

    if not pcmobj._intermediates:
        pcmobj.build()
    dm_cache = pcmobj._intermediates.get("dm", None)
    if dm_cache is None or np.linalg.norm(dm_cache - dm) >= 1e-10:
        pcmobj._get_vind(dm)
    surf = pcmobj.surface
    gridslice = surf["gslice_by_atom"]
    inter = pcmobj._intermediates
    v_grids, A, D, S, q = inter["v_grids"], inter["A"], inter["D"], inter["S"], inter["q"]
    method = (method or pcmobj.method).upper()
    epsilon = pcmobj.eps if epsilon is None else epsilon
    vK_1 = pcmobj._k_solve(v_grids, trans=True)
    dF, dA = pcm_grad.get_dF_dA(surf, pcmobj.surface_discretization_method)
    dSii_dF = -np.asarray(surf["charge_exp"]) * (2.0 / np.pi) ** 0.5 / np.asarray(surf["switch_fun"]) ** 2
    dSii = dSii_dF[:, None, None] * dF                      # (n, natm, 3)

    def by_atom(g):
        return np.asarray([g[p0:p1].sum(axis=0) for p0, p1 in gridslice])

    def term(a, kind, b):
        return 0.5 * by_atom(_pair(surf, kind, a, b))

    de = np.zeros((pcmobj.mol.natm, 3))
    if method in ("C-PCM", "CPCM", "COSMO"):
        de -= term(vK_1, "S", q)
        de -= 0.5 * np.einsum("i,inx->nx", vK_1 * q, dSii)
        return de
    if method not in ("IEF-PCM", "IEFPCM", "SS(V)PE", "SMD"):
        raise RuntimeError(f"Unknown implicit solvent model: {pcmobj.method}")
    f_epsilon = (epsilon - 1.0) / (epsilon + 1.0)
    fac_R = f_epsilon / (2.0 * np.pi)
    DA = D * A
    de_dR = fac_R * term(vK_1, "D", A * v_grids)
    vK_1_D = vK_1.dot(D)
    de_dR += 0.5 * fac_R * np.einsum("j,jnx->nx", vK_1_D * v_grids, dA)
    de_dS0 = term(vK_1, "S", q) + 0.5 * np.einsum("i,inx->nx", vK_1 * q, dSii)
    vK_1_DA = np.dot(vK_1, DA)
    de_dS1 = term(vK_1_DA, "S", q) + 0.5 * np.einsum("j,jnx->nx", vK_1_DA * q, dSii)
    Sq = np.dot(S, q)
    de_dD = term(vK_1, "D", A * Sq)
    de_dA = 0.5 * np.einsum("j,jnx->nx", vK_1_D * Sq, dA)
    if method == "SS(V)PE":
        DT_q = np.dot(D.T, q)
        ADT_q = A * DT_q
        de_dS1_T = term(vK_1, "S", ADT_q) + 0.5 * np.einsum("j,jnx->nx", vK_1 * ADT_q, dSii)
        vK_1_S = np.dot(vK_1, S)
        # pyscf's -dD^T term: G(q, D, vK_1_S A) at each point
        de_dD_T = term(q, "D", vK_1_S * A)
        de_dA_T = 0.5 * np.einsum("j,jnx->nx", vK_1_S * DT_q, dA)
        fac_K = f_epsilon / (4.0 * np.pi)
        de_dK = de_dS0 - fac_K * (de_dD + de_dA + de_dS1 + de_dD_T + de_dA_T + de_dS1_T)
    else:
        de_dK = de_dS0 - fac_R * (de_dD + de_dA + de_dS1)
    return de + de_dR - de_dK


class _MojoPCMMixin:
    """In front of pyscf's PCM (or SMD) class (:func:`attach`)."""

    __name_mixin__ = "Mojo"

    def build(self, ng=None):
        """pyscf's ``build`` with the surface matrices S and D from :func:`get_D_S`."""
        from pyscf.solvent import pcm

        if not integrals.engine() == "mojo":
            return super().build(ng)
        orig = pcm.get_D_S
        pcm.get_D_S = get_D_S
        try:
            return super().build(ng)
        finally:
            pcm.get_D_S = orig

    def _get_v(self, dms):
        """Electrostatic potential of the densities ``dms`` (nset, nao, nao) at the surface points."""
        if not supported(self):
            return super()._get_v(dms)
        coords, zetas = _surface(self)
        dms = np.asarray(dms, dtype=np.float64)
        dms = 0.5 * (dms + dms.transpose(0, 2, 1))          # (ij|k) is symmetric in ij
        return integrals.int1e_grids_dm(self.mol, dms, coords, zetas).reshape(dms.shape[0], -1)

    def _get_vmat(self, q):
        """-sum_k q_k (ij|k) for each set of surface charges ``q``."""
        if not supported(self):
            return super()._get_vmat(q)
        coords, zetas = _surface(self)
        q = np.asarray(q, dtype=np.float64).reshape(-1, coords.shape[0])
        return np.array([-integrals.int1e_grids_sum(self.mol, coords, qi, zetas) for qi in q])

    def _k_solve(self, b, trans=False):
        """K^-1 b (``trans``: K^-T b) with the LU factorisation of K, computed once per build."""
        inter = self._intermediates
        lu = inter.get("mojoscf_K_lu")
        if lu is None:
            lu = inter["mojoscf_K_lu"] = scipy.linalg.lu_factor(inter["K"])
        return scipy.linalg.lu_solve(lu, b, trans=1 if trans else 0)

    def _q_sym(self, v_grids):
        """The symmetrised surface charges of pyscf's ``_get_vind`` for potentials ``v_grids`` (nset, ng)."""
        R = self._intermediates["R"]
        q = self._k_solve(np.dot(R, v_grids.T)).T
        vK_1 = self._k_solve(v_grids.T, trans=True)
        qt = np.dot(R.T, vK_1).T
        return q, (q + qt) / 2.0

    def _get_vind(self, dms):
        if not self._intermediates:
            self.build()
        nao = dms.shape[-1]
        dms = dms.reshape(-1, nao, nao)
        if dms.shape[0] == 2:
            dms = (dms[0] + dms[1]).reshape(-1, nao, nao)
        v_grids = self.v_grids_n - self._get_v(dms)
        q, q_sym = self._q_sym(v_grids)
        vmat = self._get_vmat(q_sym)
        epcm = 0.5 * np.dot(q_sym[0], v_grids[0])
        self._intermediates["q"] = q[0]
        self._intermediates["q_sym"] = q_sym[0]
        self._intermediates["v_grids"] = v_grids[0]
        self._intermediates["dm"] = dms
        return epcm, vmat[0]

    def _B_dot_x(self, dms):
        if not self._intermediates:
            self.build()
        out_shape = dms.shape
        nao = dms.shape[-1]
        dms = dms.reshape(-1, nao, nao)
        v_grids = -self._get_v(dms)
        _, q_sym = self._q_sym(v_grids)
        return self._get_vmat(q_sym).reshape(out_shape)


def grad_qv(pcmobj, dm, q_sym=None):
    """pyscf's ``solvent.grad.pcm.grad_qv`` from one pass of :func:`mojoscf.integrals.mm_grad_terms`.

    ``2 sum_{i on A, j} D_ij sum_k q_k (nabla i j|k)`` (int3c2e_ip1) plus the
    forces ``sum_ij D_ij q_k (ij|nabla k)`` (int3c2e_ip2) on the surface
    charges, summed over the points of each atom.
    """
    if not pcmobj._intermediates:
        pcmobj.build()
    dm_cache = pcmobj._intermediates.get("dm", None)
    if dm_cache is None or np.linalg.norm(dm_cache - dm) >= 1e-10:
        pcmobj._get_vind(dm)
    if q_sym is None:
        q_sym = pcmobj._intermediates["q_sym"]
    coords, zetas = _surface(pcmobj)
    g_atoms, g_points = integrals.mm_grad_terms(pcmobj.mol, np.asarray(dm, dtype=np.float64), coords, q_sym, zetas)
    gridslice = pcmobj.surface["gslice_by_atom"]
    dq = np.asarray([g_points[p0:p1].sum(axis=0) for p0, p1 in gridslice])
    return g_atoms + dq


def _install_grad_hooks():
    """Route pyscf's ``solvent.grad.pcm.grad_qv``/``grad_solver`` and ``solvent.grad.smd.grad_solver``
    (imported at call time by PCM.grad and SMD.grad) to :func:`grad_qv` and :func:`grad_solver` for
    solvent objects with the Mojo mixin; pyscf's code runs for all others."""
    from pyscf.solvent.grad import pcm as pcm_grad
    from pyscf.solvent.grad import smd as smd_grad

    def wrap(module, name, mojo_fn):
        orig = getattr(module, name)
        if getattr(orig, "_mojoscf_orig", None) is not None:
            return

        def dispatch(pcmobj, dm, *args, **kwargs):
            if isinstance(pcmobj, _MojoPCMMixin) and supported(pcmobj) and np.ndim(dm) == 2:
                return mojo_fn(pcmobj, dm, *args, **kwargs)
            return orig(pcmobj, dm, *args, **kwargs)

        dispatch.__doc__ = orig.__doc__
        dispatch._mojoscf_orig = orig
        setattr(module, name, dispatch)

    def smd_solver(pcmobj, dm):
        from pyscf.solvent import smd

        descriptors = pcmobj.solvent_descriptors or smd.solvent_db[pcmobj.solvent]
        return grad_solver(pcmobj, dm, method="SMD", epsilon=pcmobj.eps or descriptors[5])

    wrap(pcm_grad, "grad_qv", grad_qv)
    wrap(pcm_grad, "grad_solver", grad_solver)
    wrap(smd_grad, "grad_solver", smd_solver)


def attach(mf):
    """Give the PCM/SMD solvent of ``mf`` (``mf.with_solvent``, or a solvent object itself) the Mojo
    kernels, in place; returns ``mf``.  Objects without such a solvent are returned unchanged."""
    from pyscf.solvent import pcm

    solvent = mf if isinstance(mf, pcm.PCM) else getattr(mf, "with_solvent", None)
    if isinstance(solvent, pcm.PCM) and not isinstance(solvent, _MojoPCMMixin):
        lib.set_class(solvent, (_MojoPCMMixin, type(solvent)))
        solvent._intermediates = {}         # rebuild with the Mojo surface matrices
        _install_grad_hooks()
    return mf
