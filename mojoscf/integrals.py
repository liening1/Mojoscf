"""Gaussian integrals from the Mojo integral engine, in pyscf's conventions.

The functions here take a ``pyscf.gto.Mole`` and return the same arrays as
the corresponding ``mol.intor`` calls (which use libcint):

======================  =============================================
``get_ovlp(mol)``       ``mol.intor("int1e_ovlp")``
``get_kin(mol)``        ``mol.intor("int1e_kin")``
``get_nuc(mol)``        ``mol.intor("int1e_nuc")``
``get_hcore(mol)``      ``pyscf.scf.hf.get_hcore(mol)`` (no ECP)
``int2e_s8(mol)``       ``mol.intor("int2e", aosym="s8")``
``int2e(mol)``          ``mol.intor("int2e")``
``int3c2e(mol, aux)``   ``df.incore.aux_e2(mol, aux, aosym="s2ij").T``
``int2c2e(aux)``        ``aux.intor("int2c2e")``
``cholesky_eri(mol)``   ``df.incore.cholesky_eri(mol, ...)``
======================  =============================================

Spherical and Cartesian (``mol.cart``) basis sets are supported for angular
momenta up to l = 8.  The one-electron integrals need point nuclei and no
effective core potentials; the two-electron ones (and the SCF driver's and
gradients' use of them) work for any such basis, ECPs included.
:func:`attach` makes a mojoscf SCF object use these integrals.
"""
from __future__ import annotations

import os

import numpy as np
from pyscf import gto, lib
from pyscf.df import addons as df_addons

from ._backend import get_extension

__all__ = [
    "engine",
    "set_engine",
    "available",
    "basis_tables",
    "int1e",
    "get_ovlp",
    "get_kin",
    "get_nuc",
    "get_hcore",
    "int2e_s8",
    "int2e",
    "int3c2e",
    "int2c2e",
    "cholesky_eri",
    "attach",
    "build_df",
    "get_jk",
    "unsupported_reason",
    "int1e_ip",
    "int1e_iprinv",
    "int1e_iprinv_dm",
    "get_jk_ip1",
    "grad2e",
    "grad2e_df",
]

LMAX = 8
NUC_POINT = 1
# libcint's CINTcommon_fac_sp: the factor its s and p "spherical" functions carry.
_FAC_SP = {0: 0.282094791773878143, 1: 0.488602511902919921}


_ENGINES = ("mojo", "libcint")
_ENGINE = os.environ.get("MOJOSCF_INTEGRALS", "mojo").strip().lower() or "mojo"
if _ENGINE not in _ENGINES:
    raise ValueError(f"MOJOSCF_INTEGRALS must be one of {_ENGINES}, got {_ENGINE!r}")


def engine() -> str:
    """Integral engine the mojoscf SCF driver uses: ``"mojo"`` (default) or ``"libcint"``."""
    return _ENGINE


def set_engine(name: str) -> None:
    """Select the integral engine of the SCF driver (also ``MOJOSCF_INTEGRALS=libcint``).

    With ``"mojo"`` the driver builds in-core ERIs, in-core density-fitting
    tensors and integral-direct J/K with this module whenever the molecule is
    supported (see ``unsupported_reason``) and falls back to libcint otherwise.
    """
    global _ENGINE
    name = name.strip().lower()
    if name not in _ENGINES:
        raise ValueError(f"integral engine must be one of {_ENGINES}, got {name!r}")
    _ENGINE = name


def available(mol, two_electron: bool = False, allow_ecp: bool = False) -> bool:
    """True when the Mojo engine is enabled and supports ``mol`` (see :func:`unsupported_reason`)."""
    return _ENGINE == "mojo" and unsupported_reason(mol, two_electron, allow_ecp) is None


def unsupported_reason(mol, two_electron: bool = False, allow_ecp: bool = False, allow_omega: bool = False) -> str | None:
    """Why the Mojo engine cannot handle ``mol`` (None if it can).

    With ``two_electron`` only the two-electron integrals are considered
    (four-index, three- and two-centre Coulomb integrals and their
    derivatives): effective core potentials and the nuclear charge model
    enter the one-electron Hamiltonian only, so molecules with ECPs or finite
    nuclei can still use the engine for those.  ``allow_ecp`` accepts ECPs
    for the one-electron integrals too: overlap, kinetic and point-charge
    nuclear attraction (``int1e``, ``int1e_ip``, ...) are the same Gaussian
    integrals as ``mol.intor`` returns, the ECP terms being separate
    (``ECPscalar*``) integrals.  ``allow_omega`` accepts a long-range
    operator (``mol.omega > 0``, erf(omega r) / r), which the three- and
    two-centre integrals (``int3c2e``, ``int2c2e``, ``cholesky_eri``) support.
    """
    if mol.nbas == 0:
        return "the molecule has no basis functions"
    if not two_electron and not allow_ecp and mol.has_ecp():
        return "effective core potentials are not supported"
    if int(mol._bas[:, gto.ANG_OF].max()) > LMAX:
        return f"angular momentum above l = {LMAX}"
    # pyscf marks atoms carrying an ECP with NUC_ECP; they are point charges (Z - core electrons)
    if not two_electron and (~np.isin(mol._atm[:, gto.NUC_MOD_OF], (NUC_POINT, gto.NUC_ECP))).any():
        return "only point nuclei are supported"
    omega = getattr(mol, "omega", 0.0)
    if omega and not (allow_omega and omega > 0):
        return "range-separated Coulomb operator (mol.omega) is not supported"
    return None


def _check(mol, two_electron: bool = False, allow_ecp: bool = False, allow_omega: bool = False):
    reason = unsupported_reason(mol, two_electron, allow_ecp, allow_omega)
    if reason is not None:
        raise NotImplementedError(f"mojoscf.integrals: {reason}")


_BOYS_TABLE = None


def _boys_table():
    """Boys-function interpolation table, built once per process by the extension."""
    global _BOYS_TABLE
    if _BOYS_TABLE is None:
        table = np.empty(721 * 40)
        get_extension().boys_table(table)
        _BOYS_TABLE = table
    return _BOYS_TABLE


def basis_tables(mol, allow_omega: bool = False):
    """``(atm, bas, env, nf, c2s)``: the basis in the form the Mojo engine reads.

    The tables carry no operator: ``allow_omega`` is for the callers that
    pass ``mol.omega`` to the kernels themselves (``int3c2e``, ``int2c2e``).

    ``atm``/``bas``/``env`` are pyscf's ``_atm``/``_bas``/``_env`` (as int64 /
    float64), ``nf[l]`` the number of functions per shell of angular momentum
    ``l`` and ``c2s`` the concatenated ``(ncart, nf)`` Cartesian-to-final
    transformation matrices for l = 0..lmax.
    """
    _check(mol, two_electron=True, allow_omega=allow_omega)
    atm = np.ascontiguousarray(mol._atm, dtype=np.int64)
    bas = np.ascontiguousarray(mol._bas, dtype=np.int64)
    env = np.ascontiguousarray(mol._env, dtype=np.float64)
    lmax = int(bas[:, gto.ANG_OF].max())
    mats = []
    for l in range(lmax + 1):
        if mol.cart:
            n = (l + 1) * (l + 2) // 2
            mats.append(np.eye(n) * _FAC_SP.get(l, 1.0))
        else:
            mats.append(np.asarray(gto.cart2sph(l), dtype=np.float64))
    nf = np.asarray([m.shape[1] for m in mats], dtype=np.int64)
    c2s = np.ascontiguousarray(np.concatenate([m.ravel() for m in mats]), dtype=np.float64)
    return atm, bas, env, nf, c2s


def int1e(mol):
    """``(S, T, V)``: overlap, kinetic energy and nuclear attraction matrices."""
    _check(mol, allow_ecp=True)
    tables = basis_tables(mol)
    nao = mol.nao_nr()
    s = np.empty((nao, nao))
    t = np.empty((nao, nao))
    v = np.empty((nao, nao))
    get_extension().int1e(tables, s, t, v, _boys_table())
    return s, t, v


def get_ovlp(mol):
    return int1e(mol)[0]


def get_kin(mol):
    return int1e(mol)[1]


def get_nuc(mol):
    return int1e(mol)[2]


def get_hcore(mol):
    """``T + V`` (the core Hamiltonian of a molecule without ECPs)."""
    _check(mol)
    _, t, v = int1e(mol)
    return t + v


def int2e_s8(mol, schwarz_tol: float = 1e-14):
    """Electron repulsion integrals in pyscf's 8-fold packed layout (``aosym="s8"``).

    Shell quartets whose Schwarz bound is below ``schwarz_tol`` are skipped
    (left at zero); pass ``0`` to compute every quartet.
    """
    tables = basis_tables(mol)
    nao = mol.nao_nr()
    npair = nao * (nao + 1) // 2
    eri = np.empty(npair * (npair + 1) // 2)
    get_extension().int2e_s8(tables, eri, float(schwarz_tol), _boys_table())
    return eri


def int2e(mol, schwarz_tol: float = 1e-14):
    """Full ``(nao, nao, nao, nao)`` electron repulsion tensor."""
    from pyscf import ao2mo

    return ao2mo.restore(1, int2e_s8(mol, schwarz_tol), mol.nao_nr())


def _omega(mol, omega):
    omega = float(getattr(mol, "omega", 0.0) if omega is None else omega)
    if omega < 0:
        raise NotImplementedError("mojoscf.integrals: short-range (omega < 0) operators are not supported")
    return omega


def int3c2e(mol, auxmol, omega=None):
    """``(ab|P)`` as a ``(naux, npair)`` array with the AO pair packed in ``pack_tril`` order.

    ``omega`` (default ``mol.omega``) > 0 gives the integrals of the
    long-range operator erf(omega r) / r, as libcint with ``mol.omega``.
    """
    omega = _omega(mol, omega)
    tables = basis_tables(mol, allow_omega=True)
    aux_tables = basis_tables(auxmol, allow_omega=True)
    nao = mol.nao_nr()
    npair = nao * (nao + 1) // 2
    out = np.empty((auxmol.nao_nr(), npair))
    get_extension().int3c2e(tables, aux_tables, out, _boys_table(), omega)
    return out


def int2c2e(auxmol, omega=None):
    """``(P|Q)`` as a dense ``(naux, naux)`` matrix (``omega`` as in :func:`int3c2e`)."""
    omega = _omega(auxmol, omega)
    aux_tables = basis_tables(auxmol, allow_omega=True)
    naux = auxmol.nao_nr()
    out = np.empty((naux, naux))
    get_extension().int2c2e(aux_tables, out, _boys_table(), omega)
    return out


def cholesky_eri(mol, auxbasis="weigend+etb", auxmol=None, lindep=None, omega=None):
    """pyscf's in-core DF tensor ``L^{-1} (P|ab)`` with the integrals from the Mojo engine.

    Mirrors ``pyscf.df.incore.cholesky_eri`` (Cholesky factorisation of
    ``(P|Q)``, falling back to an eigen-decomposition that drops eigenvalues
    below ``lindep`` (default pyscf's ``LINEAR_DEP_THR``) when the metric is
    not positive definite); the factorisation and triangular solve
    are LAPACK/BLAS calls through SciPy, as in pyscf.  The solve runs in place
    on the transposed (Fortran-ordered) view of the C-ordered integrals
    (``dtrsm`` from the right with L^T), so the result is C-ordered, as the
    J/K kernels need, without any copy of the (naux, npair) tensor.
    ``omega`` (default ``mol.omega``) > 0: the tensor of the long-range
    operator, as pyscf builds it for ``DF.range_coulomb(omega)``.
    """
    import scipy.linalg
    from scipy.linalg import blas as sblas

    if auxmol is None:
        auxmol = df_addons.make_auxmol(mol, auxbasis)
    from pyscf.df.incore import LINEAR_DEP_THR

    if lindep is None:
        lindep = LINEAR_DEP_THR
    omega = _omega(mol, omega)
    j2c = int2c2e(auxmol, omega)
    j3c = int3c2e(mol, auxmol, omega)
    try:
        low = scipy.linalg.cholesky(j2c, lower=True)
        # X L^T = j3c^T  <=>  X^T = L^-1 j3c
        out = sblas.dtrsm(1.0, low, j3c.T, side=1, lower=1, trans_a=1, overwrite_b=1)
        return out.T
    except scipy.linalg.LinAlgError:
        w, v = scipy.linalg.eigh(j2c)
        keep = w > lindep
        v = v[:, keep] / np.sqrt(w[keep])
        return lib.dot(v.T, j3c)


def build_df(with_df) -> bool:
    """Build ``with_df._cderi`` in core with the Mojo engine, as ``pyscf.df.DF.build`` would.

    Follows pyscf's decision: only when the tensor fits in 90% of the free
    memory and no file storage was requested.  Returns False (leaving the
    object untouched) otherwise, or when the molecule is not supported.
    """
    from pyscf import lib as pyscf_lib

    mol = with_df.mol
    omega = getattr(with_df, "omega", None) or 0.0
    if with_df._cderi is not None or omega < 0:
        return False
    if unsupported_reason(mol, two_electron=True) is not None or engine() != "mojo":
        return False
    if isinstance(getattr(with_df, "_cderi_to_save", None), str):
        return False
    auxmol = with_df.auxmol
    if auxmol is None:
        auxmol = df_addons.make_auxmol(mol, with_df.auxbasis)
    if unsupported_reason(auxmol, two_electron=True) is not None:
        return False
    nao = mol.nao_nr()
    max_memory = with_df.max_memory - pyscf_lib.current_memory()[0]
    if nao * (nao + 1) // 2 * auxmol.nao_nr() * 8 / 1e6 >= 0.9 * max_memory:
        return False
    with_df.auxmol = auxmol
    with_df._cderi = cholesky_eri(mol, auxmol=auxmol, omega=omega)
    return True


def attach(mf, schwarz_tol: float = 1e-14, auxbasis=None):
    """Make the SCF object ``mf`` take its integrals from the Mojo engine.

    ``get_ovlp`` and ``get_hcore`` return precomputed Mojo matrices (left to
    pyscf for molecules with ECPs or finite nuclei), the in-core ERI tensor
    (``mf._eri``) is built here, and for density-fitted objects the DF tensor
    (``mf.with_df._cderi``) is built from the Mojo three- and two-centre
    integrals.  Returns ``mf``.
    """
    mol = mf.mol
    _check(mol, two_electron=True)
    if available(mol):
        s, t, v = int1e(mol)
        hcore = t + v
        mf.get_ovlp = lambda mol=None: s
        mf.get_hcore = lambda mol=None: hcore
    with_df = getattr(mf, "with_df", None)
    if with_df is not None:
        auxmol = with_df.auxmol
        if auxmol is None:
            auxmol = with_df.auxmol = df_addons.make_auxmol(mol, auxbasis or with_df.auxbasis)
        with_df._cderi = cholesky_eri(mol, auxmol=auxmol)
    elif getattr(mf, "_eri", None) is None and (mol.incore_anyway or mf._is_mem_enough()):
        mf._eri = int2e_s8(mol, schwarz_tol)
    return mf


def get_jk(mol, dm, with_j=True, with_k=True, direct_scf_tol=1e-13):
    """Integral-direct J and K of symmetric density matrices, like ``pyscf.scf.hf.get_jk``.

    ``dm`` is (nao, nao) or (n, nao, nao); returns (vj, vk) of the same shape
    (None for a matrix that was not requested).  Shell quartets are screened
    with the Schwarz bounds times the density, as pyscf's ``direct_scf_tol``.
    """
    dm = np.asarray(dm, dtype=np.float64)
    single = dm.ndim == 2
    dms = np.ascontiguousarray(dm.reshape(-1, dm.shape[-2], dm.shape[-1]))
    vj = np.empty_like(dms)
    vk = np.empty_like(dms)
    get_extension().direct_jk(basis_tables(mol), _boys_table(), dms, vj, vk, bool(with_j), bool(with_k), float(direct_scf_tol))
    shape = dm.shape
    vj = vj.reshape(shape) if with_j else None
    vk = vk.reshape(shape) if with_k else None
    return vj, vk


def _int1e_ip(mol, centers, charges, want_st):
    _check(mol, allow_ecp=True)
    tables = basis_tables(mol)
    nao = mol.nao_nr()
    s = np.zeros((3, nao, nao))
    t = np.zeros((3, nao, nao))
    v = np.zeros((3, nao, nao))
    centers = np.ascontiguousarray(centers, dtype=np.float64).reshape(-1, 3)
    charges = np.ascontiguousarray(charges, dtype=np.float64).reshape(-1)
    get_extension().int1e_ip(tables, _boys_table(), centers, charges, bool(want_st), s, t, v)
    return s, t, v


def int1e_ip(mol):
    """``(ipovlp, ipkin, ipnuc)``: pyscf's ``int1e_ipovlp``, ``int1e_ipkin``, ``int1e_ipnuc`` (each (3, nao, nao))."""
    return _int1e_ip(mol, mol.atom_coords(), -mol.atom_charges().astype(np.float64), True)


def int1e_iprinv(mol, atom_id=None, origin=None):
    """pyscf's ``int1e_iprinv``: <nabla i| 1/|r - R| |j> with R the nucleus ``atom_id`` (or ``origin``)."""
    if origin is None:
        origin = mol.atom_coord(atom_id)
    return _int1e_ip(mol, np.asarray(origin, dtype=np.float64), np.ones(1), False)[2]


def int1e_iprinv_dm(mol, dm, centers=None):
    """``sum_ij dm_ij <nabla i| 1/|r - R_c| |j>`` for every centre c (default: the nuclei), shape (ncenter, 3).

    Equals ``einsum('xij,ij->x', int1e_iprinv(mol, origin=R_c), dm)`` for each
    centre, evaluated in one pass instead of one integral matrix per centre.
    """
    _check(mol, allow_ecp=True)
    if centers is None:
        centers = mol.atom_coords()
    centers = np.ascontiguousarray(centers, dtype=np.float64).reshape(-1, 3)
    dm = np.ascontiguousarray(dm, dtype=np.float64)
    out = np.zeros((centers.shape[0], 3))
    get_extension().int1e_iprinv_dm(basis_tables(mol), _boys_table(), centers, dm, out)
    return out


MM_LMAX = 4     # the charge kernels handle shell pairs up to Hermite degree 9 (g-g pairs and their derivative)


def mm_supported(mol) -> bool:
    """True if the MM-charge kernels below support ``mol`` (engine-supported, shells up to g)."""
    return available(mol, allow_ecp=True) and (mol.nbas == 0 or int(mol._bas[:, gto.ANG_OF].max()) <= MM_LMAX)


def _mm_args(mol, coords, weights, zetas):
    if not mm_supported(mol):
        raise NotImplementedError(
            unsupported_reason(mol, allow_ecp=True) or f"MM charges need shells up to l = {MM_LMAX}"
        )
    coords = np.ascontiguousarray(coords, dtype=np.float64).reshape(-1, 3)
    weights = np.ascontiguousarray(weights, dtype=np.float64).reshape(-1)
    if weights.shape[0] != coords.shape[0]:
        raise ValueError("one weight per MM charge expected")
    point = zetas is None
    zetas = np.zeros(1) if point else np.ascontiguousarray(np.broadcast_to(zetas, weights.shape), dtype=np.float64)
    return coords, weights, zetas, point


def int1e_grids_sum(mol, coords, weights, zetas=None):
    """``sum_k w_k <i| 1/|r - R_k| |j>`` for charges at ``coords`` (Bohr), shape (nao, nao).

    pyscf: ``einsum('kij,k->ij', mol.intor('int1e_grids', grids=coords), weights)``.
    With ``zetas`` the charges are unit Gaussians ``(zeta/pi)^{3/2} exp(-zeta r^2)``
    and the result is ``sum_k w_k (ij|k)`` (pyscf's ``int3c2e`` with
    ``gto.fakemol_for_charges(coords, zetas)``).  One pass over the shell
    pairs with the charges as SIMD lanes.
    """
    coords, weights, zetas, point = _mm_args(mol, coords, weights, zetas)
    nao = mol.nao_nr()
    out = np.empty((nao, nao))
    get_extension().mm_potential(basis_tables(mol), _boys_table(), coords, weights, zetas, point, out)
    return out


def int1e_grids_ip_sum(mol, coords, weights, zetas=None):
    """``sum_k w_k <nabla i| 1/|r - R_k| |j>``, shape (3, nao, nao) (Gaussian charges with ``zetas``).

    pyscf: ``einsum('kxij,k->xij', mol.intor('int1e_grids_ip', grids=coords), weights)``.
    """
    coords, weights, zetas, point = _mm_args(mol, coords, weights, zetas)
    nao = mol.nao_nr()
    mat = np.empty((3, nao, nao))
    get_extension().mm_grad(
        basis_tables(mol), _boys_table(), coords, weights, zetas, point, np.zeros(1), mat, np.zeros(0), np.zeros(0)
    )
    return mat


def int1e_grids_dm(mol, dms, coords, zetas=None):
    """``sum_ij D_ij <i| 1/|r - R_k| |j>`` for every point k: the potential of the symmetric densities
    ``dms`` ((nao, nao) or (nset, nao, nao)) at ``coords`` (Bohr), shape (nk,) or (nset, nk).

    pyscf: ``einsum('kij,ij->k', mol.intor('int1e_grids', grids=coords), dm)``; with ``zetas`` the
    points are unit Gaussians (``int3c2e`` with ``gto.fakemol_for_charges(coords, zetas)``, as in
    pyscf's PCM).  No integral is stored: each shell pair's Hermite matrices are contracted with the
    densities first.
    """
    coords, _, zetas, point = _mm_args(mol, coords, np.ones(len(np.reshape(coords, (-1, 3)))), zetas)
    dms = np.asarray(dms, dtype=np.float64)
    single = dms.ndim == 2
    nao = mol.nao_nr()
    dms = np.ascontiguousarray(dms.reshape(-1, nao, nao))
    out = np.empty((dms.shape[0], coords.shape[0]))
    get_extension().mm_esp(basis_tables(mol), _boys_table(), coords, zetas, point, dms, out)
    return out[0] if single else out


def mm_charge_forces(mol, dm, coords, weights, zetas=None):
    """``sum_ij D_ij w_k (ij|nabla_k)`` for every charge k, shape (ncharge, 3).

    This is pyscf's ``QMMMGrad.grad_hcore_mm(dm)`` (``int3c2e_ip2`` with the
    charges as unit Gaussians, or point charges without ``zetas``) for a
    symmetric density ``dm``.
    """
    coords, weights, zetas, point = _mm_args(mol, coords, weights, zetas)
    dm = np.ascontiguousarray(dm, dtype=np.float64)
    nao = mol.nao_nr()
    if dm.shape != (nao, nao):
        raise ValueError(f"dm must be ({nao}, {nao})")
    forces = np.empty((coords.shape[0], 3))
    get_extension().mm_grad(
        basis_tables(mol), _boys_table(), coords, weights, zetas, point, dm, np.zeros(0), forces, np.zeros(0)
    )
    return forces


def mm_grad_terms(mol, dm, coords, weights, zetas=None):
    """Both density-contracted derivatives of the charge potential from one pass, ``(g_atoms, g_charges)``.

    ``g_atoms[A] = 2 sum_{i on A, j} D_ij sum_k w_k <nabla i|1/|r - R_k||j>``
    (natm, 3), the term of the QM-atom gradient that pyscf's ``QMMMGrad``
    gets from its ``get_hcore`` (with ``w`` the MM charges), and
    ``g_charges = mm_charge_forces(mol, dm, coords, weights, zetas)``
    (ncharge, 3).  ``dm`` must be symmetric.  No integral matrix is formed.
    """
    coords, weights, zetas, point = _mm_args(mol, coords, weights, zetas)
    dm = np.ascontiguousarray(dm, dtype=np.float64)
    nao = mol.nao_nr()
    if dm.shape != (nao, nao):
        raise ValueError(f"dm must be ({nao}, {nao})")
    forces = np.empty((coords.shape[0], 3))
    atoms = np.empty((mol.natm, 3))
    get_extension().mm_grad(basis_tables(mol), _boys_table(), coords, weights, zetas, point, dm, np.zeros(0), forces, atoms)
    return atoms, forces


def get_jk_ip1(mol, dm, with_j=True, with_k=True, tol=1e-14):
    """Gradient J/K exactly as ``pyscf.grad.rhf.get_jk``: ``(-sum (nabla i j|kl) D_lk, -sum (nabla i j|kl) D_jk)``.

    ``dm`` is (nao, nao) or (n, nao, nao) and must be symmetric; the results
    have shape (3, nao, nao) or (n, 3, nao, nao).
    """
    dm = np.asarray(dm, dtype=np.float64)
    single = dm.ndim == 2
    dms = np.ascontiguousarray(dm.reshape(-1, dm.shape[-2], dm.shape[-1]))
    nset, nao = dms.shape[0], dms.shape[1]
    vj = np.zeros((nset, 3, nao, nao))
    vk = np.zeros((nset, 3, nao, nao))
    get_extension().jk_ip1(basis_tables(mol), _boys_table(), dms, vj, vk, bool(with_j), bool(with_k), float(tol))
    vj, vk = -vj, -vk
    if single:
        vj, vk = vj[0], vk[0]
    return (vj if with_j else None), (vk if with_k else None)


def grad2e(mol, dm_j, dm_k, j_factor=1.0, k_factor=1.0, tol=1e-14):
    """Two-electron part of the nuclear gradient at fixed densities, shape (natm, 3).

    The derivative of ``E2 = 1/2 sum_ijkl (ij|kl) G_ijkl`` with respect to the
    nuclear coordinates, where
    ``G_ijkl = j_factor Dj_ij Dj_kl - k_factor/2 sum_s (Dk_s,ik Dk_s,jl + Dk_s,il Dk_s,jk)``
    for the symmetric density ``dm_j`` and the symmetric densities
    ``dm_k`` ((nao, nao) or (n, nao, nao)).  RHF: ``grad2e(mol, D, D, 1, 0.5)``;
    UHF: ``grad2e(mol, Da + Db, (Da, Db))``.  The derivative integrals are
    contracted with ``G`` as they are evaluated (8-fold symmetry, nothing stored).
    """
    nao = mol.nao_nr()
    dmj = np.ascontiguousarray(dm_j, dtype=np.float64).reshape(nao, nao)
    dmk = np.ascontiguousarray(dm_k, dtype=np.float64).reshape(-1, nao, nao)
    de = np.zeros((mol.natm, 3))
    get_extension().grad2e(basis_tables(mol), _boys_table(), dmj, dmk, float(j_factor), float(k_factor), float(tol), de)
    return de


def _syrk_full(a_t):
    """``A^T A`` for the (k, n) array ``a_t``, Fortran-ordered to avoid a copy (BLAS ``dsyrk``, both triangles returned)."""
    from scipy.linalg import blas as sblas

    c = sblas.dsyrk(1.0, a_t, trans=1, lower=0)
    return lib.hermi_triu(c)  # Fortran-ordered result: mirrors the upper triangle in place


def grad2e_df(mol, auxmol, dm_j, orbs, occs, j_factor=1.0, k_factor=1.0, max_memory=4000, tol=1e-14, omega=0.0):
    """Two-electron part of the nuclear gradient with density fitting, shape (natm, 3).

    The derivative, at fixed densities, of the density-fitted energy
    ``E2 = j_factor/2 rho^T V^-1 rho - k_factor/2 sum_s sum_ij n_i n_j (ij|P) V^-1_PQ (Q|ij)``
    (``rho_P = (P|mu nu) Dj_mu nu``, V the Coulomb metric of ``auxmol``,
    orbitals ``orbs[s]`` with occupations ``occs[s]``), including the
    response of the auxiliary basis, as pyscf's ``df.grad`` gradients with
    ``auxbasis_response=True``.  RHF: ``dm_j = D``, ``orbs = [C_occ]``,
    ``occs = [2...]``, ``k_factor = 1/2``; UHF: ``dm_j = Da + Db``,
    ``orbs = [Ca_occ, Cb_occ]``, ``occs = [1...]``, ``k_factor = 1``.

    With c = V^-1 rho and X_s = V^-1 (P|ij)_s,

        dE2 = sum_{P, mu nu} d(mu nu|P) Gamma_P,mu nu - 1/2 sum_PQ d(P|Q) W_PQ
        Gamma_P = j_factor c_P Dj - k_factor sum_s (C n) X_s,P (C n)^T
        W = j_factor c c^T - k_factor sum_s sum_ij n_i n_j X_s,P,ij X_s,Q,ij

    The three-centre integrals and their derivatives are evaluated in Mojo,
    in blocks of auxiliary functions sized by ``max_memory`` (MB), and
    contracted as they are produced (the transforms with the orbitals use the
    sequential BLAS in the worker threads); the metric solves and W are
    SciPy/NumPy BLAS calls.  The metric is factorised as in pyscf's gradient
    code (Cholesky, eigen-decomposition fallback).  ``omega`` > 0: the same
    for the long-range operator erf(omega r) / r (integrals and metric), the
    exchange of range-separated functionals with pyscf's
    ``with_df.range_coulomb(omega)``.
    """
    from pyscf.df.grad.rhf import _gen_metric_solver

    from ._backend import worker_blas

    _check(mol, two_electron=True)
    _check(auxmol, two_electron=True)
    ext = get_extension()
    table = _boys_table()
    tables = basis_tables(mol)
    aux_tables = basis_tables(auxmol)
    seq_path, seq_prefix = worker_blas()
    nao = mol.nao_nr()
    naux = auxmol.nao_nr()
    npair = nao * (nao + 1) // 2
    dm_j = np.asarray(dm_j, dtype=np.float64).reshape(nao, nao)
    dm_tril = lib.pack_tril(dm_j + dm_j.T)
    diag = np.arange(nao)
    dm_tril[diag * (diag + 1) // 2 + diag] *= 0.5
    dpack = np.ascontiguousarray(lib.pack_tril(dm_j))
    nset = len(orbs)
    occs = [np.asarray(n, dtype=np.float64).reshape(-1) for n in occs]
    m = max((len(n) for n in occs), default=0)
    # orbital sets padded with zero columns to a common width m
    c = np.zeros((max(nset, 1), nao, m))
    nn = np.zeros((max(nset, 1), m))
    for st, (orb, n) in enumerate(zip(orbs, occs)):
        c[st, :, : len(n)] = np.asarray(orb, dtype=np.float64).reshape(nao, len(n))
        nn[st, : len(n)] = n
    cn = np.ascontiguousarray(c * nn[:, None, :])
    blk = int(max_memory * 1e6 / 8 / 2 / npair)
    blk = max(1, min(blk, naux))

    # (P|ij) of each set as packed lower triangles (they are symmetric in ij)
    mp = m * (m + 1) // 2
    rhoj = np.empty(naux)
    xs = np.empty((max(nset, 1), naux, mp))
    omega = _omega(mol, omega)
    ext.df_grad_rhs(tables, aux_tables, table, dm_tril, c, blk, rhoj, xs, seq_path, seq_prefix, omega)

    # X = V^-1 (P|ij) in place, and W on the packed columns:
    # sum_ij n_i n_j X_Pij X_Qij = sum_{i>=j} w_ij X_Pij X_Qij, w_ij = n_i n_j (2 for i > j)
    solve = _gen_metric_solver(int2c2e(auxmol, omega))
    coef = solve(rhoj)
    w = j_factor * np.outer(coef, coef)
    tri = np.tril_indices(m)
    for st in range(nset):
        xs[st] = solve(xs[st])
        wij = np.outer(nn[st], nn[st])[tri] * np.where(tri[0] == tri[1], 1.0, 2.0)
        if np.all(wij >= 0):
            w -= k_factor * _syrk_full((xs[st] * np.sqrt(wij)).T)
        else:
            w -= k_factor * lib.dot(xs[st] * wij, xs[st].T)

    de = np.zeros((auxmol.natm, 3))
    ext.grad2c(aux_tables, table, np.ascontiguousarray(w), de, omega)
    w = None
    d3 = np.zeros((mol.natm, 3))
    ext.grad_df3c(
        tables, aux_tables, table, np.ascontiguousarray(coef), dpack, float(j_factor), float(k_factor),
        xs, cn, blk, float(tol), d3, seq_path, seq_prefix, omega,
    )
    return de + d3
