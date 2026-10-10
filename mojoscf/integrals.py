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
``cholesky_eri_h5``     ``df.outcore.cholesky_eri(mol, file, ...)``
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
    "int3c2e_cols",
    "int2c2e",
    "cholesky_eri",
    "cholesky_eri_h5",
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


def int2e_s8(mol, schwarz_tol: float = 1e-14, omega: float = 0.0):
    """Electron repulsion integrals in pyscf's 8-fold packed layout (``aosym="s8"``).

    Shell quartets whose Schwarz bound is below ``schwarz_tol`` are skipped
    (left at zero); pass ``0`` to compute every quartet.  ``omega`` > 0 gives
    those of the long-range operator erf(omega r12) / r12, as pyscf's
    ``mol.intor("int2e", aosym="s8")`` inside ``mol.with_range_coulomb(omega)``.
    """
    tables = basis_tables(mol)
    nao = mol.nao_nr()
    npair = nao * (nao + 1) // 2
    eri = np.empty(npair * (npair + 1) // 2)
    get_extension().int2e_s8(tables, eri, float(schwarz_tol), _boys_table(), _omega4c(omega))
    return eri


def _omega4c(omega) -> float:
    """``omega`` for the four-centre kernels: 0 (Coulomb) or > 0 (erf(omega r) / r)."""
    omega = float(omega or 0.0)
    if omega < 0:
        raise NotImplementedError("mojoscf.integrals: short-range (omega < 0) four-centre integrals")
    return omega


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


def int3c2e_cols(mol, auxmol, a0, a1, omega=None):
    """The columns of :func:`int3c2e` for the AO pairs (i, j <= i) with i in the shells [a0, a1): the
    ``pack_tril`` columns [c0, c1), c = i (i + 1) / 2 for the first AO i of ``a0`` and of ``a1``."""
    omega = _omega(mol, omega)
    loc = mol.ao_loc_nr()
    c0, c1 = loc[a0] * (loc[a0] + 1) // 2, loc[a1] * (loc[a1] + 1) // 2
    out = np.empty((auxmol.nao_nr(), c1 - c0))
    get_extension().int3c2e_cols(basis_tables(mol, allow_omega=True), basis_tables(auxmol, allow_omega=True),
                                 _boys_table(), int(a0), int(a1), out, omega)
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


def cholesky_eri_h5(mol, erifile, auxmol=None, auxbasis="weigend+etb", dataname="j3c", max_memory=2000,
                    lindep=None, omega=None):
    """pyscf's DF tensor on disk with the integrals from the Mojo engine; returns ``erifile``.

    One contiguous ``(naux, npair)`` HDF5 dataset ``dataname``, the layout of
    ``pyscf.df.outcore.cholesky_eri`` (``DF._compatible_format``), which
    ``DF.loop`` reads and which can be memory-mapped (:func:`mojoscf.dft.ondisk_tensor`).
    Each block of AO shells is computed (:func:`int3c2e_cols`), transformed
    with the Cholesky factor of the metric (or its eigen-decomposition, as
    :func:`cholesky_eri`) and written as its columns; ``max_memory`` (MB)
    bounds the block.
    """
    import scipy.linalg
    from pyscf.df.incore import LINEAR_DEP_THR
    from pyscf.df.outcore import _create_h5file
    from scipy.linalg import blas as sblas

    if auxmol is None:
        auxmol = df_addons.make_auxmol(mol, auxbasis)
    if lindep is None:
        lindep = LINEAR_DEP_THR
    omega = _omega(mol, omega)
    j2c = int2c2e(auxmol, omega)
    low = eig = None
    try:
        low = scipy.linalg.cholesky(j2c, lower=True)
    except scipy.linalg.LinAlgError:
        w, v = scipy.linalg.eigh(j2c)
        keep = w > lindep
        eig = v[:, keep] / np.sqrt(w[keep])
    j2c = None
    naux = auxmol.nao_nr()
    loc = mol.ao_loc_nr()
    first = loc * (loc + 1) // 2                     # first pack_tril column of each shell (and the end)
    ncol_max = max(int(max(max_memory, 200) * 0.12e6 / 8 / naux), 1)   # integrals and result of a block
    feri = _create_h5file(erifile, dataname)
    try:
        dset = feri.create_dataset(dataname, (naux if eig is None else eig.shape[1], first[-1]), "f8")
        a0 = 0
        while a0 < mol.nbas:
            a1 = a0 + 1
            while a1 < mol.nbas and first[a1 + 1] - first[a0] <= ncol_max:
                a1 += 1
            j3c = int3c2e_cols(mol, auxmol, a0, a1, omega)
            if low is not None:
                dat = sblas.dtrsm(1.0, low, j3c.T, side=1, lower=1, trans_a=1, overwrite_b=1).T
            else:
                dat = lib.dot(eig.T, j3c)
            dset[:, first[a0]:first[a1]] = dat
            j3c = dat = None
            a0 = a1
    finally:
        feri.close()
    return erifile


def build_df(with_df) -> bool:
    """Build ``with_df._cderi`` with the Mojo engine where ``pyscf.df.DF.build`` would build it.

    Follows pyscf's decision: in core when the tensor fits in 90% of the free
    memory and no file was named in ``_cderi_to_save``, otherwise on disk
    (:func:`cholesky_eri_h5`, in that file or a temporary one, as pyscf).
    Returns False (leaving the object untouched) when the molecule is not
    supported and for long-range tensors that do not fit in memory.
    """
    from pyscf import lib as pyscf_lib

    mol = with_df.mol
    # the long-range objects of DF.range_coulomb(omega) carry omega on their molecule while in use
    omega = getattr(with_df, "omega", None) or getattr(mol, "omega", 0.0) or 0.0
    if with_df._cderi is not None or omega < 0:
        return False
    if unsupported_reason(mol, two_electron=True, allow_omega=omega > 0) is not None or engine() != "mojo":
        return False
    auxmol = with_df.auxmol
    if auxmol is None:
        auxmol = df_addons.make_auxmol(mol, with_df.auxbasis)
    if unsupported_reason(auxmol, two_electron=True, allow_omega=omega > 0) is not None:
        return False
    nao = mol.nao_nr()
    max_memory = with_df.max_memory - pyscf_lib.current_memory()[0]
    named = isinstance(getattr(with_df, "_cderi_to_save", None), str)
    if nao * (nao + 1) // 2 * auxmol.nao_nr() * 8 / 1e6 < 0.9 * max_memory and not named:
        with_df.auxmol = auxmol
        with_df._cderi = cholesky_eri(mol, auxmol=auxmol, omega=omega)
        return True
    if omega:
        return False
    if with_df._cderi_to_save is None:
        with_df._cderi_to_save = pyscf_lib.NamedTemporaryFile(dir=pyscf_lib.param.TMPDIR)
    cderi = with_df._cderi_to_save
    cholesky_eri_h5(mol, cderi if named else cderi.name, auxmol=auxmol, dataname=with_df._dataname,
                    max_memory=max_memory)
    with_df.auxmol = auxmol
    with_df._cderi = cderi
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


def get_jk(mol, dm, with_j=True, with_k=True, direct_scf_tol=1e-13, hermi=1, omega=0.0):
    """Integral-direct J and K of density matrices, like ``pyscf.scf.hf.get_jk``.

    ``dm`` is (nao, nao) or (n, nao, nao); returns (vj, vk) of the same shape
    (None for a matrix that was not requested).  Shell quartets are screened
    with the Schwarz bounds times the density, as pyscf's ``direct_scf_tol``.
    All densities share one pass over the integrals; ``hermi=0`` accepts
    non-symmetric densities (their antisymmetric parts go through the kernel
    as such, J being that of the symmetric parts).  ``omega`` > 0: the
    long-range operator erf(omega r12) / r12 (pyscf's ``omega`` argument).
    """
    from .kernels import _merge_anti, _sym_anti

    dm = np.asarray(dm, dtype=np.float64)
    dms = np.ascontiguousarray(dm.reshape(-1, dm.shape[-2], dm.shape[-1]))
    stack, anti = _sym_anti(dms, hermi)
    vj = np.zeros(stack.shape)
    vk = np.zeros(stack.shape)
    get_extension().direct_jk(basis_tables(mol), _boys_table(), stack, vj, vk, bool(with_j), bool(with_k),
                              float(direct_scf_tol), len(anti), _omega4c(omega))
    vj, vk = _merge_anti(vj if with_j else None, vk if with_k else None, dms.shape[0], anti)
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


def get_jk_ip1(mol, dm, with_j=True, with_k=True, tol=1e-14, omega=0.0):
    """Gradient J/K exactly as ``pyscf.grad.rhf.get_jk``: ``(-sum (nabla i j|kl) D_lk, -sum (nabla i j|kl) D_jk)``.

    ``dm`` is (nao, nao) or (n, nao, nao); the results have shape (3, nao, nao)
    or (n, 3, nao, nao).  K is right for any density, J only for symmetric
    ones (``grad._jk_ip1`` handles the others).  ``omega`` > 0: the
    long-range operator erf(omega r12) / r12.
    """
    dm = np.asarray(dm, dtype=np.float64)
    single = dm.ndim == 2
    dms = np.ascontiguousarray(dm.reshape(-1, dm.shape[-2], dm.shape[-1]))
    nset, nao = dms.shape[0], dms.shape[1]
    vj = np.zeros((nset, 3, nao, nao))
    vk = np.zeros((nset, 3, nao, nao))
    get_extension().jk_ip1(basis_tables(mol), _boys_table(), dms, vj, vk, bool(with_j), bool(with_k), float(tol),
                           _omega4c(omega))
    vj, vk = -vj, -vk
    if single:
        vj, vk = vj[0], vk[0]
    return (vj if with_j else None), (vk if with_k else None)


def grad2e(mol, dm_j, dm_k, j_factor=1.0, k_factor=1.0, tol=1e-14, omega=0.0):
    """Two-electron part of the nuclear gradient at fixed densities, shape (natm, 3).

    The derivative of ``E2 = 1/2 sum_ijkl (ij|kl) G_ijkl`` with respect to the
    nuclear coordinates, where
    ``G_ijkl = j_factor Dj_ij Dj_kl - k_factor/2 sum_s (Dk_s,ik Dk_s,jl + Dk_s,il Dk_s,jk)``
    for the symmetric density ``dm_j`` and the symmetric densities
    ``dm_k`` ((nao, nao) or (n, nao, nao)).  RHF: ``grad2e(mol, D, D, 1, 0.5)``;
    UHF: ``grad2e(mol, Da + Db, (Da, Db))``.  The derivative integrals are
    contracted with ``G`` as they are evaluated (8-fold symmetry, nothing stored).
    ``omega`` > 0: the long-range operator erf(omega r12) / r12.
    """
    nao = mol.nao_nr()
    dmj = np.ascontiguousarray(dm_j, dtype=np.float64).reshape(nao, nao)
    dmk = np.ascontiguousarray(dm_k, dtype=np.float64).reshape(-1, nao, nao)
    de = np.zeros((mol.natm, 3))
    get_extension().grad2e(basis_tables(mol), _boys_table(), dmj, dmk, float(j_factor), float(k_factor), float(tol), de,
                           _omega4c(omega))
    return de


def h1_jk(mol, dm_j=(), dm_k=(), tol=1e-14, omega=0.0):
    """First-order Coulomb and exchange matrices of all nuclear displacements at fixed densities.

    Returns ``(vj, vk)``: ``vj[A, x, s] = d/dR_Ax J[dm_j[s]]`` and
    ``vk[A, x, s] = d/dR_Ax K[dm_k[s]]`` (natm, 3, n, nao, nao), the integrals
    differentiated at all four centres (the two-electron part of pyscf's
    Hessian ``make_h1``), from one pass over the unique shell quartets; the
    densities must be symmetric.  ``omega`` > 0: erf(omega r12) / r12.
    """
    nao = mol.nao_nr()
    dmj = np.ascontiguousarray(np.asarray(dm_j, dtype=np.float64).reshape(-1, nao, nao))
    dmk = np.ascontiguousarray(np.asarray(dm_k, dtype=np.float64).reshape(-1, nao, nao))
    nj, nk = len(dmj), len(dmk)
    out = np.zeros((mol.natm, 3, nj + nk, nao, nao))
    if nj + nk:
        get_extension().h1_2e(basis_tables(mol), _boys_table(), dmj, dmk, float(tol), out, _omega4c(omega))
    return out[:, :, :nj], out[:, :, nj:]


def hess2e(mol, dm_j, dm_k, j_factor=1.0, k_factor=1.0, tol=1e-14, omega=0.0):
    """Second derivatives (natm, natm, 3, 3) of the two-electron energy of :func:`grad2e` at fixed densities.

    ``E2 = 1/2 sum_ijkl (ij|kl) G_ijkl`` with G as in :func:`grad2e` (RHF:
    ``hess2e(mol, D, D, 1, 0.5)``; UHF: ``hess2e(mol, Da + Db, (Da, Db))``):
    the two-electron part of pyscf's partial Hessian (``ej - ek`` of
    ``hessian.rhf._partial_hess_ejk``), from the second-derivative integrals
    contracted as they are produced.  ``omega`` > 0: erf(omega r12) / r12.
    """
    nao = mol.nao_nr()
    dmj = np.ascontiguousarray(dm_j, dtype=np.float64).reshape(nao, nao)
    dmk = np.ascontiguousarray(dm_k, dtype=np.float64).reshape(-1, nao, nao)
    hess = np.zeros((mol.natm, mol.natm, 3, 3))
    get_extension().hess2e(basis_tables(mol), _boys_table(), dmj, dmk, float(j_factor), float(k_factor), float(tol),
                           hess, _omega4c(omega))
    return hess


def grad2e_pairs(mol, j_pairs=(), k_pairs=(), tol=1e-14, omega=0.0):
    """Nuclear gradient (natm, 3) of ``E = sum_p c_p sum (ij|kl) L_ij R_kl + sum_q c_q sum (ij|kl) A_jk B_il``.

    ``j_pairs``: ``(c, L, R)`` with symmetric L, R (Coulomb-type products);
    ``k_pairs``: ``(c, A, B)`` with A, B both symmetric or both antisymmetric
    (exchange-type products).  The derivative integrals are contracted as they
    are produced, with the eight-fold permutational symmetry (``grad2e``
    generalised to several density pairs: the two-electron term of
    excited-state gradients).  ``omega`` > 0: erf(omega r12) / r12.
    """
    nao = mol.nao_nr()

    def stack(pairs, k):
        if not pairs:
            return np.zeros((0, nao, nao))
        return np.ascontiguousarray([np.asarray(p[k], dtype=np.float64).reshape(nao, nao) for p in pairs])

    jl, jr, kl, kr = stack(j_pairs, 1), stack(j_pairs, 2), stack(k_pairs, 1), stack(k_pairs, 2)
    jc = np.array([float(p[0]) for p in j_pairs], dtype=np.float64)
    kc = np.array([float(p[0]) for p in k_pairs], dtype=np.float64)
    de = np.zeros((mol.natm, 3))
    if not len(jc) and not len(kc):
        return de
    get_extension().grad2e_pairs(basis_tables(mol), _boys_table(), jl, jr, jc, kl, kr, kc, float(tol), de,
                                 _omega4c(omega))
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


def grad2e_df_rdm2(mol, auxmol, orbs, dm2, max_memory=4000, tol=1e-14):
    """Gradient (natm, 3) of the density-fitted energy ``E = 1/2 sum_uvwx (uv|wx) dm2[u,v,w,x]``.

    ``(uv|wx) = sum_PQ (uv|P) V^-1_PQ (Q|wx)`` with the orbitals ``orbs``
    (nao, n), at fixed orbitals and ``dm2`` (the active-space two-particle
    density of pyscf's CASSCF, ``fcisolver.make_rdm12``), including the
    response of the auxiliary basis: the 2-RDM part of pyscf's DF-CASSCF
    gradient (``grad_elec_dferi`` and ``grad_elec_auxresponse_dferi``).
    With X = V^-1 (P|uv) and G the 2-RDM symmetrised in uv and in wx,

        dE = sum_{P, mu nu} d(mu nu|P) [C D_P C^T]_mu nu - 1/2 sum_PQ d(P|Q) W_PQ
        D_P = sum_wx G_uvwx X_P,wx,   W = X G X^T

    the exchange-type terms of :func:`grad2e_df` with D_P in place of the
    fitted orbital products, so the same Mojo kernels evaluate it.
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
    c = np.ascontiguousarray(np.asarray(orbs, dtype=np.float64).reshape(1, nao, -1))
    n = c.shape[2]
    blk = max(1, min(int(max_memory * 1e6 / 8 / 2 / npair), naux))
    rhoj = np.empty(naux)
    xs = np.empty((1, naux, n * (n + 1) // 2))
    omega = _omega(mol, 0.0)
    ext.df_grad_rhs(tables, aux_tables, table, np.zeros(npair), c, blk, rhoj, xs, seq_path, seq_prefix, omega)
    x = lib.unpack_tril(_gen_metric_solver(int2c2e(auxmol, omega))(xs[0])).reshape(naux, n * n)
    g = np.asarray(dm2, dtype=np.float64).reshape(n, n, n, n)
    g = g + g.transpose(1, 0, 2, 3)
    g = (g + g.transpose(0, 1, 3, 2)) * 0.25
    d = lib.dot(x, g.reshape(n * n, n * n).T)                          # D_P,uv
    w = lib.dot(d, x.T)
    de = np.zeros((auxmol.natm, 3))
    ext.grad2c(aux_tables, table, np.ascontiguousarray(w), de, omega)
    w = x = None
    dpk = np.ascontiguousarray(lib.pack_tril(d.reshape(naux, n, n))[None])
    d3 = np.zeros((mol.natm, 3))
    ext.grad_df3c(
        tables, aux_tables, table, np.zeros(naux), np.zeros(npair), 0.0, -1.0, dpk, c, blk, float(tol), d3,
        seq_path, seq_prefix, omega,
    )
    return de + d3


def grad2e_df_casscf(mol, auxmol, mo_core, mo_cas, casdm1, casdm2, max_memory=4000, tol=1e-14):
    """Two-electron part (natm, 3) of the density-fitted CASSCF gradient in one pass over the integrals.

    The energy is E2(Dc + Da) - E2(Da) + 1/2 sum_uvwx (uv|wx) G_uvwx with
    E2(D) = 1/2 Tr D (J - K/2)[D], Dc the core and Da the active density
    (``casdm1``), G the 2-RDM (``casdm2``): the sum of :func:`grad2e_df` for
    Dc + Da, minus it for Da, and :func:`grad2e_df_rdm2`.  In the basis O of
    the core and the active natural orbitals (occupations n) every term is
    of the form O M_P O^T for each auxiliary function P,

        M_P = diag(c1_P n) - 1/2 (n n^T) o X_P
              + [active block] (-diag(ca_P w) + 1/2 (w w^T) o X_P + U^T D_P U)

    (c = V^-1 rho of Dc + Da and of Da, X = V^-1 (P|ij), D_P the 2-RDM term
    of :func:`grad2e_df_rdm2` in the active MOs, U the natural orbitals),
    and the metric terms add up to one W, so the three-centre derivative
    integrals are generated once.
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
    mo_core = np.asarray(mo_core, dtype=np.float64).reshape(nao, -1)
    mo_cas = np.asarray(mo_cas, dtype=np.float64).reshape(nao, -1)
    ncore, ncas = mo_core.shape[1], mo_cas.shape[1]
    m = ncore + ncas
    w, u = np.linalg.eigh(np.asarray(casdm1, dtype=np.float64))
    w = np.clip(w, 0.0, None)
    orb = np.ascontiguousarray(np.hstack([mo_core, mo_cas @ u])[None])     # core and natural orbitals
    n1 = np.concatenate([np.full(ncore, 2.0), w])
    blk = max(1, min(int(max_memory * 1e6 / 8 / 2 / npair), naux))
    rhoj = np.empty(naux)
    xs = np.empty((1, naux, m * (m + 1) // 2))
    omega = _omega(mol, 0.0)
    ext.df_grad_rhs(tables, aux_tables, table, np.zeros(npair), orb, blk, rhoj, xs, seq_path, seq_prefix, omega)
    solve = _gen_metric_solver(int2c2e(auxmol, omega))
    diag = np.arange(m) * (np.arange(m) + 1) // 2 + np.arange(m)
    adiag = xs[0][:, diag]                                                # (P|ii)
    c1 = solve(adiag @ n1)
    ca = solve(adiag[:, ncore:] @ w)
    x = lib.unpack_tril(solve(xs[0]))                                     # (naux, m, m)
    xs = adiag = None
    xa = x[:, ncore:, ncore:]
    # the 2-RDM term in the active MOs (u v), with X in that basis
    xu = np.einsum("ui,pij,vj->puv", u, xa, u).reshape(naux, ncas * ncas)
    g = np.asarray(casdm2, dtype=np.float64).reshape(ncas, ncas, ncas, ncas)
    g = g + g.transpose(1, 0, 2, 3)
    g = (g + g.transpose(0, 1, 3, 2)) * 0.25
    dg = lib.dot(xu, g.reshape(ncas * ncas, ncas * ncas).T)               # D_P,uv
    wmat = lib.dot(dg, xu.T) + np.outer(c1, c1) - np.outer(ca, ca)
    nn = np.outer(n1, n1)
    ww = np.outer(w, w)
    wmat -= 0.5 * lib.dot(x.reshape(naux, m * m) * nn.ravel(), x.reshape(naux, m * m).T)
    wmat += 0.5 * lib.dot(xa.reshape(naux, -1) * ww.ravel(), np.ascontiguousarray(xa).reshape(naux, -1).T)
    de = np.zeros((auxmol.natm, 3))
    ext.grad2c(aux_tables, table, np.ascontiguousarray(wmat), de, omega)
    wmat = xu = None
    mmat = x * (-0.5 * nn)
    mmat[:, ncore:, ncore:] += xa * (0.5 * ww) + np.einsum("iu,puv,vj->pij", u.T, dg.reshape(naux, ncas, ncas), u)
    idx = np.arange(m)
    mmat[:, idx, idx] += c1[:, None] * n1
    mmat[:, ncore + np.arange(ncas), ncore + np.arange(ncas)] -= ca[:, None] * w
    mpk = np.ascontiguousarray(lib.pack_tril(mmat)[None])
    x = xa = mmat = None
    d3 = np.zeros((mol.natm, 3))
    ext.grad_df3c(
        tables, aux_tables, table, np.zeros(naux), np.zeros(npair), 0.0, -1.0, mpk, orb, blk, float(tol), d3,
        seq_path, seq_prefix, omega,
    )
    return de + d3


def grad2e_df_terms(mol, auxmol, orbs, jk_terms=(), rdm2_terms=(), max_memory=4000, tol=1e-14):
    """Nuclear gradient (natm, 3) of a sum of density-fitted two-electron energies, at fixed coefficients,
    in one pass over the three-centre derivative integrals (auxiliary-basis response included).

    ``orbs`` (nao, m) is a basis O; A_P = (P|ij) over it, X = V^-1 A.

    * ``jk_terms``: ``(f, Y1, Y2)`` for f Tr(D1 (J - K/2)[D2]) with D = O Y O^T
      (Y symmetric, m x m): f [rho1^T V^-1 rho2 - 1/2 sum_PQ Tr(A_P Y2 A_Q Y1) V^-1_PQ].
    * ``rdm2_terms``: ``(f, left, right, G)`` for f sum_PQ L_P,uv V^-1_PQ R_Q,wx G_uvwx,
      L_P,uv = sum over ``(i, j)`` in ``left`` of A_P[i_u, j_v] (``i``, ``j``
      index arrays into O), R likewise from ``right``.

    Every term contributes to one matrix M_P per auxiliary function (the
    gradient is sum d(mu nu|P) [O M_P O^T]_mu nu - 1/2 sum d(P|Q) W_PQ) and
    to W, as :func:`grad2e_df_casscf` does for the CASSCF energy.
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
    orb = np.ascontiguousarray(np.asarray(orbs, dtype=np.float64).reshape(1, nao, -1))
    m = orb.shape[2]
    blk = max(1, min(int(max_memory * 1e6 / 8 / 2 / npair), naux))
    rhoj = np.empty(naux)
    xs = np.empty((1, naux, m * (m + 1) // 2))
    omega = _omega(mol, 0.0)
    ext.df_grad_rhs(tables, aux_tables, table, np.zeros(npair), orb, blk, rhoj, xs, seq_path, seq_prefix, omega)
    solve = _gen_metric_solver(int2c2e(auxmol, omega))
    a2 = lib.unpack_tril(xs[0]).reshape(naux, m * m)
    x = lib.unpack_tril(solve(xs[0]))                                    # (naux, m, m)
    xs = None
    mmat = np.zeros((naux, m, m))
    wmat = np.zeros((naux, naux))

    def right(b, y):                                                     # b_P y for all P, one GEMM
        return lib.dot(b.reshape(naux * m, m), y).reshape(naux, m, m)

    def left(y, b):                                                      # y b_P for all P, one GEMM
        return lib.dot(y, np.ascontiguousarray(b.transpose(1, 0, 2)).reshape(m, naux * m)).reshape(
            m, naux, m).transpose(1, 0, 2)

    for f, y1, y2 in jk_terms:
        y1 = np.asarray(y1, dtype=np.float64)
        y2 = np.asarray(y2, dtype=np.float64)
        c1 = solve(a2 @ y1.ravel())
        c2 = solve(a2 @ y2.ravel())
        mmat += f * (c2[:, None, None] * y1 + c1[:, None, None] * y2)
        xy1 = right(x, y1)
        xy2 = right(x, y2)
        y1xy2 = left(y1, xy2)                                              # Y1 X_P Y2
        mmat -= 0.5 * f * (y1xy2 + y1xy2.transpose(0, 2, 1))
        # Tr(X_P Y2 X_Q Y1) = sum_ij (X_P Y2)_ij (Y1 X_Q)_ij, Y1 X_Q = (X_Q Y1)^T
        t = lib.dot(xy2.reshape(naux, m * m), np.ascontiguousarray(xy1.transpose(0, 2, 1)).reshape(naux, m * m).T)
        wmat += f * (np.outer(c1, c2) + np.outer(c2, c1)) - 0.5 * f * (t + t.T)
        xy1 = xy2 = y1xy2 = t = None
    for f, left, right, g in rdm2_terms:
        g = np.asarray(g, dtype=np.float64)
        n = g.shape[0]
        g2 = g.reshape(n * n, n * n)
        xl = sum(x[:, np.asarray(i)[:, None], np.asarray(j)[None, :]] for i, j in left).reshape(naux, n * n)
        xr = sum(x[:, np.asarray(i)[:, None], np.asarray(j)[None, :]] for i, j in right).reshape(naux, n * n)
        dl = (f * lib.dot(xr, g2.T)).reshape(naux, n, n)
        dr = (f * lib.dot(xl, g2)).reshape(naux, n, n)
        for i, j in left:
            mmat[:, np.asarray(i)[:, None], np.asarray(j)[None, :]] += dl
        for i, j in right:
            mmat[:, np.asarray(i)[:, None], np.asarray(j)[None, :]] += dr
        wl = lib.dot(lib.dot(xl, g2), xr.T)
        wmat += f * (wl + wl.T)
    x = a2 = None
    de = np.zeros((auxmol.natm, 3))
    ext.grad2c(aux_tables, table, np.ascontiguousarray(wmat), de, omega)
    wmat = None
    mpk = np.ascontiguousarray(lib.pack_tril((mmat + mmat.transpose(0, 2, 1)) * 0.5)[None])
    mmat = None
    d3 = np.zeros((mol.natm, 3))
    ext.grad_df3c(
        tables, aux_tables, table, np.zeros(naux), np.zeros(npair), 0.0, -1.0, mpk, orb, blk, float(tol), d3,
        seq_path, seq_prefix, omega,
    )
    return de + d3
