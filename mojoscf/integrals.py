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
momenta up to l = 8, with point nuclei and without effective core
potentials.  :func:`attach` makes a mojoscf SCF object use these integrals.
"""
from __future__ import annotations

import numpy as np
from pyscf import gto, lib
from pyscf.df import addons as df_addons

from ._backend import get_extension

__all__ = [
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
    "unsupported_reason",
]

LMAX = 8
NUC_POINT = 1
# libcint's CINTcommon_fac_sp: the factor its s and p "spherical" functions carry.
_FAC_SP = {0: 0.282094791773878143, 1: 0.488602511902919921}


def unsupported_reason(mol) -> str | None:
    """Why the Mojo engine cannot handle ``mol`` (None if it can)."""
    if mol.nbas == 0:
        return "the molecule has no basis functions"
    if mol.has_ecp():
        return "effective core potentials are not supported"
    if int(mol._bas[:, gto.ANG_OF].max()) > LMAX:
        return f"angular momentum above l = {LMAX}"
    if (mol._atm[:, gto.NUC_MOD_OF] != NUC_POINT).any():
        return "only point nuclei are supported"
    return None


def _check(mol):
    reason = unsupported_reason(mol)
    if reason is not None:
        raise NotImplementedError(f"mojoscf.integrals: {reason}")


def basis_tables(mol):
    """``(atm, bas, env, nf, c2s)``: the basis in the form the Mojo engine reads.

    ``atm``/``bas``/``env`` are pyscf's ``_atm``/``_bas``/``_env`` (as int64 /
    float64), ``nf[l]`` the number of functions per shell of angular momentum
    ``l`` and ``c2s`` the concatenated ``(ncart, nf)`` Cartesian-to-final
    transformation matrices for l = 0..lmax.
    """
    _check(mol)
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
    tables = basis_tables(mol)
    nao = mol.nao_nr()
    s = np.empty((nao, nao))
    t = np.empty((nao, nao))
    v = np.empty((nao, nao))
    get_extension().int1e(tables, s, t, v)
    return s, t, v


def get_ovlp(mol):
    return int1e(mol)[0]


def get_kin(mol):
    return int1e(mol)[1]


def get_nuc(mol):
    return int1e(mol)[2]


def get_hcore(mol):
    """``T + V`` (the core Hamiltonian of a molecule without ECPs)."""
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
    get_extension().int2e_s8(tables, eri, float(schwarz_tol))
    return eri


def int2e(mol, schwarz_tol: float = 1e-14):
    """Full ``(nao, nao, nao, nao)`` electron repulsion tensor."""
    from pyscf import ao2mo

    return ao2mo.restore(1, int2e_s8(mol, schwarz_tol), mol.nao_nr())


def int3c2e(mol, auxmol):
    """``(ab|P)`` as a ``(naux, npair)`` array with the AO pair packed in ``pack_tril`` order."""
    tables = basis_tables(mol)
    aux_tables = basis_tables(auxmol)
    nao = mol.nao_nr()
    npair = nao * (nao + 1) // 2
    out = np.empty((auxmol.nao_nr(), npair))
    get_extension().int3c2e(tables, aux_tables, out)
    return out


def int2c2e(auxmol):
    """``(P|Q)`` as a dense ``(naux, naux)`` matrix."""
    aux_tables = basis_tables(auxmol)
    naux = auxmol.nao_nr()
    out = np.empty((naux, naux))
    get_extension().int2c2e(aux_tables, out)
    return out


def cholesky_eri(mol, auxbasis="weigend+etb", auxmol=None, lindep=1e-12):
    """pyscf's in-core DF tensor ``L^{-1} (P|ab)`` with the integrals from the Mojo engine.

    Mirrors ``pyscf.df.incore.cholesky_eri`` (Cholesky factorisation of
    ``(P|Q)``, falling back to an eigen-decomposition with ``lindep`` when the
    metric is not positive definite); the factorisation and triangular solve
    are LAPACK calls through SciPy, as in pyscf.
    """
    import scipy.linalg

    if auxmol is None:
        auxmol = df_addons.make_auxmol(mol, auxbasis)
    j2c = int2c2e(auxmol)
    j3c = int3c2e(mol, auxmol)
    try:
        low = scipy.linalg.cholesky(j2c, lower=True)
        return scipy.linalg.solve_triangular(low, j3c, lower=True, overwrite_b=True, check_finite=False)
    except scipy.linalg.LinAlgError:
        w, v = scipy.linalg.eigh(j2c)
        keep = w > lindep
        v = v[:, keep] / np.sqrt(w[keep])
        return lib.dot(v.T, j3c)


def attach(mf, schwarz_tol: float = 1e-14, auxbasis=None):
    """Make the SCF object ``mf`` take its integrals from the Mojo engine.

    ``get_ovlp`` and ``get_hcore`` return precomputed Mojo matrices, the
    in-core ERI tensor (``mf._eri``) is built here, and for density-fitted
    objects the DF tensor (``mf.with_df._cderi``) is built from the Mojo
    three- and two-centre integrals.  Returns ``mf``.
    """
    mol = mf.mol
    _check(mol)
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
