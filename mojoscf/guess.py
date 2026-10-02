"""Initial densities for broken-symmetry (BS) UHF calculations.

A UHF run started from a spin-symmetric density stays on the symmetric solution
even where a lower broken-symmetry solution exists, so BS calculations need a
spin-polarised starting density.  The builders below return a ``(2, nao, nao)``
array to pass as ``mf.kernel(dm0=...)``.
"""
from __future__ import annotations

import numpy as np


__all__ = ["atom_ao_slices", "afm_guess_by_atom", "flip_spin_on_atoms", "mix_homo_lumo_guess"]

def atom_ao_slices(mol):
    """``[(atom_index, ao_start, ao_stop)]`` for every atom."""
    return [(ia, p0, p1) for ia, (b0, b1, p0, p1) in enumerate(mol.aoslice_by_atom())]


def afm_guess_by_atom(mol, up_atoms):
    """Spin-polarised atomic-density guess: atoms in ``up_atoms`` carry alpha spin,
    all other atoms beta spin (closed-shell atoms contribute half to each channel).

    Intended for systems where every atom has one unpaired electron (hydrogen chains).
    """
    from pyscf.scf import hf

    dm = hf.init_guess_by_atom(mol)  # block-diagonal, total density
    dma = np.zeros_like(dm)
    dmb = np.zeros_like(dm)
    for ia, p0, p1 in atom_ao_slices(mol):
        blk = dm[p0:p1, p0:p1]
        if ia in up_atoms:
            dma[p0:p1, p0:p1] = blk
        else:
            dmb[p0:p1, p0:p1] = blk
    return np.array((dma, dmb))


def flip_spin_on_atoms(dm_hs, mol, flip_atoms):
    """Broken-symmetry guess from a high-spin UHF density.

    Alpha and beta densities are exchanged on the AO block of ``flip_atoms``; the
    blocks coupling them to the rest of the molecule are replaced by the spin
    average.  The result is only a starting point (it is not idempotent).
    """
    dma, dmb = dm_hs
    mask = np.zeros(mol.nao_nr(), dtype=bool)
    for ia, p0, p1 in atom_ao_slices(mol):
        if ia in flip_atoms:
            mask[p0:p1] = True
    out_a, out_b = dma.copy(), dmb.copy()
    both = np.outer(mask, mask)
    cross = np.outer(mask, ~mask) | np.outer(~mask, mask)
    out_a[both], out_b[both] = dmb[both], dma[both]
    avg = 0.5 * (dma + dmb)
    out_a[cross] = avg[cross]
    out_b[cross] = avg[cross]
    return np.array((out_a, out_b))


def mix_homo_lumo_guess(mol, angle=np.pi / 4, **kw):
    """Gaussian-style ``guess=mix``: rotate alpha HOMO/LUMO by +angle and beta by -angle.

    Starts from the converged restricted solution of ``mol``.
    """
    from pyscf import scf

    rhf = scf.RHF(mol)
    rhf.verbose = 0
    rhf.kernel()
    c = rhf.mo_coeff.copy()
    nocc = mol.nelectron // 2
    homo, lumo = c[:, nocc - 1].copy(), c[:, nocc].copy()
    ca, cb = c.copy(), c.copy()
    ca[:, nocc - 1] = np.cos(angle) * homo + np.sin(angle) * lumo
    cb[:, nocc - 1] = np.cos(angle) * homo - np.sin(angle) * lumo
    occ = rhf.mo_occ / 2
    dma = (ca * occ) @ ca.T
    dmb = (cb * occ) @ cb.T
    return np.array((dma, dmb))
