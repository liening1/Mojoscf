"""Second-order SCF (mojoscf.soscf) against pyscf.soscf.newton_ah."""
import numpy as np
import pytest
from pyscf import dft, gto, scf
from pyscf.soscf import newton_ah
from scipy.linalg import expm

import mojoscf
from mojoscf import soscf

WATER = "O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587"


def _make(mol, xc, df):
    unrestricted = mol.spin > 0
    if xc is None:
        mf = (scf.UHF if unrestricted else scf.RHF)(mol)
    else:
        mf = (dft.UKS if unrestricted else dft.RKS)(mol, xc=xc)
    return mf.density_fit() if df else mf


def _mojo(mol, xc, df):
    if xc is None:
        mf = (mojoscf.UHF if mol.spin > 0 else mojoscf.RHF)(mol)
        return mf.density_fit() if df else mf
    return mojoscf.dft.accelerate(_make(mol, xc, df))


def _rotated(mf, rng):
    """Orbitals of ``mf`` turned away from the SCF solution (so that the gradient is not small)."""
    def turn(c):
        a = rng.normal(size=(c.shape[1],) * 2) * 0.05
        return c @ expm(a - a.T)

    if mf.mo_coeff.ndim == 3:
        return np.array([turn(c) for c in mf.mo_coeff])
    return turn(mf.mo_coeff)


def _close(a, b, tol=1e-10):
    return abs(np.asarray(a) - np.asarray(b)).max() < tol * max(1.0, abs(np.asarray(b)).max())


@pytest.mark.parametrize("spin,xc,df", [
    (0, None, False), (0, "b3lyp", True), (0, "camb3lyp", False),
    (1, None, True), (1, "b3lyp", False), (1, "tpss", True),
])
def test_gen_g_hop_matches_pyscf(spin, xc, df):
    mol = gto.M(atom=WATER, basis="def2-svp", charge=spin, spin=spin, verbose=0)
    mf = _mojo(mol, xc, df).run()
    rng = np.random.default_rng(3)
    c, occ = _rotated(mf, rng), mf.mo_occ
    ref = _make(mol, xc, df)
    pyscf_g_hop = newton_ah.gen_g_hop_uhf if spin else newton_ah.gen_g_hop_rhf
    g0, hop0, hdiag0 = pyscf_g_hop(ref, c, occ)
    x = rng.normal(size=g0.size)
    y0 = hop0(x)
    fock_ao = ref.get_fock(dm=ref.make_rdm1(c, occ))
    for args in ((), (fock_ao,)):
        g, hop, hdiag = soscf.gen_g_hop(mf.newton(), c, occ, *args)
        assert _close(g, g0) and _close(hdiag, hdiag0)
        assert _close(hop(x), y0)


@pytest.mark.parametrize("spin,xc", [(0, "b3lyp"), (1, None), (1, "pbe0")])
def test_newton_scf_matches_pyscf(spin, xc, monkeypatch):
    mol = gto.M(atom=WATER, basis="def2-svp", charge=spin, spin=spin, verbose=0)
    calls = []
    ours = soscf.gen_g_hop

    def counted(*args, **kwargs):
        out = ours(*args, **kwargs)
        calls.append(out is not None)
        return out

    monkeypatch.setattr(soscf, "gen_g_hop", counted)
    mf = _mojo(mol, xc, True).newton()
    mf.conv_tol = 1e-10
    mf.kernel()
    assert mf.converged and calls and all(calls)
    ref = _make(mol, xc, True).newton()
    ref.conv_tol = 1e-10
    ref.kernel()
    assert abs(mf.e_tot - ref.e_tot) < 1e-8


def test_pyscf_paths_kept():
    """Symmetry-adapted and ROHF objects, and plain pyscf objects, keep pyscf's gen_g_hop."""
    assert newton_ah._SecondOrderRHF.gen_g_hop._mojoscf_orig is newton_ah.gen_g_hop_rhf
    assert newton_ah._SecondOrderUHF.gen_g_hop._mojoscf_orig is newton_ah.gen_g_hop_uhf
    mol = gto.M(atom=WATER, basis="def2-svp", symmetry=True, verbose=0)
    mf = mojoscf.dft.accelerate(dft.RKS(mol, xc="pbe").density_fit()).run()
    assert soscf.gen_g_hop(mf.newton(), mf.mo_coeff, mf.mo_occ) is None
    assert soscf.gen_g_hop(mf.newton(), mf.mo_coeff, mf.mo_occ, with_symmetry=False) is not None
    plain = scf.RHF(mol).run()
    assert soscf.gen_g_hop(plain.newton(), plain.mo_coeff, plain.mo_occ) is None
    mol = gto.M(atom=WATER, basis="def2-svp", charge=1, spin=1, verbose=0)
    rohf = mojoscf.dft.accelerate(dft.ROKS(mol, xc="pbe").density_fit())
    rohf.kernel()
    e = rohf.newton().kernel()
    assert abs(e - dft.ROKS(mol, xc="pbe").density_fit().newton().kernel()) < 1e-8
