"""SCF stability analysis (mojoscf.stability) against pyscf.scf.stability."""
import numpy as np
import pytest
from pyscf import dft, gto, scf
from pyscf.scf import stability as pstab
from pyscf.soscf import newton_ah

import mojoscf
from mojoscf import stability as mstab

WATER = "O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587"


def _pair(mol, xc, df):
    """(Mojo object, plain pyscf object with the same orbitals)."""
    unrestricted = mol.spin > 0

    def make():
        if xc is None:
            mf = (scf.UHF if unrestricted else scf.RHF)(mol)
        else:
            mf = (dft.UKS if unrestricted else dft.RKS)(mol, xc=xc)
        return mf.density_fit() if df else mf

    mf = make()
    if xc is None:
        mf = (mojoscf.UHF if unrestricted else mojoscf.RHF)(mol)
        if df:
            mf = mf.density_fit()
    else:
        mojoscf.dft.accelerate(mf)
    mf.conv_tol = 1e-10
    mf.kernel()
    ref = make()
    for key in ("mo_coeff", "mo_occ", "mo_energy", "e_tot", "converged"):
        setattr(ref, key, getattr(mf, key))
    return mf, ref


def _close(a, b, tol=1e-9):
    return abs(np.asarray(a) - np.asarray(b)).max() < tol * max(1.0, abs(np.asarray(b)).max())


@pytest.mark.parametrize("df", [True, False])
@pytest.mark.parametrize("xc", [None, "pbe", "b3lyp", "camb3lyp"])
def test_rhf_hessians_match_pyscf(xc, df):
    mol = gto.M(atom=WATER, basis="def2-svp", verbose=0)
    mf, ref = _pair(mol, xc, df)
    aop, g, hdiag = mstab.internal_hessian(mf)
    g0, hop, hdiag0 = newton_ah.gen_g_hop_rhf(ref, ref.mo_coeff, ref.mo_occ)
    assert _close(g, g0, 1e-8) and _close(hdiag, 2 * hdiag0)
    x = np.random.default_rng(1).normal(size=(3, g.size))
    assert _close(aop(list(x)), [2 * hop(v) for v in x])
    hop1, hop2, hd = mstab.external_hessians(mf)
    r1, rd1, r2, rd2 = pstab._gen_hop_rhf_external(ref)
    assert _close(hd, rd1) and _close(hd, rd2)
    assert _close(hop1(list(x)), [r1(v) for v in x])
    assert _close(hop2(list(x)), [r2(v) for v in x])


@pytest.mark.parametrize("df", [True, False])
@pytest.mark.parametrize("xc", [None, "b3lyp", "wb97x"])
def test_uhf_hessian_matches_pyscf(xc, df):
    mol = gto.M(atom=WATER, basis="def2-svp", charge=1, spin=1, verbose=0)
    mf, ref = _pair(mol, xc, df)
    aop, g, hdiag = mstab.internal_hessian(mf)
    g0, hop, hdiag0 = newton_ah.gen_g_hop_uhf(ref, ref.mo_coeff, ref.mo_occ)
    assert _close(g, g0, 1e-8) and _close(hdiag, 2 * hdiag0)
    x = np.random.default_rng(2).normal(size=(3, g.size))
    assert _close(aop(list(x)), [2 * hop(v) for v in x])
    assert mstab.external_hessians(mf) is None


def test_instabilities_found_and_followed():
    """Stretched H2: the RHF solution is RHF -> UHF unstable and the UHF one from it internally unstable."""
    mol = gto.M(atom="H 0 0 0; H 0 0 2.2", basis="cc-pvdz", verbose=0)
    mf = mojoscf.RHF(mol).run()
    mo_i, mo_e, stable_i, stable_e = mf.stability(external=True, return_status=True)
    ref = scf.RHF(mol).run()
    r_i, r_e, rs_i, rs_e = ref.stability(external=True, return_status=True)
    assert (stable_i, stable_e) == (rs_i, rs_e) == (True, False)
    # the UHF from the rotated orbitals breaks the spin symmetry and lies lower
    uhf = mojoscf.UHF(mol)
    dm0 = uhf.make_rdm1(mo_e, (mf.mo_occ / 2, mf.mo_occ / 2))
    uhf.kernel(dm0=dm0)
    assert uhf.e_tot < mf.e_tot - 1e-3
    # a UHF converged to the restricted solution is internally unstable
    umf = mojoscf.UHF(mol).run()
    assert abs(umf.e_tot - mf.e_tot) < 1e-8
    mo, stable = umf.stability(return_status=True)[0::2]
    assert stable is False
    uref = scf.UHF(mol).run()
    assert pstab.uhf_internal(uref, return_status=True)[1] is False
    umf.kernel(dm0=umf.make_rdm1(mo, umf.mo_occ))
    assert umf.e_tot < mf.e_tot - 1e-3


def test_accelerated_ks_stability_and_fallbacks():
    mol = gto.M(atom=WATER, basis="def2-svp", charge=1, spin=1, verbose=0)
    mf = mojoscf.dft.accelerate(dft.UKS(mol, xc="b3lyp").density_fit()).run()
    assert mstab.internal_hessian(mf) is not None
    mo, _, stable, _ = mf.stability(return_status=True)
    assert stable and mo is mf.mo_coeff
    # point-group symmetry labels: pyscf's code
    sym = gto.M(atom=WATER, basis="def2-svp", symmetry=True, verbose=0)
    mf = mojoscf.dft.accelerate(dft.RKS(sym, xc="pbe")).run()
    assert mstab.internal_hessian(mf) is None
    assert mf.stability(return_status=True)[2]
