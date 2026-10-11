"""FCI 3-particle density matrices and NEVPT2's 4-particle contractions (mojoscf.fcirdm) against pyscf."""
import numpy as np
import pytest
from pyscf import fci
from pyscf.fci import cistring
from pyscf.mrpt import nevpt2

import mojoscf  # noqa: F401  (installs the replacements)
from mojoscf import fcirdm

CASES = [(5, (3, 2)), (6, (3, 3)), (5, (2, 2)), (7, (4, 2)), (4, (4, 0)), (6, (0, 3)), (6, (1, 1)), (8, (4, 4))]


def _vectors(norb, nelec, seed):
    rng = np.random.default_rng(seed)
    na = cistring.num_strings(norb, nelec[0])
    nb = cistring.num_strings(norb, nelec[1])
    return rng.standard_normal((na, nb)), rng.standard_normal((na, nb))


def _eri(norb, seed):
    eri = np.random.default_rng(seed).standard_normal((norb,) * 4)
    eri = eri + eri.transpose(1, 0, 2, 3)
    eri = eri + eri.transpose(0, 1, 3, 2)
    return eri + eri.transpose(2, 3, 0, 1)


def test_installed():
    assert fci.rdm.make_dm123 is fcirdm.make_dm123
    assert nevpt2._contract4pdm is fcirdm.contract4pdm


@pytest.mark.parametrize("norb,nelec", CASES)
def test_make_dm123_matches_pyscf(norb, nelec):
    """<p+q r+s t+u> and the lower-rank matrices equal pyscf's, for expectation values and transition
    densities, closed and open shells."""
    bra, ket = _vectors(norb, nelec, norb + 10 * nelec[0] + nelec[1])
    orig = fcirdm.make_dm123._mojoscf_orig
    for b, k in ((ket, ket), (bra, ket)):
        ref = orig("FCI3pdm_kern_sf", b, k, norb, nelec)
        got = fcirdm.make_dm123("FCI3pdm_kern_sf", b, k, norb, nelec)
        for x, y in zip(ref, got):
            assert x.shape == y.shape
            assert abs(x - y).max() < 1e-12 * max(1.0, abs(x).max())


@pytest.mark.parametrize("norb,nelec", CASES)
@pytest.mark.parametrize("kern", ["NEVPTkern_cedf_aedf", "NEVPTkern_aedf_ecdf"])
def test_contract4pdm_matches_pyscf(norb, nelec, kern):
    """The 4-particle contractions of strongly contracted NEVPT2 equal pyscf's."""
    _, ci = _vectors(norb, nelec, 3 * norb + nelec[0])
    ci /= np.linalg.norm(ci)
    eri = _eri(norb, norb)
    ref = fcirdm.contract4pdm._mojoscf_orig(kern, eri, ci, norb, nelec)
    got = fcirdm.contract4pdm(kern, eri, ci, norb, nelec)
    assert abs(ref - got).max() < 1e-12 * max(1.0, abs(ref).max())


def test_other_kernels_fall_back(monkeypatch):
    """Kernels other than pyscf's spin-traced ones and work vectors that do not fit keep pyscf's code."""
    calls = []
    orig = fcirdm.make_dm123._mojoscf_orig
    monkeypatch.setitem(fcirdm._orig, "make_dm123", lambda *a: calls.append(a[0]) or orig(*a))
    bra, ket = _vectors(5, (2, 2), 1)
    fcirdm.make_dm123("FCI3pdm_kern_spin0", ket, ket, 5, (2, 2))
    monkeypatch.setattr(fcirdm, "_fits", lambda *a: False)
    fcirdm.make_dm123("FCI3pdm_kern_sf", ket, ket, 5, (2, 2))
    assert calls == ["FCI3pdm_kern_spin0", "FCI3pdm_kern_sf"]
