"""mojoscf.cphf: the Krylov CPHF solver in matrix products against pyscf's cphf / ucphf solvers."""
import numpy as np
import pytest
from pyscf.lib import linalg_helper
from pyscf.scf import cphf, ucphf

from mojoscf import cphf as mojo_cphf


def _operator(rng, n, scale=0.05):
    a = rng.standard_normal((n, n)) * scale
    return lambda x: (np.asarray(x).reshape(len(x), -1) @ a.T).reshape(np.shape(x))


@pytest.mark.parametrize("nset", [1, 7])
def test_krylov_matches_pyscf(rng, nset):
    n = 120
    a = rng.standard_normal((n, n)) * 0.05
    b = rng.standard_normal((nset, n))
    aop = lambda x: np.asarray(x) @ a.T
    ref = linalg_helper.krylov(aop, b, tol=1e-12, max_cycle=60)
    x = mojo_cphf.krylov(aop, b, tol=1e-12, max_cycle=60)
    assert abs(x - ref).max() < 1e-10
    assert abs(x + aop(x) - b).max() < 1.01 * abs(ref + aop(ref) - b).max() + 1e-12
    x1 = mojo_cphf.krylov(aop, b[0], tol=1e-12, max_cycle=60)
    assert x1.shape == (n,) and abs(x1 - linalg_helper.krylov(aop, b[0], tol=1e-12, max_cycle=60)).max() < 1e-10


def test_krylov_dependent_right_hand_sides(rng):
    """Linearly dependent right-hand sides take pyscf's Gram-Schmidt route and the same solution."""
    n = 80
    a = rng.standard_normal((n, n)) * 0.05
    b = rng.standard_normal((3, n))
    b = np.vstack([b, b[0] + 2 * b[1]])
    aop = lambda x: np.asarray(x) @ a.T
    x = mojo_cphf.krylov(aop, b, tol=1e-12, max_cycle=60)
    ref = linalg_helper.krylov(aop, b, tol=1e-12, max_cycle=60)
    assert abs(x - ref).max() < 1e-10
    assert abs(x + aop(x) - b).max() < 1.01 * abs(ref + aop(ref) - b).max() + 1e-12


@pytest.mark.parametrize("nset", [1, 6])
def test_solve_withs1_rhf(rng, nset):
    nmo, nocc = 14, 5
    mo_energy = np.sort(rng.standard_normal(nmo)) + np.r_[np.zeros(nocc), np.full(nmo - nocc, 1.5)]
    mo_occ = np.r_[np.full(nocc, 2.0), np.zeros(nmo - nocc)]
    h1 = rng.standard_normal((nset, nmo, nocc))
    s1 = rng.standard_normal((nset, nmo, nocc)) * 0.1
    fvind = _operator(rng, nmo * nocc)
    ref = cphf.solve(fvind, mo_energy, mo_occ, h1, s1, tol=1e-11, max_cycle=80)
    out = mojo_cphf.solve(fvind, mo_energy, mo_occ, h1, s1, tol=1e-11, max_cycle=80)
    for r, o in zip(ref, out):
        assert r.shape == o.shape and abs(r - o).max() < 1e-9
    ref = cphf.solve(fvind, mo_energy, mo_occ, h1[0], s1[0], tol=1e-11, max_cycle=80)
    out = mojo_cphf.solve(fvind, mo_energy, mo_occ, h1[0], s1[0], tol=1e-11, max_cycle=80)
    for r, o in zip(ref, out):
        assert r.shape == o.shape and abs(r - o).max() < 1e-9


def test_solve_withs1_uhf(rng):
    nset, nmo, nocca, noccb = 5, 13, 6, 4
    mo_energy = np.array([np.sort(rng.standard_normal(nmo)) + np.r_[np.zeros(n), np.full(nmo - n, 1.5)]
                          for n in (nocca, noccb)])
    mo_occ = np.array([np.r_[np.ones(n), np.zeros(nmo - n)] for n in (nocca, noccb)])
    h1 = (rng.standard_normal((nset, nmo, nocca)), rng.standard_normal((nset, nmo, noccb)))
    s1 = (rng.standard_normal((nset, nmo, nocca)) * 0.1, rng.standard_normal((nset, nmo, noccb)) * 0.1)
    fvind = _operator(rng, nmo * (nocca + noccb))
    ref = ucphf.solve(fvind, mo_energy, mo_occ, h1, s1, tol=1e-11, max_cycle=80)
    out = mojo_cphf.solve(fvind, mo_energy, mo_occ, h1, s1, tol=1e-11, max_cycle=80)
    for rs, os_ in zip(ref, out):
        for r, o in zip(rs, os_):
            assert r.shape == o.shape and abs(r - o).max() < 1e-9
