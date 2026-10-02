"""Each Mojo kernel against its NumPy / pyscf reference, on both backends."""
import numpy as np
import pytest
import scipy.linalg
from pyscf import lib, scf
from pyscf.scf import diis as pyscf_diis
from pyscf.scf import hf as pyscf_hf

import mojoscf
from mojoscf import kernels


def _sym(rng, n, scale=1.0):
    a = rng.standard_normal((n, n))
    return (a + a.T) * 0.5 * scale


def _spd(rng, n):
    a = rng.standard_normal((n, n))
    return a @ a.T + n * np.eye(n)


@pytest.mark.parametrize("transa", [False, True])
@pytest.mark.parametrize("transb", [False, True])
@pytest.mark.parametrize("shape", [(3, 4, 5), (17, 9, 33), (1, 7, 2)])
def test_gemm(use_blas, rng, transa, transb, shape):
    m, k, n = shape
    a = rng.standard_normal((k, m) if transa else (m, k))
    b = rng.standard_normal((n, k) if transb else (k, n))
    c0 = rng.standard_normal((m, n))
    opa = a.T if transa else a
    opb = b.T if transb else b
    ref = 0.7 * opa @ opb - 0.3 * c0
    out = kernels.gemm(a, b, transa, transb, alpha=0.7, beta=-0.3, out=c0.copy())
    assert np.allclose(out, ref, atol=1e-12)
    assert np.allclose(kernels.gemm(a, b, transa, transb), opa @ opb, atol=1e-12)


@pytest.mark.parametrize("n", [1, 2, 6, 25])
def test_eigh_standard(use_blas, rng, n):
    h = _sym(rng, n)
    w, c = kernels.eigh(h)
    w_ref, c_ref = scipy.linalg.eigh(h)
    c_ref = pyscf_hf._adjust_phase_(c_ref)
    assert np.allclose(w, w_ref, atol=1e-11)
    assert np.allclose(h @ c, c * w, atol=1e-10)
    assert np.allclose(c.T @ c, np.eye(n), atol=1e-10)
    # Same phase convention as pyscf
    assert np.allclose(c, c_ref, atol=1e-8)


@pytest.mark.parametrize("n", [1, 3, 12, 30])
def test_eigh_generalized(use_blas, rng, n):
    h = _sym(rng, n)
    s = _spd(rng, n)
    w, c = kernels.eigh(h, s)
    w_ref, c_ref = pyscf_hf.eig(h, s)
    assert np.allclose(w, w_ref, atol=1e-10)
    assert np.allclose(h @ c, s @ c * w, atol=1e-9)
    assert np.allclose(c.T @ s @ c, np.eye(n), atol=1e-9)
    assert np.allclose(c, c_ref, atol=1e-7)


def test_eigh_with_orthogonalizer(use_blas, rng):
    n = 14
    h = _sym(rng, n)
    s = _spd(rng, n)
    x = pyscf_hf.check_linear_dependency(s)
    w, c = kernels.eigh(h, x=x)
    mf = pyscf_hf.RHF.__new__(pyscf_hf.RHF)  # only _eigh is needed
    w_ref, c_ref = pyscf_hf.SCF._eigh(mf, h, s, x=x)
    assert np.allclose(w, w_ref, atol=1e-10)
    assert np.allclose(c, c_ref, atol=1e-8)


def test_eigh_linear_dependent_basis(use_blas, rng):
    # An overlap matrix with a (numerically) zero eigenvalue: pyscf projects it out.
    n = 8
    v = rng.standard_normal((n, n))
    s = v @ np.diag([1.0, 0.9, 0.8, 0.5, 0.3, 0.1, 1e-3, 1e-9]) @ v.T
    s = (s + s.T) / 2
    x = pyscf_hf.check_linear_dependency(s)
    assert x.shape[1] == n - 1
    h = _sym(rng, n)
    w, c = kernels.eigh(h, x=x)
    assert c.shape == (n, n - 1)
    assert np.allclose(c.T @ s @ c, np.eye(n - 1), atol=1e-8)


def test_make_rdm1(use_blas, rng):
    nao, nmo = 9, 7
    c = rng.standard_normal((nao, nmo))
    occ = np.array([2, 2, 0, 2, 0, 0, 2.0])
    dm = kernels.make_rdm1(c, occ)
    assert np.allclose(dm, pyscf_hf.make_rdm1(c, occ), atol=1e-12)
    assert np.allclose(kernels.make_rdm1(c, np.zeros(nmo)), 0.0)


def test_trace_prod_and_energy(use_blas, rng):
    n = 11
    h = rng.standard_normal((n, n))
    d = rng.standard_normal((n, n))
    v = rng.standard_normal((n, n))
    assert np.isclose(kernels.trace_prod(h, d), np.einsum("ij,ji->", h, d), atol=1e-12)
    e_tot, e2 = kernels.energy_elec(h, v, d)
    e1_ref = np.einsum("ij,ji->", h, d)
    e2_ref = 0.5 * np.einsum("ij,ji->", v, d)
    assert np.isclose(e2, e2_ref, atol=1e-12)
    assert np.isclose(e_tot, e1_ref + e2_ref, atol=1e-12)


def test_trace_prod_large_vector(use_blas, rng):
    # Long enough to take the parallel reduction path.
    n = 300
    h = rng.standard_normal((n, n))
    d = rng.standard_normal((n, n))
    assert np.isclose(kernels.trace_prod(h, d), np.einsum("ij,ji->", h, d), rtol=1e-11, atol=1e-9)


def test_get_occ(h2o):
    mf = scf.RHF(h2o)
    e = np.array([-10.0, -1.0, 1.0, -2.0, 0.0, -3.0, 0.0])
    occ, gap = kernels.get_occ(e, h2o.nelectron // 2)
    assert np.array_equal(occ, mf.get_occ(e))
    assert gap is not None
    homo, lumo = gap
    # 5 doubly occupied orbitals: -10, -3, -2, -1, 0 -> HOMO 0.0, LUMO 0.0 (degenerate)
    assert homo == 0.0 and lumo == 0.0
    occ_hf, gap_hf = kernels.get_occ(e, 4)
    assert np.array_equal(occ_hf, [2, 2, 0, 2, 0, 2, 0]) and gap_hf == (-1.0, 0.0)
    # Degenerate energies: the stable sort fills the lower index first, like numpy.
    e2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    occ2, _ = kernels.get_occ(e2, 3)
    assert np.array_equal(occ2, [2, 2, 2, 0, 0, 0, 0])
    occ3, gap3 = kernels.get_occ(e, 7)
    assert gap3 is None and np.all(occ3 == 2)


def test_get_grad(use_blas, rng):
    nao = 10
    c = rng.standard_normal((nao, nao))
    occ = np.array([2, 2, 2, 0, 0, 0, 2, 0, 0, 0.0])
    f = _sym(rng, nao)
    g = kernels.get_grad(c, occ, f)
    assert np.allclose(g, pyscf_hf.get_grad(c, occ, f), atol=1e-12)
    assert kernels.get_grad(c, np.full(nao, 2.0), f).size == 0


def test_damping_and_level_shift(use_blas, rng):
    n = 7
    f = _sym(rng, n)
    fp = _sym(rng, n)
    s = _spd(rng, n)
    d = _sym(rng, n)
    assert np.allclose(kernels.damping(f, fp, 0.3), pyscf_hf.damping(f, fp, 0.3), atol=1e-13)
    assert np.allclose(kernels.level_shift(s, d, f, 0.4), pyscf_hf.level_shift(s, d * 0.5, f, 0.4), atol=1e-11)


def test_diis_errvec(use_blas, rng):
    n = 9
    s = _spd(rng, n)
    d = _sym(rng, n)
    f = _sym(rng, n)
    assert np.allclose(kernels.diis_errvec(s, d, f), pyscf_diis.get_err_vec(s, d, f), atol=1e-11)
    x = pyscf_hf.check_linear_dependency(s)
    assert np.allclose(kernels.diis_errvec(s, d, f, x), pyscf_diis.get_err_vec(s, d, f, x), atol=1e-11)


@pytest.mark.parametrize("space", [2, 3, 6])
def test_diis_update_matches_pyscf(use_blas, rng, space):
    n = 6
    s = _spd(rng, n)
    ref = lib.diis.DIIS()
    ref.space = space
    mine = mojoscf.CDIIS()
    mine.space = space
    for step in range(12):
        f = _sym(rng, n)
        d = _sym(rng, n)
        err = pyscf_diis.get_err_vec(s, d, f)
        out_ref = ref.update(f, xerr=err)
        out = mine.update(s, d, f)
        assert mine.get_num_vec() == ref.get_num_vec()
        assert np.allclose(out, out_ref, atol=1e-9), f"step {step}"


def test_diis_update_singular_history(use_blas, rng):
    # Feeding identical error vectors makes the DIIS matrix singular: pyscf
    # switches to the pseudo-inverse; so do we.
    n = 4
    s = np.eye(n)
    ref = lib.diis.DIIS()
    mine = mojoscf.CDIIS()
    f = _sym(rng, n)
    d = _sym(rng, n)
    err = pyscf_diis.get_err_vec(s, d, f)
    for _ in range(3):
        out_ref = ref.update(f, xerr=err)
        out = mine.update(s, d, f)
    assert np.allclose(out, out_ref, atol=1e-9)


def test_diis_damp_and_corth(use_blas, rng):
    n = 6
    s = _spd(rng, n)
    x = pyscf_hf.check_linear_dependency(s)
    ref = pyscf_diis.CDIIS(Corth=x)
    ref.damp = 0.2
    mine = mojoscf.CDIIS(Corth=x)
    mine.damp = 0.2
    f_prev = None
    for _ in range(6):
        f = _sym(rng, n)
        d = _sym(rng, n)
        out_ref = ref.update(s, d, f, f_prev=f_prev)
        out = mine.update(s, d, f, f_prev=f_prev)
        assert np.allclose(out, out_ref, atol=1e-9)
        f_prev = f


def test_norm_diff(rng):
    a = rng.standard_normal((5, 5))
    b = rng.standard_normal((5, 5))
    assert np.isclose(kernels.norm_diff(a, b), np.linalg.norm(a - b), atol=1e-12)


def test_jk_dense(h2o):
    eri = h2o.intor("int2e")
    nao = h2o.nao_nr()
    rng = np.random.default_rng(7)
    dm = rng.standard_normal((nao, nao))
    dm = dm + dm.T
    vj, vk = kernels.jk_dense(eri, dm)
    vj_ref, vk_ref = pyscf_hf.dot_eri_dm(eri, dm, hermi=1)
    assert np.allclose(vj, vj_ref, atol=1e-10)
    assert np.allclose(vk, vk_ref, atol=1e-10)
