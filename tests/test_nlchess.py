"""The VV10 Hessian pair sums (mojoscf.nlchess) against pyscf's C kernels, and VV10 Hessians against pyscf."""
import ctypes

import numpy as np
import pytest
from pyscf import dft, gto, lib
from pyscf.dft import numint
from pyscf.hessian import rks as rks_hess

import mojoscf  # noqa: F401  (installs the replacements)
from mojoscf import nlchess

WATER = "O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587"
LIBDFT = lib.load_library("libdft")      # pyscf's C functions themselves


def _p(a):
    return a.ctypes.data_as(ctypes.c_void_p)


@pytest.fixture(scope="module")
def grid():
    """The thresholded NLC grid of water with pyscf's per-point VV10 quantities (B = 6, C = 0.01)."""
    mol = gto.M(atom=WATER, basis="def2-svp", verbose=0)
    dm = dft.RKS(mol, xc="pbe").run().make_rdm1()
    g = dft.gen_grid.Grids(mol)
    g.level = 0
    g.build()
    rho = numint.eval_rho(mol, numint.eval_ao(mol, g.coords, deriv=1), dm, xctype="GGA")
    keep = rho[0] >= 1e-8
    r = np.ascontiguousarray(rho[0, keep])
    gamma = (rho[1:4, keep] ** 2).sum(axis=0)
    kp = 6.0 * 1.5 * np.pi * (9 * np.pi) ** (-1 / 6)
    omega = np.sqrt(0.01 * gamma ** 2 / r ** 4 + 4 / 3 * np.pi * r)
    kappa = kp * r ** (1 / 6)
    coords = np.ascontiguousarray(g.coords[keep])
    weights = np.ascontiguousarray(g.weights[keep])
    atom = np.ascontiguousarray(g.atm_idx[keep], dtype=np.int32)
    return mol, coords, weights, r, np.ascontiguousarray(gamma), omega, kappa, atom


def _close(x, y, tol=1e-12):
    x, y = np.asarray(x), np.asarray(y)
    assert x.shape == y.shape
    assert abs(x - y).max() <= tol * max(1.0, abs(y).max()), abs(x - y).max() / max(1.0, abs(y).max())


def test_installed():
    assert isinstance(rks_hess.libdft, nlchess._Libdft)
    assert rks_hess.libdft.VXC_vv10nlc_hessian_eval_omega_derivative is LIBDFT.VXC_vv10nlc_hessian_eval_omega_derivative


def test_uwabce(grid):
    _, coords, weights, rho, _, omega, kappa, _ = grid
    n = rho.size
    ref = [np.empty(n) for _ in range(6)]
    LIBDFT.VXC_vv10nlc_hessian_eval_UWABCE(*map(_p, ref), _p(coords), _p(weights), _p(rho), _p(omega), _p(kappa),
                                           ctypes.c_int(n))
    for x, y in zip(nlchess.uwabce(coords, weights, rho, omega, kappa), ref):
        _close(x, y)


@pytest.mark.parametrize("nt", [1, 7])
def test_f_t(grid, nt):
    _, coords, weights, rho, gamma, omega, kappa, _ = grid
    n = rho.size
    rng = np.random.default_rng(nt)
    u, wq, a, b, c, _ = nlchess.uwabce(coords, weights, rho, omega, kappa)
    der = [np.empty(n) for _ in range(5)]
    LIBDFT.VXC_vv10nlc_hessian_eval_omega_derivative(*map(_p, der), _p(rho), _p(gamma), ctypes.c_double(0.01),
                                                     ctypes.c_int(n))
    kp = 6.0 * 1.5 * np.pi * (9 * np.pi) ** (-1 / 6)
    dk, d2k = kp / 6 * rho ** (-5 / 6), kp * (-5 / 36) * rho ** (-11 / 6)
    per_point = [weights, rho, omega, kappa, u, wq, a, b, c, der[0], der[1], dk, der[2], der[3], der[4], d2k]
    per_point = [np.ascontiguousarray(x) for x in per_point]
    rho_t = rng.standard_normal((nt, n)) * rho
    gamma_t = rng.standard_normal((nt, n)) * gamma
    ref = np.empty((nt, n)), np.empty((nt, n))
    LIBDFT.VXC_vv10nlc_hessian_eval_f_t(_p(ref[0]), _p(ref[1]), _p(coords), *map(_p, per_point), _p(rho_t),
                                        _p(gamma_t), ctypes.c_int(n), ctypes.c_int(nt))
    got = nlchess.f_t(coords, *per_point, rho_t, gamma_t)
    _close(got[0], ref[0])
    _close(got[1], ref[1])


def test_euw_with_weight1(grid):
    _, coords, _, rho, _, omega, kappa, _ = grid
    n = rho.size
    nd = 9
    weight1 = np.random.default_rng(3).standard_normal((nd, n))
    ref = [np.empty((nd, n)) for _ in range(3)]
    LIBDFT.VXC_vv10nlc_hessian_eval_EUW_with_weight1(*map(_p, ref), _p(coords), _p(weight1), _p(rho), _p(omega),
                                                     _p(kappa), ctypes.c_int(n), ctypes.c_int(nd))
    for x, y in zip(nlchess.euw_with_weight1(coords, weight1, rho, omega, kappa), ref):
        _close(x, y)


def test_euw_grid_response(grid):
    mol, coords, weights, rho, _, omega, kappa, atom = grid
    n = rho.size
    atom = atom.copy()
    atom[::50] = -1                                     # points without an atom: zero rows, skipped as partners
    ref = [np.empty((mol.natm, 3, n)) for _ in range(3)]
    LIBDFT.VXC_vv10nlc_hessian_eval_EUW_grid_response(*map(_p, ref), _p(coords), _p(weights), _p(rho), _p(omega),
                                                      _p(kappa), _p(atom), ctypes.c_int(n), ctypes.c_int(mol.natm))
    for x, y in zip(nlchess.euw_grid_response(coords, weights, rho, omega, kappa, atom, mol.natm), ref):
        _close(x, y)


def test_rks_vv10_hessian_matches_pyscf(monkeypatch):
    """pyscf's RKS wB97M-V Hessian with the Mojo pair sums equals the one with pyscf's C loops."""
    mol = gto.M(atom=WATER, basis="def2-svp", verbose=0)
    mf = dft.RKS(mol, xc="wb97m-v")
    mf.grids.level = 1
    mf.nlcgrids.level = 0
    mf.run(conv_tol=1e-11)
    h = mf.Hessian().kernel()
    monkeypatch.setattr(rks_hess, "libdft", rks_hess.libdft._mojoscf_orig)
    ref = mf.Hessian().kernel()
    _close(h, ref, 1e-9)


def _vv10_water(df, accel, xc="wb97m-v", nlc=None):
    mol = gto.M(atom=WATER, basis="def2-svp", verbose=0)
    mf = dft.RKS(mol, xc=xc)
    if nlc:
        mf.nlc = nlc
    mf.grids.level = 1
    mf.nlcgrids.level = 0
    if df:
        mf = mf.density_fit()
    if accel:
        mf = mojoscf.dft.accelerate(mf)
    return mf


def test_response_matches_pyscf():
    """nlchess.Response equals pyscf's get_vnlc_resp for symmetric first-order densities."""
    mf = _vv10_water(False, False).run()
    nao = mf.mol.nao_nr()
    d = np.random.default_rng(5).standard_normal((3, nao, nao))
    d = d + d.transpose(0, 2, 1)
    ref = rks_hess.get_vnlc_resp(mf, mf.mol, mf.mo_coeff, mf.mo_occ, d, 4000)
    got = nlchess.Response(mf, mf.mo_coeff, mf.mo_occ, max_set=2)(d)
    _close(got, ref, 1e-11)


@pytest.mark.parametrize("df", [True, False])
def test_cphf_operator_with_vv10(df):
    """The MO-basis coupled-perturbed operator of a VV10 functional against pyscf's gen_vind."""
    from pyscf.hessian import rhf as rhf_hess

    from mojoscf import hessian as mhess

    mf = _vv10_water(df, True).run()
    fx = mhess.cphf_operator(mf)
    assert fx is not None
    ref = rhf_hess.gen_vind(mf, mf.mo_coeff, mf.mo_occ)
    n = mf.mo_coeff.shape[1] * int((mf.mo_occ > 0).sum())
    x = np.random.default_rng(4).normal(size=(4, n))
    a, b = fx(x), ref(x)
    _close(a, np.asarray(b).reshape(a.shape), 1e-11)


@pytest.mark.parametrize("xc, nlc, df, tol", [("wb97m-v", None, False, 1e-10), ("b97m-v", None, True, 1e-9),
                                               ("b3lyp", "vv10", True, 1e-9), ("wb97m-v", None, True, 3e-8)])
def test_accelerated_vv10_hessian_matches_pyscf(xc, nlc, df, tol):
    """The accelerated RKS Hessian of VV10 functionals (native J/K, XC, CPHF and VV10 response; pyscf's VV10
    energy and Fock derivatives with the Mojo pair sums) equals pyscf's for the same SCF solution.  With
    density fitting the long-range metric of erf(omega r) / r is nearly singular: wB97M-V agrees to ~1e-8
    (7e-9 without its VV10 term), the functionals without range separation to ~1e-11."""
    from mojoscf import hessian as mhess

    mf = _vv10_water(df, True, xc, nlc)
    mf.conv_tol = 1e-11
    mf.kernel()
    ref = _vv10_water(df, False, xc, nlc)
    for key in ("mo_coeff", "mo_occ", "mo_energy", "e_tot", "converged"):
        setattr(ref, key, getattr(mf, key))
    h = mf.Hessian()
    assert isinstance(h, mhess._MojoHessMixin) and mf.do_nlc()
    assert (mhess._df_jk_reason(h, mf.mo_coeff, mf.mo_occ) is None) == df
    assert (mhess._exact_jk_reason(h, mf.mo_coeff, mf.mo_occ) is None) == (not df)
    _close(h.kernel(), ref.Hessian().kernel(), tol)


def test_uks_vv10_hessian_raises_as_pyscf():
    """pyscf has no UKS Hessian for NLC functionals; the accelerated object does not run one either."""
    from mojoscf import hessian as mhess

    mol = gto.M(atom=WATER, basis="def2-svp", charge=1, spin=1, verbose=0)
    mf = dft.UKS(mol, xc="wb97m-v")
    mf.grids.level = 0
    mf.nlcgrids.level = 0
    mf = mojoscf.dft.accelerate(mf.density_fit()).run()
    assert mhess.cphf_operator(mf) is None
    with pytest.raises(NotImplementedError):
        mf.Hessian().kernel()
