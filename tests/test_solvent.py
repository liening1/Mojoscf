"""PCM-family and SMD solvation with the Mojo kernels (mojoscf.solvent) against pyscf."""
import numpy as np
import pytest
from pyscf import dft, gto, scf
from pyscf.solvent import pcm
from pyscf.solvent.grad import pcm as pcm_grad
from pyscf.solvent.grad import smd as smd_grad

import mojoscf
from mojoscf import integrals, solvent

WATER = "O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587"
FECO2 = "Fe 0 0 0; C 1.9 0 0; O 3.05 0 0; C -1.9 0 0; O -3.05 0 0"


def test_potential_at_points_matches_pyscf():
    from pyscf import df

    mol = gto.M(atom=FECO2, basis="def2-svp", verbose=0)
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(700, 3)) * 4
    zeta = rng.uniform(0.5, 20, 700)
    dm = scf.hf.init_guess_by_minao(mol)
    dms = np.array([dm, 0.3 * dm + 0.1 * np.eye(len(dm))])
    v = df.incore.aux_e2(mol, gto.fakemol_for_charges(pts, expnt=zeta), intor="int3c2e", aosym="s1")
    assert abs(integrals.int1e_grids_dm(mol, dms, pts, zeta) - np.einsum("ijL,sij->sL", v, dms)).max() < 1e-11
    ref = np.einsum("kij,ij->k", mol.intor("int1e_grids", grids=pts), dm)
    assert abs(integrals.int1e_grids_dm(mol, dm, pts) - ref).max() < 1e-11


def _make(mol, model, xc="pbe", spin=0):
    mf = (dft.UKS if spin else dft.RKS)(mol, xc=xc).density_fit()
    if model == "SMD":
        mf = mf.SMD()
        mf.with_solvent.solvent = "water"
    else:
        mf = mf.PCM()
        mf.with_solvent.method = model
    mf.conv_tol = 1e-11
    return mf


@pytest.mark.parametrize("model", ["IEF-PCM", "C-PCM", "COSMO", "SS(V)PE", "SMD"])
def test_models_match_pyscf(model):
    mol = gto.M(atom=WATER, basis="def2-svp", verbose=0)
    ref = _make(mol, model).run()
    mf = mojoscf.dft.accelerate(_make(mol, model))
    assert isinstance(mf.with_solvent, solvent._MojoPCMMixin)
    mf.run()
    assert abs(mf.e_tot - ref.e_tot) < 1e-9
    g1 = mf.nuc_grad_method().kernel()
    for k in ("mo_coeff", "mo_occ", "mo_energy", "e_tot", "converged"):
        setattr(ref, k, getattr(mf, k))
    assert abs(g1 - ref.nuc_grad_method().kernel()).max() < 1e-10


@pytest.mark.parametrize("model", ["IEF-PCM", "C-PCM", "SS(V)PE", "SMD"])
def test_terms_match_pyscf(model):
    """S/D matrices, solver and integral gradient terms against pyscf's functions on the same state."""
    mol = gto.M(atom=FECO2, basis="def2-svp", verbose=0)
    mf = mojoscf.dft.accelerate(_make(mol, model)).run(conv_tol=1e-9)
    s = mf.with_solvent
    D0, S0 = pcm.get_D_S(s.surface, with_S=True, with_D=True)
    D1, S1 = solvent.get_D_S(s.surface, with_S=True, with_D=True)
    assert abs(S0 - S1).max() < 1e-13 and abs(D0 - D1).max() < 1e-12
    dm = mf.make_rdm1()
    mod = smd_grad if model == "SMD" else pcm_grad
    g0 = mod.grad_solver._mojoscf_orig(s, dm)
    assert abs(mod.grad_solver(s, dm) - g0).max() < 1e-11 * max(1.0, abs(g0).max())
    q0 = pcm_grad.grad_qv._mojoscf_orig(s, dm)
    assert abs(pcm_grad.grad_qv(s, dm) - q0).max() < 1e-10 * max(1.0, abs(q0).max())
    v = mf.make_rdm1()[None]
    assert abs(pcm.PCM._get_v(s, v) - s._get_v(v)).max() < 1e-11
    q = np.random.default_rng(1).normal(size=(1, S0.shape[0]))
    assert abs(pcm.PCM._get_vmat(s, q) - s._get_vmat(q)).max() < 1e-11


def test_uks_and_hf_attach():
    mol = gto.M(atom=WATER, basis="def2-svp", charge=1, spin=1, verbose=0)
    ref = _make(mol, "IEF-PCM", spin=1).run()
    mf = mojoscf.dft.accelerate(_make(mol, "IEF-PCM", spin=1)).run()
    assert abs(mf.e_tot - ref.e_tot) < 1e-9
    assert abs(mf.nuc_grad_method().kernel() - ref.nuc_grad_method().kernel()).max() < 1e-7
    # Hartree-Fock: the native loop does not run solvent objects; accelerate keeps pyscf's loop with the Mojo
    # J/K and solvent kernels, attach gives the solvent alone the kernels
    mol = gto.M(atom=WATER, basis="def2-svp", verbose=0)
    ref = scf.RHF(mol).PCM().run(conv_tol=1e-11)
    g0 = ref.nuc_grad_method().kernel()
    mf = mojoscf.accelerate(scf.RHF(mol).PCM())
    assert isinstance(mf, mojoscf.scf._MojoHFHook) and isinstance(mf.with_solvent, solvent._MojoPCMMixin)
    for mf in (mf, solvent.attach(scf.RHF(mol).PCM())):
        mf.run(conv_tol=1e-11)
        assert abs(mf.e_tot - ref.e_tot) < 1e-9
        assert abs(mf.nuc_grad_method().kernel() - g0).max() < 1e-8


def test_unattached_solvent_keeps_pyscf():
    mol = gto.M(atom=WATER, basis="sto-3g", verbose=0)
    mojoscf.dft.accelerate(_make(mol, "C-PCM")).run()          # installs the dispatching hooks
    mf = _make(mol, "C-PCM").run()
    assert not isinstance(mf.with_solvent, solvent._MojoPCMMixin)
    g = mf.nuc_grad_method().kernel()
    assert np.isfinite(g).all()
    assert solvent.attach(scf.RHF(mol)) is not None             # no solvent: unchanged


def _make_hess(mol, model, spin, df, accel, xc="pbe"):
    mf = (dft.UKS if spin else dft.RKS)(mol, xc=xc)
    mf.grids.level = 1
    if df:
        mf = mf.density_fit()
    if model == "SMD":
        mf = mf.SMD()
        mf.with_solvent.solvent = "water"
    else:
        mf = mf.PCM()
        mf.with_solvent.method = model
    mf.conv_tol = 1e-11
    return mojoscf.dft.accelerate(mf) if accel else mf


@pytest.mark.parametrize("spin, df", [(0, True), (1, True), (0, False)])
def test_cphf_operator_with_solvent(spin, df):
    """The MO-basis coupled-perturbed operator of a solvent-attached SCF (equilibrium solvation: the
    solvent's response to the first-order density added) against pyscf's gen_vind."""
    from pyscf.hessian import rhf as rhf_hess
    from pyscf.hessian import uhf as uhf_hess
    from pyscf import lib

    from mojoscf import hessian as mhess

    mol = gto.M(atom=WATER, basis="def2-svp", charge=spin, spin=spin, verbose=0)
    mf = _make_hess(mol, "C-PCM", spin, df, True).run()
    fx = mhess.cphf_operator(mf)
    assert fx is not None
    ref = (uhf_hess if spin else rhf_hess).gen_vind(mf, mf.mo_coeff, mf.mo_occ)
    cs = mf.mo_coeff if spin else [mf.mo_coeff]
    os_ = mf.mo_occ if spin else [mf.mo_occ]
    n = sum(c.shape[1] * int((o > 0).sum()) for c, o in zip(cs, os_))
    x = np.random.default_rng(6).normal(size=(4, n))
    for equilibrium in (True, False):
        with lib.temporary_env(mf.with_solvent, equilibrium_solvation=equilibrium):
            a, b = fx(x), np.asarray(ref(x)).reshape(fx(x).shape)
        assert abs(a - b).max() < 1e-11 * abs(b).max()


@pytest.mark.parametrize("model, spin, df", [("C-PCM", 0, True), ("IEF-PCM", 1, True), ("C-PCM", 0, False),
                                             ("SMD", 0, True)])
def test_solvent_hessian_matches_pyscf(model, spin, df):
    """Solvent Hessians on the native J/K, XC and MO-basis CPHF paths (with the solvent terms: the
    first-order Fock term of make_h1, the CPHF response and the solvent Hessian, their integrals from the
    Mojo kernels) equal pyscf's for the same SCF solution."""
    from mojoscf import hessian as mhess

    mol = gto.M(atom=WATER, basis="def2-svp", charge=spin, spin=spin, verbose=0)
    mf = _make_hess(mol, model, spin, df, True)
    mf.kernel()
    ref = _make_hess(mol, model, spin, df, False)
    for k in ("mo_coeff", "mo_occ", "mo_energy", "e_tot", "converged"):
        setattr(ref, k, getattr(mf, k))
    ref.with_solvent.build()
    h = mf.Hessian()
    assert isinstance(h, mhess._MojoHessMixin)
    assert (mhess._df_jk_reason(h, mf.mo_coeff, mf.mo_occ) is None) == df
    assert (mhess._exact_jk_reason(h, mf.mo_coeff, mf.mo_occ) is None) == (not df)
    a = h.kernel()
    b = ref.Hessian().kernel()
    assert abs(a - b).max() < 1e-8, abs(a - b).max()


@pytest.mark.parametrize("cart", [False, True])
def test_hessian_kernels_match_pyscf_integrals(cart):
    """The multi-set potential, the per-atom potential derivatives and the charge Hessian kernel against
    pyscf's int3c2e, int3c2e_ip1/ip2 and int3c2e_ipip1/ipvip1/ip1ip2/ipip2 with Gaussian surface charges."""
    from pyscf import df

    mol = gto.M(atom=WATER + "; Cl 2.5 0.3 -0.4", basis="def2-svp", charge=-1, cart=cart, verbose=0)
    s = dft.RKS(mol).PCM().with_solvent
    s.build()
    coords, zeta = s.surface["grid_coords"], s.surface["charge_exp"] ** 2
    gslice = s.surface["gslice_by_atom"]
    ng, nao, natm = coords.shape[0], mol.nao_nr(), mol.natm
    catom = np.concatenate([np.full(p1 - p0, a) for a, (p0, p1) in enumerate(gslice)])
    rng = np.random.default_rng(1)
    fake = gto.fakemol_for_charges(coords, expnt=zeta)
    fake.cart = cart

    def aux(intor):
        return df.incore.aux_e2(mol, fake, intor=intor, aosym="s1")

    def close(a, b, tol=1e-12):
        assert abs(a - b).max() <= tol * abs(b).max(), abs(a - b).max() / abs(b).max()

    w = rng.normal(size=(4, ng))
    w[3] = 0.0
    w[3, gslice[1][0]: gslice[1][1]] = 1.0                   # one atom's points only
    ip1 = aux("int3c2e_ip1")
    close(integrals.int1e_grids_sum_sets(mol, coords, w, zeta), np.einsum("ijk,sk->sij", aux("int3c2e"), w))
    close(integrals.int1e_grids_sum_sets(mol, coords, w, zeta, deriv=True), np.einsum("xijk,sk->sxij", ip1, w))
    dm = rng.normal(size=(nao, nao))
    dm = dm + dm.T
    g = integrals.int1e_grids_ip_dm_atoms(mol, dm, coords, zeta)
    v = np.einsum("xijk,ij->xik", ip1, 2 * dm)
    close(g, np.array([v[:, p0:p1].sum(axis=1) for p0, p1 in mol.aoslice_by_atom()[:, 2:]]))
    close(-g.sum(axis=0), np.einsum("xijk,ij->xk", aux("int3c2e_ip2"), dm))
    # pyscf's d2e_from_d2I of analytical_hess_qv: AO-AO, AO-charge and charge-charge second derivatives
    q = rng.normal(size=ng)
    sl = [slice(p0, p1) for p0, p1 in mol.aoslice_by_atom()[:, 2:]]
    ref = np.zeros((natm, natm, 3, 3))
    ipip1 = np.einsum("dijq,q->dij", aux("int3c2e_ipip1"), q).reshape(3, 3, nao, nao)
    ipvip1 = np.einsum("dijq,q->dij", aux("int3c2e_ipvip1"), q).reshape(3, 3, nao, nao)
    ip1ip2 = aux("int3c2e_ip1ip2").reshape(3, 3, nao, nao, ng)
    for a in range(natm):
        ref[a, a] += 2 * np.einsum("ij,dDij->dD", dm[sl[a]], ipip1[:, :, sl[a]])
        for b in range(natm):
            ref[a, b] += 2 * np.einsum("ij,dDij->dD", dm[sl[a], sl[b]], ipvip1[:, :, sl[a], sl[b]])
            g0, g1 = gslice[b]
            ac = 2 * np.einsum("ij,dDijq,q->dD", dm[sl[a]], ip1ip2[:, :, sl[a], :, g0:g1], q[g0:g1])
            ref[a, b] += ac
            ref[b, a] += ac.T
    cc = np.einsum("dijq,ij->dq", aux("int3c2e_ipip2"), dm).reshape(3, 3, ng)
    for a, (g0, g1) in enumerate(gslice):
        ref[a, a] += cc[:, :, g0:g1] @ q[g0:g1]
    close(integrals.mm_hess_terms(mol, dm, coords, q, catom, zeta), ref)
    perm = rng.permutation(ng)                                 # vectors of charges of several atoms
    close(integrals.mm_hess_terms(mol, dm, coords[perm], q[perm], catom[perm], zeta[perm]), ref)


@pytest.mark.parametrize("model", ["C-PCM", "IEF-PCM", "SMD"])
def test_hessian_terms_match_pyscf(model):
    """The Mojo versions of pyscf's PCM Hessian functions against the originals on the same state."""
    from pyscf.solvent.hessian import pcm as pcm_hess

    mol = gto.M(atom=WATER + "; Cl 2.5 0.3 -0.4", basis="def2-svp", charge=-1, verbose=0)
    mf = mojoscf.dft.accelerate(_make(mol, model)).run(conv_tol=1e-9)
    s = mf.with_solvent
    dm = mf.make_rdm1()
    natm = mol.natm

    def orig(name):
        return getattr(pcm_hess, name)._mojoscf_orig

    def close(a, b, tol=1e-11):
        a, b = np.asarray(a), np.asarray(b)
        assert abs(a - b).max() <= tol * max(1.0, abs(b).max()), abs(a - b).max()

    close(pcm_hess.get_dvgrids(s, dm, range(natm)), orig("get_dvgrids")(s, dm, range(natm)))
    dq = pcm_hess.get_dqsym_dx(s, dm, range(natm))
    assert pcm_hess.get_dqsym_dx(s, dm, range(natm)) is dq      # kept per density
    close(pcm_hess.analytical_grad_vmat(s, dm), orig("analytical_grad_vmat")(s, dm))
    close(pcm_hess.analytical_hess_qv(s, dm), orig("analytical_hess_qv")(s, dm))
    c, o = mf.mo_coeff, mf.mo_occ
    h1 = solvent.grad_vmat_mo(s, dm, [c], [o], max_memory=1)[0]       # one atom per block
    close(h1, np.einsum("pi,axpq,qj->axij", c, orig("analytical_grad_vmat")(s, dm), c[:, o > 0]))
    q = np.random.default_rng(2).normal(size=(5, s.surface["grid_coords"].shape[0]))
    close(s._get_vmat(q), pcm.PCM._get_vmat(s, q))


def test_surface_derivative_kernels_match_pyscf():
    """The per-atom contractions of the S and D derivatives (first and second, S, D and D^T) against
    pyscf's dense (n, n, 3) and (3, 3, n, n) arrays."""
    from pyscf.solvent.grad import pcm as pcm_grad
    from pyscf.solvent.hessian import pcm as pcm_hess

    solvent._install_hess_hooks()
    mol = gto.M(atom=WATER + "; Cl 2.5 0.3 -0.4", basis="def2-svp", charge=-1, verbose=0)
    s = dft.RKS(mol).PCM().with_solvent
    s.build()
    surf = s.surface
    gslice = surf["gslice_by_atom"]
    ng, natm = surf["grid_coords"].shape[0], mol.natm
    rng = np.random.default_rng(0)
    q, v = rng.normal(size=ng), rng.normal(size=ng)
    dF, _ = pcm_grad.get_dF_dA(surf)
    dD, dS, _ = pcm_grad.get_dD_dS(surf, dF, with_D=True, with_S=True)
    for kind, dX in (("S", dS), ("D", dD), ("DT", -dD.transpose(1, 0, 2))):
        ref = np.array([np.einsum("ijx,j->xi", dX[:, p0:p1], q[p0:p1]) for p0, p1 in gslice])
        got = solvent._dx_atoms(surf, kind, q)
        assert abs(got - ref).max() < 1e-12 * abs(ref).max(), kind
    d2D, d2S = pcm_hess.get_d2D_d2S._mojoscf_orig(surf, with_S=True, with_D=True)
    zero = np.zeros((natm, natm, 3, 3, ng))
    for kind, ref in (("S", pcm_hess.get_v_dot_d2S_dot_q._mojoscf_orig(d2S, zero, v, q, natm, gslice)),
                      ("D", pcm_hess.get_v_dot_d2D_dot_q._mojoscf_orig(d2D, v, q, natm, gslice)),
                      ("DT", pcm_hess.get_v_dot_d2DT_dot_q._mojoscf_orig(d2D, v, q, natm, gslice))):
        got = solvent._d2x_atoms(surf, kind, v, q)
        # pyscf's closed forms of the D derivatives lose ~1e-11 to cancellation; the kernel uses Boys functions
        assert abs(got - ref).max() < 1e-10 * abs(ref).max(), kind
    for method in ("SWIG", "ISWIG"):
        s.surface_discretization_method = method
        s.build()
        surf = s.surface
        ng = surf["grid_coords"].shape[0]
        w = rng.normal(size=ng)
        d2F, d2A = pcm_hess.get_d2F_d2A._mojoscf_orig(surf, method)
        assert abs(solvent._d2f_contract(surf, method, w * surf["area"]) - d2A @ w).max() < 1e-12 * abs(d2A @ w).max()
        assert abs(solvent._d2f_contract(surf, method, w * surf["switch_fun"]) - d2F @ w).max() < 1e-12 * abs(d2F @ w).max()


@pytest.mark.parametrize("model, disc", [("C-PCM", "SWIG"), ("IEF-PCM", "SWIG"), ("SS(V)PE", "SWIG"),
                                         ("SMD", "SWIG"), ("IEF-PCM", "ISWIG")])
def test_hessian_solver_term_matches_pyscf(model, disc):
    """pyscf's analytical_hess_solver and get_dqsym_dx with the surface derivatives contracted on the fly
    (no (n, n) derivative arrays, the cached LU of K) against pyscf's dense evaluation."""
    from pyscf.solvent.hessian import pcm as pcm_hess

    mol = gto.M(atom=WATER + "; Cl 2.5 0.3 -0.4", basis="def2-svp", charge=-1, verbose=0)
    mf = _make(mol, model)
    mf.with_solvent.surface_discretization_method = disc
    mf = mojoscf.dft.accelerate(mf).run(conv_tol=1e-9)
    s = mf.with_solvent
    dm = mf.make_rdm1()
    natm = mol.natm

    def close(a, b, tol=1e-11):
        assert abs(a - b).max() <= tol * abs(b).max(), abs(a - b).max() / abs(b).max()

    close(pcm_hess.get_dqsym_dx(s, dm, range(natm)), pcm_hess.get_dqsym_dx._mojoscf_orig(s, dm, range(natm)))
    close(pcm_hess.analytical_hess_solver(s, dm), pcm_hess.analytical_hess_solver._mojoscf_orig(s, dm))
