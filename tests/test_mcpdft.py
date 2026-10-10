"""MC-PDFT (pyscf.mcpdft) with the Mojo XC kernels (mojoscf.mcpdft) against pyscf."""
import numpy as np
import pytest
from pyscf import dft, gto, mcpdft, mcscf, scf
from pyscf.fci import direct_spin1
from pyscf.mcpdft import _dms, otfnal, otpd

import mojoscf
from mojoscf import mcpdft as mpdft


def _orig_energy_ot():
    fn = otfnal.otfnal.energy_ot
    return getattr(fn, "_mojoscf_orig", fn)


@pytest.fixture(scope="module")
def o2():
    """O2 triplet / cc-pVDZ, accelerated density-fitted ROHF, SA(3)-CASSCF(8,6)."""
    mol = gto.M(atom="O 0 0 0; O 0 0 1.21", basis="cc-pvdz", spin=2, verbose=0)
    mf = mojoscf.accelerate(scf.ROHF(mol).density_fit()).run(conv_tol=1e-11)
    mc = mcscf.CASSCF(mf, 6, 8).state_average_([1 / 3] * 3).run(conv_tol=1e-10)
    return mf, mc


@pytest.fixture(scope="module")
def o2_pdft(o2):
    """tPBE MC-PDFT on the SA(3)-CASSCF of the o2 fixture (grids level 2)."""
    mf, mc0 = o2
    mc = mcpdft.CASSCF(mf, "tPBE", 6, 8, grids_level=2).state_average_([1 / 3] * 3)
    mc.conv_tol = 1e-10
    mc.kernel(mc0.mo_coeff, ci0=mc0.ci)
    return mc


def test_installed_on_import():
    """Importing pyscf.mcpdft and pyscf.grad.mcpdft after mojoscf (conftest imports mojoscf first)
    installs the Mojo on-top energy, effective potentials and gradient terms."""
    from pyscf.grad import mcpdft as mcpdft_grad
    from pyscf.mcpdft import pdft_veff

    for fn in (otfnal.otfnal.energy_ot, pdft_veff.kernel, mcpdft_grad.mcpdft_HellmanFeynman_grad):
        assert getattr(fn, "_mojoscf_orig", None) is not None


@pytest.mark.parametrize("deriv", [0, 1])
def test_ontop_pair_density_matches_pyscf(o2, deriv):
    """The cumulant part of Pi (and its gradient) equals pyscf's get_ontop_pair_density minus rho_a rho_b."""
    mf, mc = o2
    mol = mf.mol
    ncore, ncas = mc.ncore, mc.ncas
    casdm1s = np.asarray(direct_spin1.make_rdm1s(mc.ci[1], ncas, mc.nelecas))
    casdm2 = direct_spin1.make_rdm12(mc.ci[1], ncas, mc.nelecas)[1]
    cascm2 = _dms.dm2_cumulant(casdm2, casdm1s)
    mo_cas = mc.mo_coeff[:, ncore:ncore + ncas]
    grids = dft.gen_grid.Grids(mol).build()
    coords = grids.coords[::7]
    ot = mcpdft.otfnal.get_transfnal(mol, "tPBE")
    ao = mol.eval_gto("GTOval_sph_deriv1", coords)
    rho = np.zeros((2, 4, coords.shape[0]))
    ref = otpd.get_ontop_pair_density(ot, rho, ao, cascm2, mo_cas, deriv)
    pi = mpdft.ontop_pair_density(mol, coords, mo_cas, cascm2, deriv)
    assert pi.shape == ref.shape
    assert abs(pi - ref).max() < 1e-11 * max(1.0, abs(ref).max())


@pytest.mark.parametrize("otxc", ["tPBE", "ftPBE", "tBLYP", "tPBE0", "tM06L"])
def test_energy_ot_matches_pyscf(o2, otxc):
    """The on-top energy of each state equals pyscf's, translated and fully translated GGAs, a
    hybrid and a meta-GGA."""
    mf, mc = o2
    ot = mcpdft.otfnal.get_transfnal(mf.mol, otxc)
    ot.grids.level = 2
    for state in range(3):
        casdm1s = np.asarray(direct_spin1.make_rdm1s(mc.ci[state], mc.ncas, mc.nelecas))
        casdm2 = direct_spin1.make_rdm12(mc.ci[state], mc.ncas, mc.nelecas)[1]
        e = ot.energy_ot(casdm1s, casdm2, mc.mo_coeff, mc.ncore, max_memory=2000)
        e0 = _orig_energy_ot()(ot, casdm1s, casdm2, mc.mo_coeff, mc.ncore, max_memory=2000)
        assert abs(e - e0) < 1e-9, (otxc, state, e, e0)


def test_mcpdft_energies(o2_pdft):
    """mcpdft.CASSCF on the accelerated reference: the same MC-PDFT state energies as pyscf's on-top
    code on the same CASSCF solution."""
    mc = o2_pdft
    mc.compute_pdft_energy_()
    e = np.asarray(mc.e_states)
    with pytest.MonkeyPatch.context() as m:
        m.setattr(otfnal.otfnal, "energy_ot", _orig_energy_ot())
        mc.compute_pdft_energy_()
        e0 = np.asarray(mc.e_states)
    assert abs(e - e0).max() < 1e-9


@pytest.mark.parametrize("mode", ["paaa_only", "aaaa_only"])
def test_pdft_veff_matches_pyscf(o2_pdft, mode):
    """The on-top energy and effective potentials (veff1, vhf_c, papa/ppaa) equal pyscf's pdft_veff.kernel."""
    from pyscf.mcpdft import pdft_veff

    mc = o2_pdft
    casdm1s = mc.make_one_casdm1s(mc.ci, state=1)
    casdm2 = mc.make_one_casdm2(mc.ci, state=1)
    dm1s = _dms.casdm1s_to_dm1s(mc, casdm1s)
    cascm2 = _dms.dm2_cumulant(casdm2, casdm1s)
    args = (mc.otfnal, dm1s, cascm2, mc.mo_coeff, mc.ncore, mc.ncas)
    kw = {mode: True}
    e, v1, v2 = pdft_veff.kernel(*args, **kw)
    e0, v10, v20 = pdft_veff.kernel._mojoscf_orig(*args, **kw)
    assert abs(e - e0) < 1e-9
    assert abs(v1 - v10).max() < 1e-9
    for name in ("vhf_c", "papa", "ppaa", "j_pc", "k_pc"):
        a, b = getattr(v2, name), getattr(v20, name)
        assert a.shape == b.shape and abs(a - b).max() < 1e-9, name


def test_hellmann_feynman_grad_matches_pyscf(o2_pdft):
    """The Hellmann-Feynman part of the MC-PDFT gradient (orbital, grid-point and weight terms of the
    on-top energy included) equals pyscf's for the same effective potentials."""
    from pyscf.grad import mcpdft as mcpdft_grad

    mc = o2_pdft
    g = mc.nuc_grad_method()
    veff1, veff2 = mc.get_pdft_veff(mc.mo_coeff, mc.ci, incl_coul=True, paaa_only=True, state=0, drop_mcwfn=True)
    fcasscf = g.make_fcasscf(0)
    fcasscf.mo_coeff, fcasscf.ci = mc.mo_coeff, mc.ci[0]
    mf_grad = mc.get_rhf_base().nuc_grad_method()
    kw = dict(mo_coeff=mc.mo_coeff, ci=mc.ci[0], mf_grad=mf_grad, auxbasis_response=True)
    de = mcpdft_grad.mcpdft_HellmanFeynman_grad(fcasscf, mc.otfnal, veff1, veff2, **kw)
    de0 = mcpdft_grad.mcpdft_HellmanFeynman_grad._mojoscf_orig(fcasscf, mc.otfnal, veff1, veff2, **kw)
    assert abs(de - de0).max() < 1e-9


def test_mcpdft_gradient_matches_pyscf(o2_pdft):
    """The SA-MC-PDFT nuclear gradient of two states equals pyscf's own MC-PDFT code."""
    from pyscf.grad import mcpdft as mcpdft_grad
    from pyscf.mcpdft import pdft_veff

    mc = o2_pdft
    de = [mc.nuc_grad_method().kernel(state=s) for s in (0, 2)]
    with pytest.MonkeyPatch.context() as m:
        m.setattr(otfnal.otfnal, "energy_ot", _orig_energy_ot())
        m.setattr(pdft_veff, "kernel", pdft_veff.kernel._mojoscf_orig)
        m.setattr(mcpdft_grad, "mcpdft_HellmanFeynman_grad", mcpdft_grad.mcpdft_HellmanFeynman_grad._mojoscf_orig)
        de0 = [mc.nuc_grad_method().kernel(state=s) for s in (0, 2)]
    for a, b in zip(de, de0):
        assert abs(a - b).max() < 1e-8
