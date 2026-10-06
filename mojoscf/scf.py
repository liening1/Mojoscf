"""RHF and UHF with the SCF iterations executed in Mojo.

The two-electron part is built natively too whenever the integrals are in core:
density fitting from pyscf's ``(naux, npair)`` tensor, or the 8-fold packed ERIs
pyscf keeps for molecules that fit in memory.  Only direct SCF (integrals
recomputed every cycle by libcint) still goes through ``mf.get_veff``.

Two ways to use it::

    import mojoscf
    mf = mojoscf.RHF(mol).run()          # a pyscf RHF object with a Mojo SCF loop
    mf = mojoscf.UHF(mol).run()          # same for UHF (open shell, broken symmetry)

    mf = pyscf.scf.RHF(mol).density_fit()
    mojoscf.accelerate(mf)               # upgrade an existing RHF/UHF object in place
    mf.kernel()

The driver :func:`kernel` is a port of :func:`pyscf.scf.hf.kernel`: it computes
the same quantities in the same order, so energies, orbitals and iteration
counts agree with pyscf to numerical precision.  Only the two-electron part
(``mf.get_veff``) is still evaluated by pyscf, since that is compiled C code
already.  Whatever the Mojo driver cannot handle (custom DIIS objects, custom
convergence checks, overridden Fock/occupation methods, ...) falls back to
pyscf's own loop, with the small glue functions still replaced by their Mojo
versions.
"""
from __future__ import annotations

import sys

import numpy as np
from pyscf import lib
from pyscf.data import nist
from pyscf.lib import logger
from pyscf.scf import chkfile
from pyscf.scf import diis as pyscf_diis
from pyscf.scf import hf as pyscf_hf
from pyscf.scf import uhf as pyscf_uhf

from . import kernels
from ._backend import blas_args, blas_config, df_block_mb, get_extension
from .diis import CDIIS

__all__ = ["RHF", "UHF", "kernel", "accelerate", "is_supported", "native_veff"]


def _c(a):
    return np.ascontiguousarray(a, dtype=np.float64)


def _is_real(*arrays) -> bool:
    return all(not np.iscomplexobj(a) for a in arrays if a is not None)


def _pair(value):
    """``(alpha, beta)`` from a scalar or a two-element sequence (pyscf convention)."""
    if isinstance(value, (tuple, list, np.ndarray)):
        a, b = value
        return float(a), float(b)
    return float(value), float(value)


# Methods the native loop re-implements instead of calling: if any of them has
# been replaced by something other than pyscf's or mojoscf's version, the native
# loop would silently ignore the replacement, so it must not be used.
_BYPASSED = ("get_fock", "eig", "_eigh", "get_occ", "make_rdm1", "energy_elec", "energy_tot", "get_grad")
_TRUSTED_MODULES = ("pyscf.scf.hf", "pyscf.scf.uhf", "mojoscf.")


def _overridden_glue(mf, include_instance=True):
    """Name of a bypassed method that ``mf`` overrides, or None."""
    if include_instance:
        for name in _BYPASSED:
            if name in vars(mf):
                return name
    for name in _BYPASSED:
        for klass in type(mf).__mro__:
            if name in klass.__dict__:
                if not klass.__module__.startswith(_TRUSTED_MODULES):
                    return name
                break
    return None


def _is_uhf(mf) -> bool:
    return isinstance(mf, pyscf_uhf.UHF)


_STANDARD_JK_MODULES = ("pyscf.scf.hf", "pyscf.scf.uhf", "pyscf.df.df_jk", "mojoscf.")


def _standard_method(mf, name) -> bool:
    """True if ``mf.<name>`` is pyscf's (or mojoscf's) own implementation."""
    if name in vars(mf):
        return False
    for klass in type(mf).__mro__:
        if name in klass.__dict__:
            return klass.__module__.startswith(_STANDARD_JK_MODULES)
    return True


def native_veff(mf):
    """How the Mojo driver can build J and K itself: ``(mode, data, reason)``.

    mode 1: density fitting with pyscf's in-core ``(naux, npair)`` tensor.
    mode 2: in-core 8-fold packed ERIs (built here if pyscf would build them).
    mode 3: integral-direct J/K from the Mojo integral engine (direct SCF).
    mode 0: not possible; ``reason`` says why and ``mf.get_veff`` is called instead.

    Integrals that have to be computed are computed by the Mojo engine
    (``mojoscf.integrals``) when it supports the molecule and is enabled
    (``integrals.engine() == "mojo"``), otherwise by pyscf/libcint.
    """
    from . import integrals

    mol = mf.mol
    nao = mol.nao_nr()
    npair = nao * (nao + 1) // 2
    for name in ("get_veff", "get_jk"):
        if not _standard_method(mf, name):
            return 0, None, f"mf.{name} is not pyscf's implementation"
    with_df = getattr(mf, "with_df", None)
    if with_df is not None:
        from pyscf.df import df as pyscf_df

        if getattr(mf, "only_dfj", False):
            return 0, None, "only_dfj (exact exchange with DF Coulomb)"
        if type(with_df) is not pyscf_df.DF:
            return 0, None, f"{type(with_df).__name__} is not the plain pyscf DF object"
        if with_df._cderi is None and not integrals.build_df(with_df):
            with_df.build()
        cderi = with_df._cderi
        if (
            isinstance(cderi, np.ndarray) and cderi.ndim == 2 and cderi.dtype == np.float64
            and cderi.shape[1] == npair
        ):
            return 1, np.ascontiguousarray(cderi), None
        return 0, None, "DF tensor is not an in-core float64 array"
    eri = getattr(mf, "_eri", None)
    if eri is None:
        if mol.incore_anyway or mf._is_mem_enough():
            # what pyscf.scf.hf.RHF.get_jk does on its first call (with the Mojo engine if possible)
            if integrals.available(mol, two_electron=True):
                eri = mf._eri = integrals.int2e_s8(mol)
            else:
                eri = mf._eri = mol.intor("int2e", aosym="s8")
        elif integrals.available(mol, two_electron=True):
            return 3, (integrals.basis_tables(mol), integrals._boys_table()), None
        else:
            reason = integrals.unsupported_reason(mol, two_electron=True) or "Mojo integral engine disabled"
            return 0, None, f"direct SCF with libcint ({reason})"
    if isinstance(eri, np.ndarray) and eri.dtype == np.float64 and eri.ndim == 1 and eri.size == npair * (npair + 1) // 2:
        return 2, np.ascontiguousarray(eri), None
    return 0, None, "ERI tensor is not the 8-fold packed float64 vector"


def kernel(mf, conv_tol=1e-10, conv_tol_grad=None, dump_chk=True, dm0=None, callback=None, conv_check=True, **kwargs):
    """Mojo SCF driver with the signature and return value of ``pyscf.scf.hf.kernel``.

    Handles closed-shell RHF and UHF objects.  Returns
    ``(scf_conv, e_tot, mo_energy, mo_coeff, mo_occ)``.
    """
    if "init_dm" in kwargs:
        raise RuntimeError('Keyword argument "init_dm" is replaced by "dm0"')
    log = logger.new_logger(mf)
    cput0 = log.init_timer()
    if conv_tol_grad is None:
        conv_tol_grad = np.sqrt(conv_tol)
        log.info("Set gradient conv threshold to %g", conv_tol_grad)

    reason = _fallback_reason(mf)
    if reason is not None:
        log.info("mojoscf: %s; running the pyscf SCF loop with Mojo glue kernels", reason)
        return pyscf_hf.kernel(
            mf, conv_tol, conv_tol_grad, dump_chk=dump_chk, dm0=dm0, callback=callback, conv_check=conv_check, **kwargs
        )

    uhf = _is_uhf(mf)
    mol = mf.mol
    s1e = mf.get_ovlp(mol)
    if dm0 is None:
        dm = mf.get_init_guess(mol, mf.init_guess, s1e=s1e, **kwargs)
    else:
        dm = dm0
    h1e = mf.get_hcore(mol)
    if not _is_real(s1e, h1e, dm) or np.ndim(h1e) != 2:
        log.info("mojoscf: complex or spin-dependent integrals; running the pyscf SCF loop with Mojo glue kernels")
        return pyscf_hf.kernel(
            mf, conv_tol, conv_tol_grad, dump_chk=dump_chk, dm0=dm0, callback=callback, conv_check=conv_check, **kwargs
        )
    # The initial guess usually carries its orbitals (pyscf tags it in make_rdm1);
    # keep them for the first get_veff call, which is by far the most expensive one
    # for density fitting.
    init_tags = (getattr(dm, "mo_coeff", None), getattr(dm, "mo_occ", None))
    if init_tags[0] is not None:
        init_tags = (np.asarray(init_tags[0]), np.asarray(init_tags[1]))
    s1e = _c(s1e)
    h1e = _c(h1e)
    dm = _c(dm)
    if uhf:
        if dm.ndim == 2:  # a closed-shell density given as guess: split it evenly
            dm = np.array((dm * 0.5, dm * 0.5))
            init_tags = (None, None)
        if dm.ndim != 3 or dm.shape[0] != 2:
            raise ValueError("mojoscf UHF expects a (2, nao, nao) density matrix")
        nspin = 2
        nocc_a, nocc_b = (int(n) for n in mf.nelec)
    else:
        if dm.ndim != 2:
            raise ValueError("mojoscf RHF expects a single (nao, nao) density matrix")
        nspin = 1
        nocc_a, nocc_b = mol.nelectron // 2, 0

    x_orth = _c(mf.check_linear_dependency(s1e, log))
    e_nuc = float(mf.energy_nuc())
    mf.scf_summary["nuc"] = e_nuc

    if dump_chk and mf.chkfile:
        chkfile.save_mol(mol, mf.chkfile)

    shift_a, shift_b = _pair(mf.level_shift) if uhf else (float(mf.level_shift), 0.0)
    damp_a, damp_b = _pair(mf.damp) if uhf else (float(mf.damp), 0.0)
    opts = {
        "conv_tol": float(conv_tol),
        "conv_tol_grad": float(conv_tol_grad),
        "max_cycle": int(mf.max_cycle),
        "diis": bool(mf.diis),
        "diis_space": int(mf.diis_space),
        "diis_start_cycle": int(mf.diis_start_cycle),
        "diis_damp": float(mf.diis_damp),
        "damp": damp_a,
        "damp_b": damp_b,
        "level_shift": shift_a,
        "level_shift_b": shift_b,
        "conv_check": bool(conv_check),
    }

    def get_veff(dm, dm_last, vhf_last, mo_coeff=None, mo_occ=None):
        # pyscf's make_rdm1 tags the density matrix with its orbitals and several
        # J/K builders (density fitting in particular) switch to a much cheaper
        # occupied-orbital algorithm when they find the tags, so keep them.
        if mo_coeff is None and dm_last is None and init_tags[0] is not None:
            mo_coeff, mo_occ = init_tags
        if mo_coeff is not None:
            dm = lib.tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
        if dm_last is None:
            return mf.get_veff(mol, dm)
        return mf.get_veff(mol, dm, dm_last, vhf_last)

    log_cb = None
    if log.verbose >= logger.INFO:

        def log_cb(cycle, e_tot, de, norm_g, norm_ddm):
            if cycle == -1:
                log.info("init E= %.15g", e_tot)
            elif cycle == -2:
                log.info(
                    "Extra cycle  E= %.15g  delta_E= %4.3g  |g|= %4.3g  |ddm|= %4.3g", e_tot, de, norm_g, norm_ddm
                )
            else:
                log.info(
                    "cycle= %d E= %.15g  delta_E= %4.3g  |g|= %4.3g  |ddm|= %4.3g",
                    cycle + 1, e_tot, de, norm_g, norm_ddm,
                )

    cb = None
    if callable(callback):

        def cb(env):
            env["dm"] = lib.tag_array(env["dm"], mo_coeff=env["mo_coeff"], mo_occ=env["mo_occ"])
            env["mf"] = mf
            env["mol"] = mol
            callback(env)

    veff_mode, veff_data, reason = native_veff(mf)
    if veff_mode == 1:
        log.info("mojoscf: density-fitted J/K built natively (naux = %d)", veff_data.shape[0])
    elif veff_mode == 2:
        log.info("mojoscf: in-core J/K built natively from 8-fold packed ERIs")
    elif veff_mode == 3:
        log.info("mojoscf: integral-direct J/K from the Mojo integral engine (direct_scf_tol = %g)", mf.direct_scf_tol)
    else:
        log.info("mojoscf: J/K from mf.get_veff (%s)", reason)
    mf.scf_summary["mojoscf_veff_mode"] = veff_mode
    veff_opts = {
        "block_mb": df_block_mb(),
        "fact_tol": 1e-14,
        "direct_tol": float(mf.direct_scf_tol),
        "incremental": bool(mf.direct_scf),
    }
    dm0_coeff = dm0_occ = None
    if init_tags[0] is not None:
        dm0_coeff = np.ascontiguousarray(init_tags[0], dtype=np.float64)
        dm0_occ = np.ascontiguousarray(init_tags[1], dtype=np.float64)

    cput1 = log.timer("initialize scf", *cput0)
    path, prefix = blas_args(s1e.shape[0])
    (seq_path, seq_prefix), _ = blas_config()
    res = get_extension().scf_kernel(
        h1e, s1e, dm, nspin, nocc_a, nocc_b, e_nuc, get_veff, x_orth, opts, log_cb, cb, path, prefix,
        seq_path, seq_prefix, veff_mode, veff_data, veff_opts, dm0_coeff, dm0_occ,
    )
    log.timer("scf iterations", *cput1)

    mf.cycles = int(res["cycles"])
    mf.scf_summary["e1"] = res["e1"]
    mf.scf_summary["e2"] = res["e2"]
    scf_conv = bool(res["converged"])
    e_tot = float(res["e_tot"])
    mo_energy = res["mo_energy"]
    mo_coeff = res["mo_coeff"]
    mo_occ = res["mo_occ"]
    if uhf:
        # pyscf updates scf_summary['gap'] and logs the HOMO/LUMO energies inside
        # get_occ; do it once for the final orbitals.
        mf.get_occ(mo_energy)
    elif "homo" in res:
        homo, lumo = res["homo"], res["lumo"]
        gap = (lumo - homo) * nist.HARTREE2EV
        mf.scf_summary["gap"] = gap
        if homo + 1e-3 > lumo:
            log.warn("HOMO %.15g == LUMO %.15g", homo, lumo)
        else:
            log.info("  HOMO = %.15g  LUMO = %.15g  gap/eV = %.5f", homo, lumo, gap)

    if dump_chk and mf.chkfile:
        mf.dump_chk({"e_tot": e_tot, "mo_energy": mo_energy, "mo_coeff": mo_coeff, "mo_occ": mo_occ})
    log.timer("scf_cycle", *cput0)
    return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ


def _fallback_reason(mf):
    """Why the native driver cannot be used for ``mf`` (None if it can)."""
    if callable(getattr(mf, "check_convergence", None)):
        return "custom check_convergence"
    if isinstance(mf.diis, lib.diis.DIIS):
        return "a DIIS instance was assigned to mf.diis"
    if mf.diis and not issubclass(mf.DIIS, (pyscf_diis.CDIIS, CDIIS)):
        return f"DIIS class {mf.DIIS.__name__} is not CDIIS"
    if mf.diis and getattr(mf, "diis_space_rollback", 0):
        return "diis_space_rollback is not supported"
    if getattr(mf, "diis_file", None):
        return "diis_file is not supported"
    if getattr(mf, "disp", None):
        return "dispersion corrections are not supported"
    name = _overridden_glue(mf)
    if name is not None:
        return f"mf.{name} is overridden"
    return None


def _is_hf(mf) -> bool:
    """True for (unrestricted or closed-shell restricted) Hartree-Fock, not ROHF, GHF or Kohn-Sham."""
    from pyscf.scf import rohf

    if not isinstance(mf, (pyscf_hf.RHF, pyscf_uhf.UHF)) or isinstance(mf, rohf.ROHF):
        return False
    try:
        from pyscf.dft.rks import KohnShamDFT
    except ImportError:  # pragma: no cover
        return True
    return not isinstance(mf, KohnShamDFT)


def _mojo_grad_method(mf, after):
    """``nuc_grad_method`` for ``mf``, called from class ``after`` of its MRO.

    The implementation ``mf`` would inherit past ``after`` decides: pyscf's
    plain RHF/UHF one gives ``mojoscf.grad.Gradients``/``UGradients``, the
    density-fitting one (``_DFHF``) ``DFGradients``/``DFUGradients``, a QM/MM
    one (``pyscf.qmmm``) the gradients of the method underneath with the Mojo
    QM/MM terms (:mod:`mojoscf.qmmm`); any other decoration (solvent, ...)
    keeps its own gradients.
    """
    from pyscf.df import df_jk

    from . import grad

    mro = type(mf).__mro__
    nxt = None
    for cls in mro[mro.index(after) + 1:]:
        if "nuc_grad_method" in cls.__dict__:
            nxt = cls.__dict__["nuc_grad_method"]
            break
    uhf = isinstance(mf, pyscf_uhf.UHF)
    if _is_hf(mf):          # mojoscf's gradients are Hartree-Fock ones (no ROHF or Kohn-Sham)
        if nxt is (pyscf_uhf.UHF if uhf else pyscf_hf.RHF).nuc_grad_method:
            return (grad.UGradients if uhf else grad.Gradients)(mf)
        if nxt is df_jk._DFHF.nuc_grad_method and not mf.istype("_Solvation"):
            return (grad.DFUGradients if uhf else grad.DFGradients)(mf)
    qmmm_itrf = sys.modules.get("pyscf.qmmm.itrf")
    if qmmm_itrf is not None and nxt is qmmm_itrf.QMMMSCF.nuc_grad_method:
        from . import qmmm

        return qmmm.qmmm_grad_for_scf(_mojo_grad_method(mf, qmmm_itrf.QMMMSCF))
    return nxt(mf)


class _MojoDFHook:
    """Placed in front of pyscf's ``_DFHF`` by ``density_fit()`` of the Mojo classes.

    ``_DFHF.nuc_grad_method`` comes first in the MRO of a density-fitted object
    and does not call ``super()``; this class routes it to the Mojo gradients.
    """

    __name_mixin__ = "Mojo"

    def nuc_grad_method(self):
        """Density-fitted nuclear gradients with Mojo derivative integrals (:mod:`mojoscf.grad`)."""
        return _mojo_grad_method(self, _MojoDFHook)


def _add_qmmm_hook(mf):
    """Put ``mojoscf.qmmm._MojoQMMMHook`` in front of a QM/MM object (pyscf.qmmm) that lacks it.

    ``accelerate`` adds it to QM/MM objects; this covers a Mojo object
    decorated afterwards (``qmmm.mm_charge(mojoscf.RHF(mol), ...)``), where
    pyscf's ``QMMMSCF`` comes first in the MRO.
    """
    qmmm_itrf = sys.modules.get("pyscf.qmmm.itrf")     # a QM/MM object implies it is loaded
    if qmmm_itrf is None or not isinstance(mf, qmmm_itrf.QMMMSCF):
        return
    from .qmmm import _MojoQMMMHook

    if not isinstance(mf, _MojoQMMMHook):
        lib.set_class(mf, (_MojoQMMMHook, type(mf)))


class _MojoGlueMixin:
    """Pieces shared by the RHF and UHF mixins: driver entry point and eigensolver."""

    DIIS = CDIIS

    def kernel(self, dm0=None, **kwargs):
        cput0 = (logger.process_clock(), logger.perf_counter())
        _add_qmmm_hook(self)
        self.dump_flags()
        self.build(self.mol)
        if dm0 is None and self.mo_coeff is not None and self.mo_occ is not None:
            dm0 = self.make_rdm1()
        if self.max_cycle > 0 or self.mo_coeff is None:
            self.converged, self.e_tot, self.mo_energy, self.mo_coeff, self.mo_occ = kernel(
                self, self.conv_tol, self.conv_tol_grad, dm0=dm0, callback=self.callback,
                conv_check=self.conv_check, **kwargs,
            )
        else:
            self.e_tot = kernel(
                self, self.conv_tol, self.conv_tol_grad, dm0=dm0, callback=self.callback,
                conv_check=self.conv_check, **kwargs,
            )[1]
        logger.timer(self, "SCF", *cput0)
        self._finalize()
        return self.e_tot

    scf = kernel

    def density_fit(self, auxbasis=None, with_df=None, only_dfj=False):
        mf = super().density_fit(auxbasis, with_df, only_dfj)
        if not isinstance(mf, _MojoDFHook):
            lib.set_class(mf, (_MojoDFHook, type(mf)))
        return mf

    def _eigh(self, h, s, overwrite=False, x=None):
        if not _is_real(h, s, x):
            return super()._eigh(h, s, overwrite, x)
        if x is None:
            return kernels.eigh(h, s)
        return kernels.eigh(h, x=x)


class _MojoRHFMixin(_MojoGlueMixin):
    """Mojo implementations of the RHF glue; mixed in front of a pyscf RHF class."""

    def make_rdm1(self, mo_coeff=None, mo_occ=None, **kwargs):
        if mo_occ is None:
            mo_occ = self.mo_occ
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if not _is_real(mo_coeff, mo_occ):
            return super().make_rdm1(mo_coeff, mo_occ, **kwargs)
        dm = kernels.make_rdm1(mo_coeff, mo_occ)
        return lib.tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)

    def energy_elec(self, dm=None, h1e=None, vhf=None):
        if dm is None:
            dm = self.make_rdm1()
        if h1e is None:
            h1e = self.get_hcore()
        if vhf is None:
            vhf = self.get_veff(self.mol, dm)
        if not _is_real(dm, h1e, vhf) or np.ndim(dm) != 2:
            return super().energy_elec(dm, h1e, vhf)
        e1 = kernels.trace_prod(h1e, dm)
        e2 = 0.5 * kernels.trace_prod(vhf, dm)
        self.scf_summary["e1"] = e1
        self.scf_summary["e2"] = e2
        if hasattr(vhf, "ecoul"):
            ecoul = vhf.ecoul.real
            exx = e2 - ecoul
            self.scf_summary["coul"] = ecoul
            self.scf_summary["exc"] = exx
            logger.debug(self, "E1 = %s  E2 = %s  Ecoul = %s  Exc = %s", e1, e2, ecoul, exx)
        else:
            logger.debug(self, "E1 = %s  E2 = %s", e1, e2)
        return e1 + e2, e2

    def get_occ(self, mo_energy=None, mo_coeff=None):
        if mo_energy is None:
            mo_energy = self.mo_energy
        if not _is_real(mo_energy):
            return super().get_occ(mo_energy, mo_coeff)
        nocc = self.mol.nelectron // 2
        mo_occ, gap = kernels.get_occ(mo_energy, nocc)
        if gap is not None:
            homo, lumo = gap
            ev_gap = (lumo - homo) * nist.HARTREE2EV
            self.scf_summary["gap"] = ev_gap
            if self.verbose >= logger.INFO:
                if homo + 1e-3 > lumo:
                    logger.warn(self, "HOMO %.15g == LUMO %.15g", homo, lumo)
                else:
                    logger.info(self, "  HOMO = %.15g  LUMO = %.15g  gap/eV = %.5f", homo, lumo, ev_gap)
        elif nocc > np.size(mo_energy):
            logger.warn(self, "Not enough orbitals for %d electrons", self.mol.nelectron)
        if self.verbose >= logger.DEBUG:
            np.set_printoptions(threshold=len(mo_energy))
            logger.debug(self, "  mo_energy =\n%s", mo_energy)
            np.set_printoptions(threshold=1000)
        return mo_occ

    def get_grad(self, mo_coeff, mo_occ, fock=None):
        if fock is None:
            dm1 = self.make_rdm1(mo_coeff, mo_occ)
            fock = self.get_hcore(self.mol) + self.get_veff(self.mol, dm1)
        if not _is_real(mo_coeff, mo_occ, fock):
            return super().get_grad(mo_coeff, mo_occ, fock)
        return kernels.get_grad(mo_coeff, mo_occ, fock)

    def nuc_grad_method(self):
        """Nuclear gradients with Mojo derivative integrals (:mod:`mojoscf.grad`)."""
        return _mojo_grad_method(self, _MojoRHFMixin)


class _MojoUHFMixin(_MojoGlueMixin):
    """Mojo implementations of the UHF glue; mixed in front of a pyscf UHF class.

    ``get_occ`` is left to pyscf: it is cheap and carries UHF-specific HOMO/LUMO
    bookkeeping that is not worth duplicating.
    """

    def make_rdm1(self, mo_coeff=None, mo_occ=None, **kwargs):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_occ is None:
            mo_occ = self.mo_occ
        if not _is_real(mo_coeff, mo_occ):
            return super().make_rdm1(mo_coeff, mo_occ, **kwargs)
        dm_a = kernels.make_rdm1(mo_coeff[0], mo_occ[0])
        dm_b = kernels.make_rdm1(mo_coeff[1], mo_occ[1])
        return lib.tag_array((dm_a, dm_b), mo_coeff=mo_coeff, mo_occ=mo_occ)

    def energy_elec(self, dm=None, h1e=None, vhf=None):
        if dm is None:
            dm = self.make_rdm1()
        if h1e is None:
            h1e = self.get_hcore()
        if isinstance(dm, np.ndarray) and dm.ndim == 2:
            dm = np.array((dm * 0.5, dm * 0.5))
        if vhf is None:
            vhf = self.get_veff(self.mol, dm)
        dm_arr = np.asarray(dm)
        if not _is_real(dm_arr, h1e, vhf) or np.ndim(h1e) != 2 or dm_arr.ndim != 3 or np.ndim(vhf) != 3:
            return super().energy_elec(dm, h1e, vhf)
        e1 = kernels.trace_prod(h1e, dm_arr[0]) + kernels.trace_prod(h1e, dm_arr[1])
        e2 = 0.5 * (kernels.trace_prod(vhf[0], dm_arr[0]) + kernels.trace_prod(vhf[1], dm_arr[1]))
        self.scf_summary["e1"] = e1
        self.scf_summary["e2"] = e2
        if hasattr(vhf, "ecoul"):
            ecoul = vhf.ecoul.real
            exx = e2 - ecoul
            self.scf_summary["coul"] = ecoul
            self.scf_summary["exc"] = exx
            logger.debug(self, "E1 = %s  E2 = %s  Ecoul = %s  Exc = %s", e1, e2, ecoul, exx)
        else:
            logger.debug(self, "E1 = %s  E2 = %s", e1, e2)
        return e1 + e2, e2

    def get_grad(self, mo_coeff, mo_occ, fock=None):
        if fock is None:
            dm1 = self.make_rdm1(mo_coeff, mo_occ)
            fock = self.get_hcore(self.mol) + self.get_veff(self.mol, dm1)
        if not _is_real(mo_coeff, mo_occ, fock):
            return super().get_grad(mo_coeff, mo_occ, fock)
        ga = kernels.get_grad(mo_coeff[0], mo_occ[0], fock[0], 1.0)
        gb = kernels.get_grad(mo_coeff[1], mo_occ[1], fock[1], 1.0)
        return np.hstack((ga, gb))

    def nuc_grad_method(self):
        """Nuclear gradients with Mojo derivative integrals (:mod:`mojoscf.grad`)."""
        return _mojo_grad_method(self, _MojoUHFMixin)


class RHF(_MojoRHFMixin, pyscf_hf.RHF):
    """Restricted Hartree-Fock whose SCF loop and glue run in Mojo.

    Behaves like :class:`pyscf.scf.hf.RHF`; see :func:`kernel` for the few
    situations in which the pyscf loop is used instead.
    """


class UHF(_MojoUHFMixin, pyscf_uhf.UHF):
    """Unrestricted Hartree-Fock whose SCF loop and glue run in Mojo.

    Behaves like :class:`pyscf.scf.uhf.UHF`, including ``init_guess_breaksym``,
    spin-dependent ``level_shift`` and broken-symmetry initial densities passed
    through ``dm0``.
    """


_accelerated_classes: dict[type, type] = {}


def unsupported_reason(mf):
    """Why :func:`accelerate` rejects ``mf`` (None if it is supported)."""
    if isinstance(mf, (_MojoRHFMixin, _MojoUHFMixin)):
        return None
    if isinstance(mf, pyscf_uhf.UHF):
        kind = "UHF"
    elif isinstance(mf, pyscf_hf.RHF):
        kind = "RHF"
        from pyscf.scf import rohf

        if isinstance(mf, rohf.ROHF):
            return "ROHF is not supported"
    else:
        return f"{type(mf).__name__} is neither a closed-shell RHF nor a UHF object"
    try:
        from pyscf.dft.rks import KohnShamDFT

        if isinstance(mf, KohnShamDFT):
            return "Kohn-Sham DFT is not supported"
    except ImportError:  # pragma: no cover
        pass
    try:
        from pyscf.soscf.newton_ah import _CIAH_SOSCF

        if isinstance(mf, _CIAH_SOSCF):
            return "second-order SCF is not supported"
    except ImportError:  # pragma: no cover
        pass
    if getattr(mf.mol, "symmetry", False):
        return "point-group symmetry is not supported"
    name = _overridden_glue(mf)
    if name is not None:
        return f"{kind} object overrides {name}, which the native loop re-implements"
    return None


def is_supported(mf) -> bool:
    """True if ``mf`` is a closed-shell RHF or a UHF object the Mojo driver can run."""
    return unsupported_reason(mf) is None


def accelerate(mf):
    """Replace the SCF loop and glue methods of an RHF/UHF object with Mojo versions.

    The object is modified in place (its class becomes a subclass of the
    original one with the Mojo mixin in front) and returned.  Density fitting,
    X2C and other decorations that only change ``get_jk``/``get_hcore`` are
    preserved; QM/MM objects (``pyscf.qmmm``) get their MM-charge terms from
    the Mojo engine (:mod:`mojoscf.qmmm`).  ROHF, Kohn-Sham, symmetry-adapted, second-order SCF objects and
    objects that override the glue methods (smearing, constrained UHF, ...) are
    rejected with ``TypeError``.
    """
    if isinstance(mf, (_MojoRHFMixin, _MojoUHFMixin)):
        return mf
    reason = unsupported_reason(mf)
    if reason is not None:
        raise TypeError(f"mojoscf.accelerate cannot accelerate {type(mf).__name__}: {reason}")
    mixin = _MojoUHFMixin if _is_uhf(mf) else _MojoRHFMixin
    cls = type(mf)
    new_cls = _accelerated_classes.get(cls)
    if new_cls is None:
        new_cls = type("Mojo" + cls.__name__, (mixin, cls), {"__module__": __name__})
        _accelerated_classes[cls] = new_cls
    mf.__class__ = new_cls
    _add_qmmm_hook(mf)
    return mf
