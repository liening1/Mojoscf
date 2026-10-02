"""RHF with the SCF iterations executed in Mojo.

Two ways to use it::

    import mojoscf
    mf = mojoscf.RHF(mol).run()          # a pyscf RHF object with a Mojo SCF loop

    mf = pyscf.scf.RHF(mol).density_fit()
    mojoscf.accelerate(mf)               # upgrade an existing RHF object in place
    mf.kernel()

The driver :func:`kernel` is a port of :func:`pyscf.scf.hf.kernel`: it computes
the same quantities in the same order, so energies, orbitals and iteration
counts agree with pyscf to numerical precision.  Only the two-electron part
(``mf.get_veff``) is still evaluated by pyscf, since that is compiled C code
already.  Whatever the Mojo driver cannot handle (custom DIIS objects, custom
convergence checks, ...) falls back to pyscf's own loop, with the small glue
functions still replaced by their Mojo versions.
"""
from __future__ import annotations

import numpy as np
from pyscf import lib
from pyscf.data import nist
from pyscf.lib import logger
from pyscf.scf import chkfile
from pyscf.scf import diis as pyscf_diis
from pyscf.scf import hf as pyscf_hf

from . import kernels
from ._backend import blas_args, get_extension
from .diis import CDIIS

__all__ = ["RHF", "kernel", "accelerate", "is_supported"]


def _c(a):
    return np.ascontiguousarray(a, dtype=np.float64)


def _is_real(*arrays) -> bool:
    return all(not np.iscomplexobj(a) for a in arrays if a is not None)


def kernel(mf, conv_tol=1e-10, conv_tol_grad=None, dump_chk=True, dm0=None, callback=None, conv_check=True, **kwargs):
    """Mojo SCF driver with the signature and return value of ``pyscf.scf.hf.kernel``.

    Returns ``(scf_conv, e_tot, mo_energy, mo_coeff, mo_occ)``.
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

    mol = mf.mol
    s1e = mf.get_ovlp(mol)
    if dm0 is None:
        dm = mf.get_init_guess(mol, mf.init_guess, s1e=s1e, **kwargs)
    else:
        dm = dm0
    h1e = mf.get_hcore(mol)
    if not _is_real(s1e, h1e, dm):
        log.info("mojoscf: complex integrals; running the pyscf SCF loop with Mojo glue kernels")
        return pyscf_hf.kernel(
            mf, conv_tol, conv_tol_grad, dump_chk=dump_chk, dm0=dm0, callback=callback, conv_check=conv_check, **kwargs
        )
    s1e = _c(s1e)
    h1e = _c(h1e)
    dm = _c(dm)
    if dm.ndim != 2:
        raise ValueError("mojoscf RHF expects a single (nao, nao) density matrix")

    x_orth = _c(mf.check_linear_dependency(s1e, log))
    nocc = mol.nelectron // 2
    e_nuc = float(mf.energy_nuc())
    mf.scf_summary["nuc"] = e_nuc

    if dump_chk and mf.chkfile:
        chkfile.save_mol(mol, mf.chkfile)

    opts = {
        "conv_tol": float(conv_tol),
        "conv_tol_grad": float(conv_tol_grad),
        "max_cycle": int(mf.max_cycle),
        "diis": bool(mf.diis),
        "diis_space": int(mf.diis_space),
        "diis_start_cycle": int(mf.diis_start_cycle),
        "diis_damp": float(mf.diis_damp),
        "damp": float(mf.damp),
        "level_shift": float(mf.level_shift),
        "conv_check": bool(conv_check),
    }

    def get_veff(dm, dm_last, vhf_last):
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
            env["mf"] = mf
            env["mol"] = mol
            callback(env)

    cput1 = log.timer("initialize scf", *cput0)
    path, prefix = blas_args()
    res = get_extension().rhf_kernel(h1e, s1e, dm, nocc, e_nuc, get_veff, x_orth, opts, log_cb, cb, path, prefix)
    log.timer("scf iterations", *cput1)

    mf.cycles = int(res["cycles"])
    mf.scf_summary["e1"] = res["e1"]
    mf.scf_summary["e2"] = res["e2"]
    if "homo" in res:
        homo, lumo = res["homo"], res["lumo"]
        gap = (lumo - homo) * nist.HARTREE2EV
        mf.scf_summary["gap"] = gap
        if homo + 1e-3 > lumo:
            log.warn("HOMO %.15g == LUMO %.15g", homo, lumo)
        else:
            log.info("  HOMO = %.15g  LUMO = %.15g  gap/eV = %.5f", homo, lumo, gap)
    scf_conv = bool(res["converged"])
    e_tot = float(res["e_tot"])
    mo_energy = res["mo_energy"]
    mo_coeff = res["mo_coeff"]
    mo_occ = res["mo_occ"]

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
    return None


class _MojoRHFMixin:
    """Mojo implementations of the RHF glue; mixed in front of a pyscf RHF class."""

    DIIS = CDIIS

    def kernel(self, dm0=None, **kwargs):
        cput0 = (logger.process_clock(), logger.perf_counter())
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

    def _eigh(self, h, s, overwrite=False, x=None):
        if not _is_real(h, s, x):
            return super()._eigh(h, s, overwrite, x)
        if x is None:
            return kernels.eigh(h, s)
        return kernels.eigh(h, x=x)


class RHF(_MojoRHFMixin, pyscf_hf.RHF):
    """Restricted Hartree-Fock whose SCF loop and glue run in Mojo.

    Behaves like :class:`pyscf.scf.hf.RHF`; see :func:`kernel` for the few
    situations in which the pyscf loop is used instead.
    """


_accelerated_classes: dict[type, type] = {}


def is_supported(mf) -> bool:
    """True if ``mf`` is a closed-shell RHF object the Mojo driver can run."""
    if not isinstance(mf, pyscf_hf.RHF):
        return False
    from pyscf.scf import rohf

    if isinstance(mf, rohf.ROHF):
        return False
    try:
        from pyscf.dft.rks import KohnShamDFT

        if isinstance(mf, KohnShamDFT):
            return False
    except ImportError:  # pragma: no cover
        pass
    try:
        from pyscf.soscf.newton_ah import _CIAH_SOSCF

        if isinstance(mf, _CIAH_SOSCF):
            return False
    except ImportError:  # pragma: no cover
        pass
    if getattr(mf.mol, "symmetry", False):
        return False
    return True


def accelerate(mf):
    """Replace the SCF loop and glue methods of an RHF object with Mojo versions.

    The object is modified in place (its class becomes a subclass of the
    original one with :class:`_MojoRHFMixin` in front) and returned.  Density
    fitting, X2C and other decorations that only change ``get_jk``/``get_hcore``
    are preserved.  ROHF, Kohn-Sham, symmetry-adapted and second-order SCF
    objects are rejected with ``TypeError``.
    """
    if isinstance(mf, _MojoRHFMixin):
        return mf
    if not is_supported(mf):
        raise TypeError(f"mojoscf.accelerate supports closed-shell RHF objects only, got {type(mf).__name__}")
    cls = type(mf)
    new_cls = _accelerated_classes.get(cls)
    if new_cls is None:
        new_cls = type("Mojo" + cls.__name__, (_MojoRHFMixin, cls), {"__module__": __name__})
        _accelerated_classes[cls] = new_cls
    mf.__class__ = new_cls
    return mf
