"""Micro-benchmarks of the individual glue kernels versus their pyscf versions."""
from __future__ import annotations

import time

import numpy as np
from pyscf import lib
from pyscf.scf import diis as pyscf_diis
from pyscf.scf import hf as pyscf_hf

import mojoscf
from mojoscf import kernels


def timeit(fn, repeat=20):
    fn()
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def bench(nao, nocc, rng):
    c = rng.standard_normal((nao, nao))
    occ = np.zeros(nao)
    occ[:nocc] = 2
    s = c @ c.T / nao + np.eye(nao)
    f = rng.standard_normal((nao, nao))
    f = f + f.T
    dm = pyscf_hf.make_rdm1(c, occ)
    h = f.copy()
    e = np.sort(rng.standard_normal(nao))
    x = pyscf_hf.check_linear_dependency(s)

    class _MF:  # minimal stand-in for get_occ
        mol = type("mol", (), {"nelectron": 2 * nocc})()
        verbose = 0
        scf_summary = {}
        mo_energy = None

    mf = _MF()
    rows = [
        ("make_rdm1", lambda: pyscf_hf.make_rdm1(c, occ), lambda: kernels.make_rdm1(c, occ)),
        ("energy_elec", lambda: (np.einsum("ij,ji->", h, dm), np.einsum("ij,ji->", f, dm)),
         lambda: kernels.energy_elec(h, f, dm)),
        ("get_occ", lambda: pyscf_hf.get_occ(mf, e), lambda: kernels.get_occ(e, nocc)),
        ("get_grad", lambda: pyscf_hf.get_grad(c, occ, f), lambda: kernels.get_grad(c, occ, f)),
        ("level_shift", lambda: pyscf_hf.level_shift(s, dm * 0.5, f, 0.2), lambda: kernels.level_shift(s, dm, f, 0.2)),
        ("diis_errvec (orth)", lambda: pyscf_diis.get_err_vec(s, dm, f, x), lambda: kernels.diis_errvec(s, dm, f, x)),
        ("eig (F, x)", lambda: pyscf_hf.SCF._eigh(pyscf_hf.RHF.__new__(pyscf_hf.RHF), f, s, x=x), lambda: kernels.eigh(f, x=x)),
    ]
    print(f"\nnao = {nao}, nocc = {nocc}")
    print(f"{'kernel':22s} {'pyscf [us]':>12s} {'mojo [us]':>12s} {'speedup':>8s}")
    for name, ref, mine in rows:
        t_ref = timeit(ref)
        t_mine = timeit(mine)
        print(f"{name:22s} {t_ref * 1e6:12.1f} {t_mine * 1e6:12.1f} {t_ref / t_mine:7.1f}x")

    # DIIS update with a warm 8-vector history; every call gets a fresh Fock
    # matrix so the DIIS system never becomes singular.
    ref = pyscf_diis.CDIIS(Corth=x)
    mine = mojoscf.CDIIS(Corth=x)
    gs = [rng.standard_normal((nao, nao)) for _ in range(64)]
    gs = [g + g.T for g in gs]
    counter = [0]

    def next_g():
        counter[0] += 1
        return gs[counter[0] % len(gs)]

    for _ in range(8):
        g = next_g()
        ref.update(s, dm, g)
        mine.update(s, dm, g)
    t_ref = timeit(lambda: ref.update(s, dm, next_g()))
    t_mine = timeit(lambda: mine.update(s, dm, next_g()))
    print(f"{'CDIIS.update':22s} {t_ref * 1e6:12.1f} {t_mine * 1e6:12.1f} {t_ref / t_mine:7.1f}x")


def main():
    info = mojoscf.backend_info()
    print(f"mojoscf {mojoscf.__version__}; BLAS: {info['blas_library'] or 'native Mojo fallback'}; "
          f"threads: pyscf={lib.num_threads()} mojo={info['parallelism_level']}")
    rng = np.random.default_rng(0)
    for nao, nocc in ((24, 5), (115, 21), (300, 60)):
        bench(nao, nocc, rng)


if __name__ == "__main__":
    main()
