"""Mojo-backed CDIIS, a drop-in for :class:`pyscf.scf.diis.CDIIS`.

The extrapolation reproduces ``pyscf.lib.diis.DIIS`` exactly (ring buffers of
``space`` vectors, bordered overlap matrix, pseudo-inverse when the overlap
matrix is singular) but keeps the vectors in NumPy buffers and performs the
error-vector construction, the overlaps and the extrapolation in Mojo.

Only the in-core variant is implemented: ``filename``/``incore`` and
``restore`` are accepted for API compatibility but the DIIS history is never
written to disk.
"""
from __future__ import annotations

import numpy as np
from pyscf import lib

from . import kernels
from ._backend import get_extension

__all__ = ["CDIIS", "SCF_DIIS", "DIIS"]


class CDIIS(lib.diis.DIIS):
    def __init__(self, mf=None, filename=None, Corth=None):
        lib.diis.DIIS.__init__(self, mf, filename)
        self.rollback = 0
        self.space = 8
        self.Corth = Corth
        self.damp = 0
        self._bufx = None
        self._bufe = None
        self._hmat = None
        self._state = np.zeros(2, dtype=np.int64)
        self._shape = None

    def _reset(self, vlen: int, elen: int) -> None:
        self._bufx = np.zeros((self.space, vlen))
        self._bufe = np.zeros((self.space, elen))
        self._hmat = np.zeros((self.space + 1, self.space + 1))
        get_extension().diis_init(self._hmat, self._state)

    def update(self, s, d, f, *args, **kwargs):
        f = np.ascontiguousarray(f, dtype=np.float64)
        errvec = kernels.diis_errvec(s, d, f, self.Corth)
        f_prev = kwargs.get("f_prev", None)
        if abs(self.damp) < 1e-6 or f_prev is None:
            xin = f
        else:
            xin = kernels.damping(f, f_prev, self.damp)
        x = xin.reshape(-1)
        if (
            self._bufx is None
            or self._bufx.shape != (self.space, x.size)
            or self._bufe.shape != (self.space, errvec.size)
        ):
            self._reset(x.size, errvec.size)
        out = np.empty_like(x)
        get_extension().diis_update(
            x, errvec, self._bufx, self._bufe, self._hmat, self._state, int(self.space), int(self.min_space), out
        )
        return out.reshape(f.shape)

    def get_num_vec(self):
        return int(self._state[1])

    # The generic push/extrapolate API of pyscf.lib.diis.DIIS operates on its
    # own buffers; the Mojo version only supports the ``update`` entry point.
    def push_err_vec(self, xerr):  # pragma: no cover - API compatibility
        raise NotImplementedError("mojoscf CDIIS only supports update(s, d, f)")

    def push_vec(self, x):  # pragma: no cover - API compatibility
        raise NotImplementedError("mojoscf CDIIS only supports update(s, d, f)")

    def extrapolate(self, nd=None):  # pragma: no cover - API compatibility
        raise NotImplementedError("mojoscf CDIIS only supports update(s, d, f)")

    def restore(self, filename, inplace=True):  # pragma: no cover - API compatibility
        raise NotImplementedError("mojoscf CDIIS keeps its history in memory only")


SCF_DIIS = DIIS = CDIIS
