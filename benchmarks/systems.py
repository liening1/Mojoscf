"""Molecular geometries used by the large-system and broken-symmetry tests."""
from __future__ import annotations

import itertools

import numpy as np

PHI = (1 + 5**0.5) / 2


def c60_atoms(bond=1.42):
    """Truncated icosahedron (C60) with the given C-C bond length in Angstrom."""
    base = []
    for sx, sy, sz in itertools.product((1, -1), repeat=3):
        base += [(0, sx * 1, sy * 3 * PHI), (sx * 1, sy * (2 + PHI), sz * 2 * PHI), (sx * PHI, sy * 2, sz * (2 * PHI + 1))]
    pts = set()
    for p in base:
        # all cyclic permutations; signs already enumerated
        for k in range(3):
            q = tuple(p[(i + k) % 3] for i in range(3))
            pts.add(tuple(round(v, 9) for v in q))
    pts = np.array(sorted(pts))
    assert len(pts) == 60, len(pts)
    pts *= bond / 2.0  # edge length of this construction is 2
    return [("C", tuple(p)) for p in pts]


def alkane_atoms(n, cc=1.53, ch=1.09):
    """All-trans linear alkane CnH2n+2."""
    ang = np.deg2rad(109.5 / 2)
    atoms = []
    xs = []
    for i in range(n):
        x = i * cc * np.sin(ang)
        y = 0.0 if i % 2 == 0 else cc * np.cos(ang)
        xs.append((x, y))
        atoms.append(("C", (x, y, 0.0)))
    for i, (x, y) in enumerate(xs):
        s = 1.0 if i % 2 == 0 else -1.0
        for sz in (1.0, -1.0):
            atoms.append(("H", (x, y - s * ch * np.cos(np.deg2rad(54.75)), sz * ch * np.sin(np.deg2rad(54.75)))))
    atoms.append(("H", (xs[0][0] - ch * np.sin(ang), xs[0][1] + ch * np.cos(ang), 0.0)))
    xe, ye = xs[-1]
    s = 1.0 if (n - 1) % 2 == 0 else -1.0
    atoms.append(("H", (xe + ch * np.sin(ang), ye + s * ch * np.cos(ang), 0.0)))
    return atoms


def water_cluster_atoms(n, spacing=2.9, seed=0):
    """n water molecules on a jittered cubic lattice with random orientations."""
    rng = np.random.default_rng(seed)
    side = int(np.ceil(n ** (1 / 3)))
    cells = list(itertools.product(range(side), repeat=3))[:n]
    atoms = []
    for c in cells:
        o = np.array(c, float) * spacing + rng.normal(scale=0.12, size=3)
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        h1 = q @ np.array([0.0, 0.757, 0.587])
        h2 = q @ np.array([0.0, -0.757, 0.587])
        atoms += [("O", tuple(o)), ("H", tuple(o + h1)), ("H", tuple(o + h2))]
    return atoms


def atoms_to_str(atoms):
    return "; ".join(f"{s} {x:.8f} {y:.8f} {z:.8f}" for s, (x, y, z) in atoms)
