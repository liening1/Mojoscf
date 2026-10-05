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


# --------------------------------------------------------------------------
# Transition-metal complexes (idealised geometries, Angstrom)
# --------------------------------------------------------------------------


def ferrocene_atoms(fe_ring=1.66, cc=1.43, ch=1.08):
    """Eclipsed ferrocene Fe(C5H5)2 (D5h)."""
    r = cc / (2 * np.sin(np.pi / 5))
    atoms = [("Fe", (0.0, 0.0, 0.0))]
    for z in (fe_ring, -fe_ring):
        for k in range(5):
            a = 2 * np.pi * k / 5
            atoms.append(("C", (r * np.cos(a), r * np.sin(a), z)))
            atoms.append(("H", ((r + ch) * np.cos(a), (r + ch) * np.sin(a), z)))
    return atoms


def _ligand(kind, pos, axis):
    """Atoms of a ligand bound through its first atom at ``pos``, pointing along the unit vector ``axis``."""
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    perp = np.cross(axis, [0.0, 0.0, 1.0] if abs(axis[2]) < 0.9 else [1.0, 0.0, 0.0])
    perp /= np.linalg.norm(perp)
    perp2 = np.cross(axis, perp)
    pos = np.asarray(pos, float)
    if kind == "H2O":
        # oxygen lone pair towards the metal, H-O-H 104.5 deg in the plane of axis and perp
        h = 0.96
        half = np.deg2rad(104.5 / 2)
        return [("O", tuple(pos))] + [
            ("H", tuple(pos + h * (np.cos(half) * axis + s * np.sin(half) * perp))) for s in (1, -1)
        ]
    if kind == "NH3":
        nh = 1.01
        out = [("N", tuple(pos))]
        # H atoms 109.5 deg from the M-N bond: 70.5 deg away from the outward axis
        c, sn = np.cos(np.deg2rad(70.5)), np.sin(np.deg2rad(70.5))
        for k in range(3):
            a = 2 * np.pi * k / 3
            out.append(("H", tuple(pos + nh * (c * axis + sn * (np.cos(a) * perp + np.sin(a) * perp2)))))
        return out
    if kind == "CO":
        return [("C", tuple(pos)), ("O", tuple(pos + 1.14 * axis))]
    if kind == "Cl":
        return [("Cl", tuple(pos))]
    raise ValueError(kind)


def octahedral_atoms(metal, ligand, bond):
    """ML6 with the ligands on the +-x, +-y, +-z axes."""
    atoms = [(metal, (0.0, 0.0, 0.0))]
    for v in np.vstack([np.eye(3), -np.eye(3)]):
        atoms += _ligand(ligand, bond * v, v)
    return atoms


def tetrahedral_atoms(metal, ligand, bond):
    """ML4 tetrahedral."""
    atoms = [(metal, (0.0, 0.0, 0.0))]
    for v in np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float) / np.sqrt(3):
        atoms += _ligand(ligand, bond * v, v)
    return atoms


def square_planar_atoms(metal, ligands, bonds):
    """ML4 square planar in the xy plane; ``ligands``/``bonds`` in the order +x, +y, -x, -y."""
    atoms = [(metal, (0.0, 0.0, 0.0))]
    for v, lig, b in zip(np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]], float), ligands, bonds):
        atoms += _ligand(lig, b * v, v)
    return atoms


def solvated_ion_atoms(metal="Fe", n_second=12, bond=2.12, r2=4.3, seed=1):
    """[M(H2O)6] with ``n_second`` second-shell waters on a jittered sphere of radius ``r2``."""
    rng = np.random.default_rng(seed)
    atoms = octahedral_atoms(metal, "H2O", bond)
    # quasi-uniform directions (Fibonacci sphere), randomly oriented waters
    k = np.arange(n_second) + 0.5
    th = np.arccos(1 - 2 * k / n_second)
    ph = np.pi * (1 + 5**0.5) * k
    for t, p in zip(th, ph):
        o = r2 * np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)]) + rng.normal(scale=0.1, size=3)
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        atoms += [("O", tuple(o)), ("H", tuple(o + q @ np.array([0.0, 0.757, 0.587]))),
                  ("H", tuple(o + q @ np.array([0.0, -0.757, 0.587])))]
    return atoms
