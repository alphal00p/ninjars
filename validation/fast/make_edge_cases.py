#!/usr/bin/env python3
"""Edge-case "ninjars-bench-case-v1" inputs for the fast engine's cross-check
against original Ninja (validation/fast/ninja_reference/).

Physical-scale kinematics (top mass 173, Higgs 125) with the degenerate
configurations the synthetic cases avoid on purpose: lightlike and vanishing
invariants (exact zeros in the S-matrix), massless and complex internal masses
mixed in one loop, infrared-divergent massless triangles and boxes (1/eps^2),
uncut momenta orthogonal to Ninja's first (and second) tadpole reference basis,
coincident shifts (zero-momentum bubbles, singular triangle Gram determinants)
and collinear box momenta, and seven-propagator top loops (2 -> 5). Unstable
and stable points share a case so that per-lane isolation is checked against
Ninja's per-point return status. Every rank from n-1 to n+1 is covered where
meaningful. Deterministic.

`--extended` writes instead the larger loops kept out of the repository (their
Ninja reference files are 100-170 kB each): seven propagators at rank n+1 and
eight at ranks n and n+1, where both codes lose up to ~1e-8 to conditioning
(validation/fast/evidence/ninja_reference.json).

Usage: make_edge_cases.py [--extended] [output directory]
       (default: ninja_reference/cases)
"""

import json
import math
import pathlib
import random
import sys

FORMAT = "ninjars-bench-case-v1"
MT2 = 173.0 ** 2
MT2C = [MT2, -173.0 * 1.5]
MH2 = 125.0 ** 2
INVSQRT2, INVSQRT3, SQRT3 = 1 / math.sqrt(2), 1 / math.sqrt(3), math.sqrt(3)


def ncoeff(rank):
    return (rank + 1) * (rank + 2) * (rank + 3) * (rank + 4) // 24


def degrees(rank):
    out = []
    for d in range(rank + 1):
        out.extend([d] * ((d + 1) * (d + 2) * (d + 3) // 6))
    return out


def mp(a, b):
    return a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]


def add(a, b):
    return [x + y for x, y in zip(a, b)]


def sub(a, b):
    return [x - y for x, y in zip(a, b)]


def scale(a, s):
    return [x * s for x in a]


def basis_e1_e2(k1, k2):
    """Ninja's Basis(k1, k2) real massless e1, e2 (both momenta massive)."""
    k1q, k2q, k1k2 = mp(k1, k1), mp(k2, k2), mp(k1, k2)
    gamma = k1k2 + math.copysign(1.0, k1k2) * math.sqrt(k1k2 * k1k2 - k1q * k2q)
    r1, r2 = k1q / gamma, k2q / gamma
    den = 1 - r1 * r2
    return scale(sub(k1, scale(k2, r1)), 1 / den), scale(sub(k2, scale(k1, r2)), 1 / den)


TADPOLE_REFS = [
    ([2.0, INVSQRT3, -INVSQRT3, INVSQRT3], [SQRT3, INVSQRT3, -INVSQRT2, INVSQRT2]),
    ([2.0, 0.5 * (1 + INVSQRT3), 0.5 * (1 - INVSQRT3), INVSQRT3],
     [SQRT3, 0.5 * (SQRT3 * INVSQRT2 + INVSQRT3), 0.5 * (1 - INVSQRT2), INVSQRT2]),
]


def tensor(rng, rank, mass=173.0):
    """Dense complex tensor with physical scaling (degree-d entries ~ mass^(rank-d))."""
    return [[rng.gauss(0, 1) * mass ** (rank - d), rng.gauss(0, 1) * mass ** (rank - d)]
            for d in degrees(rank)]


def s_matrix(v, snap):
    """(V_i - V_j)^2 with listed exact values snapped (like MadLoop's S-matrix)."""
    n = len(v)
    out = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                x = mp(sub(v[i], v[j]), sub(v[i], v[j]))
                for target in snap:
                    if abs(x - target) <= 1e-9 * max(1.0, abs(target), abs(x)):
                        x = target
                out[i][j] = x
    return out


def lane(lane_id, v, masses, mu2, rng, rank, snap=(0.0,)):
    return {"id": lane_id, "shifts": v, "masses_squared": masses, "mu_squared": mu2,
            "s_matrix": s_matrix(v, snap), "tensor": tensor(rng, rank)}


def massless_pair(e, theta, phi):
    """Back-to-back massless momenta of energy e."""
    d = [math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)]
    return [e] + scale(d, e), [e] + scale(d, -e)


def chain(momenta):
    v = [[0.0] * 4]
    for p in momenta[:-1]:
        v.append(add(v[-1], p))
    return v


def rambo(rng, sqrt_s, count):
    """RAMBO: `count` massless momenta with total (sqrt_s, 0, 0, 0)."""
    q = []
    for _ in range(count):
        c = 2 * rng.random() - 1
        phi = 2 * math.pi * rng.random()
        e = -math.log(rng.random() * rng.random())
        st = math.sqrt(1 - c * c)
        q.append([e, e * st * math.cos(phi), e * st * math.sin(phi), e * c])
    total = [sum(k[i] for k in q) for i in range(4)]
    m = math.sqrt(mp(total, total))
    b = [-x / m for x in total[1:]]
    g = total[0] / m
    a = 1 / (1 + g)
    x = sqrt_s / m
    out = []
    for k in q:
        bq = sum(bi * ki for bi, ki in zip(b, k[1:]))
        out.append([x * (g * k[0] + bq)] + [x * (k[i + 1] + b[i] * k[0] + a * bq * b[i]) for i in range(3)])
    return out


def many(extended):
    """Seven and eight top propagators (2 -> 5 and 2 -> 6 massless final
    states): beyond the n <= 6 of the synthetic cases."""
    out = []
    for n, ranks in (((7, (8,)), (8, (8, 9))) if extended else ((7, (6, 7)),)):
        for rank in ranks:
            rng = random.Random(600 + 10 * n + rank)
            lanes = []
            for k, sqrt_s in enumerate([900.0, 2500.0]):
                p1, p2 = massless_pair(sqrt_s / 2, 0.0, 0.0)
                final = rambo(rng, sqrt_s, n - 2)
                v = chain([p1, p2] + [scale(f, -1.0) for f in final])
                lanes.append(lane(f"top-{k}", v, [[MT2, 0.0]] * n, MH2, rng, rank))
                masses = [MT2C if i % 3 == 0 else ([0.0, 0.0] if i % 3 == 1 else [MH2, 0.0]) for i in range(n)]
                lanes.append(lane(f"mixed-{k}", v, masses, MH2, rng, rank))
            out.append({"format": FORMAT, "name": f"edge-many-n{n}-r{rank}", "n": n, "rank": rank,
                        "lanes": lanes})
    return out


def low_ranks():
    """Low ranks (rank - n from -5 to -1, where the triangle and bubble stages
    and their subtractions vanish or are truncated), and single tadpoles."""
    out = []
    p1, p2 = massless_pair(225.0, 0.0, 0.0)
    final = rambo(random.Random(800), 450.0, 3)
    v5 = chain([p1, p2] + [scale(f, -1.0) for f in final])
    for n, ranks in ((5, (0, 1, 2, 3)), (3, (0, 1))):
        for rank in ranks:
            rng = random.Random(810 + 10 * n + rank)
            v = v5[:n] if n == 5 else chain([p1, p2, scale(add(p1, p2), -1.0)])
            lanes = [lane("top", v, [[MT2, 0.0]] * n, MH2, rng, rank),
                     lane("mixed", v, [MT2C] + [[0.0, 0.0], [MH2, 0.0]] * ((n - 1) // 2), MH2, rng, rank)]
            out.append({"format": FORMAT, "name": f"edge-low-rank-n{n}-r{rank}", "n": n, "rank": rank,
                        "lanes": lanes})
    for rank in (0, 1, 2):
        rng = random.Random(850 + rank)
        lanes = [lane("real", [[10.0, 20.0, -30.0, 5.0]], [[MT2, 0.0]], MH2, rng, rank),
                 lane("complex", [[0.0] * 4], [MT2C], MH2, rng, rank)]
        out.append({"format": FORMAT, "name": f"edge-tadpole-n1-r{rank}", "n": 1, "rank": rank,
                    "lanes": lanes})
    return out


def cases():
    out = many(False) + low_ranks()
    # Bubbles with |p^2| << m^2 (timelike and spacelike), unequal, massless
    # and complex masses: near-lightlike cut bases, and the regime where
    # Passarino-Veltman tensor bubbles cancel catastrophically.
    for rank in (1, 2, 3):
        rng = random.Random(700 + rank)
        lanes = []
        for p2 in (1e2, 1.0, 1e-2, -1e-2):
            # (E, 0, 0, z) with E^2 - z^2 = p2 at E = 200.
            e = 200.0
            z = math.sqrt(e * e - p2)
            v = [[0.0] * 4, [e, 0.0, 0.0, z]]
            for label, masses in (("tH", [[MT2, 0.0], [MH2, 0.0]]), ("t0", [[MT2, 0.0], [0.0, 0.0]]),
                                  ("0t", [[0.0, 0.0], MT2C])):
                lanes.append(lane(f"p2={p2:g}-{label}", v, masses, MH2, rng, rank, snap=()))
        out.append({"format": FORMAT, "name": f"edge-small-p2-n2-r{rank}", "n": 2, "rank": rank,
                    "lanes": lanes})
    # Tadpole reference bases: an uncut momentum along the first basis' e1
    # (lightlike) forces the second candidate; one along the second's e1 too
    # forces the third. Lightlike, massive bubbles.
    e1a, _ = basis_e1_e2(*TADPOLE_REFS[0])
    e1b, _ = basis_e1_e2(*TADPOLE_REFS[1])
    for rank in (1, 2, 3):
        rng = random.Random(100 + rank)
        lanes = [
            lane("second-basis", [[0.0] * 4, scale(e1a, 60.0)], [[MT2, 0.0], [MT2, 0.0]], MH2, rng, rank),
            lane("second-basis-complex", [[1.0, 2.0, 3.0, 4.0], add([1.0, 2.0, 3.0, 4.0], scale(e1a, -45.0))],
                 [MT2C, [0.0, 0.0]], MH2, rng, rank),
            lane("timelike", [[0.0] * 4, [300.0, 20.0, -10.0, 40.0]], [[MT2, 0.0], [MH2, 0.0]], MH2, rng, rank),
        ]
        out.append({"format": FORMAT, "name": f"edge-tadpole-bases-n2-r{rank}", "n": 2, "rank": rank,
                    "lanes": lanes})
    for rank in (2, 3, 4):
        rng = random.Random(200 + rank)
        lanes = [
            lane("third-basis", [[0.0] * 4, scale(e1a, 80.0), scale(e1b, -70.0)],
                 [[MT2, 0.0], [MT2, 0.0], [MT2, 0.0]], MH2, rng, rank),
            lane("third-basis-mixed", [[0.0] * 4, scale(e1a, 80.0), scale(e1b, -70.0)],
                 [MT2C, [0.0, 0.0], [MH2, 0.0]], MH2, rng, rank),
        ]
        out.append({"format": FORMAT, "name": f"edge-tadpole-bases-n3-r{rank}", "n": 3, "rank": rank,
                    "lanes": lanes})
    # Infrared-divergent massless triangle (two lightlike legs) and its
    # massive/complex variants; scaleless bubbles are skipped.
    for rank in (2, 3, 4):
        rng = random.Random(300 + rank)
        lanes = []
        for k, (theta, phi, sqrt_s) in enumerate([(0.3, 0.7, 250.0), (1.9, -2.1, 600.0)]):
            p1, p2 = massless_pair(sqrt_s / 2, theta, phi)
            v = chain([p1, p2, scale(add(p1, p2), -1.0)])
            lanes.append(lane(f"massless-{k}", v, [[0.0, 0.0]] * 3, MH2, rng, rank))
            lanes.append(lane(f"top-{k}", v, [[MT2, 0.0]] * 3, MH2, rng, rank))
            lanes.append(lane(f"mixed-complex-{k}", v, [[0.0, 0.0], MT2C, [0.0, 0.0]], MH2, rng, rank))
        out.append({"format": FORMAT, "name": f"edge-ir-triangle-n3-r{rank}", "n": 3, "rank": rank,
                    "lanes": lanes})
    # Massless box with lightlike legs (1/eps^2), and gg -> HH-like boxes
    # mixing massless, real and complex internal masses.
    for rank in (3, 4, 5):
        rng = random.Random(400 + rank)
        lanes = []
        for k, (theta, phi, sqrt_s) in enumerate([(0.4, 0.1, 400.0), (2.2, 1.3, 900.0)]):
            p1, p2 = massless_pair(sqrt_s / 2, 0.0, 0.0)
            k1, k2 = massless_pair(sqrt_s / 2, theta, phi)
            v = chain([p1, p2, scale(k1, -1.0), scale(k2, -1.0)])
            lanes.append(lane(f"massless-{k}", v, [[0.0, 0.0]] * 4, MH2, rng, rank))
            # Higgs pair: |k| from sqrt(s)/2 and mH.
            e = sqrt_s / 2
            pk = math.sqrt(e * e - MH2)
            d = [math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)]
            h1 = [e] + scale(d, pk)
            h2 = [e] + scale(d, -pk)
            v = chain([p1, p2, scale(h1, -1.0), scale(h2, -1.0)])
            lanes.append(lane(f"hh-top-{k}", v, [[MT2, 0.0]] * 4, MH2, rng, rank, snap=(0.0, MH2)))
            lanes.append(lane(f"hh-mixed-{k}", v, [MT2C, [0.0, 0.0], [MT2, 0.0], MT2C], MH2, rng, rank,
                              snap=(0.0, MH2)))
        out.append({"format": FORMAT, "name": f"edge-boxes-n4-r{rank}", "n": 4, "rank": rank,
                    "lanes": lanes})
    # Unstable points next to stable ones: coincident shifts (zero-momentum
    # bubble, singular triangle Gram determinant) and exactly collinear box
    # momenta; Ninja reports UNSTABLE_KINEMATICS (or a non-finite result).
    for rank in (2, 3, 4, 5):
        rng = random.Random(500 + rank)
        good = [[0.0] * 4, [200.0, 0.0, 0.0, 200.0], [400.0, 0.0, 0.0, 0.0], [130.0, 60.0, -80.0, -30.0]]
        coincident = [[0.0] * 4, [0.0] * 4, [400.0, 0.0, 0.0, 0.0], [130.0, 60.0, -80.0, -30.0]]
        collinear = [[0.0] * 4, [100.0, 0.0, 0.0, 50.0], [300.0, 0.0, 0.0, 150.0], [0.0, 0.0, 0.0, 0.0]]
        collinear[3] = [400.0, 0.0, 0.0, 200.0]
        masses = [[MT2, 0.0], [MH2, 0.0], [MT2, 0.0], MT2C]
        lanes = [
            lane("stable", good, masses, MH2, rng, rank),
            lane("coincident-shifts", coincident, masses, MH2, rng, rank),
            lane("stable-again", good, [MT2C, [0.0, 0.0], MT2C, [MH2, 0.0]], MH2, rng, rank),
            lane("collinear", collinear, masses, MH2, rng, rank),
        ]
        out.append({"format": FORMAT, "name": f"edge-unstable-n4-r{rank}", "n": 4, "rank": rank,
                    "lanes": lanes})
    return out


def main():
    args = sys.argv[1:]
    extended = "--extended" in args
    args = [a for a in args if a != "--extended"]
    target = pathlib.Path(args[0]) if args else \
        pathlib.Path(__file__).resolve().parent / "ninja_reference/cases"
    target.mkdir(parents=True, exist_ok=True)
    for case in (many(True) if extended else cases()):
        path = target / f"{case['name']}.json"
        header = json.dumps({k: v for k, v in case.items() if k != "lanes"})[:-1]
        lanes = ",\n".join(json.dumps(lane) for lane in case["lanes"])
        path.write_text(f'{header}, "lanes": [\n{lanes}\n]}}\n')
        print(path)


if __name__ == "__main__":
    main()
