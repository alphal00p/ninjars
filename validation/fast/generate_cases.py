#!/usr/bin/env python3
"""Generate synthetic "ninjars-bench-case-v1" inputs for the fast engine and
for original Ninja's benchmark driver.

Cases: n = 2..6 with ranks n-1, n, n+1 (random real kinematics with both
timelike and spacelike invariants; lanes alternate real and complex masses;
dense random complex tensors), plus `n5_r4_gghhh`: a gg -> HHH-like
pentagon with five equal top masses 173^2 and physical 2 -> 3 kinematics.

Propagators are D_i = (q + V_i)^2 - m_i^2; the tensor follows Ninja's
TensorNumerator / MadLoop ordering (crates/ninjars/src/fast/tensor_layout.rs).
Output is deterministic (fixed seeds).

Usage: generate_cases.py [output directory] (default: validation/fast/cases)
"""

import json
import math
import pathlib
import random
import sys

FORMAT = "ninjars-bench-case-v1"


def ncoeff(rank):
    return (rank + 1) * (rank + 2) * (rank + 3) * (rank + 4) // 24


def monomial_degrees(rank):
    """Total degree of every tensor entry, in layout order."""
    out = []
    for d in range(rank + 1):
        out.extend([d] * ((d + 1) * (d + 2) * (d + 3) // 6))
    return out


def mp(a, b):
    return a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]


def diff(a, b):
    return [x - y for x, y in zip(a, b)]


def planes_ok(v, allow_null=False, scale=1.0):
    """Every box/triangle basis plane must contain timelike directions
    (real Ninja bases) and invariants must be of both signs."""
    n = len(v)

    def ok(k1, k2):
        # Real bases need a non-spacelike plane; keep well away from the
        # degenerate (collinear) limit so binary64 reductions stay well
        # conditioned (original Ninja loses digits there as well).
        k12 = mp(k1, k2)
        product = mp(k1, k1) * mp(k2, k2)
        return k12 * k12 - product > 0.1 * max(k12 * k12, abs(product), 1e-2 * scale ** 2)

    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                if not ok(diff(v[a], v[c]), diff(v[b], v[a])):
                    return False
                for d in range(c + 1, n):
                    if not ok(diff(v[a], v[d]), diff(v[b], v[a])):
                        return False
    s = [mp(diff(v[i], v[j]), diff(v[i], v[j])) for i in range(n) for j in range(i + 1, n)]
    if n >= 3 and not (any(x > 0.05 for x in s) and any(x < -0.05 for x in s)):
        return False
    return allow_null or all(abs(x) > 0.05 for x in s)


def random_kinematics(rng, n):
    while True:
        v = [[0.0] * 4] + [[2 * rng.uniform(-1, 1)] + [rng.uniform(-1, 1) for _ in range(3)]
                           for _ in range(n - 1)]
        if planes_ok(v):
            return v


def s_matrix(v):
    n = len(v)
    return [[mp(diff(v[i], v[j]), diff(v[i], v[j])) for j in range(n)] for i in range(n)]


def tensor(rng, rank, scale=1.0):
    return [[rng.gauss(0, 1) * scale ** (rank - d), rng.gauss(0, 1) * scale ** (rank - d)]
            for d in monomial_degrees(rank)]


def synthetic_case(n, rank, lanes=16):
    rng = random.Random(1000 * n + rank)
    out = []
    for lane in range(lanes):
        v = random_kinematics(rng, n)
        if lane % 2 == 0:
            masses = [[rng.uniform(0.2, 1.2), 0.0] for _ in range(n)]
        else:
            masses = [[rng.uniform(0.2, 1.2), -rng.uniform(0.0, 0.2)] for _ in range(n)]
        out.append({"id": f"p{lane}", "shifts": v, "masses_squared": masses,
                    "mu_squared": rng.uniform(0.5, 2.0), "s_matrix": None,
                    "tensor": tensor(rng, rank)})
    return {"format": FORMAT, "name": f"n{n}_r{rank}", "n": n, "rank": rank, "lanes": out}


def two_to_three(rng, sqrt_s, masses):
    """Physical 2 -> 3 momenta (incoming along z) by rejection sampling of a
    uniform three-body decay of the s-channel system."""
    e = sqrt_s / 2
    p1 = [e, 0.0, 0.0, e]
    p2 = [e, 0.0, 0.0, -e]
    while True:
        # Random energies with momentum balance through two-step decay.
        m12 = rng.uniform(masses[0] + masses[1], sqrt_s - masses[2])

        def two_body(m, ma, mb):
            p = math.sqrt(max((m * m - (ma + mb) ** 2) * (m * m - (ma - mb) ** 2), 0)) / (2 * m)
            cos = rng.uniform(-1, 1)
            phi = rng.uniform(0, 2 * math.pi)
            sin = math.sqrt(1 - cos * cos)
            d = [p * sin * math.cos(phi), p * sin * math.sin(phi), p * cos]
            return ([math.sqrt(ma * ma + p * p)] + d, [math.sqrt(mb * mb + p * p)] + [-x for x in d])

        def boost(k, frame):
            b = [x / frame[0] for x in frame[1:]]
            b2 = sum(x * x for x in b)
            if b2 == 0:
                return k
            g = 1 / math.sqrt(1 - b2)
            bp = sum(x * y for x, y in zip(b, k[1:]))
            g2 = (g - 1) / b2
            return [g * (k[0] + bp)] + [k[i + 1] + g2 * bp * b[i] + g * b[i] * k[0] for i in range(3)]

        q12, k3 = two_body(sqrt_s, m12, masses[2])
        k1, k2 = two_body(m12, masses[0], masses[1])
        k1, k2 = boost(k1, q12), boost(k2, q12)
        return p1, p2, [k1, k2, k3]


def gghhh_case(lanes=16):
    rng = random.Random(4242)
    mt2, mh = 173.0 ** 2, 125.0
    out = []
    while len(out) < lanes:
        p1, p2, (k1, k2, k3) = two_to_three(rng, 1000.0, [mh, mh, mh])
        # Loop ordering g1, g2, H1, H2, H3 with incoming gluons.
        v = [[0.0] * 4]
        for p in [p1, p2, [-x for x in k1], [-x for x in k2]]:
            v.append([a + b for a, b in zip(v[-1], p)])
        if not planes_ok(v, allow_null=True, scale=1000.0):
            continue
        out.append({"id": f"p{len(out)}", "shifts": v,
                    "masses_squared": [[mt2, 0.0] for _ in range(5)],
                    "mu_squared": mh * mh, "s_matrix": None,
                    "tensor": tensor(rng, 4, scale=173.0)})
    return {"format": FORMAT, "name": "n5_r4_gghhh", "n": 5, "rank": 4, "lanes": out}


def main():
    root = pathlib.Path(__file__).resolve().parent
    target = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else root / "cases"
    target.mkdir(parents=True, exist_ok=True)
    cases = [synthetic_case(n, r) for n in range(2, 7) for r in (n - 1, n, n + 1)]
    cases.append(gghhh_case())
    for case in cases:
        path = target / f"{case['name']}.json"
        path.write_text(json.dumps(case, indent=1) + "\n")
        print(path)


if __name__ == "__main__":
    main()
