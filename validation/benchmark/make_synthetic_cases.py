#!/usr/bin/env python3
"""Generate deterministic synthetic ninjars-bench-case-v1 inputs.

Kinematics: two massless incoming momenta along +-z with sqrt(s), n-2 outgoing
momenta from recursive random two-body decays (so the loop invariants
(V_i-V_j)^2 mix timelike and spacelike values), with the external legs in a
random cyclic order around the loop. All momenta are taken incoming:
V_0 = 0, V_i = p_1 + ... + p_i. n=2 uses one random timelike or spacelike p.
Masses: real, complex (M^2 - i M Gamma, Im <= 0), zero, or mixed. Every case
except syn-gghhh-nosmat carries an explicit S-matrix whose rounding residues are
snapped to exact 0 / external masses squared (as MadLoop passes the S-matrix).
Numerators: dense random complex tensors in Ninja's TensorNumerator order
(rank blocks 0..R, see crates/ninjars/src/fast/tensor_layout.rs), the
degree-k block scaled by scale^-k so all ranks contribute comparably.

  python3 validation/benchmark/make_synthetic_cases.py [--output-dir DIR]
"""
import argparse
import json
import math
import pathlib
import random

HERE = pathlib.Path(__file__).resolve().parent
LANES = 3


def ncoeff(rank):
    return (rank + 1) * (rank + 2) * (rank + 3) * (rank + 4) // 24


def degree_of(index):
    r = 0
    while ncoeff(r) <= index:
        r += 1
    return r


def mdot(a, b):
    return a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]


def boost(p, q):
    """Boost p from the rest frame of q to the frame where q has momentum q."""
    m = math.sqrt(mdot(q, q))
    bx, by, bz = (q[1] / q[0], q[2] / q[0], q[3] / q[0])
    b2 = bx * bx + by * by + bz * bz
    if b2 == 0.0:
        return list(p)
    gamma = q[0] / m
    bp = bx * p[1] + by * p[2] + bz * p[3]
    g2 = (gamma - 1.0) / b2
    e = gamma * (p[0] + bp)
    return [e, p[1] + g2 * bp * bx + gamma * bx * p[0],
            p[2] + g2 * bp * by + gamma * by * p[0],
            p[3] + g2 * bp * bz + gamma * bz * p[0]]


def two_body(rng, q, m1, m2):
    mq = math.sqrt(mdot(q, q))
    pstar = math.sqrt(max(0.0, (mq * mq - (m1 + m2) ** 2) * (mq * mq - (m1 - m2) ** 2))) / (2 * mq)
    cos = rng.uniform(-1, 1)
    sin = math.sqrt(1 - cos * cos)
    phi = rng.uniform(0, 2 * math.pi)
    k = [pstar * sin * math.cos(phi), pstar * sin * math.sin(phi), pstar * cos]
    p1 = [math.sqrt(m1 * m1 + pstar * pstar), *k]
    p2 = [math.sqrt(m2 * m2 + pstar * pstar), *[-x for x in k]]
    return boost(p1, q), boost(p2, q)


def decay(rng, q, masses):
    if len(masses) == 1:
        return [list(q)]
    mq = math.sqrt(mdot(q, q))
    rest_min = sum(masses[1:])
    rest_max = mq - masses[0]
    m_rest = rest_min + rng.uniform(0.05, 0.95) * (rest_max - rest_min) if len(masses) > 2 else masses[1]
    a, b = two_body(rng, q, masses[0], m_rest)
    return [a] + decay(rng, b, masses[1:])


def external_momenta(rng, n, sqrt_s, out_masses, order):
    """Return n incoming momenta summing to zero, in loop order."""
    if n == 2:
        # p and -p, with p^2 timelike or spacelike (alternating by lane).
        e = rng.uniform(50, 150)
        p = [e, rng.uniform(-0.1, 0.1) * e, rng.uniform(-0.1, 0.1) * e, rng.uniform(0, 0.8) * e]
        return [p, [-x for x in p]]
    half = sqrt_s / 2
    incoming = [[half, 0.0, 0.0, half], [half, 0.0, 0.0, -half]]
    outgoing = decay(rng, [sqrt_s, 0.0, 0.0, 0.0], out_masses)
    legs = incoming + [[-x for x in k] for k in outgoing]
    return [legs[i] for i in order]


def shifts_from(legs):
    shifts = [[0.0, 0.0, 0.0, 0.0]]
    for p in legs[:-1]:
        shifts.append([a + b for a, b in zip(shifts[-1], p)])
    return shifts


def s_matrix(shifts, known, scale):
    """(V_i - V_j)^2, snapping floating-point residues onto exact known values.

    Invariants within 1e-9*scale of 0 or of an external mass squared (e.g. the
    -6e-12 left by massless legs) are replaced by that exact value, as MadLoop's
    S-matrix and Ninja's on-shell threshold would treat them; this keeps the
    masters well defined for libraries without an on-shell threshold.
    """
    n = len(shifts)
    out = []
    for i in range(n):
        row = []
        for j in range(n):
            d = [a - b for a, b in zip(shifts[i], shifts[j])]
            s = mdot(d, d)
            for value in known:
                if abs(s - value) <= 1e-9 * scale:
                    s = value
                    break
            row.append(s)
        out.append(row)
    return out


def random_tensor(rng, rank, scale):
    tensor = []
    for k in range(ncoeff(rank)):
        w = scale ** (-degree_of(k))
        tensor.append([rng.gauss(0, 1) * w, rng.gauss(0, 1) * w])
    return tensor


def make_case(name, n, rank, seed, masses, sqrt_s=700.0, out_masses=None, order=None,
              mu_squared=1.0e4, scale=100.0, with_s_matrix=True, note=""):
    rng = random.Random(seed)
    if out_masses is None:
        out_masses = [rng.choice([0.0, 125.0, 91.1876, 10.0]) for _ in range(max(0, n - 2))]
    lanes = []
    for lane in range(LANES):
        if n == 2:
            legs = external_momenta(rng, n, sqrt_s, out_masses, [0, 1])
            if lane % 2 == 1:   # spacelike lane
                p = legs[0]
                p[0] = 0.2 * abs(p[3])
                legs = [p, [-x for x in p]]
        else:
            loop_order = order if order is not None else [0] + rng.sample(range(1, n), n - 1)
            legs = external_momenta(rng, n, sqrt_s, out_masses, loop_order)
        shifts = shifts_from(legs)
        m2 = [list(m) for m in masses]
        lanes.append({
            "id": f"{name}-lane{lane}",
            "shifts": shifts,
            "masses_squared": m2,
            "mu_squared": mu_squared,
            "s_matrix": s_matrix(shifts, sorted({0.0, *[m * m for m in out_masses]}), sqrt_s ** 2)
                        if with_s_matrix else None,
            "tensor": random_tensor(rng, rank, scale),
        })
    return {"format": "ninjars-bench-case-v1", "name": name, "n": n, "rank": rank,
            "description": note, "generator": "validation/benchmark/make_synthetic_cases.py",
            "seed": seed, "lanes": lanes}


def cases():
    mt2 = 173.0 ** 2
    mt2c = [mt2, -173.0 * 1.5]
    out = []
    seed = 1000
    mass_patterns = {
        2: [[mt2, 0.0], [0.0, 0.0]],
        3: [mt2c, mt2c, mt2c],
        4: [[mt2, 0.0], [0.0, 0.0], [mt2, 0.0], [125.0 ** 2, 0.0]],
        5: [[mt2, 0.0]] * 5,
        6: [mt2c, [mt2, 0.0], [0.0, 0.0], mt2c, [mt2, 0.0], [80.4 ** 2, -80.4 * 2.1]],
    }
    kinds = {2: "real top + massless", 3: "complex top (Gamma=1.5)", 4: "mixed real/massless",
             5: "equal real top masses", 6: "mixed complex/real/massless"}
    for n in range(2, 7):
        for rank in (n - 1, n, n + 1):
            seed += 1
            out.append(make_case(f"syn-n{n}-r{rank}", n, rank, seed, mass_patterns[n],
                                 note=f"random kinematics, {kinds[n]}, dense random tensor"))
    # gg -> HHH-like pentagons: five equal top masses, sqrt(s)=450, MH=125.
    for rank in (4, 5):
        for label, order in (("ggHHH", [0, 1, 2, 3, 4]), ("gHgHH", [0, 2, 1, 3, 4])):
            seed += 1
            out.append(make_case(f"syn-gghhh-{label}-n5-r{rank}", 5, rank, seed, [[mt2, 0.0]] * 5,
                                 sqrt_s=450.0, out_masses=[125.0] * 3, order=order,
                                 note=f"gg->HHH-like pentagon kinematics (legs {label} around the loop), "
                                      "five equal real masses 173^2, dense random tensor"))
    # Control without S-matrix: Ninja computes (V_i-V_j)^2 itself (no snapping).
    seed += 1
    out.append(make_case("syn-gghhh-nosmat-n5-r4", 5, 4, seed, [[mt2, 0.0]] * 5, sqrt_s=450.0,
                         out_masses=[125.0] * 3, order=[0, 1, 2, 3, 4], with_s_matrix=False,
                         note="as syn-gghhh-ggHHH-n5-r4 kinematics family but s_matrix null "
                              "(Ninja computes the invariants from the shifts)"))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=pathlib.Path, default=HERE / "synthetic")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for case in cases():
        path = args.output_dir / f"{case['name']}.json"
        path.write_text(json.dumps(case, indent=1) + "\n")
        print(path)


if __name__ == "__main__":
    main()
