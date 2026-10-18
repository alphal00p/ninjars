#!/usr/bin/env python3
"""Generate the rank n+2 reference suite and its rank <= n+1 control twins.

Kinematic families (all momenta incoming, V_0 = 0 unless "shifted"):
  phys    : as validation/benchmark/make_synthetic_cases.py (sqrt(s) = 700, random
            two-body decays, random cyclic leg order), binary64 momenta, S-matrix with
            rounding residues snapped to exact 0 / external masses^2 (MadLoop-style).
  int     : physical 2 -> n-2 integer 4-momenta: every (V_i - V_j)^2 is an exact binary64
            integer, the S-matrix equals the shift invariants exactly.
  shifted : int kinematics with every shift translated by the same integer vector
            (V_0 != 0; a different routing of the same external momenta).
  irml    : massless propagators, all external legs massless and exactly on shell
            (integer null vectors): soft + collinear 1/eps^2 poles.
  irsoft  : one massless propagator between two legs exactly on shell w.r.t. the adjacent
            massive propagators (M^2 = E^2 - k^2 exactly): soft 1/eps poles.
  bub0    : n = 2 with p^2 = 0 exactly (equal or one zero mass).
Masses: real, complex (M^2 - i M Gamma), mixed (real/complex/zero), zero.
Numerators: dense random complex tensors in Ninja's TensorNumerator order, degree-d block
scaled by scale^-d (as make_synthetic_cases.py).  Cases without integrand mu^2 are
"ninjars-bench-case-v1"; cases with mu^2 ("-mu2": blocks mu^0, mu^2, mu^4; "-muall": every
power k <= rank/2) are "ninjars-bench-case-v2" (crates/ninjars-cli/src/export_bench.rs):
tensor = T_0 ++ T_1 ++ ... ++ T_K, N = sum_k (mu^2)^k T_k(q), T_k dense of rank rank - 2k.
Every random stream is that of the phase-1 generator (scratchpad gen_suite.py): the 66
phase-1 cases and their 162 twins are reproduced number for number (only the "-mu2" lanes
move from the phase-1 "mu2_tensors" extension field to the v2 layout).

  gen_suite.py --out DIR            rank n+2 suite (validation/rank-n2/cases)
  gen_suite.py --out DIR --controls rank <= n+1 twins (validation/rank-n2/controls/cases)
"""
import argparse
import json
import math
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from conditioning import cayley_min, gram_min  # noqa: E402

LANES = 3
MT, MH, MW, MZ = 173.0, 125.0, 80.4, 91.1876


def ncoeff(rank):
    return (rank + 1) * (rank + 2) * (rank + 3) * (rank + 4) // 24


def degree_of(index):
    r = 0
    while ncoeff(r) <= index:
        r += 1
    return r


def mdot(a, b):
    return a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]


# ---------------------------------------------------------------- phys (copied from make_synthetic_cases.py)
def boost(p, q):
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


def phys_legs(rng, n, sqrt_s, out_masses, order, lane):
    if n == 1:
        return [[0.0, 0.0, 0.0, 0.0]]
    if n == 2:
        e = rng.uniform(50, 150)
        p = [e, rng.uniform(-0.1, 0.1) * e, rng.uniform(-0.1, 0.1) * e, rng.uniform(0, 0.8) * e]
        if lane % 2 == 1:
            p[0] = 0.2 * abs(p[3])
        return [p, [-x for x in p]]
    half = sqrt_s / 2
    incoming = [[half, 0.0, 0.0, half], [half, 0.0, 0.0, -half]]
    outgoing = decay(rng, [sqrt_s, 0.0, 0.0, 0.0], out_masses)
    legs = incoming + [[-x for x in k] for k in outgoing]
    return [legs[i] for i in order]


def snapped_s_matrix(shifts, known, scale):
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


# ---------------------------------------------------------------- exact integer kinematics
def exact_s_matrix(shifts):
    n = len(shifts)
    out = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            d = [int(a) - int(b) for a, b in zip(shifts[i], shifts[j])]
            s = d[0] * d[0] - d[1] * d[1] - d[2] * d[2] - d[3] * d[3]
            assert abs(s) < 2 ** 53
            out[i][j] = float(s)
    return out


def shifts_from(legs):
    shifts = [[0, 0, 0, 0]]
    for p in legs[:-1]:
        shifts.append([a + b for a, b in zip(shifts[-1], p)])
    return shifts


def timelike(rng, lim, emin, emax):
    k = [rng.randint(-lim, lim) for _ in range(3)]
    return [math.isqrt(sum(x * x for x in k)) + 1 + rng.randint(emin, emax), *k]


def outgoing_closure(rng, incoming, n_out, lim=120):
    """n_out outgoing integer momenta (E > 0, timelike) with sum = sum(incoming); returned
    as incoming-convention legs (negated). None if the random closure is unphysical."""
    P = [sum(p[mu] for p in incoming) for mu in range(4)]
    outs = [timelike(rng, lim, 5, 120) for _ in range(n_out - 1)]
    last = [P[mu] - sum(q[mu] for q in outs) for mu in range(4)]
    if last[0] <= 0 or mdot(last, last) <= 0:
        return None
    return [[-x for x in q] for q in outs + [last]]


def int_legs(rng, n):
    """Physical integer kinematics: 2 -> n-2 with timelike integer momenta (n >= 3)."""
    if n == 1:
        return [[0, 0, 0, 0]]
    if n == 2:
        p = [rng.randint(-300, 300) for _ in range(4)]
        return [p, [-x for x in p]]
    while True:
        inc = [timelike(rng, 150, 50, 400), timelike(rng, 150, 50, 400)]
        outs = outgoing_closure(rng, inc, n - 2)
        if outs is not None:
            legs = inc + outs
            order = [0] + rng.sample(range(1, n), n - 1)
            return [legs[i] for i in order]


def lorentzian_ok(shifts):
    """Every plane spanned by V_j - V_i, V_k - V_i (i<j<k) is Lorentzian (2x2 Gram det < 0).
    Original Ninja builds real light-cone bases in these planes and returns NaN for a
    Euclidean (spacelike) plane, which unphysical random kinematics produce."""
    n = len(shifts)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = [x - y for x, y in zip(shifts[j], shifts[i])]
                b = [x - y for x, y in zip(shifts[k], shifts[i])]
                if mdot(a, a) * mdot(b, b) - mdot(a, b) ** 2 >= 0:
                    return False
    return True


_NULL = None


def null_vectors(lim=36):
    """Integer 3-vectors with integer Euclidean norm (Pythagorean quadruples), norm <= lim."""
    global _NULL
    if _NULL is None:
        out = []
        for x in range(-lim, lim + 1):
            for y in range(-lim, lim + 1):
                for z in range(-lim, lim + 1):
                    r2 = x * x + y * y + z * z
                    r = math.isqrt(r2)
                    if r * r == r2 and 0 < r <= lim:
                        out.append((r, x, y, z))
        _NULL = out
    return _NULL


def massless_legs(rng, n, tries=200000):
    """n exactly null integer momenta (2 incoming with E>0, n-2 outgoing with E<0) summing to 0."""
    nv = null_vectors()
    if n == 3:
        # two null legs, third leg off shell (s != 0): one-mass massless triangle
        while True:
            pa, pb = list(rng.choice(nv)), list(rng.choice(nv))
            s = [pa[mu] + pb[mu] for mu in range(4)]
            if mdot(s, s) != 0:
                return [pa, pb, [-x for x in s]]
    for _ in range(tries):
        outs = [list(rng.choice(nv)) for _ in range(n - 2)]
        K = [sum(k[mu] for k in outs) for mu in range(4)]
        K2 = mdot(K, K)
        if K2 <= 0 or K2 % 2:
            continue
        cands = [v for v in nv if mdot(K, v) * 2 == K2]
        rng.shuffle(cands)
        for pa in cands:
            pb = [K[mu] - pa[mu] for mu in range(4)]
            if pb[0] > 0 and mdot(pb, pb) == 0 and list(pa) != pb:
                legs = [list(pa), pb] + [[-a for a in k] for k in outs]
                # generic: no vanishing two-particle invariant between non-adjacent legs
                return legs
    raise RuntimeError("no massless configuration found")


def onshell_pair(rng, lim=200):
    """Two integer momenta p_a, p_b with p_a^2 = p_b^2 = M^2 > 0 exactly."""
    while True:
        ka = [rng.randint(-lim, lim) for _ in range(3)]
        ea = rng.randint(max(abs(x) for x in ka) + 30, 400)
        m2 = ea * ea - sum(x * x for x in ka)
        if m2 <= 400:
            continue
        for _ in range(20000):
            kb = [rng.randint(-lim, lim) for _ in range(3)]
            e2 = m2 + sum(x * x for x in kb)
            eb = math.isqrt(e2)
            if eb * eb == e2:
                return [ea, *ka], [eb, *kb], m2


# ---------------------------------------------------------------- masses
def mass_pattern(kind, n, rng):
    real = [MT ** 2, MH ** 2, MW ** 2, MZ ** 2, 10.0 ** 2]
    width = {MT ** 2: MT * 1.5, MW ** 2: MW * 2.1, MZ ** 2: MZ * 2.5, MH ** 2: MH * 0.004, 100.0: 0.1}
    out = []
    for i in range(n):
        if kind == "real":
            m2 = rng.choice(real)
            out.append([m2, 0.0])
        elif kind == "complex":
            m2 = rng.choice(real)
            out.append([m2, -width[m2]])
        elif kind == "mixed":
            m2 = rng.choice(real + [0.0])
            out.append([m2, -width[m2]] if (m2 and rng.random() < 0.5) else [m2, 0.0])
        elif kind == "zero":
            out.append([0.0, 0.0])
        else:
            raise ValueError(kind)
    return out


def random_tensor(rng, rank, scale):
    return [[rng.gauss(0, 1) * scale ** (-degree_of(k)), rng.gauss(0, 1) * scale ** (-degree_of(k))]
            for k in range(ncoeff(rank))]


MU2_CHOICES = [1.0e4, MZ ** 2, MT ** 2]


def lane_record(name, lane, shifts, m2, smat, rank, rng, scale, mu2_ext=None):
    # rng: the tensor stream (independent of the kinematics stream, see make_case)
    rec = {"id": f"{name}-lane{lane}", "shifts": [[float(x) for x in v] for v in shifts],
           "masses_squared": m2, "mu_squared": MU2_CHOICES[lane % len(MU2_CHOICES)],
           "s_matrix": smat, "tensor": random_tensor(rng, rank, scale)}
    if mu2_ext:
        # blocks k = 1..K in order (same draw order as the phase-1 "mu2_tensors" field)
        for k, r in mu2_ext:
            assert r == rank - 2 * k
            rec["tensor"] = rec["tensor"] + random_tensor(rng, r, scale)
    return rec


GRAM_MIN = 1e-4     # min normalised Gram determinant over all sub-topologies (k = 2..4 momenta)
CAYLEY_MIN = 1e-5   # min normalised modified-Cayley determinant (non-IR families only)


def well_conditioned(shifts, m2, ir):
    """Screen out near-degenerate points (see conditioning.py): COLLIER's and Ninja's
    reductions divide by these determinants, and binary64 references lose digits there."""
    if len(shifts) < 3:
        return True
    if gram_min(shifts)[0] < GRAM_MIN:
        return False
    return ir or cayley_min(shifts, m2)[0] >= CAYLEY_MIN


def make_case(family, n, rank, masses, seed, scale=100.0, mu2_ext=None, tag=""):
    name = f"rn2-{family}-{masses}-n{n}-r{rank}{tag}"
    # kinematics and masses depend on (seed, lane) only (identical for every rank of a
    # family), the tensors on (seed, rank, tag)
    rngt = random.Random(f"{seed}-{rank}-{tag}")
    lanes = []
    for lane in range(LANES):
        rng = random.Random(f"{seed}-lane{lane}")
        for attempt in range(100000):
            if family == "phys":
                out_masses = [rng.choice([0.0, MH, MZ, 10.0]) for _ in range(max(0, n - 2))]
                order = [0] + rng.sample(range(1, n), n - 1) if n >= 3 else list(range(n))
                legs = phys_legs(rng, n, 700.0, out_masses, order, lane)
                shifts = [[0.0, 0.0, 0.0, 0.0]]
                for p in legs[:-1]:
                    shifts.append([a + b for a, b in zip(shifts[-1], p)])
                m2 = mass_pattern(masses, n, rng)
                smat = snapped_s_matrix(shifts, sorted({0.0, *[m * m for m in out_masses]}), 700.0 ** 2)
                ir = False
            elif family in ("int", "shifted"):
                shifts = shifts_from(int_legs(rng, n))
                if family == "shifted":
                    w = [rng.randint(-150, 150) for _ in range(4)]
                    shifts = [[a + b for a, b in zip(v, w)] for v in shifts]
                m2 = mass_pattern(masses, n, rng)
                smat = exact_s_matrix(shifts)
                ir = False
            elif family == "irml":
                shifts = shifts_from(massless_legs(rng, n))
                sm = exact_s_matrix(shifts)
                # every non-adjacent invariant nonzero (no forward/backward degeneracy)
                if not all(sm[i][j] != 0 for i in range(n) for j in range(n)
                           if min((i - j) % n, (j - i) % n) >= 2):
                    continue
                m2 = mass_pattern("zero", n, rng)
                smat = sm
                ir = True
            elif family == "irsoft":
                pa, pb, M2 = onshell_pair(rng)   # two incoming legs on shell (p^2 = M^2 exactly)
                outs = outgoing_closure(rng, [pa, pb], n - 2)
                if outs is None:
                    continue
                legs = [pa, pb] + [outs[i] for i in rng.sample(range(n - 2), n - 2)]
                shifts = shifts_from(legs)
                m2 = [[float(M2), 0.0], [0.0, 0.0], [float(M2), 0.0]] + mass_pattern("real", n - 3, rng)
                smat = exact_s_matrix(shifts)
                ir = True
            elif family == "bub0":
                e, x, y, z = rng.choice(null_vectors())
                k = rng.randint(1, 8)
                shifts = [[0, 0, 0, 0], [k * e, k * x, k * y, k * z]]
                m2 = [[MT ** 2, 0.0], [MT ** 2, 0.0]] if masses == "real" else \
                     [[MT ** 2, -MT * 1.5], [MT ** 2, -MT * 1.5]] if masses == "complex" else [[0.0, 0.0], [MT ** 2, 0.0]]
                smat = exact_s_matrix(shifts)
                ir = False
            else:
                raise ValueError(family)
            if lorentzian_ok(shifts) and well_conditioned(shifts, m2, ir):
                break
        else:
            raise RuntimeError(f"{name} lane {lane}: no acceptable point")
        rec = lane_record(name, lane, shifts, m2, smat, rank, rngt, scale, mu2_ext)
        rec["generator_attempts"] = attempt + 1
        lanes.append(rec)
    case = {"format": "ninjars-bench-case-v1", "name": name, "n": n, "rank": rank}
    if mu2_ext:
        assert [k for k, _ in mu2_ext] == list(range(1, len(mu2_ext) + 1))
        case.update({"format": "ninjars-bench-case-v2", "mu_powers": len(mu2_ext), "r2": "synthetic",
                     "tensor_layout": TENSOR_LAYOUT_V2})
    case.update({"family": family, "masses": masses,
                 "description": f"rank-n+2 reference suite: family {family}, masses {masses}"
                                + (f", integrand mu^2 powers 0..{len(mu2_ext)}" if mu2_ext else ""),
                 "generator": "validation/rank-n2/harness/gen_suite.py", "seed": seed,
                 "screening": {"lorentzian_planes": True, "gram_min": GRAM_MIN, "cayley_min": CAYLEY_MIN},
                 "lanes": lanes})
    return case


TENSOR_LAYOUT_V2 = ("blocks k = 0..=mu_powers of mu^(2k) T_k(q), T_k a rank-(rank-2k) tensor in "
                    "ninjars::fast::tensor_layout order")


def specs():
    """(family, n, masses) kinematics of the suite, with their seeds (phase-1 order)."""
    out = []
    for n in range(1, 7):
        for masses in ("real", "complex", "mixed"):
            out.append(("phys", n, masses))
            out.append(("int", n, masses))
        out.append(("shifted", n, "mixed"))
    for n in range(3, 7):
        out.append(("irml", n, "zero"))
        out.append(("irsoft", n, "real"))
    out.append(("int", 2, "zero"))
    for masses in ("real", "complex", "mixed"):
        out.append(("bub0", 2, masses))
    return {spec: 20261018 + i for i, spec in enumerate(out)}


def mu_blocks(rank, K):
    """[(k, rank - 2k)] for k = 1..K (K <= rank / 2)."""
    return [(k, rank - 2 * k) for k in range(1, K + 1) if rank - 2 * k >= 0]


def suite_plan():
    seeds = specs()
    out = []
    # (1) mu-free rank n+2 on every kinematics family (phase-1 cases)
    for (family, n, masses), seed in seeds.items():
        out.append(make_case(family, n, n + 2, masses, seed))
    # (2) phase-1 "-mu2" cases: int kinematics, blocks mu^2 x rank n, mu^4 x rank n-2
    for n in range(1, 7):
        for masses in ("real", "complex"):
            out.append(make_case("int", n, n + 2, masses, seeds[("int", n, masses)],
                                 mu2_ext=mu_blocks(n + 2, 2), tag="-mu2"))
    # (3) new "-mu2" cases on the other families
    extra = [("phys", n, "complex") for n in range(1, 7)] + [("shifted", n, "mixed") for n in range(1, 7)] \
        + [("bub0", 2, m) for m in ("real", "complex", "mixed")] + [("irml", n, "zero") for n in range(3, 7)]
    for family, n, masses in extra:
        out.append(make_case(family, n, n + 2, masses, seeds[(family, n, masses)],
                             mu2_ext=mu_blocks(n + 2, 2), tag="-mu2"))
    # (4) every mu power (mu^6 from n = 4, mu^8 at n = 6)
    for n in range(4, 7):
        for family, masses in (("int", "complex"), ("int", "real"), ("shifted", "mixed"), ("phys", "complex")):
            out.append(make_case(family, n, n + 2, masses, seeds[(family, n, masses)],
                                 mu2_ext=mu_blocks(n + 2, (n + 2) // 2), tag="-muall"))
    return out


def control_plan():
    """Rank <= n+1 twins on exactly the suite kinematics (Ninja-checkable)."""
    seeds = specs()
    out = []
    # phase-1 twins: ranks n-1, n, n+1 of every mu-free kinematics family
    for (family, n, masses), seed in seeds.items():
        for off in (-1, 0, 1):
            if n + off >= 0:
                out.append(make_case(family, n, n + off, masses, seed))
    # mu^2 twins at rank n+1: mu^2 x rank n-1 (rank counts mu^2 twice)
    for n in range(1, 7):
        for family, masses in (("int", "real"), ("int", "complex"), ("shifted", "mixed")):
            out.append(make_case(family, n, n + 1, masses, seeds[(family, n, masses)],
                                 mu2_ext=mu_blocks(n + 1, 1), tag="-mu2"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--controls", action="store_true", help="rank <= n+1 control twins instead of the suite")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for case in (control_plan() if args.controls else suite_plan()):
        (args.out / f"{case['name']}.json").write_text(json.dumps(case, indent=1) + "\n")
        print(case["name"])


if __name__ == "__main__":
    main()
