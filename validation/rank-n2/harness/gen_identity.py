#!/usr/bin/env python3
"""Exact identity and invariance cases for the rank n+2 tests (DESIGN.md section 7).

Every input is a small dyadic rational (integer momenta |V^mu| <~ 40, dyadic masses,
numerator coefficients k 2^-(2d), |k| < 128, for degree d), so that every derived case
below is EXACT in binary64: poly_to_tensor() raises if a single coefficient is not.
Writes validation/rank-n2/identity/cases/*.json and identity/manifest.json.

  V2   rn2-exact-q-n{n}-r{n+2}      q-only numerator of full rank n+2       (v1, 2 lanes)
       rn2-exact-mu-n{n}-r{n+2}     every mu power k <= (n+2)/2, full rank  (v2, 2 lanes)
  V7a  ...-shift                    V_i -> V_i - P, N'(q, mu^2) = N(q - P, mu^2), P per lane
  V7b  ...-perm                     lane k = lane 0 with the propagators relabelled by
                                    sigma_k (all 2n dihedral orders for n <= 4, the identity
                                    and 7 random ones for n >= 5); same integral
  V6a  rn2-id-pinch-n{n}           lane j: D_j M_j, M_j of rank n            (v2, n lanes)
       rn2-id-pinch-n{n}-sub       lane j: M_j on the n-1 propagators != j   (v1, n lanes)
                                    I_n[D_j M_j] = I_(n-1)[M_j]
  V6b  rn2-id-decomp-n{n}          N = R + sum_i D_i B_i + sum_(i<j) D_i D_j C_ij       (v2)
       rn2-id-decomp-n{n}-R/-B/-C  the pieces, every one of rank <= N+1 (v1); I[N] = sum
with D_i = (q + V_i)^2 - m_i^2 - mu^2 (mu^2 the integrand mu^2).  n = 1..7 for V2/V7,
n = 2..7 for V6 (V6b: C_ij from n = 3).  The n = 7 cases (S5b fix-up) come last in
every manifest list and use their own seeds, so every n <= 6 case is unchanged.

    gen_identity.py --out validation/rank-n2/identity"""
import argparse
import itertools
import json
import math
import pathlib
import random
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import collier_reference as cr  # noqa: E402
from gen_suite import lorentzian_ok, well_conditioned, exact_s_matrix, shifts_from, mdot, timelike  # noqa: E402

Q, cq = cr.Q, cr.cq
LAYOUT_V2 = ("blocks k = 0..=mu_powers of mu^(2k) T_k(q), T_k a rank-(rank-2k) tensor in "
             "ninjars::fast::tensor_layout order")
MASSES = {"real": [[9.0, 0.0], [16.0, 0.0], [25.0, 0.0], [49.0, 0.0], [100.0, 0.0]],
          "complex": [[25.0, -1.25], [49.0, -0.5], [100.0, -2.5], [16.0, -0.75], [36.0, -1.5]]}
MU_R2 = 64.0


def small_legs(rng, n):
    if n == 1:
        return [[0, 0, 0, 0]]
    if n == 2:
        p = [rng.randint(-9, 9) for _ in range(4)]
        return [p, [-x for x in p]]
    while True:
        inc = [timelike(rng, 8, 1, 10), timelike(rng, 8, 1, 10)]
        P = [sum(p[mu] for p in inc) for mu in range(4)]
        outs = [timelike(rng, 5, 1, 6) for _ in range(n - 3)]
        last = [P[mu] - sum(q[mu] for q in outs) for mu in range(4)]
        if last[0] <= 0 or mdot(last, last) <= 0:
            continue
        legs = inc + [[-x for x in q] for q in outs + [last]]
        order = [0] + rng.sample(range(1, n), n - 1)
        return [legs[i] for i in order]


def kinematics(rng, n, masses):
    for _ in range(100000):
        shifts = shifts_from(small_legs(rng, n))
        m2 = [list(rng.choice(MASSES[masses if masses != "mixed" else rng.choice(["real", "complex"])]))
              for _ in range(n)]
        if lorentzian_ok(shifts) and well_conditioned(shifts, m2, False):
            return [[float(x) for x in v] for v in shifts], m2
    raise RuntimeError("no kinematics")


def lane_dict(lid, shifts, m2, tensor):
    return {"id": lid, "shifts": shifts, "masses_squared": m2, "mu_squared": MU_R2,
            "s_matrix": exact_s_matrix(shifts), "tensor": tensor}


def rand_poly(rng, rank, wbits=2):
    """Dense random polynomial: coefficient (a + i b) 2^(-wbits d), |a|, |b| < 128."""
    out = {}
    for m in cr.monomials(rank):
        d = sum(m)
        c = cq(Q(rng.randint(-127, 127), 2 ** (wbits * d)), Q(rng.randint(-127, 127), 2 ** (wbits * d)))
        if not cr.is_zero(c):
            out[m] = c
    return out


def tensor_v2(blocks, rank):
    """[(k, poly)] -> concatenated v2 tensor (exactness checked)."""
    out = []
    K = max(k for k, _ in blocks)
    by = dict(blocks)
    for k in range(K + 1):
        out += cr.poly_to_tensor(by.get(k, {}), rank - 2 * k)
    return out, K


def case_doc(name, n, rank, lanes, desc, K=0):
    d = {"format": "ninjars-bench-case-v1", "name": name, "n": n, "rank": rank}
    if K:
        d.update({"format": "ninjars-bench-case-v2", "mu_powers": K, "r2": "synthetic", "tensor_layout": LAYOUT_V2})
    d.update({"description": desc, "generator": "validation/rank-n2/harness/gen_identity.py", "exact_binary64": True,
              "lanes": lanes})
    return d


def A(shift, m2):
    return cr.a_poly(tuple(Q(x) for x in shift), cq(m2[0], m2[1]))


def shift_blocks(blocks, P):
    """N(q - P) for every mu block (P exact)."""
    return [(k, cr.pshift(p, P)) for k, p in blocks]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()
    cdir = a.out / "cases"
    cdir.mkdir(parents=True, exist_ok=True)
    manifest = {"format": "ninjars-rank-n2-identities-v1",
                "description": "exact identity and invariance cases (gen_identity.py); lanes are referenced "
                               "by case name and lane index; every identity is exact in exact arithmetic",
                "V2": [], "V7a": [], "V7b": [], "V6a": [], "V6b": []}
    docs = []
    for n in range(1, 8):
        rank = n + 2
        for kind in ("q", "mu"):
            rng = random.Random(f"rn2-exact-{kind}-n{n}")
            name = f"rn2-exact-{kind}-n{n}-r{rank}"
            K = rank // 2 if kind == "mu" else 0
            base_lanes, blocks_per_lane = [], []
            for lane in range(2):
                shifts, m2 = kinematics(rng, n, "complex" if lane == 0 else "mixed")
                blocks = [(0, rand_poly(rng, rank))] + [(k, rand_poly(rng, rank - 2 * k)) for k in range(1, K + 1)]
                t, _ = tensor_v2(blocks, rank) if K else (cr.poly_to_tensor(blocks[0][1], rank), 0)
                base_lanes.append(lane_dict(f"{name}-lane{lane}", shifts, m2, t))
                blocks_per_lane.append(blocks)
            docs.append(case_doc(name, n, rank, base_lanes, f"V2: exact dyadic {kind} numerator of full rank n+2", K))
            manifest["V2"].append({"case": name, "n": n, "rank": rank, "mu_powers": K})
            # V7a: shift V_i -> V_i - P, N'(q) = N(q - P)
            sl = []
            for lane, (ld, blocks) in enumerate(zip(base_lanes, blocks_per_lane)):
                P = [rng.randint(-2, 2) for _ in range(4)]
                Pq = tuple(Q(x) for x in P)
                nb = shift_blocks(blocks, Pq)
                t, _ = tensor_v2(nb, rank) if K else (cr.poly_to_tensor(nb[0][1], rank), 0)
                shifts = [[float(x - p) for x, p in zip(v, P)] for v in ld["shifts"]]
                sl.append(lane_dict(f"{name}-shift-lane{lane}", shifts, ld["masses_squared"], t))
                manifest["V7a"].append({"case": f"{name}-shift", "lane": lane, "equals": {"case": name, "lane": lane},
                                        "P": P})
            docs.append(case_doc(f"{name}-shift", n, rank, sl, "V7a: V_i - P with N(q - P) (exact)", K))
            # V7b: relabelling
            if n >= 2:
                if n <= 4:
                    perms = [tuple((s * i + r) % n for i in range(n)) for s in (1, -1) for r in range(n)]
                else:
                    prng = random.Random(f"perm-{name}")
                    perms = [tuple(range(n))]
                    while len(perms) < 8:
                        p = tuple(prng.sample(range(n), n))
                        if p not in perms:
                            perms.append(p)
                src = base_lanes[0]
                pl = []
                for k, sg in enumerate(perms):
                    sh = [src["shifts"][i] for i in sg]
                    pl.append(lane_dict(f"{name}-perm-lane{k}", sh, [src["masses_squared"][i] for i in sg], src["tensor"]))
                    manifest["V7b"].append({"case": f"{name}-perm", "lane": k, "sigma": list(sg),
                                            "equals": {"case": name, "lane": 0}})
                docs.append(case_doc(f"{name}-perm", n, rank, pl,
                                     "V7b: lane k = lane 0 of the base case with propagator i -> sigma_k(i)", K))
    # V6a pinch
    for n in range(2, 8):
        rng = random.Random(f"rn2-id-pinch-n{n}")
        shifts, m2 = kinematics(rng, n, "complex")
        lanes, subs = [], []
        for j in range(n):
            M = rand_poly(rng, n)
            T0 = cr.pmul(A(shifts[j], m2[j]), M)
            T1 = {mm: cr.cneg(c) for mm, c in M.items()}
            t, K = tensor_v2([(0, T0), (1, T1)], n + 2)
            lanes.append(lane_dict(f"rn2-id-pinch-n{n}-lane{j}", shifts, m2, t))
            keep = [i for i in range(n) if i != j]
            sh = [shifts[i] for i in keep]
            subs.append(lane_dict(f"rn2-id-pinch-n{n}-sub-lane{j}", sh, [m2[i] for i in keep], cr.poly_to_tensor(M, n)))
            manifest["V6a"].append({"lhs": {"case": f"rn2-id-pinch-n{n}", "lane": j},
                                    "rhs": [{"case": f"rn2-id-pinch-n{n}-sub", "lane": j, "coefficient": 1}],
                                    "pinched": j})
        docs.append(case_doc(f"rn2-id-pinch-n{n}", n, n + 2, lanes, "V6a: lane j numerator D_j M_j (rank n+2)", 1))
        docs.append(case_doc(f"rn2-id-pinch-n{n}-sub", n - 1, n, subs,
                             "V6a: lane j numerator M_j on the propagators != j (rank (n-1)+1)"))
    # V6b decomposition
    for n in range(2, 8):
        rng = random.Random(f"rn2-id-decomp-n{n}")
        name = f"rn2-id-decomp-n{n}"
        full, Rl, Bl, Cl = [], [], [], []
        pairs = list(itertools.combinations(range(n), 2)) if n >= 3 else []
        for lane in range(2):
            shifts, m2 = kinematics(rng, n, "complex" if lane == 0 else "mixed")
            Ai = [A(shifts[i], m2[i]) for i in range(n)]
            R = rand_poly(rng, n + 1)
            B = [rand_poly(rng, n) for _ in range(n)]
            C = [rand_poly(rng, n - 2, wbits=1) for _ in pairs]
            T0, T1, T2 = dict(R), {}, {}
            for i in range(n):
                cr.padd(T0, cr.pmul(Ai[i], B[i]))
                cr.padd(T1, B[i], cq(-1))
            for (i, j), Cij in zip(pairs, C):
                cr.padd(T0, cr.pmul(cr.pmul(Ai[i], Ai[j]), Cij))
                cr.padd(T1, cr.pmul(cr.padd(dict(Ai[i]), Ai[j]), Cij), cq(-1))
                cr.padd(T2, Cij)
            blocks = [(0, cr.pclean(T0)), (1, cr.pclean(T1))] + ([(2, cr.pclean(T2))] if pairs else [])
            t, K = tensor_v2(blocks, n + 2)
            full.append(lane_dict(f"{name}-lane{lane}", shifts, m2, t))
            rhs = [{"case": f"{name}-R", "lane": lane, "coefficient": 1}]
            Rl.append(lane_dict(f"{name}-R-lane{lane}", shifts, m2, cr.poly_to_tensor(R, n + 1)))
            for i in range(n):
                keep = [k for k in range(n) if k != i]
                Bl.append(lane_dict(f"{name}-B-lane{len(Bl)}", [shifts[k] for k in keep], [m2[k] for k in keep],
                                    cr.poly_to_tensor(B[i], n)))
                rhs.append({"case": f"{name}-B", "lane": len(Bl) - 1, "coefficient": 1, "pinched": [i]})
            for (i, j), Cij in zip(pairs, C):
                keep = [k for k in range(n) if k not in (i, j)]
                Cl.append(lane_dict(f"{name}-C-lane{len(Cl)}", [shifts[k] for k in keep], [m2[k] for k in keep],
                                    cr.poly_to_tensor(Cij, n - 2)))
                rhs.append({"case": f"{name}-C", "lane": len(Cl) - 1, "coefficient": 1, "pinched": [i, j]})
            manifest["V6b"].append({"lhs": {"case": name, "lane": lane}, "rhs": rhs})
        K = 2 if pairs else 1
        docs.append(case_doc(name, n, n + 2, full, "V6b: N = R + sum D_i B_i + sum D_i D_j C_ij (rank n+2)", K))
        docs.append(case_doc(f"{name}-R", n, n + 1, Rl, "V6b piece R (rank n+1)"))
        docs.append(case_doc(f"{name}-B", n - 1, n, Bl, "V6b pieces B_i on the propagators != i (rank n)"))
        if Cl:
            docs.append(case_doc(f"{name}-C", n - 2, n - 2, Cl, "V6b pieces C_ij on the propagators != i, j"))
    for d in docs:
        (cdir / f"{d['name']}.json").write_text(json.dumps(d, indent=1) + "\n")
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"{len(docs)} cases; identities: " + ", ".join(f"{k} {len(v)}" for k, v in manifest.items() if isinstance(v, list)))


if __name__ == "__main__":
    main()
