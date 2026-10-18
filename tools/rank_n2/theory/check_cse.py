"""Check that the CSE-optimised code in subtractions_rank_n2_cse.txt reproduces the
per-monomial closed forms (random complex inputs)."""
import cmath
import os
import random
import re

import sympy as sp

import subtraction_formulas as SF

HERE = os.path.dirname(os.path.abspath(__file__))


def blocks():
    text = open(os.path.join(HERE, "subtractions_rank_n2_cse.txt")).read()
    out = {}
    cur = None
    for line in text.split("\n"):
        m = re.match(r"// ---------------- (\w+):", line)
        if m:
            cur = m.group(1)
            out[cur] = []
            continue
        if cur and line.strip() and not line.startswith("//"):
            out[cur].append(line)
    return out


def run_block(lines, env):
    loc = dict(env)
    loc["pow"] = lambda a, b: a ** b
    loc["out"] = {}
    for l in lines:
        l = re.sub(r"out\[\[(\d+), (\d+), (\d+)\]\]", r"out[(\1,\2,\3)]", l).rstrip(";")
        exec(l, {}, loc)
    return loc["out"]


def main(seed=5):
    rng = random.Random(seed)
    rc = lambda: complex(rng.uniform(-1, 1), rng.uniform(-1, 1))
    bl = blocks()
    worst = 0
    # tri_to_bub
    env = {f"{n}_{s}": rc() for n in ("U1", "U2", "U3", "U4") for s in (3, 4)}
    env.update({k: rc() for k in ("K1", "K2", "K3", "K4", "f", "P0", "P1", "P2")})
    env.update({f"c{i}": rc() for i in range(15, 35)})
    got = run_block(bl["tri_to_bub"], env)
    ref = {}
    for idx in range(15, 21):
        F, which = SF.tri_formula(idx, "bub")
        suf = "3" if which == "et3" else "4"
        sub = {SF.U1: env["U1_" + suf], SF.U2: env["U2_" + suf], SF.U3: env["U3_" + suf], SF.U4: env["U4_" + suf],
               SF.K1: env["K1"], SF.K2: env["K2"], SF.K3: env["K3"], SF.K4: env["K4"], SF.f: env["f"],
               SF.P0: env["P0"], SF.P1: env["P1"], SF.P2: env["P2"]}
        for s, e in F.items():
            ref[s] = ref.get(s, 0) + env[f"c{idx}"] * complex(e.evalf(subs=sub))
    for s in set(ref) | set(got):
        worst = max(worst, abs(ref.get(s, 0) - got.get(s, 0)) / (1 + abs(ref.get(s, 0))))
    # tri_to_tad
    env2 = {f"{n}_{s}": rc() for n in ("U3", "U4") for s in (3, 4)}
    env2.update({k: rc() for k in ("A0", "B0", "A1", "B1", "f0", "f1", "P")})
    env2.update({f"c{i}": rc() for i in range(15, 35)})
    got = run_block(bl["tri_to_tad"], env2)
    ref = {}
    for idx in range(15, 21):
        F, which = SF.tri_formula(idx, "tad")
        suf = "3" if which == "et3" else "4"
        sub = {SF.U3: env2["U3_" + suf], SF.U4: env2["U4_" + suf], SF.A0: env2["A0"], SF.B0: env2["B0"],
               SF.A1: env2["A1"], SF.B1: env2["B1"], SF.f0: env2["f0"], SF.f1: env2["f1"], SF.P: env2["P"]}
        for s, e in F.items():
            ref[s] = ref.get(s, 0) + env2[f"c{idx}"] * complex(e.evalf(subs=sub))
    for s in set(ref) | set(got):
        worst = max(worst, abs(ref.get(s, 0) - got.get(s, 0)) / (1 + abs(ref.get(s, 0))))
    # bub_to_tad
    env3 = {k: rc() for k in ("S2", "S3", "S4", "T2", "T3", "T4", "W", "A", "B", "f", "P")}
    env3.update({f"c{i}": rc() for i in range(20, 35)})
    got = run_block(bl["bub_to_tad"], env3)
    ref = {}
    sub = {getattr(SF, k): v for k, v in env3.items() if not k.startswith("c")}
    for idx in range(20, 35):
        for s, e in SF.bub_formula(idx).items():
            ref[s] = ref.get(s, 0) + env3[f"c{idx}"] * complex(e.evalf(subs=sub))
    for s in set(ref) | set(got):
        worst = max(worst, abs(ref.get(s, 0) - got.get(s, 0)) / (1 + abs(ref.get(s, 0))))
    print(f"CSE code vs closed forms: worst rel. difference {worst:.2e} {'PASS' if worst < 1e-12 else 'FAIL'}")


if __name__ == "__main__":
    main()
