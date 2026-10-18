#!/usr/bin/env python3
"""Size of the error of Ninja's mu-expansion convention (mu2 = tau^2 v^2, no shift) for the
box c4 at rank n+2 on reference-suite lanes with integrand mu^2 terms."""
import json
import sys
import mpmath as mp
import red
from red import FieldMp, set_field, Reduction
from e2e import build_numerator

for path, refpath in zip(sys.argv[1::2], sys.argv[2::2]):
    case = json.load(open(path))
    ref = json.load(open(refpath))
    F = FieldMp(seed=5, dps=50)
    set_field(F)
    for li, lane in enumerate(case['lanes']):
        V = [tuple(F.num(float(x)) for x in v) for v in lane['shifts']]
        m = [F.num(complex(c[0], c[1])) for c in lane['masses_squared']]
        N = build_numerator(lane, F)
        n = case['n']
        rl = ref['lanes'][li]
        scale = max(abs(mp.mpc(*rl[k])) for k in ('eps0', 'eps_m1', 'eps_m2'))
        out = {}
        for beta in (True, False):
            R = Reduction(n, V, m, N, beta_box=beta)
            for S in __import__('itertools').combinations(range(n), 5):
                R.pentagon(S)
            for S in __import__('itertools').combinations(range(n), 4):
                out.setdefault(S, {})[beta] = R.box(S).c[(2, (0,))]
        for S, d in out.items():
            rel = abs(d[False] - d[True]) / abs(d[True])
            shift = abs(d[False] - d[True]) / 6 / scale
            print(f"{case['name']} lane {li} box {S}: |c4(Ninja conv) - c4(exact)|/|c4| = {mp.nstr(rel, 3)}; "
                  f"shift of the integral (c4/6 term) / lane scale = {mp.nstr(shift, 3)}")
