"""Count which Laurent slots each NEW parent coefficient feeds (from the definition, exact)."""
import red
from red import FieldFp, set_field, Numerator, Reduction, Residue, dot, vadd, vsub, vsc, ldiv
f = FieldFp(5); set_field(f)
n = 3
V = [tuple(f.rand() for _ in range(4)) for _ in range(n)]; m = [f.rand() for _ in range(n)]
R = Reduction(n, V, m, Numerator.random(5))
T = (0, 1, 2); _, _, _, a_t, et3, et4 = R.triangle_geometry(T)
newtri = {'c15': (0, (5, 0)), 'c16': (0, (0, 5)), 'c17': (1, (3, 0)), 'c18': (1, (0, 3)), 'c19': (2, (1, 0)), 'c20': (2, (0, 1))}
def bub_slots(res, B, l, fam):
    V0, m0, uncut, e1, e2, e3, e4, a0, a1, mp12 = R.bubble_geometry(B); mp34 = dot(e3, e4)
    g3, g4 = (e3, e4) if fam == 'A' else (e4, e3)
    kk, fl = R.lin_den(B, l)
    # polynomial in (x, M) per level via fit on the layout monomials
    samples = []
    for _ in range(12):
        x, M = f.rand(), f.rand()
        P0v = vadd(a0, vsc(x, a1)); beta = m0 - dot(P0v, P0v); P2v = vsc((beta + M) / (2 * mp34), g4)
        g = res.series(vsub(P0v, V0), g3, P2v, M, 1)
        samples.append(((x, M), ldiv(g, 2 * dot(g3, kk), 2 * dot(P0v, kk) + fl, 2 * dot(P2v, kk), 0)))
    lv = R.fit_levels(samples, range(5), lambda b: [(a, j) for j in range((4 - b) // 2 + 1) for a in range(4 - b - 2 * j + 1)])
    return sorted((b, a, j) for b, d in lv.items() for (a, j), v in d.items() if not f.iszero(v))
def tad_slots(res, C, extra):
    e1t, e2t, e3t, e4t = R.tadbasis_common
    V0, m0, _, _, _ = R.geom(C); mp = dot(e1t, e2t); e4eff = vsc(1 / (2 * mp), e2t)
    samples = []
    for _ in range(6):
        M = f.rand(); P2v = vsc(m0 + M, e4eff); zero = tuple(f.zero for _ in range(4))
        g = res.series(vsub(zero, V0), e1t, P2v, M, len(extra)); mm = len(extra)
        for l in extra:
            kk, fl = R.lin_den(C, l); g = ldiv(g, 2 * dot(e1t, kk), fl, 2 * dot(P2v, kk), mm - 1); mm -= 1
        samples.append(((M,), g))
    lv = R.fit_levels(samples, range(4), lambda b: [(j,) for j in range((3 - b) // 2 + 1)])
    return sorted((b, j) for b, d in lv.items() for (j,), v in d.items() if not f.iszero(v))
for name, key in newtri.items():
    res = Residue(T, 3, 5, V[0], [et3, et4]); res.c = {key: f.rand()}
    sb = bub_slots(res, (0, 1), 2, 'A'); st = tad_slots(res, (0,), [1, 2])
    print(f"triangle {name}: feeds {len(sb)} bubble-A slots; tadpole slots {st}")
V0b, m0b, _, e1b, e2b, e3b, e4b, a0b, a1b, _ = R.bubble_geometry((0, 1))
newbub = {'c20': (0, (4, 0, 0)), 'c21': (1, (2, 0, 0)), 'c22': (2, (0, 0, 0)), 'c23': (0, (0, 4, 0)), 'c25': (0, (1, 3, 0)), 'c27': (0, (2, 2, 0)), 'c29': (0, (3, 1, 0)), 'c31': (1, (0, 2, 0)), 'c33': (1, (1, 1, 0))}
for name, key in newbub.items():
    res = Residue((0, 1), 2, 4, V0b, [e2b, e3b, e4b]); res.c = {key: f.rand()}
    print(f"bubble {name}: tadpole (1,) slots {tad_slots(res, (1,), [0])}")
