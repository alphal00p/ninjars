#!/usr/bin/env python3
"""High-precision reference values for one-loop tensor integrals of any rank.

Input: "ninjars-bench-case-v1" (mu-free dense tensor) or "ninjars-bench-case-v2" (blocks
T_0..T_K of N = sum_k (mu^2)^k T_k(q), crates/ninjars-cli/src/export_bench.rs) cases, any
rank; the legacy lane field "mu2_tensors" = [{"power": k, "tensor": [...]}, ...] of the
phase-1 suite is read as well.  mu^2 is the integrand mu^2 (qbar^2 = q^2 - mu^2), never
the lane's "mu_squared", which is the scale mu_R^2.

Output per lane: the Laurent coefficients eps0, eps_m1, eps_m2 of

    I = mu_R^(2 eps) / (i pi^(D/2) r_Gamma) int d^D qbar  N(q, mu^2) / prod_k D_k,
    D_k = (qbar + V_k)^2 - m_k^2,   r_Gamma = Gamma(1+eps) Gamma(1-eps)^2 / Gamma(1-2eps),

i.e. Ninja's (= OneLOop's) normalisation with mu_R^2 = lane "mu_squared" and eps_UV =
eps_IR, N(q) = sum_k tensor[k] q0^a q1^b q2^c q3^d in the 4-dimensional contravariant
components of qbar (Ninja's TensorNumerator order, crates/ninjars/src/fast/tensor_layout.rs).

Precision.  Every input is a binary64 number and is taken as the exact dyadic rational it
denotes.  All preprocessing is exact rational arithmetic (gmpy2.mpq, else fractions):
removal of mu^2 with mu^2 = q^2 + 2 q.V_a + V_a^2 - m_a^2 - D_a (pinching propagator a;
scaleless pieces vanish), the loop-momentum shift to each piece's reference propagator,
and the optional exact rank reduction (--reduce, see reduce_rank).  The driver reads every
mass, momentum, invariant and scale as its exact decimal expansion (the phase-1 harness
passed repr(float), the shortest round-trip decimal, which a quad build reads as a value
that differs from the binary64 input by up to 1e-17 relative).  COLLIER's tensor components
are contracted with the exact coefficients in 90-digit decimal arithmetic.

Backends (--build):
  quad     COLLIER 1.2.9 compiled with -freal-8-real-16 (the reference), --mode 1 COLI,
           --mode 2 DD (DD: N <= 6, pentagon rank <= 5, hexagon rank <= 6)
  precise  binary64 COLLIER without -ffast-math;  stock: the shipped flags
  mp       independent mpmath evaluation for N <= 2 pieces only (tadpoles: closed form;
           bubbles: Feynman parameter x with exact polynomial algebra and mp.quad for
           int x^b log(Delta(x) - i0) dx); no COLLIER code involved (mp_lowpoint.py)

Invariants (--invariants): "smatrix" uses the lane's s_matrix where given (as Ninja's
masters and NinjaRS do) and exact (V_i - V_j)^2 otherwise; "shifts" always uses the
exact (V_i - V_j)^2 of the binary64 shifts.  --input-rounding repr reproduces the phase-1
harness' driver input (shortest decimal); only for provenance comparisons.

usage: collier_reference.py CASE.json [...] --out-dir DIR [--build quad|precise|stock|mp]
       [--mode 1|2] [--invariants smatrix|shifts] [--reduce EXCESS] [--reqacc X]
       [--input-rounding exact|repr] [--jobs N] [--tag NAME]
"""
import argparse
import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal, localcontext
from math import comb

try:
    import gmpy2

    Q = gmpy2.mpq
except ImportError:  # pragma: no cover
    from fractions import Fraction as Q

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parents[2]
DEFAULT_BUILD_DIR = pathlib.Path(os.environ.get("COLLIER_BUILD", REPO / "target" / "rank-n2-collier"))
KEYS = ("eps0", "eps_m1", "eps_m2")
CONTRACTION_DIGITS = 90
QUAD_REQACC = 1e-20  # COLLIER target accuracy of the quad build (default 1e-8 truncates
#                      COLI's Gram/Cayley expansions near small Gram determinants)
DD_MAX_RANK = {5: 5, 6: 6}


# ----------------------------------------------------------------------------
# exact complex rationals and polynomials {(e0, e1, e2, e3): (re, im)}
# ----------------------------------------------------------------------------
def cq(re, im=0):
    return (Q(re), Q(im))


ZERO = cq(0)


def cadd(a, b):
    return (a[0] + b[0], a[1] + b[1])


def cmul(a, b):
    return (a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])


def cneg(a):
    return (-a[0], -a[1])


def cscale(a, f):
    return (a[0] * f, a[1] * f)


def is_zero(c):
    return c[0] == 0 and c[1] == 0


def padd(acc, poly, f=None):
    for m, c in poly.items():
        if f is not None:
            c = cmul(c, f)
        prev = acc.get(m)
        acc[m] = c if prev is None else cadd(prev, c)
    return acc


def pclean(poly):
    return {m: c for m, c in poly.items() if not is_zero(c)}


def pmul(a, b):
    out = {}
    for ma, ca in a.items():
        for mb, cb in b.items():
            m = (ma[0] + mb[0], ma[1] + mb[1], ma[2] + mb[2], ma[3] + mb[3])
            prev = out.get(m)
            p = cmul(ca, cb)
            out[m] = p if prev is None else cadd(prev, p)
    return out


def pdeg(a):
    return max((sum(m) for m in a), default=0)


def pshift(poly, v):
    """poly(l - v) as an exact polynomial in l (v: 4 exact rationals)."""
    if all(x == 0 for x in v):
        return dict(poly)
    out = {}
    for m, c in poly.items():
        terms = [((0, 0, 0, 0), c)]
        for mu in range(4):
            e = m[mu]
            if e == 0:
                continue
            new = []
            for k in range(e + 1):
                f = comb(e, k) * (-v[mu]) ** (e - k)
                if f == 0:
                    continue
                for mm, cc in terms:
                    t = list(mm)
                    t[mu] += k
                    new.append((tuple(t), cscale(cc, f)))
            terms = new
        for mm, cc in terms:
            prev = out.get(mm)
            out[mm] = cc if prev is None else cadd(prev, cc)
    return pclean(out)


def mdot(a, b):
    return a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]


ETA = (1, -1, -1, -1)


# ----------------------------------------------------------------------------
# Ninja tensor layout (port of crates/ninjars/src/fast/tensor_layout.rs)
# ----------------------------------------------------------------------------
def ncoeff(rank):
    return (rank + 1) * (rank + 2) * (rank + 3) * (rank + 4) // 24


_MONO = {}


def monomials(rank):
    if rank in _MONO:
        return _MONO[rank]
    allm = [(0, 0, 0, 0)]
    lower = [(0, 0, 0, 0)]
    for r in range(1, rank + 1):
        r1, r2, r3 = r, r * (r + 1) // 2, r * (r + 1) * (r + 2) // 6

        def bump(m, k):
            m = list(m)
            m[k] += 1
            return tuple(m)
        block = [bump(lower[0], 0)]
        block += [bump(m, 1) for m in lower[:r1]]
        block += [bump(m, 2) for m in lower[:r2]]
        block += [bump(m, 3) for m in lower[:r3]]
        allm += block
        lower = block
    _MONO[rank] = allm
    return allm


def rank_of_tensor(length):
    r = 0
    while ncoeff(r) < length:
        r += 1
    if ncoeff(r) != length:
        raise ValueError(f"tensor length {length} is not binomial(R+4,4)")
    return r


def ncoeff_mu(rank, mu_powers):
    return sum(ncoeff(rank - 2 * k) for k in range(mu_powers + 1))


def tensor_to_poly(tensor, rank=None):
    """Dense tensor [[re, im], ...] -> exact polynomial (binary64 values taken exactly)."""
    if rank is None:
        rank = rank_of_tensor(len(tensor))
    mons = monomials(rank)
    assert len(mons) == len(tensor)
    poly = {}
    for m, c in zip(mons, tensor):
        if c[0] != 0 or c[1] != 0:
            poly[m] = cq(float(c[0]), float(c[1]))
    return poly


def poly_to_tensor(poly, rank):
    """Exact polynomial -> dense binary64 tensor; raises if a coefficient is not a binary64."""
    out = []
    for m in monomials(rank):
        c = poly.get(m, ZERO)
        re, im = float(c[0]), float(c[1])
        if Q(re) != c[0] or Q(im) != c[1]:
            raise ValueError(f"coefficient of {m} is not exactly representable in binary64")
        out.append([re, im])
    for m in poly:
        if sum(m) > rank and not is_zero(poly[m]):
            raise ValueError(f"monomial {m} above rank {rank}")
    return out


def numerator_blocks(case, lane):
    """[(mu power k, exact polynomial T_k(q))] for v1, v2 and legacy mu2_tensors lanes."""
    fmt = case.get("format", "ninjars-bench-case-v1")
    rank = case["rank"]
    if fmt == "ninjars-bench-case-v2":
        K = int(case["mu_powers"])
        t = lane["tensor"]
        if len(t) != ncoeff_mu(rank, K):
            raise ValueError(f"{case['name']}: v2 tensor length {len(t)} != ncoeff_mu({rank},{K})")
        out, off = [], 0
        for k in range(K + 1):
            n = ncoeff(rank - 2 * k)
            out.append((k, tensor_to_poly(t[off:off + n], rank - 2 * k)))
            off += n
        return out
    out = [(0, tensor_to_poly(lane["tensor"]))]
    for extra in lane.get("mu2_tensors", []) or []:
        out.append((int(extra["power"]), tensor_to_poly(extra["tensor"])))
    return out


def lane_inputs(lane):
    V = [tuple(Q(float(x)) for x in v) for v in lane["shifts"]]
    M2 = [cq(float(c[0]), float(c[1])) for c in lane["masses_squared"]]
    S = None
    if lane.get("s_matrix") is not None:
        S = [[Q(float(x)) for x in row] for row in lane["s_matrix"]]
    return V, M2, S, Q(float(lane["mu_squared"]))


# ----------------------------------------------------------------------------
# integrand decomposition (exact)
# ----------------------------------------------------------------------------
def a_poly(v, m2):
    """q^2 + 2 q.v + v^2 - m2, so that mu^2 = a_poly - D with D = (qbar+v)^2 - m2."""
    p = {(2, 0, 0, 0): cq(1), (0, 2, 0, 0): cq(-1), (0, 0, 2, 0): cq(-1), (0, 0, 0, 2): cq(-1)}
    for mu in range(4):
        if v[mu] != 0:
            e = [0, 0, 0, 0]
            e[mu] = 1
            p[tuple(e)] = cq(2 * ETA[mu] * v[mu])
    p[(0, 0, 0, 0)] = cadd(cq(mdot(v, v)), cneg(m2))
    return pclean(p)


def remove_mu2(poly, power, props, V, M2, sink):
    """(mu^2)^power poly / prod_{props} D  ->  sink[props] += 4-dimensional polynomial."""
    if not props or not poly:
        return  # no propagators: scaleless, vanishes in dimensional regularisation
    if power == 0:
        padd(sink.setdefault(tuple(props), {}), poly)
        return
    a = props[0]
    remove_mu2(pmul(poly, a_poly(V[a], M2[a])), power - 1, props, V, M2, sink)
    remove_mu2({m: cneg(c) for m, c in poly.items()}, power - 1, props[1:], V, M2, sink)


def inverse(A):
    n = len(A)
    M = [list(row) + [Q(int(i == j)) for j in range(n)] for i, row in enumerate(A)]
    for c in range(n):
        p = next(r for r in range(c, n) if M[r][c] != 0)
        M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        M[c] = [x / piv for x in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [row[n:] for row in M]


def reduce_rank(poly_q, props, V, M2, sink, excess):
    """Exact rank reduction of poly_q / prod_{props} D (props: tuple, N >= 5).

    With l = q + V_r (r = props[0]) and four of the other propagators j, p_j = V_j - V_r,
    l.p_j = (D_j - D_r + f_j) / 2,  f_j = m_j^2 - m_r^2 - p_j.p_j, exactly in D dimensions,
    and l^mu = sum_j (l.p_j) v_j^mu (v_j: dual basis of the p_j).  Writing
    N = c + sum_mu l^mu M_mu(l), every piece has rank deg - 1; recursion stops at
    rank <= N + excess or N <= 4.  sink[props] += polynomial in q."""
    N = len(props)
    if not poly_q:
        return
    deg = pdeg(poly_q)
    if N <= 4 or deg <= N + excess:
        padd(sink.setdefault(tuple(props), {}), poly_q)
        return
    ref = props[0]
    others = props[1:5]
    vref = V[ref]
    p = [[V[i][mu] - vref[mu] for mu in range(4)] for i in others]
    Ainv = inverse([[p[j][nu] * ETA[nu] for nu in range(4)] for j in range(4)])  # v_j^mu = Ainv[mu][j]
    poly_l = pshift(poly_q, vref)  # l = q + V_ref
    c0 = {(0, 0, 0, 0): poly_l[(0, 0, 0, 0)]} if (0, 0, 0, 0) in poly_l else {}
    Mmu = [dict() for _ in range(4)]
    for m, c in poly_l.items():
        if sum(m) == 0:
            continue
        mu = next(k for k in range(4) if m[k] > 0)
        mm = list(m)
        mm[mu] -= 1
        padd(Mmu[mu], {tuple(mm): c})
    back = [-x for x in vref]
    if c0:
        reduce_rank(c0, props, V, M2, sink, excess)
    half, mhalf = cq(Q(1, 2)), cq(Q(-1, 2))
    for j, i in enumerate(others):
        K = {}
        for mu in range(4):
            if Ainv[mu][j] != 0:
                padd(K, Mmu[mu], cq(Ainv[mu][j]))
        K = pclean(K)
        if not K:
            continue
        Kq = pshift(K, back)  # back to the variable q
        fj = cadd(cadd(M2[i], cneg(M2[ref])), cq(-mdot(p[j], p[j])))
        reduce_rank(padd({}, Kq, half), tuple(k for k in props if k != i), V, M2, sink, excess)
        reduce_rank(padd({}, Kq, mhalf), tuple(k for k in props if k != ref), V, M2, sink, excess)
        reduce_rank(pclean(padd({}, Kq, cmul(half, fj))), props, V, M2, sink, excess)


def lane_sink(case, lane, reduce=None):
    """{props: exact polynomial in q} for one lane (mu^2 removed; optionally rank-reduced)."""
    n = case["n"]
    V, M2, _, _ = lane_inputs(lane)
    props = tuple(range(n))
    sink = {}
    for k, poly in numerator_blocks(case, lane):
        remove_mu2(poly, k, props, V, M2, sink)
    sink = {k: pclean(v) for k, v in sink.items()}
    if reduce is not None:
        red = {}
        for sub, poly in sink.items():
            if poly:
                reduce_rank(poly, sub, V, M2, red, reduce)
        sink = {k: pclean(v) for k, v in red.items()}
    return {k: v for k, v in sink.items() if v}


def collier_mominv(n):
    """(i, j) pairs: COLLIER MomInv entry k = (p_i - p_j)^2 (manual, momentum arguments)."""
    out = []
    for line in range(1, n // 2 + 1):
        count = n // 2 if (n % 2 == 0 and line == n // 2) else n
        for j in range(count):
            out.append(((j + line) % n, j))
    assert len(out) == n * (n - 1) // 2
    return out


def sink_to_jobs(sink, lane, invariants="smatrix"):
    """{props: poly(q)} -> jobs with the loop momentum l = q + V_ref of each piece's
    reference propagator (smallest shift; exactly 0 for V_0 = 0 cases)."""
    V, M2, S, mu2 = lane_inputs(lane)
    jobs = []
    for sub, poly in sorted(sink.items()):
        if not poly:
            continue
        ref = min(sub, key=lambda i: (max(abs(x) for x in V[i]), i))
        order = [ref] + [i for i in sub if i != ref]
        vref = V[ref]
        lpoly = pshift(poly, vref)
        if not lpoly:
            continue
        momvec = [[V[i][mu] - vref[mu] for mu in range(4)] for i in order[1:]]
        inv = []
        for a, b in collier_mominv(len(order)):
            ia, ib = order[a], order[b]
            if S is not None and invariants == "smatrix":
                inv.append(S[ia][ib])
            else:
                d = [x - y for x, y in zip(V[ia], V[ib])]
                inv.append(mdot(d, d))
        jobs.append({"props": order, "poly": lpoly, "rank": pdeg(lpoly), "masses2": [M2[i] for i in order],
                     "momvec": momvec, "mominv": inv, "mu2": mu2})
    return jobs


# ----------------------------------------------------------------------------
# driver I/O
# ----------------------------------------------------------------------------
class Unsupported(Exception):
    pass


def exact_decimal(q):
    """Exact decimal expansion of a dyadic rational (every binary64-derived input is dyadic)."""
    q = Q(q)
    num, den = int(q.numerator), int(q.denominator)
    k = den.bit_length() - 1
    if den != 1 << k:
        raise ValueError(f"{q} is not dyadic")
    sign = "-" if num < 0 else ""
    digits = str(abs(num) * 5 ** k)
    if k == 0:
        return sign + digits
    digits = digits.rjust(k + 1, "0")
    return f"{sign}{digits[:-k]}.{digits[-k:]}"


def fmt_input(q, rounding):
    if rounding == "repr":
        return repr(float(q))
    return exact_decimal(q)


def driver_path(build, build_dir=None):
    d = pathlib.Path(build_dir) if build_dir else DEFAULT_BUILD_DIR
    return d / "bin" / f"collier_tensor_driver-{build}"


def check_mode(jobs, mode):
    if mode == 2:
        for j in jobs:
            N = len(j["props"])
            if N > 6 or j["rank"] > DD_MAX_RANK.get(N, 99):
                raise Unsupported(f"DD branch: N={N} rank={j['rank']} not supported")


def run_driver(jobs, mode=1, build="quad", reqacc=None, rounding="exact", build_dir=None):
    if reqacc is None and build == "quad":
        reqacc = QUAD_REQACC
    check_mode(jobs, mode)
    exe = driver_path(build, build_dir)
    if not exe.exists():
        raise SystemExit(f"{exe} missing; run {HERE / 'build.sh'}")
    with tempfile.TemporaryDirectory(prefix="collier-") as tmp:
        tin, tout = pathlib.Path(tmp) / "jobs.in", pathlib.Path(tmp) / "out.txt"
        lines = [f"{len(jobs)} {max(len(j['props']) for j in jobs)} {max(j['rank'] for j in jobs)}"]
        for j in jobs:
            lines.append(f"{len(j['props'])} {j['rank']}")
            lines.append(fmt_input(j["mu2"], rounding))
            lines += [f"{fmt_input(m[0], rounding)} {fmt_input(m[1], rounding)}" for m in j["masses2"]]
            lines += [" ".join(fmt_input(x, rounding) for x in p) for p in j["momvec"]]
            lines += [fmt_input(x, rounding) for x in j["mominv"]]
        tin.write_text("\n".join(lines) + "\n")
        cmd = [str(exe), str(tin), str(tout), str(mode)] + ([repr(reqacc)] if reqacc else [])
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp)
        if proc.returncode != 0:
            raise RuntimeError(f"driver failed: {proc.stderr}\n{proc.stdout}")
        text = tout.read_text().split("\n")
    tables, cur = [], None
    for ln in text:
        if not ln.strip():
            continue
        f = ln.split()
        if f[0] == "JOB":
            cur = {"N": int(f[2]), "R": int(f[3]), "errflag": int(f[4]), "accflag": int(f[5]), "err": {}, "T": {}}
            tables.append(cur)
        elif f[0] == "ERR":
            cur["err"][int(f[1])] = float(f[2])
        else:
            m = tuple(int(x) for x in f[:4])
            v = [Decimal(x) for x in f[4:12]]
            cur["T"][m] = ((v[0], v[1]), (v[2], v[3]), (v[4], v[5]), (v[6], v[7]))
    assert len(tables) == len(jobs)
    return tables


def qdec(q):
    return Decimal(int(q.numerator)) / Decimal(int(q.denominator))


def contract(job, table):
    """sum_k c_k T(m_k) in 90-digit decimal arithmetic with the exact coefficients c_k."""
    with localcontext() as ctx:
        ctx.prec = CONTRACTION_DIGITS
        acc = [[Decimal(0), Decimal(0)] for _ in range(4)]
        absum = [0.0, 0.0, 0.0]
        err = 0.0
        for m, c in job["poly"].items():
            cr, ci = qdec(c[0]), qdec(c[1])
            ca = math.hypot(float(c[0]), float(c[1]))
            t = table["T"][m]
            for k in range(4):
                tr, ti = t[k]
                acc[k][0] += cr * tr - ci * ti
                acc[k][1] += cr * ti + ci * tr
                if k < 3:
                    absum[k] += ca * math.hypot(float(tr), float(ti))
            err += ca * table["err"].get(sum(m), 0.0)
    return {"laurent": acc, "absum": absum, "err": err, "errflag": table["errflag"], "accflag": table["accflag"]}


def evaluate_jobs(jobs, build, mode, reqacc, rounding, build_dir):
    if build == "mp":
        import mp_lowpoint
        return [mp_lowpoint.job_value(j) for j in jobs]
    tables = run_driver(jobs, mode=mode, build=build, reqacc=reqacc, rounding=rounding, build_dir=build_dir)
    return [contract(j, t) for j, t in zip(jobs, tables)]


def dstr(x):
    with localcontext() as ctx:
        ctx.prec = 40
        return str(+x)


def evaluate_case(case, build="quad", mode=1, reqacc=None, invariants="smatrix", reduce=None,
                  rounding="exact", build_dir=None):
    out = []
    all_jobs, owner, lane_err = [], [], {}
    for li, lane in enumerate(case["lanes"]):
        jobs = sink_to_jobs(lane_sink(case, lane, reduce), lane, invariants)
        try:
            check_mode(jobs, mode if build != "mp" else 1)
            if build == "mp" and any(len(j["props"]) > 2 for j in jobs):
                raise Unsupported("mp backend: only N <= 2 pieces")
        except Unsupported as e:
            lane_err[li] = str(e)
            continue
        for j in jobs:
            all_jobs.append(j)
            owner.append(li)
    values = evaluate_jobs(all_jobs, build, mode, reqacc, rounding, build_dir) if all_jobs else []
    for li, lane in enumerate(case["lanes"]):
        lid = lane.get("id", f"lane{li}")
        if li in lane_err:
            out.append({"id": lid, "unsupported": lane_err[li]})
            continue
        with localcontext() as ctx:
            ctx.prec = CONTRACTION_DIGITS
            totd = [[Decimal(0), Decimal(0)] for _ in range(4)]
            absum = [0.0, 0.0, 0.0]
            err, errflag, accflag, njobs, maxrank = 0.0, 0, 0, 0, 0
            for j, v, o in zip(all_jobs, values, owner):
                if o != li:
                    continue
                for k in range(4):
                    totd[k][0] += v["laurent"][k][0]
                    totd[k][1] += v["laurent"][k][1]
                absum = [a + b for a, b in zip(absum, v["absum"])]
                err += v["err"]
                errflag = min(errflag, v.get("errflag", 0))
                accflag = max(accflag, v.get("accflag", 0))
                njobs += 1
                maxrank = max(maxrank, j["rank"])
        tot = [complex(float(a), float(b)) for a, b in totd]
        scale = max(abs(t) for t in tot[:3])
        out.append({"id": lid,
                    "eps0": [tot[0].real, tot[0].imag],
                    "eps_m1": [tot[1].real, tot[1].imag],
                    "eps_m2": [tot[2].real, tot[2].imag],
                    "eps_m1_uv": [tot[3].real, tot[3].imag],
                    "laurent_decimal": {k: [dstr(totd[i][0]), dstr(totd[i][1])] for i, k in enumerate(KEYS)},
                    "absum": absum,
                    "condition": {k: (absum[i] / abs(tot[i]) if tot[i] != 0 else (0.0 if absum[i] == 0 else math.inf))
                                  for i, k in enumerate(KEYS)},
                    "absum_over_scale": max(absum) / scale if scale else 0.0,
                    "err_estimate": err, "err_estimate_over_scale": err / scale if scale else 0.0,
                    "collier_errflag": errflag, "collier_accflag": accflag,
                    "jobs": njobs, "max_piece_rank": maxrank})
    return out


def run_settings(args):
    return {"build": args.build, "mode": args.mode, "invariants": args.invariants, "reduce": args.reduce,
            "input_rounding": args.input_rounding,
            "reqacc": args.reqacc if args.reqacc else (QUAD_REQACC if args.build == "quad" else
                                                      (None if args.build == "mp" else 1e-8)),
            "collier": "1.2.9" if args.build != "mp" else None}


def _one(job):
    path, kw, out_dir, settings, tag = job
    case = json.loads(pathlib.Path(path).read_text())
    try:
        lanes = evaluate_case(case, **kw)
    except Unsupported as e:  # pragma: no cover
        return f"{case['name']}: skipped ({e})"
    doc = {"format": "ninjars-collier-run-v1", "case": case["name"], "case_file": str(path), "n": case["n"],
           "rank": case["rank"], "mu_powers": case.get("mu_powers", 0), "run": tag, "settings": settings,
           "normalisation": "Ninja/OneLOop: mu_R^(2eps)/(i pi^(D/2) r_Gamma) int d^D q; eps_UV = eps_IR; "
                            "mu_R^2 = lane mu_squared",
           "lanes": lanes}
    out = pathlib.Path(out_dir) / f"{case['name']}.json"
    out.write_text(json.dumps(doc, indent=1) + "\n")
    flags = ",".join(str(l.get("collier_errflag", "-")) for l in lanes)
    return f"{case['name']:40s} n={case['n']} r={case['rank']} lanes={len(lanes)} errflag={flags}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases", nargs="+", type=pathlib.Path)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    ap.add_argument("--build", default="quad", choices=["quad", "precise", "stock", "mp"])
    ap.add_argument("--build-dir", type=pathlib.Path)
    ap.add_argument("--mode", type=int, default=1)
    ap.add_argument("--reqacc", type=float)
    ap.add_argument("--invariants", choices=["smatrix", "shifts"], default="smatrix")
    ap.add_argument("--reduce", type=int, help="exact rank reduction to rank <= N + REDUCE for N >= 5 pieces")
    ap.add_argument("--input-rounding", choices=["exact", "repr"], default="exact")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    kw = dict(build=args.build, mode=args.mode, reqacc=args.reqacc, invariants=args.invariants,
              reduce=args.reduce, rounding=args.input_rounding,
              build_dir=str(args.build_dir) if args.build_dir else None)
    settings = run_settings(args)
    work = [(str(p), kw, str(args.out_dir), settings, args.tag) for p in args.cases]
    # biggest cases first for better load balance
    work.sort(key=lambda w: -pathlib.Path(w[0]).stat().st_size)
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as ex:
            for line in ex.map(_one, work):
                print(line, flush=True)
    else:
        for w in work:
            print(_one(w), flush=True)


if __name__ == "__main__":
    main()
