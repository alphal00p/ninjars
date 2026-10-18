#!/usr/bin/env python3
"""Assemble the reference files of one set from the runs of run_suite.sh.

    assemble.py --cases DIR|FILE... --work WORK/SET --out DIR [--ninja DIR] [--summary FILE] [--twins SUMMARY]

Methods per lane (directories in WORK/SET; absent ones are skipped):
  mp                 independent mpmath tadpoles/bubbles (n <= 2), exact preprocessing
  reduction-exact    60-digit integrand reduction (review_reduction/) with mpmath A0/B_n and
                     oneloopmaster's exact scalar C0/D0 (scalars-exact, 256 bits)
  reduction-exact80  the same at 80 digits (numerical uncertainty of reduction-exact)
  generic-f256       NinjaRS generic engine, Float(256), exact masters (rank <= n+1 only)
  coli-quad, dd-quad COLLIER 1.2.9 quad, COLI / DD branch
  id-dd-quad, id-coli-quad   exact rank reduction (n >= 5) + COLLIER quad pieces
  coli-quad-shifts   COLI quad with invariants from the shifts (input ambiguity of lanes
                     whose s_matrix is not exactly that of the shifts)
  coli-double, dd-double     binary64 COLLIER (diagnostic)

Published value ("source"): mp if available, else reduction-exact if every exact scalar
master of the lane agrees with COLLIER's DD or COLI scalar to <= 1e-12 ("scalar_check"),
else COLLIER (COLI quad, else DD quad).  All differences are scaled: max over
(eps0, eps_m1, eps_m2) of |x - ref| / max |ref|.
  uncertainty       numerical uncertainty of the published value: mp -> its quadrature
                    error estimate; reduction-exact -> |60 - 80 digits|; COLLIER -> the
                    largest difference to another valid COLLIER branch; floor 1e-38 (the
                    40-digit decimals); raised to input_ambiguity where s_matrix and shifts
                    are inconsistent (phys-like lanes): the largest of COLI quad smat vs shift
                    invariants, reduction-exact vs the same with shift invariants
                    (reduction-exact-shifts), the generic engine where it runs, and the
                    generic engine on the rank <= n+1 twins of the lane (--twins), and that
                    deviation extrapolated geometrically from the two highest twin ranks.
                    For mp/reduction-exact it assumes the master functions are exact (they
                    are evaluated at 50-256 bits).
  verified_to       the smallest difference to a valid COLLIER evaluation (COLI, DD, identity):
                    agreement with a fully independent code (other reduction, other masters).
  uncertainty_independent = max(uncertainty, verified_to)
  tier              A: verified_to <= 1e-8, scalar_check ok, uncertainty <= 1e-25;
                    B: verified_to <= 1e-8 and uncertainty <= 1e-9;  C: otherwise.
COLLIER runs with errflag < 0 are invalid (reported in checks_invalid)."""
import argparse
import json
import math
import pathlib
from decimal import Decimal, getcontext

getcontext().prec = 60
KEYS = ("eps0", "eps_m1", "eps_m2")
COLLIER = ("coli-quad", "dd-quad", "id-dd-quad", "id-coli-quad")
DIAG = ("coli-quad-shifts", "coli-double", "dd-double", "generic-f256", "mp", "reduction-exact",
        "reduction-exact80", "reduction-exact-shifts")
SCALAR_TOL = 1e-12
FLOOR = 1e-38


def load_run(d):
    out = {}
    if not d.is_dir():
        return None
    for f in d.glob("*.json"):
        doc = json.loads(f.read_text())
        for l in doc.get("lanes", []):
            out[l.get("id")] = l
    return out


def dec(lane):
    if lane is None or "unsupported" in lane or "error" in lane:
        return None
    d = lane.get("laurent_decimal")
    if not d:
        return None
    try:
        v = [(Decimal(d[k][0]), Decimal(d[k][1])) for k in KEYS]
    except Exception:
        return None
    if any(not (x.is_finite() and y.is_finite()) for x, y in v):
        return None
    return v


def cabs(z):
    return (z[0] * z[0] + z[1] * z[1]).sqrt()


def diff(a, b, scale):
    return float(max(cabs((x[0] - y[0], x[1] - y[1])) for x, y in zip(a, b)) / scale)


def scalar_checks(work):
    """lane id -> worst over its scalars of min(|exact - coli|, |exact - dd|) / |exact|."""
    ex, co, dd = (load_run(work / d) for d in ("scalars-exact", "scalars-coli", "scalars-dd"))
    out = {}
    if not ex:
        return out
    for sid, l in ex.items():
        e = dec(l)
        lane_id = sid.split("#")[0]
        if e is None:
            out[lane_id] = math.inf
            continue
        sc = max(cabs(x) for x in e) or Decimal(1)
        best = math.inf
        for run in (co, dd):
            if run and dec(run.get(sid)) is not None and run[sid].get("collier_errflag", 0) >= 0:
                best = min(best, diff(e, dec(run[sid]), sc))
        out[lane_id] = max(out.get(lane_id, 0.0), best)
    return out


def kin_key(lane_id):
    """Kinematics of a lane id (rank and mu tag removed): twins share it."""
    import re
    return re.sub(r"-r\d+(-mu2|-muall)?-lane", "-lane", lane_id)


def inconsistent_smatrix(lane):
    S = lane.get("s_matrix")
    if S is None:
        return False
    V = [[Decimal(x) for x in v] for v in lane["shifts"]]
    for i in range(len(V)):
        for j in range(len(V)):
            d = [a - b for a, b in zip(V[i], V[j])]
            if Decimal(S[i][j]) != d[0] * d[0] - d[1] * d[1] - d[2] * d[2] - d[3] * d[3]:
                return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", type=pathlib.Path, nargs="+", required=True, help="case directories or files")
    ap.add_argument("--work", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--ninja", type=pathlib.Path, help="original-Ninja result directory (<case>.json)")
    ap.add_argument("--summary", type=pathlib.Path)
    ap.add_argument("--case-prefix", default="", help="path prefix recorded as case_file")
    ap.add_argument("--twins", type=pathlib.Path, action="append", default=[],
                    help="summary of rank <= n+1 twins on the same kinematics (input-ambiguity calibration)")
    a = ap.parse_args()
    runs = {k: load_run(a.work / k) for k in COLLIER + DIAG}
    runs = {k: v for k, v in runs.items() if v}
    schk = scalar_checks(a.work)
    twin_amb = {}  # kinematics -> {rank: generic-engine deviation}
    for t in a.twins:
        for r in json.loads(t.read_text()):
            if r.get("input_ambiguity") is not None and r.get("d_generic-f256") is not None:
                d = twin_amb.setdefault(kin_key(r["id"]), {})
                d[r["rank"]] = max(d.get(r["rank"], 0.0), r["d_generic-f256"])
    a.out.mkdir(parents=True, exist_ok=True)
    summary = []
    files = sorted(f for c in a.cases for f in (c.glob("*.json") if c.is_dir() else [c]))
    for cf in files:
        case = json.loads(cf.read_text())
        ninja = None
        if a.ninja and (a.ninja / cf.name).exists():
            ninja = {l["id"]: l for l in json.loads((a.ninja / cf.name).read_text())["lanes"]}
        lanes_out = []
        for li, lane in enumerate(case["lanes"]):
            lid = lane["id"]
            vals = {k: dec(r.get(lid)) for k, r in runs.items()}
            invalid = {}
            for k in COLLIER + ("coli-quad-shifts", "coli-double", "dd-double"):
                l = runs.get(k, {}).get(lid)
                if l is not None and vals.get(k) is not None and l.get("collier_errflag", 0) < 0:
                    invalid[k] = l["collier_errflag"]
                    vals[k] = None
            sc_ok = schk.get(lid, 0.0) <= SCALAR_TOL
            if vals.get("mp") is not None:
                src = "mp"
            elif vals.get("reduction-exact") is not None and sc_ok:
                src = "reduction-exact"
            elif vals.get("coli-quad") is not None:
                src = "coli-quad"
            elif vals.get("dd-quad") is not None:
                src = "dd-quad"
            else:
                src = None
            if src is None:
                lanes_out.append({"id": lid, "tier": "C", "error": "no valid evaluation"})
                continue
            v = vals[src]
            scale = max(cabs(x) for x in v)
            vanishing = scale == 0
            if vanishing:
                scale = Decimal(1)
            checks = {k: diff(x, v, scale) for k, x in vals.items() if x is not None and k != src}
            amb, amb_parts = None, None
            if inconsistent_smatrix(lane):
                # different correct conventions for an s_matrix that is not that of the shifts:
                # COLLIER smat vs shifts, the reduction with shift invariants, the generic engine
                # (masters from the s_matrix), and the generic engine on rank <= n+1 twins
                amb_parts = {}
                if vals.get("coli-quad") is not None and vals.get("coli-quad-shifts") is not None:
                    amb_parts["collier_shift_invariants"] = diff(vals["coli-quad"], vals["coli-quad-shifts"], scale)
                if "reduction-exact-shifts" in checks:
                    amb_parts["reduction_shift_invariants"] = checks["reduction-exact-shifts"]
                if "generic-f256" in checks:
                    amb_parts["generic_engine"] = checks["generic-f256"]
                tw = twin_amb.get(kin_key(lid), {})
                if tw:
                    amb_parts["generic_engine_on_twins"] = max(tw.values())
                    R = case["rank"]
                    if R - 1 in tw and R - 2 in tw and tw[R - 2] > 0:
                        # the deviation grows with the rank: extrapolate one rank geometrically
                        amb_parts["generic_engine_extrapolated"] = tw[R - 1] * max(1.0, tw[R - 1] / tw[R - 2])
                amb = max(amb_parts.values()) if amb_parts else None
            if src == "mp":
                ml = runs["mp"][lid]
                u = max(FLOOR, ml.get("err_estimate_over_scale", 0.0))
            elif src == "reduction-exact":
                u = max(FLOOR, checks.get("reduction-exact80", 0.0)) if "reduction-exact80" in checks else None
            else:
                others = [checks[k] for k in ("dd-quad", "coli-quad", "id-dd-quad") if k in checks]
                u = max(others) if others else None
            if u is not None and amb is not None:
                u = max(u, amb)
            ver = [(checks[k], k) for k in COLLIER if k in checks]
            verified_to, verified_by = min(ver) if ver else (None, None)
            if src in ("coli-quad", "dd-quad"):
                verified_to, verified_by = (u, "cross-branch") if u is not None else (None, None)
            ok = verified_to is not None and verified_to <= 1e-8
            if src == "reduction-exact":
                ok = ok and sc_ok
            if ok and u is not None and u <= 1e-25:
                tier = "A"
            elif ok and u is not None and u <= 1e-9:
                tier = "B"
            else:
                tier = "C"
            cl = runs.get("coli-quad", {}).get(lid, {})
            rl = runs.get("reduction-exact", {}).get(lid, {})
            out = {"id": lid,
                   "eps0": [float(v[0][0]), float(v[0][1])],
                   "eps_m1": [float(v[1][0]), float(v[1][1])],
                   "eps_m2": [float(v[2][0]), float(v[2][1])],
                   "laurent_decimal": {k: [str(v[i][0]), str(v[i][1])] for i, k in enumerate(KEYS)},
                   "source": src, "tier": tier,
                   "uncertainty": u, "uncertainty_independent": None if u is None or verified_to is None else max(u, verified_to),
                   "verified_to": verified_to, "verified_by": verified_by,
                   "input_ambiguity": amb, "input_ambiguity_parts": amb_parts, "scale": float(scale), "vanishing": vanishing,
                   "condition": cl.get("condition"), "absum_over_scale": cl.get("absum_over_scale"),
                   "scalar_check": schk.get(lid), "checks": checks}
            if invalid:
                out["checks_invalid"] = invalid
            if vals.get("coli-quad") is not None:
                out["collier_coli_quad_decimal"] = cl["laurent_decimal"]
            if rl and "new_terms_over_scale" in rl:
                out["reduction"] = {k: rl.get(k) for k in ("recipe_vs_fit", "new_terms_over_scale", "parts_over_scale")}
            if ninja and lid in ninja:
                nl = ninja[lid]
                try:
                    nv = [(Decimal(nl[k][0]), Decimal(nl[k][1])) for k in KEYS]
                    out["ninja"] = diff(nv, v, scale)
                except Exception:
                    out["ninja"] = None
            lanes_out.append(out)
            summary.append({"case": case["name"], "lane": li, "id": lid, "n": case["n"], "rank": case["rank"],
                            "mu_powers": case.get("mu_powers", 0), "source": src, "tier": tier, "uncertainty": u,
                            "verified_to": verified_to, "input_ambiguity": amb, "scalar_check": schk.get(lid),
                            "ninja": out.get("ninja"), **{f"d_{k}": x for k, x in checks.items()}})
        doc = {"format": "ninjars-reference-collier-v1", "case": case["name"],
               "case_file": a.case_prefix + cf.name, "n": case["n"], "rank": case["rank"],
               "mu_powers": case.get("mu_powers", 0),
               "normalisation": "Ninja/OneLOop: mu_R^(2eps) / (i pi^(D/2) r_Gamma) int d^D q, eps_UV = eps_IR, "
                                "mu_R^2 = lane mu_squared; numerator in the 4-dimensional components of q and the "
                                "integrand mu^2 (v2 blocks), D_k = (qbar + V_k)^2 - m_k^2, qbar^2 = q^2 - mu^2",
               "reference": "validation/rank-n2/README.md: source mp | reduction-exact | coli-quad | dd-quad per lane",
               "lanes": lanes_out}
        (a.out / cf.name).write_text(json.dumps(doc, indent=1) + "\n")
    if a.summary:
        a.summary.write_text(json.dumps(summary, indent=1) + "\n")
    tiers = {}
    for s in summary:
        tiers[s["tier"]] = tiers.get(s["tier"], 0) + 1
    print(f"{len(summary)} lanes, tiers {dict(sorted(tiers.items()))}")


if __name__ == "__main__":
    main()
