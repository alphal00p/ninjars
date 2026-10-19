#!/usr/bin/env python3
"""Collect the rank n+2 measurements into docs/benchmark/data/rank_n2.json.

Inputs (validation/benchmark/collier/results/, written by run_bench.py and
run_accuracy.py):
- collier.json        COLLIER 1.2.10 per case and build/branch: block-median timings
                      per round, and the binary64 Laurent coefficients, flags and
                      error estimates of every lane (validation pass);
- ninjars-fast.json   `ninjars case-bench` (fast engine) per case, width 1 and 32, per
                      round;
- ninjars-generic.json `ninjars case-bench --engines generic` (one round);
- accuracy.json       `ninjars case-check` verdicts and scaled differences (fast and
                      generic f64, generic DoubleFloat, Float(128), Float(256));
- environment.json    CPU, date, rounds, blocks, foreign CPU time per run;
- instructions.json   retired instructions per point (count_instructions.py), optional;
- accuracy-final-review.json  NinjaRS and COLLIER on the final review's fresh cases
                      (run_final_review.py), optional.
The high-precision references are validation/rank-n2/reference (suite, synthetic) and
validation/rank-n2/controls/reference (rank n+1 twins).

Timings: per case and quantity the best (smallest) of the rounds' block medians, in
microseconds per phase-space point. Scaled differences follow case-check: the largest
|x - ref| over the three Laurent coefficients over the lane scale, here in exact decimal
arithmetic against the 40-digit reference decimals.

usage: collect.py [--results DIR] [--output docs/benchmark/data/rank_n2.json]
"""
import argparse
import json
import math
import pathlib
import re
import statistics
import sys
from decimal import Decimal, getcontext

getcontext().prec = 80
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_bench import dump  # noqa: E402  (compact JSON, one line per case or lane)
REPO = HERE.parents[2]
RN2 = REPO / "validation" / "rank-n2"
KEYS = ("eps0", "eps_m1", "eps_m2")
SHA256 = "460624ea709211506b642343a7f6be41d3883ccb52598b14950f8e07c0262713"

# Rows of the per-case timing table (the other cases enter the per-n aggregates).
DISPLAY = (
    [f"syn-n{n}-r{n + 2}" for n in range(1, 7)]
    + ["rn2-int-complex-n7-r9",
       "rn2-int-zero-n2-r4", "rn2-irml-zero-n4-r6", "rn2-irml-zero-n6-r8",
       "rn2-int-complex-n4-r6-mu2", "rn2-int-complex-n6-r8-mu2", "rn2-int-real-n7-r9-muall",
       "rn2-shifted-mixed-n5-r7"]
)


def load(path):
    return json.loads(pathlib.Path(path).read_text())


def us(ns):
    return None if ns is None else ns / 1000.0


def best(rounds, case, *path):
    """Smallest median over rounds of rounds[r][case][path...]['median_ns_per_point']."""
    vals = []
    for rnd in rounds:
        node = rnd.get(case)
        for key in path:
            if node is None:
                break
            node = node.get(key) if isinstance(node, dict) else None
        if node is not None:
            vals.append(node["median_ns_per_point"])
    return min(vals) if vals else None


def width_entry(case_rounds, width, quantity):
    vals = []
    for entry in case_rounds:
        for w in entry["widths"]:
            if w["width"] == width:
                if quantity in w.get("summary_ns_per_point", {}):
                    vals.append(w["summary_ns_per_point"][quantity])
    return min(vals) if vals else None


def ninjars_times(rounds, case, numerator_key):
    entries = [r[case] for r in rounds if case in r]
    if not entries:
        return None
    out = {}
    for width in (1, 32):
        out[f"w{width}"] = {q: us(width_entry(entries, width, k)) for q, k in (
            ("total", "total"), ("reduction", "reduction"), ("masters", "masters"),
            ("numerator", numerator_key))}
    out["failed_lanes"] = max(w.get("failed_lanes", 0) for e in entries for w in e["widths"])
    return out


def classify(name):
    m = re.match(r"(syn|rn2)-(?:([a-z0-9]+)-([a-z]+)-)?n(\d+)-r(\d+)(?:-(mu2|muall))?$", name)
    if name.startswith("syn-"):
        return {"family": "syn", "masses": "real/complex", "mu": None}
    fam, masses, mu = m.group(2), m.group(3), m.group(6)
    return {"family": fam, "masses": masses, "mu": mu}


def reference(name):
    for d in (RN2 / "reference", RN2 / "controls" / "reference"):
        p = d / f"{name}.json"
        if p.exists():
            return load(p)
    return None


def scaled_difference(values, ref_lane):
    """values: [re0, im0, re1, im1, re2, im2] binary64 floats."""
    refv = []
    for k in KEYS:
        refv += [Decimal(x) for x in ref_lane["laurent_decimal"][k]]
    scale = ref_lane.get("scale")
    if not scale:
        scale = max(math.hypot(float(refv[2 * j]), float(refv[2 * j + 1])) for j in range(3)) or 1.0
    diff = 0.0
    for j in range(3):
        dr = Decimal(values[2 * j]) - refv[2 * j]
        di = Decimal(values[2 * j + 1]) - refv[2 * j + 1]
        diff = max(diff, math.hypot(float(dr), float(di)))
    return diff / scale, scale


def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"lanes": 0}
    return {"lanes": len(xs), "median": statistics.median(xs), "max": max(xs),
            "p90": sorted(xs)[min(len(xs) - 1, int(0.9 * len(xs)))]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=pathlib.Path, default=HERE / "results")
    ap.add_argument("--output", type=pathlib.Path, default=REPO / "docs" / "benchmark" / "data" / "rank_n2.json")
    args = ap.parse_args()
    res = args.results
    env = load(res / "environment.json")
    collier = load(res / "collier.json")
    fast = load(res / "ninjars-fast.json")["rounds"]
    generic_path = res / "ninjars-generic.json"
    generic = load(generic_path)["rounds"] if generic_path.exists() else []
    accuracy_path = res / "accuracy.json"
    accuracy = load(accuracy_path) if accuracy_path.exists() else None
    meta = collier["meta"]
    twins = {pathlib.Path(p).stem for p in env.get("twins", [])}
    variants = ("precise-coli", "precise-dd", "stock-coli", "stock-dd", "precise-coli-cache")

    # ------------------------------------------------------------ timing --
    cases = []
    for name, m in meta.items():
        entry = {"case": name, "n": m["n"], "rank": m["rank"], "mu_powers": m["mu_powers"],
                 "twin": name in twins, **classify(name),
                 "pieces": max(len(p) for p in m["pieces"]),
                 "coefficients": max(sum(q["coefficients"] for q in p) for p in m["pieces"]),
                 "shifted": classify(name)["family"] == "shifted" or name == "syn-n1-r3",
                 "ninjars": ninjars_times(fast, name, "tensor_numerator"),
                 "generic": ninjars_times(generic, name, "numerator"), "collier": {}}
        for v in variants:
            q = {k: us(best(collier["rounds"], name, v, k)) for k in (
                "total_finite", "total_laurent", "calls_finite", "contraction", "delta_switch", "scale_switch")}
            if q["total_finite"] is not None:
                entry["collier"][v] = q
        cases.append(entry)
    cases.sort(key=lambda c: (c["twin"], c["n"], c["case"]))

    def ratio(c, variant="precise-coli", quantity="total_finite", width="w32"):
        a = c["ninjars"] and c["ninjars"][width]["total"]
        b = c["collier"].get(variant, {}).get(quantity)
        return a / b if a and b else None

    by_n = []
    suite = [c for c in cases if not c["twin"]]
    for n in sorted({c["n"] for c in suite}):
        cs = [c for c in suite if c["n"] == n]
        row = {"n": n, "cases": len(cs)}
        for label, variant, quantity, width in (
                ("coli", "precise-coli", "total_finite", "w32"), ("coli_w1", "precise-coli", "total_finite", "w1"),
                ("coli_laurent", "precise-coli", "total_laurent", "w32"),
                ("dd", "precise-dd", "total_finite", "w32"), ("stock_coli", "stock-coli", "total_finite", "w32"),
                ("coli_cache", "precise-coli-cache", "total_finite", "w32")):
            rs = [ratio(c, variant, quantity, width) for c in cs]
            rs = [r for r in rs if r]
            row[label] = ({"cases": len(rs), "geomean": geomean(rs), "min": min(rs), "max": max(rs)} if rs
                          else None)
        row["ninjars_w32_total_geomean"] = geomean([c["ninjars"]["w32"]["total"] for c in cs if c["ninjars"]])
        row["collier_coli_geomean"] = geomean([c["collier"]["precise-coli"]["total_finite"] for c in cs
                                               if "precise-coli" in c["collier"]])
        gs = [c["generic"]["w32"]["total"] / c["ninjars"]["w32"]["total"] for c in cs
              if c["generic"] and c["ninjars"]]
        row["generic_over_fast_w32"] = geomean(gs)
        row["generic_reduction_over_fast_w32"] = geomean(
            [c["generic"]["w32"]["reduction"] / c["ninjars"]["w32"]["reduction"] for c in cs
             if c["generic"] and c["ninjars"]])
        shares = [c["ninjars"]["w32"]["numerator"] / c["ninjars"]["w32"]["total"] for c in cs if c["ninjars"]]
        row["numerator_share_w32"] = {"median": statistics.median(shares), "min": min(shares), "max": max(shares)}
        shares = [c["ninjars"]["w32"]["reduction"] / c["ninjars"]["w32"]["total"] for c in cs if c["ninjars"]]
        row["reduction_share_w32"] = {"median": statistics.median(shares), "min": min(shares), "max": max(shares)}
        laurent = [c["collier"]["precise-coli"]["total_laurent"] / c["collier"]["precise-coli"]["total_finite"]
                   for c in cs if "precise-coli" in c["collier"]]
        row["collier_laurent_over_finite"] = geomean(laurent)
        cache = [c["collier"]["precise-coli-cache"]["total_finite"] / c["collier"]["precise-coli"]["total_finite"]
                 for c in cs if "precise-coli-cache" in c["collier"] and "precise-coli" in c["collier"]]
        row["collier_cache_over_nocache"] = geomean(cache)
        by_n.append(row)
    cache_gain = [c["collier"]["precise-coli-cache"]["total_finite"] / c["collier"]["precise-coli"]["total_finite"]
                  for c in suite if "precise-coli-cache" in c["collier"] and "precise-coli" in c["collier"]]
    stock_gain = [c["collier"]["stock-coli"]["total_finite"] / c["collier"]["precise-coli"]["total_finite"]
                  for c in suite if "stock-coli" in c["collier"] and "precise-coli" in c["collier"]]
    overheads = {k: statistics.median([c["collier"]["precise-coli"][k] for c in suite if "precise-coli" in c["collier"]])
                 for k in ("delta_switch", "scale_switch")}

    # retired instructions per point (count_instructions.py), when measured
    instructions = None
    if (res / "instructions.json").exists():
        raw = load(res / "instructions.json")["cases"]
        instructions = [{"case": name, "n": meta[name]["n"] if name in meta else None, **row}
                        for name, row in raw.items()]
        instructions.sort(key=lambda r: (r["n"] or 0, r["case"]))

    # ------------------------------------------------- rank n+1 twins --
    index = {c["case"]: c for c in cases}
    twin_rows = []
    for n in range(1, 8):
        a, b = index.get(f"rn2-int-complex-n{n}-r{n + 2}"), index.get(f"rn2-int-complex-n{n}-r{n + 1}")
        if not a or not b or not a["ninjars"] or not b["ninjars"]:
            continue
        row = {"n": n}
        for w in ("w1", "w32"):
            for q in ("reduction", "total", "numerator", "masters"):
                row[f"{q}_{w}"] = [a["ninjars"][w][q], b["ninjars"][w][q], a["ninjars"][w][q] / b["ninjars"][w][q]]
        ca, cb = a["collier"].get("precise-coli"), b["collier"].get("precise-coli")
        if ca and cb:
            row["collier"] = [ca["total_finite"], cb["total_finite"], ca["total_finite"] / cb["total_finite"]]
        twin_rows.append(row)

    # ---------------------------------------------------------- accuracy --
    acc_runs = (accuracy or {}).get("runs", {})
    ninja_lane = {}
    for key, label in (("fast/f64", "fast"), ("generic/f64", "generic")):
        for lane in acc_runs.get(key, []):
            ninja_lane.setdefault(lane["id"], {})[label] = lane
    lanes = []
    for name, m in meta.items():
        if name in twins:
            continue
        ref = reference(name)
        if ref is None:
            continue
        ref_lanes = {l["id"]: l for l in ref["lanes"]}
        for i, lane_id in enumerate(m["lane_ids"]):
            rl = ref_lanes.get(lane_id)
            if rl is None:
                continue
            row = {"case": name, "id": lane_id, "n": m["n"], "rank": m["rank"], "mu_powers": m["mu_powers"],
                   "tier": rl.get("tier"), "uncertainty": rl.get("uncertainty"),
                   # COLLIER's contraction condition, as case-check reads it
                   "condition": rl.get("absum_over_scale",
                                       max((rl.get("condition") or {}).values() or [0.0])),
                   "new_terms_over_scale": (rl.get("reduction") or {}).get("new_terms_over_scale")}
            for v in ("precise-coli", "precise-dd", "stock-coli", "stock-dd"):
                runs = collier["lanes"].get(name, {}).get(v)
                if not runs:
                    continue
                cl = runs[i]
                vals = cl["eps0"] + cl["eps_m1"] + cl["eps_m2"]
                err, scale = scaled_difference(vals, rl)
                row[v] = {"error": err, "estimate": cl["err_estimate"] / scale, "errflag": cl["errflag"],
                          "accflag": cl["accflag"], "absum_over_scale": cl["absum"] / scale}
            for label in ("fast", "generic"):
                nl = ninja_lane.get(lane_id, {}).get(label)
                if nl:
                    row[label] = {"error": nl["scaled_difference"], "verdict": nl["verdict"], "status": nl["status"]}
            lanes.append(row)
    lanes.sort(key=lambda r: (r["n"], r["case"], r["id"]))

    def err(r, key):
        return r[key]["error"] if key in r and r[key].get("error") is not None else None

    # Binary64 comparison set: tier A and B lanes whose condition (COLLIER's, or the
    # cancellation of the new rank n+2 terms) is at most 1e6, as case-check judges binary64.
    def judged(r):
        cond = max(r["condition"] or 0.0, r["new_terms_over_scale"] or 0.0)
        return r["tier"] in ("A", "B") and cond <= 1e6

    acc_by_n = []
    for n in sorted({r["n"] for r in lanes}):
        rs = [r for r in lanes if r["n"] == n and judged(r)]
        row = {"n": n, "lanes": len(rs), "tier_c": sum(1 for r in lanes if r["n"] == n and r["tier"] == "C"),
               "ill_conditioned": sum(1 for r in lanes if r["n"] == n and r["tier"] in ("A", "B") and not judged(r))}
        for key in ("fast", "generic", "precise-coli", "precise-dd", "stock-coli"):
            row[key] = stats([err(r, key) for r in rs])
            row[f"{key}_all_ab"] = stats([err(r, key) for r in lanes if r["n"] == n and r["tier"] in ("A", "B")])
        both = [r for r in rs if err(r, "fast") is not None and err(r, "precise-coli") is not None]
        row["fast_better_than_coli"] = sum(1 for r in both if err(r, "fast") <= err(r, "precise-coli"))
        row["compared"] = len(both)
        acc_by_n.append(row)

    # COLLIER's own accuracy information against its actual error (tier A/B lanes)
    flags = {}
    for v in ("precise-coli", "precise-dd"):
        rs = [r for r in lanes if v in r and r["tier"] in ("A", "B")]
        est_ok = [r for r in rs if r[v]["error"] <= max(r[v]["estimate"], 1e-15)]
        flags[v] = {
            "lanes": len(rs),
            "errflag_negative": sum(1 for r in rs if r[v]["errflag"] < 0),
            "accflag": {str(a): sum(1 for r in rs if r[v]["accflag"] == a) for a in (0, -1, -2)},
            "error_above_1e-8": sum(1 for r in rs if r[v]["error"] > 1e-8),
            "error_above_1e-8_with_accflag_0": sum(1 for r in rs if r[v]["error"] > 1e-8 and r[v]["accflag"] == 0),
            "accflag_negative_error_below_1e-8": sum(1 for r in rs if r[v]["error"] <= 1e-8 and r[v]["accflag"] < 0),
            "estimate_covers_error": len(est_ok),
            "worst_underestimate": max((r[v]["error"] / max(r[v]["estimate"], 1e-15) for r in rs), default=None),
            "estimate_zero": sum(1 for r in rs if r[v]["estimate"] == 0),
            "median_estimate_over_error": statistics.median(
                [r[v]["estimate"] / r[v]["error"] for r in rs if r[v]["error"] > 0]) if rs else None,
        }

    # Agreement with the binary64 COLLIER 1.2.9 diagnostics of the references (checks
    # coli-double / dd-double: precise build, exact decimal inputs, 90-digit contraction).
    # Errors below 1e-14 are rounding of the binary64 contraction and inputs.
    agreement = {}
    for v, check in (("precise-coli", "coli-double"), ("precise-dd", "dd-double")):
        pairs = []
        for r in lanes:
            if v not in r:
                continue
            rl = {l["id"]: l for l in reference(r["case"])["lanes"]}[r["id"]]
            old = (rl.get("checks") or {}).get(check)
            if old is not None:
                pairs.append((r["id"], r[v]["error"], old))
        resolved = [(i, a, b) for i, a, b in pairs if max(a, b) > 1e-14]
        differ = [(i, a, b) for i, a, b in resolved if max(a, b) > 2 * min(a, b)]
        agreement[v] = {"lanes": len(pairs), "above_1e-14": len(resolved),
                        "median_ratio": statistics.median([a / b for _, a, b in resolved if b > 0]) if resolved
                        else None,
                        "differ_2x": [{"id": i, "error_1.2.10": a, "error_1.2.9": b} for i, a, b in differ]}

    # the final review's fresh cases (run_final_review.py), when measured
    fresh = None
    if (res / "accuracy-final-review.json").exists():
        fr = load(res / "accuracy-final-review.json")["lanes"]
        fresh = {"lanes": len(fr), "cases": len({l["case"] for l in fr}), "by_n": []}
        for n in sorted({l["n"] for l in fr}):
            rs = [l for l in fr if l["n"] == n]
            row = {"n": n, "lanes": len(rs)}
            for key in ("fast", "generic", "coli", "dd"):
                row[key] = stats([l.get(key) for l in rs])
                row[key]["above_1e-6"] = sum(1 for l in rs if (l.get(key) or 0) > 1e-6)
            fresh["by_n"].append(row)
        bad = [l for l in fr if max(l.get("fast") or 0, l.get("generic") or 0) > 1e-6]
        unstable = [l for l in fr if l.get("generic_stability_test") == "Unstable"]
        fresh["ninjars_above_1e-6"] = len(bad)
        fresh["ninjars_above_1e-6_flagged"] = sum(1 for l in bad if l.get("generic_stability_test") == "Unstable")
        fresh["ninjars_above_1e-5_unflagged"] = sum(
            1 for l in bad if max(l.get("fast") or 0, l.get("generic") or 0) > 1e-5
            and l.get("generic_stability_test") != "Unstable")
        fresh["flagged"] = len(unstable)
        fresh["above_1e-6_condition_min"] = min((l["condition"] for l in bad), default=None)
        fresh["above_1e-6_coli"] = stats([l.get("coli") for l in bad])
        fresh["flagged_below_1e-6"] = sum(1 for l in unstable if max(l.get("fast") or 0, l.get("generic") or 0) <= 1e-6)
        fresh["flagged_coli"] = stats([l.get("coli") for l in unstable])
        fresh["flagged_condition_min"] = min((l["condition"] for l in unstable), default=None)
        fresh["coli_above_1e-6"] = [{"id": l["id"], "coli": l["coli"], "fast": l.get("fast"),
                                     "generic": l.get("generic"), "flags": l.get("coli_flags")}
                                    for l in fr if (l.get("coli") or 0) > 1e-6]

    # case-check verdict table (docs/rank-n2.md section 8.3), per n and run
    verdict_table = {}
    for key, rows in acc_runs.items():
        table = {}
        for lane in rows:
            t = table.setdefault(str(lane["n"]), {"lanes": 0, "pass": 0, "fail": 0, "report-only": 0,
                                                    "known-failure": 0, "other": 0})
            t["lanes"] += 1
            t[lane["verdict"] if lane["verdict"] in t else "other"] += 1
        checked = [l["scaled_difference"] for l in rows if l["verdict"] in ("pass", "fail")
                   and l["scaled_difference"] is not None]
        tier_a = [l["scaled_difference"] for l in rows if l["verdict"] in ("pass", "fail") and l["tier"] == "A"
                  and l["scaled_difference"] is not None]
        verdict_table[key] = {"by_n": table, "largest_checked": max(checked) if checked else None,
                              "largest_tier_a": max(tier_a) if tier_a else None,
                              "exit_status": (accuracy or {}).get("exit_status", {}).get(key.split("/")[1])}

    out = {
        "date": env["date"],
        "host": env.get("processor", ""),
        "cpu": env["cpu"],
        "ninjars_head": env.get("ninjars_head"),
        "oneloopmaster": env.get("oneloopmaster"),
        "accuracy_head": (accuracy or {}).get("ninjars_head"),
        "rounds": env["rounds"],
        "blocks": {"ninjars": env["blocks"], "collier": env.get("collier_blocks"),
                   "generic": env.get("generic_blocks"), "min_block_ms": env["min_block_ms"]},
        "foreign_cpu_s": sum(r.get("foreign_s", 0.0) for r in env["runs"]),
        "measured_cpu_s": sum(r.get("wall_s", 0.0) for r in env["runs"]),
        "collier": {"version": collier["version"], "sha256": SHA256,
                    "builds": {"precise": "gfortran -O2 -funroll-loops (shipped Release flags without -ffast-math)",
                               "stock": "gfortran -O2 -funroll-loops -ffast-math (shipped Release flags)"},
                    "reqacc": 1e-8},
        "display": [c for c in DISPLAY if c in index],
        "cases": cases,
        "by_n": by_n,
        "collier_cache_over_nocache": {"geomean": geomean(cache_gain), "min": min(cache_gain, default=None),
                                       "max": max(cache_gain, default=None)},
        "collier_stock_over_precise": {"geomean": geomean(stock_gain), "min": min(stock_gain, default=None),
                                       "max": max(stock_gain, default=None)},
        "collier_overheads_us": overheads,
        "twins": twin_rows,
        "accuracy": {"by_n": acc_by_n, "collier_flags": flags, "collier_vs_1.2.9": agreement,
                     "final_review": fresh},
        "case_check": verdict_table,
        "instructions": instructions,
    }
    args.output.write_text(dump(out))
    print(f"wrote {args.output}: {len(cases)} cases, {len(lanes)} lanes")


if __name__ == "__main__":
    main()
