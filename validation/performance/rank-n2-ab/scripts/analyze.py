#!/usr/bin/env python3
"""Ratios of an interleaved A/B (or A/A) benchmark, design section 8.

usage: analyze.py [--tsv out.tsv] [--json out.json] [--label L] ROOT [ROOT ...]

Each ROOT holds the run directories NN-A / NN-B written by ab.sh. Per key
(source, case, width, metric) the statistic is the median over runs of the
per-run `median_ns_per_point`; the ratio is B/A. Several ROOTs (replicates on
different cores) are analysed separately and combined by the geometric mean
of their per-key ratios.

Metrics (design 8, step 5): case-bench reduction and total; pipeline-bench
reduction, masters and total. Acceptance: every ratio <= 1.03, ratios in
(1.03, 1.05] need a rerun, none > 1.05, and the geometric mean of all
reduction and total ratios <= 1.01.

Also checked: the per-lane results (pipeline `lane_results`, case-bench
`reference_check`) of all runs of both sides are identical.
"""
import argparse
import collections
import json
import math
import pathlib
import statistics as st

METRICS = {"case": ("reduction", "total"), "pipe": ("reduction", "masters", "total")}


def load(root):
    vals = collections.defaultdict(lambda: {"A": [], "B": []})
    lanes = collections.defaultdict(lambda: {"A": set(), "B": set()})
    runs = sorted(p for p in root.iterdir() if p.is_dir() and p.name[-2:] in ("-A", "-B"))
    for r in runs:
        side = r.name[-1]
        for f in sorted(r.glob("pipeline-*.json")):
            d = json.loads(f.read_text())
            name = f.stem[len("pipeline-"):]
            for t in d["per_point_timing"]:
                for m in METRICS["pipe"]:
                    vals[("pipe", name, t["width"], m)][side].append(t[m]["median_ns_per_point"])
                lanes[("pipe", name, t["width"])][side].add(json.dumps(t.get("lane_results"), sort_keys=True))
        f = r / "cases.json"
        if f.exists():
            for c in json.loads(f.read_text())["cases"]:
                for t in c["per_point_timing"]:
                    for m in METRICS["case"]:
                        if m in t:
                            vals[("case", c["case"], t["width"], m)][side].append(t[m]["median_ns_per_point"])
                    lanes[("case", c["case"], t["width"])][side].add(json.dumps(t.get("reference_check"), sort_keys=True))
    return runs, vals, lanes


def gm(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def rel_spread(xs):
    """Half the relative interquartile range of the per-run medians."""
    if len(xs) < 4:
        return float("nan")
    q = st.quantiles(xs, n=4)
    return (q[2] - q[0]) / (2 * st.median(xs))


def analyse(root):
    runs, vals, lanes = load(root)
    out = {}
    for k, v in vals.items():
        if v["A"] and v["B"]:
            a, b = st.median(v["A"]), st.median(v["B"])
            out[k] = dict(a=a, b=b, ratio=b / a, na=len(v["A"]), nb=len(v["B"]),
                          spread_a=rel_spread(v["A"]), spread_b=rel_spread(v["B"]),
                          runs_a=v["A"], runs_b=v["B"])
    differ = [k for k, v in lanes.items() if len(v["A"] | v["B"]) != 1]
    return runs, out, differ


def summary(ratios, label):
    lines = []
    for src in ("case", "pipe"):
        for m in METRICS[src]:
            xs = [r for k, r in ratios.items() if k[0] == src and k[3] == m]
            if xs:
                lines.append(f"  {src:4s} {m:9s} n={len(xs):3d} geomean={gm(xs):.4f} "
                             f"min={min(xs):.3f} max={max(xs):.3f} >1.03: {sum(x > 1.03 for x in xs)} "
                             f">1.05: {sum(x > 1.05 for x in xs)}")
    rt = [r for k, r in ratios.items() if k[3] in ("reduction", "total")]
    allr = list(ratios.values())
    head = (f"{label}: keys={len(ratios)} geomean(reduction+total, n={len(rt)})={gm(rt):.4f} "
            f"geomean(all)={gm(allr):.4f} max={max(allr):.3f} "
            f">1.03: {sum(x > 1.03 for x in allr)} >1.05: {sum(x > 1.05 for x in allr)}")
    return head, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", type=pathlib.Path)
    ap.add_argument("--tsv", type=pathlib.Path)
    ap.add_argument("--json", type=pathlib.Path)
    ap.add_argument("--label", default="B/A")
    args = ap.parse_args()
    per_root = {}
    report = {"label": args.label, "replicates": {}}
    for root in args.roots:
        runs, res, differ = analyse(root)
        per_root[root.name] = res
        na = sum(r.name.endswith("-A") for r in runs)
        nb = sum(r.name.endswith("-B") for r in runs)
        ratios = {k: v["ratio"] for k, v in res.items()}
        head, lines = summary(ratios, f"{args.label} [{root.name}] runs A={na} B={nb}")
        print(head)
        print("\n".join(lines))
        sp = [v["spread_a"] for v in res.values() if not math.isnan(v["spread_a"])]
        if sp:
            print(f"  run-to-run spread of A (half IQR / median): median {st.median(sp) * 100:.2f}%  "
                  f"max {max(sp) * 100:.2f}%")
        print(f"  per-lane results identical across all A and B runs: "
              f"{'yes' if not differ else 'NO: ' + str(differ[:6])}")
        report["replicates"][root.name] = {
            "runs_a": na, "runs_b": nb, "summary": head, "lane_results_identical": not differ,
            "differing_keys": [list(map(str, k)) for k in differ],
        }
    keys = sorted(set.intersection(*(set(r) for r in per_root.values())))
    combined = {k: gm([per_root[n][k]["ratio"] for n in per_root]) for k in keys}
    if len(per_root) > 1:
        head, lines = summary(combined, f"{args.label} [combined: geometric mean over {len(per_root)} replicates]")
        print(head)
        print("\n".join(lines))
    allr = list(combined.values())
    rt = [r for k, r in combined.items() if k[3] in ("reduction", "total")]
    report["combined"] = {
        "keys": len(combined), "geomean_reduction_total": gm(rt), "geomean_all": gm(allr),
        "max": max(allr), "above_1_03": sum(x > 1.03 for x in allr), "above_1_05": sum(x > 1.05 for x in allr),
    }
    flagged = [(k, r) for k, r in sorted(combined.items(), key=lambda x: -x[1]) if r > 1.03]
    print("ratios > 1.03 (combined):" + ("" if flagged else " none"))
    for k, r in flagged:
        per = "  ".join(f"{n}={per_root[n][k]['ratio']:.3f}" for n in per_root)
        print(f"  {k[0]} {k[1]} w{k[2]} {k[3]}: {r:.3f}  ({per})")
    print("top 10:", ", ".join(f"{k[1]} w{k[2]} {k[3]} {r:.3f}"
                               for k, r in sorted(combined.items(), key=lambda x: -x[1])[:10]))
    print("bottom 5:", ", ".join(f"{k[1]} w{k[2]} {k[3]} {r:.3f}"
                                 for k, r in sorted(combined.items(), key=lambda x: x[1])[:5]))
    report["above_1_03"] = [{"key": list(map(str, k)), "ratio": r} for k, r in flagged]
    if args.tsv:
        with open(args.tsv, "w") as f:
            names = list(per_root)
            f.write("source\tcase\twidth\tmetric\tratio\t" + "\t".join(
                f"{n}:A_ns\t{n}:B_ns\t{n}:ratio\t{n}:spreadA\t{n}:spreadB" for n in names) + "\n")
            for k in keys:
                row = [k[0], k[1], str(k[2]), k[3], f"{combined[k]:.4f}"]
                for n in names:
                    v = per_root[n][k]
                    row += [f"{v['a']:.1f}", f"{v['b']:.1f}", f"{v['ratio']:.4f}",
                            f"{v['spread_a']:.4f}", f"{v['spread_b']:.4f}"]
                f.write("\t".join(row) + "\n")
    if args.json:
        args.json.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
