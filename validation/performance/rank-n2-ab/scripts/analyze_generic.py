#!/usr/bin/env python3
"""Ratios of the interleaved generic-engine `ninjars bench` A/B (ab_generic.sh).

usage: analyze_generic.py ROOT [--tsv out.tsv]

Per (run name, batch size) the median over runs of `median_ns_per_point`;
the ratio is B/A (design 8, check g: <= 1.05). The representative results of
A and B must agree exactly.
"""
import argparse
import collections
import json
import math
import pathlib
import statistics as st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=pathlib.Path)
    ap.add_argument("--tsv", type=pathlib.Path)
    args = ap.parse_args()
    vals = collections.defaultdict(lambda: {"A": [], "B": []})
    results = collections.defaultdict(set)
    runs = sorted(p for p in args.root.iterdir() if p.is_dir() and p.name[-2:] in ("-A", "-B"))
    for r in runs:
        side = r.name[-1]
        for f in sorted(r.glob("*.json")):
            d = json.loads(f.read_text())
            for c in d["cases"]:
                key = (f.stem, c["batch_size"])
                vals[key][side].append(c["median_ns_per_point"])
                results[key].add(json.dumps(c.get("representative_result")))
    rows = []
    for k in sorted(vals):
        a, b = vals[k]["A"], vals[k]["B"]
        if a and b:
            rows.append((k, st.median(a), st.median(b), st.median(b) / st.median(a), len(a), len(b)))
    na = sum(r.name.endswith("-A") for r in runs)
    nb = sum(r.name.endswith("-B") for r in runs)
    ratios = [r[3] for r in rows]
    gm = math.exp(sum(map(math.log, ratios)) / len(ratios))
    print(f"generic bench: runs A={na} B={nb}; {len(rows)} keys; geomean {gm:.4f}; "
          f"min {min(ratios):.3f}; max {max(ratios):.3f}; > 1.05: {sum(x > 1.05 for x in ratios)}")
    for k, a, b, r, x, y in rows:
        print(f"  {k[0]:24s} w{k[1]:<3d} A {a:14.0f} ns  B {b:14.0f} ns  ratio {r:.4f}")
    same = all(len(v) == 1 for v in results.values())
    print("representative results identical across all runs:", "yes" if same else
          "NO: " + str([k for k, v in results.items() if len(v) != 1]))
    if args.tsv:
        with open(args.tsv, "w") as f:
            f.write("run\tbatch_size\tA_ns_per_point\tB_ns_per_point\tratio\truns_A\truns_B\n")
            for k, a, b, r, x, y in rows:
                f.write(f"{k[0]}\t{k[1]}\t{a:.1f}\t{b:.1f}\t{r:.4f}\t{x}\t{y}\n")


if __name__ == "__main__":
    main()
