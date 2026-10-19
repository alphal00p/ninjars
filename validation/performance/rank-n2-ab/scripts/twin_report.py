#!/usr/bin/env python3
"""Rank n+2 cost against the rank n+1 twins, from the output of twin_cost.sh.

usage: twin_report.py OUT_DIR [--tsv-prefix PREFIX]

Fast engine (case-bench): per pair, the median over repetitions of the
per-point reduction (replayed numerator and masters) and total
(TensorNumerator/MuTensorNumerator + native masters), at widths 1 and 32;
ratio rank n+2 / rank n+1. Instructions per point (ab_icount.rs) likewise.
Generic engine (`ninjars bench` on the n-gon twins): per n and precision,
median ns per point and ratio; the fast engine on the same fixtures.
"""
import argparse
import collections
import json
import math
import pathlib
import statistics as st


def gm(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=pathlib.Path)
    ap.add_argument("--tsv-prefix", type=pathlib.Path)
    args = ap.parse_args()
    out = args.out
    pairs = [l.split("\t") for l in (out / "pairs.tsv").read_text().splitlines() if l.strip()]
    case_name = lambda path: pathlib.Path(path).stem
    timing = collections.defaultdict(list)
    for f in sorted(out.glob("cases-*.json")):
        for c in json.loads(f.read_text())["cases"]:
            for t in c["per_point_timing"]:
                for m in ("reduction", "total"):
                    timing[(c["case"], t["width"], m)].append(t[m]["median_ns_per_point"])
    icount = {}
    if (out / "icount.tsv").exists():
        for line in (out / "icount.tsv").read_text().splitlines():
            f = line.split("\t")
            icount[(f[0], int(f[1]))] = (float(f[2]), float(f[4]))
    rows = []
    for fam, n, hi, lo in pairs:
        h, l = case_name(hi), case_name(lo)
        rec = {"family": fam, "n": int(n), "rank_n2": h, "twin": l}
        for w in (1, 32):
            for m in ("reduction", "total"):
                a, b = timing.get((h, w, m)), timing.get((l, w, m))
                rec[f"{m}_w{w}_n2"] = st.median(a) if a else float("nan")
                rec[f"{m}_w{w}_n1"] = st.median(b) if b else float("nan")
                rec[f"{m}_w{w}_ratio"] = rec[f"{m}_w{w}_n2"] / rec[f"{m}_w{w}_n1"] if a and b else float("nan")
            if (h, w) in icount and (l, w) in icount:
                rec[f"ins_replay_w{w}_ratio"] = icount[(h, w)][0] / icount[(l, w)][0]
                rec[f"ins_live_w{w}_ratio"] = icount[(h, w)][1] / icount[(l, w)][1]
        rows.append(rec)
    print("Fast engine, rank n+2 / rank n+1 twin (same kinematics), median over families")
    print(f"{'n':>2s} {'set':6s} {'pairs':>5s} {'red w1':>7s} {'red w32':>7s} {'tot w1':>7s} {'tot w32':>7s} "
          f"{'ins red w32':>11s} {'ins live w32':>12s}  red w32 range")
    for mu in (False, True):
        for n in range(2, 8):
            sel = [r for r in rows if r["n"] == n and r["family"].endswith("-mu2") == mu]
            if not sel:
                continue
            med = lambda k: st.median([r[k] for r in sel if k in r and not math.isnan(r[k])] or [float("nan")])
            rng = [r["reduction_w32_ratio"] for r in sel if not math.isnan(r["reduction_w32_ratio"])]
            print(f"{n:2d} {'mu2' if mu else 'q-only':6s} {len(sel):5d} {med('reduction_w1_ratio'):7.2f} "
                  f"{med('reduction_w32_ratio'):7.2f} {med('total_w1_ratio'):7.2f} {med('total_w32_ratio'):7.2f} "
                  f"{med('ins_replay_w32_ratio'):11.2f} {med('ins_live_w32_ratio'):12.2f}  "
                  f"{min(rng):.2f}-{max(rng):.2f}" if rng else "")
    print()
    print("Fast engine per pair (width 32): reduction ns/point rank n+2 | twin | ratio; total ratio")
    for r in rows:
        print(f"  {r['rank_n2']:30s} {r['reduction_w32_n2']:10.0f} {r['reduction_w32_n1']:10.0f} "
              f"{r['reduction_w32_ratio']:6.2f}   total {r['total_w32_ratio']:6.2f}")
    # generic engine n-gon twins
    gen = {}
    for f in sorted((out / "generic").glob("ngon-*.json")):
        stem = f.stem  # ngon-n{n}-r{r}-{prec}
        parts = stem.split("-")
        n, r, prec = int(parts[1][1:]), int(parts[2][1:]), parts[3]
        d = json.loads(f.read_text())
        for c in d["cases"]:
            gen[(n, r, prec, c["batch_size"])] = (c["median_ns_per_point"], c["diagnostic"]["failed_lanes"])
    grows = []
    if gen:
        print()
        print("n-gon twins (ninjars bench): ns/point rank n+1 -> rank n+2, ratio")
        for prec in ("f64", "fast", "double", "float128"):
            for n in range(2, 8):
                for w in (1, 8, 32):
                    lo, hi = gen.get((n, n + 1, prec, w)), gen.get((n, n + 2, prec, w))
                    if lo and hi:
                        grows.append((prec, n, w, lo[0], hi[0], hi[0] / lo[0], lo[1], hi[1]))
                        print(f"  {prec:8s} n={n} w{w:<3d} {lo[0]:14.0f} -> {hi[0]:14.0f}  {hi[0] / lo[0]:5.2f}"
                              f"{'  failed lanes ' + str((lo[1], hi[1])) if lo[1] or hi[1] else ''}")
    if args.tsv_prefix:
        cols = list(rows[0].keys())
        with open(f"{args.tsv_prefix}-fast.tsv", "w") as f:
            f.write("\t".join(cols) + "\n")
            for r in rows:
                f.write("\t".join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r.get(c, "")) for c in cols) + "\n")
        with open(f"{args.tsv_prefix}-ngon.tsv", "w") as f:
            f.write("engine_precision\tn\twidth\trank_n1_ns\trank_n2_ns\tratio\tfailed_n1\tfailed_n2\n")
            for g in grows:
                f.write(f"{g[0]}\t{g[1]}\t{g[2]}\t{g[3]:.1f}\t{g[4]:.1f}\t{g[5]:.4f}\t{g[6]}\t{g[7]}\n")


if __name__ == "__main__":
    main()
