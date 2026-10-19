#!/usr/bin/env python3
"""Compare user-space instruction counts per point of two builds (ab_icount.rs).

usage: icount_compare.py --a a-1.tsv [a-2.tsv ...] --b b-1.tsv [b-2.tsv ...] [--tsv out.tsv] [--json out.json]

Per (case, width) and path (replay = replayed pure reduction, live = tensor
numerator + native masters), the median over repetitions of the instructions
per point; the ratio is B/A. Acceptance (design 5.5): within 0.5 %.
The reproducibility of each side is the largest relative deviation of a
repetition from its median.
"""
import argparse
import json
import statistics as st


def load(paths):
    rows = {}
    for p in paths:
        for line in open(p):
            f = line.rstrip("\n").split("\t")
            if len(f) < 6:
                continue
            key = (f[0], int(f[1]))
            rows.setdefault(key, []).append(tuple(map(float, f[2:6])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs="+", required=True)
    ap.add_argument("--b", nargs="+", required=True)
    ap.add_argument("--tsv")
    ap.add_argument("--json")
    args = ap.parse_args()
    a, b = load(args.a), load(args.b)
    keys = sorted(set(a) & set(b))
    out = []
    worst = {"replay": (0, None), "live": (0, None)}
    repro = {"A": 0.0, "B": 0.0}
    for k in keys:
        rec = {"case": k[0], "width": k[1]}
        for side, rows in (("A", a[k]), ("B", b[k])):
            for i, path in ((0, "replay"), (2, "live")):
                xs = [r[i] for r in rows]
                m = st.median(xs)
                rec[f"{path}_ins_{side}"] = m
                repro[side] = max(repro[side], max(abs(x / m - 1) for x in xs))
            rec[f"replay_cyc_{side}"] = st.median(r[1] for r in rows)
            rec[f"live_cyc_{side}"] = st.median(r[3] for r in rows)
        for path in ("replay", "live"):
            r = rec[f"{path}_ins_B"] / rec[f"{path}_ins_A"]
            rec[f"{path}_ratio"] = r
            if abs(r - 1) > abs(worst[path][0] - 1) or worst[path][1] is None:
                worst[path] = (r, k)
        out.append(rec)
    print(f"{len(keys)} (case, width) points; repetitions A={len(args.a)} B={len(args.b)}; "
          f"max deviation of a repetition from its median: A {repro['A'] * 100:.3f}%  B {repro['B'] * 100:.3f}%")
    print(f"{'case':28s} {'w':>3s} {'replay A':>11s} {'replay B':>11s} {'ratio':>7s} {'live A':>11s} {'live B':>11s} {'ratio':>7s}")
    for r in out:
        flag = " *" if abs(r["replay_ratio"] - 1) > 0.005 or abs(r["live_ratio"] - 1) > 0.005 else ""
        print(f"{r['case']:28s} {r['width']:3d} {r['replay_ins_A']:11.0f} {r['replay_ins_B']:11.0f} "
              f"{r['replay_ratio']:7.4f} {r['live_ins_A']:11.0f} {r['live_ins_B']:11.0f} {r['live_ratio']:7.4f}{flag}")
    for path in ("replay", "live"):
        rs = [r[f"{path}_ratio"] for r in out]
        print(f"{path}: min {min(rs):.4f} max {max(rs):.4f}; outside 0.5%: "
              f"{sum(abs(x - 1) > 0.005 for x in rs)}; worst {worst[path][0]:.4f} at {worst[path][1]}")
    if args.tsv:
        cols = list(out[0].keys())
        with open(args.tsv, "w") as f:
            f.write("\t".join(cols) + "\n")
            for r in out:
                f.write("\t".join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"points": len(keys), "reproducibility": repro,
                       "worst": {p: {"ratio": w[0], "key": list(w[1])} for p, w in worst.items()},
                       "outside_0_5_percent": {p: sum(abs(r[f'{p}_ratio'] - 1) > 0.005 for r in out)
                                               for p in ("replay", "live")}}, f, indent=1)


if __name__ == "__main__":
    main()
