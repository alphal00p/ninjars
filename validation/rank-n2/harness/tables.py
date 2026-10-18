#!/usr/bin/env python3
"""Markdown tables of the assembled sets (validation/rank-n2/summary/<set>.json).

    tables.py SUMMARY.json [--by family|n] [--title T]

Per group: lanes, tiers A/B/C, sources, the largest uncertainty, verified_to and input
ambiguity, and the largest error of COLLIER COLI / DD quad, binary64 COLI and Ninja
against the published value."""
import argparse
import collections
import json
import pathlib
import re


def group_of(r, by):
    name = r["case"]
    m = re.match(r"(?:rn2-)?([a-z0-9]+)-([a-z]+)-n(\d)", name)
    fam = f"{m.group(1)}-{m.group(2)}" if m and not name.startswith("syn") else re.sub(r"-n\d.*", "", name)
    if r.get("mu_powers"):
        fam += f" mu^{2 * r['mu_powers']}"
    return (fam, r["n"]) if by == "family" else (r["n"], r["rank"] - r["n"])


def fmt(v):
    return "-" if v is None else (f"{v:.0e}" if v else "0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary", type=pathlib.Path)
    ap.add_argument("--by", default="family", choices=["family", "n"])
    a = ap.parse_args()
    rows = json.loads(a.summary.read_text())
    agg = collections.defaultdict(list)
    for r in rows:
        agg[group_of(r, a.by)].append(r)

    def mx(lst, k):
        v = [x.get(k) for x in lst if x.get(k) is not None]
        return max(v) if v else None
    head = "| family | n |" if a.by == "family" else "| n | rank - n |"
    print(head + " lanes | A/B/C | source | max unc. | max verified_to | max input amb. | COLI quad err | DD quad err "
          "| COLI binary64 err | Ninja err |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    tot = collections.Counter()
    for key, lst in sorted(agg.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        t = collections.Counter(x["tier"] for x in lst)
        tot.update(t)
        src = collections.Counter({"mp": "mp", "reduction-exact": "red", "coli-quad": "coli", "dd-quad": "dd"}[x["source"]]
                                  for x in lst)
        print(f"| {key[0]} | {key[1]} | {len(lst)} | {t['A']}/{t['B']}/{t['C']} | "
              + ",".join(f"{k}{v}" for k, v in sorted(src.items())) + " | "
              + " | ".join(fmt(mx(lst, k)) for k in ("uncertainty", "verified_to", "input_ambiguity", "d_coli-quad",
                                                     "d_dd-quad", "d_coli-double", "ninja")) + " |")
    print(f"\n{len(rows)} lanes: tier A {tot['A']}, B {tot['B']}, C {tot['C']}")


if __name__ == "__main__":
    main()
