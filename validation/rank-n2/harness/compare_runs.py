#!/usr/bin/env python3
"""Lane-by-lane comparison of result files (harness runs, reference files, Ninja results).

    compare_runs.py REF_DIR OTHER_DIR [--json OUT] [--quiet]

For every <case>.json (or <case>.collier.json) in REF_DIR that OTHER_DIR also has, lanes
are matched by "id" (falling back to position) and the scaled difference
    max_k |other_k - ref_k| / max_k |ref_k|,   k in (eps0, eps_m1, eps_m2)
is computed from the 40-digit "laurent_decimal" strings where both sides have them,
otherwise from the binary64 pairs."""
import argparse
import json
import math
import pathlib
from decimal import Decimal, getcontext

getcontext().prec = 60
KEYS = ("eps0", "eps_m1", "eps_m2")


def load(d, name):
    for nm in (f"{name}.json", f"{name}.collier.json"):
        p = pathlib.Path(d) / nm
        if p.exists():
            return json.loads(p.read_text())
    return None


def values(lane):
    dec = lane.get("laurent_decimal")
    if dec:
        return [(Decimal(dec[k][0]), Decimal(dec[k][1])) for k in KEYS], True
    out = []
    for k in KEYS:
        v = lane.get(k)
        if v is None or v[0] is None:
            return None, False
        out.append((Decimal(v[0]), Decimal(v[1])))
    return out, False


def cabs(a):
    return (a[0] * a[0] + a[1] * a[1]).sqrt()


def lane_diff(ref, other):
    r, rd = values(ref)
    o, od = values(other)
    if r is None or o is None:
        return None, None
    scale = max(cabs(x) for x in r)
    if scale == 0:
        scale = Decimal(1)
    d = max(cabs((a[0] - b[0], a[1] - b[1])) for a, b in zip(r, o)) / scale
    return float(d), rd and od


def compare(ref_dir, other_dir):
    rows = []
    for p in sorted(pathlib.Path(ref_dir).glob("*.json")):
        name = p.name[:-len(".collier.json")] if p.name.endswith(".collier.json") else p.stem
        ref = json.loads(p.read_text())
        other = load(other_dir, name)
        if other is None or "lanes" not in ref:
            continue
        olanes = {l.get("id"): l for l in other["lanes"]}
        for i, rl in enumerate(ref["lanes"]):
            ol = olanes.get(rl.get("id"))
            if ol is None and i < len(other["lanes"]):
                ol = other["lanes"][i]
            if ol is None or "unsupported" in ol or "unsupported" in rl:
                continue
            d, dec = lane_diff(rl, ol)
            if d is None:
                continue
            rows.append({"case": name, "lane": i, "id": rl.get("id"), "diff": d, "decimal": dec})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref", type=pathlib.Path)
    ap.add_argument("other", type=pathlib.Path)
    ap.add_argument("--json", type=pathlib.Path)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    rows = compare(a.ref, a.other)
    if not a.quiet:
        for r in rows:
            print(f"{r['case']:40s} l{r['lane']} {r['diff']:.2e}{'' if r['decimal'] else ' (binary64)'}")
    if rows:
        ds = sorted(r["diff"] for r in rows)
        print(f"lanes {len(rows)}  median {ds[len(ds) // 2]:.2e}  max {ds[-1]:.2e}")
    if a.json:
        a.json.write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
