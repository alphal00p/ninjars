#!/usr/bin/env python3
"""Check every identity of identity/manifest.json on the reference values themselves.

For V6a/V6b: |I[lhs] - sum_k c_k I[rhs_k]| / scale(lhs); for V7a/V7b: |I[case] - I[equals]|.
Evaluated with the published references (identity/reference) and, as an independent
measure of the harness, with any run directory given by --run (e.g. coli-quad).

    verify_identities.py --manifest identity/manifest.json --refs identity/reference [--run DIR]... [--json OUT]"""
import argparse
import json
import pathlib
import sys
from decimal import Decimal

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from assemble import dec, cabs  # noqa: E402


def table(d):
    out = {}
    for f in pathlib.Path(d).glob("*.json"):
        doc = json.loads(f.read_text())
        name = doc.get("case")
        for i, l in enumerate(doc.get("lanes", [])):
            out[(name, i)] = dec(l)
    return out


def residual(t, lhs, rhs):
    a = t.get((lhs["case"], lhs["lane"]))
    if a is None or any(t.get((r["case"], r["lane"])) is None for r in rhs):
        return None
    acc = [[Decimal(0), Decimal(0)] for _ in range(3)]
    for r in rhs:
        v = t[(r["case"], r["lane"])]
        c = Decimal(r.get("coefficient", 1))
        for k in range(3):
            acc[k][0] += c * v[k][0]
            acc[k][1] += c * v[k][1]
    scale = max(cabs(x) for x in a) or Decimal(1)
    return float(max(cabs((a[k][0] - acc[k][0], a[k][1] - acc[k][1])) for k in range(3)) / scale)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=pathlib.Path, required=True)
    ap.add_argument("--refs", type=pathlib.Path, required=True)
    ap.add_argument("--run", type=pathlib.Path, action="append", default=[])
    ap.add_argument("--json", type=pathlib.Path)
    a = ap.parse_args()
    man = json.loads(a.manifest.read_text())
    sources = {"reference": table(a.refs)} | {r.name: table(r) for r in a.run}
    out = {}
    for kind in ("V6a", "V6b", "V7a", "V7b"):
        rows = []
        for e in man[kind]:
            if kind in ("V6a", "V6b"):
                lhs, rhs = e["lhs"], e["rhs"]
            else:
                lhs, rhs = {"case": e["case"], "lane": e["lane"]}, [dict(e["equals"], coefficient=1)]
            rows.append({"lhs": lhs, **{s: residual(t, lhs, rhs) for s, t in sources.items()}})
        out[kind] = rows
        for s in sources:
            vals = [r[s] for r in rows if r[s] is not None]
            if vals:
                print(f"{kind:4s} {s:16s} identities {len(vals):3d}  max residual {max(vals):.2e}")
    if a.json:
        a.json.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
