#!/usr/bin/env python3
"""Compare the references with the review's 60-digit-preprocessing values (provenance/review-hpref)
and with the phase-1 suite references (provenance/phase1-reference, optional).

    compare_hpref.py --hpref DIR --refs DIR --work WORK/suite [--phase1 DIR] [--json OUT]

Per lane (scaled differences, see assemble.py):
  compat_vs_hpref   this harness with the review's driver input (repr decimals, --input-rounding
                    repr) against hpref: the port of the review's collier_hp.py
  input_rounding    COLI quad with repr inputs against COLI quad with exact inputs: the error
                    the shortest-decimal driver input put into hpref
  collier_error     COLI quad (exact inputs) against the new reference: COLLIER's own error
  new_vs_hpref      the new reference against hpref; it is bounded by
                    input_rounding + collier_error (+ the new uncertainty)."""
import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from assemble import dec, diff, cabs  # noqa: E402


def lanes(d, name):
    for nm in (f"{name}.json", f"{name}.collier.json"):
        p = pathlib.Path(d) / nm
        if p.exists():
            return {l["id"]: l for l in json.loads(p.read_text())["lanes"]}
    return {}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hpref", type=pathlib.Path, required=True)
    ap.add_argument("--refs", type=pathlib.Path, required=True)
    ap.add_argument("--work", type=pathlib.Path, required=True)
    ap.add_argument("--phase1", type=pathlib.Path)
    ap.add_argument("--json", type=pathlib.Path)
    a = ap.parse_args()
    rows = []
    for f in sorted(a.hpref.glob("rn2-*.json")):
        name = json.loads(f.read_text())["case"]
        hp = lanes(a.hpref, name)
        ref = lanes(a.refs, name)
        compat = lanes(a.work / "coli-quad-repr", name)
        coli = lanes(a.work / "coli-quad", name)
        old = lanes(a.phase1, name) if a.phase1 else {}
        for lid, h in hp.items():
            r = ref.get(lid)
            if r is None:
                continue
            rv = dec(r)
            scale = max(cabs(x) for x in rv)
            row = {"case": name, "id": lid, "source": r["source"], "tier": r["tier"],
                   "uncertainty": r["uncertainty"], "uncertainty_independent": r["uncertainty_independent"],
                   "new_vs_hpref": diff(dec(h), rv, scale),
                   "compat_vs_hpref": diff(dec(compat[lid]), dec(h), scale) if lid in compat else None,
                   "input_rounding": diff(dec(compat[lid]), dec(coli[lid]), scale) if lid in compat and lid in coli else None,
                   "collier_error": diff(dec(coli[lid]), rv, scale) if lid in coli else None}
            if lid in old:
                row["new_vs_phase1"] = diff(dec(old[lid]), rv, scale)
                row["phase1_uncertainty"] = old[lid].get("uncertainty")
            bound = (row["input_rounding"] or 0) + (row["collier_error"] or 0) + (r["uncertainty"] or 0)
            row["within_explained_budget"] = row["new_vs_hpref"] <= 1.01 * bound + 1e-36
            rows.append(row)
    for r in rows:
        print(f"{r['id']:36s} {r['source']:16s} new-hpref {r['new_vs_hpref']:.1e}  compat-hpref "
              f"{r['compat_vs_hpref']:.1e}  input {r['input_rounding']:.1e}  collier {r['collier_error']:.1e}  "
              f"unc {r['uncertainty']:.1e}  ok={r['within_explained_budget']}"
              + (f"  phase1 {r['new_vs_phase1']:.1e} (stated {r['phase1_uncertainty']:.1e})" if "new_vs_phase1" in r else ""))
    print(f"lanes {len(rows)}; max compat-hpref {max(r['compat_vs_hpref'] for r in rows):.2e}; "
          f"max new-hpref {max(r['new_vs_hpref'] for r in rows):.2e}; "
          f"all within budget: {all(r['within_explained_budget'] for r in rows)}")
    if a.json:
        a.json.write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
