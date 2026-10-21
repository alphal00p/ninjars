#!/usr/bin/env python3
"""Print the three gates and accuracy columns from a results.json (path argument)."""
import json, sys
d = json.load(open(sys.argv[1]))
print("date", d.get("date"), "host", d.get("host"), "width", d.get("batch_width"))
print(f"{'case':26s} {'(a) red':>8s} {'(b) mast':>8s} {'(c) e2e':>8s} | {'nr red':>7s} {'nj red':>7s} {'nr mas':>7s} {'ol mas':>7s} {'nr tot':>8s} {'ml tot':>8s} | {'dNinja':>9s} {'dE2E':>9s}")
ra, rb, rc = [], [], []
for c in d["cases"]:
    a = c["ninjars_identical_us"]["reduction"] / c["ninja_identical_us"]["reduction"]
    b = c["ninjars_identical_us"]["masters"] / c["ninja_identical_us"]["masters"]
    e = c["ninjars_us"]["total"] / c["madloop_us"]["total"]
    ra.append(a); rb.append(b); rc.append(e)
    print(f"{c['id']:26s} {a:8.3f} {b:8.3f} {e:8.3f} | {c['ninjars_identical_us']['reduction']:7.2f} {c['ninja_identical_us']['reduction']:7.2f} {c['ninjars_identical_us']['masters']:7.2f} {c['ninja_identical_us']['masters']:7.2f} {c['ninjars_us']['total']:8.2f} {c['madloop_us']['total']:8.2f} | {c['max_scaled_difference_vs_ninja']:9.2e} {c['max_scaled_difference_end_to_end']:9.2e}")
print(f"range (a) {min(ra):.3f}-{max(ra):.3f}  (b) {min(rb):.3f}-{max(rb):.3f}  (c) {min(rc):.3f}-{max(rc):.3f}")
print("synthetic:")
for s in d["synthetic"]:
    nj = s["ninja_us"]
    print(f"{s['id']:26s} red {s['ninjars_us']['reduction']:8.2f}/{nj['reduction']:8.2f}={s['ninjars_us']['reduction']/nj['reduction']:.3f} mas {s['ninjars_us']['masters']:8.2f}/{nj['masters']:8.2f}={s['ninjars_us']['masters']/nj['masters']:.3f} tot {s['ninjars_us']['total']/nj['total']:.3f} failed {s['failed_lanes']} d {s['max_scaled_difference_vs_ninja']:.2e}")
