#!/usr/bin/env python3
"""Address span of the hot non-generic rank <= n+1 functions of the replayed
reduction (results/layout-syn-n2-r2.txt, item 5) in a CLI binary, and the
rank n+2 code linked inside that span.

usage: hot_span.py BINARY [--list]
"""
import re
import subprocess
import sys

HOT = [r"<ninjars::fast::engine::FastAmplitude>::refresh_active$",
       r"<ninjars::fast::engine::FastAmplitude>::prepare$",
       r"<ninjars::fast::engine::Calls>::prepare_out$",
       r"<ninjars::fast::geometry::Basis>::from_single$",
       r"<ninjars::fast::geometry::Basis>::complete$",
       r"ninjars::fast::geometry::reference_vectors_ok$",
       r"ninjars::fast::subtractions::tadpole_from_bubble_x1$",
       r"<ninjars::fast::replay::ReplayNumerator as ninjars::fast::numerator::FastNumerator>::(?!\w*_x2)\w+$",
       r"<ninjars::fast::replay::ReplayMasters as ninjars::fast::masters::FastMasters>::evaluate$"]
X2 = re.compile(r"engine_x2|bases_x2|subtractions_x2|subtractions_series_x2|_x2\b|RankN2")

out = subprocess.run(["nm", "-n", "-S", "-C", "--defined-only", sys.argv[1]],
                     capture_output=True, text=True).stdout
syms = []
for line in out.splitlines():
    p = line.split(" ", 3)
    if len(p) == 4 and p[2] in "tTwW":
        syms.append((int(p[0], 16), int(p[1], 16), p[3]))
hot = [s for s in syms if any(re.search(h, s[2]) for h in HOT)]
lo = min(s[0] for s in hot)
hi = max(s[0] + s[1] for s in hot)
x2 = [s for s in syms if lo <= s[0] < hi and X2.search(s[2])]
print(f"{len(hot)} hot functions span {lo:#x}..{hi:#x} = {(hi - lo) / 1024:.0f} KiB; "
      f"rank n+2 code inside: {len(x2)} functions, {sum(s[1] for s in x2) / 1024:.0f} KiB")
if "--list" in sys.argv:
    for s in hot:
        print(f"  hot {s[0]:#x} {s[1]:6d} {s[2][:110]}")
    for s in x2:
        print(f"  x2  {s[0]:#x} {s[1]:6d} {s[2][:110]}")
