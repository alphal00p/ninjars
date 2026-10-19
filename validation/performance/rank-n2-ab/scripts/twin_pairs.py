#!/usr/bin/env python3
"""List rank n+2 cases with their rank n+1 twin (same kinematics), as TSV:
family, n, rank n+2 file, rank n+1 file (paths relative to ROOT).

usage: twin_pairs.py ROOT

- syn: validation/benchmark/synthetic-rank-n2/syn-n{n}-r{n+2} against
  validation/benchmark/synthetic/syn-n{n}-r{n+1} (n = 2..6);
- the rank n+2 suite validation/rank-n2/cases/rn2-<family>-n{n}-r{n+2}[-mu2]
  against validation/rank-n2/controls/cases/rn2-<family>-n{n}-r{n+1}[-mu2]
  (n = 2..7 where both exist).
"""
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
rows = []
for n in range(2, 7):
    hi = f"validation/benchmark/synthetic-rank-n2/syn-n{n}-r{n + 2}.json"
    lo = f"validation/benchmark/synthetic/syn-n{n}-r{n + 1}.json"
    if (root / hi).exists() and (root / lo).exists():
        rows.append(("syn", n, hi, lo))
for fam in ("int-complex", "int-real", "int-mixed", "shifted-mixed", "phys-complex", "phys-real"):
    for suffix in ("", "-mu2"):
        for n in range(2, 8):
            hi = f"validation/rank-n2/cases/rn2-{fam}-n{n}-r{n + 2}{suffix}.json"
            lo = f"validation/rank-n2/controls/cases/rn2-{fam}-n{n}-r{n + 1}{suffix}.json"
            if (root / hi).exists() and (root / lo).exists():
                rows.append((fam + suffix, n, hi, lo))
for r in rows:
    print("\t".join(map(str, r)))
