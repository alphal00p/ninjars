#!/usr/bin/env python3
"""Compare two `ninjars reduce` JSON outputs: results (values per lane) and other top-level keys.
usage: cmp_reduce.py OLD NEW"""
import json, sys
from decimal import Decimal, getcontext
getcontext().prec = 400
a, b = (json.load(open(p)) for p in sys.argv[1:3])
diff_keys = [k for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
print("top-level keys that differ:", diff_keys)
order = b["result_order"]
for la, lb in zip(sorted(a["results"], key=lambda l: l["lane"]), sorted(b["results"], key=lambda l: l["lane"])):
    worst = []
    vals = {}
    for key, va, vb in zip(order, la["values"], lb["values"]):
        za = [Decimal(x) for x in va]; zb = [Decimal(x) for x in vb]
        d = ((za[0] - zb[0]) ** 2 + (za[1] - zb[1]) ** 2).sqrt()
        m = (zb[0] ** 2 + zb[1] ** 2).sqrt()
        vals[key] = (d, m)
    scale = max(m for d, m in vals.values()) or Decimal(1)
    print("lane", lb["lane"], {k: f"{float(d / scale):.2e}" for k, (d, m) in vals.items()},
          "status", la.get("status"), lb.get("status"),
          {k: (la.get(k), lb.get(k)) for k in la if k not in ("values", "lane") and la.get(k) != lb.get(k)})
