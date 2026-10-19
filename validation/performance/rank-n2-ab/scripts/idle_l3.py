#!/usr/bin/env python3
"""Busy threads per L3 complex over a window (default 10 s): idle_l3.py [seconds]."""
import collections, sys, time

def snap():
    d = {}
    for line in open('/proc/stat'):
        if line.startswith('cpu') and line[3].isdigit():
            w = line.split(); v = list(map(int, w[1:])); d[int(w[0][3:])] = (v[3] + v[4], sum(v))
    return d

a = snap(); time.sleep(float(sys.argv[1]) if len(sys.argv) > 1 else 10); b = snap()
busy = {c: 1 - (b[c][0] - a[c][0]) / max(1, b[c][1] - a[c][1]) for c in a}
groups = collections.OrderedDict()
for c in sorted(busy):
    key = open(f'/sys/devices/system/cpu/cpu{c}/cache/index3/shared_cpu_list').read().strip()
    groups.setdefault(key, []).append(c)
rows = sorted(groups.items(), key=lambda kv: sum(busy[c] for c in kv[1]))
for k, cs in rows:
    idle = [c for c in cs if busy[c] < 0.02]
    print(f'{k:20s} busy_threads={sum(busy[c] for c in cs):5.2f} max={max(busy[c] for c in cs):.2f} idle_cpus={idle}')
