#!/usr/bin/env python3
"""Per physical core busy fraction over a window, for CPUs 0-127 and their SMT siblings 256-383.
usage: idle_pairs.py [seconds] [lo] [hi]"""
import sys, time
def snap():
    d = {}
    for line in open('/proc/stat'):
        if line.startswith('cpu') and line[3].isdigit():
            w = line.split(); v = list(map(int, w[1:])); d[int(w[0][3:])] = (v[3] + v[4], sum(v))
    return d
sec = float(sys.argv[1]) if len(sys.argv) > 1 else 10
lo = int(sys.argv[2]) if len(sys.argv) > 2 else 0
hi = int(sys.argv[3]) if len(sys.argv) > 3 else 127
a = snap(); time.sleep(sec); b = snap()
busy = {c: 1 - (b[c][0] - a[c][0]) / max(1, b[c][1] - a[c][1]) for c in a}
rows = []
for c in range(lo, hi + 1):
    sib = [int(x) for x in open(f'/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list').read().strip().replace('-', ',').split(',')]
    other = [s for s in sib if s != c]
    l3 = open(f'/sys/devices/system/cpu/cpu{c}/cache/index3/shared_cpu_list').read().strip()
    rows.append((busy[c] + sum(busy[s] for s in other), c, other, busy[c], [busy[s] for s in other], l3))
rows.sort()
for tot, c, other, bc, bo, l3 in rows[:40]:
    print(f'cpu {c:3d} busy {bc:.3f} sibling {other} busy {[round(x,3) for x in bo]} l3 {l3}')
