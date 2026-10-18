import json, sys
from decimal import Decimal, getcontext
getcontext().prec = 60
recs = [json.loads(l) for l in open(sys.argv[1])]
OLD = {0: ('-4.31269666370595857744e-17', '3.79346580325708373964e-17'),
       1: ('-5.81770124992702747462e-17', '3.19195457337303775317e-17'),
       2: ('-6.16594826112180847195e-17', '-1.06572031691224518515e-16')}
D = lambda s: Decimal(s)
def dabs(re, im): return (re * re + im * im).sqrt()
C = lambda v: [complex(*z) for z in v]
for r in recs:
    if r['kind'] == 'master_check':
        print('master_check bits', r['bits'], 'lane', r['lane'], 'bad_calls', r['bad_calls'],
              [(f['call'], f['family'], f['info']) for f in r['flagged']])
lanes = sorted({r['lane'] for r in recs if 'lane' in r})
for lane in lanes:
    R = [r for r in recs if r.get('lane') == lane]
    gen = {(r.get('carrier'), r.get('bits'), r['masters']): r for r in R if r['kind'] == 'generic'}
    for k, r in sorted(gen.items(), key=str):
        if 'error' in r: print('lane', lane, k, 'ERROR', r['error'][:120])
    raw = {b: gen.get(('Float', b, 'hp_raw')) for b in (256, 400)}
    if not all(x and 'hp' in x for x in raw.values()):
        print('lane', lane, 'missing raw'); continue
    hp = {b: (D(raw[b]['hp'][0][0]), D(raw[b]['hp'][0][1])) for b in raw}
    ref = hp[400]; aref = dabs(*ref)
    print('=== lane %d exact eps0 (Float400 raw) = %s %s i   tracked bits 256:%s 400:%s' % (
        lane, f'{ref[0]:.20e}', f'{ref[1]:+.20e}', raw[256]['hp'][0][2:], raw[400]['hp'][0][2:]))
    d = dabs(hp[256][0] - ref[0], hp[256][1] - ref[1]) / aref
    old = (D(OLD[lane][0]), D(OLD[lane][1]))
    do = dabs(old[0] - ref[0], old[1] - ref[1]) / aref
    print('    |F256raw - F400raw|/|eps0| = %.3e   |old substituted - F400raw|/|eps0| = %.3e' % (d, do))
    for k2 in [('Float', 256, 'hpfix'), ('Float', 400, 'hpfix'), ('Float', 256, 'qp_scalar_all'), ('Float', 400, 'qp_scalar_all'), ('DoubleFloat', None, 'hp')]:
        if k2 in gen and 'hp' in gen[k2]:
            z = (D(gen[k2]['hp'][0][0]), D(gen[k2]['hp'][0][1]))
            print('    %-34s |x - F400raw|/|eps0| = %.3e' % (str(k2), dabs(z[0] - ref[0], z[1] - ref[1]) / aref))
    E = C(raw[400]['values'])
    e0 = abs(E[0])
    b64 = [r for r in R if r['kind'] == 'binary64'][0]
    ex = [r for r in R if r['kind'] == 'binary64_exact_masters']
    cands = [('Ninja+OneLOop (reference)', C(b64['ninja'])),
             ('fast, native masters', C(b64['fast_native'])),
             ('fast, Ninja OneLOop masters', C(b64['fast_ninja_masters']))]
    if ex: cands.append(('fast, exact masters (rounded)', C(ex[0]['fast_exact_masters'])))
    for k2 in [('Float', 256, 'ninja'), ('Float', 256, 'native')]:
        if k2 in gen and 'values' in gen[k2]: cands.append(('generic F256, %s masters' % k2[2], C(gen[k2]['values'])))
    print('    statuses: native %s ninja-masters %s (unmatched %s) exact %s' % (
        b64['fast_native_status'], b64['fast_ninja_masters_status'], b64['fast_ninja_masters_unmatched'],
        (ex[0]['status'], ex[0]['unmatched']) if ex else None))
    for name, v in cands:
        print('    %-34s eps0 %.3e   eps-1 %.3e   eps-2 %.3e  (/|exact eps0|)' % (
            name, abs(v[0] - E[0]) / e0, abs(v[1] - E[1]) / e0, abs(v[2] - E[2]) / e0))
