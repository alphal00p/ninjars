import json,sys
for l in open(sys.argv[1]):
    r=json.loads(l)
    if 'rows' not in r: print(r); continue
    print(r['lane'], 'bits', r['bits'], 'exact eps0', r['exact'][0], 'exact_2bits_eps0', r['exact_2bits_eps0'][:60], '...', 'eps0abs %.4e'%r['eps0_abs'], r['fast_statuses'])
    for x in r['rows']: print('   %-34s eps0/|eps0| %.3e  eps-1/|eps0| %.3e  eps-2 %.3e'%(x['name'],x['err_over_eps0'][0],x['err_over_eps0'][1],x['err_over_eps0'][2]))
