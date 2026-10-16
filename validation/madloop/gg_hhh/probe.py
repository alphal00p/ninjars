#!/usr/bin/env python3
"""Probe one unsquared gg->HHH MadLoop graph, never its counterterm driver."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'references/MG5_aMC_v3_7_0').is_dir())


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph',type=int,required=True)
    parser.add_argument('--exports-dir',type=Path,default=ROOT/'target/reference/madloop-gg-hhh-external')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    exports=args.exports_dir.resolve(); out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    metadata=json.loads((exports/'graphs.json').read_text())
    graph=next(g for g in metadata['graphs'] if g['filter_id']==args.graph)
    assert graph['multiplier']==1
    export=exports/graph['directory']; source=export/'SubProcesses/P0_gg_hhh'
    work=out/f'probe-{args.graph}';work.mkdir(exist_ok=True)
    commands=[]
    def run(command,cwd,stdout):
        commands.append({'argv':list(map(str,command)),'cwd':str(cwd)})
        subprocess.run(list(map(str,command)),cwd=cwd,stdout=stdout,stderr=subprocess.STDOUT,check=True)
    calls=[s for s in graph['loop_calls'] if not s.startswith('#')]
    assert len(calls)==1
    call=calls[0]; n=graph['loop_propagators']
    match=re.match(r'CALL %\(proc_prefix\)sLOOP_'+str(n)+'_'+str(n)+r'\(1,([1-7],){'+str(n)+r'}',call)
    assert match,call
    order=list(map(int,match.group(0).split('(1,')[1].strip(',').split(',')))
    assert len(order)==n and len(set(order))==n and {1,2}.issubset(order)
    couplings=re.findall(r'(GC_\d+),MP__\1',call)
    assert len(couplings)==n and couplings.count('GC_5')==2 and couplings.count('GC_37')==n-2
    amp_match=re.search(r',([0-9]+),1,1,([0-9]+),AMPL\(1,\2\),S\(\2\)\)$',call)
    assert amp_match and int(amp_match[1])==n
    amplitude=int(amp_match[2])
    color=[c for row in graph['color_amplitudes'] for c in row if c['amplitude_number']==amplitude]
    assert color==[{'fermion_factor':-1,'rational_color_coefficient':'1','imaginary':False,'nc_power':0,'amplitude_number':amplitude}]
    assert "(('Tr', (1, 2)),)" in graph['loop_color_basis']
    num=(source/'loop_num.f').read_text();ct=(source/'CT_interface.f').read_text()
    nwave=int(re.search(r'PARAMETER \(NWAVEFUNCS=([0-9]+)\)',num)[1])
    nct=int(re.search(r'PARAMETER \(NLOOPAMPS=[0-9]+, NCTAMPS=([0-9]+)\)',(source/'loop_matrix.f').read_text())[1])
    assert amplitude==nct+1
    for key,value in [('NEXTERNAL',5),('NLOOPAMPS',amplitude),('NWAVEFUNCS',nwave),('MAXLCOUPLINGS',n),('NCOMB',4)]:
        assert f'PARAMETER ({key}={value})' in num,(key,value)
    assert 'IF(.NOT.MP_DONE.AND.CTMODE.EQ.0)' in num
    assert f'PARAMETER (NLOOPAMPS={amplitude}, NCTAMPS={nct})' in (source/'loop_matrix.f').read_text()
    assert 'PARAMETER (NORMALIZATION = 1.D0/(16.D0*3.14159265358979323846D0*' in ct
    assert 'CALL CTSXCUT(CTMODE,LSCALE,MU_R,NLOOPLINE,ML5_0_LOOPNUM' in ct
    seeds=set(re.findall(r'CALL (LCUT_AF|LCUT_F)\(',num));assert len(seeds)==1
    bridge=[c for section in graph['external_wavefunction_calls'] for c in section if re.match(r'CALL SSSS?1_1',c)]
    trees={i:str(i) for i in range(1,6)}
    leaves={i:[i] for i in range(1,6)}
    tree_couplings=[]
    for tree_call in bridge:
        assert re.fullmatch(r'CALL SSSS?1_1\((?:W\(1,[1-7]\),){2,3}(GC_30|GC_HHHH),mdl_MH,mdl_WH,W\(1,[6-7]\)\)',tree_call)
        ids=list(map(int,re.findall(r'W\(1,([1-7])\)',tree_call)))
        child=ids[:-1];destination=ids[-1]
        assert destination not in trees and all(i in trees for i in child)
        trees[destination]='H['+','.join(sorted(trees[i] for i in child))+']'
        leaves[destination]=sorted(j for i in child for j in leaves[i])
        tree_couplings+=re.findall(r'GC_30|GC_HHHH',tree_call)
    assert sorted(j for i in order for j in leaves[i])==[1,2,3,4,5]
    assert len(bridge)==nwave-5
    integrated=call.replace('%(proc_prefix)s','ML5_0_').replace(f'AMPL(1,{amplitude}),S({amplitude})','RES,STABLE')
    integrated=re.sub(r'MP__(GC_\d+)',r'CMPLX(\1,KIND=16)',integrated)
    integrated=integrated.replace('CMPLX(MP__mdl_MT,KIND=16)','CMPLX(mdl_MT,KIND=16)')
    replacements={'NLOOP':str(n),'NWAVEFUNCS':str(nwave),'NLOOPAMPS':str(amplitude),'AMPLNUM':str(amplitude),'WAVEFUNCTION_ORDER':','.join(map(str,order)),
                  'BRIDGE_COUPLING':'GC_30' if 'GC_30' in tree_couplings else 'DCMPLX(0D0,0D0)',
                  'QUARTIC_COUPLING':'GC_HHHH' if 'GC_HHHH' in tree_couplings else 'DCMPLX(0D0,0D0)',
                  'COUPLINGS':','.join(couplings),'BRIDGE_CALL':'\n'.join('      '+s for s in bridge),
                  'INTEGRATED_CALL':'      '+integrated}
    text=(HERE/'raw_gg_hhh.f').read_text()
    for key,value in replacements.items():text=text.replace('@'+key+'@',value)
    assert '@' not in text
    harness=work/'raw_gg_hhh.f';harness.write_text(text)
    with (work/'build.log').open('w') as log:
        run(['make','-j2','libdhelas','libmodel'],export/'Source',log)
        objects=[]
        for name in ['loop_num','CT_interface']:
            obj=work/(name+'.o');objects.append(obj)
            run(['gfortran','-O2','-fPIC','-ffixed-line-length-none','-I'+str(source),'-c',source/(name+'.f'),'-o',obj],work,log)
        binary=work/'raw-gg-hhh'
        run(['gfortran','-O2','-ffixed-line-length-none','-I'+str(source),harness,*objects,
             '-L'+str(export/'lib'),'-ldhelas','-lmodel','-lcts','-o',binary],work,log)
    def parse(path,physical):
        rows=[];momenta={};wavefunctions={};inputs=None;epsilon=None;stable=None
        for line in path.read_text().splitlines():
            fields=line.split()
            if not fields:continue
            if fields[0]=='NUM':
                values=list(map(float,fields[2:]));assert len(values)==12 and all(map(math.isfinite,values))
                assert values[8:10]==values[10:12]
                rows.append({'id':int(fields[1]),'q':[values[i:i+2] for i in range(0,8,2)],'raw':values[8:10]})
            elif fields[0]=='P':momenta[fields[1]]=list(map(float,fields[2:]))
            elif fields[0]=='W':wavefunctions.setdefault(fields[1],[]).append(list(map(float,fields[3:])))
            elif fields[0]=='INPUT':inputs=list(map(float,fields[1:]))
            elif fields[0]=='LAURENT':
                stable=fields[1]=='T';values=list(map(float,fields[2:]));assert len(values)==6
                if all(map(math.isfinite,values)):epsilon=[values[i:i+2] for i in range(0,6,2)]
        assert len(rows)==70 and len({r['id'] for r in rows})==70
        assert len(momenta)==5 and len(wavefunctions)==nwave and len(inputs)==12
        assert all(map(math.isfinite,inputs)) and all(math.isfinite(x) for p in momenta.values() for x in p)
        assert all(math.isfinite(x) for w in wavefunctions.values() for c in w for x in c)
        assert all(c[1]==0 for w in wavefunctions.values() for c in w[:4])
        signed=[[-c[0] for c in wavefunctions[str(leg)][:4]] for leg in order]
        assert all(abs(sum(p[i] for p in signed))<1e-12 for i in range(4))
        shifts=[[0.0]*4]
        for p in signed[:-1]:shifts.append([a+b for a,b in zip(shifts[-1],p)])
        if physical:assert stable and epsilon is not None,'physical fixed-mode evaluation failed'
        return {'inputs_mt_mh_muR_wh_reimGC5_reimGC37_reimGC30_reimGCHHHH':inputs,'external_momenta':momenta,
                'external_polarizations':{str(i):wavefunctions[str(i)][4:8] for i in (1,2)},
                'helas_wavefunctions':wavefunctions,'cuttools_shifts':shifts,'numerator_samples':rows,
                'raw_laurent':epsilon,'stable':stable}
    cases=[]
    states=[('pp',1,1,0,0),('pm',1,-1,0,0),('mp',-1,1,0,0),('mm',-1,-1,0,0),
            ('generic_complex',1,1,0,1),('ward0',1,1,1,0),('ward1',1,1,2,0),('ward_both',1,1,3,0)]
    for mode in (1,4):
        for state,h1,h2,ward,generic in states:
            stdout=work/f'{state}-mode{mode}.txt'
            with stdout.open('w') as log:run([binary,export/'Cards/param_card.dat',h1,h2,ward,mode,generic],work,log)
            cases.append({'state':state,'helicity':[h1,h2],'ward_mask':ward,'ctmode':mode,
                          'integrated_diagnostic_only':state not in ('pp','pm','mp','mm'),**parse(stdout,state in ('pp','pm','mp','mm'))})
    result={'schema_version':1,'graph_filter_id':args.graph,'graph_metadata':graph,'loop_propagators':n,
            'wavefunction_order':order,'wavefunction_trees':{str(i):t for i,t in trees.items()},'wavefunction_leaves':{str(i):v for i,v in leaves.items()},'tree_couplings':tree_couplings,'fermion_seed':seeds.pop(),'couplings':couplings,'cases':cases,
            'fixed_ctmodes':[1,4],'loop_library':2,'commands':commands,
            'raw_grid':'70 affine complex unisolvent points for total loop degree <=4',
            'convention':'unsquared graph only; R1 retained; no R2/UV, color/helicity average, or final-state symmetry factor',
            'counterterm_exclusion':'only loop_num and CT_interface linked; selected loop amplitude '+str(amplitude)+'; automatic MP_BORN_AMPS_AND_WFS replaced by STOP91 trap; explicit modes1/4 and MP_DONE=true',
            'linear_color_factor':{'fermion':-1,'trace':'Tr(Ta Tb)=delta_ab/2','delta_ab_coefficient':'-1/2'},
            'candidate_raw_factor':'-1/2; must be independently confirmed with FeynKit routing/numerator',
            'candidate_laurent_factor':'-8*pi^2; CT_interface normalization1/(16*pi^2)',
            'input_precision':'external wavefunctions/couplings computed in binary64 then promoted unchanged for explicit mode4; not high-precision physical inputs',
            'source_sha256':{str(p):digest(p) for p in [Path(__file__),HERE/'raw_gg_hhh.f',harness,binary,source/'loop_num.f',source/'CT_interface.f',source/'loop_matrix.f',export/'Cards/param_card.dat',export/'Cards/MadLoopParams.dat',export/'lib/libdhelas.a',export/'lib/libmodel.a',export/'lib/libcts.a']}}
    (work/'reference.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(work/'reference.json')


if __name__=='__main__':main()
