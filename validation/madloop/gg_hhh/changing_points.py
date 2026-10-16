#!/usr/bin/env python3
"""Fixed-mode MadLoop graph55 checks at explicit changing binary64 phase space.

The original frozen ggHHH driver is copied into a new artifact directory and
only its fixed momentum assignments are replaced by a five-row input read.
No reference export, original driver or physical fixture is modified.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[3]


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_point(point):
    assert point['helicities']==[1,1,0,0,0]
    assert point['model']==[173,173,125,116639/10**10,59/500,132507/1000,22797/250]
    assert point['scale_squared']==10000
    momenta=point['momenta']
    assert len(momenta)==5 and all(len(p)==4 for p in momenta)
    assert all(math.isfinite(x) for p in momenta for x in p)
    scale=max(1,max(abs(x) for p in momenta for x in p))
    assert all(abs(momenta[0][i]+momenta[1][i]-sum(p[i] for p in momenta[2:]))<=1e-12*scale for i in range(4))
    for index,p in enumerate(momenta):
        mass2=0 if index<2 else 125**2
        residual=p[0]**2-sum(x*x for x in p[1:])-mass2
        assert abs(residual)<=1e-12*max(1,p[0]**2,mass2)
        assert p[0]>0


def parse_output(path,point,mode,order):
    samples=[];momenta={};waves={};inputs=None;laurent=None;stable=None;reported_mode=None;state=None
    for line in path.read_text().splitlines():
        fields=line.split()
        if not fields:continue
        if fields[0]=='NUM':
            values=list(map(float,fields[2:]));assert len(values)==12 and all(map(math.isfinite,values))
            assert values[8:10]==values[10:12]
            samples.append({'id':int(fields[1]),'q':[values[i:i+2] for i in range(0,8,2)],'raw':values[8:10]})
        elif fields[0]=='STATE':state=list(map(int,fields[1:]))
        elif fields[0]=='CTMODE':reported_mode=int(fields[1])
        elif fields[0]=='P':momenta[int(fields[1])]=list(map(float,fields[2:]))
        elif fields[0]=='W':waves.setdefault(int(fields[1]),[]).append(list(map(float,fields[3:])))
        elif fields[0]=='INPUT':inputs=list(map(float,fields[1:]))
        elif fields[0]=='LAURENT':
            stable=fields[1]=='T';values=list(map(float,fields[2:]));assert len(values)==6 and all(map(math.isfinite,values))
            laurent=[values[i:i+2] for i in range(0,6,2)]
    assert len(samples)==70 and {r['id'] for r in samples}==set(range(1,71))
    assert stable and laurent is not None
    assert len(inputs)==12 and all(map(math.isfinite,inputs))
    assert set(momenta)==set(range(1,6)) and set(waves)==set(range(1,6))
    assert [momenta[i] for i in range(1,6)]==point['momenta'],'reference momentum read changed a binary64 input'
    assert all(len(w)==8 and all(len(z)==2 and all(map(math.isfinite,z)) for z in w) for w in waves.values())
    assert inputs[:4]==[173,125,100,0]
    assert state==[1,1,0,0] and reported_mode==mode
    shifts=[[0.]*4]
    for leg in order[:-1]:
        shifts.append([x-waves[leg][i][0] for i,x in enumerate(shifts[-1])])
    return {'external_momenta':{str(i):momenta[i] for i in momenta},
            'external_polarizations':{str(i):waves[i][4:8] for i in (1,2)},
            'helas_wavefunctions':{str(i):waves[i] for i in waves},
            'inputs_mt_mh_muR_wh_reimGC5_reimGC37_reimGC30_reimGCHHHH':inputs,
            'numerator_samples':samples,'cuttools_shifts':shifts,'raw_laurent':laurent,'stable':stable}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pool',type=Path,required=True,help='JSON {points:[{id,momenta,helicities,model,scale_squared}]}')
    p.add_argument('--baseline-probe-dir',type=Path,default=ROOT/'target/reference/madloop-gg-hhh-probes/probe-55')
    p.add_argument('--export-dir',type=Path,default=ROOT/'target/reference/madloop-gg-hhh-external/graph-55')
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    pool=json.loads(a.pool.read_text());points=pool['points'];assert len(points)>=3
    assert len({p['id'] for p in points})==len(points)
    for point in points:validate_point(point)
    assert len({json.dumps(p['momenta']) for p in points})==len(points),'changing workload requires distinct phase-space points'
    base=a.baseline_probe_dir.resolve();reference=json.loads((base/'reference.json').read_text())
    assert reference['graph_filter_id']==55 and reference['loop_propagators']==5
    for name,expected in reference['source_sha256'].items():assert digest(Path(name))==expected,('changed baseline source',name)
    export=a.export_dir.resolve();source=export/'SubProcesses/P0_gg_hhh'
    original=(base/'raw_gg_hhh.f').read_text()
    begin=original.index('      P=0D0\n');end=original.index('      W=DCMPLX',begin)
    reader='''      CALL GET_COMMAND_ARGUMENT(7,ARG)
      OPEN(UNIT=71,FILE=TRIM(ARG),STATUS='OLD',ACTION='READ')
      DO J=1,5
        READ(71,*) P(:,J)
      ENDDO
      CLOSE(71)
'''
    harness=out/'raw_changing_points.f';harness.write_text(original[:begin]+reader+original[end:])
    commands=[]
    def run(command,stdout):
        command=list(map(str,command));commands.append(command)
        subprocess.run(command,cwd=out,stdout=stdout,stderr=subprocess.STDOUT,check=True)
    with (out/'build.log').open('w') as log:
        objects=[]
        for name in ('loop_num','CT_interface'):
            obj=out/(name+'.o');objects.append(obj)
            run(['gfortran','-O2','-fPIC','-ffixed-line-length-none','-I'+str(source),'-c',source/(name+'.f'),'-o',obj],log)
        binary=out/'raw-changing-points'
        run(['gfortran','-O2','-ffixed-line-length-none','-I'+str(source),harness,*objects,'-L'+str(export/'lib'),'-ldhelas','-lmodel','-lcts','-o',binary],log)
    cases=[]
    for index,point in enumerate(points):
        pointfile=out/f'point-{index}.txt'
        pointfile.write_text('\n'.join(' '.join(format(x,'.17e') for x in row) for row in point['momenta'])+'\n')
        for mode in (1,4):
            stdout=out/f'point-{index}-mode{mode}.txt'
            with stdout.open('w') as log:run([binary,export/'Cards/param_card.dat',1,1,0,mode,0,pointfile],log)
            cases.append({'point_id':point['id'],'state':'pp','ctmode':mode,**parse_output(stdout,point,mode,reference['wavefunction_order'])})
    routing_path=ROOT/'validation/evidence/gg-hhh-raw.json'
    routing=next(g for g in json.loads(routing_path.read_text())['graphs'] if g['feynkit_orientation']==26)
    assert routing['madloop_filter_id']==55
    sources=[routing_path,Path(__file__),a.pool.resolve(),base/'reference.json',base/'raw_gg_hhh.f',harness,binary,
             source/'loop_num.f',source/'CT_interface.f',source/'coupl.inc',export/'Cards/param_card.dat',export/'Cards/MadLoopParams.dat',
             export/'lib/libdhelas.a',export/'lib/libmodel.a',export/'lib/libcts.a']
    result={'schema_version':1,'feynkit_orientation':26,'graph_filter_id':55,'points':points,'cases':cases,
            'wavefunction_order':reference['wavefunction_order'],'verified_routing':routing,
            'raw_factor':'-1/2','linear_laurent_factor':'-8*pi^2',
            'internal_madloop_raw_wrapper_checks':len(points)*2*70,
            'independent_raw_comparison_status':'pending separate shared FeynKit provider comparison',
            'input_precision':'Explicit shared binary64 momentum rows are read unchanged; mode4 promotes the same external wavefunctions/couplings. No high-precision-input oracle.',
            'convention':'unsquared top pentagon, four-dimensional numerator plus R1, no R2/UV; fixed pp; explicit modes1/4, no switching',
            'baseline_reference_sha256':digest(base/'reference.json'),
            'commands':commands,'source_sha256':{str(f):digest(f) for f in sources},'performance_acceptance':False}
    (out/'reference.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(out/'reference.json')


if __name__=='__main__':main()
