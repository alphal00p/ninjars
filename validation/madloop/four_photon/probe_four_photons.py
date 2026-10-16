#!/usr/bin/env python3
"""Compile stock unsquared box routines; record fixed-mode probes and provenance."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p/'references/MG5_aMC_v3_7_0').is_dir())
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--graph', type=int, required=True)
parser.add_argument('--output-dir', type=Path, default=ROOT/'target/reference/madloop-four-photon')
parser.add_argument('--exports-dir', type=Path, help='Read existing isolated graph exports without modifying their recorded probe outputs.')
args = parser.parse_args()
OUT = args.output_dir.resolve()
EXPORTS = (args.exports_dir or OUT).resolve()
OUT.mkdir(parents=True, exist_ok=True)
metadata = json.loads((EXPORTS/'graphs.json').read_text())
graph = next(g for g in metadata['graphs'] if g['filter_id'] == args.graph)
export = EXPORTS/graph['directory']
source = export/'SubProcesses/P0_aa_aa'
work = OUT/f'probe-{args.graph}'
work.mkdir(exist_ok=True)
commands = []
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def run(command, cwd, stdout=None):
    commands.append({'argv': [str(x) for x in command], 'cwd': str(cwd)})
    subprocess.run([str(x) for x in command], cwd=cwd, stdout=stdout, stderr=subprocess.STDOUT, check=True)

assert graph['multiplier'] == 1
calls = [call for call in graph['loop_calls'] if not call.startswith('#')]
assert len(calls) == 1
match = re.match(r'CALL %\(proc_prefix\)sLOOP_4_4\(1,([1-4]),([1-4]),([1-4]),([1-4]),', calls[0])
assert match
order = [int(x) for x in match.groups()]
assert sorted(order) == [1, 2, 3, 4]
expected = ('CALL %(proc_prefix)sLOOP_4_4(1,'+','.join(map(str, order))+','
            + 'DCMPLX(mdl_MT),CMPLX(MP__mdl_MT,KIND=16),'*4
            + 'GC_2,MP__GC_2,'*4 + '4,1,1,1,AMPL(1,1),S(1))')
assert calls == [expected], calls
color = [entry for row in graph['color_amplitudes'] for entry in row]
assert color == [{'fermion_factor': -1, 'rational_color_coefficient': '1', 'imaginary': False,
                  'nc_power': 1, 'amplitude_number': 1}], color
assert "(('ColorOne', ()),)" in graph['loop_color_basis']
matrix = (source/'loop_matrix.f').read_text()
num = (source/'loop_num.f').read_text()
ct = (source/'CT_interface.f').read_text()
declaration = re.search(r'SUBROUTINE ML5_0_LOOP_4_4\((.*?)\)',ct,re.S)
assert declaration
declaration = re.sub(r'[\s$]','',declaration.group(1))
assert declaration == ('LID,W1,W2,W3,W4,'
                       + ','.join(f'M{i},MP_M{i}' for i in range(1,5))+','
                       + ','.join(f'C{i},MP_C{i}' for i in range(1,5))
                       + ',RANK,LSYMFACT,LMULTIPLIER,AMPLN,RES,STABLE')
assert 'SUBROUTINE ML5_0_LOOPNUM(Q,RES)' in num
assert 'SUBROUTINE ML5_0_LOOPNUMHEL(Q,RES,H)' in num
assert 'PARAMETER (NLOOPAMPS=1, NCTAMPS=0)' in matrix
for key, value in [('NEXTERNAL',4),('NLOOPAMPS',1),('NWAVEFUNCS',4),('MAXLCOUPLINGS',4),('NCOMB',16)]:
    assert f'PARAMETER ({key}={value})' in num
assert 'PARAMETER (NORMALIZATION = 1.D0/(16.D0*3.14159265358979323846D0*' in ct
assert 'CALL CTSXCUT(CTMODE,LSCALE,MU_R,NLOOPLINE,ML5_0_LOOPNUM' in ct
function = re.search(r'^      LOGICAL FUNCTION ML5_0_IS_HEL_SELECTED\(HELID\).*?^      END\s*$', matrix, re.M|re.S)
assert function
helper = work/'helicity_selection.f'
helper.write_text(function.group(0)+'\n')
harness = work/'raw_four_photon.f'
harness.write_text((HERE/'raw_four_photon.f').read_text().replace('@WAVEFUNCTION_ORDER@', ','.join(map(str,order))))
with (work/'build.log').open('w') as log:
    run(['make','-j2','libdhelas','libmodel','libcuttools'], export/'Source', log)
    objects=[]
    for name in ['loop_num','CT_interface','mp_born_amps_and_wfs']:
        obj = work/f'{name}.o'
        run(['gfortran','-O2','-fPIC','-ffixed-line-length-none','-I'+str(source),'-c',source/f'{name}.f','-o',obj],work,log)
        objects.append(obj)
    binary = work/'raw-four-photon'
    run(['gfortran','-O2','-ffixed-line-length-none','-I'+str(source),harness,helper,*objects,
         '-L'+str(export/'lib'),'-ldhelas','-lmodel','-lcts','-o',binary],work,log)

def parse(path, require_stable=True):
    rows=[]; momenta={}; polarization={}; wavefunctions={}; epsilon=None; inputs=None
    for line in path.read_text().splitlines():
        fields=line.split()
        if not fields: continue
        if fields[0]=='NUM':
            values=list(map(float,fields[2:])); assert len(values)==12
            assert values[8:10]==values[10:12], 'raw and processed numerator differ'
            assert all(math.isfinite(v) for v in values)
            rows.append({'id':int(fields[1]),'q':[values[i:i+2] for i in range(0,8,2)],'raw':values[8:10]})
        elif fields[0]=='P': momenta[fields[1]]=list(map(float,fields[2:]))
        elif fields[0]=='W':
            wavefunctions.setdefault(fields[1],[]).append(list(map(float,fields[3:])))
            if 5<=int(fields[2])<=8:
                polarization.setdefault(fields[1],[]).append(list(map(float,fields[3:])))
        elif fields[0]=='INPUT': inputs=list(map(float,fields[1:]))
        elif fields[0]=='LAURENT':
            stable=fields[1]=='T'
            if require_stable: assert stable, 'fixed-mode graph evaluation unstable'
            values=list(map(float,fields[2:])); assert len(values)==6
            finite=all(math.isfinite(v) for v in values)
            if require_stable: assert finite, 'non-finite physical Laurent coefficient'
            epsilon=[values[i:i+2] for i in range(0,6,2)] if finite else None
    assert len(rows)==70 and len({row['id'] for row in rows})==70
    assert len(momenta)==4 and len(polarization)==4 and len(inputs)==4
    assert all(math.isfinite(v) for p in momenta.values() for v in p)
    assert all(math.isfinite(v) for p in polarization.values() for c in p for v in c)
    assert all(math.isfinite(v) for v in inputs)
    # This follows CT_interface's PL=-Re(W(1:4,WE)) and cumulative PCT construction.
    signed=[[-v[0] for v in wavefunctions[str(leg)][:4]] for leg in order]
    assert all(v[1]==0 for row in wavefunctions.values() for v in row[:4])
    assert [sum(p[i] for p in signed) for i in range(4)]==[0.0]*4
    pct=[[0.0]*4]
    for p in signed[:-1]: pct.append([a+b for a,b in zip(pct[-1],p)])
    return {'inputs_mt_muR_reGC2_imGC2':inputs,'external_momenta':momenta,
            'external_polarizations':polarization,'helas_wavefunctions':wavefunctions,
            'numerator_samples':rows,'raw_laurent':epsilon,
            'cuttools_shifts':pct, 'stable':stable}

cases=[]
for mode in (1,4):
    for state,helicity,ward in [('pppp',[1,1,1,1],0),('ppmm',[1,1,-1,-1],0),('mpmp',[-1,1,-1,1],0),('generic_complex',[1,1,1,1],0)]:
        path=work/f'{state}-mode{mode}.txt'
        with path.open('w') as stream:
            run([binary,export/'Cards/param_card.dat',*map(str,helicity),str(ward),str(mode),str(int(state=='generic_complex'))],work,stream)
        cases.append({'state':state,'helicity':helicity,'ward_mask':ward,'ctmode':mode,
                      'integrated_diagnostic_only':state=='generic_complex',
                      'stdout':str(path.relative_to(OUT)),**parse(path, require_stable=state!='generic_complex')})
sources={str(p.resolve()):digest(p) for p in
         [Path(__file__),HERE/'raw_four_photon.f',harness,helper,binary,
          source/'loop_num.f',source/'CT_interface.f',source/'mp_born_amps_and_wfs.f',
          source/'loop_matrix.f',export/'Cards/param_card.dat',export/'Cards/MadLoopParams.dat',
          export/'lib/libdhelas.a',export/'lib/libmodel.a',export/'lib/libcts.a']}
result={'schema_version':1,'graph_filter_id':args.graph,'graph_metadata':graph,
        'convention':'4D numerator plus R1; no R2/UV, squaring, interference, helicity average, or phase-space symmetry factor',
        'fixed_ctmodes':[1,4],'loop_library':2,
        'quadruple_inputs':'same binary64 wavefunctions/couplings promoted exactly, not independent high-precision physical inputs',
        'wavefunction_order':order,
        'audited_loop_routine_declaration':declaration,
        'linear_factor':{'fermion':-1,'rational':'1','nc_power':1,'Nc':3,'value':-3},
        'ct_interface_normalization':'1/(16*pi^2)',
        'candidate_prepared_laurent_factor':'-48*pi^2; pending independent FeynKit numerator/routing match',
        'raw_grid':'70 unisolvent affine complex samples, total degree <=4 in q0,q1,q2,q3',
        'source_sha256':sources,'commands':commands,'cases':cases,
        'exports_directory':str(EXPORTS),
        'generic_polarization_contract':'W(5:8,leg) assigned directly from fixture exact small-integer complex vectors; integrated output is diagnostic only'}
(work/'reference.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(work/'reference.json')
