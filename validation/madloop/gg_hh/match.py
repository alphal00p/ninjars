#!/usr/bin/env python3
"""Compile and record independent exact FeynKit/MadLoop graph matching."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'Cargo.toml').is_file())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--cargo-artifacts',type=Path,required=True)
 p.add_argument('--reference',type=Path,required=True)
 p.add_argument('--fixtures',type=Path,required=True)
 p.add_argument('--output-dir',type=Path,required=True)
 a=p.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
 names=['ninjars_graph','feynkit_graph','symbolica','serde_json'];libs={}
 for line in a.cargo_artifacts.read_text().splitlines():
  r=json.loads(line)
  if r.get('reason')=='compiler-artifact' and r['target']['name'] in names:
   for f in r['filenames']:
    if f.endswith('.rlib'):libs[r['target']['name']]=Path(f)
 assert set(libs)==set(names)
 binary=out/'match-gg-hh';cmd=['rustc','--edition=2024',str(HERE/'match.rs'),'-o',str(binary),'-L','dependency='+str(libs['symbolica'].parent)]
 for n,f in libs.items():cmd+=['--extern',n+'='+str(f)]
 env=dict(os.environ,CARGO_CRATE_NAME='gg_hh_match')
 with (out/'build.log').open('w') as log:subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 run=[str(binary),str(a.reference.resolve()),str(a.fixtures.resolve())]
 with (out/'stdout.json').open('w') as stdout,(out/'stderr.log').open('w') as stderr:
  completed=subprocess.run(run,cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
 result=json.loads((out/'stdout.json').read_text())
 result['exit_code']=completed.returncode
 result['criterion']='abs(actual-reference)/max(1,abs(reference)) <=1e-9 for every raw complex sample'
 result['normalization']={'raw_factor':'-1/2','linear_laurent_factor':'-8*pi^2',
    'proof':'One fermion-loop minus sign times Tr(Ta Tb)=delta_ab/2, projected by delta_ab/8 exactly once. Independent symbolic temporal-polarization zero-momentum controls verify each complete FeynKit numerator and the triangle Higgs bridge. MadLoop loop amplitude2 has fermion factor -1 and color tensor Tr(1,2); CT_interface contributes1/(16*pi^2). Couplings are GC11=GC5=iG, GC94=GC37=-i yt/sqrt2, GC69=GC30=-6i lambda vev.',
    'integrated_comparison':'not performed by this raw matching probe'}
 result['reference_scope']={'input_precision':'MadLoop binary64 wavefunctions/momenta/couplings converted to exact rationals before binding; not an independent high-precision physical input oracle',
    'polarizations':'Unisolvence is in loop components after each fixed polarization binding; generic and Ward states add independent controls.',
    'independent_checks':result['checks'],'internal_madloop_raw_wrapper_checks':json.loads(a.reference.read_text())['internal_raw_wrapper_checks']}
 sources=[HERE/'match.rs',Path(__file__),a.reference.resolve(),a.fixtures.resolve()/'feynkit-graphs.json',ROOT/'Cargo.lock',ROOT/'dependencies/manifest.json']
 sources+=sorted(a.fixtures.resolve().glob('top-*.dot'))
 result['provenance']={'recorded_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
   'compile_command':cmd,'run_command':run,'compiler':subprocess.check_output(['rustc','-Vv'],text=True),
   'binary_sha256':digest(binary),'rlib_sha256':{str(f):digest(f) for f in libs.values()},
   'source_sha256':{str(f):digest(f) for f in sources},
   'dependency_heads':{n:subprocess.check_output(['git','-C',str(ROOT/f),'rev-parse','HEAD'],text=True).strip() for n,f in [('symbolica','references/symbolica'),('feynkit','references/gammaloop_ninja_shared')]}}
 (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(out/'result.json')
 if completed.returncode or not result['pass']:raise SystemExit(1)
if __name__=='__main__':main()
