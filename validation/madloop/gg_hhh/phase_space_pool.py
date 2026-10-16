#!/usr/bin/env python3
"""Compile the exact Rust pool generator using an existing locked Cargo build."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cargo-artifacts',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    names={'ninjars_graph','symbolica','serde_json'};libs={}
    for line in a.cargo_artifacts.read_text().splitlines():
        r=json.loads(line)
        if r.get('reason')=='compiler-artifact' and r['target']['name'] in names:
            for f in r['filenames']:
                if f.endswith('.rlib'):libs[r['target']['name']]=Path(f)
    assert set(libs)==names
    hashes={str(f):digest(f) for f in libs.values()}
    binary=out/'phase-space-pool'
    command=['rustc','--edition=2024',str(HERE/'phase_space_pool.rs'),'-o',str(binary),'-L','dependency='+str(libs['symbolica'].parent)]
    for name,path in libs.items():command+=['--extern',name+'='+str(path)]
    env=dict(os.environ,CARGO_CRATE_NAME='gg_hhh_phase_space_pool')
    with (out/'build.log').open('w') as log:subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    with (out/'pool.json').open('w') as stdout:subprocess.run([binary],cwd=ROOT,env=env,stdout=stdout,check=True)
    assert hashes=={str(f):digest(f) for f in libs.values()}
    result={'schema_version':1,'compile_command':command,'run_command':[str(binary)],'compiler':subprocess.check_output(['rustc','-Vv'],text=True),
            'binary_sha256':digest(binary),'pool_sha256':digest(out/'pool.json'),'rlib_sha256':hashes,
            'source_sha256':{str(f):digest(f) for f in [Path(__file__),HERE/'phase_space_pool.rs',ROOT/'Cargo.lock',ROOT/'dependencies/manifest.json']}}
    (out/'provenance.json').write_text(json.dumps(result,indent=2)+'\n');print(out/'pool.json')


if __name__=='__main__':main()
