#!/usr/bin/env python3
"""Compile/run independent FeynKit fixture generation from locked Cargo artifacts.

The Rust source uses existing model, generator, color projector, graph routing,
Idenso contraction and numerator-preparation APIs. Python only orchestrates
validation files and writes exact string-valued numerical fixture inputs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'Cargo.toml').is_file())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(name):
    momenta = {'0':['150','0','0','150'], '1':['150','0','0','-150'],
               '2':['150','15*sqrt(11)','0','20*sqrt(11)'],
               '3':['150','-15*sqrt(11)','0','-20*sqrt(11)']}
    plus = [['0','-1/sqrt(2)','-1𝑖/sqrt(2)','0'],
            ['0','-1/sqrt(2)','1𝑖/sqrt(2)','0']]
    minus = [['0','1/sqrt(2)','-1𝑖/sqrt(2)','0'],
             ['0','1/sqrt(2)','1𝑖/sqrt(2)','0']]
    states = [(plus[0],plus[1]), (plus[0],minus[1]),
              (minus[0],plus[1]), (minus[0],minus[1]),
              (['1+1𝑖','2','-1+2𝑖','3'], ['2','1-1𝑖','3','4+1𝑖']),
              (momenta['0'],plus[1]), (plus[0],momenta['1']),
              (momenta['0'],momenta['1'])]
    return {'schema_version':1, 'graph':name+'.dot', 'model':'sm', 'scale_squared':'10000',
            'color_dimension_invariants':True,
            'parameter_overrides':{'MT':'173','ymt':'173','MH':'125','WT':'0','WH':'0',
                                   'Gf':'116639/10^10','aS':'59/500','aEWM1':'132507/1000','MZ':'22797/250'},
            'samples':[{'external_momenta':momenta,
                        'external_polarizations':{'0':a,'1':b}} for a,b in states]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cargo-artifacts',type=Path,required=True,
                        help='JSON-lines output of cargo build -p ninjars-graph --locked --message-format=json')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--rustc',default='rustc')
    args=parser.parse_args()
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    names=['ninjars_graph','feynkit_graph','feynkit_amplitude','feynkit_generator','symbolica','serde_json']
    # Host units (build scripts, proc macros) share deps/ with the target units
    # and are listed in no fixed order; keep the rlibs with ninjars_graph's profile.
    candidates={}
    for line in args.cargo_artifacts.read_text().splitlines():
        record=json.loads(line)
        if record.get('reason')!='compiler-artifact': continue
        name=record['target']['name']
        for filename in record['filenames']:
            if name in names and filename.endswith('.rlib'):
                candidates.setdefault(name,[]).append((json.dumps(record['profile'],sort_keys=True),Path(filename)))
    if set(candidates)!=set(names): parser.error(f'missing required artifacts: {set(names)-set(candidates)}')
    profiles={profile for profile,_ in candidates['ninjars_graph']}
    if len(profiles)!=1: parser.error('ninjars_graph must have exactly one profile')
    libraries={}
    for name,entries in candidates.items():
        paths={path for profile,path in entries if profile in profiles}
        if len(paths)!=1: parser.error(f'{name}: expected one rlib with the ninjars_graph profile, found {len(paths)}')
        libraries[name]=paths.pop()
    directories={p.parent for name,p in libraries.items() if name!='ninjars_graph'}
    if len(directories)!=1: parser.error('dependency rlibs must share a directory')
    binary=out/'prepare-feynkit'
    command=[args.rustc,'--edition=2024',str(HERE/'prepare_feynkit.rs'),'-o',str(binary),
             '-L',f'dependency={next(iter(directories))}']
    for name,path in libraries.items(): command+=['--extern',f'{name}={path}']
    env=dict(os.environ,CARGO_CRATE_NAME='gg_hh_fixture_generator')
    with (out/'build.log').open('w') as log:
        subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    run=[str(binary),str(out)]
    with (out/'run.log').open('w') as log:
        subprocess.run(run,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    metadata=json.loads((out/'feynkit-graphs.json').read_text())
    for graph in metadata['graphs']:
        path=out/(graph['name']+'.json')
        path.write_text(json.dumps(fixture(graph['name']),indent=2)+'\n')
    metadata['states']=['pp','pm','mp','mm','generic_complex','ward0','ward1','ward_both']
    metadata['phase_space']='sqrt(s)=300, MH=125, MT=173; final p=(150, +/-15sqrt(11), 0, +/-20sqrt(11))'
    metadata['input_precision']='exact symbolic strings; MadLoop binary64 input conversion must be recorded separately'
    metadata['provenance']={'compile_command':command,'run_command':run,
        'compiler':subprocess.check_output([args.rustc,'-Vv'],text=True),
        'binary_sha256':digest(binary),'cargo_artifacts_sha256':digest(args.cargo_artifacts),
        'rlib_sha256':{str(path):digest(path) for path in libraries.values()},
        'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in [HERE/'prepare_feynkit.rs',Path(__file__),ROOT/'Cargo.lock',ROOT/'dependencies/manifest.json']},
        'fixture_sha256':{p.name:digest(p) for p in sorted(out.glob('top-*')) if p.suffix in ('.dot','.json')}}
    (out/'feynkit-graphs.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(out/'feynkit-graphs.json')


if __name__=='__main__':main()
