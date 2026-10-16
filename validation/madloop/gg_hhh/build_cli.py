#!/usr/bin/env python3
"""Build/copy an immutable development CLI for correctness, never timing acceptance."""
import argparse
from datetime import datetime,timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('build_record',ROOT/'validation/performance/build.py')
build=importlib.util.module_from_spec(spec);spec.loader.exec_module(build)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    before=build.production_snapshot();dependencies=build.checkpoint.dependencies()
    config=build.cargo_configuration()
    command=['cargo','build','-p','ninjars-cli','--all-features','--locked','-j8','--message-format=json']
    with (out/'cargo-artifacts.jsonl').open('w') as log,(out/'build.log').open('w') as err:
        subprocess.run(command,cwd=ROOT,stdout=log,stderr=err,check=True)
    source=None
    for line in (out/'cargo-artifacts.jsonl').read_text().splitlines():
        r=json.loads(line)
        if r.get('reason')=='compiler-artifact' and r['target']['name']=='ninjars' and r.get('executable'):
            source=Path(r['executable'])
    assert source is not None
    after=build.production_snapshot()
    assert before['source_sha256']==after['source_sha256'] and before['git_head']==after['git_head']
    assert dependencies==build.checkpoint.dependencies() and config==build.cargo_configuration()
    binary=out/'ninjars';shutil.copy2(source,binary)
    digest=build.checkpoint.digest
    record={'schema_version':1,'recorded_utc':datetime.now(timezone.utc).isoformat(),
            'command':command,'compiler':build.checkpoint.command('rustc','-Vv'),
            'binary_sha256':digest(binary),'profile':'dev; correctness validation only',
            'performance_acceptance':False,'source_commit':after['git_head'],
            'source_sha256':after['source_sha256'],'checkout':after,'dependencies':dependencies,
            'dependency_manifest':json.loads((ROOT/'dependencies/manifest.json').read_text()),
            'cargo_configuration':config,'environment':{k:os.environ.get(k) for k in build.ENVIRONMENT},
            'build_log_sha256':digest(out/'build.log'),'cargo_artifacts_sha256':digest(out/'cargo-artifacts.jsonl'),
            'recorder_sha256':digest(Path(__file__))}
    (out/'build.json').write_text(json.dumps(record,indent=2)+'\n');print(out/'build.json')


if __name__=='__main__':main()
