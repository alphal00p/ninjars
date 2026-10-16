#!/usr/bin/env python3
"""External own-process CPU attribution; never performance-acceptance evidence."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import datetime

HERE = Path(__file__).resolve().parent

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repository', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--perf', type=Path)
    parser.add_argument('--workload', choices=['n5', 'bubble2', 'bubble3'], default='n5')
    parser.add_argument('--diagnostic-build', type=Path, help='instrument.py build.json; execute spans without perf')
    parser.add_argument('--width', type=int, default=1)
    parser.add_argument('--rounds', type=int, default=30000)
    args = parser.parse_args()
    if not args.perf and not args.diagnostic_build:
        parser.error('--perf or --diagnostic-build is required')
    repo, out = args.repository.resolve(), args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    baseline = json.loads(gzip.decompress((repo/'validation/evidence/shared-workload-diagnostic-timing.json.gz').read_bytes()))
    old = repo/'target/performance/shared-workloads-diagnostic-newpins-release'
    library = old/'libninjars_shared_provider_probe.so'
    input_path = old/'input.txt'
    reference = repo/'target/reference-uncached-v2'
    ninja, olo = reference/'ninja-build/src/.libs/libninja.a', reference/'oneloop/libavh_olo.a'
    for path in [library, input_path, ninja, olo]:
        if baseline['artifacts_sha256'].get(str(path)) != digest(path):
            raise RuntimeError(f'immutable baseline artifact mismatch: {path}')
    diagnostic_build = None
    if args.diagnostic_build:
        diagnostic_build = json.loads(args.diagnostic_build.read_text())
        library = args.diagnostic_build.resolve().parent/'libninjars_shared_provider_probe.so'
        if not diagnostic_build.get('success') or digest(library) != diagnostic_build['library_sha256']:
            raise RuntimeError('diagnostic library build identity mismatch')
    for path in [library,input_path]: shutil.copyfile(path,out/path.name)
    library = out/library.name
    headers = sorted((repo/'validation/performance/shared_provider').glob('*.h*'))
    for path in headers:
        if baseline['source_sha256'].get(str(path.relative_to(repo))) != digest(path):
            raise RuntimeError(f'shared ABI/header differs from immutable library baseline: {path}')
    headers += sorted((reference/'ninja-source').rglob('*.h'))
    headers += sorted((reference/'ninja-source').rglob('*.hh'))
    headers += sorted((reference/'ninja-build').rglob('*.h'))
    headers += sorted((reference/'ninja-build').rglob('*.hh'))
    source = HERE/('probe.cc' if args.workload == 'n5' else 'bubble.cc')
    source_paths = [source, HERE/'control.hh', HERE/'diagnostic.hh', HERE/'run.py', *headers,
                    repo/'examples/graphs/physical/gg-hhh/top-pentagon-26.dot']
    source_before = {str(path):digest(path) for path in source_paths}
    (out/'source-before.json').write_text(json.dumps(source_before,indent=2)+'\n')
    environment = dict(os.environ, RAYON_NUM_THREADS='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    if diagnostic_build:
        environment['NINJARS_PROFILE_OUTPUT'] = str(out/'kernel-spans.json')
    commands=[]
    def run(label,command,stdin=None):
        command=list(map(str,command))
        with (out/f'{label}.stdout').open('w') as stdout,(out/f'{label}.stderr').open('w') as stderr:
            result=subprocess.run(command,cwd=repo,env=environment,input=stdin,text=True,stdout=stdout,stderr=stderr,timeout=180)
        commands.append({'label':label,'argv':command,'returncode':result.returncode})
        (out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        if result.returncode:
            (out/'failure.json').write_text(json.dumps({'success':False,'label':label,
                'commands':commands,'source_before':source_before,
                'source_after':{str(path):digest(path) for path in source_paths}},indent=2)+'\n')
            raise RuntimeError(f'{label} failed: see {out}/{label}.stderr')
        return (out/f'{label}.stdout').read_text()
    fortran=Path(subprocess.check_output(['gfortran','-print-file-name=libgfortran.so'],text=True).strip())
    binary=out/'probe'
    run('build',['g++','-O3','-g','-fno-omit-frame-pointer','-std=c++17','-Wall','-Wextra','-Werror',
        '-I'+str(repo/'validation/performance/shared_provider'),'-I'+str(reference/'ninja-source/src'),
        '-I'+str(reference/'ninja-build/src'),'-I'+str(reference/'ninja-build'),source,library,ninja,olo,fortran,
        '-lquadmath','-ldl','-Wl,-rpath,'+str(out),'-Wl,-rpath,'+str(fortran.parent),'-o',binary])
    dot=repo/'examples/graphs/physical/gg-hhh/top-pentagon-26.dot'
    input_text=input_path.read_text()
    selector = dot if args.workload == 'n5' else args.workload[-1]
    run('check',[binary,selector,args.width,1,'-','-'],input_text)
    if diagnostic_build:
        stdout=run('record',[binary,selector,args.width,args.rounds,'-','-'],input_text)
    else:
        control,ack=out/'control.fifo',out/'ack.fifo'
        os.mkfifo(control);os.mkfifo(ack)
        try:
            stdout=run('record',[args.perf,'record','-e','cpu-clock:u','-F','997','--call-graph','dwarf,16384','-D','-1',
               '--control',f'fifo:{control},{ack}','-o',out/'perf.data','--',binary,selector,args.width,args.rounds,control,ack],input_text)
        finally:
            control.unlink();ack.unlink()
    rows=[json.loads(line) for line in stdout.splitlines() if line.startswith('{')]
    if len(rows)!=1 or not rows[0]['success']:raise RuntimeError('missing profile correctness result')
    if not diagnostic_build:
        run('self-report',[args.perf,'report','-i',out/'perf.data','--stdio','--no-children','--percent-limit','0.25','--sort','dso,symbol','--call-graph','none','--field-separator',';'])
        run('callgraph-report',[args.perf,'report','-i',out/'perf.data','--stdio','--children','--percent-limit','0.5','--sort','dso,symbol','--call-graph','none','--field-separator',';'])
        run('stacks',[args.perf,'script','-i',out/'perf.data','--ns','-F','comm,pid,tid,time,event,ip,sym,dso'])
    source_after = {str(path):digest(path) for path in source_paths}
    if source_before != source_after:
        (out/'failure.json').write_text(json.dumps({'success':False,'source_before':source_before,
            'source_after':source_after,'commands':commands},indent=2)+'\n')
        raise RuntimeError('driver/source headers changed during profile')
    record=rows[0]
    record.update({'baseline':'Immutable library from the retained shared-provider diagnostic; actual source hashes below, not rebuilt at current HEAD',
       'baseline_dependency_heads':baseline['dependency_heads'],'baseline_source_sha256':baseline['source_sha256'],
       'profile_source_sha256':source_before,'sources_unchanged':True,
       'artifact_sha256':{str(path):digest(path) for path in [binary,library,input_path,ninja,olo] + ([args.perf,out/'perf.data'] if not diagnostic_build else [out/'kernel-spans.json'])},
       'diagnostic_build':diagnostic_build,
       'kernel_execution':'Symbolica interpreter; native-kernels feature not enabled/selected',
       'current_git_head_for_driver_context':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
       'commands':commands,'sampling_hz_requested':None if diagnostic_build else 997,'callgraph':None if diagnostic_build else 'dwarf,16384','event':None if diagnostic_build else 'cpu-clock:u',
       'recorded_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'sampling_boundary':'Counters/perf disabled during setup, capture, warmup, preparation and correctness; enabled only around repeated Rust reduction and checksum',
       'host_isolation_verified':False,'performance_acceptance':False,
       'limitations':['CPU samples are statistical attribution, not exact per-kernel call counts or standalone kernel timings','Same fixed physical point only; changing-input and bubble profiles remain separate','Common callback/FFI and frozen-master lookup overhead remain included','No optimized-gain or acceptance claim from an instrumented own-process profile']})
    (out/'result.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(out/'result.json')

if __name__=='__main__':main()
