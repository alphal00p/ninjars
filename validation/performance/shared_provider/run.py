#!/usr/bin/env python3
"""Build and run validation-only shared numerator/master correctness probes."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--reference-build-dir', type=Path, default=ROOT/'target/reference-uncached-v2')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if list(output.iterdir()):
        parser.error('output directory must be empty')
    reference = args.reference_build_dir.resolve()
    spec=importlib.util.spec_from_file_location('reference_build',ROOT/'validation/reference/build.py')
    reference_build=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference_build)
    metadata=json.loads((reference/'revisions.json').read_text())
    ninja=reference/'ninja-build/src/.libs/libninja.a'
    olo=reference/'oneloop/libavh_olo.a'
    config=reference_build.configuration(reference)
    if (metadata.get('configuration')!=config
        or metadata.get('libraries_sha256')!={p.name:digest(p) for p in [ninja,olo]}
        or metadata.get('input_files_sha256')!=reference_build.library_inputs(reference)
        or not config['oneloop'] or not config['higher_rank'] or config['master_result_cache']):
        raise RuntimeError('reference library provenance/configuration missing or changed; rebuild uncached libraries')
    commands=[]
    def run(label,command,parse=False,accepted_codes=(0,)):
        command=list(map(str,command))
        result=subprocess.run(command,cwd=ROOT,text=True,capture_output=True)
        (output/f'{label}.stdout').write_text(result.stdout)
        (output/f'{label}.stderr').write_text(result.stderr)
        commands.append({'label':label,'argv':command,'returncode':result.returncode})
        (output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        if result.returncode not in accepted_codes:
            raise RuntimeError(f'{label} failed; see {output}/{label}.stderr')
        if parse:
            return [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
        return result.stdout
    source_paths=[p for p in HERE.rglob('*') if p.is_file() and 'target' not in p.parts and p.suffix in {'.rs','.h','.hh','.cc','.py','.toml','.lock','.md'}]
    initial={str(p.relative_to(ROOT)):digest(p) for p in source_paths}
    built=run('cargo-build',['cargo','build','--manifest-path',HERE/'Cargo.toml','--locked','--message-format=json'],True)
    artifacts=[Path(name) for row in built if row.get('reason')=='compiler-artifact' and row['target']['name']=='ninjars_shared_provider_probe' for name in row['filenames'] if name.endswith('.so')]
    if len(artifacts)!=1:
        raise RuntimeError('expected one provider shared library artifact')
    library=output/'libninjars_shared_provider_probe.so'
    shutil.copy2(artifacts[0],library)
    tests=run('cargo-tests-build',['cargo','test','--manifest-path',HERE/'Cargo.toml','--locked','--no-run','--message-format=json'],True)
    test_bins=[row['executable'] for row in tests if row.get('reason')=='compiler-artifact' and row.get('executable') and row['profile'].get('test')]
    if len(test_bins)!=1:
        raise RuntimeError('expected one Rust probe test executable')
    fortran=Path(run('fortran-library',['gfortran','-print-file-name=libgfortran.so']).strip())
    executables={}
    for name,source in [('numerator','probe.cc'),('master','master_probe.cc')]:
        executable=output/('shared-'+name+'-probe')
        run('cpp-'+name+'-build',['g++','-O2','-std=c++17','-Wall','-Wextra','-Werror',
            '-I'+str(reference/'ninja-source/src'),'-I'+str(reference/'ninja-build/src'),'-I'+str(reference/'ninja-build'),
            HERE/source,library,ninja,olo,fortran,'-lquadmath','-Wl,-rpath,'+str(output),'-Wl,-rpath,'+str(fortran.parent),'-o',executable])
        executables[name]=executable
    # All compilers have completed before either runtime check begins.
    run('rust-tests',[test_bins[0],'--test-threads=1'])
    run('known-c0-reproducer',[test_bins[0],'--ignored','--exact',
        'masters::tests::euclidean_c0_exceptional_point_must_be_finite','--test-threads=1'],
        accepted_codes=(0,101))
    c0_bug_reproduced=commands[-1]['returncode']==101
    results={}
    for name,executable in executables.items():
        rows=run('cpp-'+name+'-probe',[executable],True)
        if len(rows)!=1 or not rows[0].get('success'):
            raise RuntimeError('missing successful C++ '+name+' probe result')
        results[name]=rows[0]
    if initial!={str(p.relative_to(ROOT)):digest(p) for p in source_paths}:
        raise RuntimeError('probe source changed during validation')
    report={'success':True,'probes':results, 'scope':'shared Rust numerator and union-master replay correctness; not a matched-provider performance run',
        'production_core_ffi':False,'shared_master_provider_implemented':True,'performance_acceptance':False,
        'outstanding_c0_exception':{'reproduced':c0_bug_reproduced,'integral':'C0',
            'invariants':[-1,-2,-3],'masses_squared':[2,3,5],'scale_squared':7,
            'current_failure':'MasterFailure::NonFiniteResult','required_behavior':'finite result; positive-mass Euclidean point',
            'acceptance_blocker':True},
        'source_sha256':initial,'artifacts_sha256':{str(p):digest(p) for p in [library,*executables.values(),ninja,olo]},
        'reference_revision_record_sha256':digest(reference/'revisions.json'),
        'commands':commands,'rustc':run('rustc-version',['rustc','--version']).strip(),
        'cxx':run('cxx-version',['g++','--version']).splitlines()[0],
        'symbolica_revision':subprocess.check_output(['git','-C',ROOT/'references/symbolica','rev-parse','HEAD'],text=True).strip(),
        'oneloopmaster_revision':subprocess.check_output(['git','-C',ROOT/'references/oneloopmaster','rev-parse','HEAD'],text=True).strip()}
    (output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'success':True,'evidence':str(output/'result.json')}))

if __name__=='__main__':
    main()
