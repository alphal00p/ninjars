#!/usr/bin/env python3
"""Check or explicitly sample separate matched physical workload boundaries."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

from run_physical import ROOT, HERE, digest, runtime_sources
from timing_statistics import summarize_record


def host_snapshot():
    def read(path):
        path = Path(path)
        return path.read_text() if path.exists() else None
    cpus = sorted(os.sched_getaffinity(0))
    return {'allowed_cpu_affinity': cpus, 'loadavg': read('/proc/loadavg'),
            'proc_stat': read('/proc/stat'), 'cpuinfo': read('/proc/cpuinfo'),
            'smt_siblings': {str(cpu): read(f'/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list') for cpu in cpus},
            'cpu_governors': {str(cpu): read(f'/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor') for cpu in cpus},
            'process_cpu_snapshot': subprocess.check_output(['ps', '-eo', 'pid,comm,pcpu,psr'], text=True)}


def protocol():
    pool = json.loads((ROOT/'validation/evidence/gg-hhh-changing-pool.json').read_text())
    reference = json.loads((ROOT/'validation/evidence/gg-hhh-changing-madloop.json').read_text())
    if pool['points'] != reference['points'] or len(pool['points']) != 3 or reference['raw_factor'] != '-1/2' or reference['linear_laurent_factor'] != '-8*pi^2':
        raise RuntimeError('physical reference boundary changed')
    lines = ['3']
    for point in pool['points']:
        case, = [case for case in reference['cases'] if case['ctmode'] == 4 and case['point_id'] == point['id']]
        if not case['stable'] or len(case['numerator_samples']) != 70:
            raise RuntimeError('incomplete/stable independent reference required')
        lines += [' '.join(map(repr, row)) for row in point['momenta']]
        lines += [' '.join(map(str, point['helicities'])), ' '.join(map(repr, point['model'])), repr(point['scale_squared']), '70']
        for sample in case['numerator_samples']:
            lines.append(' '.join(map(repr, [-x for pair in sample['q'] for x in pair]+[-.5*x for x in sample['raw']])))
        lines.append(' '.join(repr(-8*math.pi**2*x) for pair in case['raw_laurent'] for x in pair))
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--timing', action='store_true', help='explicit diagnostic timings, never automatic acceptance')
    mode.add_argument('--allocation-diagnostics', action='store_true', help='separate instrumented Rust allocation run; no timings')
    parser.add_argument('--release', action='store_true', help='optimized correctness/allocation build; timing always uses release')
    parser.add_argument('--cpu', type=int, help='pin runtime to this already-allowed CPU; does not reserve its sibling')
    parser.add_argument('--widths', default='1,8,32,128,1024,31')
    parser.add_argument('--scopes', default='0,1,2,3', help='0 preparation,1 shared reduction,2 shared-live complete,3 production complete')
    parser.add_argument('--rounds', type=int, default=16)
    parser.add_argument('--trials', type=int, default=8)
    parser.add_argument('--reference-build-dir', type=Path, default=ROOT/'target/reference-uncached-v2')
    parser.add_argument('--physical-controls', type=Path, default=ROOT/'validation/evidence/shared-physical-provider-probe.json', help='separate raw/oracle/empty-batch report; its exact source revisions remain explicit')
    args = parser.parse_args()
    if args.cpu is not None and args.cpu not in os.sched_getaffinity(0):
        parser.error('chosen CPU is outside permitted affinity')
    if args.timing and args.cpu is None:
        parser.error('explicit timing requires a selected CPU; host/sibling isolation is still reviewed separately')
    if args.rounds < 1 or args.trials < 4 or args.trials % 2:
        parser.error('positive rounds and >=4 balanced trials required')
    for env_name in ['RUSTFLAGS', 'CFLAGS', 'CXXFLAGS']:
        if any(flag in os.environ.get(env_name, '') for flag in ['fast-math', 'unsafe-fp-math', 'finite-math-only']):
            parser.error(f'unsupported relaxed floating-point option in {env_name}')
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        parser.error('output directory must be empty')
    # Reuse the existing reference builder's exact library/config provenance.
    import importlib.util
    spec = importlib.util.spec_from_file_location('reference_build', ROOT/'validation/reference/build.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    reference = args.reference_build_dir.resolve()
    record = json.loads((reference/'revisions.json').read_text())
    config = builder.configuration(reference)
    ninja, olo = reference/'ninja-build/src/.libs/libninja.a', reference/'oneloop/libavh_olo.a'
    if (record.get('configuration') != config or not config['oneloop'] or not config['higher_rank'] or config['master_result_cache']
            or record.get('libraries_sha256') != {p.name: digest(p) for p in [ninja, olo]}
            or record.get('input_files_sha256') != builder.library_inputs(reference)):
        raise RuntimeError('uncached reference backend provenance mismatch')
    paths = runtime_sources()
    before = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    controls_path = args.physical_controls.resolve()
    controls = json.loads(controls_path.read_text())
    if not controls.get('success') or not controls.get('unchanged_runtime_sources') or 0 not in controls.get('integrated_widths', []) or 0 not in controls.get('preparation_widths', []) or controls.get('raw_components_checked') != 210:
        raise RuntimeError('complete separate physical controls are required')
    input_text = protocol()
    (out/'input.txt').write_text(input_text)
    commands = []
    environment = dict(os.environ, RAYON_NUM_THREADS='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')

    def run(label, command, stdin=None, affinity=False):
        command = list(map(str, command))
        pin = (lambda: os.sched_setaffinity(0, {args.cpu})) if affinity and args.cpu is not None else None
        with (out/f'{label}.stdout').open('w') as stdout, (out/f'{label}.stderr').open('w') as stderr:
            result = subprocess.run(command, cwd=ROOT, text=True, env=environment, input=stdin, stdout=stdout, stderr=stderr, preexec_fn=pin)
        commands.append({'label': label, 'argv': command, 'returncode': result.returncode})
        (out/'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
        if result.returncode:
            raise RuntimeError(f'{label} failed: {out}/{label}.stderr')
        return (out/f'{label}.stdout').read_text()

    command = ['cargo', 'build', '--manifest-path', HERE/'Cargo.toml', '--locked', '--message-format=json']
    if args.timing or args.release:
        command.append('--release')
    if args.allocation_diagnostics:
        command += ['--features', 'allocation-diagnostics']
    build = run('cargo-build', command)
    artifacts = [Path(p) for line in build.splitlines() if line.startswith('{') for row in [json.loads(line)]
                 if row.get('reason') == 'compiler-artifact' and row['target']['name'] == 'ninjars_shared_provider_probe'
                 for p in row['filenames'] if p.endswith('.so')]
    if len(artifacts) != 1:
        raise RuntimeError('expected one Rust provider artifact')
    library = out/'libninjars_shared_provider_probe.so'
    shutil.copy2(artifacts[0], library)
    fortran = Path(subprocess.check_output(['gfortran', '-print-file-name=libgfortran.so'], text=True).strip())
    binary = out/'physical-timing'
    command = ['g++', '-O3', '-std=c++17', '-Wall', '-Wextra', '-Werror',
               '-I'+str(reference/'ninja-source/src'), '-I'+str(reference/'ninja-build/src'), '-I'+str(reference/'ninja-build'),
               HERE/'physical_timing.cc', library, ninja, olo, fortran, '-lquadmath',
               '-Wl,-rpath,'+str(out), '-Wl,-rpath,'+str(fortran.parent), '-o', binary]
    if args.allocation_diagnostics:
        command.insert(1, '-DSP_ALLOCATION_DIAGNOSTICS')
    run('cpp-build', command)
    actual_mode = 'timing' if args.timing else 'allocations' if args.allocation_diagnostics else 'check'
    host_before = host_snapshot()
    stdout = run('physical-workloads', [binary, ROOT/'examples/graphs/physical/gg-hhh/top-pentagon-26.dot', actual_mode, args.rounds, args.trials, args.widths, args.scopes], input_text, affinity=True)
    host_after = host_snapshot()
    results = [json.loads(line) for line in stdout.splitlines() if line.startswith('{')]
    if len(results) != 1 or not results[0]['success']:
        raise RuntimeError('missing successful workload report')
    result = results[0]
    if args.timing:
        for item in result['records']:
            item['summary'] = summarize_record(item)
            item['minimum_observed_block_ns'] = min(min(pair['original_ns'], pair['rust_ns'])*item['rounds'] for pair in item['samples'])
            item['block_duration_below_20ms'] = item['minimum_observed_block_ns'] < 20_000_000
    after = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    result['unchanged_runtime_sources'] = before == after
    result['success'] &= before == after
    result['host_before'], result['host_after'] = host_before, host_after
    result['selected_cpu'] = args.cpu
    result['rust_profile'] = 'release' if args.timing or args.release else 'dev'
    result['thread_environment'] = {key: environment[key] for key in ['RAYON_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS']}
    result['host_isolation_verified'] = False
    result['performance_acceptance'] = False
    result['allocation_scope'] = 'Rust GlobalAlloc only; C++ and native GMP/MPFR allocations are unobserved. Separate build, absent from primary timings.'
    result['normal_allocation_costs_excluded_from_timings'] = False
    result['input_boundary'] = 'Five raw positive-energy momenta, five helicities, seven free model inputs and scale squared. Both complete paths include the same FeynKit numerical preparation and coefficient evaluation.'
    result['original_preparation_batching'] = 'Same width as NinjaRS; shared batched physical preparation precedes scalar original reduction.'
    for item in result['records']:
        item['shared_preparation_width'] = item['width']
        item['original_reduction_batch_width'] = 1 if item['scope'] != 'preparation' else None
        item['rust_reduction_batch_width'] = item['width'] if item['scope'] != 'preparation' else None
    result['warmup_validation_scope'] = 'Integrated Laurent values and internal finite-split identities; preparation baseline checks equality. Raw and Float256 oracle controls are separate.'
    result['separate_physical_controls'] = {
        'path': str(controls_path), 'sha256': digest(controls_path),
        'dependency_heads': controls.get('dependency_heads'),
        'source_sha256': controls.get('source_sha256'),
        'all_runtime_sources_match_this_run': controls.get('source_sha256') == before,
        'integrated_widths': controls['integrated_widths'],
        'preparation_widths': controls['preparation_widths'],
        'raw_components_checked': controls['raw_components_checked'],
        'raw_max_scaled_error': controls['raw_max_scaled_error'],
        'coefficient_oracle_max_scaled_error': controls['coefficient_oracle_max_scaled_error'],
        'precision_oracle_boundary': controls.get('precision_oracle_boundary'),
    }
    result['common_provider_ffi_included'] = True
    result['production_complete_is_unmodified_cli_latency'] = False
    result['source_sha256'] = before
    result['artifacts_sha256'] = {str(p): digest(p) for p in [library, binary, ninja, olo, fortran, out/'input.txt']}
    result['reference_build_configuration'] = config
    result['dependency_manifest'] = json.loads((ROOT/'dependencies/manifest.json').read_text())
    result['dependency_heads'] = {name: subprocess.check_output(['git', '-C', ROOT/'references'/name, 'rev-parse', 'HEAD'], text=True).strip() for name in ['symbolica', 'oneloopmaster', 'gammaloop_ninja_shared']}
    result['compiler_versions'] = {'rustc': subprocess.check_output(['rustc', '-Vv'], text=True), 'cxx': subprocess.check_output(['g++', '--version'], text=True)}
    result['commands'] = commands
    result['logs'] = {p.name: {'sha256': digest(p), 'text': p.read_text()} for p in out.glob('*.stderr')}
    (out/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(out/'result.json')
    if not result['success']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
