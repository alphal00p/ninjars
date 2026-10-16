#!/usr/bin/env python3
"""Record a controlled Rust/original-Ninja complete-evaluation comparison."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('performance_build', Path(__file__).with_name('build.py'))
build_provenance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_provenance)
THREADS = {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
           'RAYON_NUM_THREADS': '1', 'SYMBOLICA_HIDE_BANNER': '1'}
BUILD_NAMES = {'cargo', 'rustc', 'cc1', 'cc1plus', 'f951', 'g++', 'gcc', 'gfortran',
               'make', 'cmake', 'ninja', 'ld', 'ld.lld'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


@contextmanager
def lock(directory, name, operation):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/name
    # Only the process which successfully creates the file owns its cleanup.
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            json.dump({'pid': os.getpid(), 'operation': operation}, stream)
        yield
    finally:
        path.unlink()


def git(*args, directory=ROOT):
    return subprocess.check_output(['git', '-C', str(directory), *args], text=True).strip()


def source_snapshot(output):
    files = set(subprocess.check_output(['git', 'ls-files', '-z', '--cached', '--others',
                                         '--exclude-standard'], cwd=ROOT).decode().split('\0'))
    return {name: digest(ROOT/name) for name in sorted(files) if name
            and (ROOT/name).is_file() and not (ROOT/name).is_relative_to(output)}


def build_processes():
    found = []
    for path in Path('/proc').glob('[0-9]*/comm'):
        try:
            name = path.read_text().strip()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if name in BUILD_NAMES:
            process = {'pid': int(path.parent.name), 'executable': name}
            # Do not collect command lines or environments: either can contain
            # credentials. Parent/state/cwd are enough to locate interference.
            try:
                fields = dict(line.split(':', 1) for line in (path.parent/'status').read_text().splitlines()
                              if ':' in line)
                process.update(parent_pid=int(fields['PPid']), state=fields['State'].strip())
                process['cwd'] = os.readlink(path.parent/'cwd')
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                process['details_unavailable'] = True
            found.append(process)
    return sorted(found, key=lambda process: process['pid'])


def host(cpu):
    details = {'system': platform.platform(), 'machine': platform.machine(), 'cpu': cpu,
               'controller_affinity': sorted(os.sched_getaffinity(0)),
               'load_average': os.getloadavg(), 'active_build_processes': build_processes()}
    information = Path('/proc/cpuinfo').read_text()
    for block in information.split('\n\n'):
        fields = dict(line.split(':', 1) for line in block.splitlines() if ':' in line)
        fields = {key.strip(): value.strip() for key, value in fields.items()}
        if fields.get('processor') == str(cpu):
            details['cpu_description'] = {key: fields[key] for key in
                                          ['model name', 'vendor_id', 'cpu family', 'model', 'stepping', 'microcode']
                                          if key in fields}
            break
    policy = Path(f'/sys/devices/system/cpu/cpu{cpu}/cpufreq')
    for name in ['scaling_governor', 'scaling_cur_freq', 'scaling_min_freq', 'scaling_max_freq']:
        try:
            details[name] = (policy/name).read_text().strip()
        except (FileNotFoundError, PermissionError):
            details[name] = None
    return details


def capture_host(output, cpu, phase):
    snapshot = host(cpu)
    snapshot['recorded_unix_ns'] = time.time_ns()
    # Save before applying the quiet-window guard, so a rejected run retains
    # precisely the processes which caused rejection, even if they then exit.
    write_json(output/f'host-{phase}.json', snapshot)
    return snapshot


def failure_record(error, output):
    return {'error': str(error), 'performance_acceptance': False,
            'recorded_unix_ns': time.time_ns(), 'active_build_processes': build_processes(),
            'host_snapshots': {path.stem: json.loads(path.read_text())
                               for path in sorted(output.glob('host-*.json'))}}


def json_output(text):
    # Original OneLOop can emit a license banner before its one-line JSON.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        lines = [line for line in text.splitlines() if line.startswith('{')]
        if len(lines) != 1:
            raise ValueError('expected exactly one JSON result')
        return json.loads(lines[0])


def pair(value):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError('missing complex coefficient')
    real, imaginary = map(float, value)
    if not math.isfinite(real) or not math.isfinite(imaginary):
        raise ValueError('nonfinite complex coefficient')
    return complex(real, imaginary)


def scaled_error(actual, expected):
    # Rescaling prevents intermediate norms/subtractions overflowing for finite
    # binary64 values. This is exactly |a-b| / max(1, |b|).
    scale = max(1., abs(actual.real), abs(actual.imag), abs(expected.real), abs(expected.imag))
    return abs(actual/scale-expected/scale)/max(1./scale, abs(expected/scale))


def check_results(rust, original):
    rows = rust['results']
    if not original['success'] or not original['pre_post_equal']:
        raise ValueError('original reference has failed/nonrepeatable rows')
    checks = []
    for phase in ['pre_results', 'post_results']:
        reference = original[phase]
        if len(reference) != len(rows):
            raise ValueError('Rust and original sample counts differ')
        for index, (actual, expected) in enumerate(zip(rows, reference)):
            if actual['lane'] != index or expected['lane'] != index:
                raise ValueError('result order differs from the source pool')
            if actual['status']['kind'] != 'success' or expected['status'] != 0 or not expected['finite']:
                raise ValueError('comparison contains failed numerical rows')
            if len(actual['values']) != 5 or len(expected['epsilon']) != 3:
                raise ValueError('unexpected coefficient output layout')
            errors = [scaled_error(pair(a), pair(b)) for a, b in zip(actual['values'][:3], expected['epsilon'])]
            if len(errors) != 3 or any(not math.isfinite(error) or error > 1e-9 for error in errors):
                raise ValueError(f'Laurent comparison failed in {phase} row {index}: {errors}')
            split_errors = []
            for values in [[pair(value) for value in actual['values']],
                           [*map(pair, expected['epsilon']), pair(expected['cut_constructible']), pair(expected['rational'])]]:
                split_errors.append(scaled_error(values[3]+values[4], values[0]))
            if any(not math.isfinite(error) or error > 1e-9 for error in split_errors):
                raise ValueError(f'finite-part consistency failed: {split_errors}')
            checks.append({'phase': phase, 'lane': index, 'laurent_scaled_errors': errors,
                           'finite_split_errors': split_errors})
    return checks


def validate_build_record(record, binary):
    if not isinstance(record, dict):
        raise ValueError('Rust build provenance must be a JSON object')
    if record.get('binary_sha256') != digest(binary):
        raise ValueError('Rust build record does not match the supplied executable')
    command = record.get('command')
    if not isinstance(command, list) or '--release' not in command:
        raise ValueError('retained timings require an explicit optimized Rust build command')
    text = json.dumps([record.get('environment', {}), record.get('cargo_configuration', {})]).lower()+' '+' '.join(command).lower()
    if any(flag in text for flag in ['fast-math', 'fast_math', '-ofast', 'unsafe-fp-math']):
        raise ValueError('fast-math is incompatible with this comparison')
    current = build_provenance.production_snapshot()['source_sha256']
    recorded = record.get('checkout', {}).get('source_sha256', {})
    # The first retained record also hashed fixtures/validation sources. They
    # did not build the executable. Keep that record unchanged, but compare the
    # same production input scope as the reproducible build recorder now uses.
    recorded = {name: value for name, value in recorded.items()
                if name.startswith(('crates/', 'dependencies/')) or name in build_provenance.TOP_LEVEL}
    if current != recorded:
        changed = sorted(name for name in current.keys() | recorded.keys()
                         if current.get(name) != recorded.get(name))
        raise ValueError(f'Rust build production sources differ from the current checkout: {changed}')
    dependencies = build_provenance.checkpoint.dependencies()
    if dependencies != record.get('dependencies'):
        raise ValueError('Rust build dependency heads, tracked changes or patch records differ from the actual vendored repositories')
    configuration = build_provenance.cargo_configuration()
    if configuration != record.get('cargo_configuration'):
        raise ValueError('Rust build Cargo configuration differs from the current configuration')
    return {'production_source_sha256': current, 'dependencies': dependencies,
            'cargo_configuration': configuration}


def compare(args, output):
    binary, fixture = args.ninjars.resolve(), args.fixture.resolve()
    allowed = sorted(os.sched_getaffinity(0))
    cpu = allowed[0] if args.cpu is None else args.cpu
    if cpu not in allowed:
        raise ValueError(f'CPU {cpu} is outside the allowed affinity set')
    if not args.smoke and not args.rust_build_record:
        raise ValueError('--rust-build-record is required for retained optimized timings')
    build_record = json.loads(args.rust_build_record.read_text()) if args.rust_build_record else None
    build_validation = None
    if args.rust_build_record:
        build_validation = validate_build_record(build_record, binary)
        write_json(output/'rust-build.json', build_record)
        write_json(output/'build-validation.json', build_validation)
    dependencies = build_provenance.checkpoint.dependencies()
    configuration = build_provenance.cargo_configuration()
    write_json(output/'dependency-state.json', dependencies)
    sources = source_snapshot(output)
    write_json(output/'source-sha256.json', sources)
    manifest = ROOT/'dependencies/manifest.json'
    initial = {'binary_sha256': digest(binary), 'fixture_sha256': digest(fixture),
               'dependency_manifest_sha256': digest(manifest), 'source_snapshot_sha256': digest(output/'source-sha256.json'),
               'dependency_state_sha256': digest(output/'dependency-state.json'),
               'cargo_configuration': configuration,
               'git_head': git('rev-parse', 'HEAD'), 'git_status': git('status', '--porcelain'),
               'allowed_affinity': allowed, 'child_affinity': [cpu], 'thread_environment': THREADS,
               'rust_build_record_sha256': digest(args.rust_build_record) if args.rust_build_record else None,
               'python': sys.version, 'controller_command': sys.argv, 'working_directory': str(ROOT)}
    commands = []
    environment = {**os.environ, **THREADS}

    def run(label, command, data=None, parse=True):
        command = list(map(str, command))
        record = {'label': label, 'argv': command, 'cwd': str(ROOT), 'cpu_affinity': [cpu],
                  'stdin_sha256': hashlib.sha256(data.encode()).hexdigest() if data is not None else None}
        commands.append(record)
        write_json(output/'commands.json', commands)
        started = time.time_ns()
        result = subprocess.run(command, input=data, capture_output=True, text=True, cwd=ROOT,
                                env=environment, preexec_fn=lambda: os.sched_setaffinity(0, {cpu}))
        record.update(returncode=result.returncode, wall_time_ns=time.time_ns()-started)
        (output/f'{label}.stdout').write_text(result.stdout)
        (output/f'{label}.stderr').write_text(result.stderr)
        write_json(output/'commands.json', commands)
        if result.returncode:
            raise RuntimeError(f'{label} failed; see {output}/{label}.stderr')
        return json_output(result.stdout) if parse else None

    bundle = output/'pool'
    run('export', [binary, 'prepare', fixture, '--reference-bench-dir', bundle])
    pool_manifest = json.loads((bundle/'manifest.json').read_text())
    if pool_manifest['propagator_count'] < 1 or not pool_manifest['samples']:
        raise ValueError('empty or unsupported source pool')
    changing = args.workload != 'repeated'
    if changing and (len(pool_manifest['samples']) < 2 or args.points < 2):
        raise ValueError('changing workloads require at least two source rows and points')
    build = args.reference_build_dir.resolve()
    runner = output/'ninja-benchmark'
    command = [sys.executable, ROOT/'validation/reference/build.py', '--benchmark', '--disable-avholo-cache',
               '--build-dir', build, '--numerator-header', bundle/'numerator.hh', '--output', runner, '--jobs', '2']
    if (build/'ninja-build/src/.libs/libninja.a').exists():
        command.append('--runner-only')
    with lock(build, '.ninjars-build.lock', 'performance reference build'):
        run('reference-build', command, parse=False)
    reference_artifacts = {str(path.relative_to(output)): digest(path)
                           for path in [runner, runner.with_suffix('.json'), bundle/'numerator.hh',
                                        bundle/'pool.inputs', bundle/'manifest.json', bundle/'prepared.json']}
    flags = ['--mu-sampling'] if args.mu_extraction == 'sampling' else []
    rust_flags = ['--mu-extraction', args.mu_extraction]
    reduced = run('rust-reduce', [binary, 'reduce', fixture, '--precision', 'f64', '--batch-size', '2',
                                  '--cut-stop', '1', *rust_flags])
    pool_text = (bundle/'pool.inputs').read_text().rstrip()+'\n'
    preflight_input = pool_text+'1\n0 0 1 1 1 1\n'
    (output/'preflight.inputs').write_text(preflight_input)
    preflight = run('reference-preflight', [runner, '--cut-stop', '1', *flags], preflight_input)
    preflight_checks = check_results(reduced, preflight)
    write_json(output/'preflight-checks.json', preflight_checks)

    workloads = ['repeated', 'changing'] if args.workload == 'both' else [args.workload]
    jobs = [(index, int(workload == 'changing'), args.points, args.warmup, args.rounds, args.iterations)
            for index, workload in enumerate(workloads)]
    full_input = pool_text+str(len(jobs))+'\n'+''.join(' '.join(map(str, job))+'\n' for job in jobs)
    (output/'benchmark.inputs').write_text(full_input)
    benchmark_command = [binary, 'bench', fixture, '--precision', 'f64', '--batch-sizes', args.batch_sizes,
                         '--points', str(args.points), '--warmup', str(args.warmup), '--rounds', str(args.rounds),
                         '--iterations', str(args.iterations), '--workload', args.workload,
                         '--cut-stop', '1', *rust_flags]
    if args.rust_build_record and validate_build_record(build_record, binary) != build_validation:
        raise ValueError('Rust build provenance changed before timing')
    if dependencies != build_provenance.checkpoint.dependencies() or configuration != build_provenance.cargo_configuration():
        raise ValueError('dependency state or Cargo configuration changed before timing')
    before = capture_host(output, cpu, 'before')
    if before['active_build_processes'] and not args.smoke:
        raise RuntimeError('build processes are active; retain timings only after the build window is quiet')
    # Each process retains its internal numerical session. Order is explicit so
    # independent repeats can reverse it to investigate drift.
    if args.order == 'rust-first':
        rust_bench = run('rust-bench', benchmark_command)
        original_bench = run('reference-bench', [runner, '--cut-stop', '1', *flags], full_input)
    else:
        original_bench = run('reference-bench', [runner, '--cut-stop', '1', *flags], full_input)
        rust_bench = run('rust-bench', benchmark_command)
    after = capture_host(output, cpu, 'after')
    if after['active_build_processes'] and not args.smoke:
        raise RuntimeError('build processes appeared during the timing window')
    if not rust_bench['success'] or (not args.smoke and rust_bench['build_profile'] != 'release'):
        raise ValueError('Rust benchmark failed or is not optimized')
    final_checks = check_results(reduced, original_bench)
    reference_jobs = {job['workload']: job for job in original_bench['jobs']}
    ratios = []
    for case in rust_bench['cases']:
        original = reference_jobs[case['workload']]
        if (case['timed_points_per_round'] != original['timed_points_per_round']
                or len(case['round_ns']) != args.rounds or len(original['round_ns']) != args.rounds):
            raise ValueError('benchmark iteration/round counts do not match')
        if any(not math.isfinite(value) or value <= 0 for value in [*case['round_ns'], *original['round_ns']]):
            raise ValueError('nonpositive or nonfinite timing sample')
        if case['diagnostic']['failed_lanes'] or case['warmup_final_pass_failed_lanes']:
            raise ValueError('Rust timing case contains failed lanes')
        lane = case['representative_source_lane']
        for value, expected in zip(case['representative_result'][:3], reduced['results'][lane]['values'][:3]):
            if scaled_error(pair(value), pair(expected)) > 1e-9:
                raise ValueError('Rust benchmark representative differs from its validated reduction')
        reference_median = statistics.median(original['round_ns'])/original['timed_points_per_round']
        rust_median = statistics.median(case['round_ns'])/case['timed_points_per_round']
        ratios.append({'workload': case['workload'], 'batch_size': case['batch_size'],
                       'rust_round_ns': case['round_ns'], 'reference_round_ns': original['round_ns'],
                       'points_per_round': original['timed_points_per_round'],
                       'rust_median_ns_per_point': rust_median, 'reference_median_ns_per_point': reference_median,
                       'rust_over_reference_latency': rust_median/reference_median,
                       'rust_over_reference_throughput': reference_median/rust_median,
                       'full_batches': case['full_batches'], 'tail_lanes': case['tail_lanes']})
    if digest(binary) != initial['binary_sha256'] or digest(fixture) != initial['fixture_sha256']:
        raise ValueError('binary or fixture changed during comparison')
    if digest(manifest) != initial['dependency_manifest_sha256'] or source_snapshot(output) != sources:
        raise ValueError('source/dependency snapshot changed during comparison')
    if dependencies != build_provenance.checkpoint.dependencies():
        raise ValueError('actual vendored dependency heads or tracked changes changed during comparison')
    if configuration != build_provenance.cargo_configuration():
        raise ValueError('Cargo configuration changed during comparison')
    if git('rev-parse', 'HEAD') != initial['git_head']:
        raise ValueError('checkout revision changed during comparison')
    if any(digest(output/path) != value for path, value in reference_artifacts.items()):
        raise ValueError('reference binary or exported input changed during comparison')
    if args.rust_build_record and digest(args.rust_build_record) != initial['rust_build_record_sha256']:
        raise ValueError('Rust build provenance changed during comparison')
    report = {'schema_version': 1, 'operation': 'complete_evaluation_performance_comparison',
              'correctness_passed': True, 'performance_acceptance': False, 'smoke': args.smoke,
              'timing_scope': 'diagnostic smoke only' if args.smoke else 'optimized complete evaluation; acceptance gates pending',
              'acceptance_limitations': ['shared-master reduction is not measured', 'complete fixed-suite and uncertainty gates are pending',
                                         'DoubleFloat/arbitrary-precision/Dual reference performance is not measured'],
              'criterion': 'each total Laurent coefficient: abs(a-b)/max(1,abs(b)) <= 1e-9',
              'finite_split_policy': 'each implementation checked internally; cross-basis splits retained without equating them',
              'comparison_order': args.order, 'cut_stop': 1, 'coefficient_mode': 'integrated',
              'mu_extraction': args.mu_extraction, 'pool_manifest': pool_manifest,
              'cache_policy': {'rust_master_results': 'not cached', 'original_master_results': 'disabled at configure time'},
              'sampling_scales': {'rust': 'topology-wide magnitudes and exact-zero fallback; see Rust report',
                                  'original': original_bench['sampling_scale_policy']},
              'uncertainty_policy': 'raw rounds and ratios of observed medians; no confidence-interval claim',
              'reference_scalar_policy': 'one scalar serial point loop per workload; same reference rounds reused for each Rust batch width',
              'provenance': initial, 'reference_artifacts_sha256': reference_artifacts,
              'build_validation': build_validation, 'dependency_state': dependencies,
              'host_before': before, 'host_after': after,
              'commands': commands, 'preflight_checks': preflight_checks, 'final_checks': final_checks,
              'ratios': ratios, 'rust_bench': rust_bench, 'original_bench': original_bench,
              'rust_reduce': reduced, 'reference_build': json.loads(runner.with_suffix('.json').read_text())}
    write_json(output/'comparison.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fixture', type=Path)
    parser.add_argument('--ninjars', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True, help='new or empty artifact directory')
    parser.add_argument('--rust-build-record', type=Path)
    parser.add_argument('--reference-build-dir', type=Path, default=ROOT/'target/reference-uncached-v2')
    parser.add_argument('--cpu', type=int)
    parser.add_argument('--batch-sizes', default='1,8,32,128,1024')
    parser.add_argument('--points', type=int, default=1025)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=1)
    parser.add_argument('--workload', choices=['repeated', 'changing', 'both'], default='both')
    parser.add_argument('--mu-extraction', choices=['expansion', 'sampling'], default='expansion')
    parser.add_argument('--order', choices=['rust-first', 'reference-first'], default='rust-first')
    parser.add_argument('--smoke', action='store_true', help='diagnostic timings; permit debug binary and active builds')
    args = parser.parse_args()
    if min(args.points, args.warmup, args.rounds, args.iterations) < 1:
        parser.error('point/warmup/round/iteration counts must be positive')
    try:
        widths = list(map(int, args.batch_sizes.split(',')))
    except ValueError:
        parser.error('batch sizes must be comma-separated integers')
    if not widths or min(widths) < 1 or len(set(widths)) != len(widths):
        parser.error('batch sizes must be positive and distinct')
    if max(args.points*args.rounds*args.iterations, args.points*args.warmup, *widths) >= 1 << 64:
        parser.error('workload count overflow')
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('artifact directory is not empty; choose a fresh output directory')
    with lock(output, '.ninjars-performance.lock', 'complete evaluation comparison'):
        try:
            report = compare(args, output)
        except Exception as error:
            write_json(output/'failure.json', failure_record(error, output))
            raise
    print(f'{len(report["ratios"])} timing comparisons; correctness passed; performance acceptance remains false')
    print(output/'comparison.json')


if __name__ == '__main__':
    main()
