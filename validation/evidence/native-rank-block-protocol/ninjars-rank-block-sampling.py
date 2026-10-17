#!/usr/bin/env python3
"""Strict, build-free sampling of reviewed NinjaRS 8b8682 release artifacts.

Default: preflight only. --sample is deliberately explicit and never builds.
This script belongs to evidence, not the frozen measured source tree.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time

sys.dont_write_bytecode = True
SOURCE = '8b868220f3a438a7f78e0e5329cd011646fef71c'
ROOT = Path('/shared/external/ninja/references/ninjars_native_block_harness')
RELEASE = ROOT / 'target/validation/native-block-release32'
HERE = ROOT / 'validation/performance/shared_provider'
WIDTHS = (1, 8, 31, 32, 128, 1024)
CONTROL_WIDTHS = [0, 1, 3, 8, 31, 32, 128, 1024]
ROUNDS = {1: 384, 8: 48, 31: 15, 32: 12, 128: 3, 1024: 3}
TRIALS = 8
SETTINGS = (('native', None), ('native', 32),
            ('native-rank', None), ('native-rank', 32))
COUNTERS = ['numerator_direct_calls', 'numerator_mu_calls', 'numerator_t3_calls',
            'numerator_t2_calls', 'numerator_direct_rows', 'numerator_mu_rows',
            'numerator_t3_rows', 'numerator_t2_rows', 'preparation_calls',
            'preparation_rows', 'wavefunction_calls', 'rust_numerator_wire_calls',
            'rust_numerator_wire_rows', 'rust_master_wire_calls',
            'rust_master_wire_rows', 'shared_live_backend_calls',
            'shared_live_backend_rows', 'live_flag_delta',
            'direct_rust_master_calls', 'direct_rust_master_rows']


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def strict_json(text):
    def bad_constant(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    return json.loads(text, parse_constant=bad_constant)


def read_json(path):
    return strict_json(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def command(argv, *, environment=None):
    return subprocess.check_output(list(map(str, argv)), cwd=ROOT, env=environment,
                                   text=True, stderr=subprocess.STDOUT)


def git(*args):
    return command(['git', *args]).strip()


def label(setting):
    mode, bound = setting
    return mode + ('-whole' if bound is None else '-fixed32')


def dependencies():
    result = {}
    for name in ('symbolica', 'oneloopmaster', 'gammaloop_ninja_shared'):
        path = ROOT / 'references' / name
        require(not git('-C', path, 'status', '--porcelain'),
                'dependency has tracked/untracked modifications: ' + name)
        result[name] = git('-C', path, 'rev-parse', 'HEAD')
    return result


def frozen_checkout():
    require(not git('status', '--porcelain'), 'frozen checkout is dirty')
    command(['git', 'merge-base', '--is-ancestor', SOURCE, 'HEAD'])
    changes = git('diff', '--name-only', SOURCE, 'HEAD').splitlines()
    require(all(path.startswith('validation/evidence/') or path.endswith('.md')
                for path in changes),
            'HEAD differs from the frozen revision beyond documentation/evidence')
    return {'source_revision': SOURCE, 'checkout_head': git('rev-parse', 'HEAD'),
            'documentation_or_evidence_changes': changes}


def checked_environment():
    # No loader injection or mutable inherited search-path overrides.
    for name in ('LD_PRELOAD', 'LD_AUDIT', 'LD_DEBUG', 'GLIBC_TUNABLES'):
        require(not os.environ.get(name), 'unset loader override: ' + name)
    env = dict(os.environ, LD_LIBRARY_PATH=str(RELEASE), RAYON_NUM_THREADS='1',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    return env


def stat_signature(path):
    s = Path(path).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def control_report(report, selection, bound, *, all_catalog=False):
    from kernel_selection import validate_report
    require(report.get('success') is True, 'physical control did not succeed')
    validate_report(report, selection, True, native_block_rows=bound,
                    all_catalog=all_catalog, execution_controls=bound is not None)
    require(report.get('integrated_widths') == CONTROL_WIDTHS,
            'physical control does not cover the exact complete width set')
    require(set(report.get('preparation_widths', [])) == {0, *WIDTHS},
            'physical preparation coverage changed')
    require(report.get('raw_components_checked') == 210,
            'physical raw numerator grid is incomplete')
    require(report.get('batch_laurent_components_checked') == 3 * sum(CONTROL_WIDTHS),
            'physical Laurent control count changed')
    for name in ('raw_max_scaled_error', 'coefficient_oracle_max_scaled_error',
                 'jit_interpreter_max_scaled_error', 'integrated_max_scaled_error',
                 'madloop_integrated_max_scaled_error',
                 'changed_model_coefficient_oracle_max_scaled_error'):
        value = report.get(name)
        require(type(value) in (float, int) and math.isfinite(value)
                and 0 <= value <= 1e-9, 'invalid physical error: ' + name)
    require(report.get('preparation_failure_controls') == 4
            and report.get('transactional_preparation_checked') is True
            and report.get('same_preparation_both_paths') is True,
            'physical transactional/preparation controls missing')
    for name in ('live_original_backend_calls', 'live_original_backend_rows',
                 'live_rust_backend_calls', 'live_rust_backend_rows',
                 'live_rust_repeat_backend_calls', 'live_rust_repeat_backend_rows'):
        require(type(report.get(name)) is int and report[name] > 0,
                'actual live master execution missing: ' + name)


def loader_check(binary, library, environment):
    needed = command(['readelf', '-d', binary], environment=environment)
    output = command(['ldd', binary], environment=environment)
    require('not found' not in output, 'unresolved runtime library')
    matches = []
    loaded = {}
    for line in output.splitlines():
        # Handles both absolute DT_NEEDED entries and SONAME => /path entries.
        match = re.search(r'(?:=>\s*)?(/\S+)\s+\(0x[0-9a-fA-F]+\)', line)
        if match:
            path = Path(match.group(1)).resolve(strict=True)
            loaded[str(path)] = sha(path)
            if path.name == library.name:
                matches.append(path)
    require(matches == [library.resolve(strict=True)],
            'loader did not resolve exactly the frozen copied Rust library')
    require(library.name in needed, 'Rust provider missing from DT_NEEDED')
    return {'readelf_dynamic': needed, 'ldd': output,
            'loaded_libraries_sha256': loaded,
            'provider_exact_target': str(library.resolve(strict=True))}


def compiler_controls(manifest, rich, files):
    names = ['RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS', 'CFLAGS', 'CXXFLAGS',
             'CARGO_PROFILE_RELEASE_OPT_LEVEL', 'CARGO_PROFILE_RELEASE_LTO',
             'CARGO_PROFILE_RELEASE_CODEGEN_UNITS']
    env = manifest.get('compiler_environment')
    require(isinstance(env, dict) and all(name in env for name in names),
            'release compiler environment is not completely recorded')
    for name in names[:4]:
        flags = env[name] or ''
        require(isinstance(flags, str), 'invalid compiler flags: ' + name)
        require(not any(flag in flags for flag in (
            'fast-math', 'unsafe-fp-math', 'finite-math-only', '-Ofast',
            'allocation-diagnostics', 'execution-diagnostics')),
            'relaxed floating point or instrumentation in release flags: ' + name)
    for name, expected in [('OPT_LEVEL', '3'), ('LTO', 'thin'), ('CODEGEN_UNITS', '1')]:
        require(env['CARGO_PROFILE_RELEASE_' + name] in (None, expected),
                'release profile override: ' + name)
    for name in names[:2]:
        flags = (env[name] or '').replace('\x1f', ' ')
        require(not re.search(r'(?:-C\s*|--codegen(?:=|\s+))'
                              r'(?:opt-level|lto|codegen-units)(?:=|\s)', flags),
                'Rust codegen profile override')
    builds = [row for row in rich.get('commands', []) if row.get('label') == 'cargo-build']
    require(len(builds) == 1 and builds[0].get('returncode') == 0,
            'successful physical release cargo command absent')
    cargo = builds[0]['argv']
    require('--release' in cargo and '--locked' in cargo
            and '--features' in cargo
            and cargo[cargo.index('--features') + 1] == 'native-kernels'
            and not any('diagnostics' in arg for arg in cargo),
            'physical artifact lacks the noninstrumented native release build')
    cxx = manifest.get('timing_compile_command')
    require(isinstance(cxx, list) and all(isinstance(arg, str) for arg in cxx),
            'actual timing compiler command absent')
    require('-O3' in cxx and '-o' in cxx
            and cxx[cxx.index('-o') + 1] == str(files['physical-timing'])
            and str(files['libninjars_shared_provider_probe.so']) in cxx
            and str(HERE / 'physical_timing.cc') in cxx
            and not any('DIAGNOSTICS' in arg or 'fast-math' in arg
                        or 'finite-math-only' in arg or arg == '-Ofast' for arg in cxx),
            'timing artifact command differs from noninstrumented O3 shared-library build')
    return {'compiler_environment': env, 'cargo_command': builds[0],
            'timing_compile_command': cxx}


def preflight(expected_manifest_sha, environment):
    from kernel_selection import validate_coverage
    from run_physical import runtime_sources
    from run_timing import protocol
    checkout = frozen_checkout()
    manifest_path = RELEASE / 'release-artifacts.json'
    require(sha(manifest_path) == expected_manifest_sha,
            'reviewed release manifest hash does not match')
    manifest = read_json(manifest_path)
    require(manifest.get('source_revision') == SOURCE, 'wrong release source revision')
    source = {str(p.relative_to(ROOT)): sha(p) for p in runtime_sources()}
    require(manifest.get('source_sha256') == source, 'release runtime source mismatch')
    deps = dependencies()
    require(manifest.get('dependency_heads') == deps, 'release dependency mismatch')
    artifacts = manifest.get('artifacts_sha256')
    require(isinstance(artifacts, dict) and artifacts, 'release artifact hashes absent')
    require(all(Path(p).is_absolute() and sha(p) == digest
                for p, digest in artifacts.items()), 'release artifact changed')
    files = {name: RELEASE / name for name in (
        'result.json', 'all-whole.json', 'native-fixed32.json',
        'native-rank-fixed32.json', 'physical-probe', 'physical-timing',
        'libninjars_shared_provider_probe.so', 'input.txt')}
    require(all(str(path) in artifacts for path in files.values()),
            'manifest lacks a required report, runner, library, or input hash')
    require(files['input.txt'].read_text() == protocol(), 'physical input protocol mismatch')
    rich = read_json(files['result.json'])
    require(rich.get('rust_profile') == 'release'
            and rich.get('unchanged_runtime_sources') is True
            and rich.get('source_sha256') == source
            and rich.get('dependency_heads') == deps,
            'primary physical report is not the matched immutable release control')
    control_report(rich, 'native-rank', 32)
    for path, digest in rich.get('artifacts_sha256', {}).items():
        require(artifacts.get(path) == digest and sha(path) == digest,
                'rich physical artifact differs from release manifest: ' + path)
    require(str(files['libninjars_shared_provider_probe.so'])
            in rich.get('artifacts_sha256', {}), 'physical control library unbound')
    compiler = compiler_controls(manifest, rich, files)
    reports = {}
    for name, mode in [('native-fixed32.json', 'native'),
                       ('native-rank-fixed32.json', 'native-rank')]:
        report = read_json(files[name])
        control_report(report, mode, 32)
        reports[name] = report
    require(all(rich.get(key) == value
                for key, value in reports['native-rank-fixed32.json'].items()),
            'raw primary control does not match rich report')
    whole = read_json(files['all-whole.json'])
    control_report(whole, 'symbolica', None, all_catalog=True)
    reports['all-whole.json'] = whole
    union_entries = rich.get('master_union_entries')
    require(type(union_entries) is int and union_entries > 0
            and all(report.get('master_union_entries') == union_entries
                    for report in reports.values()),
            'controlled union-entry counts are invalid or inconsistent')
    # The primary Whole report is symbolica, not native: select validated catalog rows.
    selected = {}
    for mode, bound in SETTINGS:
        report = reports['all-whole.json' if bound is None else mode + '-fixed32.json']
        row = (next(x for x in report['catalog_parity']
                    if x['coverage']['selection'] == mode)
               if bound is None else report['execution_parity'][1])
        validate_coverage(row['coverage'], mode, True, native_block_rows=bound)
        selected[label((mode, bound))] = {
            'report': 'all-whole.json' if bound is None else mode + '-fixed32.json',
            'selected_parity_row': row,
            'primary_report_selection': report['reduction_kernels']['selection']}
    # Independently recheck the same uncached reference configuration and inputs.
    reference_candidates = [Path(p).parents[3] for p in rich['artifacts_sha256']
                            if p.endswith('/ninja-build/src/.libs/libninja.a')]
    require(len(reference_candidates) == 1, 'uncached reference location ambiguous')
    reference = reference_candidates[0]
    spec = importlib.util.spec_from_file_location('frozen_reference_builder',
                                                 ROOT / 'validation/reference/build.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    config = builder.configuration(reference)
    revision = read_json(reference / 'revisions.json')
    require(config == revision.get('configuration') and config['oneloop']
            and config['higher_rank'] and not config['master_result_cache'],
            'uncached original Ninja configuration differs')
    reference_libs = [reference / 'ninja-build/src/.libs/libninja.a',
                      reference / 'oneloop/libavh_olo.a']
    require(revision.get('libraries_sha256') == {p.name: sha(p) for p in reference_libs}
            and revision.get('input_files_sha256') == builder.library_inputs(reference),
            'original Ninja libraries or build inputs changed')
    require(all(artifacts.get(str(p)) == sha(p) for p in reference_libs),
            'original libraries differ from manifest')
    loader = loader_check(files['physical-timing'],
                          files['libninjars_shared_provider_probe.so'], environment)
    for path, digest in rich['artifacts_sha256'].items():
        if Path(path).name.startswith('libgfortran.so'):
            require(loader['loaded_libraries_sha256'].get(str(Path(path).resolve())) == digest,
                    'loaded Fortran runtime differs from controlled runtime')
    profile = (HERE / 'Cargo.toml').read_text()
    require('[profile.release]\nlto = "thin"\ncodegen-units = 1' in profile,
            'frozen aligned release profile is absent')
    # Commands in the reviewed manifest bind compiler choices. The driver does
    # not infer a release build merely from ELF bytes or mutate any build flags.
    return {'checkout': checkout, 'release_manifest_sha256': expected_manifest_sha,
            'source_sha256': source, 'dependency_heads': deps,
            'artifacts_sha256': artifacts, 'selected_physical_controls': selected,
            'controlled_union_entries': union_entries,
            'reference_configuration': config, 'loader': loader,
            'compiler': compiler,
            'driver_sha256': sha(__file__),
            'reference_builder_sha256': sha(ROOT / 'validation/reference/build.py')}


def schedule():
    result = []
    for sweep in range(4):
        for width_index, width in enumerate(WIDTHS):
            # Every setting occupies each process position once at each width.
            order = (sweep + width_index) % len(SETTINGS)
            for position in range(len(SETTINGS)):
                setting = SETTINGS[(order + position) % len(SETTINGS)]
                result.append({'sweep': sweep, 'position': position, 'width': width,
                               'rounds': ROUNDS[width], 'trials': TRIALS,
                               'selection': setting[0], 'native_block_rows': setting[1],
                               'setting': label(setting)})
    return result


def validate_timing(report, job, controlled_union_entries):
    from kernel_selection import validate_report
    from timing_statistics import summarize_record
    require(report.get('success') is True and report.get('mode') == 'timing',
            'missing successful timing report')
    require(set(report) == {
        'success', 'reduction_kernels', 'mode', 'clock_pair_median_ns',
        'performance_acceptance', 'original_production_backend_counters_available',
        'counter_columns', 'records', 'frozen_backend_call_delta',
        'frozen_backend_row_delta', 'checksum', 'union_entries',
        'union_replay_rows', 'union_misses'}, 'unexpected timing report fields')
    require(report.get('performance_acceptance') is False
            and report.get('original_production_backend_counters_available') is False,
            'primary timing artifact unexpectedly instrumented or accepted')
    validate_report(report, job['selection'], True,
                    native_block_rows=job['native_block_rows'])
    require(report.get('counter_columns') == COUNTERS, 'counter schema changed')
    for name in ('frozen_backend_call_delta', 'frozen_backend_row_delta', 'union_misses'):
        require(type(report.get(name)) is int and report[name] == 0,
                'frozen reduction executed backend or missed union: ' + name)
    require(type(report.get('checksum')) in (int, float)
            and math.isfinite(report['checksum']), 'invalid retained checksum')
    require(type(report.get('clock_pair_median_ns')) in (int, float)
            and math.isfinite(report['clock_pair_median_ns'])
            and report['clock_pair_median_ns'] >= 0, 'invalid clock overhead')
    require(type(controlled_union_entries) is int and controlled_union_entries > 0
            and type(report.get('union_entries')) is int
            and report['union_entries'] == controlled_union_entries,
            'union-entry count differs from release physical controls')
    require(type(report.get('union_replay_rows')) is int
            and report['union_replay_rows'] > 0, 'missing frozen replay rows')
    records = report.get('records')
    require(isinstance(records, list) and len(records) == 2,
            'exactly repeated/changing records required')
    for row, workload in zip(records, ('repeated', 'changing')):
        require(set(row) == {
            'scope', 'workload', 'width', 'rounds', 'contiguous_blocks',
            'counters_outside_blocks', 'retained_reducer_sessions',
            'prebound_numerator_states', 'correctness_components',
            'max_scaled_error', 'samples', 'raw_block_columns',
            'raw_block_nanoseconds', 'original_counter_deltas',
            'rust_counter_deltas'}, 'unexpected timing record fields')
        width = job['width']
        require(type(row.get('width')) is int and type(row.get('rounds')) is int
                and (row.get('scope'), row.get('workload'), row.get('width'),
                 row.get('rounds')) == ('shared-reduction', workload, width, job['rounds']),
                'timing matrix record mismatch')
        require(row.get('contiguous_blocks') is True
                and row.get('counters_outside_blocks') is True
                and type(row.get('retained_reducer_sessions')) is int
                and row['retained_reducer_sessions'] == 1
                and type(row.get('prebound_numerator_states')) is int
                and row['prebound_numerator_states'] == 3,
                'shared reduction timing boundary changed')
        require(type(row.get('correctness_components')) is int
                and row['correctness_components'] == 9 * width
                and type(row.get('max_scaled_error')) in (int, float)
                and math.isfinite(row['max_scaled_error'])
                and 0 <= row['max_scaled_error'] <= 1e-9,
                'warmup physical Laurent validation incomplete')
        samples, blocks = row.get('samples'), row.get('raw_block_nanoseconds')
        require(isinstance(samples, list) and len(samples) == TRIALS
                and isinstance(blocks, list) and len(blocks) == TRIALS
                and row.get('raw_block_columns') == ['original_ns', 'rust_ns'],
                'raw paired timing block schema/count mismatch')
        for index, (pair, raw) in enumerate(zip(samples, blocks)):
            require(pair.get('order') == ('rust-first' if index % 2 else 'original-first'),
                    'unbalanced or reordered paired sampling')
            require(isinstance(raw, list) and len(raw) == 2
                    and all(type(x) in (int, float) and math.isfinite(x) and x > 0
                            for x in raw), 'invalid raw block duration')
            require(raw[0] / job['rounds'] == pair['original_ns']
                    and raw[1] / job['rounds'] == pair['rust_ns'],
                    'raw timing normalization mismatch')
        scale = TRIALS * job['rounds']
        original = [10 * width, 0, 25 * width, 20 * width] * 2 + [0] * 12
        rust = [5, 0, 15, 10, 10 * width, 0, 25 * width, 20 * width,
                0, 0, 0, 30, 55 * width, 5, 50 * width, 0, 0, 0, 0, 0]
        for name, expected in [('original_counter_deltas', original),
                               ('rust_counter_deltas', rust)]:
            values = row.get(name)
            require(isinstance(values, list) and all(type(v) is int for v in values)
                    and values == [v * scale for v in expected],
                    'scope1 provider/preparation/master counter mismatch: ' + name)
        row['summary'] = summarize_record(row)
        row['minimum_observed_block_ns'] = min(min(x) for x in blocks)
        row['block_duration_below_20ms'] = row['minimum_observed_block_ns'] < 20_000_000
    return report


def comparisons(results):
    # Per-sweep ratios are across distinct processes, not original/Rust pairs.
    values = {}
    for item in results:
        for row in item['report']['records']:
            values[(item['job']['sweep'], row['width'], row['workload'],
                    item['job']['setting'])] = statistics.median(
                        sample['rust_ns'] for sample in row['samples'])
    contrasts = [('block-with-generic', 'native-fixed32', 'native-whole'),
                 ('block-with-rank', 'native-rank-fixed32', 'native-rank-whole'),
                 ('rank-with-whole', 'native-rank-whole', 'native-whole'),
                 ('rank-with-fixed32', 'native-rank-fixed32', 'native-fixed32')]
    out = []
    for width in WIDTHS:
        for workload in ('repeated', 'changing'):
            for name, numerator, denominator in contrasts:
                ratios = [values[(s, width, workload, numerator)] /
                          values[(s, width, workload, denominator)] for s in range(4)]
                median = statistics.median(ratios)
                out.append({'contrast': name, 'width': width, 'workload': workload,
                            'numerator': numerator, 'denominator': denominator,
                            'four_sweep_process_median_ratios': ratios, 'median': median,
                            'mad': statistics.median(abs(x - median) for x in ratios),
                            'confidence_interval': None,
                            'scope': 'Distinct process medians within each rotation; '
                                     'not paired blocks, no cross-variant bootstrap CI.'})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reviewed-release-manifest-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cpu', type=int, default=75)
    parser.add_argument('--sample', action='store_true', help='explicit sampling, after review')
    args = parser.parse_args()
    require(re.fullmatch('[0-9a-f]{64}', args.reviewed_release_manifest_sha256) is not None,
            'a reviewed SHA256 digest is required')
    require(args.cpu in os.sched_getaffinity(0), 'requested CPU is not allowed')
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    require(not any(out.iterdir()), 'output directory must be empty')
    require(not out.is_relative_to(ROOT), 'evidence output must stay outside frozen checkout')
    sys.path.insert(0, str(HERE))
    from run_timing import host_snapshot
    environment = checked_environment()
    shutil.copyfile(__file__, out / 'driver.py')
    write_json(out / 'schedule.json', schedule())
    write_json(out / 'request.json', {'argv': sys.argv, 'source_revision': SOURCE,
               'fixed_rounds': ROUNDS, 'trials': TRIALS, 'processes': 96,
               'records': 192, 'paired_blocks': 1536, 'individual_blocks': 3072,
               'selected_cpu': args.cpu, 'selected_cpu_siblings':
               Path(f'/sys/devices/system/cpu/cpu{args.cpu}/topology/thread_siblings_list').read_text(),
               'environment': {key: environment[key] for key in (
                   'LD_LIBRARY_PATH', 'RAYON_NUM_THREADS', 'OMP_NUM_THREADS',
                   'OPENBLAS_NUM_THREADS')},
               'no_timing_while_building': 'Coordinator must reserve a build-free window; '
                                         'the numerical lock alone does not block compilers.',
               'performance_acceptance': False, 'host_isolation_verified': False,
               'duration_policy': 'Every individual block must be >=20ms for a suitable '
                                  'diagnostic summary; retain all shorter blocks as a pilot. '
                                  'No adaptive rounds, selective reruns or discarded samples.',
               'claims': 'Four settings share one current executable/library. '
                         'Rank/block effects are conditional within the 2x2 matrix. '
                         'No causal comparison to historical source revisions.',
               'head_count_limitation': 'Provider counters measure calls and rows, not '
                         'heads. Source audit gives n5/r4 T2 output heads per point '
                         'original=40, Rust=60 in all four settings; not an observed '
                         'head counter. Configuration metadata does not count rank-guard '
                         'hits or individual native block invocations.',
               'scope': 'Scope1 only: callbacks, geometry/reduction, frozen master '
                         'lookup/contraction and staging. No physical preparation or '
                         'actual master backend. Separate instrumented preparation/master '
                         'and complete-evaluation passes remain necessary.'})
    write_json(out / 'host-preflight.json', host_snapshot())
    before = preflight(args.reviewed_release_manifest_sha256, environment)
    write_json(out / 'preflight.json', before)
    print('Strict preflight passed; ' + ('sampling requested' if args.sample else 'no sampling'),
          flush=True)
    if not args.sample:
        return
    binaries = [RELEASE / 'physical-timing', RELEASE / 'libninjars_shared_provider_probe.so',
                RELEASE / 'input.txt', RELEASE / 'release-artifacts.json']
    signatures = {str(path): stat_signature(path) for path in binaries}
    commands, results = [], []
    # The lock serializes all numerical executables in this coordinated window.
    with open('/tmp/ninjars-symbolica-tests.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        write_json(out / 'host-window-before.json', host_snapshot())
        for index, job in enumerate(schedule()):
            require(all(stat_signature(path) == sig for path, sig in signatures.items()),
                    'frozen artifact metadata changed during sampling')
            name = f'{index:03d}-s{job["sweep"]}-w{job["width"]}-{job["setting"]}'
            argv = ['taskset', '-c', str(args.cpu), str(RELEASE / 'physical-timing'),
                    str(ROOT / 'examples/graphs/physical/gg-hhh/top-pentagon-26.dot'),
                    'timing', str(job['rounds']), str(TRIALS), str(job['width']), '1',
                    job['selection']]
            if job['native_block_rows'] is not None:
                argv.append(str(job['native_block_rows']))
            entry = {'index': index, 'job': job, 'argv': argv,
                     'start_utc_seconds': time.time(), 'status': 'running'}
            commands.append(entry)
            write_json(out / 'commands.json', commands)
            with (out / (name + '.stdout')).open('w') as stdout, \
                    (out / (name + '.stderr')).open('w') as stderr, \
                    (RELEASE / 'input.txt').open() as stdin:
                completed = subprocess.run(argv, cwd=ROOT, env=environment, stdin=stdin,
                                           stdout=stdout, stderr=stderr)
            entry.update(returncode=completed.returncode, end_utc_seconds=time.time(),
                         status='complete',
                         stdout_sha256=sha(out / (name + '.stdout')),
                         stderr_sha256=sha(out / (name + '.stderr')))
            write_json(out / 'commands.json', commands)
            require(completed.returncode == 0, 'workload failed; raw logs retained: ' + name)
            text = (out / (name + '.stdout')).read_text()
            reports = [strict_json(line) for line in text.splitlines() if line.startswith('{')]
            require(len(reports) == 1, 'missing or ambiguous timing JSON: ' + name)
            report = validate_timing(reports[0], job, before['controlled_union_entries'])
            results.append({'job': job, 'report': report, 'raw_stdout': name + '.stdout',
                            'raw_stderr': name + '.stderr'})
            write_json(out / (name + '.validated.json'), report)
            print(f'Completed {index + 1}/96: sweep {job["sweep"] + 1}, '
                  f'width {job["width"]}, {job["setting"]}', flush=True)
        write_json(out / 'host-window-after.json', host_snapshot())
    after = preflight(args.reviewed_release_manifest_sha256, environment)
    # ldd includes ASLR addresses; retain that raw evidence without treating its
    # changing addresses as a change to immutable code or loader targets.
    def identity(record):
        record = dict(record)
        record['loader'] = {key: value for key, value in record['loader'].items()
                            if key not in ('ldd', 'readelf_dynamic')}
        return record
    require(identity(before) == identity(after),
            'source/dependency/artifact/loader identity changed during window')
    write_json(out / 'postflight.json', after)
    short = [{'job': item['job'], 'workload': row['workload'],
              'minimum_observed_block_ns': row['minimum_observed_block_ns']}
             for item in results for row in item['report']['records']
             if row['block_duration_below_20ms']]
    summary = {'success': True, 'records': 192, 'processes': 96,
               'paired_blocks': 1536, 'individual_blocks': 3072,
               'pilot': bool(short), 'duration_suitable': not short,
               'short_records': short, 'duration_floor_ns': 20_000_000,
               'performance_acceptance': False, 'host_isolation_verified': False,
               'diagnostic_only': True, 'contrasts': comparisons(results),
               'uncertainty': 'Each record retains original/Rust stratified paired '
                 'bootstrap intervals. Cross-setting contrasts retain four process '
                 'rotation ratios, median/MAD, and no claimed bootstrap interval. '
                 'Neither establishes host isolation.',
               'all_raw_blocks_retained': True, 'adaptive_sampling': False}
    write_json(out / 'summary.json', summary)
    write_json(out / 'archive-sha256.json', {p.name: sha(p) for p in sorted(out.iterdir())
                                           if p.is_file() and p.name != 'archive-sha256.json'})
    print('Complete. ' + ('PILOT: duration floor missed.' if short else
                         'Duration floor passed; shared-host diagnostic only.'), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'FAILED: {error}', file=sys.stderr, flush=True)
        raise
