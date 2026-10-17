#!/usr/bin/env python3
"""Check or explicitly sample separate matched physical workload boundaries."""
import argparse
import json
import math
import os
import re
from pathlib import Path
import shutil
import subprocess

from run_physical import ROOT, HERE, digest, runtime_sources
from timing_statistics import summarize_record
from execution_report import validate_execution_record
from kernel_selection import (add_arguments, native_enabled, validate_arguments, validate_report,
                              validate_pruning_preflight)


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
    mode.add_argument('--execution-diagnostics', action='store_true', help='separate instrumented stage/backend pass; never primary timing')
    parser.add_argument('--release', action='store_true', help='optimized correctness/allocation build; timing always uses release')
    parser.add_argument('--cpu', type=int, help='pin runtime to this already-allowed CPU; does not reserve its sibling')
    parser.add_argument('--widths', default='1,8,32,128,1024,31')
    parser.add_argument('--scopes', help='0 preparation,1 shared reduction,2 shared-live complete,3 production complete; timing defaults to1, other modes to all four')
    parser.add_argument('--rounds', type=int, default=384)
    parser.add_argument('--trials', type=int, default=8)
    parser.add_argument('--reference-build-dir', type=Path, default=ROOT/'target/reference-uncached-v2')
    parser.add_argument('--physical-controls', type=Path, default=ROOT/'validation/evidence/shared-physical-provider-probe.json', help='separate raw/oracle/empty-batch report; its exact source revisions remain explicit')
    add_arguments(parser)
    args = parser.parse_args()
    try:
        validate_arguments(args)
    except ValueError as error:
        parser.error(str(error))
    if args.scopes is None:
        args.scopes = '1' if args.timing else '0,1,2,3'
    if args.cpu is not None and args.cpu not in os.sched_getaffinity(0):
        parser.error('chosen CPU is outside permitted affinity')
    if args.timing and args.cpu is None:
        parser.error('explicit timing requires a selected CPU; host/sibling isolation is still reviewed separately')
    if args.rounds < 1 or args.trials < 4 or args.trials % 2:
        parser.error('positive rounds and >=4 balanced trials required')
    if args.timing and args.rounds % 3:
        parser.error('timing rounds must contain complete cycles of the three phase-space points')
    for key, expected in [('OPT_LEVEL', '3'), ('LTO', 'thin'), ('CODEGEN_UNITS', '1')]:
        actual = os.environ.get('CARGO_PROFILE_RELEASE_' + key)
        if actual is not None and actual != expected:
            parser.error('release profile override conflicts with main workspace: ' + key)
    for env_name in ['RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS', 'CFLAGS', 'CXXFLAGS']:
        if any(flag in os.environ.get(env_name, '') for flag in ['fast-math', 'unsafe-fp-math', 'finite-math-only']):
            parser.error(f'unsupported relaxed floating-point option in {env_name}')
    for key in ['RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS']:
        flags = os.environ.get(key, '').replace('\x1f', ' ')
        if re.search(r'(?:-C\s*|--codegen(?:=|\s+))(?:opt-level|lto|codegen-units)(?:=|\s)', flags):
            parser.error('codegen profile overrides must be removed for this matched-profile runner')
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
    if args.prune_swapped_arithmetic:
        validate_pruning_preflight(controls, args.reduction_kernels, native_enabled(args), before,
                                  required_widths=[int(value) for value in args.widths.split(',')],
                                  native_block_rows=args.native_block_rows)
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

    command = ['cargo', 'build', '--manifest-path', HERE/'Cargo.toml', '--locked', '-j4', '--message-format=json']
    if native_enabled(args):
        command += ["--features", "native-kernels"]
    if args.timing or args.release:
        command.append('--release')
    if args.allocation_diagnostics:
        command += ['--features', 'allocation-diagnostics']
    if args.execution_diagnostics:
        command += ['--features', 'execution-diagnostics']
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
    if args.execution_diagnostics:
        command.insert(1, '-DSP_EXECUTION_DIAGNOSTICS')
    run('cpp-build', command)
    actual_mode = 'timing' if args.timing else 'allocations' if args.allocation_diagnostics else 'execution' if args.execution_diagnostics else 'check'
    host_before = host_snapshot()
    workload_command = [binary, ROOT/'examples/graphs/physical/gg-hhh/top-pentagon-26.dot', actual_mode, args.rounds, args.trials, args.widths, args.scopes, args.reduction_kernels]
    if args.native_block_rows is not None:
        workload_command.append(args.native_block_rows)
    if args.prune_swapped_arithmetic:
        workload_command.append('--prune-swapped-arithmetic')
    stdout = run('physical-workloads', workload_command, input_text, affinity=True)
    host_after = host_snapshot()
    results = [json.loads(line) for line in stdout.splitlines() if line.startswith('{')]
    if len(results) != 1 or not results[0]['success']:
        raise RuntimeError('missing successful workload report')
    result = results[0]
    validate_report(result, args.reduction_kernels, native_enabled(args), native_block_rows=args.native_block_rows,
                    prune_swapped_arithmetic=args.prune_swapped_arithmetic)
    if args.execution_diagnostics:
        for item in result['records']:
            item['validated_execution_breakdown'] = validate_execution_record(item)
    if args.timing:
        for item in result['records']:
            item['summary'] = summarize_record(item)
            if len(item['raw_block_nanoseconds']) != len(item['samples']):
                raise RuntimeError('raw contiguous block times are missing')
            for pair, raw in zip(item['samples'], item['raw_block_nanoseconds']):
                if len(raw) != 2 or raw[0] / item['rounds'] != pair['original_ns'] or raw[1] / item['rounds'] != pair['rust_ns']:
                    raise RuntimeError('raw block and normalized timing disagree')
            item['minimum_observed_block_ns'] = min(min(raw) for raw in item['raw_block_nanoseconds'])
            item['block_duration_below_20ms'] = item['minimum_observed_block_ns'] < 20_000_000
    after = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    result['unchanged_runtime_sources'] = before == after
    result['success'] &= before == after
    result['host_before'], result['host_after'] = host_before, host_after
    result['selected_cpu'] = args.cpu
    result['selected_scopes'] = [int(value) for value in args.scopes.split(',')]
    result['rust_profile'] = 'release' if args.timing or args.release else 'dev'
    result['rust_release_settings'] = {'opt_level': 3, 'lto': 'thin', 'codegen_units': 1, 'matches_main_workspace_profile': True}
    result['measurement_generation'] = 'contiguous blocks, one retained reducer, prebound coefficient states; distinct from dba7411 baseline'
    result['sampling_boundary'] = 'One clock pair around all rounds, with provider counters and checksum reads outside. Rounds contain full three-point cycles.'
    result['shared_reduction_boundary'] = 'Three immutable prepared numerator/input states, one retained reducer/session; provider pointer selection, numerical geometry/reduction, common numerator calls, frozen lookup/contraction and ordinary staging included. No coefficient preparation or backend arithmetic.'
    result['execution_diagnostics'] = {'enabled': args.execution_diagnostics, 'observed_phase_index': 1, 'calls_per_implementation_record': 1,
        'unit': 'integer nanoseconds from monotonic clocks, calls and evaluated rows',
        'nesting': 'Shared backend execution is nested in shared ABI spans; Rust preparation provider is nested in full C++ preparation. Additive accounting subtracts each nested span once; numerator and full preparation are disjoint from masters.',
        'master_backend_scope': 'Actual provider execution, including its ordinary validation/conversion and expression arithmetic; no result-cache hits. Original rank2/3 bubbles are grouped calls, not fabricated individual family timings. Scale initialization is separate and retained in remainder.',
        'master_lookup_transport_scope': 'Rust shared-master ABI, decoding, exact-key lookup, output copying and validation, excluding its nested backend. Caller-side C++/Rust packing remains in remainder.',
        'remainder_scope': 'Core reduction, contraction, caller-side staging/transport, unassigned initialization and instrumentation overhead; not a pure arithmetic measurement.',
        'cross_run_subtraction': False, 'aggregate_timing_instrumented': False}
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
        'pruning_configuration_and_parity_validated': args.prune_swapped_arithmetic,
        'integrated_widths': controls['integrated_widths'],
        'preparation_widths': controls['preparation_widths'],
        'raw_components_checked': controls['raw_components_checked'],
        'raw_max_scaled_error': controls['raw_max_scaled_error'],
        'coefficient_oracle_max_scaled_error': controls['coefficient_oracle_max_scaled_error'],
        'precision_oracle_boundary': controls.get('precision_oracle_boundary'),
    }
    result['numerator_provider'] = {'shared_out_of_line_abi': True, 'coefficient_heads': 'Factored multi-output SymJIT O2, evaluated only in physical preparation', 'reduction_callbacks': 'Same PolynomialNumerator<f64> and generic Symbolica batched interpreter for direct/Mu/T3/T2; callbacks are not JIT compiled', 'execution_span_scope': 'Individual direct/Mu/T3/T2 provider ABI decoding, interpreted evaluation and output writing; calls/rows/ns reported separately'}
    result['common_provider_ffi_included'] = True
    result['production_complete_is_unmodified_cli_latency'] = False
    result['source_sha256'] = before
    result['artifacts_sha256'] = {str(p): digest(p) for p in [library, binary, ninja, olo, fortran, out/'input.txt']}
    result['reference_build_configuration'] = config
    result['dependency_manifest'] = json.loads((ROOT/'dependencies/manifest.json').read_text())
    result['dependency_heads'] = {name: subprocess.check_output(['git', '-C', ROOT/'references'/name, 'rev-parse', 'HEAD'], text=True).strip() for name in ['symbolica', 'oneloopmaster', 'gammaloop_ninja_shared']}
    result['compiler_environment'] = {key: os.environ.get(key) for key in ['RUSTFLAGS', 'CFLAGS', 'CXXFLAGS', 'CARGO_ENCODED_RUSTFLAGS', 'CARGO_PROFILE_RELEASE_OPT_LEVEL', 'CARGO_PROFILE_RELEASE_LTO', 'CARGO_PROFILE_RELEASE_CODEGEN_UNITS']}
    result['compiler_versions'] = {'rustc': subprocess.check_output(['rustc', '-Vv'], text=True), 'cxx': subprocess.check_output(['g++', '--version'], text=True)}
    result['commands'] = commands
    result['logs'] = {p.name: {'sha256': digest(p), 'text': p.read_text()} for p in out.glob('*.stderr')}
    (out/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(out/'result.json')
    if not result['success']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
