#!/usr/bin/env python3
"""Check the persistent uncached runner; short timings are protocol controls only."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


def parse_pool(path):
    tokens = iter(path.read_text().split())
    assert next(tokens) == 'NINJARS_BENCH_V1'
    n, rank, parameters, rows = (int(next(tokens)) for _ in range(4))
    samples = []
    for _ in range(rows):
        scale, matrix = float(next(tokens)), int(next(tokens))
        shifts = [[float(next(tokens)) for _ in range(6)] for _ in range(n)]
        values = [[float(next(tokens)), float(next(tokens))] for _ in range(parameters)]
        matrix = [float(next(tokens)) for _ in range(n*n)] if matrix else None
        samples.append({'scale': scale, 'propagators': shifts, 'parameters': values, 'matrix': matrix})
    assert next(tokens, None) is None, 'pool.inputs must stop before the job count'
    return {'n': n, 'rank': rank, 'parameters': parameters, 'samples': samples}


def row_text(row):
    lines = [f'{row["scale"]:.17e} {int(row["matrix"] is not None)}']
    lines += [' '.join(format(value, '.17e') for value in propagator) for propagator in row['propagators']]
    lines += [' '.join(format(value, '.17e') for value in parameter) for parameter in row['parameters']]
    if row['matrix'] is not None:
        lines.append(' '.join(format(value, '.17e') for value in row['matrix']))
    return '\n'.join(lines)


def protocol(pool, jobs=None):
    if jobs is None:
        jobs = [(7, 0, 5, 1, 2, 2), (19, 1, 7, 2, 3, 1)]
    return ('NINJARS_BENCH_V1\n'
            f'{pool["n"]} {pool["rank"]} {pool["parameters"]} {len(pool["samples"])}\n'
            + '\n'.join(row_text(row) for row in pool['samples'])
            + f'\n{len(jobs)}\n'
            + ''.join(' '.join(map(str, job))+'\n' for job in jobs))


def json_result(result):
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
    assert len(rows) == 1, (result.stdout, result.stderr)
    return rows[0]


def result_checksum(rows):
    value = 14695981039346656037
    data = struct.pack('<Q', len(rows))
    for row in rows:
        data += struct.pack('<QQ', row['lane'], row['status'])
        for pair in [*row['epsilon'], row['cut_constructible'], row['rational']]:
            data += struct.pack('<dd', *pair)
    for byte in data:
        value = ((value ^ byte)*1099511628211) & ((1 << 64)-1)
    return format(value, '016x')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True,
                        help='common-template export with numerator.hh and pool.inputs')
    parser.add_argument('--build-dir', type=Path, default=ROOT/'target/reference-uncached-v2')
    parser.add_argument('--cached-build-dir', type=Path, default=ROOT/'target/reference')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--no-build', action='store_true')
    parser.add_argument('--ninjars', type=Path, help='copied NinjaRS executable for independent Laurent parity')
    parser.add_argument('--fixture', type=Path, help='fixture used to export this pool')
    args = parser.parse_args()
    if bool(args.ninjars) != bool(args.fixture):
        parser.error('--ninjars and --fixture must be supplied together')
    bundle, build = args.bundle.resolve(), args.build_dir.resolve()
    executable, control = build/'ninja-benchmark', build/'ninja-benchmark-streamed-control'
    if not args.no_build:
        for extra in [['--benchmark', '--output', executable], ['--output', control]]:
            subprocess.run([sys.executable, ROOT/'validation/reference/build.py', '--runner-only',
                            '--disable-avholo-cache', '--build-dir', build,
                            '--numerator-header', bundle/'numerator.hh', *extra], check=True, cwd=ROOT)
    for binary in [executable, control]:
        recorded = json.loads(binary.with_suffix('.json').read_text())
        if (recorded['generated_numerator_sha256'] != hashlib.sha256((bundle/'numerator.hh').read_bytes()).hexdigest()
                or recorded['executable_sha256'] != hashlib.sha256(binary.read_bytes()).hexdigest()):
            parser.error('runner/header provenance mismatch; rebuild without --no-build')
    pool = parse_pool(bundle/'pool.inputs')
    source_rows = len(pool['samples'])
    singleton_extension = len(pool['samples']) == 1
    if singleton_extension:
        # Protocol-only changing-scale control; never represent this as a new
        # phase-space point in a performance report.
        second = copy.deepcopy(pool['samples'][0])
        second['scale'] = 2*second['scale']
        pool['samples'].append(second)
    evidence = {'schema_version': 1, 'performance_acceptance': False,
                'purpose': 'protocol and correctness controls; short loop timings are not benchmark evidence',
                'binary_sha256': hashlib.sha256(executable.read_bytes()).hexdigest(),
                'build_metadata': json.loads(executable.with_suffix('.json').read_text()),
                'source_sha256': {name: hashlib.sha256((ROOT/'validation/reference'/name).read_bytes()).hexdigest()
                                  for name in ['benchmark_runner.cc', 'build.py', 'test_benchmark_runner.py']},
                'bundle_sha256': {name: hashlib.sha256((bundle/name).read_bytes()).hexdigest()
                                  for name in ['numerator.hh', 'pool.inputs', 'manifest.json']},
                'singleton_pool_extended_with_second_scale': singleton_extension,
                'checks': {}}
    if args.ninjars:
        evidence['ninjars_sha256'] = hashlib.sha256(args.ninjars.read_bytes()).hexdigest()
        evidence['fixture_sha256'] = hashlib.sha256(args.fixture.read_bytes()).hexdigest()

    def invoke(input_text, flags=()):
        return subprocess.run([executable, *flags], input=input_text, capture_output=True, text=True)

    def streamed(row, flags=()):
        request = f'{pool["n"]} {pool["rank"]} 1 {row["scale"]:.17e}\n'
        request += '\n'.join(' '.join(format(value, '.17e') for value in values)
                             for values in [*row['propagators'], *row['parameters']])+'\n'
        with tempfile.TemporaryDirectory(prefix='ninjars-reference-matrix-') as directory:
            command = [control, *flags]
            if row['matrix'] is not None:
                path = Path(directory)/'s-matrix'
                path.write_text(str(pool['n'])+'\n'+' '.join(map(str, row['matrix']))+'\n')
                command += ['--s-matrix', path]
            result = subprocess.run(command, input=request, capture_output=True, text=True, check=True)
            return json_result(result)

    class PersistentProtocol(unittest.TestCase):
        def compare(self, report, samples, flags=()):
            self.assertTrue(report['success'])
            self.assertTrue(report['pre_post_equal'])
            self.assertEqual(report['pre_checksum'], result_checksum(report['pre_results']))
            self.assertEqual(report['post_checksum'], result_checksum(report['post_results']))
            self.assertEqual(report['pre_results'], report['post_results'])
            self.assertEqual(len(report['pre_results']), len(samples))
            for actual, sample in zip(report['pre_results'], samples):
                expected = streamed(sample, flags)
                self.assertEqual(actual['status'], 0)
                self.assertEqual(expected['status'], 0)
                for key in ['epsilon', 'rational', 'cut_constructible']:
                    self.assertEqual(actual[key], expected[key], (key, actual, expected))
            for job in report['jobs']:
                self.assertEqual(job['warmup_final_pass_failed_rows'], 0)
                self.assertEqual(len(job['round_ns']), job['rounds'])
                self.assertTrue(all(math.isfinite(t) and t > 0 for t in job['round_ns']))

        def test_repeated_changing_and_control_flags(self):
            for flags in [[], ['--mu-sampling'], ['--cut-stop', '2']]:
                with self.subTest(flags=flags):
                    result = invoke(protocol(pool), flags)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    report = json_result(result)
                    self.compare(report, pool['samples'], flags)
                    self.assertEqual([job['job_id'] for job in report['jobs']], [7, 19])
                    self.assertEqual([job['timed_points_per_round'] for job in report['jobs']], [10, 7])
                    evidence['checks']['controls '+repr(flags)] = report

        def test_rebinding_permutation_and_single_row(self):
            varied = copy.deepcopy(pool)
            # Include duplicate and changed parameter rows, separated by other
            # kinematics, to expose stale callback buffers and pointer lifetimes.
            extra = copy.deepcopy(varied['samples'][0])
            extra['parameters'] = [[0., 0.] for _ in extra['parameters']]
            complex_row = copy.deepcopy(extra)
            if complex_row['parameters']:
                complex_row['parameters'][0] = [1., 2.]
            varied['samples'] = [extra, complex_row, *reversed(varied['samples']), copy.deepcopy(varied['samples'][0])]
            result = invoke(protocol(varied))
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json_result(result)
            self.compare(report, varied['samples'])
            evidence['checks']['rebind_permuted'] = report
            varied['samples'] = varied['samples'][-1:]
            result = invoke(protocol(varied, [(0, 0, 1, 1, 2, 1)]))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.compare(json_result(result), varied['samples'])

        def test_malformed_shapes_jobs_and_trailing_input(self):
            baseline = protocol(pool)
            invalid = ['', 'NINJARS_BENCH_V2\n', baseline+'unexpected\n',
                       baseline.rsplit('\n', 3)[0], protocol(pool, []),
                       protocol(pool, [(0, 0, 0, 1, 1, 1)]),
                       protocol(pool, [(0, 0, 1, 0, 1, 1)]),
                       protocol(pool, [(0, 0, 1, 1, 0, 1)]),
                       protocol(pool, [(0, 0, 1, 1, 1, 0)]),
                       protocol(pool, [(0, 2, 1, 1, 1, 1)]),
                       protocol(pool, [(0, 0, 1, 1, 1, 1)]*2),
                       protocol(pool, [(0, 0, 1 << 63, 1, 1, 3)]),
                       baseline.replace('NINJARS_BENCH_V1\n', 'NINJARS_BENCH_V1\n-1 ', 1)]
            for field in ['n', 'rank', 'parameters']:
                changed = copy.deepcopy(pool)
                changed[field] += 1
                invalid.append(protocol(changed))
            changed = copy.deepcopy(pool)
            changed['samples'] = []
            invalid.append(protocol(changed))
            changed = copy.deepcopy(pool)
            changed['samples'] = changed['samples'][:1]
            invalid.append(protocol(changed))
            for scale in [0., -1., float('nan'), float('inf')]:
                changed = copy.deepcopy(pool)
                changed['samples'][0]['scale'] = scale
                invalid.append(protocol(changed))
            changed = copy.deepcopy(pool)
            changed['samples'][0]['propagators'][0][0] = float('inf')
            invalid.append(protocol(changed))
            for text in invalid:
                result = invoke(text)
                self.assertEqual(result.returncode, 2, (text, result.stdout, result.stderr))
                self.assertFalse(any(line.startswith('{') for line in result.stdout.splitlines()))
            for flags in [['--full-box-coefficients'], ['--cut-stop', '7'], ['--cut-stop', '-1'], ['--mu-sampling-scale', '2']]:
                self.assertEqual(invoke(baseline, flags).returncode, 2)
            evidence['checks']['malformed_inputs_rejected'] = len(invalid)+4

        def test_s_matrix_validation_and_reset(self):
            varied = copy.deepcopy(pool)
            for row in varied['samples']:
                # An explicit S-matrix override exercises pointer replacement;
                # the fresh reference receives the same supplied matrix.
                row['matrix'] = [0. if i == j else -1.
                                 for i in range(pool['n']) for j in range(pool['n'])]
            # Derived matrices and explicitly supplied matrices alternate, so a
            # previous external pointer cannot leak into the following sample.
            varied['samples'] += copy.deepcopy(pool['samples'])
            result = invoke(protocol(varied))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.compare(json_result(result), varied['samples'])
            varied['samples'][0]['matrix'][0] = 1.
            self.assertEqual(invoke(protocol(varied)).returncode, 2)
            varied['samples'][0]['matrix'][0] = 0.
            varied['samples'][0]['matrix'][1] += 1.
            self.assertEqual(invoke(protocol(varied)).returncode, 2)

        def test_cache_enabled_library_is_rejected(self):
            config = args.cached_build_dir/'ninja-build/config.h'
            if not config.exists() or '#define NINJA_USE_ONELOOP_WITH_CACHE 1' not in config.read_text():
                self.skipTest('no cached reference build available')
            result = subprocess.run([sys.executable, ROOT/'validation/reference/build.py', '--runner-only',
                                     '--benchmark', '--build-dir', args.cached_build_dir,
                                     '--numerator-header', bundle/'numerator.hh'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('master-result cache enabled', result.stderr)
            evidence['checks']['cached_build_rejected'] = True

        def test_changed_library_provenance_is_rejected(self):
            with tempfile.TemporaryDirectory(prefix='ninjars-reference-provenance-') as directory:
                copied = Path(directory)
                for relative in ['ninja-build/src/.libs/libninja.a', 'oneloop/libavh_olo.a', 'ninja-build/config.h']:
                    target = copied/relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.symlink_to(build/relative)
                revisions = json.loads((build/'revisions.json').read_text())
                revisions['libraries_sha256']['libninja.a'] = '0'*64
                (copied/'revisions.json').write_text(json.dumps(revisions))
                result = subprocess.run([sys.executable, ROOT/'validation/reference/build.py', '--runner-only',
                                         '--benchmark', '--build-dir', copied,
                                         '--numerator-header', bundle/'numerator.hh'], capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn('provenance/configuration is missing or changed', result.stderr)

        def test_copied_source_and_generated_header_changes_are_rejected(self):
            with tempfile.TemporaryDirectory(prefix='ninjars-reference-inputs-') as directory:
                copied = Path(directory)
                revisions = json.loads((build/'revisions.json').read_text())
                for relative in [*revisions['input_files_sha256'], 'ninja-build/src/.libs/libninja.a',
                                 'oneloop/libavh_olo.a']:
                    target = copied/relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.symlink_to(build/relative)
                (copied/'revisions.json').write_text(json.dumps(revisions))
                command = [sys.executable, ROOT/'validation/reference/build.py', '--runner-only',
                           '--benchmark', '--build-dir', copied, '--numerator-header', bundle/'numerator.hh']
                for relative in ['ninja-source/src/ninja/types.hh', 'ninja-build/src/ninja/ninja_config.h']:
                    with self.subTest(relative=relative):
                        target = copied/relative
                        original = target.read_text()
                        target.unlink()
                        target.write_text(original+'\n// controlled mutation\n')
                        result = subprocess.run(command, capture_output=True, text=True)
                        self.assertEqual(result.returncode, 2)
                        self.assertIn('sources/generated headers provenance is missing or changed', result.stderr)
                        target.unlink()
                        target.symlink_to(build/relative)
                del revisions['input_files_sha256']
                (copied/'revisions.json').write_text(json.dumps(revisions))
                result = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn('sources/generated headers provenance is missing or changed', result.stderr)
                evidence['checks']['changed_source_generated_header_and_legacy_record_rejected'] = True

        @unittest.skipUnless(pool['n'] == 3 and pool['rank'] >= 2, 'triangle exceptional-point control')
        def test_nonfinite_reference_rows_suppress_timing(self):
            exceptional = copy.deepcopy(pool)
            row = exceptional['samples'][-1]
            row['propagators'] = [[0., 0., 0., 0., 1., 0.], [0., 1., 0., 0., 1., 0.],
                                  [0., 0., 1., 0., 1., 0.]]
            row['matrix'] = None
            result = invoke(protocol(exceptional))
            self.assertEqual(result.returncode, 1, result.stderr)
            report = json_result(result)
            self.assertFalse(report['success'])
            self.assertFalse(report['pre_results'][-1]['finite'])
            self.assertEqual(report['jobs'], [])
            self.assertEqual(report['post_results'], [])
            evidence['checks']['nonfinite_reference_suppresses_timing'] = report

        @unittest.skipUnless(args.ninjars, 'no copied NinjaRS binary/fixture requested')
        def test_full_laurents_match_rust_and_partial_cut_diagnostics(self):
            checks = []
            for flags, rust_flags in [([], []), (['--mu-sampling'], ['--mu-extraction', 'sampling']),
                                      (['--cut-stop', '2'], ['--cut-stop', '2'])]:
                report = json_result(invoke(protocol(pool), flags))
                command = [str(args.ninjars.resolve()), 'reduce', str(args.fixture.resolve()),
                           '--precision', 'f64', '--batch-size', '2', *rust_flags]
                result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
                reduced = json.loads(result.stdout)
                self.assertEqual(len(reduced['results']), source_rows)
                full_reduction = '--cut-stop' not in flags
                for row, original in zip(reduced['results'], report['pre_results']):
                    self.assertEqual(row['status']['kind'], 'success')
                    errors = []
                    for value, target in zip(row['values'][:3], original['epsilon']):
                        actual, expected = complex(*map(float, value)), complex(*target)
                        error = abs(actual-expected)/max(1., abs(expected))
                        self.assertTrue(math.isfinite(error))
                        if full_reduction:
                            self.assertLessEqual(error, 1e-9,
                                                 (command, row['lane'], error, actual, expected))
                        errors.append(error)
                    checks.append({'command': command, 'lane': row['lane'],
                                   'laurent_scaled_errors': errors, 'rust_values': row['values'],
                                   'original': original, 'passed': max(errors) <= 1e-9,
                                   'correctness_gate': full_reduction,
                                   'scope': 'full reduction' if full_reduction else 'partial-cut diagnostic; mismatch origin unresolved'})
            # Keep decompositions visible, but do not equate finite-part splits
            # belonging to different reduction/master bases.
            evidence['checks']['rust_laurents'] = checks

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PersistentProtocol))
    evidence['passed'] = result.wasSuccessful()
    evidence['tests_run'] = result.testsRun
    output = args.output or build/'benchmark-protocol-validation.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2)+'\n')
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
