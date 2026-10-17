#!/usr/bin/env python3
"""Pure Python protocol regressions; every timing/counter fixture below is FAKE.

No benchmark executable, compiler, physical preflight, or numerical kernel runs.
The invented durations and union size must never be used as performance evidence.
"""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from collections import Counter, defaultdict

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('sampling_driver',
                                             '/tmp/ninjars-rank-block-sampling.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
sys.path.insert(0, str(driver.HERE))
from kernel_selection import FAMILIES

FAKE_UNION_ENTRIES = 17


def fixture(setting=('native', None), width=1):
    """Hand-constructed fake wire report matching the current C++ schema."""
    mode, bound = setting
    rounds = driver.ROUNDS[width]
    job = {'sweep': 0, 'position': 0, 'width': width, 'rounds': rounds,
           'trials': 8, 'selection': mode, 'native_block_rows': bound,
           'setting': driver.label(setting)}
    coverage = {
        'selection': mode, 'native_feature_enabled': True,
        'topology': {'propagators': 5, 'rank': 4},
        'families': {name: 'native-rust' if i < 5 else 'symbolica-interpreter'
                     for i, name in enumerate(FAMILIES)},
        'native_execution': {'policy': 'whole' if bound is None else 'fixed-rows',
                             'block_rows': bound, 'scope': 'native-kernel-rows'}}
    if mode == 'native-rank':
        coverage['conditional_corrections'] = {
            'backend': 'native-rust', 'guard': 'all-active-source-kind-structural-zeros',
            'fallback': 'reported-generic-families'}
    report = {'success': True, 'reduction_kernels': coverage, 'mode': 'timing',
              'clock_pair_median_ns': 31.0, 'performance_acceptance': False,
              'original_production_backend_counters_available': False,
              'counter_columns': list(driver.COUNTERS), 'records': [],
              'frozen_backend_call_delta': 0, 'frozen_backend_row_delta': 0,
              'checksum': 0.125, 'union_entries': FAKE_UNION_ENTRIES,
              'union_replay_rows': 100, 'union_misses': 0}
    for workload in ('repeated', 'changing'):
        samples, blocks = [], []
        for trial in range(8):
            # Fake times deliberately above the floor, exactly normalized.
            raw = [rounds * (30_000_000 + trial * 100),
                   rounds * (40_000_000 + trial * 200)]
            blocks.append(raw)
            samples.append({'original_ns': raw[0] / rounds,
                            'rust_ns': raw[1] / rounds,
                            'order': 'rust-first' if trial % 2 else 'original-first'})
        original = [10*width, 0, 25*width, 20*width,
                    10*width, 0, 25*width, 20*width] + [0]*12
        rust = [5, 0, 15, 10, 10*width, 0, 25*width, 20*width,
                0, 0, 0, 30, 55*width, 5, 50*width, 0, 0, 0, 0, 0]
        report['records'].append({
            'scope': 'shared-reduction', 'workload': workload, 'width': width,
            'rounds': rounds, 'contiguous_blocks': True, 'counters_outside_blocks': True,
            'retained_reducer_sessions': 1, 'prebound_numerator_states': 3,
            'correctness_components': 9*width, 'max_scaled_error': 1e-16,
            'samples': samples, 'raw_block_columns': ['original_ns', 'rust_ns'],
            'raw_block_nanoseconds': blocks,
            'original_counter_deltas': [v * rounds * 8 for v in original],
            'rust_counter_deltas': [v * rounds * 8 for v in rust]})
    return report, job


class ProtocolTests(unittest.TestCase):
    def reject(self, mutation, setting=('native', None)):
        report, job = fixture(setting)
        mutation(report, job)
        with self.assertRaises((RuntimeError, ValueError, TypeError, KeyError)):
            driver.validate_timing(report, job, FAKE_UNION_ENTRIES)

    def test_all_four_settings_all_six_widths(self):
        for setting in driver.SETTINGS:
            for width in driver.WIDTHS:
                with self.subTest(setting=setting, width=width):
                    report, job = fixture(setting, width)
                    output = driver.validate_timing(report, job, FAKE_UNION_ENTRIES)
                    self.assertEqual(len(output['records']), 2)
                    self.assertFalse(output['records'][0]['block_duration_below_20ms'])

    def test_wrong_mode_and_selection(self):
        self.reject(lambda r, j: r.update(mode='check'))
        self.reject(lambda r, j: r['reduction_kernels'].update(selection='native-rank'))

    def test_wrong_policy_bound_and_rank_coverage(self):
        self.reject(lambda r, j: r['reduction_kernels']['native_execution'].update(
            policy='fixed-rows', block_rows=32))
        self.reject(lambda r, j: r['reduction_kernels']['native_execution'].update(
            block_rows=7), ('native', 32))
        self.reject(lambda r, j: r['reduction_kernels'].pop('conditional_corrections'),
                    ('native-rank', 32))

    def test_callback_counter_changes(self):
        for name in ('original_counter_deltas', 'rust_counter_deltas'):
            for index in (0, 1, 3, 4, 7):
                with self.subTest(name=name, index=index):
                    self.reject(lambda r, j, n=name, i=index:
                                r['records'][0][n].__setitem__(i, r['records'][0][n][i]+1))

    def test_preparation_or_actual_backend_counter(self):
        for name in ('original_counter_deltas', 'rust_counter_deltas'):
            for index in (8, 9, 10, 15, 16, 17, 18, 19):
                with self.subTest(name=name, index=index):
                    self.reject(lambda r, j, n=name, i=index:
                                r['records'][0][n].__setitem__(i, 1))
        for field in ('frozen_backend_call_delta', 'frozen_backend_row_delta', 'union_misses'):
            self.reject(lambda r, j, f=field: r.update({f: 1}))

    def test_union_control_count_and_boolean_rejected(self):
        for value in (0, 18, True, 17.0):
            self.reject(lambda r, j, v=value: r.update(union_entries=v))
        self.reject(lambda r, j: r.update(union_replay_rows=0))

    def test_schema_and_instrumented_records(self):
        self.reject(lambda r, j: r['counter_columns'].reverse())
        self.reject(lambda r, j: r['records'].pop())
        self.reject(lambda r, j: r['records'][0].update(width=True))
        self.reject(lambda r, j: r['records'][0].update(scope='production-complete'))
        self.reject(lambda r, j: r['records'][0].update(retained_reducer_sessions=True))
        for field in ('execution_records', 'allocation_records', 'allocation_counts',
                      'allocation_scopes', 'rust_allocator_only'):
            self.reject(lambda r, j, f=field: r['records'][0].update({f: []}))
        self.reject(lambda r, j: r.update(execution_records=[]))

    def test_order_block_count_and_normalization(self):
        self.reject(lambda r, j: r['records'][0]['samples'][0].update(order='rust-first'))
        self.reject(lambda r, j: r['records'][0]['samples'].pop())
        self.reject(lambda r, j: r['records'][0]['raw_block_nanoseconds'].pop())
        self.reject(lambda r, j: r['records'][0]['raw_block_nanoseconds'][0].__setitem__(0, 1))
        self.reject(lambda r, j: r['records'][0]['raw_block_columns'].reverse())

    def test_nonfinite_and_wrong_type_numeric_fields(self):
        for value in (float('nan'), float('inf'), float('-inf'), True, '1', None):
            self.reject(lambda r, j, v=value: r.update(checksum=v))
            self.reject(lambda r, j, v=value: r['records'][0].update(max_scaled_error=v))
        self.reject(lambda r, j: r['records'][0].update(max_scaled_error=1.1e-9))
        self.reject(lambda r, j: r['records'][0]['raw_block_nanoseconds'][0].__setitem__(0, True))

    def test_strict_json_rejects_constants(self):
        for value in ('NaN', 'Infinity', '-Infinity'):
            with self.assertRaises(ValueError):
                driver.strict_json('{"unrelated_field": ' + value + '}')
        self.assertEqual(driver.strict_json('{"value": 1}'), {'value': 1})

    def test_short_block_is_retained_and_flagged(self):
        report, job = fixture()
        row = report['records'][0]
        row['raw_block_nanoseconds'][0][0] = 19_999_999
        row['samples'][0]['original_ns'] = 19_999_999/job['rounds']
        before = copy.deepcopy(row['raw_block_nanoseconds'])
        output = driver.validate_timing(report, job, FAKE_UNION_ENTRIES)
        self.assertTrue(output['records'][0]['block_duration_below_20ms'])
        self.assertEqual(output['records'][0]['raw_block_nanoseconds'], before)

    def test_schedule_is_complete_deterministic_and_position_balanced(self):
        jobs = driver.schedule()
        self.assertEqual(jobs, driver.schedule())
        self.assertEqual(len(jobs), 96)
        per_sweep_width = defaultdict(set)
        positions = defaultdict(Counter)
        for job in jobs:
            per_sweep_width[job['sweep'], job['width']].add(job['setting'])
            positions[job['width'], job['setting']][job['position']] += 1
            self.assertEqual(job['rounds'], {1:384, 8:48, 31:15, 32:12, 128:3, 1024:3}[job['width']])
            self.assertEqual(job['rounds'] % 3, 0)
            self.assertEqual(job['trials'], 8)
        expected = {driver.label(setting) for setting in driver.SETTINGS}
        self.assertEqual(len(per_sweep_width), 24)
        self.assertTrue(all(value == expected for value in per_sweep_width.values()))
        self.assertEqual(len(positions), 24)
        self.assertTrue(all(count == Counter({0:1, 1:1, 2:1, 3:1})
                            for count in positions.values()))


if __name__ == '__main__':
    print('FAKE fixtures only: pure Python validation, no numerical executable or build.')
    unittest.main(verbosity=2)
