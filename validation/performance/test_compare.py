#!/usr/bin/env python3
"""Fast failure-path controls; no compiler, license or timing run required."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('performance_compare', Path(__file__).with_name('compare.py'))
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.binary = self.root/'ninjars'
        self.binary.write_bytes(b'compiled executable')
        self.sources = {'Cargo.toml': 'manifest', 'Cargo.lock': 'lock',
                        'crates/ninjars/src/lib.rs': 'source', 'dependencies/manifest.json': 'dependencies'}
        self.dependencies = {'symbolica': {'head': 'revision', 'tracked_status': 'M file.rs',
                                           'tracked_diff_sha256': 'patch', 'bundled_patches': []}}
        self.configuration = {'/workspace/.cargo/config.toml': {'sha256': 'configuration',
                                                               'build_rustflags': None, 'target_rustflags': {}}}
        self.record = {'binary_sha256': compare.digest(self.binary), 'command': ['cargo', 'build', '--release'],
                       'environment': {}, 'checkout': {'source_sha256': copy.deepcopy(self.sources)},
                       'dependencies': copy.deepcopy(self.dependencies),
                       'cargo_configuration': copy.deepcopy(self.configuration)}
        self.enterContext(patch.object(compare.build_provenance, 'production_snapshot',
                                       side_effect=lambda: {'source_sha256': self.sources}))
        self.enterContext(patch.object(compare.build_provenance.checkpoint, 'dependencies',
                                       side_effect=lambda: self.dependencies))
        self.enterContext(patch.object(compare.build_provenance, 'cargo_configuration',
                                       side_effect=lambda: self.configuration))

    def test_current_and_legacy_records_validate_same_production_inputs(self):
        current = compare.validate_build_record(self.record, self.binary)
        self.record['checkout']['source_sha256']['validation/performance/compare.py'] = 'old harness'
        self.record['checkout']['source_sha256']['examples/graphs/bubble.json'] = 'old fixture'
        self.assertEqual(compare.validate_build_record(self.record, self.binary), current)

    def test_changed_added_and_removed_production_files_are_rejected(self):
        for operation in ['change', 'add', 'remove']:
            with self.subTest(operation=operation):
                saved = self.sources.copy()
                if operation == 'change':
                    self.sources['crates/ninjars/src/lib.rs'] = 'changed source'
                elif operation == 'add':
                    self.sources['crates/ninjars/src/new.rs'] = 'new source'
                else:
                    del self.sources['Cargo.lock']
                with self.assertRaisesRegex(ValueError, 'production sources differ'):
                    compare.validate_build_record(self.record, self.binary)
                self.sources = saved

    def test_actual_dependency_revision_and_tracked_changes_are_rejected(self):
        for key in ['head', 'tracked_status', 'tracked_diff_sha256']:
            with self.subTest(key=key):
                saved = copy.deepcopy(self.dependencies)
                self.dependencies['symbolica'][key] = 'changed'
                with self.assertRaisesRegex(ValueError, 'actual vendored repositories'):
                    compare.validate_build_record(self.record, self.binary)
                self.dependencies = saved

    def test_configuration_added_changed_and_removed_are_rejected(self):
        for configuration in [{}, {'/workspace/.cargo/config.toml': {'sha256': 'modified'}},
                              {**self.configuration, '/home/test/.cargo/config': {'sha256': 'new'}}]:
            with self.subTest(configuration=configuration):
                with patch.object(compare.build_provenance, 'cargo_configuration', return_value=configuration):
                    with self.assertRaisesRegex(ValueError, 'Cargo configuration differs'):
                        compare.validate_build_record(self.record, self.binary)

    def test_binary_profile_and_fast_math_flags_are_rejected(self):
        invalid = [('binary_sha256', 'wrong', 'supplied executable'),
                   ('command', ['cargo', 'build'], 'optimized Rust build'),
                   ('environment', {'RUSTFLAGS': '-C llvm-args=-enable-unsafe-fp-math'}, 'fast-math'),
                   ('cargo_configuration', {'config': {'build_rustflags': '-ffast-math'}}, 'fast-math')]
        for key, value, message in invalid:
            with self.subTest(key=key):
                record = {**self.record, key: value}
                with self.assertRaisesRegex(ValueError, message):
                    compare.validate_build_record(record, self.binary)
        for record in [{}, [], None]:
            with self.subTest(empty_record=record):
                with self.assertRaises(ValueError):
                    compare.validate_build_record(record, self.binary)

    def test_failure_keeps_rejection_snapshot_after_process_exits(self):
        snapshot = {'cpu': 0, 'active_build_processes': [{'pid': 123, 'parent_pid': 100,
                                                        'executable': 'rustc', 'state': 'R', 'cwd': '/build'}]}
        with patch.object(compare, 'host', return_value=snapshot):
            compare.capture_host(self.root, 0, 'before')
        with patch.object(compare, 'build_processes', return_value=[]):
            failure = compare.failure_record(RuntimeError('build processes are active'), self.root)
        self.assertEqual(failure['host_snapshots']['host-before']['active_build_processes'], snapshot['active_build_processes'])
        self.assertEqual(failure['active_build_processes'], [])
        self.assertFalse(failure['performance_acceptance'])
        json.dumps(failure)

    def test_unowned_lock_is_preserved(self):
        lock = self.root/'lock'
        lock.write_text('another owner')
        with self.assertRaises(FileExistsError):
            with compare.lock(self.root, 'lock', 'test'):
                self.fail('an occupied lock must not be acquired')
        self.assertEqual(lock.read_text(), 'another owner')


class CorrectnessTests(unittest.TestCase):
    def test_scaled_error_handles_large_finite_coefficients(self):
        self.assertAlmostEqual(compare.scaled_error(complex(-1e308, 1e308), complex(1e308, -1e308)), 2.)
        self.assertEqual(compare.scaled_error(0j, 0j), 0.)

    def test_every_post_coefficient_and_status_is_checked(self):
        rust = {'results': [{'lane': 0, 'status': {'kind': 'success'},
                             'values': [[3., 0.], [2., 0.], [1., 0.], [2., 0.], [1., 0.]]}]}
        row = {'lane': 0, 'status': 0, 'finite': True, 'epsilon': [[3., 0.], [2., 0.], [1., 0.]],
               'cut_constructible': [2., 0.], 'rational': [1., 0.]}
        original = {'success': True, 'pre_post_equal': True, 'pre_results': [copy.deepcopy(row)],
                    'post_results': [copy.deepcopy(row)]}
        self.assertEqual(len(compare.check_results(rust, original)), 2)
        for coefficient in range(3):
            with self.subTest(coefficient=coefficient):
                changed = copy.deepcopy(original)
                changed['post_results'][0]['epsilon'][coefficient][0] += 1.
                with self.assertRaisesRegex(ValueError, 'Laurent comparison failed in post_results'):
                    compare.check_results(rust, changed)
        original['post_results'][0]['status'] = 1
        with self.assertRaisesRegex(ValueError, 'failed numerical rows'):
            compare.check_results(rust, original)


if __name__ == '__main__':
    unittest.main()
