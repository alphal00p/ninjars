import argparse
import contextlib
import copy
import io
import unittest
from kernel_selection import (add_arguments, native_enabled, validate_arguments, validate_coverage,
                              validate_report, FAMILIES, SELECTIONS, LEGACY_SELECTIONS,
                              NATIVE_SELECTIONS, MAX_NATIVE_BLOCK_ROWS, supported_selections)


SUPPORTED = {'propagators': 5, 'rank': 4}


def coverage(selection, compiled=True, topology=SUPPORTED):
    count = {'symbolica': 0, 'native-denominators': 2, 'native': 5,
             'interpreted-rank': 0, 'native-rank': 5}[selection]
    result = {'selection': selection, 'native_feature_enabled': compiled,
              'families': {key: 'native-rust' if i < count else 'symbolica-interpreter' for i, key in enumerate(FAMILIES)}}
    if topology is not None:
        result['topology'] = dict(topology)
    if selection in ('interpreted-rank', 'native-rank'):
        result['conditional_corrections'] = {
            'backend': 'native-rust' if selection == 'native-rank' else 'symbolica-interpreter',
            'guard': 'all-active-source-kind-structural-zeros',
            'fallback': 'reported-generic-families',
        }
    return result


def execution_coverage(selection, native_block_rows=None):
    result = coverage(selection)
    result['native_execution'] = {
        'policy': 'whole' if native_block_rows is None else 'fixed-rows',
        'block_rows': native_block_rows,
        'scope': 'native-kernel-rows',
    }
    return result


def execution_report(selection, native_block_rows):
    return {
        'reduction_kernels': execution_coverage(selection, native_block_rows),
        'integrated_widths': [0, 1, 3, 8],
        'execution_parity': [
            {'coverage': execution_coverage(selection, bound),
             'complex_components_checked': 60, 'max_scaled_error': 1e-15}
            for bound in (None, native_block_rows)
        ],
    }


class KernelSelectionTests(unittest.TestCase):
    def test_default_compatibility_and_same_artifact_option(self):
        parser = argparse.ArgumentParser()
        add_arguments(parser, all_catalog=True)
        args = parser.parse_args([])
        self.assertEqual(args.reduction_kernels, 'symbolica')
        self.assertIsNone(args.native_block_rows)
        validate_arguments(args)
        self.assertFalse(native_enabled(args))
        self.assertFalse(native_enabled(parser.parse_args(['--reduction-kernels', 'interpreted-rank'])))
        self.assertTrue(native_enabled(parser.parse_args(['--reduction-kernels', 'native-rank'])))
        for argv in [['--native-catalog'], ['--reduction-kernels', 'native'], ['--all-reduction-kernels']]:
            self.assertTrue(native_enabled(parser.parse_args(argv)))

    def test_native_block_arguments_are_explicit_positive_native_only(self):
        parser = argparse.ArgumentParser()
        add_arguments(parser, all_catalog=True)
        for selection in NATIVE_SELECTIONS:
            for bound in (1, 32, MAX_NATIVE_BLOCK_ROWS):
                args = parser.parse_args(['--reduction-kernels', selection, '--native-block-rows', str(bound)])
                validate_arguments(args)
                self.assertEqual(args.native_block_rows, bound)
                self.assertTrue(native_enabled(args))
        for value in ('0', '-1', '1.5', 'true', 'nan', str(MAX_NATIVE_BLOCK_ROWS + 1)):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(['--reduction-kernels', 'native', '--native-block-rows', value])
        for selection in ('symbolica', 'interpreted-rank'):
            args = parser.parse_args(['--reduction-kernels', selection, '--native-block-rows', '32'])
            self.assertFalse(native_enabled(args))
            with self.assertRaises(ValueError):
                validate_arguments(args)
            args.native_catalog = True
            with self.assertRaises(ValueError):
                validate_arguments(args)
        for selection in SELECTIONS:
            args = parser.parse_args(['--reduction-kernels', selection, '--native-block-rows', '32',
                                      '--all-reduction-kernels'])
            with self.assertRaises(ValueError):
                validate_arguments(args)
        for value in (True, False, 0, -1, 1.0, '1', [], {}, MAX_NATIVE_BLOCK_ROWS + 1):
            args = argparse.Namespace(reduction_kernels='native', native_block_rows=value)
            with self.assertRaises(ValueError):
                validate_arguments(args)

    def test_execution_metadata_keeps_historical_whole_compatible(self):
        for selection in SELECTIONS:
            validate_report({'reduction_kernels': coverage(selection)}, selection, True)
            current = {'reduction_kernels': execution_coverage(selection)}
            validate_report(current, selection, True)
            for mutation in [
                lambda c: c.update(native_execution=None),
                lambda c: c.update(native_execution={}),
                lambda c: c.update(native_execution='whole'),
                lambda c: c['native_execution'].update(policy='fixed-rows'),
                lambda c: c['native_execution'].update(block_rows=0),
                lambda c: c['native_execution'].update(block_rows=32),
                lambda c: c['native_execution'].update(scope='phase-space-lanes'),
                lambda c: c['native_execution'].update(observed_rows=32),
                lambda c: c['native_execution'].pop('scope'),
            ]:
                bad = copy.deepcopy(current)
                mutation(bad['reduction_kernels'])
                with self.assertRaises(ValueError):
                    validate_report(bad, selection, True)

    def test_fixed_execution_requires_exact_metadata_and_native_selection(self):
        for selection in NATIVE_SELECTIONS:
            fixed = execution_coverage(selection, 32)
            validate_coverage(fixed, selection, True, native_block_rows=32)
            # Timing reports read back configuration without adding numerical scopes.
            validate_report({'reduction_kernels': fixed}, selection, True, native_block_rows=32)
            for mutation in [
                lambda c: c.pop('native_execution'),
                lambda c: c.update(native_execution=None),
                lambda c: c.update(native_execution=[]),
                lambda c: c['native_execution'].update(policy='whole'),
                lambda c: c['native_execution'].update(policy='fixed'),
                lambda c: c['native_execution'].update(block_rows=None),
                lambda c: c['native_execution'].update(block_rows=31),
                lambda c: c['native_execution'].update(block_rows=32.0),
                lambda c: c['native_execution'].update(scope='native-callback-rows'),
                lambda c: c['native_execution'].pop('block_rows'),
                lambda c: c['native_execution'].update(unexpected=True),
            ]:
                bad = copy.deepcopy(fixed)
                mutation(bad)
                with self.assertRaises(ValueError):
                    validate_report({'reduction_kernels': bad}, selection, True, native_block_rows=32)
            with self.assertRaises(ValueError):
                validate_report({'reduction_kernels': fixed}, selection, True)
        boolean_bound = execution_coverage('native', 1)
        boolean_bound['native_execution']['block_rows'] = True
        with self.assertRaises(ValueError):
            validate_report({'reduction_kernels': boolean_bound}, 'native', True, native_block_rows=1)
        for value in (True, False, 0, -1, 1.0, '1', [], {}, MAX_NATIVE_BLOCK_ROWS + 1):
            with self.assertRaises(ValueError):
                validate_report({'reduction_kernels': coverage('native')}, 'native', True,
                                native_block_rows=value)
        for selection in ('symbolica', 'interpreted-rank'):
            with self.assertRaises(ValueError):
                validate_report({'reduction_kernels': execution_coverage(selection, 32)},
                                selection, True, native_block_rows=32)

    def test_fixed_physical_controls_require_whole_then_fixed_full_parity(self):
        for selection in NATIVE_SELECTIONS:
            good = execution_report(selection, 32)
            validate_report(good, selection, True, native_block_rows=32, execution_controls=True)
            for mutation in [
                lambda r: r.pop('execution_parity'),
                lambda r: r.update(execution_parity=None),
                lambda r: r.update(execution_parity={}),
                lambda r: r['execution_parity'].pop(),
                lambda r: r['execution_parity'].append(copy.deepcopy(r['execution_parity'][1])),
                lambda r: r['execution_parity'].reverse(),
                lambda r: r['execution_parity'].__setitem__(0, None),
                lambda r: r['execution_parity'][0].pop('coverage'),
                lambda r: r['execution_parity'][0].update(coverage=None),
                lambda r: r['execution_parity'][0]['coverage'].pop('native_execution'),
                lambda r: r['execution_parity'][1]['coverage'].pop('native_execution'),
                lambda r: r['execution_parity'][0]['coverage'].update(selection='symbolica'),
                lambda r: r['execution_parity'][1]['coverage'].update(native_feature_enabled=False),
                lambda r: r['execution_parity'][0]['coverage'].update(topology={'propagators': 6, 'rank': 4}),
                lambda r: r['execution_parity'][1]['coverage'].pop('topology'),
                lambda r: r['execution_parity'][0]['coverage']['families'].update(integration='native-rust'),
                lambda r: r['execution_parity'][0].update(max_scaled_error=float('nan')),
                lambda r: r['execution_parity'][1].update(max_scaled_error=float('inf')),
                lambda r: r['execution_parity'][0].update(max_scaled_error=-1e-15),
                lambda r: r['execution_parity'][1].update(max_scaled_error=1.1e-9),
                lambda r: r['execution_parity'][0].update(max_scaled_error=True),
                lambda r: r['execution_parity'][1].pop('max_scaled_error'),
                lambda r: r['execution_parity'][0].update(complex_components_checked=59),
                lambda r: r['execution_parity'][1].update(complex_components_checked=60.0),
                lambda r: r['execution_parity'][0].update(complex_components_checked=True),
                lambda r: r['execution_parity'][1].pop('complex_components_checked'),
            ]:
                bad = copy.deepcopy(good)
                mutation(bad)
                with self.assertRaises(ValueError):
                    validate_report(bad, selection, True, native_block_rows=32, execution_controls=True)
        bad = execution_report('native-rank', 32)
        bad['execution_parity'][1]['coverage']['conditional_corrections']['guard'] = 'primal-only'
        with self.assertRaises(ValueError):
            validate_report(bad, 'native-rank', True, native_block_rows=32, execution_controls=True)

    def test_execution_controls_require_bound_and_valid_integrated_widths(self):
        with self.assertRaises(ValueError):
            validate_report({'reduction_kernels': execution_coverage('native')}, 'native', True,
                            execution_controls=True)
        for widths in (None, [], [0], [True], [-1, 3], [1.0], ['1'], {'width': 1}):
            bad = execution_report('native', 32)
            bad['integrated_widths'] = widths
            with self.assertRaises(ValueError):
                validate_report(bad, 'native', True, native_block_rows=32, execution_controls=True)
        bad = execution_report('native', 32)
        bad.pop('integrated_widths')
        with self.assertRaises(ValueError):
            validate_report(bad, 'native', True, native_block_rows=32, execution_controls=True)

    def test_all_catalog_controls_remain_whole(self):
        good = {'reduction_kernels': execution_coverage('symbolica'), 'integrated_widths': [0, 1, 3, 8],
                'catalog_parity': [{'coverage': execution_coverage(name), 'complex_components_checked': 60,
                                   'max_scaled_error': 0.0} for name in SELECTIONS]}
        validate_report(good, 'symbolica', True, all_catalog=True)
        for index, name in enumerate(SELECTIONS):
            bad = copy.deepcopy(good)
            bad['catalog_parity'][index]['coverage'] = execution_coverage(name, 32)
            with self.assertRaises(ValueError):
                validate_report(bad, 'symbolica', True, all_catalog=True)
        with self.assertRaises(ValueError):
            validate_report(execution_report('native', 32), 'native', True,
                            all_catalog=True, native_block_rows=32)

    def test_each_family_and_feature_is_checked(self):
        for selection in SELECTIONS:
            good = {'reduction_kernels': coverage(selection)}
            validate_report(good, selection, True)
            for family in FAMILIES:
                bad = copy.deepcopy(good)
                old = bad['reduction_kernels']['families'][family]
                bad['reduction_kernels']['families'][family] = 'native-rust' if old == 'symbolica-interpreter' else 'symbolica-interpreter'
                with self.assertRaises(ValueError):
                    validate_report(bad, selection, True)
            for mutation in [lambda c: c.update(native_feature_enabled=False),
                             lambda c: c.update(selection='unknown'),
                             lambda c: c['families'].pop('integration'),
                             lambda c: c['families'].pop('tadpole_source_projection'),
                             lambda c: c['families'].update(unexpected='native-rust')]:
                bad = copy.deepcopy(good)
                mutation(bad['reduction_kernels'])
                with self.assertRaises(ValueError):
                    validate_report(bad, selection, True)
        validate_report({'reduction_kernels': coverage('symbolica', False)}, 'symbolica', False)

    def test_catalog_parity_requires_all_rows_and_finite_coefficients(self):
        good = {'reduction_kernels': coverage('symbolica'), 'integrated_widths': [0, 1, 3, 8],
                'catalog_parity': [{'coverage': coverage(name), 'complex_components_checked': 60,
                                   'max_scaled_error': 1e-15} for name in SELECTIONS]}
        validate_report(good, 'symbolica', True, all_catalog=True)
        for mutation in [lambda r: r['catalog_parity'].pop(),
                         lambda r: r['catalog_parity'].reverse(),
                         lambda r: r['catalog_parity'][1].update(max_scaled_error=float('nan')),
                         lambda r: r['catalog_parity'][1].update(max_scaled_error=1e-8),
                         lambda r: r['catalog_parity'][1].update(complex_components_checked=59)]:
            bad = copy.deepcopy(good)
            mutation(bad)
            with self.assertRaises(ValueError):
                validate_report(bad, 'symbolica', True, all_catalog=True)

    def test_conditional_metadata_is_required_and_exact(self):
        for selection in ('interpreted-rank', 'native-rank'):
            good = {'reduction_kernels': coverage(selection)}
            for mutation in [
                lambda c: c.pop('topology'),
                lambda c: c.update(topology={'propagators': 4, 'rank': 4}),
                lambda c: c.update(topology={'propagators': 5, 'rank': 5}),
                lambda c: c.update(topology={'propagators': True, 'rank': 4}),
                lambda c: c.pop('conditional_corrections'),
                lambda c: c.update(conditional_corrections=None),
                lambda c: c['conditional_corrections'].pop('guard'),
                lambda c: c['conditional_corrections'].update(guard='primal-only'),
                lambda c: c['conditional_corrections'].update(fallback='none'),
                lambda c: c['conditional_corrections'].update(backend='unknown'),
                lambda c: c['conditional_corrections'].update(unexpected=True),
            ]:
                bad = copy.deepcopy(good)
                mutation(bad['reduction_kernels'])
                with self.assertRaises(ValueError):
                    validate_report(bad, selection, True)
            bad = copy.deepcopy(good)
            bad['reduction_kernels']['conditional_corrections']['backend'] = (
                'symbolica-interpreter' if selection == 'native-rank' else 'native-rust')
            with self.assertRaises(ValueError):
                validate_report(bad, selection, True)
        generic = coverage('native')
        generic['conditional_corrections'] = coverage('native-rank')['conditional_corrections']
        with self.assertRaises(ValueError):
            validate_report({'reduction_kernels': generic}, 'native', True)
        validate_report({'reduction_kernels': coverage('interpreted-rank', False)}, 'interpreted-rank', False)
        with self.assertRaises(ValueError):
            validate_report({'reduction_kernels': coverage('native-rank', False)}, 'native-rank', False)

    def test_topology_rejects_impossible_rank_without_an_arbitrary_propagator_limit(self):
        for selection in LEGACY_SELECTIONS:
            for topology in [{'propagators': 2, 'rank': 4}, {'propagators': 5, 'rank': 7}]:
                with self.assertRaises(ValueError):
                    validate_report({'reduction_kernels': coverage(selection, True, topology)}, selection, True)
            for topology in [{'propagators': 2, 'rank': 3}, {'propagators': 42, 'rank': 43}]:
                validate_report({'reduction_kernels': coverage(selection, True, topology)}, selection, True)

    def test_topology_aware_all_modes_and_historical_report_compatibility(self):
        for topology, compiled, expected in [
            (SUPPORTED, True, SELECTIONS),
            ({'propagators': 4, 'rank': 3}, True, LEGACY_SELECTIONS),
            (SUPPORTED, False, ('symbolica', 'interpreted-rank')),
            ({'propagators': 4, 'rank': 3}, False, ('symbolica',)),
            (None, True, LEGACY_SELECTIONS),
        ]:
            self.assertEqual(supported_selections(topology, compiled), expected)
            good = {'reduction_kernels': coverage('symbolica', compiled, topology),
                    'integrated_widths': [1, 3],
                    'catalog_parity': [{'coverage': coverage(mode, compiled, topology),
                                       'complex_components_checked': 20,
                                       'max_scaled_error': 0.} for mode in expected]}
            validate_report(good, 'symbolica', compiled, all_catalog=True)
            missing = copy.deepcopy(good)
            missing['catalog_parity'].pop()
            with self.assertRaises(ValueError):
                validate_report(missing, 'symbolica', compiled, all_catalog=True)
            extra = copy.deepcopy(good)
            extra['catalog_parity'].append({'coverage': coverage('native-rank'),
                                           'complex_components_checked': 20, 'max_scaled_error': 0.})
            with self.assertRaises(ValueError):
                validate_report(extra, 'symbolica', compiled, all_catalog=True)
            mismatched = copy.deepcopy(good)
            mismatched['catalog_parity'][0]['coverage']['topology'] = {'propagators': 6, 'rank': 4}
            with self.assertRaises(ValueError):
                validate_report(mismatched, 'symbolica', compiled, all_catalog=True)
        # Historical reports cannot introduce rank modes without their metadata.
        for selection in ('interpreted-rank', 'native-rank'):
            with self.assertRaises(ValueError):
                validate_report({'reduction_kernels': coverage(selection, True, None)}, selection, True)


if __name__ == '__main__':
    unittest.main()
