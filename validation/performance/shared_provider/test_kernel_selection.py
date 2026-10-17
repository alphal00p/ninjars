import argparse
import copy
import unittest
from kernel_selection import add_arguments, native_enabled, validate_report, FAMILIES, SELECTIONS, LEGACY_SELECTIONS, supported_selections


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


class KernelSelectionTests(unittest.TestCase):
    def test_default_compatibility_and_same_artifact_option(self):
        parser = argparse.ArgumentParser()
        add_arguments(parser, all_catalog=True)
        args = parser.parse_args([])
        self.assertEqual(args.reduction_kernels, 'symbolica')
        self.assertFalse(native_enabled(args))
        self.assertFalse(native_enabled(parser.parse_args(['--reduction-kernels', 'interpreted-rank'])))
        self.assertTrue(native_enabled(parser.parse_args(['--reduction-kernels', 'native-rank'])))
        for argv in [['--native-catalog'], ['--reduction-kernels', 'native'], ['--all-reduction-kernels']]:
            self.assertTrue(native_enabled(parser.parse_args(argv)))

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
