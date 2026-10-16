import argparse
import copy
import unittest
from kernel_selection import add_arguments, native_enabled, validate_report, FAMILIES, SELECTIONS


def coverage(selection, compiled=True):
    count = {'symbolica': 0, 'native-denominators': 2, 'native': 4}[selection]
    return {'selection': selection, 'native_feature_enabled': compiled,
            'families': {key: 'native-rust' if i < count else 'symbolica-interpreter' for i, key in enumerate(FAMILIES)}}


class KernelSelectionTests(unittest.TestCase):
    def test_default_compatibility_and_same_artifact_option(self):
        parser = argparse.ArgumentParser()
        add_arguments(parser, all_catalog=True)
        args = parser.parse_args([])
        self.assertEqual(args.reduction_kernels, 'symbolica')
        self.assertFalse(native_enabled(args))
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


if __name__ == '__main__':
    unittest.main()
