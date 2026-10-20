"""Pruning configuration and physical report contracts; legacy fixtures stay off."""
import argparse
import copy
import unittest

from kernel_selection import (add_arguments, native_enabled, validate_arguments,
                              validate_coverage, validate_report, SELECTIONS,
                              NATIVE_SELECTIONS, validate_pruning_preflight)
from test_kernel_selection import coverage, execution_coverage


def pruning_coverage(selection='symbolica', enabled=True, bound=None, *, n=5, rank=4):
    result = execution_coverage(selection, bound)
    result['topology'] = {'propagators': n, 'rank': rank}
    if not enabled:
        full = swapped = None
    elif n < 2 or rank < n - 2:
        full = swapped = 0
    else:
        full, swapped = ((1, 0), (3, 1), (7, 3), (13, 7))[rank - (n - 2)]
    result['pruned_swapped_arithmetic'] = {
        'enabled': enabled,
        'selected_backend': 'symbolica-interpreter' if enabled else None,
        'full_columns': full,
        'swapped_columns': swapped,
        'guarded_rank_correction': enabled and selection in ('interpreted-rank', 'native-rank'),
        'scope': 'projection-only-configuration',
        'numerator_callbacks': 'unchanged-full',
    }
    return result


def parity_row(selected, enabled=True, bound=None):
    return {'coverage': pruning_coverage(selected, enabled, bound),
            'complex_components_checked': 60, 'max_scaled_error': 1e-15,
            'numerator_call_row_deltas': [12, 3, 24, 18, 36, 9, 72, 54]}


def pruning_report(selection='symbolica', bound=None):
    return {'reduction_kernels': pruning_coverage(selection, True, bound),
            'integrated_widths': [0, 1, 3, 8],
            'pruning_parity': [parity_row(selection, enabled, bound) for enabled in (False, True)]}


def validate_pruning(report, selection='symbolica', bound=None, **kwargs):
    validate_report(report, selection, True, native_block_rows=bound,
                    prune_swapped_arithmetic=True, pruning_controls=True, **kwargs)


class PrunedSwappedTests(unittest.TestCase):
    def test_flag_defaults_off_without_enabling_native_or_relaxing_policy(self):
        parser = argparse.ArgumentParser()
        add_arguments(parser, all_catalog=True)
        plain = parser.parse_args([])
        self.assertFalse(plain.prune_swapped_arithmetic)
        enabled = parser.parse_args(['--prune-swapped-arithmetic'])
        validate_arguments(enabled)
        self.assertTrue(enabled.prune_swapped_arithmetic)
        self.assertFalse(native_enabled(enabled))
        for selection in SELECTIONS:
            args = parser.parse_args(['--reduction-kernels', selection, '--prune-swapped-arithmetic'])
            validate_arguments(args)
            self.assertEqual(native_enabled(args), selection in NATIVE_SELECTIONS)
            args.native_block_rows = 32
            if selection in NATIVE_SELECTIONS:
                validate_arguments(args)
            else:
                with self.assertRaises(ValueError):
                    validate_arguments(args)
            args.all_reduction_kernels = True
            with self.assertRaises(ValueError):
                validate_arguments(args)
        for value in (0, 1, 'true', None, [], {}):
            args = parser.parse_args([])
            args.prune_swapped_arithmetic = value
            with self.assertRaises(ValueError):
                validate_arguments(args)

    def test_historical_default_off_and_explicit_disabled_are_distinct(self):
        for selection in SELECTIONS:
            historical = coverage(selection)
            validate_coverage(historical, selection, True)
            current = pruning_coverage(selection, False)
            validate_coverage(current, selection, True)
            with self.assertRaises(ValueError):
                validate_coverage(historical, selection, True, prune_swapped_arithmetic=True)
            for key, value in [('enabled', 0), ('enabled', 1), ('enabled', 'false'),
                               ('selected_backend', 'symbolica-interpreter'), ('full_columns', 0),
                               ('swapped_columns', 0), ('guarded_rank_correction', 0),
                               ('guarded_rank_correction', True), ('scope', 'executed'),
                               ('numerator_callbacks', 'trimmed'), ('extra', False)]:
                bad = copy.deepcopy(current)
                bad['pruned_swapped_arithmetic'][key] = value
                with self.subTest(selection=selection, key=key, value=value), self.assertRaises(ValueError):
                    validate_coverage(bad, selection, True)
            for key in current['pruned_swapped_arithmetic']:
                bad = copy.deepcopy(current)
                del bad['pruned_swapped_arithmetic'][key]
                with self.assertRaises(ValueError):
                    validate_coverage(bad, selection, True)
        for selection in ('symbolica', 'native-denominators', 'native'):
            validate_coverage(coverage(selection, topology=None), selection, True)

    def test_all_degrees_inactive_layouts_and_interpreted_backend_are_explicit(self):
        for n, rank, expected in [(3, 1, (1, 0)), (3, 2, (3, 1)), (3, 3, (7, 3)),
                                  (3, 4, (13, 7)), (1, 1, (0, 0)), (5, 0, (0, 0))]:
            good = pruning_coverage(n=n, rank=rank)
            info = good['pruned_swapped_arithmetic']
            self.assertEqual((info['full_columns'], info['swapped_columns']), expected)
            validate_coverage(good, 'symbolica', True, prune_swapped_arithmetic=True)
            for key in ('full_columns', 'swapped_columns'):
                for value in (True, False, float(info[key]), info[key] + 1, None):
                    bad = copy.deepcopy(good)
                    bad['pruned_swapped_arithmetic'][key] = value
                    with self.assertRaises(ValueError):
                        validate_coverage(bad, 'symbolica', True, prune_swapped_arithmetic=True)
        for selection in SELECTIONS:
            for bound in ((None, 32) if selection in NATIVE_SELECTIONS else (None,)):
                good = pruning_coverage(selection, bound=bound)
                validate_coverage(good, selection, True, native_block_rows=bound,
                                  prune_swapped_arithmetic=True)
                self.assertEqual(good['families'], coverage(selection)['families'])
                for key, value in [('enabled', 0), ('enabled', 1), ('enabled', 'true'),
                                   ('selected_backend', 'native-rust'), ('scope', 'executed'),
                                   ('scope', 'phase-space-lanes'), ('numerator_callbacks', 'trimmed'),
                                   ('guarded_rank_correction', not good['pruned_swapped_arithmetic']['guarded_rank_correction']),
                                   ('guarded_rank_correction', 1), ('extra', True)]:
                    bad = copy.deepcopy(good)
                    bad['pruned_swapped_arithmetic'][key] = value
                    with self.subTest(selection=selection, key=key), self.assertRaises(ValueError):
                        validate_coverage(bad, selection, True, native_block_rows=bound,
                                          prune_swapped_arithmetic=True)
                for key in good['pruned_swapped_arithmetic']:
                    bad = copy.deepcopy(good)
                    del bad['pruned_swapped_arithmetic'][key]
                    with self.assertRaises(ValueError):
                        validate_coverage(bad, selection, True, native_block_rows=bound,
                                          prune_swapped_arithmetic=True)
        for key in ('topology', 'native_execution', 'pruned_swapped_arithmetic'):
            bad = pruning_coverage()
            del bad[key]
            with self.assertRaises(ValueError):
                validate_coverage(bad, 'symbolica', True, prune_swapped_arithmetic=True)
        for selection in SELECTIONS:
            feature_off = pruning_coverage(selection)
            feature_off['native_feature_enabled'] = False
            if selection in NATIVE_SELECTIONS:
                with self.assertRaises(ValueError):
                    validate_coverage(feature_off, selection, False, prune_swapped_arithmetic=True)
            else:
                validate_coverage(feature_off, selection, False, prune_swapped_arithmetic=True)

    def test_full_then_pruned_all_five_heads_and_callback_deltas_are_required(self):
        for selection in SELECTIONS:
            good = pruning_report(selection)
            validate_pruning(good, selection)
            for mutation in [
                lambda r: r.pop('pruning_parity'),
                lambda r: r.update(pruning_parity=None),
                lambda r: r.update(pruning_parity={}),
                lambda r: r['pruning_parity'].pop(),
                lambda r: r['pruning_parity'].reverse(),
                lambda r: r['pruning_parity'].append(copy.deepcopy(r['pruning_parity'][1])),
                lambda r: r['pruning_parity'].__setitem__(1, copy.deepcopy(r['pruning_parity'][0])),
                lambda r: r['pruning_parity'].__setitem__(0, None),
                lambda r: r['pruning_parity'][0].pop('coverage'),
                lambda r: r['pruning_parity'][0]['coverage'].pop('pruned_swapped_arithmetic'),
                lambda r: r['pruning_parity'][0]['coverage'].pop('native_execution'),
                lambda r: r['pruning_parity'][1]['coverage'].pop('topology'),
                lambda r: r['pruning_parity'][0]['coverage'].update(topology={'propagators': 6, 'rank': 4}),
                lambda r: r['pruning_parity'][1]['coverage']['families'].update(integration='native-rust'),
                lambda r: r['pruning_parity'][0]['coverage'].update(native_feature_enabled=False),
                lambda r: r['pruning_parity'][0].pop('max_scaled_error'),
                lambda r: r['pruning_parity'][1].pop('complex_components_checked'),
                lambda r: r['pruning_parity'][0].pop('numerator_call_row_deltas'),
                lambda r: r['pruning_parity'][1]['numerator_call_row_deltas'].__setitem__(0, 13),
                lambda r: r['pruning_parity'][1]['numerator_call_row_deltas'].pop(),
            ]:
                bad = copy.deepcopy(good)
                mutation(bad)
                with self.subTest(selection=selection, mutation=mutation), self.assertRaises(ValueError):
                    validate_pruning(bad, selection)
        for key, values in {
            'max_scaled_error': [float('nan'), float('inf'), -1e-15, 1.1e-9, True, None],
            'complex_components_checked': [36, 59, 61, 60.0, True, None],
            'numerator_call_row_deltas': [None, {}, [0] * 8, [0] * 4 + [1] * 4,
                                         [1] * 4 + [0] * 4, [1] * 7, [1] * 9,
                                         [True] * 8, [1.0] * 8, [-1] + [1] * 7],
        }.items():
            for value in values:
                bad = pruning_report()
                bad['pruning_parity'][0][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    validate_pruning(bad)
        with self.assertRaises(ValueError):
            validate_report(pruning_report(), 'symbolica', True, pruning_controls=True)
        for widths in (None, [], [0], [True], [1.0], [-1, 3]):
            bad = pruning_report()
            bad['integrated_widths'] = widths
            with self.assertRaises(ValueError):
                validate_pruning(bad)

    def test_pruning_is_preserved_by_execution_and_catalog_controls(self):
        for selection in NATIVE_SELECTIONS:
            good = pruning_report(selection, 32)
            good['execution_parity'] = [parity_row(selection, bound=bound) for bound in (None, 32)]
            validate_pruning(good, selection, 32, execution_controls=True)
            for section in ('pruning_parity', 'execution_parity'):
                for index in range(2):
                    bad = copy.deepcopy(good)
                    bad[section][index]['coverage']['native_execution']['block_rows'] = 31
                    with self.assertRaises(ValueError):
                        validate_pruning(bad, selection, 32, execution_controls=True)
            for index in range(2):
                bad = copy.deepcopy(good)
                bad['execution_parity'][index]['coverage']['pruned_swapped_arithmetic']['enabled'] = False
                with self.assertRaises(ValueError):
                    validate_pruning(bad, selection, 32, execution_controls=True)
        good = pruning_report()
        good['catalog_parity'] = [parity_row(selection) for selection in SELECTIONS]
        validate_pruning(good, all_catalog=True)
        for index, selection in enumerate(SELECTIONS):
            bad = copy.deepcopy(good)
            bad['catalog_parity'][index]['coverage'] = pruning_coverage(selection, False)
            with self.assertRaises(ValueError):
                validate_pruning(bad, all_catalog=True)
        bad = pruning_report('native-rank')
        bad['pruning_parity'][0]['coverage']['conditional_corrections']['guard'] = 'primal-only'
        with self.assertRaises(ValueError):
            validate_pruning(bad, 'native-rank')

    def test_enabled_timing_preflight_requires_current_matching_physical_controls(self):
        sources = {'kernel.rs': 'a' * 64, 'physical_probe.cc': 'b' * 64}
        for selection in SELECTIONS:
            for bound in ((None, 32) if selection in NATIVE_SELECTIONS else (None,)):
                good = pruning_report(selection, bound)
                good['source_sha256'] = sources.copy()
                if bound is not None:
                    good['execution_parity'] = [parity_row(selection, bound=value) for value in (None, bound)]
                validate_pruning_preflight(good, selection, True, sources, required_widths=[1, 8],
                                          native_block_rows=bound)
                for mutation in [
                    lambda r: r.pop('reduction_kernels'),
                    lambda r: r.update(reduction_kernels=None),
                    lambda r: r.pop('source_sha256'),
                    lambda r: r.update(source_sha256={}),
                    lambda r: r.update(source_sha256=None),
                    lambda r: r['source_sha256'].pop('kernel.rs'),
                    lambda r: r['source_sha256'].update(unrelated='c' * 64),
                    lambda r: r['source_sha256'].update(**{'physical_probe.cc': 'c' * 64}),
                    lambda r: r.pop('pruning_parity'),
                    lambda r: r['reduction_kernels'].pop('pruned_swapped_arithmetic'),
                    lambda r: r['reduction_kernels'].update(selection='invalid'),
                    lambda r: r['pruning_parity'].reverse(),
                    lambda r: r['pruning_parity'][1]['coverage']['native_execution'].update(block_rows=7),
                ]:
                    bad = copy.deepcopy(good)
                    mutation(bad)
                    with self.assertRaises(ValueError):
                        validate_pruning_preflight(bad, selection, True, sources, required_widths=[1, 8],
                                                  native_block_rows=bound)
                if bound is not None:
                    bad = copy.deepcopy(good)
                    bad.pop('execution_parity')
                    with self.assertRaises(ValueError):
                        validate_pruning_preflight(bad, selection, True, sources, required_widths=[1, 8],
                                                  native_block_rows=bound)
        for invalid_sources in (None, {}, [], 'current'):
            with self.assertRaises(ValueError):
                validate_pruning_preflight(pruning_report(), 'symbolica', True, invalid_sources,
                                          required_widths=[1, 8])
        good = pruning_report()
        good['source_sha256'] = sources.copy()
        for widths in (None, [], [0], [True], [1.0], [-1], [1025], [1, 1024], [31, 32]):
            with self.assertRaises(ValueError):
                validate_pruning_preflight(good, 'symbolica', True, sources, required_widths=widths)
        for widths in ([0, 1, 1, 3, 8], [0, 1.0, 3, 8], [0, True, 3, 8]):
            bad = copy.deepcopy(good)
            bad['integrated_widths'] = widths
            for row in bad['pruning_parity']:
                row['complex_components_checked'] = int(5 * sum(widths))
            with self.assertRaises(ValueError):
                validate_pruning_preflight(bad, 'symbolica', True, sources, required_widths=[1, 8])
        validate_pruning_preflight(good, 'symbolica', True, sources, required_widths=[8, 1, 8])


if __name__ == '__main__':
    unittest.main()
