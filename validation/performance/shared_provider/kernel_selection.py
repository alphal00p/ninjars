"""Validate kernel coverage and execution configuration independently of providers."""
import argparse
import ctypes
import math

LEGACY_SELECTIONS = ('symbolica', 'native-denominators', 'native')
RANK_SELECTIONS = ('interpreted-rank', 'native-rank')
SELECTIONS = LEGACY_SELECTIONS + RANK_SELECTIONS
NATIVE_SELECTIONS = ('native-denominators', 'native', 'native-rank')
FAMILIES = ('t2_denominators', 't3_denominators', 'bubble_source_projection',
            'triangle_bubble_correction', 'tadpole_source_projection', 'other_reduction', 'integration', 'accumulation')
MAX_NATIVE_BLOCK_ROWS = (1 << (8 * ctypes.sizeof(ctypes.c_size_t))) - 1


def _validate_native_block_rows(value):
    if value is not None and (type(value) is not int or not 0 < value <= MAX_NATIVE_BLOCK_ROWS):
        raise ValueError('native block rows must be a positive size_t integer')


def _parse_native_block_rows(value):
    try:
        rows = int(value)
        _validate_native_block_rows(rows)
    except ValueError as error:
        raise argparse.ArgumentTypeError('native block rows must be a positive size_t integer') from error
    return rows


def add_arguments(parser, *, all_catalog=False):
    parser.add_argument('--reduction-kernels', choices=SELECTIONS, default='symbolica')
    parser.add_argument('--prune-swapped-arithmetic', action='store_true',
                        help='use interpreted positive-head swapped arithmetic only for complete bubble projection; callbacks remain full')
    parser.add_argument('--native-catalog', action='store_true',
                        help='compile the optional catalog even for interpreted selection, allowing same-artifact controls')
    parser.add_argument('--native-block-rows', type=_parse_native_block_rows,
                        help='configure a positive native-kernel row bound for an explicitly native selection')
    if all_catalog:
        parser.add_argument('--all-reduction-kernels', action='store_true',
                            help='check every topology-supported selection in one binary against the same numerator and frozen masters')


def validate_arguments(args):
    if type(getattr(args, 'prune_swapped_arithmetic', False)) is not bool:
        raise ValueError('prune-swapped-arithmetic must be a boolean')
    rows = getattr(args, 'native_block_rows', None)
    _validate_native_block_rows(rows)
    if rows is not None:
        if getattr(args, 'all_reduction_kernels', False):
            raise ValueError('--native-block-rows cannot be combined with --all-reduction-kernels')
        if args.reduction_kernels not in NATIVE_SELECTIONS:
            raise ValueError('--native-block-rows requires a native reduction selection')


def native_enabled(args):
    return (args.native_catalog or args.reduction_kernels in NATIVE_SELECTIONS
            or getattr(args, 'all_reduction_kernels', False))


def _topology(coverage):
    if 'topology' not in coverage:
        return None
    topology = coverage['topology']
    if (not isinstance(topology, dict) or set(topology) != {'propagators', 'rank'}
            or type(topology['propagators']) is not int or topology['propagators'] <= 0
            or type(topology['rank']) is not int or topology['rank'] < 0
            or topology['rank'] > topology['propagators'] + 1):
        raise ValueError('invalid reduction topology')
    return topology


def supported_selections(topology, compiled):
    """No topology means a historical three-mode report, never a rank-mode report."""
    if topology is None:
        return LEGACY_SELECTIONS
    result = ['symbolica']
    if compiled:
        result.extend(('native-denominators', 'native'))
    if topology == {'propagators': 5, 'rank': 4}:
        result.append('interpreted-rank')
        if compiled:
            result.append('native-rank')
    return tuple(result)


def _validate_native_execution(coverage, selection, native_block_rows):
    _validate_native_block_rows(native_block_rows)
    if native_block_rows is not None and selection not in NATIVE_SELECTIONS:
        raise ValueError('fixed-row execution requires a native reduction selection')
    if 'native_execution' not in coverage and native_block_rows is None:
        # Reports predating the execution-policy interface used Whole only.
        return
    expected = {
        'policy': 'whole' if native_block_rows is None else 'fixed-rows',
        'block_rows': native_block_rows,
        'scope': 'native-kernel-rows',
    }
    execution = coverage.get('native_execution')
    if (not isinstance(execution, dict) or execution != expected
            or (native_block_rows is not None and type(execution.get('block_rows')) is not int)):
        raise ValueError('native execution configuration does not match the requested policy/bound')


def _validate_pruned_swapped(coverage, selection, enabled):
    if type(enabled) is not bool:
        raise ValueError('prune-swapped-arithmetic must be a boolean')
    if 'pruned_swapped_arithmetic' not in coverage and not enabled:
        # Historical reports and old constructors had pruning disabled.
        return
    topology = _topology(coverage)
    full = swapped = None
    if enabled:
        if topology is None:
            raise ValueError('configured pruning requires explicit topology')
        if 'native_execution' not in coverage:
            raise ValueError('configured pruning requires explicit native policy metadata')
        n, rank = topology['propagators'], topology['rank']
        if n < 2 or rank < n - 2:
            full = swapped = 0
        else:
            degree = rank - (n - 2)
            # Cardinalities of the existing T2 callback layout, not algebra.
            full, swapped = (1, 3, 7, 13)[degree], (0, 1, 3, 7)[degree]
    expected = {
        'enabled': enabled,
        'selected_backend': 'symbolica-interpreter' if enabled else None,
        'full_columns': full,
        'swapped_columns': swapped,
        'guarded_rank_correction': enabled and selection in RANK_SELECTIONS,
        'scope': 'projection-only-configuration',
        'numerator_callbacks': 'unchanged-full',
    }
    actual = coverage.get('pruned_swapped_arithmetic')
    if (not isinstance(actual, dict) or actual != expected
            or type(actual.get('enabled')) is not bool
            or type(actual.get('guarded_rank_correction')) is not bool
            or (enabled and any(type(actual.get(field)) is not int
                                for field in ('full_columns', 'swapped_columns')))):
        raise ValueError('configured pruning must retain full callbacks and report interpreted selected graphs')


def validate_coverage(coverage, selection, compiled, *, native_block_rows=None,
                      prune_swapped_arithmetic=False):
    if selection not in SELECTIONS:
        raise ValueError('unknown reduction selection')
    count = {'symbolica': 0, 'native-denominators': 2, 'native': 5,
             'interpreted-rank': 0, 'native-rank': 5}[selection]
    expected = {name: 'native-rust' if i < count else 'symbolica-interpreter'
                for i, name in enumerate(FAMILIES)}
    if (coverage.get('selection') != selection
            or type(coverage.get('native_feature_enabled')) is not bool
            or coverage['native_feature_enabled'] != compiled
            or coverage.get('families') != expected
            or (selection in NATIVE_SELECTIONS and not compiled)):
        raise ValueError('actual reduction coverage does not match requested selection/build')
    topology = _topology(coverage)
    if selection in RANK_SELECTIONS:
        expected_alternate = {
            'backend': 'native-rust' if selection == 'native-rank' else 'symbolica-interpreter',
            'guard': 'all-active-source-kind-structural-zeros',
            'fallback': 'reported-generic-families',
        }
        if (topology != {'propagators': 5, 'rank': 4}
                or coverage.get('conditional_corrections') != expected_alternate):
            raise ValueError('rank correction topology/conditional coverage mismatch')
    elif 'conditional_corrections' in coverage:
        raise ValueError('generic selection must not claim conditional rank corrections')
    _validate_native_execution(coverage, selection, native_block_rows)
    _validate_pruned_swapped(coverage, selection, prune_swapped_arithmetic)


def _expected_components(report):
    widths = report.get('integrated_widths')
    if (not isinstance(widths, list) or not widths
            or any(type(width) is not int or width < 0 for width in widths)):
        raise ValueError('parity requires nonnegative integer integrated widths')
    expected = 5 * sum(widths)
    if expected <= 0:
        raise ValueError('parity requires at least one integrated lane')
    return expected


def _validate_parity_components(row, expected_checks, description):
    error = row.get('max_scaled_error')
    if (type(error) not in (float, int) or not math.isfinite(error) or not 0 <= error <= 1e-9
            or type(row.get('complex_components_checked')) is not int
            or row['complex_components_checked'] != expected_checks):
        raise ValueError('incomplete or failed frozen-master ' + description + ' parity')


def validate_report(report, selection, compiled, *, all_catalog=False, native_block_rows=None,
                    execution_controls=False, prune_swapped_arithmetic=False, pruning_controls=False):
    if all_catalog and native_block_rows is not None:
        raise ValueError('all-catalog controls require Whole execution')
    if execution_controls and native_block_rows is None:
        raise ValueError('execution controls require an explicit native block bound')
    if pruning_controls and not prune_swapped_arithmetic:
        raise ValueError('pruning controls require explicit pruning opt-in')
    primary = report['reduction_kernels']
    validate_coverage(primary, selection, compiled, native_block_rows=native_block_rows,
                      prune_swapped_arithmetic=prune_swapped_arithmetic)
    if all_catalog:
        topology = _topology(primary)
        selections = supported_selections(topology, compiled)
        rows = report.get('catalog_parity', [])
        if not isinstance(rows, list) or len(rows) != len(selections):
            raise ValueError('all topology-supported same-artifact controls are required')
        expected_checks = _expected_components(report)
        for row, name in zip(rows, selections):
            if not isinstance(row, dict) or not isinstance(row.get('coverage'), dict):
                raise ValueError('invalid catalog parity row')
            validate_coverage(row['coverage'], name, compiled,
                              prune_swapped_arithmetic=prune_swapped_arithmetic)
            if _topology(row['coverage']) != topology:
                raise ValueError('catalog parity topology changed between selections')
            _validate_parity_components(row, expected_checks, 'catalog')
    if execution_controls:
        rows = report.get('execution_parity')
        if not isinstance(rows, list) or len(rows) != 2:
            raise ValueError('Whole then FixedRows same-selection execution controls are required')
        expected_checks = _expected_components(report)
        topology = _topology(primary)
        for row, bound in zip(rows, (None, native_block_rows)):
            if not isinstance(row, dict) or not isinstance(row.get('coverage'), dict):
                raise ValueError('invalid execution parity row')
            coverage = row['coverage']
            if 'native_execution' not in coverage:
                raise ValueError('execution parity requires explicit policy metadata')
            validate_coverage(coverage, selection, compiled, native_block_rows=bound,
                              prune_swapped_arithmetic=prune_swapped_arithmetic)
            if (_topology(coverage) != topology or coverage['families'] != primary['families']
                    or coverage.get('conditional_corrections') != primary.get('conditional_corrections')):
                raise ValueError('kernel coverage changed between execution policies')
            _validate_parity_components(row, expected_checks, 'execution')

    if pruning_controls:
        rows = report.get('pruning_parity')
        if not isinstance(rows, list) or len(rows) != 2:
            raise ValueError('full then pruned same-artifact controls are required')
        expected_checks = _expected_components(report)
        expected_callbacks = None
        for row, enabled in zip(rows, (False, True)):
            if not isinstance(row, dict) or not isinstance(row.get('coverage'), dict):
                raise ValueError('invalid pruning parity row')
            coverage = row['coverage']
            if 'pruned_swapped_arithmetic' not in coverage:
                raise ValueError('pruning parity requires explicit configured metadata')
            if 'native_execution' not in coverage:
                raise ValueError('pruning parity requires explicit native policy metadata')
            validate_coverage(coverage, selection, compiled, native_block_rows=native_block_rows,
                              prune_swapped_arithmetic=enabled)
            if (_topology(coverage) != _topology(primary)
                    or coverage['families'] != primary['families']
                    or coverage.get('conditional_corrections') != primary.get('conditional_corrections')
                    or coverage.get('native_execution') != primary.get('native_execution')):
                raise ValueError('kernel selection, topology or native policy changed between pruning controls')
            _validate_parity_components(row, expected_checks, 'pruning')
            callbacks = row.get('numerator_call_row_deltas')
            if (not isinstance(callbacks, list) or len(callbacks) != 8
                    or any(type(value) is not int or value < 0 for value in callbacks)
                    or sum(callbacks[:4]) == 0 or sum(callbacks[4:]) == 0):
                raise ValueError('pruning controls require eight successful numerator call/row deltas')
            if expected_callbacks is None:
                expected_callbacks = callbacks
            elif callbacks != expected_callbacks:
                raise ValueError('pruning changed the full numerator callback counts')


def validate_pruning_preflight(controls, selection, compiled, source_sha256, *, required_widths,
                              native_block_rows=None):
    """Enabled-only runner gate; historical default-off reports remain accepted separately."""
    if (not isinstance(controls, dict) or not isinstance(controls.get('reduction_kernels'), dict)
            or not isinstance(source_sha256, dict) or not source_sha256
            or controls.get('source_sha256') != source_sha256):
        raise ValueError('pruning requires physical controls matching every current runtime source')
    validate_report(controls, selection, compiled, native_block_rows=native_block_rows,
                    execution_controls=native_block_rows is not None,
                    prune_swapped_arithmetic=True, pruning_controls=True)
    control_widths = controls['integrated_widths']
    if len(set(control_widths)) != len(control_widths):
        raise ValueError('pruning physical control widths must not contain duplicates')
    if (not isinstance(required_widths, list) or not required_widths
            or any(type(width) is not int or not 0 < width <= 1024 for width in required_widths)
            or not set(required_widths).issubset(control_widths)):
        raise ValueError('pruning physical controls must cover every requested timing width')
