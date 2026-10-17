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
    parser.add_argument('--native-catalog', action='store_true',
                        help='compile the optional catalog even for interpreted selection, allowing same-artifact controls')
    parser.add_argument('--native-block-rows', type=_parse_native_block_rows,
                        help='configure a positive native-kernel row bound for an explicitly native selection')
    if all_catalog:
        parser.add_argument('--all-reduction-kernels', action='store_true',
                            help='check every topology-supported selection in one binary against the same numerator and frozen masters')


def validate_arguments(args):
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


def validate_coverage(coverage, selection, compiled, *, native_block_rows=None):
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
                    execution_controls=False):
    if all_catalog and native_block_rows is not None:
        raise ValueError('all-catalog controls require Whole execution')
    if execution_controls and native_block_rows is None:
        raise ValueError('execution controls require an explicit native block bound')
    primary = report['reduction_kernels']
    validate_coverage(primary, selection, compiled, native_block_rows=native_block_rows)
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
            validate_coverage(row['coverage'], name, compiled)
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
            validate_coverage(coverage, selection, compiled, native_block_rows=bound)
            if (_topology(coverage) != topology or coverage['families'] != primary['families']
                    or coverage.get('conditional_corrections') != primary.get('conditional_corrections')):
                raise ValueError('kernel coverage changed between execution policies')
            _validate_parity_components(row, expected_checks, 'execution')
