"""Validate generic fallback and conditional coverage independently of providers."""
import math

LEGACY_SELECTIONS = ('symbolica', 'native-denominators', 'native')
RANK_SELECTIONS = ('interpreted-rank', 'native-rank')
SELECTIONS = LEGACY_SELECTIONS + RANK_SELECTIONS
NATIVE_SELECTIONS = ('native-denominators', 'native', 'native-rank')
FAMILIES = ('t2_denominators', 't3_denominators', 'bubble_source_projection',
            'triangle_bubble_correction', 'tadpole_source_projection', 'other_reduction', 'integration', 'accumulation')


def add_arguments(parser, *, all_catalog=False):
    parser.add_argument('--reduction-kernels', choices=SELECTIONS, default='symbolica')
    parser.add_argument('--native-catalog', action='store_true',
                        help='compile the optional catalog even for interpreted selection, allowing same-artifact controls')
    if all_catalog:
        parser.add_argument('--all-reduction-kernels', action='store_true',
                            help='check every topology-supported selection in one binary against the same numerator and frozen masters')


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


def validate_coverage(coverage, selection, compiled):
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


def validate_report(report, selection, compiled, *, all_catalog=False):
    validate_coverage(report['reduction_kernels'], selection, compiled)
    if all_catalog:
        topology = _topology(report['reduction_kernels'])
        selections = supported_selections(topology, compiled)
        rows = report.get('catalog_parity', [])
        if len(rows) != len(selections):
            raise ValueError('all topology-supported same-artifact controls are required')
        expected_checks = 5 * sum(report['integrated_widths'])
        for row, name in zip(rows, selections):
            validate_coverage(row['coverage'], name, compiled)
            if _topology(row['coverage']) != topology:
                raise ValueError('catalog parity topology changed between selections')
            error = row.get('max_scaled_error')
            if (type(error) not in (float, int) or not math.isfinite(error) or not 0 <= error <= 1e-9
                    or type(row.get('complex_components_checked')) is not int
                    or row['complex_components_checked'] != expected_checks or expected_checks <= 0):
                raise ValueError('incomplete or failed frozen-master catalog parity')
