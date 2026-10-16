"""Validate explicit reduction coverage independently of numerator/master choices."""
import math

SELECTIONS = ('symbolica', 'native-denominators', 'native')
FAMILIES = ('t2_denominators', 't3_denominators', 'bubble_source_projection',
            'triangle_bubble_correction', 'other_reduction', 'integration', 'accumulation')


def add_arguments(parser, *, all_catalog=False):
    parser.add_argument('--reduction-kernels', choices=SELECTIONS, default='symbolica')
    parser.add_argument('--native-catalog', action='store_true',
                        help='compile the optional catalog even for interpreted selection, allowing same-artifact controls')
    if all_catalog:
        parser.add_argument('--all-reduction-kernels', action='store_true',
                            help='check all three selections in one binary against the same fixed numerator and frozen masters')


def native_enabled(args):
    return args.native_catalog or args.reduction_kernels != 'symbolica' or getattr(args, 'all_reduction_kernels', False)


def validate_coverage(coverage, selection, compiled):
    if selection not in SELECTIONS:
        raise ValueError('unknown reduction selection')
    count = {'symbolica': 0, 'native-denominators': 2, 'native': 4}[selection]
    expected = {name: 'native-rust' if i < count else 'symbolica-interpreter'
                for i, name in enumerate(FAMILIES)}
    if (coverage.get('selection') != selection
            or type(coverage.get('native_feature_enabled')) is not bool
            or coverage['native_feature_enabled'] != compiled
            or coverage.get('families') != expected
            or (selection != 'symbolica' and not compiled)):
        raise ValueError('actual reduction coverage does not match requested selection/build')


def validate_report(report, selection, compiled, *, all_catalog=False):
    validate_coverage(report['reduction_kernels'], selection, compiled)
    if all_catalog:
        rows = report.get('catalog_parity', [])
        if len(rows) != len(SELECTIONS):
            raise ValueError('all three same-artifact catalog controls are required')
        expected_checks = 5 * sum(report['integrated_widths'])
        for row, name in zip(rows, SELECTIONS):
            validate_coverage(row['coverage'], name, True)
            error = row.get('max_scaled_error')
            if (type(error) not in (float, int) or not math.isfinite(error) or not 0 <= error <= 1e-9
                    or type(row.get('complex_components_checked')) is not int
                    or row['complex_components_checked'] != expected_checks or expected_checks <= 0):
                raise ValueError('incomplete or failed frozen-master catalog parity')
