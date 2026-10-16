#!/usr/bin/env python3
"""Gate the immutable four-photon matrix and independent pole-control reports."""
import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path

STATES = ('pppp', 'ppmm', 'mpmp', 'generic_complex')
PRECISIONS = ('f64', 'double-float', 'float')
MODES = ('expansion', 'sampling')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def unique_coverage(rows, key, expected):
    keys = [key(row) for row in rows]
    return len(keys) == len(expected) and len(set(keys)) == len(keys) and set(keys) == expected


def small(value, target='1e-9'):
    value = Decimal(str(value))
    return value.is_finite() and 0 <= value <= Decimal(target)


def validate(matrix, poles):
    rows = matrix['checks']
    expected = {(o, p, m, lane) for o in range(6) for p in PRECISIONS
                for m in MODES for lane in range(4)}
    key = lambda row: (row['orientation'], row['precision'], row['mu_extraction'], row['lane'])
    physical = [row for row in rows if row['lane'] < 3]
    generic = [row for row in rows if row['lane'] == 3]
    state_contract = all(row['state'] == STATES[row['lane']] and
                         row['diagnostic_only'] == (row['lane'] == 3) for row in rows)

    def row_pass(row, mode):
        comparisons = [c for c in row['comparisons'] if c['reference_mode'] == mode]
        if row['result']['status']['kind'] != 'success' or row['process_returncode'] != 0 or len(comparisons) != 1:
            return False
        comparison = comparisons[0]
        errors = comparison['scaled_errors']
        return (comparison['reference_stable'] and errors is not None and len(errors) == 3
                and all(small(error) for error in errors)
                and comparison['decomposition_scaled_error'] is not None
                and small(comparison['decomposition_scaled_error']))

    mu = [row for row in matrix['controls'] if row['kind'] == 'mu_mode_equivalence']
    expected_mu = {(o, p, state) for o in range(6) for p in PRECISIONS for state in STATES}
    mu_key = lambda row: (row['orientation'], row['precision'], row['state'])
    summed = [row for row in matrix['summed_graphs'] if row['reference_mode'] == 4]
    expected_sums = {(p, m, state) for p in PRECISIONS for m in MODES for state in STATES}
    sum_key = lambda row: (row['precision'], row['mu_extraction'], row['state'])
    accuracy = poles['integrated_pole_accuracy']
    targets = {'f64': '1e-9', 'double-float': '1e-25', 'float': '1e-65'}
    gates = {
        'complete_matrix': unique_coverage(rows, key, expected),
        'physical_row_count_108': len(physical) == 108,
        'generic_row_count_36': len(generic) == 36,
        'state_and_diagnostic_contract': state_contract,
        'working_precision': all(row['working_bits'] == {'f64': 53, 'double-float': 106, 'float': 256}[row['precision']]
                                 for row in rows),
        'process_success': len(matrix['provenance']['commands']) == 42 and
                           all(c['returncode'] == 0 for c in matrix['provenance']['commands']),
        'unchanged_runtime_sources': not matrix['changed_sources'],
        'physical_mode4_laurents_and_decomposition': all(row_pass(row, 4) for row in physical),
        'complete_mu_mode_controls': unique_coverage(mu, mu_key, expected_mu),
        'physical_mu_mode_equivalence': all(len(row['errors']) == 3 and all(small(e) for e in row['errors'])
                                          for row in mu if row['state'] != 'generic_complex'),
        'complete_summed_graph_controls': unique_coverage(summed, sum_key, expected_sums),
        'physical_summed_graphs_mode4': all(len(row['scaled_errors']) == 3 and all(small(e) for e in row['scaled_errors'])
                                          for row in summed if row['state'] != 'generic_complex'),
        'same_cli_for_exact_poles': matrix['provenance']['binary']['sha256'] == poles['provenance']['ninjars_binary_sha256'],
        'complete_exact_pole_reference_controls': unique_coverage(
            poles['rows'], lambda row: (row['orientation'], row['lane']),
            {(orientation, lane) for orientation in range(6) for lane in range(4)}),
        'physical_exact_poles_reference_mode4': all(
            len([c for c in row['comparisons'] if c['mode'] == 4]) == 1 and
            all(c['stable'] and small(c['scaled_error']) for c in row['comparisons'] if c['mode'] == 4)
            for row in poles['rows'] if row['lane'] < 3),
        'exact_uv_cancellation': poles['exact_sum_zero'] == [True] * 4,
        'complete_integrated_pole_controls': unique_coverage(accuracy, key, expected),
        'fixed_precision_targets': poles['integrated_pole_accuracy_targets'] == targets and
                                   all(row['target'] == targets[row['precision']] for row in accuracy),
        'physical_integrated_pole_accuracy': all(row['scaled_error'] is not None and
                                                small(row['scaled_error'], targets[row['precision']])
                                                for row in accuracy if row['lane'] < 3),
    }
    diagnostics = {
        'physical_mode1_laurents_and_decomposition': all(row_pass(row, 1) for row in physical),
        'generic_mode1_laurents_and_decomposition': all(row_pass(row, 1) for row in generic),
        'generic_mode4_laurents_and_decomposition': all(row_pass(row, 4) for row in generic),
        'generic_mu_mode_equivalence': all(len(row['errors']) == 3 and all(small(e) for e in row['errors'])
                                         for row in mu if row['state'] == 'generic_complex'),
        'generic_summed_graphs_mode4': all(len(row['scaled_errors']) == 3 and all(small(e) for e in row['scaled_errors'])
                                         for row in summed if row['state'] == 'generic_complex'),
        'generic_integrated_pole_accuracy': all(row['scaled_error'] is not None and
                                               small(row['scaled_error'], targets[row['precision']])
                                               for row in accuracy if row['lane'] == 3),
    }
    maxima = {}
    for precision in PRECISIONS:
        maxima[precision] = {}
        for label, selected in [('physical', [r for r in accuracy if r['precision'] == precision and r['lane'] < 3]),
                                ('including_generic_diagnostics', [r for r in accuracy if r['precision'] == precision])]:
            maxima[precision][label] = (str(max(Decimal(row['scaled_error']) for row in selected))
                                         if selected and all(row['scaled_error'] is not None for row in selected) else None)
    return {'physical_passed': all(gates.values()), 'gates': gates, 'diagnostics': diagnostics,
            'physical_rows': len(physical), 'generic_rows': len(generic),
            'independent_pole_error_maxima': maxima,
            'precision_convergence': 'The matrix Float256 convergence rows are informational; the independent analytic pole targets above are gated.',
            'performance_acceptance': False,
            'scope': 'Six separate unsquared top-quark boxes at the fixed phase-space point and three helicity states; generic complex polarizations are diagnostic. Four-dimensional numerators retain R1 and exclude explicit R2/UV counterterms.',
            'finite_part_input_oracle': 'MadLoop wavefunctions/couplings are binary64, including fixed mode4. No independent higher-precision physical-input oracle is claimed for finite parts.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--poles', type=Path, required=True)
    parser.add_argument('--build-record', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text())
    poles = json.loads(args.poles.read_text())
    build = json.loads(args.build_record.read_text())
    result = validate(matrix, poles)
    result['gates']['matching_build_binary'] = build['binary_sha256'] == matrix['provenance']['binary']['sha256']
    result['physical_passed'] = all(result['gates'].values())
    result['schema_version'] = 1
    result['provenance'] = {'matrix': {'path': str(args.matrix), 'sha256': digest(args.matrix)},
                            'poles': {'path': str(args.poles), 'sha256': digest(args.poles)},
                            'build_record': {'path': str(args.build_record), 'sha256': digest(args.build_record)},
                            'aggregator_sha256': digest(Path(__file__))}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(args.output)
    if not result['physical_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
