#!/usr/bin/env python3
"""Check the complete f64 SymJIT matrix against fixed-mode MadLoop and exact poles."""
import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path

from aggregate import small, unique_coverage
from compare import precision_error


def validate(matrix, poles):
    rows = matrix['checks']
    states = ('pppp', 'ppmm', 'mpmp', 'generic_complex')
    modes = ('expansion', 'sampling')
    expected = {(o, mode, lane) for o in range(6) for mode in modes for lane in range(4)}
    key = lambda row: (row['orientation'], row['mu_extraction'], row['lane'])
    pole_values = {(r['orientation'], r['lane']): r['value'] for r in poles['rows']}
    errors = []
    for row in rows:
        if row['result']['status']['kind'] == 'success':
            errors.append({**{k: row[k] for k in ('orientation', 'mu_extraction', 'lane')},
                           'scaled_error': precision_error(row['result']['values'][1],
                                                          pole_values[row['orientation'], row['lane']])})
    def passes(row):
        comparisons = [c for c in row['comparisons'] if c['reference_mode'] == 4]
        return (row['process_returncode'] == 0 and row['result']['status']['kind'] == 'success'
                and len(comparisons) == 1 and comparisons[0]['reference_stable']
                and len(comparisons[0]['scaled_errors']) == 3
                and all(small(e) for e in comparisons[0]['scaled_errors'])
                and small(comparisons[0]['decomposition_scaled_error']))
    mu = [r for r in matrix['controls'] if r['kind'] == 'mu_mode_equivalence']
    summed = [r for r in matrix['summed_graphs'] if r['reference_mode'] == 4]
    gates = {
        'symjit_o2_selected': matrix['numerator_evaluator'] == 'symjit-o2',
        'complete_matrix': unique_coverage(rows, key, expected),
        'f64_states': all(r['precision'] == 'f64' and r['working_bits'] == 53
                          and r['state'] == states[r['lane']] and r['diagnostic_only'] == (r['lane'] == 3) for r in rows),
        'process_success': len(matrix['provenance']['commands']) == 18
                           and all(c['returncode'] == 0 for c in matrix['provenance']['commands']),
        'unchanged_sources': not matrix['changed_sources'],
        'identical_prepared_inputs': len(matrix['preparations']) == 6 and
                                    {f"prepared-{r['orientation']}.json": r['sha256'] for r in matrix['preparations']}
                                    == poles['provenance']['prepared_sha256'],
        'physical_laurents_and_decomposition': all(passes(r) for r in rows if r['lane'] < 3),
        'complete_mu_controls': unique_coverage(mu, lambda r: (r['orientation'], r['state']),
                                               {(o, state) for o in range(6) for state in states}),
        'physical_mu_equivalence': all(len(r['errors']) == 3 and all(small(e) for e in r['errors'])
                                      for r in mu if r['state'] != 'generic_complex'),
        'complete_sums': unique_coverage(summed, lambda r: (r['mu_extraction'], r['state']),
                                        {(m, s) for m in modes for s in states}),
        'physical_summed_graphs': all(len(r['scaled_errors']) == 3 and all(small(e) for e in r['scaled_errors'])
                                     for r in summed if r['state'] != 'generic_complex'),
        'complete_pole_controls': unique_coverage(errors, key, expected),
        'physical_exact_poles': all(small(r['scaled_error']) for r in errors if r['lane'] < 3),
        'independent_exact_sum_zero': poles['exact_sum_zero'] == [True] * 4,
    }
    return {'passed': all(gates.values()), 'gates': gates, 'physical_rows': sum(r['lane'] < 3 for r in rows),
            'generic_rows': sum(r['lane'] == 3 for r in rows), 'criterion': 'scaled 1e-9 per complex coefficient',
            'generic_mode4_passed': all(passes(r) for r in rows if r['lane'] == 3),
            'independent_pole_max_error': str(max(Decimal(r['scaled_error']) for r in errors)),
            'performance_acceptance': False,
            'scope': 'f64 SymJIT O2 coefficient preparation followed by ordinary NinjaRS reduction; six boxes, both mu modes, three physical helicities and generic complex diagnostics'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--poles', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = validate(json.loads(args.matrix.read_text()), json.loads(args.poles.read_text()))
    paths = [args.matrix, args.poles, Path(__file__), Path(__file__).with_name('aggregate.py'),
             Path(__file__).with_name('compare.py')]
    result['source_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(args.output)
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
