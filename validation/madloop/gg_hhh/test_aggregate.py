#!/usr/bin/env python3
"""Mutation regressions for the gg->HHH evidence gate, using real reports."""
import argparse
import copy
import json
import gzip
import tempfile
import zlib

from aggregate import digest, validate
from pathlib import Path
from reference_io import read_json
from archive_integrated import coverage, verify
from reference_io import archive_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--poles', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    matrix = read_json(args.matrix)
    poles = read_json(args.poles)
    assert validate(matrix, poles)['physical_passed'], 'unmodified reference reports must pass'
    checks = []
    with tempfile.TemporaryDirectory() as directory:
        archive=Path(directory)/'matrix.json.gz'
        manifest={**archive_json(args.matrix,archive),'coverage_and_runtime_guard':coverage(matrix)}
        assert verify(archive,manifest)==matrix
        checks.append({'case':'lossless matrix archive identity and coverage','passed':True})
        data=archive.read_bytes()
        for label, damaged in [('truncated matrix archive',data[:-20]),
                               ('corrupted matrix archive',data[:20]+bytes([data[20]^1])+data[21:])]:
            archive.write_bytes(damaged)
            try:verify(archive,manifest)
            except (EOFError,gzip.BadGzipFile,ValueError,zlib.error):pass
            else:raise AssertionError(f'accepted {label}')
            checks.append({'case':label,'rejected':True})

    def reject(name, mutation):
        changed_matrix, changed_poles = copy.deepcopy(matrix), copy.deepcopy(poles)
        mutation(changed_matrix, changed_poles)
        result = validate(changed_matrix, changed_poles)
        assert not result['physical_passed'], f'gate accepted {name}'
        checks.append({'case': name, 'rejected_by': [k for k, v in result['gates'].items() if not v]})

    reject('missing physical row', lambda m, p: m['checks'].pop(0))
    reject('duplicate row replaces another sample', lambda m, p: m['checks'].__setitem__(0, copy.deepcopy(m['checks'][1])))
    reject('unsuccessful process with readable output', lambda m, p: m['provenance']['commands'][0].__setitem__('returncode', 1))
    reject('mu-mode mismatch', lambda m, p: next(r for r in m['controls'] if r['kind'] == 'mu_mode_equivalence' and r['state'] in ('pp','pm','mp','mm'))['errors'].__setitem__(0, '1'))
    reject('missing all mu-mode controls', lambda m, p: m.__setitem__('controls', [r for r in m['controls'] if r['kind'] != 'mu_mode_equivalence']))
    reject('summed-graph mismatch', lambda m, p: next(r for r in m['summed_graphs'] if r['reference_mode'] == 4 and r['state'] in ('pp','pm','mp','mm'))['scaled_errors'].__setitem__(0, 1))
    reject('missing independent pole row', lambda m, p: p['integrated_pole_accuracy'].pop(0))
    reject('DoubleFloat misses fixed accuracy target', lambda m, p: next(r for r in p['integrated_pole_accuracy'] if r['precision'] == 'double-float' and r['lane'] < 4).__setitem__('scaled_error', '1e-20'))
    reject('altered pole output despite stale comparison flags', lambda m, p: m['checks'][0]['result']['values'][1].__setitem__(0, '1e10'))
    reject('altered finite output despite stale flags', lambda m,p:m['checks'][0]['result']['values'][0].__setitem__(0,'1e10'))
    def balanced(m,p):
        chosen=[r for r in m['checks'] if r['precision']=='double-float' and r['mu_extraction']=='expansion' and r['lane']==0][:2]
        chosen[0]['result']['values'][1]=['1','0'];chosen[1]['result']['values'][1]=['-1','0']
    reject('balanced pole corruption hidden by exact graph sum',balanced)
    reject('corrupt exact pole control value',lambda m,p:p['rows'][0].__setitem__('value',['1','0']))
    reject('relaxed precision target', lambda m, p: p['integrated_pole_accuracy_targets'].__setitem__('double-float', '1e-8'))
    reject('changed runtime source', lambda m, p: m['changed_sources'].append('injected mutation'))
    reject('changed exact-pole source', lambda m,p:p['provenance']['source_sha256_after'].__setitem__('injected mutation','bad hash'))
    reject('changed exact-pole library', lambda m,p:p['provenance']['rlib_sha256_after'].__setitem__('injected mutation','bad hash'))
    reject('Float result at wrong working precision', lambda m, p: next(r for r in m['checks'] if r['precision'] == 'float').__setitem__('working_bits', 53))

    for name, mode, lane, diagnostic in [
        ('generic mismatch remains diagnostic', 4, 4, 'generic_and_ward_mode4_laurents_and_decomposition'),
        ('Ward mismatch remains diagnostic', 4, 5, 'generic_and_ward_mode4_laurents_and_decomposition'),
        ('mode1 mismatch remains diagnostic', 1, 0, 'physical_mode1_laurents_and_decomposition'),
    ]:
        changed = copy.deepcopy(matrix)
        row = next(row for row in changed['checks'] if row['lane'] == lane)
        next(c for c in row['comparisons'] if c['reference_mode'] == mode)['scaled_errors'][0] = 1
        result = validate(changed, poles)
        assert result['physical_passed'] and not result['diagnostics'][diagnostic]
        checks.append({'case': name, 'physical_gate_passed': True, 'diagnostic_passed': False})

    args.output.write_text(json.dumps({
        'passed': True, 'checks': checks,
        'matrix_sha256': digest(args.matrix), 'poles_sha256': digest(args.poles),
        'test_source_sha256': digest(Path(__file__)),
        'aggregator_sha256': digest(Path(__file__).with_name('aggregate.py')),
    }, indent=2)+'\n')
    print(f'{len(checks)} mutation regressions passed')


if __name__ == '__main__':
    main()
