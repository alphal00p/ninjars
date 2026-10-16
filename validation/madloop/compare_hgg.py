#!/usr/bin/env python3
"""Compare physical NinjaRS graph coefficients to fixed-mode MadLoop evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ninjars', type=Path, default=ROOT/'target/debug/ninjars')
    parser.add_argument('--reference', type=Path, default=ROOT/'validation/evidence/physical-hgg-madloop.json')
    parser.add_argument('--output', type=Path, default=ROOT/'target/reference/madloop-hgg-comparison.json')
    args = parser.parse_args()
    binary = args.ninjars.resolve()
    source_paths = [
        ROOT/f'examples/graphs/physical/hgg/top-triangle-{orientation}.{suffix}'
        for orientation in range(2) for suffix in ['dot', 'json']
    ] + [Path(__file__).resolve(), ROOT/'crates/ninjars-graph/src/physical.rs',
         ROOT/'crates/ninjars-cli/src/prepare.rs', ROOT/'crates/ninjars-cli/src/runtime.rs']
    provenance = {
        'binary': {'path': str(binary), 'sha256': sha256(binary)},
        'source_sha256': {str(path.relative_to(ROOT)): sha256(path) for path in source_paths},
        'dependency_manifest': {'path': 'dependencies/manifest.json',
                                'sha256': sha256(ROOT/'dependencies/manifest.json')},
        'working_directory': str(ROOT),
        'runs': [],
    }
    reference = json.loads(args.reference.read_text())
    expected = {(case['feynkit_orientation'], case['sample_lane']): case
                for case in reference['cases'] if case['ctmode'] == 4}
    checks, failed, maximum = [], False, 0.0
    for orientation in range(2):
        fixture = ROOT/f'examples/graphs/physical/hgg/top-triangle-{orientation}.json'
        for precision in ['f64', 'double-float', 'float']:
            for batch in [1, 2]:
                command = [str(binary), 'reduce', str(fixture),
                           '--precision', precision, '--batch-size', str(batch)]
                if precision == 'float':
                    command += ['--bits', '256']
                output = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
                provenance['runs'].append({'orientation': orientation, 'precision': precision,
                                           'batch_size': batch, 'command': command,
                                           'returncode': output.returncode})
                if not output.stdout.strip():
                    raise RuntimeError(output.stderr)
                actual = json.loads(output.stdout)
                failed |= output.returncode != 0
                for lane, row in enumerate(actual['results']):
                    expected_row = expected[orientation, lane]
                    errors = []
                    if row['status']['kind'] == 'success':
                        for index, value in enumerate(row['values'][:3]):
                            value = complex(*map(float, value))
                            target = complex(*expected_row['epsilon_ninjars_normalization'][index])
                            error = abs(value-target)/max(1.0, abs(target))
                            if not math.isfinite(error):
                                raise RuntimeError('nonfinite physical comparison')
                            errors.append(error)
                        for part in range(2):
                            finite = float(row['values'][0][part])
                            split = sum(float(row['values'][i][part]) for i in [3, 4])
                            errors.append(abs(finite-split)/max(1.0, abs(finite)))
                    else:
                        failed = True
                    maximum = max([maximum, *errors])
                    passed = len(errors) == 5 and max(errors) <= 1e-9
                    failed |= not passed
                    checks.append({'orientation': orientation, 'lane': lane,
                                   'state': expected_row['state'], 'precision': precision,
                                   'working_bits': actual['precision']['bits'], 'batch_size': batch,
                                   'result': row, 'reference': expected_row['epsilon_ninjars_normalization'],
                                   'scaled_errors': errors, 'passed': passed})
    if sha256(binary) != provenance['binary']['sha256']:
        raise RuntimeError('NinjaRS binary changed during the physical comparison')
    for path in source_paths:
        if sha256(path) != provenance['source_sha256'][str(path.relative_to(ROOT))]:
            raise RuntimeError(f'physical comparison source changed during execution: {path}')
    if sha256(ROOT/'dependencies/manifest.json') != provenance['dependency_manifest']['sha256']:
        raise RuntimeError('dependency manifest changed during the physical comparison')
    result = {'schema_version': 1, 'passed': not failed, 'maximum_scaled_error': maximum,
              'criterion': 'each complex Laurent coefficient: abs(a-b)/max(1,abs(b)) <= 1e-9; finite=cc+rational',
              'reference_mode': 4, 'reference_sha256': hashlib.sha256(args.reference.read_bytes()).hexdigest(),
              'reference_input_precision': 'binary64 couplings and wavefunctions, promoted exactly for fixed quadruple arithmetic',
              'high_precision_input_oracle': False,
              'reference_precision_note': 'fixed quadruple arithmetic promotes the same binary64 physical inputs; this is not an independently high-precision physical-input oracle',
              'raw_numerator_gate': 'separate Rust physical test, 90 points on degree-two unisolvent complex grids',
              'performance_acceptance': False, 'provenance': provenance, 'checks': checks}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(f'{len(checks)} coefficient rows; max scaled error {maximum:.6e}; passed={not failed}')
    print(args.output)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
