#!/usr/bin/env python3
"""Check each box's UV pole independently by exact numerator differentiation."""
import argparse
import datetime
from decimal import Decimal, localcontext
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
STATES = ('pppp', 'ppmm', 'mpmp', 'generic_complex')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ninjars', type=Path, required=True)
    parser.add_argument('--symbolica-rlib', type=Path, required=True)
    parser.add_argument('--serde-json-rlib', type=Path, required=True)
    parser.add_argument('--rustc', default='rustc')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--integrated', type=Path,
                        help='Optional completed compare.py result for direct pole accuracy controls.')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    binary = args.ninjars.resolve()
    libraries = {'symbolica': args.symbolica_rlib.resolve(),
                 'serde_json': args.serde_json_rlib.resolve()}
    if len({p.parent for p in libraries.values()}) != 1:
        parser.error('rlibs must come from one locked workspace dependency directory')
    commands = []
    files = []
    for orientation in range(6):
        fixture = ROOT/f'examples/graphs/physical/four-photon/top-box-{orientation}.json'
        command = [str(binary), 'prepare', str(fixture)]
        output = out/f'prepared-{orientation}.json'
        with output.open('w') as stdout, (out/f'prepared-{orientation}.stderr').open('w') as stderr:
            subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=True)
        commands.append(command)
        files.append(output)
    probe = out/'uv-poles'
    command = [args.rustc, '--edition=2024', str(HERE/'uv_poles.rs'), '-o', str(probe),
               '-L', f'dependency={args.symbolica_rlib.resolve().parent}']
    for name, path in libraries.items():
        command += ['--extern', f'{name}={path}']
    env = dict(os.environ, CARGO_CRATE_NAME='four_photon_uv_control')
    with (out/'build.log').open('w') as stream:
        subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=stream, check=True)
    commands.append(command)
    command = [str(probe), *map(str, files)]
    with (out/'exact.json').open('w') as stdout, (out/'probe.stderr').open('w') as stderr:
        subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, check=True)
    commands.append(command)
    exact = json.loads((out/'exact.json').read_text())
    raw_path = ROOT/'validation/evidence/four-photon-raw-generic.json'
    reference_path = ROOT/'validation/evidence/four-photon-generic-madloop.json'
    raw = json.loads(raw_path.read_text())
    assert raw['pass']
    mapping = {row['feynkit_orientation']: row['madloop_filter_id'] for row in raw['graphs']}
    reference = json.loads(reference_path.read_text())
    references = {(graph['graph_filter_id'], case['state'], case['ctmode']): case
                  for graph in reference['graphs'] for case in graph['cases']}
    for row in exact['rows']:
        value = complex(*map(float, row['value']))
        row['state'] = STATES[row['lane']]
        row['diagnostic_only'] = row['lane'] == 3
        row['comparisons'] = []
        for mode in (1, 4):
            case = references[mapping[row['orientation']], row['state'], mode]
            target = -48 * math.pi**2 * complex(*case['raw_laurent'][1])
            error = abs(value-target)/max(1., abs(target))
            if not math.isfinite(error):
                raise ValueError('nonfinite pole comparison')
            row['comparisons'].append({'mode': mode, 'target': [target.real, target.imag],
                                       'scaled_error': error, 'stable': case['stable'],
                                       'passed': error <= 1e-9 and case['stable']})
    sources = [HERE/'uv_poles.rs', Path(__file__).resolve(), raw_path, reference_path,
               ROOT/'crates/ninjars-cli/src/prepare.rs', ROOT/'Cargo.lock',
               ROOT/'dependencies/manifest.json', ROOT/'references/oneloopmaster/src/two_point.rs']
    sources += sorted((ROOT/'examples/graphs/physical/four-photon').glob('*'))
    accuracy = []
    accuracy_targets = {'f64': '1e-9', 'double-float': '1e-25', 'float': '1e-65'}
    if args.integrated:
        integrated_path = args.integrated.resolve()
        integrated = json.loads(integrated_path.read_text())
        poles = {(row['orientation'], row['lane']): row['value'] for row in exact['rows']}
        with localcontext() as context:
            context.prec = 120
            for row in integrated['checks']:
                error = None
                if row['result']['status']['kind'] == 'success':
                    actual = list(map(Decimal, row['result']['values'][1]))
                    expected = list(map(Decimal, poles[row['orientation'], row['lane']]))
                    norm = sum(value*value for value in expected).sqrt()
                    error = sum((a-b)**2 for a, b in zip(actual, expected)).sqrt()/max(Decimal(1), norm)
                    if not error.is_finite():
                        raise ValueError('nonfinite integrated-pole accuracy control')
                target = accuracy_targets[row['precision']]
                accuracy.append({
                    'orientation': row['orientation'], 'lane': row['lane'], 'state': row['state'],
                    'precision': row['precision'], 'mu_extraction': row['mu_extraction'],
                    'diagnostic_only': row['diagnostic_only'],
                    'scaled_error': None if error is None else str(error), 'target': target,
                    'passed': error is not None and error <= Decimal(target),
                })
        sources.append(integrated_path)
    result = {
        'schema_version': 1,
        'physical_passed': all(r['comparisons'][1]['passed'] for r in exact['rows'] if not r['diagnostic_only']),
        'generic_diagnostic_passed': all(r['comparisons'][1]['passed'] for r in exact['rows'] if r['diagnostic_only']),
        'criterion': 'abs(pole-reference)/max(1,abs(reference)) <= 1e-9; fixed mode4 is primary',
        'formula': 'Delta_q^2 N / 192 in four-dimensional +--- signature',
        'normalization_derivation': 'oneloopmaster A0000 pole is (m^2)^3/24; three mass-squared derivatives divided by 3! give 1/24 for a box metric pairing. The quartic numerator contracts the three pairings to Delta_q^2 N/8. The supplied overall factor is applied exactly once.',
        'scope': 'Independent per-graph single-pole control and exact six-graph UV cancellation; finite parts and reduction coefficients are not tested here.',
        'integrated_pole_accuracy_targets': accuracy_targets,
        'integrated_pole_accuracy': accuracy,
        'integrated_physical_poles_passed': all(row['passed'] for row in accuracy if not row['diagnostic_only']) if accuracy else None,
        'input_precision_note': 'Exact fixture inputs are differentiated and evaluated at 256 bits. MadLoop input wavefunctions/couplings are binary64, also in fixed mode4. This is not an independent high-precision physical-input oracle.',
        **exact,
        'provenance': {
            'recorded_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'commands': commands, 'compiler': subprocess.check_output([args.rustc, '-Vv'], text=True),
            'ninjars_binary_sha256': digest(binary), 'probe_binary_sha256': digest(probe),
            'rlib_sha256': {str(p): digest(p) for p in libraries.values()},
            'prepared_sha256': {p.name: digest(p) for p in files},
            'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in sources if p.is_file()},
        },
    }
    (out/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(out/'result.json')
    if not result['physical_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
