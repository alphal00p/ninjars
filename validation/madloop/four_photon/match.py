#!/usr/bin/env python3
"""Compile the independent Rust graph probe against explicit workspace artifacts.

The rlibs must come from one locked workspace build. Rust rejects incompatible
crate identities. No production source or Cargo manifest is edited by this tool.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=ROOT/'validation/evidence/four-photon-madloop.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--graph-rlib', type=Path, required=True)
    parser.add_argument('--symbolica-rlib', type=Path, required=True)
    parser.add_argument('--feynkit-graph-rlib', type=Path, required=True)
    parser.add_argument('--serde-json-rlib', type=Path, required=True)
    parser.add_argument('--rustc', default='rustc')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    reference = args.reference.resolve()
    libraries = {
        'ninjars_graph': args.graph_rlib.resolve(),
        'symbolica': args.symbolica_rlib.resolve(),
        'feynkit_graph': args.feynkit_graph_rlib.resolve(),
        'serde_json': args.serde_json_rlib.resolve(),
    }
    dirs = {p.parent for p in libraries.values()}
    if len(dirs) != 1 or any(not p.is_file() for p in libraries.values()):
        parser.error('supply existing rlibs from the same workspace dependency directory')
    binary = out/'match-four-photons'
    command = [args.rustc, '--edition=2024', str(HERE/'match.rs'), '-o', str(binary),
               '-L', f'dependency={next(iter(dirs))}']
    for name, path in libraries.items():
        command += ['--extern', f'{name}={path}']
    env = dict(os.environ, CARGO_CRATE_NAME='four_photon_probe')
    with (out/'build.log').open('w') as stream:
        subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    run = [str(binary), str(reference)]
    with (out/'stdout.json').open('w') as stdout, (out/'stderr.log').open('w') as stderr:
        completed = subprocess.run(run, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, check=False)
    result = json.loads((out/'stdout.json').read_text())
    result['schema_version'] = 1
    result['exit_code'] = completed.returncode
    result['criterion'] = 'hypot(actual-expected)/max(1,hypot(expected)) <= 1e-9 for every raw sample'
    result['normalization'] = {
        'raw_factor': '-3',
        'proof': 'FeynKit exact temporal-polarization zero-momentum control is -12*GC_2^4*MT^4: fermion-loop sign(-1), internal color trace(3), Dirac trace(4). MadLoop LOOPNUM omits its separately audited color-amplitude factor (-1)*Nc=-3. No external color projector or spin/helicity average is applied.',
        'linear_laurent_factor': '-48*pi^2',
        'linear_laurent_derivation': 'The audited stock CT_interface multiplies its Laurent result by 1/(16*pi^2). Undoing this normalization and applying the independently checked raw factor gives -48*pi^2.',
        'integrated_comparison': 'Not performed by this raw-numerator probe.',
    }
    reference_data = json.loads(reference.read_text())
    internal_checks = sum(len(case['numerator_samples']) for graph in reference_data['graphs'] for case in graph['cases'])
    states = sorted({case['state'] for graph in reference_data['graphs'] for case in graph['cases'] if case['ctmode'] == 4})
    result['scope'] = {
        'polarization_states': states,
        'polarizations': 'Fixed polarization states listed above; polynomial unisolvence is in four loop components after polarization binding, not in arbitrary polarization variables.',
        'precision': 'The reference wavefunctions and coupling are binary64 inputs. This is not an independent high-precision physical-input oracle.',
        'mode_selection': 'Fixed mode4 rows only. Mode1 repeats in the input artifact are not counted again.',
        'internal_madloop_checks': f'The raw/LOOPNUM wrapper checks at both fixed modes comprise {internal_checks} comparisons; the independent FeynKit check comprises {result["checks"]}.',
    }
    sources = [HERE/'match.rs', Path(__file__).resolve(), reference,
               ROOT/'crates/ninjars-graph/src/physical.rs', ROOT/'crates/ninjars-graph/src/components.rs',
               ROOT/'crates/ninjars-graph/tests/four_photons.rs', ROOT/'Cargo.toml', ROOT/'Cargo.lock',
               ROOT/'dependencies/manifest.json']
    sources += sorted((ROOT/'examples/graphs/physical/four-photon').glob('*'))
    result['provenance'] = {
        'recorded_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'compile_command': command, 'run_command': run,
        'compiler': subprocess.check_output([args.rustc, '-Vv'], text=True),
        'binary_sha256': digest(binary),
        'rlib_sha256': {str(p): digest(p) for p in libraries.values()},
        'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in sources if p.is_file()},
        'graph_dependency_heads': {
            name: subprocess.check_output(['git', '-C', str(ROOT/path), 'rev-parse', 'HEAD'], text=True).strip()
            for name, path in [('symbolica','references/symbolica'), ('feynkit','references/gammaloop_ninja_shared')]
        },
    }
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(out/'result.json')
    if completed.returncode or not result['pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
