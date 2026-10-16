#!/usr/bin/env python3
"""Run the Rust numerical physical-preparation probe from locked Cargo artifacts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cargo-artifacts', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        parser.error('output directory must be empty')
    names = {'ninjars_graph', 'feynkit_kinematics', 'symbolica', 'serde_json'}
    libraries = {}
    for line in args.cargo_artifacts.read_text().splitlines():
        record = json.loads(line)
        if record.get('reason') != 'compiler-artifact':
            continue
        name = record['target']['name']
        for path in record['filenames']:
            if name in names and path.endswith('.rlib'):
                libraries[name] = Path(path)
    if set(libraries) != names:
        parser.error(f'missing artifacts: {names-set(libraries)}')
    directories = {path.parent for name, path in libraries.items() if name != 'ninjars_graph'}
    if len(directories) != 1:
        parser.error('all dependency artifacts must come from one build directory')
    paths = [HERE/'src/main.rs', Path(__file__), ROOT/'Cargo.lock', ROOT/'dependencies/manifest.json',
             ROOT/'examples/graphs/physical/four-photon/top-box-0.json']
    before = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    artifact_before = {str(path): digest(path) for path in libraries.values()}
    binary = out/'physical-input-probe'
    command = ['rustc', '--edition=2024', str(HERE/'src/main.rs'), '-Dwarnings', '-o', str(binary),
               '-L', f'dependency={next(iter(directories))}']
    for name, path in libraries.items():
        command += ['--extern', f'{name}={path}']
    env = dict(os.environ, CARGO_MANIFEST_DIR=str(HERE), CARGO_CRATE_NAME='physical_input_probe')
    with (out/'build.log').open('w') as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    run = [str(binary)]
    with (out/'stdout.json').open('w') as stdout, (out/'stderr.log').open('w') as stderr:
        subprocess.run(run, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, check=True)
    result = json.loads((out/'stdout.json').read_text())
    after = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    artifact_after = {str(path): digest(path) for path in libraries.values()}
    result['unchanged_sources'] = before == after
    result['unchanged_artifacts'] = artifact_before == artifact_after
    result['passed'] &= result['unchanged_sources'] and result['unchanged_artifacts']
    result['provenance'] = {'compile_command': command, 'run_command': run,
        'compiler': subprocess.check_output(['rustc', '-Vv'], text=True),
        'binary_sha256': digest(binary), 'source_sha256': before,
        'dependency_manifest': json.loads((ROOT/'dependencies/manifest.json').read_text()),
        'cargo_artifacts_sha256': digest(args.cargo_artifacts),
        'rlib_sha256': artifact_before, 'rlib_sha256_after': artifact_after,
        'logs': {name: {'text': (out/name).read_text(), 'sha256': digest(out/name)}
                 for name in ['build.log', 'stderr.log']}}
    (out/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(out/'result.json')
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
