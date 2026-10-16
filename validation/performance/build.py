#!/usr/bin/env python3
"""Build the optimized CLI and record its actual sources, flags and executable."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('checkpoint', ROOT/'validation/record_checkpoint.py')
checkpoint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checkpoint)
ENVIRONMENT = ('RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS', 'CC', 'CXX', 'PKG_CONFIG',
               'CFLAGS', 'CXXFLAGS', 'CARGO_TARGET_DIR', 'CARGO_BUILD_TARGET',
               'CARGO_PROFILE_RELEASE_OPT_LEVEL', 'CARGO_PROFILE_RELEASE_LTO',
               'CARGO_PROFILE_RELEASE_CODEGEN_UNITS', 'CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS')
TOP_LEVEL = {'Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', 'flake.nix', 'flake.lock'}


def production_snapshot():
    snapshot = checkpoint.source_snapshot()
    snapshot['source_sha256'] = {
        name: value for name, value in snapshot['source_sha256'].items()
        if name.startswith(('crates/', 'dependencies/')) or name in TOP_LEVEL
    }
    return snapshot


def cargo_configuration():
    # Hash configuration rather than copying it: registry or environment
    # sections can contain credentials unrelated to this build record.
    directories = [ROOT, *ROOT.parents]
    cargo_home = Path(os.environ.get('CARGO_HOME', Path.home()/'.cargo'))
    paths = {directory/'.cargo'/name for directory in directories for name in ('config', 'config.toml')}
    paths.update(cargo_home/name for name in ('config', 'config.toml'))
    result = {}
    for path in sorted(paths):
        if not path.is_file():
            continue
        parsed = tomllib.loads(path.read_text())
        result[str(path)] = {
            'sha256': checkpoint.digest(path),
            'build_rustflags': parsed.get('build', {}).get('rustflags'),
            'target_rustflags': {key: value.get('rustflags') for key, value in parsed.get('target', {}).items()
                                 if isinstance(value, dict) and 'rustflags' in value},
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'target/performance/build',
                        help='new or empty directory for rust-build.json and build.log')
    parser.add_argument('--jobs', type=int, default=8)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('artifact directory must be new or empty')
    output.mkdir(parents=True, exist_ok=True)
    command = ['cargo', 'build', '--release', '-j', str(args.jobs), '-p', 'ninjars-cli', '--all-features', '--locked']
    environment = {key: os.environ.get(key) for key in ENVIRONMENT}
    configuration = cargo_configuration()
    declared_flags = json.dumps([environment, configuration]).lower()
    if any(flag in declared_flags for flag in ('fast-math', 'fast_math', '-ofast', 'unsafe-fp-math')):
        parser.error('fast-math is incompatible with the controlled comparison')
    if environment['CARGO_BUILD_TARGET']:
        parser.error('cross compilation is unsupported by the same-host comparison')
    before, dependencies = production_snapshot(), checkpoint.dependencies()
    metadata = json.loads(subprocess.check_output(
        ['cargo', 'metadata', '--format-version', '1', '--no-deps', '--locked'], cwd=ROOT, text=True))
    binary = Path(metadata['target_directory'])/'release/ninjars'
    log = output/'build.log'
    with log.open('w') as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise SystemExit(f'optimized build failed; see {log}')
    after = production_snapshot()
    if (before['git_head'] != after['git_head'] or before['source_sha256'] != after['source_sha256']
            or dependencies != checkpoint.dependencies() or configuration != cargo_configuration()):
        raise SystemExit('production sources, dependency state or Cargo configuration changed during the build')
    record = {
        'schema_version': 1, 'recorded_utc': datetime.now(timezone.utc).isoformat(),
        'binary': str(binary), 'binary_sha256': checkpoint.digest(binary), 'command': command,
        'compiler': checkpoint.command('rustc', '-Vv'), 'cargo': checkpoint.command('cargo', '-V'),
        'environment': environment, 'cargo_configuration': configuration,
        'release_profile': tomllib.loads((ROOT/'Cargo.toml').read_text())['profile']['release'],
        'build_log_sha256': checkpoint.digest(log), 'build_log_tail': log.read_text().splitlines()[-3:],
        'checkout': after, 'dependencies': dependencies,
        'notes': ['Sources and dependency state checked before and after the completed build.',
                  'Environment uses an explicit allowlist; credentials and Symbolica license are excluded.'],
    }
    path = output/'rust-build.json'
    path.write_text(json.dumps(record, indent=2)+'\n')
    print(path)


if __name__ == '__main__':
    main()
