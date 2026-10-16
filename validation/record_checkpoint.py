#!/usr/bin/env python3
"""Record a completed workspace test run and the sources used to reproduce it.

Run after validation has finished and sources are frozen. This records evidence;
it does not infer performance acceptance from passing tests or timing metadata.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
TEST_COMMAND = (
    "RUST_MIN_STACK=134217728 cargo test --workspace --all-features --locked "
    "-- --include-ignored --test-threads=1 --nocapture"
)


def command(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_snapshot() -> dict:
    paths = subprocess.check_output(
        ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=ROOT,
    ).decode().split('\0')
    # Evidence files change when this recorder runs. Capture production and
    # validation sources separately from generated records and prose ledgers.
    included = ('crates/', 'validation/', 'tools/', 'dependencies/', 'examples/')
    top_level = {'Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', 'flake.nix', 'flake.lock'}
    hashes = {
        path: digest(ROOT / path)
        for path in sorted(set(paths))
        if path and (ROOT / path).is_file()
        and (path.startswith(included) or path in top_level)
        and not path.startswith('validation/evidence/')
        and not path.endswith('.md')
    }
    return {'git_head': command('git', 'rev-parse', 'HEAD'),
            'git_status': command('git', 'status', '--porcelain=v1'),
            'source_sha256': hashes}


def dependencies() -> dict:
    manifest = json.loads((ROOT / 'dependencies/manifest.json').read_text())
    result = {}
    for entry in manifest['dependencies']:
        directory = ROOT / entry['path']
        patch = subprocess.check_output(['git', 'diff', '--binary', 'HEAD'], cwd=directory)
        result[entry['name']] = {
            'head': command('git', 'rev-parse', 'HEAD', cwd=directory),
            'tracked_status': command('git', 'status', '--porcelain=v1', '--untracked-files=no', cwd=directory),
            'tracked_diff_sha256': hashlib.sha256(patch).hexdigest(),
            'bundled_patches': entry.get('patches', []),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('test_log', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, action='append', default=[])
    args = parser.parse_args()
    log = args.test_log.read_text()
    results = [tuple(map(int, row)) for row in re.findall(
        r'test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored', log)]
    if (not results or 'Doc-tests ninjars_symbolica' not in log
            or 'test result: FAILED' in log or any(failed or ignored for _, failed, ignored in results)):
        raise RuntimeError('a complete passing all-feature workspace log with ignored tests enabled is required')
    related = {}
    for path in args.evidence:
        report = json.loads(path.read_text())
        relative = str(path.resolve().relative_to(ROOT))
        related[relative] = {'sha256': digest(path), 'passed': report.get('passed'),
                             'checks': len(report.get('checks', []))}
    cpu_info = Path('/proc/cpuinfo')
    cpu_models = sorted(set(re.findall(r'^model name\s*:\s*(.+)$', cpu_info.read_text(), re.M))) if cpu_info.exists() else []
    # Explicit allowlist: never serialize credentials or the Symbolica license.
    settings = {key: os.environ.get(key) for key in (
        'RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS', 'CC', 'CXX', 'CFLAGS', 'CXXFLAGS',
        'OMP_NUM_THREADS', 'RAYON_NUM_THREADS', 'RUST_MIN_STACK',
    )}
    record = {
        'schema_version': 1, 'recorded_utc': datetime.now(timezone.utc).isoformat(),
        'workspace_tests': {'passed': sum(row[0] for row in results), 'failed': 0, 'ignored': 0,
                            'command': TEST_COMMAND, 'log_sha256': digest(args.test_log)},
        'checkout': source_snapshot(), 'dependencies': dependencies(),
        'rustc': command('rustc', '-Vv'),
        'host': {'system': platform.system(), 'architecture': platform.machine(), 'cpu_models': cpu_models,
                 'affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                 'load_average_at_recording': os.getloadavg() if hasattr(os, 'getloadavg') else None},
        'environment': settings, 'related_evidence': related,
        'performance_acceptance': False, 'full_goal_complete': False,
        'scope': 'Test checkpoint; host metadata is recorded after testing, not a timing comparison or CPU-frequency control.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + '\n')
    print(f"Recorded {record['workspace_tests']['passed']} passing tests: {args.output}")


if __name__ == '__main__':
    main()
