#!/usr/bin/env python3
"""Archive/verify a complete ggHHH matrix without changing any numerical rows."""
import argparse
import hashlib
import json
from pathlib import Path

from reference_io import archive_json, read_bytes, read_json


def identity(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def coverage(matrix):
    expected = {(g, p, m, lane) for g in range(50)
                for p in ('f64', 'double-float', 'float')
                for m in ('expansion', 'sampling') for lane in range(8)}
    keys = [(r['orientation'], r['precision'], r['mu_extraction'], r['lane'])
            for r in matrix['checks']]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError('matrix has missing, duplicate or unexpected graph/state rows')
    return {
        'rows': len(keys),
        'physical_rows': sum(r['lane'] < 4 for r in matrix['checks']),
        'diagnostic_rows': sum(r['lane'] >= 4 for r in matrix['checks']),
        'runtime_source_sha256_before': matrix['provenance']['source_sha256'],
        'runtime_changed_sources_after': matrix['changed_sources'],
        'runtime_guard_passed': not matrix['changed_sources'],
        'cli_binary_sha256': matrix['provenance']['binary']['sha256'],
        'performance_acceptance': False,
    }


def verify(archive, manifest):
    raw = read_bytes(archive)
    checks = {
        'uncompressed_sha256': hashlib.sha256(raw).hexdigest(),
        'compressed_sha256': identity(archive),
        'uncompressed_bytes': len(raw),
        'compressed_bytes': Path(archive).stat().st_size,
    }
    if any(manifest[key] != value for key, value in checks.items()):
        raise ValueError('matrix archive identity mismatch')
    matrix = json.loads(raw)
    if coverage(matrix) != manifest['coverage_and_runtime_guard']:
        raise ValueError('matrix archive coverage/runtime metadata mismatch')
    return matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    if args.archive.exists() or args.manifest.exists():
        parser.error('archive and manifest must be new files')
    matrix = read_json(args.input)
    metadata = coverage(matrix)
    manifest = {
        'schema_version': 1,
        **archive_json(args.input, args.archive),
        'coverage_and_runtime_guard': metadata,
        'archiver_sha256': identity(Path(__file__)),
        'reader_sha256': identity(Path(__file__).with_name('reference_io.py')),
    }
    verify(args.archive, manifest)
    args.manifest.write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    print(args.manifest)


if __name__ == '__main__':
    main()
