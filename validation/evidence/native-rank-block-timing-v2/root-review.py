#!/usr/bin/env python3
"""Verify and summarize the complete, preselected rank/block timing run.

Run only after the driver exits successfully. This script does not run or
select benchmarks, discard blocks, or pool process confidence intervals.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True

SOURCE = Path('/tmp/ninjars-rank-block-timing-v2')
DEST = Path('/shared/external/ninja/validation/evidence/native-rank-block-timing-v2')
WIDTHS = (1, 8, 31, 32, 128, 1024)
SETTINGS = ('native-whole', 'native-fixed32', 'native-rank-whole', 'native-rank-fixed32')


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def strict(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result

    def invalid(value):
        raise ValueError('nonfinite JSON: ' + value)

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def read(name):
    return strict((SOURCE / name).read_text())


def immutable(record):
    record = dict(record)
    record['loader'] = {key: value for key, value in record['loader'].items()
                        if key not in ('ldd', 'readelf_dynamic')}
    return record


def main():
    require(not DEST.exists(), 'refusing to replace existing review')
    summary = read('summary.json')
    require(summary['success'] is True and summary['diagnostic_only'] is True
            and summary['performance_acceptance'] is False, 'run incomplete or scope changed')
    archived = read('archive-sha256.json')
    files = {path.name for path in SOURCE.iterdir() if path.is_file()}
    require(set(archived) == files - {'archive-sha256.json'}, 'incomplete evidence hash map')
    require(all(sha(SOURCE / name) == digest for name, digest in archived.items()),
            'evidence hash changed')
    preflight, postflight = read('preflight.json'), read('postflight.json')
    require(immutable(preflight) == immutable(postflight),
            'pre/postflight immutable identities differ')
    require(sha(SOURCE / 'driver.py') == preflight['driver_sha256'], 'copied driver identity differs')
    spec = importlib.util.spec_from_file_location('retained_rank_block_driver', SOURCE / 'driver.py')
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    statistics_path = driver.HERE / 'timing_statistics.py'
    require(sha(statistics_path) == preflight['source_sha256'][str(statistics_path.relative_to(driver.ROOT))],
            'paired summary implementation differs')
    sys.path.insert(0, str(driver.HERE))
    from timing_statistics import summarize_record
    commands, schedule = read('commands.json'), read('schedule.json')
    require(schedule == driver.schedule(), 'preselected schedule differs')
    require(len(commands) == len(schedule) == summary['processes'] == 96, 'process count differs')
    by_case, records, blocks, raw_blocks, short, results = {}, 0, 0, 0, [], []
    minimum = None
    maximum_error = 0.0
    for index, (command, job) in enumerate(zip(commands, schedule)):
        require(command['index'] == index and command['job'] == job
                and command['status'] == 'complete' and command['returncode'] == 0,
                'command order/status differs')
        name = f'{index:03d}-s{job["sweep"]}-w{job["width"]}-{job["setting"]}'
        require(command['stdout_sha256'] == archived[name + '.stdout']
                and command['stderr_sha256'] == archived[name + '.stderr'],
                'command/output provenance differs')
        raw = [strict(line) for line in (SOURCE / (name + '.stdout')).read_text().splitlines()
               if line.startswith('{')]
        require(len(raw) == 1, 'raw report count differs')
        report = read(name + '.validated.json')
        results.append({'job': job, 'report': report})
        require(len(raw[0]['records']) == len(report['records']) == 2, 'record count differs')
        require(all(type(report[key]) is int and report[key] == 0
                    for key in ('frozen_backend_call_delta', 'frozen_backend_row_delta', 'union_misses')),
                'actual master computation or frozen-union miss')
        for original, row in zip(raw[0]['records'], report['records']):
            require(all(row[key] == value for key, value in original.items()),
                    'validated report changed a raw field')
            require(row['width'] == job['width'], 'row width differs')
            require(row['scope'] == 'shared-reduction', 'timing scope differs')
            pairs = row['raw_block_nanoseconds']
            require(len(pairs) == len(row['samples']) == 8
                    and all(len(pair) == 2 for pair in pairs), 'paired block shape differs')
            require(all(pair[0] / job['rounds'] == sample['original_ns']
                        and pair[1] / job['rounds'] == sample['rust_ns']
                        for pair, sample in zip(pairs, row['samples'])), 'normalization differs')
            expected_ratio = statistics.median(sample['rust_ns'] / sample['original_ns']
                                               for sample in row['samples'])
            require(row['summary']['rust_over_original']['median'] == expected_ratio,
                    'paired median differs')
            require(summarize_record(row) == row['summary'], 'deterministic paired summary differs')
            require(all(row[name][cell] == 0 for name in ('original_counter_deltas', 'rust_counter_deltas')
                        for cell in (8, 9, 10, 15, 16, 17, 18, 19)),
                    'scope1 includes preparation or actual master execution')
            key = (row['width'], row['workload'], job['setting'])
            by_case.setdefault(key, []).append({
                'sweep': job['sweep'], 'validated_report': name + '.validated.json',
                'ratio': row['summary']['rust_over_original'],
                'timings': row['summary']['timings'],
                'minimum_observed_block_ns': row['minimum_observed_block_ns'],
            })
            records += 1
            blocks += len(row['samples'])
            raw_blocks += sum(len(pair) for pair in pairs)
            floor = row['minimum_observed_block_ns']
            require(floor == min(min(pair) for pair in pairs), 'record floor differs')
            minimum = floor if minimum is None else min(minimum, floor)
            maximum_error = max(maximum_error, row['max_scaled_error'])
            if floor < 20_000_000:
                short.append({'job': job, 'workload': row['workload'],
                              'minimum_observed_block_ns': floor})
            # The driver validates exact typed counters against its fixed
            # source-derived arrays. Retain its complete counter payloads in
            # the archive; no master duration is inferred from lookup traffic.
    require((records, blocks, raw_blocks) == (192, 1536, 3072), 'observed block counts differ')
    require(short == summary['short_records'] and bool(short) == summary['pilot'],
            'duration-floor classification differs')
    require(driver.comparisons(results) == summary['contrasts'], 'retained cross-setting contrasts differ')
    require(set(by_case) == {(w, load, mode) for w in WIDTHS
                            for load in ('repeated', 'changing') for mode in SETTINGS},
            'case coverage differs')
    cases = []
    for key in sorted(by_case):
        sweeps = by_case[key]
        require([x['sweep'] for x in sweeps] == list(range(4)), 'sweep coverage differs')
        ratios = [x['ratio']['median'] for x in sweeps]
        median = statistics.median(ratios)
        cases.append({'width': key[0], 'workload': key[1], 'setting': key[2],
                      'four_processes': sweeps, 'median_of_process_paired_ratios': median,
                      'mad_of_process_paired_ratios': statistics.median(abs(x - median) for x in ratios),
                      'pooled_confidence_interval': None})
    result = {
        'source_directory': str(SOURCE), 'source_archive_hash_map_sha256': sha(SOURCE / 'archive-sha256.json'),
        'review_script_sha256': sha(Path(__file__)), 'complete_run_verified': True,
        'source_dependency_artifact_loader_identity_unchanged': True,
        'deterministic_driver_summaries_and_contrasts_reproduced': True,
        'processes': 96, 'records': records, 'paired_blocks': blocks, 'individual_blocks': raw_blocks,
        'minimum_individual_block_ns': minimum, 'pilot': bool(short),
        'maximum_scaled_laurent_error': maximum_error, 'actual_master_calls_and_rows_zero': True,
        'duration_suitable': not short, 'performance_acceptance': False, 'diagnostic_only': True,
        'cases': cases, 'contrasts': summary['contrasts'],
        'scope': 'Shared-provider reduction including callbacks, frozen master lookup/contraction and staging; actual master execution and physical preparation excluded.',
        'uncertainty': 'Each process retains its paired-block bootstrap interval. Median/MAD across four processes is descriptive only; no pooled interval. Shared host not isolated.',
        'limitation': 'Kernel configuration does not count rank-guard/native-block hits. Original/Rust T2 heads differ (40/60 per point by source audit).',
    }
    DEST.mkdir(parents=True)
    (DEST / 'root-review.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: result[key] for key in ('processes', 'records', 'paired_blocks',
        'individual_blocks', 'minimum_individual_block_ns', 'pilot')}, indent=2))


if __name__ == '__main__':
    main()
