"""Paired timing summaries; these do not determine performance acceptance."""
import math
import random
import statistics
from collections import Counter


ORDERS = ('original-first', 'rust-first')


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _positive_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a finite positive number')
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError(f'{name} must be finite') from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be a finite positive number')
    return value


def _distribution(values):
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError('timing conversion overflowed or underflowed')
    median = statistics.median(values)
    if not math.isfinite(median) or median <= 0:
        raise ValueError('timing median overflowed or underflowed')
    return {
        'median': median,
        'mad': statistics.median(abs(value - median) for value in values),
    }


def _quantile(sorted_values, probability):
    position = (len(sorted_values) - 1) * probability
    lo = math.floor(position)
    hi = math.ceil(position)
    weight = position - lo
    return sorted_values[lo] * (1 - weight) + sorted_values[hi] * weight


def summarize_record(record, *, bootstrap_samples=10000, seed=1729):
    """Summarize paired blocks already normalized to ns per complete batch.

    Required record fields are width, rounds and samples. Each sample has exactly
    original_ns, rust_ns and order. Both process orders must occur equally often,
    at least twice. Resampling keeps each original/Rust block pair together and
    preserves the observed count in each order stratum. The reported ratio is
    median(rust_ns/original_ns), so values above one mean Rust took longer.

    The percentile interval describes uncertainty over observed paired blocks;
    it establishes neither host isolation nor a performance acceptance result.
    """
    if not isinstance(record, dict):
        raise ValueError('record must be an object')
    try:
        width = _positive_integer(record['width'], 'width')
        rounds = _positive_integer(record['rounds'], 'rounds')
        samples = record['samples']
    except KeyError as error:
        raise ValueError(f'missing timing field {error.args[0]}') from error
    bootstrap_samples = _positive_integer(bootstrap_samples, 'bootstrap_samples')
    if bootstrap_samples < 2:
        raise ValueError('at least two bootstrap samples are required')
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError('seed must be an integer')
    if not isinstance(samples, list) or not samples:
        raise ValueError('samples must be a nonempty paired list')
    times = {'original': [], 'rust': []}
    ratios = {order: [] for order in ORDERS}
    counts = Counter()
    for sample in samples:
        if not isinstance(sample, dict) or set(sample) != {'original_ns', 'rust_ns', 'order'}:
            raise ValueError('each sample must contain exactly original_ns, rust_ns and order')
        order = sample['order']
        if not isinstance(order, str) or order not in ORDERS:
            raise ValueError('unknown process order')
        original = _positive_number(sample['original_ns'], 'original_ns')
        rust = _positive_number(sample['rust_ns'], 'rust_ns')
        ratio = rust / original
        if not math.isfinite(ratio) or ratio <= 0:
            raise ValueError('paired timing ratio overflowed or underflowed')
        times['original'].append(original)
        times['rust'].append(rust)
        ratios[order].append(ratio)
        counts[order] += 1
    if counts[ORDERS[0]] != counts[ORDERS[1]] or counts[ORDERS[0]] < 2:
        raise ValueError('both process orders need equal counts and at least two pairs each')
    rng = random.Random(seed)
    bootstrapped = sorted(
        statistics.median([
            rng.choice(ratios[order])
            for order in ORDERS for _ in ratios[order]
        ])
        for _ in range(bootstrap_samples)
    )
    if any(not math.isfinite(value) or value <= 0 for value in bootstrapped):
        raise ValueError('bootstrap median overflowed or underflowed')
    result = {
        'width': width,
        'rounds_per_block': rounds,
        'paired_blocks': len(samples),
        'counts_by_order': dict(counts),
        'timings': {},
        'rust_over_original': {
            **_distribution([ratio for order in ORDERS for ratio in ratios[order]]),
            'median_by_order': {order: statistics.median(ratios[order]) for order in ORDERS},
            'confidence_level': 0.95,
            'confidence_interval': [_quantile(bootstrapped, 0.025), _quantile(bootstrapped, 0.975)],
            'bootstrap_samples': bootstrap_samples,
            'bootstrap_seed': seed,
            'method': 'percentile bootstrap of paired median ratios, stratified by process order',
        },
        'mad_definition': 'unscaled median absolute deviation from the sample median',
        'uncertainty_scope': 'Observed paired timing blocks only; does not establish host isolation or performance acceptance.',
    }
    for implementation, values in times.items():
        try:
            per_point = [value / width for value in values]
            throughput = [width * 1e9 / value for value in values]
        except OverflowError as error:
            raise ValueError('timing conversion overflowed') from error
        result['timings'][implementation] = {
            'ns_per_batch': _distribution(values),
            'ns_per_point': _distribution(per_point),
            'points_per_second': _distribution(throughput),
        }
    return result
