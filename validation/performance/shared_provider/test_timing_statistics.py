"""Synthetic unit controls for paired timing statistics; no timing execution."""
import copy
import math
import unittest

from timing_statistics import summarize_record


def record():
    return {
        'width': 8,
        'rounds': 100,
        'samples': [
            {'original_ns': value, 'rust_ns': 2 * value, 'order': order}
            for order in ('original-first', 'rust-first') for value in (80, 160)
        ],
    }


class TimingStatistics(unittest.TestCase):
    def test_constant_ratio_and_units(self):
        result = summarize_record(record(), bootstrap_samples=100)
        ratio = result['rust_over_original']
        self.assertEqual((ratio['median'], ratio['mad'], ratio['confidence_interval']), (2, 0, [2, 2]))
        original = result['timings']['original']
        self.assertEqual(original['ns_per_batch'], {'median': 120, 'mad': 40})
        self.assertEqual(original['ns_per_point'], {'median': 15, 'mad': 5})
        self.assertEqual(original['points_per_second'], {'median': 75_000_000, 'mad': 25_000_000})
        self.assertEqual(result['counts_by_order'], {'original-first': 2, 'rust-first': 2})

    def test_paired_ratio_is_not_ratio_of_medians(self):
        data = record()
        for sample, a, b in zip(data['samples'], (1, 2, 10, 100), (2, 8, 30, 100)):
            sample.update(original_ns=a, rust_ns=b)
        result = summarize_record(data, bootstrap_samples=100)
        self.assertEqual(result['rust_over_original']['median'], 2.5)
        self.assertNotEqual(result['rust_over_original']['median'], 19 / 6)

    def test_seed_and_input_are_stable(self):
        data = record()
        data['samples'][1]['rust_ns'] = 480
        before = copy.deepcopy(data)
        first = summarize_record(data, bootstrap_samples=1000, seed=27)
        self.assertEqual(first, summarize_record(data, bootstrap_samples=1000, seed=27))
        self.assertEqual(data, before)
        low, high = first['rust_over_original']['confidence_interval']
        self.assertLessEqual(low, first['rust_over_original']['median'])
        self.assertGreaterEqual(high, first['rust_over_original']['median'])

    def test_reject_invalid_times(self):
        for value in (0, -1, math.inf, -math.inf, math.nan, True, '3', None, 10**1000):
            for name in ('original_ns', 'rust_ns'):
                with self.subTest(value=str(value), name=name):
                    data = record()
                    data['samples'][0][name] = value
                    with self.assertRaises(ValueError):
                        summarize_record(data, bootstrap_samples=10)

    def test_reject_incomplete_and_unbalanced_pairs(self):
        for change in (
            lambda data: data['samples'][0].pop('rust_ns'),
            lambda data: data['samples'].pop(),
            lambda data: data.update(samples=[]),
            lambda data: data.update(samples=data['samples'][::2]),
            lambda data: data['samples'][0].update(order='parallel'),
            lambda data: data['samples'][0].update(extra=1),
            lambda data: data.update(samples={'original_ns': [], 'rust_ns': []}),
        ):
            data = record()
            change(data)
            with self.assertRaises(ValueError):
                summarize_record(data, bootstrap_samples=10)

    def test_reject_invalid_dimensions(self):
        for name in ('width', 'rounds'):
            for value in (0, -1, True, 2.5, None):
                data = record()
                data[name] = value
                with self.assertRaises(ValueError):
                    summarize_record(data, bootstrap_samples=10)
        data = record()
        del data['width']
        with self.assertRaises(ValueError):
            summarize_record(data)

    def test_reject_invalid_bootstrap_controls(self):
        for value in (0, 1, True, 1.5):
            with self.assertRaises(ValueError):
                summarize_record(record(), bootstrap_samples=value)
        with self.assertRaises(ValueError):
            summarize_record(record(), seed=True)

    def test_reject_conversion_extremes(self):
        for original, rust in ((1e-300, 1e300), (1e300, 1e-300), (5e-324, 5e-324), (1e308, 1e308), (1, 1e308)):
            data = record()
            for sample in data['samples']:
                sample.update(original_ns=original, rust_ns=rust)
            with self.assertRaises(ValueError):
                summarize_record(data, bootstrap_samples=10)


if __name__ == '__main__':
    unittest.main()
