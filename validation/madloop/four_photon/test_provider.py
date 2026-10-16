#!/usr/bin/env python3
"""Guard against accepting incomplete or mismatched compiled-provider evidence."""
import copy
import json
from pathlib import Path
import unittest

from check_provider import validate

ROOT = Path(__file__).resolve().parents[3]


class ProviderEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matrix = json.loads((ROOT/'validation/evidence/four-photon-symjit-o2-matrix.json').read_text())
        cls.poles = json.loads((ROOT/'validation/evidence/four-photon-uv.json').read_text())

    def test_retained_matrix_passes(self):
        self.assertTrue(validate(self.matrix, self.poles)['passed'])

    def test_corrupted_controls_fail(self):
        mutations = [
            lambda m: m['checks'].pop(),
            lambda m: m.__setitem__('numerator_evaluator', 'symbolica'),
            lambda m: m['checks'][0].__setitem__('working_bits', 106),
            lambda m: m['checks'][0]['comparisons'][1]['scaled_errors'].__setitem__(0, 1e-5),
            lambda m: m['checks'][0]['result']['values'][1].__setitem__(0, '99'),
            lambda m: m['controls'][0]['errors'].__setitem__(0, '0.01'),
            lambda m: m['provenance']['commands'][0].__setitem__('returncode', 1),
            lambda m: m['changed_sources'].append('modified source'),
            lambda m: m['preparations'][0].__setitem__('sha256', 'different prepared numerator'),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                matrix = copy.deepcopy(self.matrix)
                mutate(matrix)
                self.assertFalse(validate(matrix, self.poles)['passed'])


if __name__ == '__main__':
    unittest.main()
