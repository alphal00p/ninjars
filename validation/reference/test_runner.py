#!/usr/bin/env python3
"""Check reference protocol framing, known masters, and invalid input handling."""
import json
import math
import pathlib
import subprocess
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
EXECUTABLE = ROOT / "target/reference/ninja-reference"


class ReferenceProtocol(unittest.TestCase):
    def test_bubble_and_triangle(self):
        result = subprocess.run([EXECUTABLE],
            input=(ROOT / "validation/reference/smoke.inputs").read_text(),
            capture_output=True, text=True, check=True)
        values = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
        self.assertEqual(len(values), 2)
        for value in values:
            self.assertEqual(value["status"], 0)
            self.assertGreater(value["nanoseconds_per_evaluation"], 0)
            self.assertEqual(value["rational"], [0, 0])
        self.assertTrue(math.isclose(values[0]["epsilon"][0][0], -0.1520447048200203, abs_tol=1e-13))
        self.assertEqual(values[0]["epsilon"][1:], [[1, 0], [0, 0]])
        self.assertTrue(math.isclose(values[1]["epsilon"][0][0], -0.37747010458616526, abs_tol=1e-13))
        self.assertEqual(values[1]["epsilon"][1:], [[0, 0], [0, 0]])

    def test_invalid_rank_is_rejected(self):
        result = subprocess.run([EXECUTABLE], input="2 4 1 1\n", capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)

    def test_truncated_sample_is_rejected(self):
        result = subprocess.run([EXECUTABLE], input="2 0 1 1\n0 0\n", capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
