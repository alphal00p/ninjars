"""Checks on the frozen changing-phase-space boundary and its reference output."""
import copy
import json
from pathlib import Path
import unittest
from changing_points import validate_point

ROOT=Path(__file__).resolve().parents[3]


class ChangingPointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pool=json.loads((ROOT/'validation/evidence/gg-hhh-changing-pool.json').read_text())
        cls.reference=json.loads((ROOT/'validation/evidence/gg-hhh-changing-madloop.json').read_text())

    def test_three_genuine_phase_space_points_at_fixed_helicity(self):
        points=self.pool['points']
        self.assertEqual(len(points),3)
        self.assertEqual([p['momenta'][0][0]*2 for p in points],[450,525,600])
        for point in points:validate_point(point)

    def test_complete_fixed_mode_reference_and_identical_inputs(self):
        points={p['id']:p for p in self.pool['points']}
        cases=self.reference['cases']
        self.assertEqual(len(cases),6)
        self.assertEqual({(c['point_id'],c['ctmode']) for c in cases},{(p,m) for p in points for m in (1,4)})
        self.assertEqual(self.reference['internal_madloop_raw_wrapper_checks'],420)
        for case in cases:
            self.assertTrue(case['stable'])
            self.assertEqual(len(case['numerator_samples']),70)
            self.assertEqual([case['external_momenta'][str(i)] for i in range(1,6)],points[case['point_id']]['momenta'])

    def test_off_shell_point_rejected(self):
        point=copy.deepcopy(self.pool['points'][0]);point['momenta'][2][0]+=1
        with self.assertRaises(AssertionError):validate_point(point)

    def test_wrong_model_point_rejected(self):
        point=copy.deepcopy(self.pool['points'][0]);point['model'][0]=172
        with self.assertRaises(AssertionError):validate_point(point)

    def test_changing_helicity_is_not_the_fixed_helicity_workload(self):
        point=copy.deepcopy(self.pool['points'][0]);point['helicities'][1]=-1
        with self.assertRaises(AssertionError):validate_point(point)


if __name__=='__main__':unittest.main()
