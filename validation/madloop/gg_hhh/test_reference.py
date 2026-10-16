"""Mutation controls for archived reference integrity and raw-grid coverage."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from collect import validate_graph
from reference_io import archive_json,read_bytes,read_json

ROOT=Path(__file__).resolve().parents[3]


class ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.archive=ROOT/'validation/evidence/gg-hhh-madloop.json.gz'
        cls.reference=read_json(cls.archive)
        cls.graph=cls.reference['graphs'][0]

    def test_archive_exact_bytes_and_metadata(self):
        metadata=read_json(ROOT/'validation/evidence/gg-hhh-reference-archive.json')
        raw=read_bytes(self.archive);compressed=self.archive.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),metadata['uncompressed_sha256'])
        self.assertEqual(hashlib.sha256(compressed).hexdigest(),metadata['compressed_sha256'])
        self.assertEqual(len(raw),metadata['uncompressed_bytes'])
        self.assertEqual(len(compressed),metadata['compressed_bytes'])
        self.assertEqual(compressed[3]&8,0) # gzip has no filename metadata
        self.assertEqual(compressed[4:8],b'\0'*4)

    def test_deterministic_exact_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'source.json'
            payload=b'{ "exact": "12345678901234567890/7", "value": [1.25, -2.5] }\n'
            path.write_bytes(payload)
            a=Path(directory)/'first.json.gz';b=Path(directory)/'other-name.json.gz'
            self.assertEqual(archive_json(path,a),archive_json(path,b))
            self.assertEqual(a.read_bytes(),b.read_bytes())
            self.assertEqual(read_bytes(a),payload)
            self.assertEqual(read_json(a),json.loads(payload))

    def test_archive_full_coverage(self):
        self.assertEqual(len(self.reference['graphs']),50)
        self.assertEqual(self.reference['expected_raw_checks'],28000)
        self.assertEqual(self.reference['internal_raw_wrapper_checks'],56000)
        for graph in self.reference['graphs']:validate_graph(graph)

    def reject(self,mutate):
        graph=copy.deepcopy(self.graph);mutate(graph)
        with self.assertRaises(AssertionError):validate_graph(graph)

    def test_missing_state_rejected(self):
        self.reject(lambda g:g['cases'].pop())

    def test_duplicate_case_rejected(self):
        self.reject(lambda g:g['cases'].__setitem__(0,g['cases'][1]))

    def test_duplicate_grid_point_rejected(self):
        self.reject(lambda g:g['cases'][0]['numerator_samples'].__setitem__(0,g['cases'][0]['numerator_samples'][1]))

    def test_nonfinite_numerator_rejected(self):
        self.reject(lambda g:g['cases'][0]['numerator_samples'][0]['raw'].__setitem__(0,float('nan')))

    def test_unstable_physical_reference_rejected(self):
        self.reject(lambda g:g['cases'][0].__setitem__('stable',False))

    def test_merged_graph_rejected(self):
        self.reject(lambda g:g['graph_metadata'].__setitem__('multiplier',2))

    def test_corrupt_gzip_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bad.json.gz';path.write_bytes(self.archive.read_bytes()[:-20])
            with self.assertRaises((EOFError,gzip.BadGzipFile)):read_json(path)


if __name__=='__main__':unittest.main()
