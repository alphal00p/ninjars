import copy
import unittest

from execution_report import validate_execution_record


def record(scope='shared-reduction'):
    entries = []
    for implementation in ['original', 'rust']:
        spans = [[0, 0, 0] for _ in range(49)]
        spans[0] = [2, 2, 10]
        spans[47 if implementation == 'rust' else 48] = [3, 8, 40]
        backend = 0
        if scope == 'shared-live-complete':
            spans[11 if implementation == 'rust' else 25] = [1, 8, 30]
            backend = 30
        prep = 20 if scope == 'shared-live-complete' else 0
        if prep:
            spans[4] = [1, 8, 15]
        entries.append(dict(implementation=implementation, overall_ns=100,
                            numerator_provider_abi_ns=10, preparation_provider_ns=15 if prep else 0,
                            preparation_total_ns=prep, preparation_transport_ns=5 if prep else 0,
                            master_backend_ns=backend, master_lookup_transport_ns=40-backend,
                            remainder_ns=50-prep, non_master_elapsed_ns=100-backend,
                            reduction_excluding_master_and_preparation_ns=100-backend-prep,
                            rust_spans=[cell for span in spans for cell in span],
                            preparation_total_span=[1, 8, prep] if prep else [0, 0, 0],
                            original_backend_spans=[[0, 0, 0] for _ in range(17)]))
    return dict(scope=scope, execution_records=entries)


class ExecutionAccounting(unittest.TestCase):
    def test_frozen_lookup_is_not_backend_execution(self):
        source = record()
        original = copy.deepcopy(source)
        result = validate_execution_record(source)
        self.assertEqual(source, original)
        for item in result:
            self.assertEqual(item['master_backend_totals'], dict(calls=0, rows=0, inclusive_ns=0))
            self.assertEqual(item['shared_master_abi_totals_inclusive'], dict(calls=3, rows=8, inclusive_ns=40))
            self.assertEqual(item['additive_ns'], dict(overall=100, numerator=10, preparation=0,
                             actual_backend=0, shared_abi_excluding_backend=40, remainder=50))

    def test_nested_backend_and_preparation_are_subtracted_once(self):
        result = validate_execution_record(record('shared-live-complete'))
        for item in result:
            self.assertEqual(item['master_backend_totals'], dict(calls=1, rows=8, inclusive_ns=30))
            self.assertEqual(item['additive_ns'], dict(overall=100, numerator=10, preparation=20,
                             actual_backend=30, shared_abi_excluding_backend=10, remainder=30))

    def test_rejects_zero_duration_backend_call_in_frozen_scope(self):
        source = record()
        source['execution_records'][0]['rust_spans'][25*3] = 1
        with self.assertRaisesRegex(ValueError, 'exactly zero backend'):
            validate_execution_record(source)

    def test_rejects_invalid_dimensions_values_and_duplicate_implementation(self):
        for mutate in [lambda r: r['execution_records'][0]['rust_spans'].pop(),
                       lambda r: r['execution_records'][0]['rust_spans'].__setitem__(0, True),
                       lambda r: r['execution_records'][1].__setitem__('implementation', 'original'),
                       lambda r: r['execution_records'][0].__setitem__('overall_ns', -1)]:
            source = record()
            mutate(source)
            with self.assertRaises(ValueError):
                validate_execution_record(source)

    def test_rejects_accounting_and_nesting_errors(self):
        for mutate in [lambda r: r['execution_records'][0].__setitem__('remainder_ns', 51),
                       lambda r: r['execution_records'][0]['rust_spans'].__setitem__(25*3+2, 41),
                       lambda r: r['execution_records'][0]['preparation_total_span'].__setitem__(2, 1),
                       lambda r: r['execution_records'][0].__setitem__('overall_ns', 1)]:
            source = record('shared-live-complete')
            mutate(source)
            with self.assertRaises(ValueError):
                validate_execution_record(source)

    def test_rejects_correct_totals_from_wrong_backend_or_wire_source(self):
        source = record('shared-live-complete')
        spans = source['execution_records'][0]['rust_spans']
        spans[11*3:11*3+3], spans[25*3:25*3+3] = spans[25*3:25*3+3], spans[11*3:11*3+3]
        with self.assertRaisesRegex(ValueError, 'backend source'):
            validate_execution_record(source)
        source = record()
        spans = source['execution_records'][0]['rust_spans']
        spans[47*3:47*3+3], spans[48*3:48*3+3] = spans[48*3:48*3+3], spans[47*3:47*3+3]
        with self.assertRaisesRegex(ValueError, 'provider source'):
            validate_execution_record(source)

    def test_production_sources_are_disjoint_and_original_init_is_detail(self):
        source = record('shared-live-complete')
        source['scope'] = 'production-complete'
        for entry in source['execution_records']:
            entry['rust_spans'][5*3:] = [0] * (44*3)
            entry['master_lookup_transport_ns'] = 0
            entry['remainder_ns'] = 40
            if entry['implementation'] == 'original':
                entry['original_backend_spans'][6] = [1, 8, 30]
                entry['original_backend_spans'][-1] = [1, 0, 3]
            else:
                entry['rust_spans'][39*3:39*3+3] = [1, 8, 30]
        self.assertEqual(len(validate_execution_record(source)), 2)
        wrong = copy.deepcopy(source)
        wrong['execution_records'][1]['rust_spans'][39*3:39*3+3] = [0, 0, 0]
        wrong['execution_records'][1]['original_backend_spans'][6] = [1, 8, 30]
        with self.assertRaisesRegex(ValueError, 'backend source'):
            validate_execution_record(wrong)

    def test_complete_scope_requires_real_backend_calls(self):
        source = record()
        source['scope'] = 'production-complete'
        with self.assertRaisesRegex(ValueError, 'actual master backend'):
            validate_execution_record(source)


if __name__ == '__main__':
    unittest.main()
