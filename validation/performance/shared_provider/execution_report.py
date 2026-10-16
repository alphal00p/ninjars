"""Validate and name one instrumented pass; never combine separate noisy runs."""
MASTER_KINDS = ['A0', 'B0', 'B1', 'B11', 'B111', 'C0', 'D0', 'DB0'] + [f'An{n}' for n in range(5)] + ['An_other']
RUST_NAMES = (['numerator.direct', 'numerator.mu', 'numerator.t3', 'numerator.t2', 'physical.prepare']
              + [f'{source}.{kind}' for source in ['shared-rust.backend', 'shared-original.backend', 'direct-rust.backend'] for kind in MASTER_KINDS]
              + ['shared-rust.abi', 'shared-original.abi'])
ORIGINAL_NAMES = MASTER_KINDS + ['B0+B1+B11', 'B0+B1+B11+B111', 'initialize-scale']


def _integer(value):
    if type(value) is not int or value < 0:
        raise ValueError('span cells must be nonnegative integer counts/nanoseconds')
    return value


def _spans(names, values):
    if len(names) != len(values) or any(len(row) != 3 for row in values):
        raise ValueError('incorrect span dimensions')
    return {name: dict(zip(['calls', 'rows', 'inclusive_ns'], map(_integer, row))) for name, row in zip(names, values)}


def validate_execution_record(record):
    """Return named spans and exact additive accounting; do not mutate input."""
    output = []
    entries = record['execution_records']
    if len(entries) != 2 or [entry['implementation'] for entry in entries] != ['original', 'rust']:
        raise ValueError('exactly one original and one Rust diagnostic call required')
    for entry in entries:
        cells = entry['rust_spans']
        if len(cells) != len(RUST_NAMES) * 3:
            raise ValueError('incorrect Rust span dimensions')
        rust = _spans(RUST_NAMES, [cells[i:i+3] for i in range(0, len(cells), 3)])
        original = _spans(ORIGINAL_NAMES, entry['original_backend_spans'])
        backend_records = [span for name, span in rust.items() if '.backend.' in name] + list(original.values())[:-1]
        backend = {column: sum(span[column] for span in backend_records) for column in ['calls', 'rows', 'inclusive_ns']}
        wire_records = [rust[name] for name in RUST_NAMES[-2:]]
        wire = {column: sum(span[column] for span in wire_records) for column in ['calls', 'rows', 'inclusive_ns']}
        shared_backend_ns = sum(span['inclusive_ns'] for name, span in rust.items() if name.startswith('shared-') and '.backend.' in name)
        numerator_ns = sum(rust[name]['inclusive_ns'] for name in RUST_NAMES[:4])
        preparation_provider_ns = rust['physical.prepare']['inclusive_ns']
        preparation = _spans(['prepare'], [entry['preparation_total_span']])['prepare']
        preparation_ns = preparation['inclusive_ns']
        if preparation_ns < preparation_provider_ns:
            raise ValueError('nested preparation provider exceeds full preparation span')
        if wire['inclusive_ns'] < shared_backend_ns:
            raise ValueError('nested backend exceeds its containing ABI span')
        lookup_ns = wire['inclusive_ns'] - shared_backend_ns
        total = _integer(entry['overall_ns'])
        partitioned = numerator_ns + preparation_ns + backend['inclusive_ns'] + lookup_ns
        if partitioned > total:
            raise ValueError('disjoint stage spans exceed overall interval')
        expected = {
            'numerator_provider_abi_ns': numerator_ns,
            'preparation_provider_ns': preparation_provider_ns,
            'preparation_total_ns': preparation_ns,
            'preparation_transport_ns': preparation_ns - preparation_provider_ns,
            'master_backend_ns': backend['inclusive_ns'],
            'master_lookup_transport_ns': lookup_ns,
            'remainder_ns': total - partitioned,
            'non_master_elapsed_ns': total - backend['inclusive_ns'],
            'reduction_excluding_master_and_preparation_ns': total - backend['inclusive_ns'] - preparation_ns,
        }
        for key, value in expected.items():
            if _integer(entry[key]) != value:
                raise ValueError(f'inconsistent same-pass accounting: {key}')
        if record['scope'] == 'shared-reduction' and any(backend.values()):
            raise ValueError('frozen replay must have exactly zero backend calls/rows/time')
        if record['scope'] == 'preparation' and (any(backend.values()) or any(wire.values()) or numerator_ns):
            raise ValueError('preparation unexpectedly performed reduction provider work')
        if record['scope'] in ['shared-live-complete', 'production-complete'] and not all(backend.values()):
            raise ValueError('complete evaluation must execute an actual master backend')
        # Source identity is a correctness condition, not merely a timer label.
        def totals(prefix):
            return {column: sum(span[column] for name, span in rust.items() if name.startswith(prefix))
                    for column in ['calls', 'rows', 'inclusive_ns']}
        groups = {source: totals(source + '.backend.') for source in ['shared-rust', 'shared-original', 'direct-rust']}
        groups['original-oneloop'] = {column: sum(span[column] for span in list(original.values())[:-1])
                                      for column in ['calls', 'rows', 'inclusive_ns']}
        expected_wire = 'shared-' + entry['implementation'] + '.abi'
        for name in RUST_NAMES[-2:]:
            should_call = record['scope'] in ['shared-reduction', 'shared-live-complete'] and name == expected_wire
            if should_call:
                if not rust[name]['calls'] or not rust[name]['rows']:
                    raise ValueError('missing expected shared provider lookup/transport')
            elif any(rust[name].values()):
                raise ValueError('unexpected shared provider source')
        expected_backend = None
        if record['scope'] == 'shared-live-complete':
            expected_backend = 'shared-' + entry['implementation']
        elif record['scope'] == 'production-complete':
            expected_backend = 'direct-rust' if entry['implementation'] == 'rust' else 'original-oneloop'
        for source, span in groups.items():
            if source == expected_backend:
                if not span['calls'] or not span['rows']:
                    raise ValueError('missing expected master backend source')
            elif any(span.values()):
                raise ValueError('unexpected master backend source')
        if expected_backend != 'original-oneloop' and any(original['initialize-scale'].values()):
            raise ValueError('unexpected original backend initialization')
        output.append({'implementation': entry['implementation'], 'rust_spans_by_family': rust,
                       'original_backend_spans_by_family': original,
                       'preparation_total': preparation, 'master_backend_totals': backend, 'shared_master_abi_totals_inclusive': wire,
                       'additive_ns': dict(overall=total, numerator=numerator_ns, preparation=preparation_ns,
                                           actual_backend=backend['inclusive_ns'], shared_abi_excluding_backend=lookup_ns,
                                           remainder=total-partitioned)})
    return output
