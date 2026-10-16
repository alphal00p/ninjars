#!/usr/bin/env python3
"""Collect unchanged isolated MadLoop probes for the independent Rust matcher."""
import argparse
import hashlib
import json
from pathlib import Path
import re

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--exports-dir', type=Path, required=True)
parser.add_argument('--probes-dir', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
exports = args.exports_dir.resolve()
probes = args.probes_dir.resolve()
manifest_path = exports/'graphs.json'
manifest = json.loads(manifest_path.read_text())
assert manifest['original_model_unchanged']
assert len(manifest['graphs']) == 6

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

graphs = []
checks = 0
for graph in manifest['graphs']:
    source = exports/graph['directory']/'SubProcesses/P0_aa_aa/loop_num.f'
    seeds = re.findall(r'\bCALL (LCUT_AF|LCUT_F)\(', source.read_text())
    assert len(seeds) == 1, seeds
    path = probes/f"probe-{graph['filter_id']}/reference.json"
    reference = json.loads(path.read_text())
    assert reference['graph_filter_id'] == graph['filter_id']
    assert reference['graph_metadata']['multiplier'] == 1
    reference['original_reference_sha256'] = digest(path)
    reference['fermion_seed'] = seeds[0]
    reference['ordered_external_legs'] = reference['wavefunction_order']
    reference['seed_source_sha256'] = digest(source)
    for case in reference['cases']:
        assert len(case['numerator_samples']) == 70
        if case['ctmode'] == 4:
            checks += 70
    graphs.append(reference)
result = {
    'schema_version': 1, 'expected_raw_checks': checks, 'graphs': graphs,
    'scope': 'Six isolated top-quark four-photon graphs on 70-point affine quartic grids. Raw matching selects fixed mode4 rows; mode1 repetitions remain separate.',
    'input_precision': 'Reference wavefunctions and GC_2 are binary64 inputs. Quadruple reduction promotes those inputs; this is not an independent high-precision physical-input oracle.',
    'generation': {
        'manifest_sha256': digest(manifest_path),
        'model_patch': manifest['model_patch'],
        'original_model_unchanged': manifest['original_model_unchanged'],
        'source_sha256': manifest['source_sha256'],
        'commands': manifest['commands'],
        'collector_sha256': digest(Path(__file__)),
    },
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
print(args.output)
