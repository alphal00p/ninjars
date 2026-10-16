#!/usr/bin/env python3
"""Collect isolated gg→HHH graph probes, requiring full fixed-state coverage."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

STATES={'pp','pm','mp','mm','generic_complex','ward0','ward1','ward_both'}


def validate_graph(reference):
    cases=reference['cases']
    assert len(cases)==16
    assert {(c['ctmode'],c['state']) for c in cases}=={(m,s) for m in (1,4) for s in STATES}
    assert reference['graph_metadata']['multiplier']==1
    for case in cases:
        rows=case['numerator_samples']
        assert len(rows)==70 and {r['id'] for r in rows}==set(range(1,71))
        assert len({json.dumps(r['q']) for r in rows})==70
        assert all(len(r['q'])==4 and len(r['raw'])==2 for r in rows)
        assert all(math.isfinite(v) for r in rows for pair in r['q']+[r['raw']] for v in pair)
        assert len(case['external_momenta'])==5 and len(case['external_polarizations'])==2
        if case['state'] in {'pp','pm','mp','mm'}:
            assert case['stable'] and len(case['raw_laurent'])==3
            assert all(math.isfinite(v) for pair in case['raw_laurent'] for v in pair)


def collect(exports,probes,partial=False):
    manifest_path=exports/'graphs.json';manifest=json.loads(manifest_path.read_text());graphs=[]
    assert manifest['original_model_unchanged']
    assert len(manifest['graphs'])==50
    assert len({g['filter_id'] for g in manifest['graphs']})==50
    assert Counter(g['loop_propagators'] for g in manifest['graphs'])=={3:8,4:18,5:24}
    for graph in manifest['graphs']:
        path=probes/f'probe-{graph["filter_id"]}/reference.json'
        if not path.exists() and partial:continue
        reference=json.loads(path.read_text())
        assert reference['graph_filter_id']==graph['filter_id']
        validate_graph(reference)
        reference['original_reference_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        graphs.append(reference)
    if not partial:assert len(graphs)==50
    return {'schema_version':1,'partial':partial,'graphs':graphs,
            'expected_raw_checks':len(graphs)*8*70,
            'internal_raw_wrapper_checks':len(graphs)*16*70,
            'generation_manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            'collector_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'input_precision':'Binary64 MadLoop parameters/momenta/wavefunctions promoted identically in mode4; no independent high-precision input oracle'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exports-dir',type=Path,required=True)
    p.add_argument('--probes-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--partial',action='store_true',help='Diagnostic collection only; raw matcher rejects partial input')
    a=p.parse_args();result=collect(a.exports_dir,a.probes_dir,a.partial)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(a.output)


if __name__=='__main__':main()
