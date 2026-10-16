#!/usr/bin/env python3
"""Collect complete isolated graph probes without changing numerical results."""
import argparse
import hashlib
import json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--exports-dir',type=Path,required=True)
p.add_argument('--probes-dir',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--partial',action='store_true',help='Diagnostic collection of available graphs; never full acceptance')
a=p.parse_args();manifest=json.loads((a.exports_dir/'graphs.json').read_text());graphs=[]
for graph in manifest['graphs']:
 path=a.probes_dir/f'probe-{graph["filter_id"]}/reference.json'
 if not path.exists() and a.partial:continue
 r=json.loads(path.read_text());assert r['graph_filter_id']==graph['filter_id']
 r['original_reference_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();graphs.append(r)
if not a.partial:assert len(graphs)==8 and manifest['original_model_unchanged']
result={'schema_version':1,'partial':a.partial,'graphs':graphs,
        'expected_raw_checks':sum(70 for g in graphs for c in g['cases'] if c['ctmode']==4),
        'internal_raw_wrapper_checks':sum(len(c['numerator_samples']) for g in graphs for c in g['cases']),
        'generation_manifest_sha256':hashlib.sha256((a.exports_dir/'graphs.json').read_bytes()).hexdigest(),
        'collector_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'input_precision':'Binary64 MadLoop parameters/momenta/wavefunctions promoted identically in mode4; no independent high-precision input oracle'}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(a.output)
