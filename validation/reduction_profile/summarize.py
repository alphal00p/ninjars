#!/usr/bin/env python3
"""Archive diagnostic counters without promoting them to acceptance timings."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_spans(result, spans):
    if not result.get('success') or not result.get('sources_unchanged') or result.get('union_misses') != 0:
        raise ValueError('run failed numerical, source, or frozen-union guard')
    rounds, width = result['rounds'], result['width']
    if not isinstance(rounds,int) or rounds <= 0 or not isinstance(width,int) or width <= 0:
        raise ValueError('invalid dimensions')
    if not spans.get('records') or spans.get('performance_acceptance') is not False:
        raise ValueError('missing or mislabeled diagnostic records')
    identities = set()
    for row in spans['records']:
        identity = tuple(row[key] for key in ['kind','label','parent','file','line','inputs','outputs'])
        if identity in identities: raise ValueError('duplicate semantic record')
        identities.add(identity)
        if row['kind'] not in ['phase','kernel']: raise ValueError('unknown span kind')
        for key in ['calls','rows','inclusive_ns','exclusive_ns']:
            if not isinstance(row[key],int) or row[key] < 0: raise ValueError('invalid counter')
        if row['calls'] % rounds or row['rows'] % rounds: raise ValueError('incomplete fixed-workload counters')
        if row['exclusive_ns'] > row['inclusive_ns']: raise ValueError('invalid nested accounting')
        if row['kind'] == 'kernel' and Path(row['file']).name in ['kernel.rs','diagnostic_profile.rs']:
            raise ValueError('track_caller did not reach the semantic owner')
        if row['kind'] == 'kernel' and row['exclusive_ns'] != row['inclusive_ns']:
            raise ValueError('unexpected nested exact kernel')
    phases = {row['label']:row for row in spans['records'] if row['kind']=='phase'}
    geometry = {'high_cuts::box_batch':5,'cuts::triangle_batch':10,'cuts::bubble_batch':10,'cuts::tadpole_batch':5} if result['scope'].startswith('shared-reduction') else {'cuts::bubble_batch':1,'cuts::tadpole_batch':2}
    for label, lanes in geometry.items():
        row=phases[label]
        if row['calls'] != rounds or row['rows'] != rounds*width*lanes:
            raise ValueError('geometry not recomputed once per fixed batch')


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--repository',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--profile-dir',type=Path,action='append',default=[])
    parser.add_argument('--span-dir',type=Path,action='append',required=True)
    args=parser.parse_args();repo=args.repository.resolve()
    # This is an explicit post-run audit of the pre-existing reference build.
    # The original runs' own source snapshots are retained, not backfilled.
    spec=importlib.util.spec_from_file_location('reference_build',repo/'validation/reference/build.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    reference=repo/'target/reference-uncached-v2'
    record=json.loads((reference/'revisions.json').read_text())
    if module.configuration(reference)!=record['configuration'] or record['configuration']['master_result_cache']:
        raise ValueError('reference configuration changed')
    if module.library_inputs(reference)!=record['input_files_sha256']:
        raise ValueError('copied reference input identity changed')
    runs=[];builds={}
    for directory in args.span_dir:
        result=json.loads((directory/'result.json').read_text());spans=json.loads((directory/'kernel-spans.json').read_text())
        validate_spans(result,spans)
        build=result.pop('diagnostic_build');build_key=build['library_sha256'];builds[build_key]=build
        rows=[]
        for row in spans['records']:
            row=dict(row);source=Path(row['file']);lines=source.read_text().splitlines()
            row['source_context']='\n'.join(lines[max(0,row['line']-4):row['line']+2]) if row['kind']=='kernel' else None
            row['calls_per_reduction']=row['calls']/result['rounds'];row['rows_per_reduction']=row['rows']/result['rounds']
            row['inclusive_ns_per_reduction']=row['inclusive_ns']/result['rounds'];row['exclusive_ns_per_reduction']=row['exclusive_ns']/result['rounds']
            rows.append(row)
        runs.append({'directory':str(directory.resolve()),'result_sha256':digest(directory/'result.json'),
            'span_sha256':digest(directory/'kernel-spans.json'),'result':result,'build_library_sha256':build_key,'records':rows})
    profiles=[]
    for directory in args.profile_dir:
        result=json.loads((directory/'result.json').read_text())
        if not result.get('success') or not result.get('sources_unchanged') or result.get('union_misses')!=0:raise ValueError('invalid uninstrumented profile')
        profiles.append({'directory':str(directory.resolve()),'result':result,'result_sha256':digest(directory/'result.json'),
            'self_report':(directory/'self-report.stdout').read_text(),'callgraph_report':(directory/'callgraph-report.stdout').read_text(),
            'perf_record_stderr':(directory/'record.stderr').read_text()})
    output={'schema_version':1,'success':True,'performance_acceptance':False,'instrumented_builds':builds,'spans':runs,'external_profiles':profiles,
       'reference_post_run_audit':{'configuration':record['configuration'],'revisions_sha256':digest(reference/'revisions.json'),'input_files_sha256':record['input_files_sha256'],'matched':True},
       'numerator_execution':'External physical coefficient-head preparation uses SymJIT O2; actual shared PolynomialNumerator direct/T3/T2 reduction callbacks are interpreted for both physical and synthetic profiles.',
       'kernel_execution':'Symbolica interpreter. Native denominator feature/selection is absent from these runs.',
       'limitations':['Own-process CPU sampling and intrusive diagnostic spans; not parity timings or host-isolation evidence.',
          'Optimized immutable library has incomplete unwind chains: self-symbol percentages are available, but inclusive caller attribution is incomplete.',
          'Per-kernel spans include timer and map overhead, change code layout, and cannot be used to infer the cost of empty paths.',
          'Kernel span totals exclude shared numerator callbacks, frozen-master lookup, and much coordinator/FFI work.',
          'Bubble polynomial controls differ from the historical 10–16x benchmark fixture; that complete gate remains outstanding.',
          'All workloads here repeat one point. The separate measurement harness owns changing-input and paired-provider spans.']}
    raw=(json.dumps(output,indent=2,allow_nan=False)+'\n').encode()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_bytes(gzip.compress(raw,mtime=0))
    print(json.dumps({'output':str(args.output),'sha256':digest(args.output),'uncompressed_sha256':hashlib.sha256(raw).hexdigest(),'uncompressed_bytes':len(raw),'runs':len(runs),'profiles':len(profiles)}))

if __name__=='__main__':main()
