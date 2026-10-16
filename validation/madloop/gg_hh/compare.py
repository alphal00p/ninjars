#!/usr/bin/env python3
"""Compare eight unsquared NinjaRS gg->HH graphs to fixed-mode MadLoop."""
from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
CARRIERS = ('f64', 'double-float', 'float')
MU_MODES = ('expansion', 'sampling')
STATES = ('pp', 'pm', 'mp', 'mm', 'generic_complex', 'ward0', 'ward1', 'ward_both')
COEFFICIENTS = ('finite', 'single_pole', 'double_pole')
CRITERION = 1e-9
FACTOR = -8 * math.pi**2


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pair(value):
    result = complex(*map(float, value))
    if not math.isfinite(result.real) or not math.isfinite(result.imag):
        raise ValueError('nonfinite complex comparison value')
    return result


def scaled_error(actual, expected):
    result = abs(pair(actual)-pair(expected))/max(1., abs(pair(expected)))
    if not math.isfinite(result):
        raise ValueError('nonfinite comparison error')
    return result


def precision_error(actual, expected):
    """Compare printed components without discarding the extra output digits."""
    with localcontext() as context:
        context.prec = 120
        a = list(map(Decimal, actual))
        e = list(map(Decimal, expected))
        error = sum((x-y)**2 for x, y in zip(a, e)).sqrt()
        scale = max(Decimal(1), sum(x*x for x in e).sqrt())
        return str(error/scale)


def source_paths():
    paths = [ROOT/'Cargo.toml', ROOT/'Cargo.lock', ROOT/'dependencies/manifest.json',
             Path(__file__).resolve()]
    for name in ('ninjars', 'ninjars-graph'):
        paths += list((ROOT/f'crates/{name}/src').rglob('*.rs'))
    paths += [p for p in (ROOT/'crates/ninjars-symbolica/src').rglob('*.rs')
              if p.name not in ('reference.rs', 'rust_codegen.rs')]
    paths += [ROOT/f'crates/ninjars-cli/src/{name}.rs'
              for name in ('prepare', 'runtime', 'reduce')]
    paths += list((ROOT/'examples/graphs/physical/gg-hh').glob('*.dot'))
    paths += list((ROOT/'examples/graphs/physical/gg-hh').glob('*.json'))
    return sorted(set(paths))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ninjars', type=Path, required=True,
                        help='Copied executable; never relink this path during a run.')
    parser.add_argument('--build-record', type=Path, required=True)
    parser.add_argument('--reference', type=Path,
                        default=ROOT/'validation/evidence/gg-hh-madloop.json')
    parser.add_argument('--raw-evidence', type=Path,
                        default=ROOT/'validation/evidence/gg-hh-raw.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--orientations', type=int, nargs='+', default=list(range(8)))
    parser.add_argument('--precisions', choices=CARRIERS, nargs='+', default=list(CARRIERS))
    parser.add_argument('--mu-modes', choices=MU_MODES, nargs='+', default=list(MU_MODES))
    parser.add_argument('--numerator-evaluator', choices=('symbolica', 'symjit-o2'), default='symbolica')
    parser.add_argument('--batch-size', type=int, default=3,
                        help='Eight samples with width3 exercise an incomplete final batch.')
    args = parser.parse_args()
    if args.numerator_evaluator == 'symjit-o2' and args.precisions != ['f64']:
        parser.error('symjit-o2 requires --precisions f64')
    if args.batch_size < 1 or len(set(args.orientations)) != len(args.orientations) or any(i not in range(8) for i in args.orientations):
        parser.error('positive batch size and distinct orientations in0..7 required')
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    binary = args.ninjars.resolve()
    build_record = json.loads(args.build_record.read_text())
    recorded_sha = build_record.get('binary_sha256', build_record.get('binary',{}).get('sha256'))
    if recorded_sha != digest(binary):
        raise ValueError('immutable binary does not match its build record')
    reference_path = args.reference.resolve()
    raw_path = args.raw_evidence.resolve()
    reference = json.loads(reference_path.read_text())
    raw = json.loads(raw_path.read_text())
    if not raw['pass'] or raw['checks'] != 4480:
        raise ValueError('independent raw-numerator gate has not passed')
    mappings = {row['feynkit_orientation']: row for row in raw['graphs']}
    if set(mappings) != set(range(8)):
        raise ValueError('eight verified graph mappings required')
    expected = {(g['graph_filter_id'], c['state'], c['ctmode']): c
                for g in reference['graphs'] for c in g['cases']}
    sources = {str(p.relative_to(ROOT)): digest(p) for p in source_paths()}
    provenance = {'binary': {'path': str(binary), 'sha256': digest(binary)},
                  'build_record_sha256':digest(args.build_record),'build_record':build_record,
                  'source_sha256': sources, 'working_directory': str(ROOT),
                  'reference_sha256': digest(reference_path),
                  'raw_evidence_sha256': digest(raw_path), 'commands': [],
                  'git_head': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  'dependency_heads': {name:subprocess.check_output(['git','-C',str(ROOT/path),'rev-parse','HEAD'],text=True).strip()
                                       for name,path in [('symbolica','references/symbolica'),('feynkit','references/gammaloop_ninja_shared'),('oneloopmaster','references/oneloopmaster')]}}
    checks = []
    outputs = {}
    preparations = []
    for orientation in args.orientations:
        topology = 'triangle' if orientation < 2 else 'box'
        fixture = ROOT/f'examples/graphs/physical/gg-hh/top-{topology}-{orientation}.json'
        command = [str(binary), 'prepare', str(fixture)]
        prepared = subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
        provenance['commands'].append({'command':command,'returncode':prepared.returncode})
        (out/f'prepared-{orientation}.json').write_text(prepared.stdout)
        (out/f'prepared-{orientation}.stderr').write_text(prepared.stderr)
        if prepared.returncode:
            raise RuntimeError(f'preparation failed for orientation{orientation}: {prepared.stderr}')
        document = json.loads(prepared.stdout)
        if len(document['samples']) != 8 or len(document['propagator_edges']) != (3 if orientation < 2 else 4):
            raise ValueError('unexpected sample count or loop propagator count')
        if any(sample['overall_factor'] != ('-1/74375' if orientation < 2 else '-1') for sample in document['samples']):
            raise ValueError('fermion sign or Higgs bridge normalization changed')
        preparations.append({'orientation':orientation,'sha256':digest(out/f'prepared-{orientation}.json'),
                             'mapping':mappings[orientation]})
        for precision in args.precisions:
            for mu_mode in args.mu_modes:
                stem = f'graph-{orientation}-{precision}-{mu_mode}'
                command = [str(binary),'reduce',str(fixture),'--precision',precision,
                           '--batch-size',str(args.batch_size),'--mu-extraction',mu_mode,'--diagnostics']
                if precision == 'float':
                    command += ['--bits','256']
                if args.numerator_evaluator != 'symbolica':
                    command += ['--numerator-evaluator', args.numerator_evaluator]
                started=time.monotonic()
                completed = subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
                elapsed=time.monotonic()-started
                (out/f'{stem}.json').write_text(completed.stdout)
                (out/f'{stem}.stderr').write_text(completed.stderr)
                provenance['commands'].append({'command':command,'returncode':completed.returncode,
                                               'operational_elapsed_seconds':elapsed,'timing_is_benchmark':False,
                                               'stdout_sha256':digest(out/f'{stem}.json')})
                actual = json.loads(completed.stdout) if completed.stdout.strip() else None
                if actual is not None:
                    if args.numerator_evaluator != 'symbolica' and actual['numerator_provider']['evaluator'] != args.numerator_evaluator:
                        raise ValueError('requested numerator evaluator was not used')
                    if actual['rank'] != (2 if orientation < 2 else 4) or actual['box_reduction']['mu_extraction'] != mu_mode or actual['cut_stop'] != 1:
                        raise ValueError('unexpected rank, extraction mode or cut stopping in reduction output')
                    if len(actual['results']) != 8:
                        raise ValueError('unexpected output lane count')
                for lane,state in enumerate(STATES):
                    row=actual['results'][lane] if actual else {'status':{'kind':'process_error'},'values':None}
                    outputs[orientation,precision,mu_mode,lane]=row
                    comparisons=[]
                    for reference_mode in (1,4):
                        target_case=expected[mappings[orientation]['madloop_filter_id'],state,reference_mode]
                        target=([[FACTOR*x for x in value] for value in target_case['raw_laurent']]
                                if target_case['raw_laurent'] is not None else None)
                        errors=None
                        decomposition_error=None
                        if row['status']['kind']=='success' and target is not None:
                            errors=[scaled_error(value,truth) for value,truth in zip(row['values'][:3],target)]
                            finite=pair(row['values'][0]);split=pair(row['values'][3])+pair(row['values'][4])
                            decomposition_error=abs(finite-split)/max(1.,abs(finite))
                        passed=(errors is not None and max(errors)<=CRITERION and decomposition_error<=CRITERION
                                and target_case['stable'])
                        comparisons.append({'reference_mode':reference_mode,'reference_stable':target_case['stable'],
                                            'target':target,'scaled_errors':errors,
                                            'decomposition_scaled_error':decomposition_error,'passed':passed})
                    checks.append({'orientation':orientation,'madloop_filter_id':mappings[orientation]['madloop_filter_id'],
                                   'lane':lane,'state':state,'precision':precision,'working_bits':actual['precision']['bits'] if actual else None,
                                   'mu_extraction':mu_mode,'batch_size':args.batch_size,'diagnostic_only':lane>=4,
                                   'result':row,'comparisons':comparisons,'process_returncode':completed.returncode})
                failures=sum(not c['comparisons'][1]['passed'] for c in checks[-8:])
                print(f'{stem}: {8-failures}/8 rows pass fixed-mode4, elapsed {elapsed:.2f}s',flush=True)
                # Keep completed rows even if a later invocation is interrupted.
                (out/'progress.json').write_text(json.dumps(checks,indent=2,allow_nan=False)+'\n')
    controls=[]
    for orientation in args.orientations:
        for lane,state in enumerate(STATES):
            for precision in args.precisions:
                if all((orientation,precision,m,lane) in outputs and outputs[orientation,precision,m,lane]['status']['kind']=='success' for m in MU_MODES):
                    a=outputs[orientation,precision,'expansion',lane]['values']
                    b=outputs[orientation,precision,'sampling',lane]['values']
                    errors=[precision_error(x,y) for x,y in zip(a[:3],b[:3])]
                    controls.append({'kind':'mu_mode_equivalence','orientation':orientation,'state':state,
                                     'precision':precision,'diagnostic_only':lane>=4,'errors':errors,'passed':all(Decimal(e)<=Decimal('1e-9') for e in errors)})
            for mode in args.mu_modes:
                if 'float' not in args.precisions: continue
                truth=outputs[orientation,'float',mode,lane]
                if truth['status']['kind']!='success': continue
                for precision in ('f64','double-float'):
                    if precision not in args.precisions: continue
                    row=outputs[orientation,precision,mode,lane]
                    if row['status']['kind']!='success': continue
                    controls.append({'kind':'precision_convergence','orientation':orientation,'state':state,
                                     'precision':precision,'mu_extraction':mode,
                                     'target':'NinjaRS Float256 at exact fixture inputs; convergence control, not independent oracle',
                                     'errors':[precision_error(x,y) for x,y in zip(row['values'][:3],truth['values'][:3])]})
    sums=[]
    if set(args.orientations)==set(range(8)):
        for precision in args.precisions:
            for mode in args.mu_modes:
                for lane,state in enumerate(STATES):
                    rows=[outputs[o,precision,mode,lane] for o in range(8)]
                    if any(r['status']['kind']!='success' for r in rows): continue
                    for reference_mode in (1,4):
                        actual=[sum(pair(r['values'][i]) for r in rows) for i in range(3)]
                        target=[sum(FACTOR*pair(expected[mappings[o]['madloop_filter_id'],state,reference_mode]['raw_laurent'][i]) for o in range(8)) for i in range(3)]
                        errors=[abs(a-e)/max(1.,abs(e)) for a,e in zip(actual,target)]
                        sums.append({'precision':precision,'mu_extraction':mode,'state':state,'reference_mode':reference_mode,
                                     'actual':[[x.real,x.imag] for x in actual],'target':[[x.real,x.imag] for x in target],
                                     'scaled_errors':errors,'passed':max(errors)<=CRITERION,'diagnostic_only':lane>=4})
    changed=[name for name,value in sources.items() if digest(ROOT/name)!=value]
    if digest(binary)!=provenance['binary']['sha256']: changed.append('CLI binary')
    if digest(reference_path)!=provenance['reference_sha256']: changed.append('reference evidence')
    if digest(raw_path)!=provenance['raw_evidence_sha256']: changed.append('raw evidence')
    physical=[c for c in checks if not c['diagnostic_only']]
    diagnostics=[c for c in checks if c['diagnostic_only']]
    gate=lambda rows,mode: all(next(c for c in r['comparisons'] if c['reference_mode']==mode)['passed'] for r in rows)
    process_success = all(c['returncode'] == 0 for c in provenance['commands'])
    physical_mu = [c for c in controls if c['kind']=='mu_mode_equivalence' and not c['diagnostic_only']]
    physical_sums = [c for c in sums if c['reference_mode']==4 and not c['diagnostic_only']]
    full_coverage = set(args.orientations)==set(range(8)) and set(args.precisions)==set(CARRIERS) and set(args.mu_modes)==set(MU_MODES)
    physical_passed = (bool(physical) and gate(physical,4) and process_success and not changed
                       and all(c['passed'] for c in physical_mu) and all(c['passed'] for c in physical_sums))
    summary={'schema_version':1,'physical_passed':physical_passed,
             'full_matrix_coverage':full_coverage,'process_success':process_success,
             'physical_mu_mode_checks_passed':all(c['passed'] for c in physical_mu),
             'physical_summed_graphs_passed':all(c['passed'] for c in physical_sums),
             'numerator_evaluator':args.numerator_evaluator,
             'physical_mode1_passed':gate(physical,1) and not changed,
             'generic_and_ward_diagnostic_passed':gate(diagnostics,4) and not changed,
             'physical_rows':len(physical),'diagnostic_rows':len(diagnostics),
             'criterion':f'each complex Laurent coefficient abs(a-b)/max(1,abs(b)) <= {CRITERION}; finite=cc+rational',
             'primary_reference_mode':4,'all_reference_modes_recorded':[1,4],
             'normalization':{'raw_factor':-0.5,'returned_laurent_factor':'-8*pi^2','numeric_factor':FACTOR,
                              'derivation':'Independently verified fermion sign times projected Tr(TaTb), giving -1/2; Higgs bridge once; undo CT_interface normalization1/(16*pi^2).'},
             'input_precision_note':'NinjaRS evaluates exact rational/radical fixture bindings at the selected precision. MadLoop prints binary64 wavefunctions/couplings, promoted to quadruple arithmetic in fixed mode4. This is not an independently high-precision physical-input oracle.',
             'high_precision_input_oracle':False,'performance_acceptance':False,
             'preparations':preparations,'checks':checks,'controls':controls,'summed_graphs':sums,
             'provenance':provenance,'changed_sources':changed}
    (out/'result.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(f"physical rows {len(physical)}, mode4 pass={summary['physical_passed']}, generic diagnostics pass={summary['generic_and_ward_diagnostic_passed']}",flush=True)
    print(out/'result.json',flush=True)
    if not summary['physical_passed']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
