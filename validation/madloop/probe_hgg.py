#!/usr/bin/env python3
"""Build and run unsquared H->gg graph probes from isolated stock exports.

This is validation tooling. It links the generated HELAS numerator and
CutTools interface directly and never calls the matrix-element driver,
counterterm amplitude, color square, or Born interference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def parse_probe(path: Path) -> dict:
    rows, coefficients, inputs, polarization = [], None, None, {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == 'INPUT':
            inputs = list(map(float, fields[1:]))
        elif fields[0] == 'W':
            leg, component = map(int, fields[1:3])
            if leg in (2, 3) and component in range(5, 9):
                polarization.setdefault(str(leg - 1), []).append(list(map(float, fields[3:])))
        elif fields[0] == 'NUM':
            numbers = list(map(float, fields[2:]))
            if len(numbers) != 12 or numbers[8:10] != numbers[10:12]:
                raise RuntimeError('LOOPNUM(q) != LOOPNUMHEL(-q,H) with unit graph multiplier')
            rows.append({'index': int(fields[1]),
                         'q': [numbers[2*i:2*i+2] for i in range(4)],
                         'raw_numerator': numbers[8:10]})
        elif fields[0] == 'LAURENT':
            if fields[1] != 'T':
                raise RuntimeError(f'CutTools reports an unstable reference: {path}')
            coefficients = list(map(float, fields[2:]))
    if (len(rows) != 15 or coefficients is None or len(coefficients) != 6
            or inputs is None or len(inputs) != 6
            or set(polarization) != {'1', '2'}
            or any(len(components) != 4 for components in polarization.values())):
        raise RuntimeError(f'incomplete reference output: {path}')
    if ([row['index'] for row in rows] != list(range(1, 16))
            or len({tuple(tuple(component) for component in row['q']) for row in rows}) != 15):
        raise RuntimeError(f'raw sampling grid is missing or repeats points: {path}')
    numbers = [*coefficients, *inputs,
               *(value for components in polarization.values() for component in components for value in component),
               *(value for row in rows for component in [*row['q'], row['raw_numerator']] for value in component)]
    if not all(math.isfinite(value) for value in numbers):
        raise RuntimeError(f'nonfinite reference: {path}')
    return {'inputs': dict(zip(['MT', 'MH', 'mu_R', 'g_s', 'yukawa_re', 'yukawa_im'], inputs)),
            'polarizations': polarization, 'raw_samples': rows,
            'epsilon_raw': [coefficients[2*i:2*i+2] for i in range(3)],
            'epsilon_ninjars_normalization': [
                [-8*math.pi**2*value for value in coefficients[2*i:2*i+2]]
                for i in range(3)], 'stable': True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exports', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'target/reference/madloop-hgg-probe')
    parser.add_argument('--fc', default='gfortran')
    parser.add_argument('--ctmode', type=int, choices=[1, 4], nargs='+', default=[1, 4],
                        help='explicit double or quadruple reduction; no adaptive fallback')
    args = parser.parse_args()
    exports, output = args.exports.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((exports/'graphs.json').read_text())
    if metadata['group_subprocesses'] or metadata['loop_optimized_output']:
        raise RuntimeError('reference export must disable grouping and optimized loop folding')
    if len(metadata['graphs']) != 2:
        raise RuntimeError('expected two isolated top orientations')
    cases, sources = [], {}
    harness = Path(__file__).with_name('raw_hgg.f')
    sources[str(harness.relative_to(ROOT))] = sha256(harness)
    for orientation, graph in enumerate(metadata['graphs']):
        graph_id = graph['filter_id']
        if graph['multiplier'] != 1 or graph['reference_controls']['CTModeRun'] != 1:
            raise RuntimeError('reference multiplier/explicit precision controls differ')
        if not graph.get('loop_color_basis', '').startswith("{(('Tr', (2, 3)),):"):
            raise RuntimeError('missing or unexpected recorded Tr(Ta Tb) color basis')
        expected_call = ('CALL %(proc_prefix)sLOOP_3_3(1,1,2,3,'
                         'DCMPLX(mdl_MT),CMPLX(MP__mdl_MT,KIND=16),'
                         'DCMPLX(mdl_MT),CMPLX(MP__mdl_MT,KIND=16),'
                         'DCMPLX(mdl_MT),CMPLX(MP__mdl_MT,KIND=16),'
                         'GC_37,MP__GC_37,GC_5,MP__GC_5,GC_5,MP__GC_5,'
                         '3,1,1,2,AMPL(1,2),S(2))')
        calls = [call for call in graph['loop_calls'] if not call.startswith('#')]
        if calls != [expected_call]:
            raise RuntimeError('generated loop signature/couplings differ from the audited harness')
        color = [entry for row in graph['color_amplitudes'] for entry in row
                 if entry['amplitude_number'] == 2]
        if len(color) != 1 or color[0] != {
            'fermion_factor': -1, 'rational_color_coefficient': '1',
            'imaginary': False, 'nc_power': 0, 'amplitude_number': 2,
        }:
            raise RuntimeError('unexpected linear color coefficient for loop amplitude 2')
        export = exports/graph['directory']
        source = export/'SubProcesses/P0_h_gg'
        # The needed selection helper is copied verbatim; no generated routine
        # is edited, and the matrix-element routine itself is never linked.
        matrix = (source/'loop_matrix.f').read_text()
        function = re.search(r'^      LOGICAL FUNCTION ML5_0_IS_HEL_SELECTED\(HELID\).*?^      END\s*$',
                             matrix, flags=re.M | re.S)
        if function is None:
            raise RuntimeError('missing stock helicity helper')
        helper = output/f'{graph_id}-helicity_selection.f'
        helper.write_text(function.group(0)+'\n')
        run(['make', '-j2', 'libdhelas', 'libmodel', 'libcuttools'], export/'Source')
        objects = []
        for name in ['loop_num', 'CT_interface', 'mp_born_amps_and_wfs']:
            path, obj = source/f'{name}.f', output/f'{graph_id}-{name}.o'
            sources[f'{graph["directory"]}/SubProcesses/P0_h_gg/{name}.f'] = sha256(path)
            run([args.fc, '-O2', '-fPIC', '-ffixed-line-length-none', '-I'+str(source),
                 '-c', str(path), '-o', str(obj)], output)
            objects.append(str(obj))
        sources[f'{graph["directory"]}/Cards/param_card.dat'] = sha256(export/'Cards/param_card.dat')
        binary = output/f'raw-hgg-{graph_id}'
        run([args.fc, '-O2', '-ffixed-line-length-none', '-I'+str(source), str(harness),
             str(helper), *objects, '-L'+str(export/'lib'), '-ldhelas', '-lmodel', '-lcts',
             '-o', str(binary)], output)
        for ctmode in args.ctmode:
            for lane, (state, helicity, ward) in enumerate([('pp', '1', '0'), ('pm', '-1', '0'), ('ww', '1', '3')]):
                path = output/f'graph-{graph_id}-{state}-mode{ctmode}.txt'
                with path.open('w') as stream:
                    subprocess.run([str(binary), str(export/'Cards/param_card.dat'), '1', helicity,
                                    ward, str(ctmode)], cwd=output, stdout=stream, check=True)
                cases.append({'graph_filter_id': graph_id, 'feynkit_orientation': orientation,
                              'sample_lane': lane, 'state': state, 'ctmode': ctmode,
                              **parse_probe(path)})
    result = {'schema_version': 1, 'reference': 'MG5_aMC_v3_7_0 / CutTools 1.9.3 / OneLOop 3.6',
              'convention': 'four-dimensional numerator plus R1; no R2 or UV; no squaring or interference',
              'reference_graph_metadata': metadata, 'source_sha256': sources,
              'integrated_routine': 'ML5_0_LOOP_3_3', 'ctmodes': args.ctmode, 'loop_library': 2,
              'quadruple_inputs': 'identical binary64 couplings and wavefunctions promoted exactly; reduction arithmetic is quadruple',
              'linear_color_projection': '-1/2 (loop coefficient -1 times Tr(Ta Tb)=delta_ab/2)',
              'ct_interface_normalization': '1/(16*pi^2)',
              'ninjars_conversion': 'multiply raw Laurent coefficients by -8*pi^2',
              'laurent_order': ['finite', 'single_pole', 'double_pole'],
              'routing': ['q_FeynKit=-q_CutTools', 'q_FeynKit=q_CutTools'],
              'raw_sampling': '15-point unisolvent affine complex grid for total degree two in four components',
              'denominator_order': [2, 0, 1], 'cases': cases}
    path = output/'reference.json'
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(path)


if __name__ == '__main__':
    main()
