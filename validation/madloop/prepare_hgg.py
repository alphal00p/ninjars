#!/usr/bin/env python3
"""Generate isolated top H->gg graphs with stock MadLoop (validation only).

Discovery observes the existing loop-filter predicate before MadLoop groups
numerically equivalent graphs. Each production reference export then uses the
unmodified filter with one recorded graph ID and asserts multiplier == 1.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def configure_export(destination: Path) -> dict:
    """Fix the backend and disable automatic precision/phase-space changes."""
    from madgraph.various.banner import MadLoopParam
    from models.check_param_card import ParamCard

    controls = {'MLReductionLib': '1', 'CTModeRun': 1, 'CTModeInit': 1,
                'CTLoopLibrary': 2, 'ImprovePSPoint': -1,
                'UseQPIntegrandForCutTools': False, 'UseLoopFilter': False}
    ml_path = destination / 'Cards/MadLoopParams.dat'
    ml = MadLoopParam(str(ml_path))
    for name, value in controls.items():
        ml[name] = value
    ml.write(str(ml_path))
    card_path = destination / 'Cards/param_card.dat'
    card = ParamCard(str(card_path))
    for pdg in (6, 25):
        card['decay'].get((pdg,)).value = 0.0
    card.write(str(card_path))
    return controls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--madgraph', type=Path, default=ROOT / 'references/MG5_aMC_v3_7_0')
    parser.add_argument('--output', type=Path, default=ROOT / 'target/reference/madloop-hgg')
    parser.add_argument('--discover-only', action='store_true')
    args = parser.parse_args()
    args.madgraph = args.madgraph.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    # MadGraph/PLY writes parser and command files in its working directory.
    os.chdir(args.output)
    sys.path.insert(0, str(args.madgraph))
    from madgraph.interface.master_interface import MasterCmd
    from madgraph.loop.loop_diagram_generation import LoopAmplitude

    base_filter = 'n==3 and all(abs(p)==6 for p in loop_pdgs)'
    discovered = []
    original = LoopAmplitude.get_loop_filter

    def observe(filterdef):
        predicate = original(filterdef)

        def record(diagram, structures, model, graph_id):
            selected = predicate(diagram, structures, model, graph_id)
            if selected:
                discovered.append({
                    'filter_id': graph_id,
                    'loop_pdgs': list(diagram.get_loop_lines_pdgs()),
                    'attached_pdgs': list(diagram.get_pdgs_attached_to_loop(structures)),
                    'vertices': [
                        {'id': v.get('id'), 'legs': [dict(leg) for leg in v.get('legs')]}
                        for v in diagram.get('vertices')
                    ],
                })
            return selected
        return record

    def command(interface, line):
        interface.exec_cmd(line, errorhandling=False, printcmd=False, precmd=True, postcmd=True)

    def configure():
        interface = MasterCmd()
        for line in [
            'set automatic_html_opening False',
            'set group_subprocesses False',
            'set loop_optimized_output False',
            'set output_dependencies internal',
            'import model loop_sm',
        ]:
            command(interface, line)
        return interface

    LoopAmplitude.get_loop_filter = staticmethod(observe)
    try:
        discovery = configure()
        command(discovery, f'generate h > g g QED=1 [virt=QCD] --loop_filter="{base_filter}"')
    finally:
        LoopAmplitude.get_loop_filter = staticmethod(original)
    if len(discovered) != 2 or len({g['filter_id'] for g in discovered}) != 2:
        raise RuntimeError(f'expected two oriented top triangles, got {discovered}')

    args.output.mkdir(parents=True, exist_ok=True)
    for graph in discovered:
        if args.discover_only:
            continue
        interface = configure()
        graph_filter = f'{base_filter} and id=={graph["filter_id"]}'
        command(interface, f'generate h > g g QED=1 [virt=QCD] --loop_filter="{graph_filter}"')
        amplitudes = interface._curr_amps
        loops = [d for a in amplitudes for d in a.get('loop_diagrams')]
        if len(loops) != 1 or loops[0].get('multiplier') != 1:
            raise RuntimeError('reference graph grouping was not fully isolated')
        graph['multiplier'] = loops[0].get('multiplier')
        destination = args.output / f'graph-{graph["filter_id"]}'
        if destination.exists():
            raise FileExistsError(f'refusing to overwrite existing export: {destination}')
        command(interface, f'output standalone {destination}')
        graph['directory'] = destination.name
        graph['reference_controls'] = configure_export(destination)
        elements = interface._curr_matrix_elements.get_matrix_elements()
        if len(elements) != 1:
            raise RuntimeError('expected one isolated matrix element')
        element = elements[0]
        writer = interface._curr_helas_model
        graph['external_wavefunction_calls'] = writer.get_born_ct_helas_calls(element, include_CT=False)
        graph['loop_calls'] = writer.get_loop_amp_helas_calls(element)
        graph['loop_color_basis'] = repr(element.get('loop_color_basis'))
        graph['color_amplitudes'] = [
            [
                {'fermion_factor': coefficient[0], 'rational_color_coefficient': str(coefficient[1]),
                 'imaginary': coefficient[2], 'nc_power': coefficient[3], 'amplitude_number': number}
                for coefficient, number in row
            ] for row in element.get_loop_color_amplitudes()
        ]

    metadata = {
        'process': 'h > g g QED=1 [virt=QCD]',
        'model': 'loop_sm',
        'group_subprocesses': False,
        'loop_optimized_output': False,
        'discovery_filter': base_filter,
        'graphs': discovered,
        'status': 'discovery only' if args.discover_only else 'isolated reference exports',
        'target_convention': 'four-dimensional numerators plus R1, no R2 or UV; zero widths',
        'warning': 'Exported standard drivers include counterterms. They are not the no-R2/UV graph validation harness.',
    }
    path = args.output / 'graphs.json'
    path.write_text(json.dumps(metadata, indent=2) + '\n')
    print(path)


if __name__ == '__main__':
    main()
