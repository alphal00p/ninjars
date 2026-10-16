#!/usr/bin/env python3
"""Discover/export isolated stock MadLoop top-quark gg -> HH graphs.

This validation-only harness observes the unmodified loop selection predicate
before numerical grouping, then exports each graph separately with multiplier
one. Its standard generated driver is never used for the no-R2/UV comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'references/MG5_aMC_v3_7_0').is_dir())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_hashes(root):
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*'))
            if p.is_file() and p.suffix not in ('.pyc', '.pkl') and '__pycache__' not in p.parts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'target/reference/madloop-gg-hh')
    parser.add_argument('--madgraph-dir', type=Path, default=ROOT / 'references/MG5_aMC_v3_7_0')
    parser.add_argument('--discover-only', action='store_true')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    mg = args.madgraph_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'graphs.json').exists():
        raise FileExistsError('use a fresh output directory to preserve existing evidence')
    model_before = model_hashes(mg / 'models/loop_sm')
    os.chdir(out)
    sys.path.insert(0, str(mg))
    from madgraph.interface.master_interface import MasterCmd
    from madgraph.loop.loop_diagram_generation import LoopAmplitude

    spec = importlib.util.spec_from_file_location('prepare_hgg', ROOT / 'validation/madloop/prepare_hgg.py')
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    commands = []
    process = 'g g > h h QCD=2 QED=2 [virt=QCD]'
    base_filter = 'n in [3,4] and all(abs(p)==6 for p in loop_pdgs)'
    graphs = []

    def command(interface, line):
        commands.append(line)
        interface.exec_cmd(line, errorhandling=False, printcmd=False, precmd=True, postcmd=True)

    def configure():
        interface = MasterCmd()
        for line in ['set automatic_html_opening False', 'set group_subprocesses False',
                     'set loop_optimized_output False', 'set output_dependencies internal',
                     'set gauge Feynman', 'import model loop_sm']:
            command(interface, line)
        return interface

    stock_filter = LoopAmplitude.get_loop_filter

    def observe(filterdef):
        predicate = stock_filter(filterdef)

        def record(diagram, structures, model, graph_id):
            selected = predicate(diagram, structures, model, graph_id)
            if selected:
                loop_pdgs = list(diagram.get_loop_lines_pdgs())
                graphs.append({'filter_id': graph_id, 'loop_pdgs': loop_pdgs,
                               'loop_propagators': len(loop_pdgs),
                               'attached_pdgs': list(diagram.get_pdgs_attached_to_loop(structures)),
                               'vertices': [{'id': v.get('id'), 'legs': [dict(leg) for leg in v.get('legs')]}
                                            for v in diagram.get('vertices')]})
            return selected
        return record

    manifest = {'schema_version': 1, 'process': process, 'model': 'unmodified loop_sm',
                'original_model_sha256': model_before,
                'source_sha256': {'prepare_madloop.py': digest(Path(__file__)),
                                 'prepare_hgg.py': digest(ROOT / 'validation/madloop/prepare_hgg.py')},
                'commands': commands, 'graphs': graphs, 'discovery_filter': base_filter,
                'group_subprocesses': False, 'loop_optimized_output': False,
                'convention': 'four-dimensional unsquared graph numerator plus R1; no R2/UV; zero widths',
                'warning': 'Standard generated matrix drivers include counterterms; only direct LOOPNUM/CT_interface calls are intended for validation.'}

    def checkpoint():
        (out / 'graphs.json').write_text(json.dumps(manifest, indent=2) + '\n')

    try:
        LoopAmplitude.get_loop_filter = staticmethod(observe)
        try:
            interface = configure()
            command(interface, f'generate {process} --loop_filter="{base_filter}"')
        finally:
            LoopAmplitude.get_loop_filter = staticmethod(stock_filter)
        manifest['post_discovery_loop_count'] = sum(len(a.get('loop_diagrams')) for a in interface._curr_amps)
        checkpoint()
        counts = {size: sum(g['loop_propagators'] == size for g in graphs) for size in (3, 4)}
        manifest['discovered_topology_counts'] = counts
        if counts != {3: 2, 4: 6} or len({g['filter_id'] for g in graphs}) != 8:
            raise RuntimeError(f'expected two triangles and six boxes; observed {counts}')
        for graph in graphs:
            if args.discover_only:
                continue
            interface = configure()
            command(interface, f'generate {process} --loop_filter="{base_filter} and id=={graph["filter_id"]}"')
            loops = [d for amplitude in interface._curr_amps for d in amplitude.get('loop_diagrams')]
            if len(loops) != 1 or loops[0].get('multiplier') != 1:
                raise RuntimeError('isolated graph must have exactly one loop and multiplier one')
            graph['multiplier'] = loops[0].get('multiplier')
            destination = out / f'graph-{graph["filter_id"]}'
            if destination.exists():
                raise FileExistsError(destination)
            command(interface, f'output standalone {destination}')
            graph['directory'] = destination.name
            graph['reference_controls'] = common.configure_export(destination)
            elements = interface._curr_matrix_elements.get_matrix_elements()
            if len(elements) != 1:
                raise RuntimeError('expected one isolated matrix element')
            element = elements[0]
            writer = interface._curr_helas_model
            graph['external_wavefunction_calls'] = writer.get_born_ct_helas_calls(element, include_CT=False)
            graph['loop_calls'] = writer.get_loop_amp_helas_calls(element)
            graph['loop_color_basis'] = repr(element.get('loop_color_basis'))
            graph['color_amplitudes'] = [
                [{'fermion_factor': c[0], 'rational_color_coefficient': str(c[1]),
                  'imaginary': c[2], 'nc_power': c[3], 'amplitude_number': n} for c, n in row]
                for row in element.get_loop_color_amplitudes()]
            graph['generated_sources_sha256'] = {
                str(p.relative_to(destination)): digest(p)
                for p in sorted((destination / 'SubProcesses').rglob('*.f'))}
            checkpoint()
        manifest['original_model_unchanged'] = model_hashes(mg / 'models/loop_sm') == model_before
        if not manifest['original_model_unchanged']:
            raise RuntimeError('original model source changed')
        manifest['status'] = 'discovery only' if args.discover_only else 'eight individually exported top-quark graphs'
    finally:
        checkpoint()
    print(out / 'graphs.json')


if __name__ == '__main__':
    main()
