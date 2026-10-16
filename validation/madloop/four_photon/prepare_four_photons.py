#!/usr/bin/env python3
"""Export six isolated top-quark four-photon graphs with explicit MadLoop controls."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo-root', type=Path, default=next(p for p in Path(__file__).resolve().parents if (p/'references/MG5_aMC_v3_7_0').is_dir()))
parser.add_argument('--output-dir', type=Path)
parser.add_argument('--madgraph-dir', type=Path)
args = parser.parse_args()
ROOT = args.repo_root.resolve()
OUT = (args.output_dir or ROOT/'target/reference/madloop-four-photon').resolve()
OUT.mkdir(parents=True, exist_ok=True)
MG = (args.madgraph_dir or ROOT/'references/MG5_aMC_v3_7_0').resolve()
SOURCE = MG / 'models/loop_sm'
MODEL = OUT / 'loop_sm_qed_validation'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def hashes(root):
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*'))
            if p.is_file() and p.suffix not in ('.pyc', '.pkl') and '__pycache__' not in p.parts}

before = hashes(SOURCE)
if MODEL.exists():
    raise FileExistsError(MODEL)
shutil.copytree(SOURCE, MODEL, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.pkl'))
orders = MODEL / 'coupling_orders.py'
original = orders.read_text()
assert original.count('perturbative_expansion =0') == 1
orders.write_text(original.replace('perturbative_expansion =0', 'perturbative_expansion =1'))
model_changes = {p: {'original': before[p], 'copied': h} for p, h in hashes(MODEL).items()
                 if h != before[p]}
assert set(model_changes) == {'coupling_orders.py'}
os.chdir(OUT)
sys.path.insert(0, str(MG))
from madgraph.interface.master_interface import MasterCmd
from madgraph.loop.loop_diagram_generation import LoopAmplitude

spec = importlib.util.spec_from_file_location('prepare_hgg', ROOT / 'validation/madloop/prepare_hgg.py')
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)
commands = []
process = 'a a > a a [virt=QED]'
base_filter = 'n==4 and all(abs(p)==6 for p in loop_pdgs)'

def command(interface, line):
    commands.append(line)
    interface.exec_cmd(line, errorhandling=False, printcmd=False, precmd=True, postcmd=True)

def configure():
    interface = MasterCmd()
    for line in ['set automatic_html_opening False', 'set group_subprocesses False',
                 'set loop_optimized_output False', 'set output_dependencies internal',
                 'set gauge Feynman', f'import model {MODEL}']:
        command(interface, line)
    assert 'QED' in interface._curr_model['perturbation_couplings']
    return interface

graphs = []
stock_filter = LoopAmplitude.get_loop_filter
def observe(filterdef):
    predicate = stock_filter(filterdef)
    def record(diagram, structures, model, graph_id):
        selected = predicate(diagram, structures, model, graph_id)
        if selected:
            graphs.append({'filter_id': graph_id,
                           'loop_pdgs': list(diagram.get_loop_lines_pdgs()),
                           'attached_pdgs': list(diagram.get_pdgs_attached_to_loop(structures)),
                           'vertices': [{'id': v.get('id'), 'legs': [dict(leg) for leg in v.get('legs')]}
                                        for v in diagram.get('vertices')]})
        return selected
    return record

manifest = {'process': process, 'model': str(MODEL), 'original_model_sha256': before,
            'model_changes': model_changes, 'model_patch': 'QED perturbative_expansion: 0 -> 1 only',
            'source_sha256': {'prepare_four_photons.py': digest(Path(__file__)),
                             'prepare_hgg.py': digest(ROOT/'validation/madloop/prepare_hgg.py')},
            'commands': commands, 'graphs': graphs,
            'convention': '4D numerator plus R1, no explicit R2 or UV; no squaring or interference',
            'group_subprocesses': False, 'loop_optimized_output': False,
            'limitations': ['validation-only QED metadata enables existing tree vertices; not a complete QED loop model']}
def checkpoint():
    (OUT/'graphs.json').write_text(json.dumps(manifest, indent=2)+'\n')

try:
    LoopAmplitude.get_loop_filter = staticmethod(observe)
    try:
        interface = configure()
        command(interface, f'generate {process} --loop_filter="{base_filter}"')
    finally:
        LoopAmplitude.get_loop_filter = staticmethod(stock_filter)
    checkpoint()
    if len(graphs) != 6 or len({g['filter_id'] for g in graphs}) != 6:
        raise RuntimeError(f'expected six distinct oriented boxes, observed {len(graphs)}')
    for graph in graphs:
        interface = configure()
        command(interface, f'generate {process} --loop_filter="{base_filter} and id=={graph["filter_id"]}"')
        loops = [d for amplitude in interface._curr_amps for d in amplitude.get('loop_diagrams')]
        if len(loops) != 1 or loops[0].get('multiplier') != 1:
            raise RuntimeError('graph isolation/grouping assertion failed')
        graph['multiplier'] = loops[0].get('multiplier')
        destination = OUT/f'graph-{graph["filter_id"]}'
        command(interface, f'output standalone {destination}')
        graph['directory'] = destination.name
        graph['reference_controls'] = common.configure_export(destination)
        elements = interface._curr_matrix_elements.get_matrix_elements()
        assert len(elements) == 1
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
            for p in sorted((destination/'SubProcesses').rglob('*.f'))}
        checkpoint()
    assert hashes(SOURCE) == before, 'original model source changed'
    manifest['original_model_unchanged'] = True
    manifest['status'] = 'six individually exported oriented top boxes'
    checkpoint()
finally:
    checkpoint()
print(OUT/'graphs.json')
