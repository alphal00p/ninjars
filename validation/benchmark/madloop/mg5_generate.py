#!/usr/bin/env python3
"""Generate one isolated MadLoop loop diagram (optimized output) and time it.

Runs stock MG5_aMC 3.7.0 through its own command interpreter (the same
commands an mg5_aMC input file would contain), with two harness-only hooks:

* the stock loop-filter predicate is wrapped to *record* the selected loop
  diagram (filter id, loop PDGs, attached PDGs, vertices) for verification;
* the exporter's make() (the serial `make` of Source/ that `output standalone`
  runs) is timed separately so that generation and Source compilation can be
  reported apart. make() itself is executed unchanged.

CutTools is linked from an explicit directory compiled with the current
compiler (MG5 refuses a CutTools library built by another gfortran version);
set via the interface attribute, as validation/madloop/*/prepare_*.py do.
No configuration is saved (all `set` commands use --no_save) and the MG5
installation is not modified.

  python3 mg5_generate.py --process "g g > h h h QCD=2 QED=3 [virt=QCD]" \
      --loop-filter "... and id==55" --ninja-lib DIR --cuttools-dir DIR \
      --output DEST --record DEST.json
"""
import argparse
import json
import os
import pathlib
import sys
import time

T_START = time.perf_counter()

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--madgraph-dir", type=pathlib.Path, default=ROOT / "references/MG5_aMC_v3_7_0")
    parser.add_argument("--process", required=True)
    parser.add_argument("--loop-filter", required=True)
    parser.add_argument("--model", default="loop_sm")
    parser.add_argument("--ninja-lib", type=pathlib.Path, required=True)
    parser.add_argument("--cuttools-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--record", type=pathlib.Path, required=True)
    parser.add_argument("--allow-many", action="store_true",
                        help="full-process mode: keep every diagram passing the filter (no single-diagram assertion)")
    args = parser.parse_args()
    mg = args.madgraph_dir.resolve()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    workdir = out.parent
    workdir.mkdir(parents=True, exist_ok=True)
    os.chdir(workdir)  # MG5/PLY write parser tables and history files here

    t0 = time.perf_counter()
    sys.path.insert(0, str(mg))
    from madgraph.interface.master_interface import MasterCmd
    from madgraph.loop.loop_diagram_generation import LoopAmplitude
    from madgraph.loop import loop_exporters
    t_import = time.perf_counter() - t0

    selected = []
    stock_filter = LoopAmplitude.get_loop_filter

    def observe(filterdef):
        predicate = stock_filter(filterdef)

        def record(diagram, structures, model, graph_id):
            keep = predicate(diagram, structures, model, graph_id)
            if keep:
                selected.append({
                    "filter_id": graph_id,
                    "loop_pdgs": list(diagram.get_loop_lines_pdgs()),
                    "attached_pdgs": list(diagram.get_pdgs_attached_to_loop(structures)),
                    "vertices": [{"id": v.get("id"), "legs": [dict(leg) for leg in v.get("legs")]}
                                 for v in diagram.get("vertices")]})
            return keep
        return record

    make_times = []
    stock_make = loop_exporters.LoopProcessExporterFortranSA.make

    def timed_make(self, *a, **k):
        s = time.perf_counter()
        try:
            return stock_make(self, *a, **k)
        finally:
            make_times.append(time.perf_counter() - s)

    commands = [
        "set automatic_html_opening False --no_save",
        "set gauge Feynman --no_save",
        "set group_subprocesses False --no_save",
        "set loop_optimized_output True --no_save",
        "set output_dependencies external --no_save",
        f"set ninja {args.ninja_lib.resolve()} --no_save",
        "set collier None --no_save",
        f"import model {args.model}",
        f'generate {args.process} --loop_filter="{args.loop_filter}"',
        f"output standalone {out}",
    ]
    timings = {}
    LoopAmplitude.get_loop_filter = staticmethod(observe)
    loop_exporters.LoopProcessExporterFortranSA.make = timed_make
    try:
        t0 = time.perf_counter()
        interface = MasterCmd()
        interface._cuttools_dir = str(args.cuttools_dir.resolve())
        timings["interface_setup"] = time.perf_counter() - t0
        for line in commands:
            key = line.split()[0] if not line.startswith("set") else "set"
            if key == "output":
                # The filter isolates the diagram at generation time; `output`
                # later rebuilds the amplitude's diagram list for the HELAS
                # matrix element (e.g. adds the type -6 orientation entry of a
                # fermion loop), so count before output.
                pre_output_loops = [d for amp in interface._curr_amps for d in amp.get("loop_diagrams")]
            t0 = time.perf_counter()
            interface.exec_cmd(line, errorhandling=False, printcmd=False, precmd=True, postcmd=True)
            timings[key] = timings.get(key, 0.0) + time.perf_counter() - t0
    finally:
        LoopAmplitude.get_loop_filter = staticmethod(stock_filter)
        loop_exporters.LoopProcessExporterFortranSA.make = stock_make

    loops = pre_output_loops
    post_output_loops = [d for amp in interface._curr_amps for d in amp.get("loop_diagrams")]
    elements = interface._curr_matrix_elements.get_matrix_elements()
    element = elements[0]
    try:
        loop_groups = element.get("loop_groups")
    except Exception:
        loop_groups = None
    t_end = time.perf_counter()
    source_make = sum(make_times)
    record = {
        "schema": "ninjars-reference-madloop-generation-v1",
        "process": args.process, "loop_filter": args.loop_filter, "model": args.model,
        "commands": commands,
        "cuttools_dir": str(args.cuttools_dir.resolve()),
        "ninja_lib": str(args.ninja_lib.resolve()),
        "output": str(out),
        "selected_loop_diagrams": selected,
        "n_loop_diagrams_after_filter": len(loops),
        "multipliers": [d.get("multiplier") for d in loops],
        "post_output_loop_diagram_types": [d.get("type") for d in post_output_loops],
        "n_matrix_elements": len(elements),
        "n_loop_groups": len(loop_groups) if loop_groups is not None else None,
        "timings_seconds": {
            "python_imports": t_import,
            **timings,
            "source_make_inside_output": source_make,
            "output_without_source_make": timings.get("output", 0.0) - source_make,
            "generation_total_in_process": (t_end - T_START) - source_make,
            "process_total_in_process": t_end - T_START,
        },
        "note": ("generation = python start..end of output, minus the serial `make` of Source/ "
                 "that `output standalone` runs (reported separately as source_make_inside_output)."),
    }
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, indent=2, default=str) + "\n")
    if not args.allow_many and (len(loops) != 1 or len(selected) != 1 or loops[0].get("multiplier") != 1
                                or (loop_groups is not None and len(loop_groups) != 1)):
        raise RuntimeError(f"expected exactly one loop diagram, got {len(loops)} (selected {len(selected)})")
    print(args.record)


if __name__ == "__main__":
    main()
