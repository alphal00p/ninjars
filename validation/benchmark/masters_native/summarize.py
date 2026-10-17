#!/usr/bin/env python3
"""Tabulate oneloopmaster native masters vs OneLOop Fortran on identical call streams.

Reads validation/benchmark/masters_native/results.json (masters-native-bench) and
validation/benchmark/ninja/<case>.json (OneLOop per-family replay timings of the
'o3' build) and writes validation/benchmark/masters_native/SUMMARY.md.
"""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
NINJA = HERE.parent / "ninja"
BUILD = "o3"


def us(ns):
    return f"{ns / 1000:.2f}" if ns is not None else "-"


def ratio(a, b):
    return f"{a / b:.0f}x" if a is not None and b else "-"


def main():
    native = json.loads((HERE / "results.json").read_text())
    lines = [
        "# oneloopmaster native masters vs OneLOop Fortran (identical call streams)",
        "",
        "Native: `NativeEvaluator<f64>` (feature `generated-evaluators`, direct ahead-of-time Rust, no",
        "interpreter/JIT), one scalar `evaluate` per call, built with NinjaRS's release profile",
        "(thin LTO, 1 CGU). OneLOop: the same calls replayed through `ninja::avh_olo` (Ninja build",
        f"`{BUILD}`, OneLOop-3.6, gfortran -O3). Both pinned to CPU 29; microseconds per phase-space",
        "point (one lane's calls), median of >=7 blocks >=25 ms. 'B' native is B0 only, whereas OneLOop's",
        "rank-2 bubble call (olo_b11) returns B11, B00, B1 and B0. max diff = largest scaled",
        "difference of native vs recorded OneLOop results (D0/C0/B0/A0).",
        "",
        "| case | OneLOop all | native all | ratio | OL D0 | nat D0 | OL C0 | nat C0 | OL B | nat B0 | OL A0 | nat A0 | max diff |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in native["cases"]:
        name = case["case"]
        ref = json.loads((NINJA / f"{name}.json").read_text())["timing"][BUILD]["median_ns"]
        t = {k: v["median_ns"] for k, v in case["timing"].items()}
        diff = max(case["max_scaled_difference_vs_oneloop"].values())
        lines.append(
            f"| {name} | {us(ref['masters_only'])} | {us(t.get('all_in_call_order'))} | "
            f"{ratio(t.get('all_in_call_order'), ref['masters_only'])} | "
            f"{us(ref.get('masters_boxes'))} | {us(t.get('boxes'))} | {us(ref.get('masters_triangles'))} | "
            f"{us(t.get('triangles'))} | {us(ref.get('masters_bubbles'))} | {us(t.get('bubbles_b0_only'))} | "
            f"{us(ref.get('masters_tadpoles'))} | {us(t.get('tadpoles'))} | {diff:.1e} |")
    lines += [
        "",
        "Cause of the native C0/D0 cost: the f64 dilogarithm of the generated code",
        "(`oneloopmaster/src/native/primitives.rs`, `<f64 as NativeFloat>::dilog`) calls Symbolica's",
        "registered `polylog(2, z)` for `Complex<f64>`, which converts to MPFR `Float` and sums a series",
        "(`references/symbolica` 2c06f245, `src/transcendental.rs` `polylog_eval_complex_f64`).",
        "Measured with `DILOG_TEST=1 masters-native-bench /dev/null` on CPU 29: 12-33 us for real",
        "arguments in (-3, 1), 108 us at z=2, 270 us at 0.5+0.5i, 2.1 ms at 0.9+0.3i, 0.69 ms at 0.3-0.8i.",
        "",
    ]
    (HERE / "SUMMARY.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
