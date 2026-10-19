#!/usr/bin/env python3
"""Tabulate oneloopmaster native masters vs OneLOop Fortran on identical call streams.

Reads validation/benchmark/masters_native/results/<label>.json (masters-native-bench,
one file per oneloopmaster/Symbolica build, see build_native.py and
results/<label>.build.json) and the OneLOop replay timings of the same calls in
validation/benchmark/ninja/<case>.json (build o3-fortranrules: MadLoop's Ninja
flags; the masters-only replay goes through the same libavh_olo.a in every
build), and writes validation/benchmark/masters_native/SUMMARY.md.
"""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
NINJA = HERE.parent / "ninja"
BUILD = "o3-fortranrules"
PRIMARY = "olm-a92b148-sym-e679e8c1-tensor"
LABELS = {
    "olm-3023fa6-sym-e679e8c1-tensor":
        "oneloopmaster main 3023fa6 (264a6ce plus the B1111 tensor family, conditioned C0 dilogarithm "
        "differences, the contour box for real three-mass boxes, exact Kallen edge roots, far-root r1, "
        "labelling-independent real four-mass D0, collinear few-mass boxes and commutative DoubleFloat "
        "products) + Symbolica e679e8c1; the final benchmark revision (the pinned 149ddff returns the same "
        "bits on every call)",
    "olm-264a6ce-sym-e679e8c1-tensor":
        "oneloopmaster main 264a6ce (continuation and degenerate-kinematics fixes, hand-written C0/D0 "
        "and tensor binary64 kernels) + Symbolica e679e8c1; the previous benchmark revision",
    "olm-a92b148-sym-e679e8c1-tensor":
        "oneloopmaster `claude/native-masters` a92b148 (direct `dilog_complex_f64`, NativeTensorEvaluator "
        "B11/B111 for Ninja's rank-2/3 bubble calls) + Symbolica e679e8c1 (2c06f245 + binary64 dilogarithm)",
    "olm-c9dd82f-sym-e679e8c1":
        "oneloopmaster main c9dd82f + Symbolica e679e8c1 (registered polylog(2) callback now binary64); "
        "rank-2 bubbles as B0 only",
    "olm-c9dd82f-sym-2c06f245":
        "oneloopmaster main c9dd82f + Symbolica 2c06f245 (the NinjaRS workspace pin): MPFR dilogarithm "
        "(historical; first measurement of this stream)",
}


def us(ns):
    return f"{ns / 1000:.2f}" if ns is not None else "-"


def ratio(a, b):
    if a is None or not b:
        return "-"
    r = a / b
    return f"{r:.2f}x" if r < 10 else f"{r:.0f}x"


def bubbles(t):
    for key in ("bubbles", "bubbles_b0_only"):
        if key in t:
            return t[key]
    return None


def table(label, data):
    tensor = data.get("tensor_bubbles", False)
    lines = [
        f"## `{label}`", "", LABELS.get(label, ""), "",
        "| case | OneLOop all | native all | native/OneLOop | OL D0 | nat D0 | OL C0 | nat C0 | OL B | "
        f"nat {'B (tensor)' if tensor else 'B0 only'} | OL A0 | nat A0 | max diff | relabel |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in data["cases"]:
        name = case["case"]
        path = NINJA / f"{name}.json"
        if not path.exists():
            continue
        ref = json.loads(path.read_text())["timing"]
        if BUILD not in ref:
            continue
        ref = ref[BUILD]["median_ns"]
        t = {k: v["median_ns"] for k, v in case["timing"].items()}
        diff = max(case["max_scaled_difference_vs_oneloop"].values())
        sym = case.get("max_scaled_relabelling_deviation_native")
        sym = f"{max(sym.values()):.0e}" if sym else "n/a"
        skipped = f" ({case['skipped_calls']} calls skipped)" if case.get("skipped_calls") else ""
        lines.append(
            f"| {name}{skipped} | {us(ref['masters_only'])} | {us(t.get('all_in_call_order'))} | "
            f"{ratio(t.get('all_in_call_order'), ref['masters_only'])} | "
            f"{us(ref.get('masters_boxes'))} | {us(t.get('boxes'))} | {us(ref.get('masters_triangles'))} | "
            f"{us(t.get('triangles'))} | {us(ref.get('masters_bubbles'))} | {us(bubbles(t))} | "
            f"{us(ref.get('masters_tadpoles'))} | {us(t.get('tadpoles'))} | {diff:.1e} | {sym} |")
    return lines + [""]


def main():
    results = {p.stem: json.loads(p.read_text()) for p in sorted((HERE / "results").glob("*.json"))
               if not p.name.endswith(".build.json")}
    order = [l for l in LABELS if l in results] + [l for l in results if l not in LABELS]
    lines = [
        "# oneloopmaster native masters vs OneLOop Fortran (identical call streams)",
        "",
        "Native: oneloopmaster's direct ahead-of-time Rust backend (`NativeEvaluator<f64>`, feature",
        "`generated-evaluators`; no expression interpreter, no SymJIT), one scalar `evaluate` per call,",
        "NinjaRS release profile (thin LTO, 1 CGU). OneLOop: the same calls (`../ninja/<case>.masters.json`,",
        f"recorded from original Ninja) replayed through `ninja::avh_olo` (build `{BUILD}`, OneLOop-3.6,",
        "gfortran -O3, OneLOop warnings off). Microseconds per phase-space point (one lane's calls),",
        "median of >=7 blocks >=25 ms. OneLOop replay and olm-264a6ce native pinned to CPU 250 (2026-10-18),",
        "olm-3023fa6 native on CPU 250 (2026-10-19; earlier labels: native on CPU 29, see",
        "../cpu-crosscheck.md). 'OL B' is every bubble call (olo_b11 for",
        "Ninja's rank-2 bubbles: B11, B00, B1, B0); 'nat B0 only' evaluates just B0 for those calls",
        "(less work), 'nat B (tensor)' evaluates NativeTensorEvaluator B11 = [B0, B1, B11] (B111 for",
        "rank-3 bubbles), i.e. everything Ninja consumes. max diff = largest |native - OneLOop| / max|OneLOop|",
        "over each call's outputs; relabel = largest deviation of native C0/D0 under the cyclic and",
        "reflected relabellings of the propagators (a native self-consistency check).",
        "",
    ]
    for label in order:
        lines += table(label, results[label])
    lines += [
        "## Notes",
        "",
        "* **Dilogarithm.** With the NinjaRS workspace's Symbolica pin (2c06f245) the generated code's f64",
        "  `dilog` goes through Symbolica's registered `polylog(2, z)` callback, which converts to MPFR and",
        "  sums a series (12 us - 2 ms per call): C0/D0 were 25-500x slower than OneLOop. Symbolica e679e8c1",
        "  (`claude/oneloopmaster-integration-dilog`: 2c06f245 + a binary64 Bernoulli-series dilogarithm,",
        "  the specialization described in oneloopmaster's `STATUS.md`/`patches/README.md`) evaluates it",
        "  in 17-110 ns (`DILOG_TEST=1`); oneloopmaster a92b148 calls that kernel directly.",
        "* **Defects of earlier revisions, fixed in 264a6ce.** a92b148's native C0 with one massless and",
        "  two massive internal lines above the massive-pair threshold had the wrong imaginary part (e.g.",
        "  C0(0, 0, 490000; 173^2, 0, 173^2; mu^2=1e4)), syn-n4-r3/r4 had O(1) C0/D0 deviations, and",
        "  syn-gghhh-ggHHH-n5-r5 had a D0 near a vanishing Gram determinant at 6.8e-10 (OneLOop: 7.9e-15,",
        "  checked against 256-bit values). At 264a6ce the native C0 of syn-n6-r6 with nearly coincident",
        "  roots was at ~2e-13 versus OneLOop's 3e-15 (OneLOop uses a dedicated near-coincident dilogarithm",
        "  difference).",
        "* **Changes in 3023fa6.** 3cbe08d conditions the coincident-root dilogarithm differences (that C0",
        "  is now at 2.2e-15 against 256-bit values). 9c5490b (contour-box edge roots from exact Kallen",
        "  functions) moves the D0 of syn-gghhh-ggHHH-n5-r5 near a vanishing Gram determinant from 4.4e-12",
        "  to 3.6e-11 (OneLOop 7.9e-15) and syn-n5-r5's largest deviation from 6.2e-14 to 4.5e-13; the",
        "  physical streams keep 8.3e-14. Against the 264a6ce run, native boxes take up to 14 % longer",
        "  (4-9 % on the physical streams), and families without code changes moved by -10 to +6 %. Run",
        "  alternately on CPU 250 (best of three, five physical streams), 3023fa6's boxes take 3-5 % longer",
        "  than 264a6ce's, triangles and bubbles within 2 %, tadpoles 7-9 % less. Native/OneLOop is",
        "  0.25-0.57x on the physical streams (0.24-0.54x at 264a6ce).",
        "",
    ]
    (HERE / "SUMMARY.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
