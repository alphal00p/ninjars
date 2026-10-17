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
        "median of >=7 blocks >=25 ms, both pinned to CPU 29. 'OL B' is every bubble call (olo_b11 for",
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
        "* **Native C0 defect (unchanged by the dilogarithm).** C0 with one massless and two massive",
        "  internal lines above the massive-pair threshold has the wrong imaginary part, e.g.",
        "  C0(0, 0, 490000; 173^2, 0, 173^2; mu^2=1e4): native 3.4587e-5 (real; |OneLOop| = 3.4587e-5),",
        "  OneLOop -5.698e-6 - 3.411e-5 i (OneLOop 3.4 and 3.6 agree; the pair cut at p^2 > 4 m^2 requires",
        "  an imaginary part). In syn-n6-r6, C0(0, 70988, 271317; 173^2, 0, 173^2) is not even invariant",
        "  under relabelling (deviation 1.7). No physical case has such a triangle (all-massive top loops).",
        "* `syn-n6-r7`: the native B111 differs from OneLOop by 2.5e-9 (scaled) on one rank-3 bubble;",
        "  `syn-gghhh-ggHHH-n5-r5`: a D0 differs by 6.8e-10. Not investigated further (synthetic).",
        "",
    ]
    (HERE / "SUMMARY.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
