#!/usr/bin/env python3
"""Compare the machine code of the rank <= n+1 hot functions of two binaries.

usage: dis_compare.py [--no-alias] BIN_A BIN_B [REGEX]

Functions are matched by demangled name (generic instances included) among
the symbols whose name matches REGEX (default: the fast engine, its geometry,
tensor numerators and replay, the native master providers and oneloopmaster's
native binary64 masters, and the case/pipeline bench drivers), excluding the
rank n+2 modules. Each body is disassembled with objdump and normalised
(addresses, immediates that are addresses, and jump targets removed), then
compared. Prints identical/different counts and, for every differing
function, the instruction counts of both sides.

By default B's `FastAmplitude::evaluate_up_to_n1` is matched with A's
`evaluate` (for a base from before the rank n+2 split). `--no-alias` compares
the names as they are, for two binaries that both have `evaluate_up_to_n1`
(main 0d0d7e7 or later).
"""
import collections
import re
import subprocess
import sys

# The CLI package `ninjars-cli` builds the binary crate `ninjars` ([[bin]] name),
# so its modules demangle as `ninjars::case_bench::...`; the library crate
# `ninjars` has no modules of these names.
DEFAULT = (r"ninjars::fast::|ninjars_symbolica::oneloop_native|oneloop::native::|"
           r"ninjars::case_bench::|ninjars::pipeline_bench::|ninjars::fast_runtime::")
EXCLUDE = re.compile(r"engine_x2|bases_x2|subtractions_x2|subtractions_series_x2|_x2\b|x2::|RankN2|B1111|BubbleRank4|bubble::<2, 4, 5>")


def symbols(binary, pattern, alias=False):
    """Text symbols matching `pattern`; with `alias`, B's renamed rank <= n+1
    body `FastAmplitude::evaluate_up_to_n1` is matched with A's `evaluate`, and
    B's new `evaluate` dispatchers are listed separately."""
    out = subprocess.run(["nm", "-C", "-S", "--defined-only", binary], capture_output=True, text=True).stdout
    found = collections.defaultdict(list)
    for line in out.splitlines():
        parts = line.split(" ", 3)
        if len(parts) < 4 or parts[2] not in "tTwW":
            continue
        addr, size, _, name = parts
        if pattern.search(name) and not EXCLUDE.search(name):
            if alias:
                if "FastAmplitude>::evaluate::<" in name:
                    name = "[dispatcher] " + name
                name = name.replace("FastAmplitude>::evaluate_up_to_n1::<", "FastAmplitude>::evaluate::<")
            found[name].append((int(addr, 16), int(size, 16)))
    return found


NORM = [
    (re.compile(r"^\s*[0-9a-f]+:\s*"), ""),
    (re.compile(r"\s(# )?[0-9a-f]+ <.*$"), " ADDR"),
    (re.compile(r"0x[0-9a-f]+"), "HEX"),
    (re.compile(r"\s+"), " "),
]


def body(binary, addr, size):
    out = subprocess.run(["objdump", "-d", "--no-show-raw-insn", "-C", f"--start-address={addr}",
                          f"--stop-address={addr + size}", binary], capture_output=True, text=True).stdout
    lines = []
    started = False
    for line in out.splitlines():
        if line.endswith(">:"):
            started = True
            continue
        if not started or not line.strip():
            continue
        for rx, rep in NORM:
            line = rx.sub(rep, line)
        line = line.strip()
        if line and not line.startswith(("nop", "data16", "cs nop", "xchg %ax,%ax", "int3")):
            lines.append(line)
    return lines


def main():
    args = sys.argv[1:]
    alias = "--no-alias" not in args
    args = [arg for arg in args if arg != "--no-alias"]
    a_bin, b_bin = args[0], args[1]
    pattern = re.compile(args[2] if len(args) > 2 else DEFAULT)
    a, b = symbols(a_bin, pattern), symbols(b_bin, pattern, alias=alias)
    common = sorted(set(a) & set(b))
    same, diff = 0, []
    for name in common:
        if len(a[name]) != len(b[name]):
            diff.append((name, sum(s for _, s in a[name]), sum(s for _, s in b[name]), "instance count"))
            continue
        for (aa, sa), (ab, sb) in zip(sorted(a[name], key=lambda x: x[1]), sorted(b[name], key=lambda x: x[1])):
            la, lb = body(a_bin, aa, sa), body(b_bin, ab, sb)
            if la == lb:
                same += 1
            else:
                diff.append((name, len(la), len(lb), "code"))
    print(f"functions compared: {same + len(diff)}; identical after normalisation: {same}; different: {len(diff)}")
    print(f"only in A: {len(set(a) - set(b))}; only in B (not rank n+2): {len(set(b) - set(a))}")
    for name, x, y, why in diff:
        print(f"  DIFF [{why}] {x} -> {y} instructions: {name[:200]}")
    for name in sorted(set(a) - set(b))[:40]:
        print(f"  only A: {name[:200]}")
    for name in sorted(set(b) - set(a))[:40]:
        print(f"  only B: {name[:200]}")


if __name__ == "__main__":
    main()
