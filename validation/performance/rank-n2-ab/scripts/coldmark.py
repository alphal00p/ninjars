#!/usr/bin/env python3
"""Experiment only (not applied to the crates): insert #[cold] before every
rank n+2 function of the fast engine that is not #[inline]/#[inline(always)],
so that LLVM emits them into .text.unlikely.* sections.

usage: coldmark.py <checkout>/crates/ninjars/src/fast

rust-lld concatenates .text.unlikely.* with the other .text.* input sections
in input order; the cold sections are only placed apart when the CLI is linked
with -z keep-text-section-prefix (relink_cli.sh).
"""
import pathlib
import re
import sys

D = pathlib.Path(sys.argv[1])
RULES = {
    "engine_x2.rs": r"^\s*(pub\(super\) )?fn (new|evaluate|prepare|boxes|triangles|bubbles|tadpoles|integrate|"
                    r"clear|prepare_out|rank|mu_expansion|mu_expansion_x2|t3_expansion|t2_expansion|"
                    r"t3_expansion_x2|t2_expansion_x2)\b",
    "bases_x2.rs": r"^\s*(pub\(super\) )?fn (new|complete|bubble_basis|bubble_basis_x2|choose)\b",
    "subtractions_series_x2.rs": r"^\s*pub\(super\) fn (bubble_from_triangle_all|bubble_from_triangle_spurious|"
                                 r"tadpole_from_triangle|tadpole_from_bubble)\b",
    "tensor.rs": r"^\s*fn (mu_expansion_x2|t3_expansion_x2|t2_expansion_x2|reserve_x2)\b",
    "tensor_mu.rs": r"^\s*fn (mu_expansion_x2|t3_expansion_x2|t2_expansion_x2|reserve_x2)\b",
    "replay.rs": r"^\s*fn \w+_x2\b",
    "adapters.rs": r"^\s*fn \w+_x2\b",
}
total = 0
for name, rx in RULES.items():
    p = D / name
    out, in_tests, n = [], False, 0
    for line in p.read_text().splitlines(keepends=True):
        if re.match(r"^mod tests", line):
            in_tests = True
        prev = out[-1] if out else ""
        if not in_tests and re.match(rx, line) and not re.search(r"#\[(cold|inline)", prev):
            out.append(re.match(r"^\s*", line).group(0) + "#[cold]\n")
            n += 1
        out.append(line)
    p.write_text("".join(out))
    print(name, n)
    total += n
print("total", total)
