#!/usr/bin/env python3
"""Generic-engine rank twins for the rank n+2 cost measurement.

usage: make_twin_fixtures.py OUT_DIR

Writes, for n = 2..7 and R in {n+1, n+2}, an n-gon in FeynKit's compact DOT
dialect (phi3 model, unit internal masses, the layout of
examples/graphs/{triangle,box,heptagon-rank8}.dot) with the graph numerator

    N = (1 + q0/2 + q1/3 - q2/5 + q3/7)^R + mu2 (2 - q0 + q3/3)^(R-2) + mu2^floor(R/2)

(full rank R in q and in the mu2-weighted sense), and a schema-version-1 JSON
fixture with two exact off-shell points: the first n-1 momenta of the two
samples of examples/graphs/heptagon-rank8.json. The twins of one n share
their kinematics and differ only in the numerator rank.
"""
import json
import pathlib
import sys

def heptagon_samples(root):
    d = json.loads((root / "examples/graphs/heptagon-rank8.json").read_text())
    return [s["external_momenta"] for s in d["samples"]]


def numerator(rank):
    q = "ninjars::q"
    lin = f"(1 + {q}0/2 + {q}1/3 - {q}2/5 + {q}3/7)^{rank}"
    mu = f"ninjars::mu2*(2 - {q}0 + {q}3/3)^{rank - 2}" if rank >= 2 else "ninjars::mu2"
    return f"{lin} + {mu} + ninjars::mu2^{rank // 2}"


def dot(name, n, rank):
    v = "abcdefghij"[:n]
    lines = [f"digraph {name.replace('-', '_')} {{", f'    num = "{numerator(rank)}";', "    ext [style=invis];"]
    lines.append(f'    ext -> {v[0]} [particle="phi", num="1"];')
    if n == 2:
        lines.append(f'    {v[1]} -> ext [particle="phi", num="1"];')
    else:
        lines.append(f'    ext -> {v[1]} [particle="phi", num="1"];')
        for x in v[2:]:
            lines.append(f'    {x} -> ext [particle="phi", num="1"];')
    for i in range(n):
        extra = ", lmb_id=0" if i == 0 else ""
        lines.append(f'    {v[i]} -> {v[(i + 1) % n]} [particle="phi"{extra}, num="1"];')
    lines.append("}")
    return "\n".join(lines) + "\n"


def main():
    out = pathlib.Path(sys.argv[1])
    root = pathlib.Path(__file__).resolve().parents[4] if len(sys.argv) < 3 else pathlib.Path(sys.argv[2])
    samples = heptagon_samples(root)
    out.mkdir(parents=True, exist_ok=True)
    for n in range(2, 8):
        for rank in (n + 1, n + 2):
            name = f"ngon-n{n}-r{rank}"
            (out / f"{name}.dot").write_text(dot(name, n, rank))
            fixture = {
                "schema_version": 1,
                "graph": f"{name}.dot",
                "model": "phi3",
                "scale_squared": "1",
                "parameter_overrides": {"UFO::mass": "1"},
                "samples": [{"external_momenta": {str(i): s[str(i)] for i in range(n - 1)}} for s in samples],
            }
            (out / f"{name}.json").write_text(json.dumps(fixture, indent=1) + "\n")
    print(f"wrote {len(list(out.glob('*.json')))} fixtures to {out}")


if __name__ == "__main__":
    main()
