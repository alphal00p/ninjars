#!/usr/bin/env python3
"""Summarize the arbitrary-precision runs of `run.sh` for the benchmark report.

Reads p700.json, p1000.json, dd.json and fast.json (written by `run.sh`) and
writes arb_summary.json next to them and docs/benchmark/data/arbitrary_precision.json.
Correct digits of a result x are -log10(|x - ref| / |ref|) for the finite part,
with the 1000-bit evaluation as reference; the reported value of the finite
part is the 1000-bit one, rounded to 200 significant digits.
"""

import json
import math
import pathlib
from decimal import Decimal, getcontext

getcontext().prec = 400

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DIGITS = 200


def load(name):
    report = json.loads((HERE / name).read_text())
    order = report["result_order"]
    return [
        {key: tuple(Decimal(part) for part in value) for key, value in zip(order, lane["values"])}
        for lane in sorted(report["results"], key=lambda lane: lane["lane"])
    ]


def modulus(z):
    return (z[0] * z[0] + z[1] * z[1]).sqrt()


def digits(value, reference):
    error = modulus((value[0] - reference[0], value[1] - reference[1]))
    if error == 0:
        return math.inf
    return float(-(error / modulus(reference)).log10())


def significant(value):
    return format(value, f".{DIGITS - 1}e").replace("e-0", "e-").replace("e+0", "e+")


def main():
    reference, p700, dd, fast = (load(name) for name in ("p1000.json", "p700.json", "dd.json", "fast.json"))
    points = []
    for index, exact in enumerate(reference):
        poles = [modulus(exact[key]) for key in ("single_pole", "double_pole", "rational")]
        points.append(
            {
                "id": str(index),
                "re": significant(exact["finite"][0]),
                "im": significant(exact["finite"][1]),
                "digits_700_vs_1000": digits(p700[index]["finite"], exact["finite"]),
                "digits_double_float": digits(dd[index]["finite"], exact["finite"]),
                "digits_binary64_fast": digits(fast[index]["finite"], exact["finite"]),
                "max_abs_pole_1000": f"{max(poles):.2e}",
            }
        )
    summary = json.dumps({"points": points}, indent=1) + "\n"
    (HERE / "arb_summary.json").write_text(summary)
    (ROOT / "docs" / "benchmark" / "data" / "arbitrary_precision.json").write_text(summary)
    for point in points:
        print(
            point["id"],
            f"700 bits {point['digits_700_vs_1000']:.1f}",
            f"double-float {point['digits_double_float']:.1f}",
            f"binary64 {point['digits_binary64_fast']:.1f}",
            f"max |pole|, |rational| {point['max_abs_pole_1000']}",
        )


if __name__ == "__main__":
    main()
