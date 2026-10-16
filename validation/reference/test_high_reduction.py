#!/usr/bin/env python3
"""Record original Ninja results for independent four/six-line reduction tests."""
import json
import pathlib
import subprocess
import sys

from test_generated_runner import ROOT, BUILD, evaluate


def main():
    generator = ROOT / "target/debug/examples/generate_reference"
    shifts = [[0, 0, 0, 0], [2, 1, 0, 0], [4, 0, 1, 0],
              [7, 0, 0, 1], [11, 1, 2, 1], [16, 2, -1, 2]]
    cases = []
    for n in [4, 5, 6]:
        for label, expression, rank in [
            ("constant", "1", 0),
            ("polynomial", "1+q0+2*q1*q2+mu2", 2),
            ("higher", f"q0^{n+1}+mu2^2*q1^{n-3}+mu2*q2^2+1", n+1),
        ]:
            name = f"high-{n}-{label}"
            header = BUILD / f"{name}.hh"
            executable = BUILD / name
            subprocess.run(["flock", "/tmp/ninjars-symbolica-tests.lock", generator,
                            str(n), expression, header], check=True)
            subprocess.run([sys.executable, ROOT / "validation/reference/build.py",
                            "--runner-only", "--numerator-header", header,
                            "--output", executable], check=True)
            # Masses are exactly representable, with a negative imaginary part
            # only in one lane. Identical polynomials/denominators feed both codes.
            for complex_mass in [False, True]:
                masses = [[i+1, -(i+1)/16 if complex_mass else 0] for i in range(n)]
                request = f"{n} {rank} 1 1\n"
                request += "".join(" ".join(map(str, [*p, *m]))+"\n"
                                   for p, m in zip(shifts, masses))
                reference, = evaluate(executable, [request])
                cases.append(dict(name=name+("-complex" if complex_mass else "-real"),
                                  expression=expression, rank=rank, shifts=shifts[:n],
                                  masses_squared=masses, scale_squared=1, reference=reference,
                                  artifact=json.loads(executable.with_suffix(".json").read_text())))
    exceptions = []
    # Preserve rejected initial controls: the original backend reports status0
    # with nonfinite coefficients. They are diagnosed separately, not converted
    # into successful comparisons or used to relax the fixed accuracy gate.
    for n, rank, label, points in [
        (4, 2, "polynomial", [[0,0,0,0],[1,2,0,1],[3,-1,1,0],[-2,1,3,2]]),
        (5, 0, "constant", [[0,0,0,0],[2,1,0,0],[3,0,1,0],[4,0,0,1],[1,2,3,4]]),
    ]:
        masses = [[i+1,0] for i in range(n)]
        request = f"{n} {rank} 1 1\n" + "".join(" ".join(map(str,[*p,*m]))+"\n" for p,m in zip(points,masses))
        run = subprocess.run([BUILD/f"high-{n}-{label}"], input=request, text=True, capture_output=True, check=True)
        reference, = [json.loads(line) for line in run.stdout.splitlines() if line.startswith("{")]
        exceptions.append(dict(shifts=points,masses_squared=masses,rank=rank,
                               expression="1+q0+2*q1*q2+mu2" if rank else "1",
                               reference=reference,reason="original returns nonfinite coefficients; diagnostic pending"))
    output = ROOT / "validation/evidence/high-reduction.json"
    output.write_text(json.dumps(dict(cases=cases, reference_exceptions=exceptions, timings_are_acceptance_evidence=False,
                                     full_ninja_acceptance=False), indent=2)+"\n")
    print(f"Recorded {len(cases)} original Ninja high-cut controls: {output}")


if __name__ == "__main__":
    main()
