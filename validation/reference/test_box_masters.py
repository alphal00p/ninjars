#!/usr/bin/env python3
"""Independent D0 values through original Ninja/OneLOop and scalar reduction."""
import hashlib
import json
import math
import pathlib
import subprocess
from fractions import Fraction

ROOT = pathlib.Path(__file__).resolve().parents[2]
BUILD = ROOT / "target/reference"

def run_json(executable, requests, extra=()):
    output = subprocess.run([executable,*extra], input=requests, text=True, capture_output=True, check=True)
    return [json.loads(line) for line in output.stdout.splitlines() if line.startswith("{")]

def main():
    libraries = [BUILD / "ninja-build/src/.libs/libninja.a", BUILD / "oneloop/libavh_olo.a"]
    fortran = pathlib.Path(subprocess.check_output(["gfortran", "-print-file-name=libgfortran.so"], text=True).strip())
    source = ROOT / "validation/reference/box_master_probe.cc"
    executable = BUILD / "box-master-probe"
    subprocess.run(["g++", "-std=c++11", "-O3", "-I"+str(BUILD/"ninja-source/src"), "-I"+str(BUILD/"ninja-build/src"), source, *libraries, fortran, "-Wl,-rpath,"+str(fortran.parent), "-lquadmath", "-o", executable], check=True)
    spatial = [[0,0,0,0],[0,1,0,0],[0,0,2,0],[0,0,0,3]]
    timelike = [[0,0,0,0],[2,1,0,0],[3,0,1,0],[4,0,0,1]]
    cases = []
    def add(name, masses, shifts=None, invariants=None, scale=1):
        cases.append(dict(name=name, invariants=invariants, masses_squared=masses, scale_squared=scale, shifts=shifts))
    # Literal test data: equality to the original amplitude checks its S-matrix
    # routing against the explicit wrapper argument order independently.
    spatial_invariants = [-1,-5,-13,-9,-4,-10]
    add("unequal_real_spacelike", [[1,0],[2,0],[3,0],[4,0]], spatial, spatial_invariants)
    complex_masses = [[1,"-1/8"],[2,"-1/4"],[3,"-3/8"],[4,"-1/2"]]
    add("unequal_complex_timelike", complex_masses, timelike, [3,-1,-1,15,8,2])
    add("massless_offshell", [[0,0]]*4, spatial, spatial_invariants)
    add("two_massive_offshell", [[0,0],[0,0],[2,0],[3,0]], spatial, spatial_invariants)
    add("timelike_unit_mass", [[1,0]]*4, timelike, [3,-1,-1,15,8,2])
    add("massless_onshell_poles", [[0,0]]*4, invariants=[0,0,0,0,-3,-5], scale=2)
    # Only the independent binary64 reference needs this numerical conversion.
    def number(value): return str(float(Fraction(value)))
    requests = "".join(" ".join(map(number,[*case["invariants"], *[part for mass in case["masses_squared"] for part in mass],case["scale_squared"]]))+"\n" for case in cases)
    results = run_json(executable,requests)
    assert len(results)==len(cases)
    for case,result in zip(cases,results):
        assert all(pair is not None and all(math.isfinite(x) for x in pair) for pair in result["epsilon"]), (case,result)
        case["epsilon"] = result["epsilon"]
        if case["shifts"] is not None:
            request = f'4 0 1 {case["scale_squared"]}\n'
            request += "".join(" ".join(map(number,[*shift,*mass]))+"\n" for shift,mass in zip(case["shifts"],case["masses_squared"]))
            request += "1 0\n"
            integrated, = run_json(BUILD/"ninja-reference",request)
            assert integrated["status"]==0, integrated
            for a,b in zip(integrated["epsilon"],result["epsilon"]):
                assert a is not None and all(abs(x-y)<=1e-9*(1+abs(y)) for x,y in zip(a,b)), (case,integrated)
            case["constant_numerator_reduction"] = integrated
    metadata=json.loads((BUILD/"revisions.json").read_text())
    metadata["libraries_sha256"]={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in libraries}
    metadata["probe_sha256"]=hashlib.sha256(source.read_bytes()).hexdigest()
    exceptional_request=" ".join(map(number,[*spatial_invariants,*[part for mass in complex_masses for part in mass],1]))+"\n"
    exceptional, = run_json(executable,exceptional_request)
    assert exceptional["epsilon"] == [[0,0],[0,0],[0,0]]
    # The original cache key omits s14 (called p4 in avholo_interface.cc).
    cache_inputs=[[-1,-5,-13,s14,-4,-10,1,0,2,0,3,0,4,0,1] for s14 in [-9,-8]]
    cache_requests="".join(" ".join(map(number,values))+"\n" for values in cache_inputs)
    cleared=run_json(executable,cache_requests)
    retained=run_json(executable,cache_requests,["--keep-cache"])
    assert abs(cleared[0]["epsilon"][0][0]-cleared[1]["epsilon"][0][0])>1e-6
    assert retained[0]["epsilon"]==retained[1]["epsilon"]==cleared[0]["epsilon"]
    evidence={"invariant_order":["s21","s32","s43","s14","s31","s42"],"epsilon_order":["finite","single_pole","double_pole"],"cases":cases,"reference":metadata,"timings_are_acceptance_evidence":False,
        "cache_key_probe":{"argument_order":"six invariants, four complex squared masses, squared scale","inputs":cache_inputs,"cache_cleared":cleared,"cache_retained":retained,"finding":"Original cached D0 key omits s14; changing only this invariant incorrectly reuses the first result. All acceptance controls in cases clear the cache."},
        "exceptional_reference_points":[{"invariants":spatial_invariants,"masses_squared":complex_masses,"scale_squared":1,**exceptional,
        "reason":"OneLOop 3.4 boxc returns zero when all candidate Kallen lambdas are nonpositive; avh_olo_foriregi.f90:5689. Ninja suppresses the Fortran diagnostic. These zeros are not correctness controls."}]}
    path=ROOT/"validation/evidence/box-masters.json"
    path.write_text(json.dumps(evidence,indent=2)+"\n")
    print(f"Validated {len(cases)} D0 controls, including five constant-numerator reductions: {path}")

if __name__ == "__main__": main()
