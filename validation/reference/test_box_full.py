#!/usr/bin/env python3
"""Probe original higher-rank full-box reconstruction with a generated c5 term."""
import hashlib
import json
import pathlib
import subprocess

ROOT=pathlib.Path(__file__).resolve().parents[2]
BUILD=ROOT/"target/reference"

def main():
    header=BUILD/"box-c5-only.hh"
    subprocess.run([ROOT/"target/debug/examples/generate_reference","4","(a0*q0-a1*q1-a2*q2-a3*q3)*mu2^2",header,"a0","a1","a2","a3"],check=True)
    source=ROOT/"validation/reference/box_full_probe.cc"
    executable=BUILD/"box-full-probe"
    fortran=pathlib.Path(subprocess.check_output(["gfortran","-print-file-name=libgfortran.so"],text=True).strip())
    subprocess.run(["g++","-std=c++11","-O3","-I"+str(BUILD/"ninja-source/src"),"-I"+str(BUILD/"ninja-build/src"),"-I"+str(BUILD),source,BUILD/"ninja-build/src/.libs/libninja.a",BUILD/"oneloop/libavh_olo.a",fortran,"-Wl,-rpath,"+str(fortran.parent),"-lquadmath","-o",executable],check=True)
    output=subprocess.check_output([executable],text=True)
    rows=[json.loads(line) for line in output.splitlines() if line.startswith("{")]
    assert len(rows)==2
    for row in rows:
        assert row["status"]==0,row
        for index in [0,1,2,4]: assert max(map(abs,row["coefficients"][index]))<1e-9,row
        assert abs(row["coefficients"][5][0]-1)<1e-9 and abs(row["coefficients"][5][1])<1e-9,row
        row["reconstruction_matches"]=all(abs(a-b)<1e-9*(1+abs(b)) for a,b in zip(row["reconstructed"],row["numerator"]))
    assert abs(rows[0]["coefficients"][3][0]-14)<1e-9 and not rows[0]["reconstruction_matches"],rows
    assert max(map(abs,rows[1]["coefficients"][3]))<1e-9 and rows[1]["reconstruction_matches"],rows
    reference=json.loads((BUILD/"revisions.json").read_text())
    reference["libraries_sha256"]={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in [BUILD/"ninja-build/src/.libs/libninja.a",BUILD/"oneloop/libavh_olo.a"]}
    evidence={"reference":reference,"probe_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"numerator_sha256":hashlib.sha256(header.read_bytes()).hexdigest(),"numerator":"(q dot Vort)*mu2^2, with Vfirst=0 and Vort taken from the original cut", "method":"Call unchanged original evaluateFullBox through access-control-only header macro; generated Symbolica numerator and direct original residue evaluation", "observations":rows,"scope":"Spurious c3 reconstruction only; c0/c4 and hence direct box integrated contribution are unaffected."}
    path=ROOT/"validation/evidence/box-full-reconstruction.json";path.write_text(json.dumps(evidence,indent=2)+"\n")
    print(f"Confirmed original full-box no-mu c3 discrepancy at scale15; mu-expansion control passes: {path}")

if __name__=="__main__":main()
