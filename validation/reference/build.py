#!/usr/bin/env python3
"""Build unchanged Ninja and bundled OneLOop in disposable build directories."""
import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[2]


def run(command, cwd):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, check=True)


def build_libraries(build, jobs):
    source = build / "ninja-source"
    shutil.copytree(ROOT / "references/ninja", source,
                    ignore=shutil.ignore_patterns(".git"), dirs_exist_ok=True)
    masters = build / "oneloop"
    masters.mkdir(exist_ok=True)
    for path in [ROOT / "references/MG5_aMC_v3_7_0/vendor/IREGI/src/oneloop/avh_olo_foriregi.f90",
                 ROOT / "validation/reference/avh_olo_alias.f90"]:
        shutil.copy2(path, masters / path.name)
    run(["gfortran", "-O3", "-fPIC", "-c", "avh_olo_foriregi.f90", "avh_olo_alias.f90"], masters)
    run(["ar", "rcs", "libavh_olo.a", "avh_olo_foriregi.o", "avh_olo_alias.o"], masters)
    run(["autoreconf", "-fi"], source)
    ninja_build = build / "ninja-build"
    ninja_build.mkdir(exist_ok=True)
    run([source / "configure", "--disable-shared", "--disable-autoflags",
         "--disable-f90module", "--disable-gosam",
         "--with-avholo=-L" + str(masters) + " -lavh_olo", "FCINCLUDE=-I" + str(masters),
         "CXXFLAGS=-O3 -DNDEBUG", "FCFLAGS=-O3"], ninja_build)
    # Building only the library avoids upstream parallel make dependencies on
    # generated Fortran module installation files unrelated to this C++ runner.
    run(["make", "-C", "src", "-j", str(jobs), "libninja.la"], ninja_build)
    revisions = {}
    for name in ["ninja", "MG5_aMC_v3_7_0"]:
        checkout = ROOT / "references" / name
        if (checkout / ".git").exists():
            commit = subprocess.check_output(
                ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
            ).strip()
            dirty = subprocess.check_output(
                ["git", "-C", str(checkout), "status", "--porcelain"], text=True
            ).strip()
            revisions[name] = {"commit": commit, "dirty": bool(dirty)}
        else:
            revisions[name] = {"source": "release directory", "oneloop_sha256":
                hashlib.sha256((masters / "avh_olo_foriregi.f90").read_bytes()).hexdigest()}
    revisions["compiler"] = subprocess.check_output(["g++", "--version"], text=True).splitlines()[0]
    (build / "revisions.json").write_text(json.dumps(revisions, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--build-dir", type=pathlib.Path, default=ROOT / "target/reference")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--numerator-header", type=pathlib.Path,
                        help="Symbolica-generated Ninja callback header")
    parser.add_argument("--runner-only", action="store_true",
                        help="reuse previously built reference libraries and their revision record")
    parser.add_argument("--output", type=pathlib.Path, help="override the runner executable path")
    args = parser.parse_args()
    build = args.build_dir.resolve()
    build.mkdir(parents=True, exist_ok=True)
    source = build / "ninja-source"
    masters = build / "oneloop"
    ninja_build = build / "ninja-build"
    if not args.runner_only:
        build_libraries(build, args.jobs)
    libraries = [ninja_build / "src/.libs/libninja.a", masters / "libavh_olo.a"]
    if not all(path.exists() for path in libraries + [build / "revisions.json"]):
        parser.error("reference libraries are missing; run once without --runner-only")
    name = "ninja-generated-reference" if args.numerator_header else "ninja-reference"
    executable = args.output.resolve() if args.output else build / name
    executable.parent.mkdir(parents=True, exist_ok=True)
    extra_flags = []
    metadata = json.loads((build / "revisions.json").read_text())
    if args.numerator_header:
        data = args.numerator_header.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        # Separate outputs can compile concurrently without replacing each
        # other's numerator while the C++ preprocessor is reading it.
        header = build / f"generated-{digest}.hh"
        header.write_bytes(data)
        extra_flags = ["-I" + str(build), f'-DNINJARS_GENERATED_NUMERATOR_HEADER="{header.name}"']
        metadata["generated_numerator_sha256"] = digest
    fortran_library = pathlib.Path(subprocess.check_output(
        ["gfortran", "-print-file-name=libgfortran.so"], text=True
    ).strip())
    run(["g++", "-O3", "-DNDEBUG", "-std=c++11", "-I" + str(source / "src"),
         "-I" + str(ninja_build / "src"), *extra_flags,
         ROOT / "validation/reference/tensor_runner.cc", *libraries,
         fortran_library, "-Wl,-rpath," + str(fortran_library.parent), "-lquadmath",
         "-o", executable], build)
    metadata["libraries_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                     for path in libraries}
    executable.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(executable)


if __name__ == "__main__":
    main()
