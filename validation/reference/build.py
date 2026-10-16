#!/usr/bin/env python3
"""Build unchanged Ninja and bundled OneLOop in disposable build directories."""
import argparse
import hashlib
import json
import pathlib
import re
import shutil
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configuration(build):
    path = build / "ninja-build/config.h"
    contents = path.read_text()
    enabled = lambda name: bool(re.search(r"^\s*#\s*define\s+" + name + r"\s+1\b", contents, re.M))
    return {"config_sha256": sha256(path),
            "oneloop": enabled("NINJA_USE_ONELOOP"),
            "master_result_cache": enabled("NINJA_USE_ONELOOP_WITH_CACHE"),
            "higher_rank": enabled("NINJA_X1RANK")}


def library_inputs(build):
    """Copied numerical sources plus every generated header on include paths.

    Keep paths relative to the disposable build directory. Recording the exact
    set also detects an added header which could shadow an original include.
    Numerator headers are runner-specific and recorded separately.
    """
    paths = {path for path in (build / "ninja-source/src").rglob("*") if path.is_file()}
    paths.update(path for path in (build / "ninja-build").rglob("*")
                 if path.is_file() and path.suffix.lower() in {".h", ".hh", ".hpp", ".hxx"})
    paths.update(build / "oneloop" / name for name in ["avh_olo_foriregi.f90", "avh_olo_alias.f90"])
    return {str(path.relative_to(build)): sha256(path) for path in sorted(paths)}


def run(command, cwd, commands=None):
    print("+", " ".join(map(str, command)), flush=True)
    if commands is not None:
        commands.append({"argv": list(map(str, command)), "cwd": str(cwd)})
    subprocess.run(list(map(str, command)), cwd=cwd, check=True)


def build_libraries(build, jobs, disable_cache=False):
    commands = []
    source = build / "ninja-source"
    shutil.copytree(ROOT / "references/ninja", source,
                    ignore=shutil.ignore_patterns(".git"), dirs_exist_ok=True)
    masters = build / "oneloop"
    masters.mkdir(exist_ok=True)
    for path in [ROOT / "references/MG5_aMC_v3_7_0/vendor/IREGI/src/oneloop/avh_olo_foriregi.f90",
                 ROOT / "validation/reference/avh_olo_alias.f90"]:
        shutil.copy2(path, masters / path.name)
    initial_masters = {path.name: sha256(path) for path in masters.glob("*.f90")}
    run(["gfortran", "-O3", "-fPIC", "-c", "avh_olo_foriregi.f90", "avh_olo_alias.f90"], masters, commands)
    run(["ar", "rcs", "libavh_olo.a", "avh_olo_foriregi.o", "avh_olo_alias.o"], masters, commands)
    run(["autoreconf", "-fi"], source, commands)
    ninja_build = build / "ninja-build"
    ninja_build.mkdir(exist_ok=True)
    run([source / "configure", "--disable-shared", "--disable-autoflags",
         "--disable-f90module", "--disable-gosam",
         *(["--disable-avholo_cache"] if disable_cache else []),
         "--with-avholo=-L" + str(masters) + " -lavh_olo", "FCINCLUDE=-I" + str(masters),
         "CXXFLAGS=-O3 -DNDEBUG", "FCFLAGS=-O3"], ninja_build, commands)
    config = configuration(build)
    if not config["oneloop"] or not config["higher_rank"] or config["master_result_cache"] == disable_cache:
        raise RuntimeError("configured Ninja does not match the requested OneLOop/cache/higher-rank policy")
    inputs = library_inputs(build)
    # Building only the library avoids upstream parallel make dependencies on
    # generated Fortran module installation files unrelated to this C++ runner.
    run(["make", "-C", "src", "-j", str(jobs), "libninja.la"], ninja_build, commands)
    if (library_inputs(build) != inputs
            or {path.name: sha256(path) for path in masters.glob("*.f90")} != initial_masters):
        raise RuntimeError("copied reference sources or generated headers changed during library compilation")
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
    revisions["fortran_compiler"] = subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0]
    revisions["configuration"] = config
    revisions["build_commands"] = commands
    revisions["oneloop_source_sha256"] = sha256(masters / "avh_olo_foriregi.f90")
    revisions["alias_sha256"] = sha256(masters / "avh_olo_alias.f90")
    revisions["libraries_sha256"] = {
        path.name: sha256(path) for path in [ninja_build / "src/.libs/libninja.a", masters / "libavh_olo.a"]}
    revisions["input_files_sha256"] = inputs
    (build / "revisions.json").write_text(json.dumps(revisions, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--build-dir", type=pathlib.Path,
                        help="default target/reference, or target/reference-uncached-v2 with --benchmark")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--numerator-header", type=pathlib.Path,
                        help="Symbolica-generated Ninja callback header")
    parser.add_argument("--runner-only", action="store_true",
                        help="reuse previously built reference libraries and their revision record")
    parser.add_argument("--output", type=pathlib.Path, help="override the runner executable path")
    parser.add_argument("--disable-avholo-cache", "--disable-avholo_cache", action="store_true",
                        help="configure original Ninja with --disable-avholo_cache")
    parser.add_argument("--benchmark", action="store_true",
                        help="build the persistent benchmark runner; requires generated callbacks and uncached masters")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.benchmark and not args.numerator_header:
        parser.error("--benchmark requires --numerator-header")
    disable_cache = args.disable_avholo_cache or args.benchmark
    build = (args.build_dir or ROOT / ("target/reference-uncached-v2" if args.benchmark else "target/reference")).resolve()
    build.mkdir(parents=True, exist_ok=True)
    source = build / "ninja-source"
    masters = build / "oneloop"
    ninja_build = build / "ninja-build"
    if not args.runner_only:
        build_libraries(build, args.jobs, disable_cache)
    libraries = [ninja_build / "src/.libs/libninja.a", masters / "libavh_olo.a"]
    if not all(path.exists() for path in libraries + [build / "revisions.json"]):
        parser.error("reference libraries are missing; run once without --runner-only")
    config = configuration(build)
    if disable_cache and config["master_result_cache"]:
        parser.error("reference libraries have the master-result cache enabled; rebuild in a dedicated uncached directory")
    name = "ninja-benchmark" if args.benchmark else "ninja-generated-reference" if args.numerator_header else "ninja-reference"
    executable = args.output.resolve() if args.output else build / name
    executable.parent.mkdir(parents=True, exist_ok=True)
    extra_flags = []
    metadata = json.loads((build / "revisions.json").read_text())
    library_hashes = {path.name: sha256(path) for path in libraries}
    if args.benchmark and (metadata.get("configuration") != config
                           or metadata.get("libraries_sha256") != library_hashes
                           or not config["oneloop"] or not config["higher_rank"]):
        parser.error("benchmark library provenance/configuration is missing or changed; rebuild without --runner-only")
    if args.benchmark or "input_files_sha256" in metadata:
        if metadata.get("input_files_sha256") != library_inputs(build):
            parser.error("copied reference sources/generated headers provenance is missing or changed; rebuild without --runner-only")
    metadata["configuration"] = config
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
    runner = ROOT / "validation/reference" / ("benchmark_runner.cc" if args.benchmark else "tensor_runner.cc")
    runner_commands = []
    run(["g++", "-O3", "-DNDEBUG", "-std=c++11", "-I" + str(source / "src"),
         "-I" + str(ninja_build / "src"), "-I" + str(ninja_build), *extra_flags,
         runner, *libraries,
         fortran_library, "-Wl,-rpath," + str(fortran_library.parent), "-lquadmath",
         "-o", executable], build, runner_commands)
    if "input_files_sha256" in metadata and library_inputs(build) != metadata["input_files_sha256"]:
        raise RuntimeError("copied reference sources or generated headers changed during runner compilation")
    metadata["libraries_sha256"] = library_hashes
    metadata["runner_sha256"] = sha256(runner)
    metadata["runner_commands"] = runner_commands
    metadata["runner_compiler"] = subprocess.check_output(["g++", "--version"], text=True).splitlines()[0]
    metadata["executable_sha256"] = sha256(executable)
    metadata["execution"] = "persistent scalar benchmark control" if args.benchmark else "streamed correctness requests"
    metadata["performance_acceptance"] = False
    executable.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(executable)


if __name__ == "__main__":
    main()
