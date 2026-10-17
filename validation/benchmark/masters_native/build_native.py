#!/usr/bin/env python3
"""Build masters-native-bench against exact oneloopmaster / Symbolica git revisions.

  python3 validation/benchmark/masters_native/build_native.py --label LABEL \
      --oneloop REV --symbolica REV [--tensor] [--jobs 24]

Each revision is exported with `git archive` (no change to the shared
repositories under references/) into target/benchmark/masters-native-src/, a
copy of this crate's manifest is generated in
target/benchmark/masters-native/<LABEL>/ that points at those snapshots (src/
is the committed harness, symlinked), and a release binary is built with the
committed Cargo.lock, offline. --tensor enables the NativeTensorEvaluator
B11/B111 route for Ninja's rank-2/3 bubble calls (needs a oneloopmaster with
src/native/tensor.rs). Provenance (revisions, commands, binary sha256) is
written to target/benchmark/masters-native/<LABEL>/build.json and printed.
"""
import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REFS = pathlib.Path("/shared/external/ninja/references")
SRC = ROOT / "target/benchmark/masters-native-src"
OUT = ROOT / "target/benchmark/masters-native"


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def snapshot(repo, rev):
    full = subprocess.check_output(["git", "-C", str(REFS / repo), "rev-parse", f"{rev}^{{commit}}"], text=True).strip()
    dest = SRC / f"{repo}-{full[:8]}"
    if not dest.is_dir():
        SRC.mkdir(parents=True, exist_ok=True)
        tmp = SRC / f".{dest.name}.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        archive = subprocess.run(["git", "-C", str(REFS / repo), "archive", full], check=True, capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", str(tmp)], input=archive, check=True)
        tmp.rename(dest)
    return full, dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--oneloop", required=True, help="oneloopmaster revision")
    ap.add_argument("--symbolica", required=True, help="Symbolica revision (also numerica, graphica)")
    ap.add_argument("--tensor", action="store_true")
    ap.add_argument("--jobs", type=int, default=24)
    args = ap.parse_args()

    oneloop_rev, oneloop_dir = snapshot("oneloopmaster", args.oneloop)
    symbolica_rev, symbolica_dir = snapshot("symbolica", args.symbolica)

    build = OUT / args.label
    build.mkdir(parents=True, exist_ok=True)
    manifest = (HERE / "Cargo.toml").read_text()
    manifest = re.sub(r'oneloop = \{ path = "[^"]*"', f'oneloop = {{ path = "{oneloop_dir}"', manifest)
    manifest = manifest.replace('"../../../references/symbolica"', f'"{symbolica_dir}"')
    manifest = manifest.replace('"../../../references/symbolica/lib/', f'"{symbolica_dir}/lib/')
    assert str(oneloop_dir) in manifest and str(symbolica_dir) in manifest
    (build / "Cargo.toml").write_text(manifest)
    shutil.copy(HERE / "Cargo.lock", build / "Cargo.lock")
    link = build / "src"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(HERE / "src")

    command = ["cargo", "build", "--release", "--offline", "-j", str(args.jobs)]
    if args.tensor:
        command += ["--features", "tensor"]
    env = dict(os.environ, CARGO_TARGET_DIR=str(build / "target"))
    print("+", " ".join(command), f"(in {build})", flush=True)
    subprocess.run(command, cwd=build, env=env, check=True)
    binary = build / "target/release/masters-native-bench"
    record = {
        "label": args.label,
        "oneloopmaster_revision": oneloop_rev,
        "symbolica_revision": symbolica_rev,
        "tensor_feature": args.tensor,
        "command": command,
        "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
        "harness_main_rs_sha256": sha256(HERE / "src/main.rs"),
        "binary": str(binary),
        "binary_sha256": sha256(binary),
    }
    (build / "build.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
