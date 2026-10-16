#!/usr/bin/env python3
"""Fetch pinned source dependencies and apply their reviewed local patches."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def git(checkout, *args, capture=False):
    return subprocess.run(
        ["git", "-C", str(checkout), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    ).stdout


def prepare(dependency):
    checkout = ROOT / dependency["path"]
    revision = dependency["revision"]
    patches = []
    if len(dependency.get("patches", [])) > 1:
        raise RuntimeError("Use one consolidated base-to-head patch per dependency")
    for entry in dependency.get("patches", []):
        patch = ROOT / entry["path"]
        if hashlib.sha256(patch.read_bytes()).hexdigest() != entry["sha256"]:
            raise RuntimeError(f"Patch checksum mismatch: {patch}")
        patches.append(patch)

    if not checkout.exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", dependency["url"], str(checkout)],
            check=True,
        )
        git(checkout, "checkout", "--detach", revision)
    elif not (checkout / ".git").exists():
        raise RuntimeError(f"Existing path is not a Git checkout: {checkout}")

    head = git(checkout, "rev-parse", "HEAD", capture=True).strip()
    allowed = [revision, *dependency.get("patched_revisions", [])]
    if head not in allowed:
        raise RuntimeError(
            f"{checkout}: found {head}, expected one of {allowed}. "
            "Existing checkout left unchanged."
        )
    if patches:
        # Check the consolidated diff atomically. Reverse-check recognizes
        # already applied patches without committing or modifying local history.
        reverse = subprocess.run(
            ["git", "-C", str(checkout), "apply", "--reverse", "--check", *map(str, reversed(patches))],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if reverse.returncode:
            git(checkout, "apply", "--check", *map(str, patches))
            git(checkout, "apply", *map(str, patches))
    print(f"Ready: {dependency['name']} ({head}, {len(patches)} patches)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", action="append", help="Prepare only this named dependency; repeatable")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "dependencies/manifest.json").read_text())
    dependencies = manifest["dependencies"]
    if args.only:
        unknown = set(args.only) - {dependency["name"] for dependency in dependencies}
        if unknown:
            parser.error(f"Unknown dependencies: {', '.join(sorted(unknown))}")
        dependencies = [dependency for dependency in dependencies if dependency["name"] in args.only]
    for dependency in dependencies:
        prepare(dependency)


if __name__ == "__main__":
    main()
