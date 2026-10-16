#!/usr/bin/env python3
"""Generate and compile scalar kernel controls in the existing Rust workspace."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'target/rust-kernel-probe')
    parser.add_argument('--evidence', type=Path, default=ROOT/'target/rust-kernel-probe/evidence.json')
    parser.add_argument('--release', action='store_true')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    source = output/'src'
    source.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, SYMBOLICA_HIDE_BANNER='1', RUST_MIN_STACK='134217728')
    command = ['cargo', 'run', '--locked', '-p', 'ninjars-symbolica', '--features', 'rust-codegen']
    if args.release:
        command.append('--release')
    subprocess.run([*command, '--example', 'generate_rust_kernels', '--', str(source)],
                   cwd=ROOT, env=environment, check=True)
    shutil.copyfile(Path(__file__).with_name('probe.rs'), source/'probe.rs')
    wrapper = ROOT/'crates/ninjars-symbolica/examples/ninjars_rust_kernel_probe_generated.rs'
    # An exclusive create protects a user's existing file. A temporary target
    # reuses the root lockfile, profiles, patches, and compilation cache.
    with wrapper.open('x') as stream:
        stream.write('#[cfg(feature="rust-codegen")]\n'
                     f'#[path={json.dumps(str(source/"probe.rs"))}] mod probe;\n'
                     '#[cfg(feature="rust-codegen")] fn main() {'
                     'std::thread::Builder::new().stack_size(128*1024*1024).spawn(probe::run)'
                     '.unwrap().join().unwrap();}\n'
                     '#[cfg(not(feature="rust-codegen"))] fn main() {}\n')
    try:
        result = subprocess.run([*command, '--example', 'ninjars_rust_kernel_probe_generated'],
                                cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                text=True, check=True)
        evidence = json.loads(result.stdout)
        evidence['generated_sha256'] = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(source.glob('*.rs'))
        }
        evidence['generated_sha256']['metadata.json'] = hashlib.sha256(
            (source/'metadata.json').read_bytes()).hexdigest()
        evidence['validation_source_sha256'] = {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(__file__).resolve(), Path(__file__).with_name('probe.rs').resolve()]
        }
        evidence['profile'] = 'release' if args.release else 'dev'
        evidence['rustc'] = subprocess.check_output(['rustc', '--version'], text=True).strip()
        args.evidence.parent.mkdir(parents=True, exist_ok=True)
        args.evidence.write_text(json.dumps(evidence, indent=2)+'\n')
        print(args.evidence)
    finally:
        wrapper.unlink()


if __name__ == '__main__':
    main()
