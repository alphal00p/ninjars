#!/usr/bin/env python3
"""Compile and validate emitted batch kernels in the existing Rust workspace."""
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
    parser.add_argument('--jobs', type=int, default=8)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    source = output/'src'
    source.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, SYMBOLICA_HIDE_BANNER='1', RUST_MIN_STACK='134217728')
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    command = ['cargo', 'run', '--locked', '-j', str(args.jobs), '-p',
               'ninjars-symbolica', '--features', 'rust-codegen']
    if args.release:
        command.append('--release')
    validation_sources = [Path(__file__).resolve(), Path(__file__).with_name('probe.rs').resolve()]
    implementation_sources = [
        ROOT/'Cargo.toml', ROOT/'Cargo.lock', ROOT/'dependencies/manifest.json',
        ROOT/'crates/ninjars-symbolica/Cargo.toml',
        ROOT/'crates/ninjars-symbolica/examples/generate_rust_kernels.rs',
        *sorted((ROOT/'crates/ninjars-symbolica/src').glob('*.rs')),
    ]
    def hashes(paths: list[Path]) -> dict[str, str]:
        return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in paths}
    source_hashes = hashes(validation_sources + implementation_sources)
    generation_command = [*command, '--example', 'generate_rust_kernels', '--', str(source)]
    subprocess.run(generation_command,
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
        build_command = [*command, '--example', 'ninjars_rust_kernel_probe_generated',
                         '--message-format=json']
        build_command[1] = 'build'
        build = subprocess.run(build_command, cwd=ROOT, env=environment,
                               stdout=subprocess.PIPE, text=True, check=True)
        messages = [json.loads(line) for line in build.stdout.splitlines()]
        executables = {item['executable'] for item in messages
                       if item.get('reason') == 'compiler-artifact'
                       and item['target']['name'] == 'ninjars_rust_kernel_probe_generated'
                       and item.get('executable')}
        if len(executables) != 1:
            raise RuntimeError(f'expected one compiled probe, found {executables}')
        executable = Path(executables.pop())
        result = subprocess.run([str(executable)],
                                cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                text=True, check=True)
        if source_hashes != hashes(validation_sources + implementation_sources):
            raise RuntimeError('probe sources changed during generation/build/validation')
        evidence = json.loads(result.stdout)
        evidence['generated_sha256'] = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(source.glob('*.rs'))
        }
        evidence['generated_sha256']['metadata.json'] = hashlib.sha256(
            (source/'metadata.json').read_bytes()).hexdigest()
        evidence['validation_source_sha256'] = hashes(validation_sources)
        evidence['implementation_source_sha256'] = hashes(implementation_sources)
        evidence['binary_sha256'] = hashlib.sha256(executable.read_bytes()).hexdigest()
        evidence['wrapper_sha256'] = hashlib.sha256(wrapper.read_bytes()).hexdigest()
        evidence['commands'] = [generation_command, build_command, [str(executable)]]
        evidence['dependencies'] = json.loads((ROOT/'dependencies/manifest.json').read_text())
        evidence['profile'] = 'release' if args.release else 'dev'
        evidence['rustc'] = subprocess.check_output(['rustc', '--version'], text=True).strip()
        evidence['source_commit'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        args.evidence.parent.mkdir(parents=True, exist_ok=True)
        args.evidence.write_text(json.dumps(evidence, indent=2)+'\n')
        print(args.evidence)
    finally:
        wrapper.unlink()


if __name__ == '__main__':
    main()
