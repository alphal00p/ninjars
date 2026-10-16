#!/usr/bin/env python3
"""Compile only diagnostic copied adapters against explicitly hashed cached rlibs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

HERE = Path(__file__).resolve().parent

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repository', required=True, type=Path)
    parser.add_argument('--cargo-artifacts', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    repo, out = args.repository.resolve(), args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    records = []
    for line in args.cargo_artifacts.read_text().splitlines():
        try: records.append(json.loads(line))
        except ValueError: pass
    rlibs = {}; library_paths = []
    for row in records:
        if row.get('reason') == 'compiler-artifact' and row['profile']['opt_level'] == '3':
            for filename in row['filenames']:
                if filename.endswith('.rlib'): rlibs[row['target']['name']] = Path(filename)
        if row.get('reason') == 'build-script-executed': library_paths.extend(row['linked_paths'])
    required = ['ninjars','symbolica','oneloop','symbolica_utils','ninjars_graph','feynkit_graph','feynkit_kinematics']
    artifact_before = {str(rlibs[name]): digest(rlibs[name]) for name in required}
    source_dir = repo/'crates/ninjars-symbolica/src'
    bridge_dir = repo/'validation/performance/shared_provider/src'
    sources = sorted(source_dir.glob('*.rs')) + sorted(bridge_dir.glob('*.rs')) + [HERE/'diagnostic.rs', HERE/'instrument.py']
    before = {str(path):digest(path) for path in sources}
    adapter, bridge = out/'adapter', out/'bridge'
    shutil.copytree(source_dir, adapter); shutil.copytree(bridge_dir, bridge)
    shutil.copyfile(HERE/'diagnostic.rs', adapter/'diagnostic_profile.rs')
    path = adapter/'lib.rs'; path.write_text(path.read_text()+'\n#[doc(hidden)]\npub mod diagnostic_profile;\n')
    path = adapter/'kernel.rs'; text = path.read_text()
    text = text.replace('    fn evaluate(', '    #[track_caller]\n    fn evaluate(')
    text = text.replace('    pub fn evaluate_batch(', '    #[track_caller]\n    pub fn evaluate_batch(')
    needle = '    ) -> Result<()> {\n        BatchShape {'
    assert text.count(needle) == 1
    text = text.replace(needle, '    ) -> Result<()> {\n        let _diagnostic = crate::diagnostic_profile::Span::kernel(rows, self.inputs, self.outputs);\n        BatchShape {')
    path.write_text(text)
    path = adapter/'cuts.rs'; text = path.read_text()
    needle = '    fn evaluate(&mut self, rows: usize) -> Result<()> {'
    assert text.count(needle) == 1
    text = text.replace(needle,'    #[track_caller]\n'+needle+'\n        let _diagnostic = crate::diagnostic_profile::Span::kernel(rows, if rows == 0 { 0 } else { self.inputs.len()/rows }, self.columns*self.width);')
    path.write_text(text)
    phase_count = 0
    for filename in ['amplitude','cuts','high_cuts','triangle_kernels','bubble_kernels','tadpole_kernels','box_kernels','pentagon_kernels','integration']:
        path = adapter/f'{filename}.rs'; text = path.read_text()
        def add_phase(match):
            nonlocal phase_count
            name, signature = match.group(1), match.group(0)
            rows = next((name+'.len()' for name in ['requests','e12','cuts','scales'] if re.search(r'\b'+name+r'\s*:',signature)), None)
            if re.search(r'\brows\s*:\s*usize',signature): rows = 'rows'
            if rows is None and filename == 'integration' and name == 'contract_batch': rows = 'output.len()'
            if rows is None: raise RuntimeError(f'unknown row metadata: {filename} {name}')
            phase_count += 1
            return signature+f'\n let _diagnostic_phase = crate::diagnostic_profile::Span::phase("{filename}::{name}", {rows});\n'
        text = re.sub(r'fn (\w+_batch)\([^{}]*?\)\s*->\s*Result<\(\)>\s*\{',add_phase,text,flags=re.S)
        path.write_text(text)
    path = bridge/'lib.rs'; text = path.read_text()
    text += '''\n#[unsafe(no_mangle)]\npub extern "C" fn sp_kernel_profile_start() -> i32 { guard(|| { ninjars_symbolica::diagnostic_profile::start(); Ok(()) }) }\n#[unsafe(no_mangle)]\npub extern "C" fn sp_kernel_profile_stop() -> i32 { guard(|| {\n let report = ninjars_symbolica::diagnostic_profile::stop();\n let path = std::env::var("NINJARS_PROFILE_OUTPUT").map_err(|e| e.to_string())?;\n std::fs::write(path, report).map_err(|e| e.to_string())?; Ok(())\n}) }\n'''
    path.write_text(text)
    commands = []
    environment = dict(os.environ)
    def compile(label, command):
        command = list(map(str, command)); commands.append(command)
        with (out/f'{label}.stdout').open('w') as stdout,(out/f'{label}.stderr').open('w') as stderr:
            build_env = dict(environment, CARGO_CRATE_NAME=command[command.index('--crate-name')+1])
            result = subprocess.run(command, cwd=repo, env=build_env, stdout=stdout, stderr=stderr)
        if result.returncode: raise RuntimeError(f'{label} failed; see {out}/{label}.stderr')
    dependencies = rlibs['ninjars'].parent
    common = ['--edition=2024','-C','opt-level=3','-C','debuginfo=1','-C','force-frame-pointers=yes','-C','force-unwind-tables=yes','-L',f'dependency={dependencies}']
    for path in sorted(set(library_paths)): common += ['-L',path]
    extern = lambda names: [item for name in names for item in ['--extern',f'{name}={rlibs[name]}']]
    built_adapter = out/'libninjars_symbolica_profile.rlib'
    compile('adapter',['rustc',adapter/'lib.rs','--crate-name','ninjars_symbolica','--crate-type','rlib',*common,'-C','metadata=diagnostic_kernel_profile',*extern(['ninjars','symbolica','oneloop','symbolica_utils']),'-o',built_adapter])
    library = out/'libninjars_shared_provider_probe.so'
    compile('bridge',['rustc',bridge/'lib.rs','--crate-name','ninjars_shared_provider_probe','--crate-type','cdylib',*common,'-L',f'dependency={out}','--extern',f'ninjars_symbolica={built_adapter}',*extern(['ninjars','symbolica','ninjars_graph','feynkit_graph','feynkit_kinematics']),'-o',library])
    after = {str(path):digest(path) for path in sources}
    artifact_after = {path:digest(path) for path in artifact_before}
    if before != after or artifact_before != artifact_after: raise RuntimeError('source/artifact changed during build')
    report = {'success':True,'performance_acceptance':False,'source_sha256':before,'cached_rlib_sha256':artifact_before,
       'cargo_artifact_manifest_sha256':digest(args.cargo_artifacts),'instrumented_source_sha256':{str(p):digest(p) for p in sorted(adapter.glob('*.rs'))+sorted(bridge.glob('*.rs'))},
       'library_sha256':digest(library),'commands':commands,'semantic_phase_entries':phase_count,
       'rustc':subprocess.check_output(['rustc','-Vv'],text=True),'production_files_modified':False,
       'note':'Diagnostic adapter/bridge rebuilt from recorded sources against hashed cached dependencies; no native kernel selection, no parity timing claim.'}
    (out/'build.json').write_text(json.dumps(report,indent=2)+'\n'); print(out/'build.json')

if __name__ == '__main__': main()
