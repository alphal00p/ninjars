#!/usr/bin/env python3
"""Run same-provider ggHHH physical correctness controls; never report timings."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess

from kernel_selection import add_arguments, native_enabled, validate_arguments, validate_report

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def runtime_sources():
    paths = [ROOT/'Cargo.lock', ROOT/'dependencies/manifest.json',
             ROOT/'examples/graphs/physical/gg-hhh/top-pentagon-26.dot',
             ROOT/'validation/evidence/gg-hhh-changing-pool.json',
             ROOT/'validation/evidence/gg-hhh-changing-madloop.json',
             ROOT/'references/gammaloop_ninja_shared/crates/feynkit-model/data/sm.json.zlib']
    for directory in [HERE, ROOT/'crates/ninjars', ROOT/'crates/ninjars-symbolica', ROOT/'crates/ninjars-graph']:
        paths += [p for p in directory.rglob('*') if p.is_file() and 'target' not in p.parts and p.suffix in {'.rs', '.toml', '.lock', '.h', '.hh', '.cc', '.py'}]
    # Include all tracked dependency source files, capturing local patches too.
    for name in ['symbolica', 'oneloopmaster', 'gammaloop_ninja_shared']:
        directory = ROOT/'references'/name
        tracked = subprocess.check_output(['git', '-C', directory, 'ls-files', '-z']).split(b'\0')
        paths += [directory/p.decode() for p in tracked if p and (directory/p.decode()).is_file() and Path(p.decode()).suffix in {'.rs', '.toml', '.lock'}]
    return sorted(set(paths))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--reference-build-dir', type=Path, default=ROOT/'target/reference-uncached-v2')
    parser.add_argument('--maximum-integrated-batch', type=int, default=32)
    parser.add_argument('--release', action='store_true', help='use the optimized Rust configuration used by timing probes')
    add_arguments(parser, all_catalog=True)
    args = parser.parse_args()
    try:
        validate_arguments(args)
    except ValueError as error:
        parser.error(str(error))
    if args.all_reduction_kernels and args.reduction_kernels != "symbolica":
        parser.error("all-catalog checks use interpreted as the primary selection")
    if args.maximum_integrated_batch < 3:
        parser.error('maximum integrated batch must include the three distinct points')
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        parser.error('output directory must be empty')
    reference = args.reference_build_dir.resolve()
    spec = importlib.util.spec_from_file_location('reference_build', ROOT/'validation/reference/build.py')
    reference_build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference_build)
    record = json.loads((reference/'revisions.json').read_text())
    ninja, olo = reference/'ninja-build/src/.libs/libninja.a', reference/'oneloop/libavh_olo.a'
    config = reference_build.configuration(reference)
    if (record.get('configuration') != config or not config['oneloop'] or not config['higher_rank'] or config['master_result_cache']
            or record.get('libraries_sha256') != {p.name: digest(p) for p in [ninja, olo]}
            or record.get('input_files_sha256') != reference_build.library_inputs(reference)):
        raise RuntimeError('uncached reference build provenance mismatch')
    pool = json.loads((ROOT/'validation/evidence/gg-hhh-changing-pool.json').read_text())
    madloop = json.loads((ROOT/'validation/evidence/gg-hhh-changing-madloop.json').read_text())
    if pool['points'] != madloop['points'] or len(pool['points']) != 3:
        raise RuntimeError('MadLoop and shared numerical preparation inputs differ')
    if madloop['raw_factor'] != '-1/2' or madloop['linear_laurent_factor'] != '-8*pi^2':
        raise RuntimeError('unreviewed normalization')
    cases = {(c['point_id'], c['ctmode']): c for c in madloop['cases']}
    if len(cases) != 6:
        raise RuntimeError('expected independent mode1/mode4 references')
    protocol = ['3']
    for point in pool['points']:
        case = cases[point['id'], 4]
        if not case['stable'] or len(case['numerator_samples']) != 70:
            raise RuntimeError('unstable/incomplete mode4 reference')
        for row in point['momenta']:
            protocol.append(' '.join(map(repr, row)))
        protocol += [' '.join(map(str, point['helicities'])), ' '.join(map(repr, point['model'])), repr(point['scale_squared']), '70']
        # Verified graph26 routing: FeynKit K -> -MadLoop K. Raw color/fermion
        # factor is applied once to the unsquared MadLoop numerator.
        for sample in case['numerator_samples']:
            values = [-x for pair in sample['q'] for x in pair]
            values += [-.5*x for x in sample['raw']]
            protocol.append(' '.join(map(repr, values)))
        protocol.append(' '.join(repr(-8*math.pi**2*x) for pair in case['raw_laurent'] for x in pair))
    (out/'input.txt').write_text('\n'.join(protocol)+'\n')
    paths = runtime_sources()
    before = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    commands = []

    def run(label, command, stdin=None):
        command = list(map(str, command))
        with (out/f'{label}.stdout').open('w') as stdout, (out/f'{label}.stderr').open('w') as stderr:
            result = subprocess.run(command, cwd=ROOT, text=True, input=stdin, stdout=stdout, stderr=stderr)
        commands.append({'label': label, 'argv': command, 'returncode': result.returncode})
        (out/'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
        if result.returncode:
            raise RuntimeError(f'{label} failed: {out}/{label}.stderr')
        return (out/f'{label}.stdout').read_text()

    build_command = ['cargo', 'build', '--manifest-path', HERE/'Cargo.toml', '--locked', '-j4', '--message-format=json']
    if native_enabled(args):
        build_command += ["--features", "native-kernels"]
    if args.release:
        build_command.append('--release')
    build = run('cargo-build', build_command)
    artifacts = [Path(p) for line in build.splitlines() if line.startswith('{')
                 for row in [json.loads(line)] if row.get('reason') == 'compiler-artifact' and row['target']['name'] == 'ninjars_shared_provider_probe'
                 for p in row['filenames'] if p.endswith('.so')]
    if len(artifacts) != 1:
        raise RuntimeError('expected one Rust provider artifact')
    library = out/'libninjars_shared_provider_probe.so'
    shutil.copy2(artifacts[0], library)
    fortran = Path(subprocess.check_output(['gfortran', '-print-file-name=libgfortran.so'], text=True).strip())
    binary = out/'physical-probe'
    run('cpp-build', ['g++', '-O2', '-std=c++17', '-Wall', '-Wextra', '-Werror',
                     '-I'+str(reference/'ninja-source/src'), '-I'+str(reference/'ninja-build/src'), '-I'+str(reference/'ninja-build'),
                     HERE/'physical_probe.cc', library, ninja, olo, fortran, '-lquadmath',
                     '-Wl,-rpath,'+str(out), '-Wl,-rpath,'+str(fortran.parent), '-o', binary])
    probe_command = [binary, ROOT/'examples/graphs/physical/gg-hhh/top-pentagon-26.dot', args.maximum_integrated_batch, "all" if args.all_reduction_kernels else args.reduction_kernels]
    if args.native_block_rows is not None:
        probe_command.append(args.native_block_rows)
    stdout = run('physical-probe', probe_command, '\n'.join(protocol)+'\n')
    results = [json.loads(line) for line in stdout.splitlines() if line.startswith("{")]
    if len(results) != 1:
        raise RuntimeError("expected exactly one physical probe report")
    result = results[0]
    validate_report(result, args.reduction_kernels, native_enabled(args), all_catalog=args.all_reduction_kernels,
                    native_block_rows=args.native_block_rows, execution_controls=args.native_block_rows is not None)
    if not result['success']:
        raise RuntimeError('physical provider controls did not pass')
    # Mode1 is separately visible, with no automatic switching or substitution.
    mode1 = []
    for i, point in enumerate(pool['points']):
        expected = [complex(*z)*(-8*math.pi**2) for z in cases[point['id'], 1]['raw_laurent']]
        actual = [complex(*z) for z in result['rust'][i*5:i*5+3]]
        errors = [abs(a-e)/max(1., abs(e)) for a, e in zip(actual, expected)]
        mode1.append({'point_id': point['id'], 'max_scaled_error': max(errors), 'passes_1e_minus_9': max(errors) <= 1e-9})
    after = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    result['unchanged_runtime_sources'] = before == after
    result['success'] &= before == after
    result['madloop_mode1_diagnostic'] = mode1
    result['rust_profile'] = 'release' if args.release else 'dev'
    result['numerator_provider'] = {'coefficient_heads': 'factored SymJIT O2 during physical preparation',
                                    'reduction_callbacks': 'shared interpreted PolynomialNumerator<f64> direct/Mu/T3/T2',
                                    'independent_of_reduction_selection': True}
    result['scope'] = 'Same factored SymJIT coefficient provider, FeynKit wavefunctions/model binding/routing and shared live masters; separate frozen-union reduction controls. No production-backend or timing acceptance.'
    result['physical_input_boundary'] = 'Explicit five positive-energy binary64 four-momenta, five helicities, seven free model values and scale squared; wavefunctions, dependent model coefficients, coefficient heads, propagators and S matrix evaluated within every complete-path call.'
    result['precision_oracle_boundary'] = 'Float256 exactly promotes binary64 physical component inputs after wavefunction evaluation; this checks coefficient arithmetic, not high-precision physical wavefunctions.'
    result['source_sha256'] = before
    result['artifacts_sha256'] = {str(p): digest(p) for p in [library, binary, ninja, olo, fortran, out/'input.txt']}
    result['dependency_manifest'] = json.loads((ROOT/'dependencies/manifest.json').read_text())
    result['dependency_heads'] = {name: subprocess.check_output(['git', '-C', ROOT/'references'/name, 'rev-parse', 'HEAD'], text=True).strip() for name in ['symbolica', 'oneloopmaster', 'gammaloop_ninja_shared']}
    result['commands'] = commands
    result['logs'] = {p.name: {'sha256': digest(p), 'text': p.read_text()} for p in out.glob('*.stderr')}
    (out/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(out/'result.json')
    if not result['success']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
