//! Generation-only audit of opt-in Symbolica instruction optimization.
use oneloop::rust_codegen::{EmitOptions, RustEmitter};
use oneloop::{A0, B0, C0, D0, OneLoopExpressions, ScalarIntegral, dB0};
use std::{collections::HashSet, path::PathBuf, time::Instant};
use symbolica::{
    atom::{Atom, AtomCore, Symbol},
    domains::{float::Complex, rational::Rational},
    evaluate::{ExportedInstructions, Instruction},
};

#[derive(Default, Debug)]
struct Counts {
    bodies: usize,
    instructions: usize,
    arithmetic: usize,
    calls: usize,
    branches: usize,
    max_arity: usize,
    temporary_slots: usize,
}

fn inventory(
    graph: &ExportedInstructions<Complex<Rational>>,
    seen: &mut HashSet<(Symbol, Vec<String>)>,
    counts: &mut Counts,
) {
    counts.bodies += 1;
    counts.instructions += graph.instructions.len();
    counts.temporary_slots += graph.temporary_count;
    for instruction in &graph.instructions {
        match instruction {
            Instruction::Add(_, args, _) | Instruction::Mul(_, args, _) => {
                counts.arithmetic += args.len().saturating_sub(1);
                counts.max_arity = counts.max_arity.max(args.len());
            }
            Instruction::IfElse(..) => counts.branches += 1,
            Instruction::Fun(..) => counts.calls += 1,
            _ => {}
        }
    }
    for child in &graph.sub_evaluators {
        if seen.insert((child.symbol, child.tags.clone())) {
            inventory(&child.instructions, seen, counts);
        }
    }
}

fn main() {
    let mut arguments = std::env::args().skip(1);
    let mode = arguments.next().expect("mode: baseline, cse, cpe");
    let directory = PathBuf::from(arguments.next().expect("output directory"));
    assert!(arguments.next().is_none());
    assert!(["baseline", "cse", "cpe"].contains(&mode.as_str()));
    std::thread::Builder::new()
        .stack_size(128 * 1024 * 1024)
        .spawn(move || {
            std::fs::create_dir_all(&directory).unwrap();
            let context = OneLoopExpressions::new();
            let mut emitter = RustEmitter::default();
            for (family, symbol, name) in [
                (ScalarIntegral::A0, A0(), "a0"),
                (ScalarIntegral::B0, B0(), "b0"),
                (ScalarIntegral::DB0, dB0(), "db0"),
                (ScalarIntegral::C0, C0(), "c0"),
                (ScalarIntegral::D0, D0(), "d0"),
            ] {
                let started = Instant::now();
                let parameters = (0..family.arity())
                    .map(|index| symbolica::symbol!(format!("instruction_probe_{name}_{index}")).to_atom())
                    .collect::<Vec<_>>();
                let calls = [0, -1, -2].map(|tag| {
                    let mut args = vec![Atom::num(tag)];
                    args.extend(parameters.iter().cloned());
                    symbol.call(&args)
                });
                let evaluator = Atom::evaluator_multiple(&calls, &parameters)
                    .function_map(context.function_map().clone())
                    .direct_translation(true)
                    .horner_iterations(0)
                    .build()
                    .unwrap();
                let build_seconds = started.elapsed().as_secs_f64();
                let exported = evaluator.export_instructions();
                let mut counts = Counts::default();
                inventory(&exported, &mut HashSet::new(), &mut counts);
                drop(exported);
                let module = emitter.emit_batch(evaluator, EmitOptions {
                    inline_arithmetic: matches!(family, ScalarIntegral::A0 | ScalarIntegral::B0 | ScalarIntegral::DB0),
                }).unwrap();
                let scalar_bytes = module.source.find("use super::BatchWorkspace;").unwrap();
                eprintln!("mode={mode} family={name} counts={counts:?} scalar_bytes={scalar_bytes} batch_bytes={} build_seconds={build_seconds:.6} total_seconds={:.6}", module.source.len()-scalar_bytes, started.elapsed().as_secs_f64());
                std::fs::write(directory.join(format!("generated_{name}.rs")), module.source).unwrap();
            }
            std::fs::write(directory.join("generated_constants.rs"), emitter.constants_source()).unwrap();
        })
        .unwrap()
        .join()
        .unwrap();
}
