mod bench;
mod case_bench;
#[cfg(feature = "reference-codegen")]
mod compare;
mod draw;
mod export_bench;
mod fast_runtime;
mod numerator_bench;
mod numerator_coefficients;
mod pipeline_bench;
mod prepare;
mod reduce;
#[cfg(feature = "reference-codegen")]
mod reference;
#[cfg(feature = "reference-codegen")]
mod reference_bench;
mod runtime;

use clap::{Parser, Subcommand};

/// Generic, batched one-loop reduction using Symbolica and FeynKit.
#[derive(Parser)]
#[command(name = "ninjars", version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Prepare a FeynKit DOT graph or a JSON fixture using exact symbolic inputs.
    Prepare(prepare::Args),
    /// Reduce graphs with any number of propagators and fully specified samples.
    Reduce(reduce::Args),
    /// Measure warmed Rust evaluation; reference performance acceptance is separate.
    Bench(bench::Args),
    /// Compare prepared samples against original Ninja and OneLOop.
    #[cfg(feature = "reference-codegen")]
    Compare(compare::Args),
    /// Export fixture samples as a shared tensor-numerator bench case (ninjars-bench-case-v1).
    ExportBench(export_bench::Args),
    /// Physical tensor-numerator generation timings and per-point preparation benchmark.
    NumeratorBench(numerator_bench::Args),
    /// End-to-end binary64 pipeline benchmark (physical numerator, fast
    /// reduction, native masters) with a per-component breakdown.
    PipelineBench(pipeline_bench::Args),
    /// Binary64 fast-engine benchmark of shared bench cases (ninjars-bench-case-v1).
    CaseBench(case_bench::Args),
    /// Draw FeynKit DOT graphs as Typst documents with Linnest.
    Draw(draw::Args),
}

fn main() {
    // Symbolica documents this presentation-only setting. Keep machine-readable
    // stdout free of its startup banner; licensing checks are unaffected.
    // SAFETY: the standalone binary has not initialized Symbolica or started any
    // worker threads, and no other environment access runs concurrently here.
    unsafe { std::env::set_var("SYMBOLICA_HIDE_BANNER", "1") };
    let result = match Cli::parse().command {
        Command::Prepare(args) => prepare::run(args).map_err(|error| error.to_string()),
        Command::Reduce(args) => std::thread::Builder::new()
            .name("ninjars-reduce".into())
            // oneloopmaster's exact expression preparation requires this stack.
            .stack_size(128 * 1024 * 1024)
            .spawn(move || reduce::run(args).map_err(|error| error.to_string()))
            .expect("cannot start reduction worker")
            .join()
            .unwrap_or_else(|_| Err("reduction worker panicked".into())),
        Command::Bench(args) => std::thread::Builder::new()
            .name("ninjars-bench".into())
            .stack_size(128 * 1024 * 1024)
            .spawn(move || bench::run(args).map_err(|error| error.to_string()))
            .expect("cannot start benchmark worker")
            .join()
            .unwrap_or_else(|_| Err("benchmark worker panicked".into())),
        Command::ExportBench(args) => std::thread::Builder::new()
            .name("ninjars-export-bench".into())
            .stack_size(128 * 1024 * 1024)
            .spawn(move || export_bench::run(args).map_err(|error| error.to_string()))
            .expect("cannot start export worker")
            .join()
            .unwrap_or_else(|_| Err("export worker panicked".into())),
        Command::NumeratorBench(args) => std::thread::Builder::new()
            .name("ninjars-numerator-bench".into())
            .stack_size(128 * 1024 * 1024)
            .spawn(move || numerator_bench::run(args).map_err(|error| error.to_string()))
            .expect("cannot start numerator benchmark worker")
            .join()
            .unwrap_or_else(|_| Err("numerator benchmark worker panicked".into())),
        Command::Draw(args) => draw::run(args).map_err(|error| error.to_string()),
        Command::CaseBench(args) => std::thread::Builder::new()
            .name("ninjars-case-bench".into())
            .stack_size(128 * 1024 * 1024)
            .spawn(move || case_bench::run(args).map_err(|error| error.to_string()))
            .expect("cannot start case benchmark worker")
            .join()
            .unwrap_or_else(|_| Err("case benchmark worker panicked".into())),
        Command::PipelineBench(args) => std::thread::Builder::new()
            .name("ninjars-pipeline-bench".into())
            .stack_size(128 * 1024 * 1024)
            .spawn(move || pipeline_bench::run(args).map_err(|error| error.to_string()))
            .expect("cannot start pipeline benchmark worker")
            .join()
            .unwrap_or_else(|_| Err("pipeline benchmark worker panicked".into())),
        #[cfg(feature = "reference-codegen")]
        Command::Compare(args) => std::thread::Builder::new()
            .name("ninjars-compare".into())
            .stack_size(128 * 1024 * 1024)
            .spawn(move || compare::run(args).map_err(|error| error.to_string()))
            .expect("cannot start comparison worker")
            .join()
            .unwrap_or_else(|_| Err("comparison worker panicked".into())),
    };
    if let Err(error) = result {
        eprintln!("ninjars: {error}");
        std::process::exit(1);
    }
}
