mod bench;
#[cfg(feature = "reference-codegen")]
mod compare;
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
    /// Reduce graphs through six propagators with fully specified samples.
    Reduce(reduce::Args),
    /// Measure warmed Rust evaluation; reference performance acceptance is separate.
    Bench(bench::Args),
    /// Compare prepared samples against original Ninja and OneLOop.
    #[cfg(feature = "reference-codegen")]
    Compare(compare::Args),
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
