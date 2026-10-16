mod prepare;
#[cfg(feature = "reference-codegen")]
mod reference;

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
}

fn main() {
    // Symbolica documents this presentation-only setting. Keep machine-readable
    // stdout free of its startup banner; licensing checks are unaffected.
    // SAFETY: the standalone binary has not initialized Symbolica or started any
    // worker threads, and no other environment access runs concurrently here.
    unsafe { std::env::set_var("SYMBOLICA_HIDE_BANNER", "1") };
    let result = match Cli::parse().command {
        Command::Prepare(args) => prepare::run(args),
    };
    if let Err(error) = result {
        eprintln!("ninjars: {error}");
        std::process::exit(1);
    }
}
