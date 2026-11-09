//! Reduce box mu-expansion conditioning fixtures with chosen engines and
//! carriers; one JSON line per (fixture, carrier).
//!
//! ```text
//! cargo run --release -p ninjars-symbolica --example box_mu_conditioning -- \
//!     [--carriers fast,fast-conditioned,fast-unchecked,f64,dd,float:107,float:400] FIXTURE.json...
//! ```
//!
//! Carriers: `fast` (binary64 fast engine, default options),
//! `fast-conditioned` (`conditioned_cuts`), `fast-unchecked` (zero
//! `stability_threshold` and `relative_tolerance`), `f64` / `dd` (generic
//! engine in binary64 / double-double) and `float:BITS` (generic engine in
//! MPFR at BITS bits). `float:BITS` lines print the finite part to the
//! carrier's digits, so two of them at >= 100 digits give a reference.

#[path = "../tests/common/conditioning.rs"]
mod conditioning;

use conditioning::{Error, Outcome, fast, generic, parse_fixture, relative_error};
use ninjars::Float;
use ninjars::fast::FastOptions;
use serde_json::json;
use symbolica::domains::float::DoubleFloat;

fn run(fixture: &conditioning::Fixture, carrier: &str) -> Result<Vec<Outcome>, Error> {
    match carrier {
        "fast" => fast(fixture, FastOptions::default()),
        "fast-conditioned" => fast(
            fixture,
            FastOptions {
                conditioned_cuts: true,
                ..FastOptions::default()
            },
        ),
        "fast-unchecked" => fast(
            fixture,
            FastOptions {
                stability_threshold: 0.,
                relative_tolerance: 0.,
                ..FastOptions::default()
            },
        ),
        "f64" => generic(fixture, 0f64, 53),
        "dd" => generic(fixture, DoubleFloat::from(0.), 106),
        other => {
            let bits: u32 = other
                .strip_prefix("float:")
                .ok_or_else(|| format!("unknown carrier {other}"))?
                .parse()?;
            generic(fixture, Float::with_val(bits, 0), bits)
        }
    }
}

fn main() -> Result<(), Error> {
    let mut carriers = vec!["fast".to_string(), "dd".to_string()];
    let mut files = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--carriers" {
            carriers = args
                .next()
                .ok_or("--carriers")?
                .split(',')
                .map(str::to_string)
                .collect();
        } else {
            files.push(arg);
        }
    }
    for file in &files {
        let fixture = parse_fixture(&std::fs::read_to_string(file)?)?;
        for carrier in &carriers {
            let start = std::time::Instant::now();
            let outcomes = run(&fixture, carrier)?;
            let seconds = start.elapsed().as_secs_f64();
            for (lane, outcome) in outcomes.iter().enumerate() {
                let error = match (&outcome.laurent, &fixture.references[lane]) {
                    (Some(l), Some(r)) => Some(relative_error(l[0], r)),
                    _ => None,
                };
                println!(
                    "{}",
                    json!({
                        "fixture": fixture.name, "file": file, "carrier": carrier, "lane": lane,
                        "status": outcome.status, "finite": outcome.finite_text,
                        "rel_error": error,
                        "parts": outcome.parts.map(|p| [[p[0].re, p[0].im], [p[1].re, p[1].im]]),
                        "seconds": seconds,
                    })
                );
            }
        }
    }
    Ok(())
}
