//! Per-lane look at the cancellation ladder (`cancellation_ladder`) on
//! dumped pyAmpliCol lanes (one fixture per JSON line, the
//! `ninjars-box-mu-conditioning-v1` shape with a single `tensor`).
//!
//! ```text
//! cargo run --release -p ninjars-symbolica --features native-masters \
//!     --example ladder_diag -- [--base dd|f64|BITS] [--ref BITS] LANES.jsonl
//! ```
//!
//! Each lane is first made consistent at `--ref` bits (shells projected,
//! other invariants from the projected shifts), its reference computed there,
//! and its inputs rounded to the base carrier (as a caller's cure in that
//! carrier gives them). Then the base carrier's generic reduction (with the
//! stability test) and the ladder's rescue. One JSON line per lane: base
//! status, measured loss `L`, base and rescued relative errors, rungs, and
//! the cut sizes carrying the largest term moduli.

#[path = "../tests/common/conditioning.rs"]
mod conditioning;
#[path = "../tests/common/ladder.rs"]
mod ladder;

use conditioning::{Error, parse_fixture};
use ladder::{consistent, run};
use ninjars::{Complex, DoubleFloat, Float};
use serde_json::json;

fn rel(a: Complex<f64>, b: Complex<f64>) -> f64 {
    let d = Complex::new(a.re - b.re, a.im - b.im);
    (d.re.hypot(d.im)) / b.re.hypot(b.im)
}

fn main() -> Result<(), Error> {
    let mut base = "dd".to_string();
    let mut reference = 512u32;
    let mut target: Option<f64> = None;
    let mut files = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--base" => base = args.next().ok_or("--base")?,
            "--ref" => reference = args.next().ok_or("--ref")?.parse()?,
            "--target" => target = Some(args.next().ok_or("--target")?.parse()?),
            _ => files.push(arg),
        }
    }
    for file in files {
        for line in std::fs::read_to_string(&file)?.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let fixture = parse_fixture(line)?;
            let lane = consistent(&fixture, reference);
            let started = std::time::Instant::now();
            let reference_run = run(&lane, Float::with_val(reference, 0), false, None)?;
            let (truth, lref) = (reference_run.value, reference_run.lost_bits);
            let tref = started.elapsed().as_secs_f64();
            let started = std::time::Instant::now();
            let base_run = match base.as_str() {
                "dd" => run(&lane, DoubleFloat::from(0.0), true, target)?,
                "f64" => run(&lane, 0f64, true, target)?,
                bits => run(&lane, Float::with_val(bits.parse()?, 0), true, target)?,
            };
            let t = started.elapsed().as_secs_f64();
            let out = json!({
                "name": fixture.name, "file": file, "n": lane.n, "rank": lane.rank,
                "truth": [truth.re, truth.im], "ref_lost_bits": lref, "ref_seconds": tref,
                "status": base_run.status, "lost_bits": base_run.lost_bits, "base_rel_err": rel(base_run.value, truth),
                "base_and_ladder_seconds": t,
                "max_term_moduli_by_cut_size": base_run.by_size,
                "ladder": base_run.ladder.map(|r| json!({"rel_err": rel(r.value, truth), "rungs": r.rungs, "rung_estimate": r.estimate, "projected": r.projected, "outcome": format!("{:?}", r.outcome)})),
            });
            println!("{out}");
        }
    }
    Ok(())
}
