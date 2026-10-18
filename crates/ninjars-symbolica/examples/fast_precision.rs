//! Arbitrary-precision reference for the binary64 conditioning of the fast
//! engine and of original Ninja (see `tests/common/precision.rs`).
//!
//! For each selected lane, the generic NinjaRS engine reduces the case's tensor
//! numerator in `Float(--bits)` arithmetic with exactly the OneLOop masters
//! original Ninja used; the binary64 errors of the fast engine (same masters)
//! and of original Ninja are reported relative to the lane's largest exact
//! Laurent/split modulus. The generic engine accepts any number of propagators.
//!
//! ```text
//! cargo run --profile perf -p ninjars-symbolica --example fast_precision -- \
//!     [--bits 256] [--lanes 0,2] CASE.json NINJA.json [MASTERS.json]
//! ```

#[path = "../tests/common/ninja_reference.rs"]
mod ninja_reference;
#[path = "../tests/common/precision.rs"]
mod precision;

use ninja_reference::{Error, crosscheck, parse_case, parse_ninja};
use serde_json::json;

fn main() -> Result<(), Error> {
    let mut bits = 256u32;
    let mut only: Option<Vec<usize>> = None;
    let mut files = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--bits" => bits = args.next().ok_or("--bits")?.parse()?,
            "--lanes" => {
                only = Some(
                    args.next()
                        .ok_or("--lanes")?
                        .split(',')
                        .map(str::parse)
                        .collect::<Result<_, _>>()?,
                )
            }
            _ => files.push(arg),
        }
    }
    if files.len() < 2 {
        return Err("expected CASE.json NINJA.json [MASTERS.json]".into());
    }
    let case = parse_case(&std::fs::read_to_string(&files[0])?)?;
    let masters = files.get(2).map(std::fs::read_to_string).transpose()?;
    let ninja = parse_ninja(&std::fs::read_to_string(&files[1])?, masters.as_deref())?;
    let report = crosscheck(&case, &ninja)?;
    for lane in precision::precision(&case, &ninja, bits, only.as_deref())? {
        let index = case
            .points
            .iter()
            .position(|p| p.id == lane.id)
            .unwrap_or(0);
        println!(
            "{}",
            json!({"case": case.name, "lane": lane.id, "bits": bits,
                "exact": lane.exact.iter().map(|z| [z.re, z.im]).collect::<Vec<_>>(),
                "scale": lane.scale, "fast_status": format!("{:?}", lane.fast_status),
                "fast_laurent_error": lane.fast_laurent, "ninja_laurent_error": lane.ninja_laurent,
                "fast_split_error": lane.fast_split, "ninja_split_error": lane.ninja_split,
                "fast_vs_ninja": report.lanes[index].laurent})
        );
    }
    Ok(())
}
