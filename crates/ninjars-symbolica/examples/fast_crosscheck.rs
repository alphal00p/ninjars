//! Cross-check the fast engine against original Ninja's recorded results with
//! identical (OneLOop) master integrals; see `tests/common/ninja_reference.rs`.
//!
//! ```text
//! cargo run --profile perf -p ninjars-symbolica --example fast_crosscheck -- \
//!     CASE.json NINJA.json [MASTERS.json] [-- CASE.json NINJA.json ...]
//! cargo run --profile perf -p ninjars-symbolica --example fast_crosscheck -- \
//!     --benchmark-dir /path/to/validation/benchmark
//! ```
//!
//! `NINJA.json` is `ninja_bench.cc` output (inline `master_call_list`) or a
//! `validation/benchmark/ninja/<case>.json` result file, whose calls are then
//! read from `MASTERS.json` (`<case>.masters.json`). `--benchmark-dir` checks
//! every `cases/*.json` and `synthetic/*.json` of a benchmark directory
//! (`validation/benchmark` of the main checkout, or `validation/fast/ninja_reference`)
//! against its `ninja/` results. One JSON line per case.
//!
//! With `--features native-masters` each line also reports oneloopmaster's
//! native masters (`OneLoopNativeMasters`) against OneLOop on every recorded
//! call (`native_vs_oneloop`: largest relative difference per family and
//! output, relative to the larger modulus of each Laurent triple) and the
//! complete fast path, fast engine + native masters, against original Ninja +
//! OneLOop (`native_full_max_scaled`).

#[path = "../tests/common/ninja_reference.rs"]
mod ninja_reference;

use ninja_reference::{Error, crosscheck, parse_case, parse_ninja};
use serde_json::json;
use std::path::{Path, PathBuf};

fn run(case: &Path, ninja: &Path, masters: Option<&Path>) -> Result<(), Error> {
    let parsed = parse_case(&std::fs::read_to_string(case)?)?;
    let masters = masters.map(std::fs::read_to_string).transpose()?;
    let lanes = parse_ninja(&std::fs::read_to_string(ninja)?, masters.as_deref())?;
    let report = crosscheck(&parsed, &lanes)?;
    #[cfg(feature = "native-masters")]
    let native = native::compare(&parsed, &lanes)?;
    #[cfg(not(feature = "native-masters"))]
    let native = serde_json::Value::Null;
    let lanes: Vec<_> = report
        .lanes
        .iter()
        .map(|l| {
            json!({"id": l.id, "status": format!("{:?}", l.status), "ninja_status": l.ninja_status,
                "scale": l.scale, "laurent": l.laurent, "cut_constructible": l.cut_constructible,
                "rational": l.rational})
        })
        .collect();
    println!(
        "{}",
        json!({"case": report.name, "n": report.n, "rank": report.rank,
            "max_scaled_laurent": report.max_laurent(), "max_scaled_split": report.max_split(),
            "unmatched_masters": report.unmatched_masters, "statuses_match": report.statuses_match(),
            "batches_bitwise": report.batches_bitwise,
            "batch_failures": report.batch_failures.len(), "native": native, "lanes": lanes})
    );
    Ok(())
}

#[cfg(feature = "native-masters")]
mod native {
    use super::ninja_reference::{Batch, Case, Error, NinjaLane, values};
    use ninjars::fast::{
        C64, FastAmplitude, FastMaster, FastMasters, FastOptions, FastResult, FastStatus,
        TensorNumerator,
    };
    use ninjars_symbolica::OneLoopNativeMasters;
    use serde_json::{Value, json};
    use std::collections::BTreeMap;

    pub fn compare(case: &Case, ninja: &[NinjaLane]) -> Result<Value, Error> {
        let mut masters = OneLoopNativeMasters::new()?;
        let mut worst: BTreeMap<String, f64> = BTreeMap::new();
        let mut worst_call: BTreeMap<String, Value> = BTreeMap::new();
        for call in ninja.iter().flat_map(|l| &l.calls) {
            let mut out = vec![[C64::ZERO; 3]; call.family.outputs()];
            let mut ok = [false];
            masters.evaluate(call.family, &call.args, &mut out, &mut ok)?;
            for (k, (native, oneloop)) in out.iter().zip(&call.values).enumerate() {
                let scale = oneloop.iter().map(|z| z.abs()).fold(0., f64::max);
                let error = if !ok[0] {
                    f64::INFINITY
                } else if scale == 0. {
                    native.iter().map(|z| z.abs()).fold(0., f64::max)
                } else {
                    (0..3)
                        .map(|j| (native[j] - oneloop[j]).abs())
                        .fold(0., f64::max)
                        / scale
                };
                let name = match call.family {
                    FastMaster::BubbleRank2 | FastMaster::BubbleRank3 => {
                        ["B0", "B1", "B11", "B111"][k].to_string()
                    }
                    family => format!("{family:?}"),
                };
                let entry = worst.entry(name.clone()).or_insert(0.);
                if error.is_nan() || error > *entry {
                    *entry = error;
                    let args: Vec<[f64; 2]> = call.args.iter().map(|z| [z.re, z.im]).collect();
                    let pair = |v: &[C64; 3]| v.map(|z| [z.re, z.im]);
                    worst_call.insert(
                        name,
                        json!({"args": args, "native": pair(&out[k]), "oneloop": pair(&call.values[k]),
                            "ok": ok[0]}),
                    );
                }
            }
        }
        let mut amplitude = FastAmplitude::new(case.n, case.rank, FastOptions::default())?;
        let mut full: f64 = 0.;
        let mut failed = 0;
        for (index, reference) in ninja.iter().enumerate() {
            if reference.status != 0 {
                continue;
            }
            let mut batch = Batch::new(case, &[index]);
            let mut tensor =
                std::mem::replace(&mut batch.tensor, TensorNumerator::new(case.rank, 0));
            let mut results = [FastResult::default()];
            let mut statuses = [FastStatus::NotEvaluated];
            amplitude.evaluate(
                &batch.input(),
                &mut tensor,
                &mut masters,
                &mut results,
                &mut statuses,
            )?;
            if statuses[0] != FastStatus::Success {
                failed += 1;
                continue;
            }
            let got = values(&results[0]);
            let scale = reference
                .values
                .iter()
                .map(|z| z.abs())
                .fold(1e-300, f64::max);
            for (g, r) in got.iter().zip(&reference.values).take(3) {
                full = full.max((*g - *r).abs() / scale);
            }
        }
        Ok(
            json!({"native_vs_oneloop": worst, "native_worst_calls": worst_call,
            "native_full_max_scaled": full, "native_failed_lanes": failed}),
        )
    }
}

fn main() -> Result<(), Error> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("--benchmark-dir") {
        let dir = PathBuf::from(args.get(1).ok_or("--benchmark-dir needs a path")?);
        for sub in ["cases", "synthetic"] {
            if !dir.join(sub).is_dir() {
                continue;
            }
            let mut files: Vec<PathBuf> = std::fs::read_dir(dir.join(sub))?
                .map(|e| e.map(|e| e.path()))
                .collect::<Result<_, _>>()?;
            files.sort();
            for case in files
                .iter()
                .filter(|p| p.extension().is_some_and(|e| e == "json"))
            {
                let stem = case.file_stem().ok_or("case name")?.to_string_lossy();
                let ninja = dir.join("ninja").join(format!("{stem}.json"));
                let masters = dir.join("ninja").join(format!("{stem}.masters.json"));
                run(case, &ninja, masters.exists().then_some(masters.as_path()))?;
            }
        }
        return Ok(());
    }
    for group in args.split(|a| a == "--") {
        match group {
            [case, ninja] => run(Path::new(case), Path::new(ninja), None)?,
            [case, ninja, masters] => {
                run(Path::new(case), Path::new(ninja), Some(Path::new(masters)))?
            }
            [] => {}
            _ => return Err("expected CASE.json NINJA.json [MASTERS.json]".into()),
        }
    }
    Ok(())
}
