//! Binary64 benchmark of shared bench cases ("ninjars-bench-case-v1").
//!
//! Each case gives per-lane propagator data and dense Ninja tensor
//! coefficients, exactly as the C++ reference driver
//! (`validation/benchmark/ninja_bench.cc`) consumes them. For every batch
//! width this times, per phase-space point:
//!
//! - `reduction`: `FastAmplitude::evaluate` with recorded numerator outputs and
//!   master values replayed at O(1) cost (pure reduction);
//! - `reduction_and_tensor_numerator`: the live `TensorNumerator` with replayed
//!   masters (the difference is the tensor-numerator time);
//! - `masters`: the recorded master calls replayed through oneloopmaster's
//!   native backend, without reduction;
//! - `total`: tensor numerator, reduction and native masters together.

use std::{error::Error, fs, hint::black_box, path::PathBuf, time::Duration};

use clap::Args as ClapArgs;
use ninjars::fast::{
    C64, FastAmplitude, FastInput, FastMaster, FastMasters, FastOptions, FastResult, FastStatus,
    RMom, RecordingMasters, RecordingNumerator, TensorNumerator, tensor_layout::ncoeff,
};
use ninjars_symbolica::OneLoopNativeMasters;
use serde_json::{Value, json};

use crate::numerator_bench::{affinity, measure};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(ClapArgs)]
pub struct Args {
    /// One or more "ninjars-bench-case-v1" files.
    pub cases: Vec<PathBuf>,
    #[arg(long, value_delimiter = ',', default_value = "1,32")]
    pub widths: Vec<usize>,
    /// Timed blocks per measurement; the median is reported.
    #[arg(long, default_value = "7")]
    pub blocks: usize,
    /// Minimum duration of one timed block in milliseconds.
    #[arg(long, default_value = "20")]
    pub min_block_ms: f64,
    /// Directory of original-Ninja results ("<case>.json") to check against.
    #[arg(long)]
    pub reference_dir: Option<PathBuf>,
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

struct Lane {
    id: String,
    shifts: Vec<RMom>,
    masses: Vec<C64>,
    mu2: f64,
    s_matrix: Option<Vec<f64>>,
    tensor: Vec<C64>,
}

struct Case {
    name: String,
    n: usize,
    rank: usize,
    lanes: Vec<Lane>,
}

fn number(value: &Value) -> Result<f64> {
    value.as_f64().ok_or_else(|| "expected a number".into())
}

fn complex(value: &Value) -> Result<C64> {
    Ok(C64::new(number(&value[0])?, number(&value[1])?))
}

fn load(path: &PathBuf) -> Result<Case> {
    let value: Value = serde_json::from_str(&fs::read_to_string(path)?)?;
    if value["format"] != "ninjars-bench-case-v1" {
        return Err(format!("{}: not a ninjars-bench-case-v1 file", path.display()).into());
    }
    let n = value["n"].as_u64().ok_or("n")? as usize;
    let rank = value["rank"].as_u64().ok_or("rank")? as usize;
    let mut lanes = Vec::new();
    for lane in value["lanes"].as_array().ok_or("lanes")? {
        let shifts = lane["shifts"]
            .as_array()
            .ok_or("shifts")?
            .iter()
            .map(|p| {
                Ok([
                    number(&p[0])?,
                    number(&p[1])?,
                    number(&p[2])?,
                    number(&p[3])?,
                ])
            })
            .collect::<Result<Vec<_>>>()?;
        let masses = lane["masses_squared"]
            .as_array()
            .ok_or("masses_squared")?
            .iter()
            .map(complex)
            .collect::<Result<Vec<_>>>()?;
        let s_matrix = match &lane["s_matrix"] {
            Value::Null => None,
            rows => Some(
                rows.as_array()
                    .ok_or("s_matrix")?
                    .iter()
                    .flat_map(|row| row.as_array().into_iter().flatten())
                    .map(number)
                    .collect::<Result<Vec<_>>>()?,
            ),
        };
        let tensor = lane["tensor"]
            .as_array()
            .ok_or("tensor")?
            .iter()
            .map(complex)
            .collect::<Result<Vec<_>>>()?;
        if shifts.len() != n || masses.len() != n || tensor.len() != ncoeff(rank) {
            return Err(format!("{}: malformed lane", path.display()).into());
        }
        lanes.push(Lane {
            id: lane["id"].as_str().unwrap_or("").to_string(),
            shifts,
            masses,
            mu2: number(&lane["mu_squared"])?,
            s_matrix,
            tensor,
        });
    }
    if lanes.is_empty() {
        return Err(format!("{}: no lanes", path.display()).into());
    }
    Ok(Case {
        name: value["name"].as_str().unwrap_or("").to_string(),
        n,
        rank,
        lanes,
    })
}

/// Records every master call with its arguments.
struct ArgumentRecorder<M> {
    inner: M,
    calls: Vec<(FastMaster, Vec<C64>, usize)>,
}

impl<M: FastMasters> FastMasters for ArgumentRecorder<M> {
    fn evaluate(
        &mut self,
        family: FastMaster,
        args: &[C64],
        out: &mut [[C64; 3]],
        ok: &mut [bool],
    ) -> ninjars::Result<()> {
        self.inner.evaluate(family, args, out, ok)?;
        self.calls.push((family, args.to_vec(), ok.len()));
        Ok(())
    }
}

fn reference_check(
    reference_dir: &Option<PathBuf>,
    case: &Case,
    ids: &[String],
    results: &[FastResult],
    statuses: &[FastStatus],
) -> Result<Value> {
    let Some(dir) = reference_dir else {
        return Ok(Value::Null);
    };
    let path = dir.join(format!("{}.json", case.name));
    let Ok(text) = fs::read_to_string(&path) else {
        return Ok(Value::Null);
    };
    let reference: Value = serde_json::from_str(&text)?;
    let lanes = reference["lanes"].as_array().ok_or("reference lanes")?;
    let mut worst = 0_f64;
    let mut checked = 0;
    for (lane, id) in ids.iter().enumerate().take(case.lanes.len()) {
        let Some(expected) = lanes.iter().find(|l| l["id"] == id.as_str()) else {
            continue;
        };
        if statuses[lane] != FastStatus::Success {
            continue;
        }
        let expected = [
            complex(&expected["eps0"])?,
            complex(&expected["eps_m1"])?,
            complex(&expected["eps_m2"])?,
        ];
        let scale = expected.iter().fold(0_f64, |m, z| m.max(z.abs()));
        if scale == 0. {
            continue;
        }
        checked += 1;
        for (a, b) in results[lane].laurent.iter().zip(&expected) {
            worst = worst.max((*a - *b).abs() / scale);
        }
    }
    Ok(json!({
        "reference": path,
        "lanes_checked": checked,
        "max_scaled_laurent_difference": worst,
    }))
}

pub fn run(args: Args) -> Result<()> {
    let min_block = Duration::from_secs_f64(args.min_block_ms * 1e-3);
    let mut masters = OneLoopNativeMasters::new()?;
    let mut reports = Vec::new();
    for path in &args.cases {
        let case = load(path)?;
        let (n, rank) = (case.n, case.rank);
        let mut amplitude = FastAmplitude::new(n, rank, FastOptions::default())?;
        let mut widths = Vec::new();
        for &width in &args.widths {
            if width == 0 {
                return Err("timing widths must be positive".into());
            }
            let lanes: Vec<&Lane> = (0..width)
                .map(|i| &case.lanes[i % case.lanes.len()])
                .collect();
            let shifts: Vec<RMom> = lanes
                .iter()
                .flat_map(|l| l.shifts.iter().copied())
                .collect();
            let masses: Vec<C64> = lanes
                .iter()
                .flat_map(|l| l.masses.iter().copied())
                .collect();
            let scales: Vec<f64> = lanes.iter().map(|l| l.mu2).collect();
            let s_matrix: Option<Vec<f64>> = lanes
                .iter()
                .map(|l| l.s_matrix.clone())
                .collect::<Option<Vec<_>>>()
                .map(|rows| rows.concat());
            let ids: Vec<String> = lanes.iter().map(|l| l.id.clone()).collect();
            let mut tensor = TensorNumerator::new(rank, width);
            for (lane, data) in lanes.iter().enumerate() {
                tensor
                    .lane_coefficients_mut(lane)
                    .copy_from_slice(&data.tensor);
            }
            let input = FastInput {
                lanes: width,
                shifts: &shifts,
                masses_squared: &masses,
                scales_squared: &scales,
                s_matrix: s_matrix.as_deref(),
            };
            let mut results = vec![FastResult::default(); width];
            let mut statuses = vec![FastStatus::NotEvaluated; width];

            let mut recording_numerator =
                RecordingNumerator::new(TensorNumerator::new(rank, width));
            recording_numerator
                .inner_mut()
                .coefficients_mut()
                .copy_from_slice(tensor.coefficients());
            let mut recorder = ArgumentRecorder {
                inner: RecordingMasters::new(OneLoopNativeMasters::new()?),
                calls: Vec::new(),
            };
            amplitude.evaluate(
                &input,
                &mut recording_numerator,
                &mut recorder,
                &mut results,
                &mut statuses,
            )?;
            let failed = statuses
                .iter()
                .filter(|s| **s != FastStatus::Success)
                .count();
            let mut failure_kinds = std::collections::BTreeMap::<String, usize>::new();
            for status in statuses.iter().filter(|s| **s != FastStatus::Success) {
                *failure_kinds.entry(format!("{status:?}")).or_default() += 1;
            }
            let check = reference_check(&args.reference_dir, &case, &ids, &results, &statuses)?;
            let mut replay_numerator = recording_numerator.replay();
            let mut replay_masters = recorder.inner.replay();
            let master_calls = recorder.calls;

            let reduction = measure(args.blocks, min_block, width, || {
                replay_numerator.rewind();
                replay_masters.rewind();
                amplitude.evaluate(
                    &input,
                    &mut replay_numerator,
                    &mut replay_masters,
                    &mut results,
                    &mut statuses,
                )?;
                black_box(&results);
                Ok(())
            })?;
            let reduction_and_tensor = measure(args.blocks, min_block, width, || {
                replay_masters.rewind();
                amplitude.evaluate(
                    &input,
                    &mut tensor,
                    &mut replay_masters,
                    &mut results,
                    &mut statuses,
                )?;
                black_box(&results);
                Ok(())
            })?;
            let mut master_out: Vec<Vec<[C64; 3]>> = master_calls
                .iter()
                .map(|(family, _, rows)| vec![[C64::ZERO; 3]; rows * family.outputs()])
                .collect();
            let mut master_ok: Vec<Vec<bool>> = master_calls
                .iter()
                .map(|(_, _, rows)| vec![false; *rows])
                .collect();
            let masters_only = measure(args.blocks, min_block, width, || {
                for ((call, out), ok) in master_calls
                    .iter()
                    .zip(master_out.iter_mut())
                    .zip(master_ok.iter_mut())
                {
                    masters.evaluate(call.0, &call.1, out, ok)?;
                }
                black_box(&master_out);
                Ok(())
            })?;
            let total = measure(args.blocks, min_block, width, || {
                amplitude.evaluate(
                    &input,
                    &mut tensor,
                    &mut masters,
                    &mut results,
                    &mut statuses,
                )?;
                black_box(&results);
                Ok(())
            })?;
            let median = |v: &Value| v["median_ns_per_point"].as_f64().unwrap_or(f64::NAN);
            let tensor_ns = median(&reduction_and_tensor) - median(&reduction);
            let sum = tensor_ns + median(&reduction) + median(&masters_only);
            let mut families = std::collections::BTreeMap::<String, usize>::new();
            for (family, _, rows) in &master_calls {
                *families.entry(format!("{family:?}")).or_default() += rows;
            }
            widths.push(json!({
                "width": width,
                "failed_lanes": failed,
                "failure_kinds": failure_kinds,
                "reduction": reduction,
                "reduction_and_tensor_numerator": reduction_and_tensor,
                "masters": masters_only,
                "total": total,
                "summary_ns_per_point": {
                    "tensor_numerator": tensor_ns,
                    "reduction": median(&reduction),
                    "masters": median(&masters_only),
                    "total": median(&total),
                    "sum_of_components": sum,
                    "sum_over_total": sum / median(&total),
                },
                "master_rows_per_batch": families,
                "reference_check": check,
            }));
        }
        reports.push(json!({
            "case": case.name,
            "file": path,
            "n": n,
            "rank": rank,
            "lanes": case.lanes.len(),
            "per_point_timing": widths,
        }));
    }
    let report = json!({
        "schema_version": 1,
        "operation": "case-bench",
        "command": std::env::args().collect::<Vec<_>>(),
        "cases": reports,
        "cpu_affinity": affinity(),
        "boundary": "per point: TensorNumerator expansions (tensor_numerator = reduction_and_tensor_numerator - reduction), ninjars::fast reduction (replayed numerator and masters), oneloopmaster native masters (recorded calls replayed); total uninstrumented",
    });
    let report = serde_json::to_string_pretty(&report)? + "\n";
    if let Some(path) = args.output {
        fs::write(path, report)?;
    } else {
        print!("{report}");
    }
    Ok(())
}
