//! Physical tensor-numerator generation and per-point preparation timings.
//!
//! Builds a `PhysicalTensorNumerator` from a fixture's graph/model/color
//! policy and a shared phase-space pool's runtime model parameters, reports
//! the cold setup stages, optionally checks the prepared tensors and propagator
//! data against an exported bench case, and measures warmed per-point
//! preparation (wavefunctions, coefficient heads, tensor scatter; propagator
//! data separately) at the requested batch widths.

use std::{
    collections::BTreeMap,
    error::Error,
    fs,
    hint::black_box,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

use clap::{Args as ClapArgs, ValueEnum};
use ninjars_graph::{
    CoefficientBackend, GenerationTimings, Model, NumeratorMode, PhysicalNumeratorSpec,
    PhysicalTensorNumerator,
};
use serde::Deserialize;
use serde_json::{Value, json};
use symbolica::{
    atom::Atom,
    domains::float::{Complex, Real},
    parser::ParseSettings,
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Clone, Copy, ValueEnum)]
pub enum Backend {
    SymjitO2,
    Interpreter,
}

#[derive(ClapArgs)]
pub struct Args {
    /// Schema-version-1 JSON fixture supplying graph, model and color policy.
    pub input: PathBuf,
    /// Shared phase-space pool ("ninjars-phase-space-pool-v1").
    #[arg(long)]
    pub points: PathBuf,
    /// Re-apply the model's Feynman rules instead of the DOT's local factors.
    #[arg(long)]
    pub feynman_rules: bool,
    /// R2 from the numerator: `off` (four-dimensional loop momentum, R1 only)
    /// or `fdh` (K.K -> q.q - mu^2 in FDH; the tensor gains mu^2 blocks).
    #[arg(long, value_enum, default_value = "off")]
    pub r2: crate::prepare::R2,
    #[arg(long, value_enum, default_value = "symjit-o2")]
    pub backend: Backend,
    /// Factor coefficient heads exactly before compilation.
    #[arg(long)]
    pub factor: bool,
    #[arg(long)]
    pub rank: Option<usize>,
    #[arg(long, value_delimiter = ',', default_value = "1,8,32")]
    pub widths: Vec<usize>,
    /// Timed blocks per measurement; the median is reported.
    #[arg(long, default_value = "9")]
    pub blocks: usize,
    /// Minimum duration of one timed block in milliseconds.
    #[arg(long, default_value = "20")]
    pub min_block_ms: f64,
    /// Only build the numerator and report generation timings.
    #[arg(long)]
    pub generation_only: bool,
    /// Bench case ("ninjars-bench-case-v1") whose lanes correspond to the pool points.
    #[arg(long)]
    pub check_case: Option<PathBuf>,
    /// MadLoop's OSThres for the kinematic matrix.
    #[arg(long, default_value = "1e-8")]
    pub on_shell_threshold: f64,
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

#[derive(Deserialize)]
pub(crate) struct PoolPoint {
    pub(crate) id: String,
    pub(crate) momenta: Vec<[f64; 4]>,
    pub(crate) helicities: Vec<i8>,
    pub(crate) model: Vec<f64>,
    pub(crate) scale_squared: f64,
}

#[derive(Deserialize)]
pub(crate) struct Pool {
    pub(crate) format: String,
    pub(crate) model_parameters: Vec<String>,
    #[serde(default)]
    pub(crate) fixed_parameters: BTreeMap<String, String>,
    pub(crate) points: Vec<PoolPoint>,
}

pub(crate) fn ns(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e9
}

pub(crate) fn generation_json(t: &GenerationTimings) -> Value {
    let p = &t.preparation;
    json!({
        "graph_import": ns(p.import),
        "feynman_rules": ns(p.feynman_rules),
        "propagators": ns(p.propagators),
        "contraction": ns(p.contraction),
        "color_invariants": ns(t.color),
        "polynomial": ns(t.polynomial),
        "prepare_total": ns(p.import + p.feynman_rules + p.propagators + p.contraction + t.color + t.polynomial),
        "coefficient_planning": ns(t.planning),
        "coefficient_factoring": ns(t.factoring),
        "tensor_plan": ns(t.tensor_plan),
        "planning_total": ns(t.planning + t.factoring + t.tensor_plan),
        "evaluator_construction": ns(t.evaluator),
        "symjit_compilation": ns(t.compilation),
        "compilation_total": ns(t.evaluator + t.compilation),
        "kinematics_setup": ns(t.kinematics),
        "total": ns(t.total()),
    })
}

pub(crate) fn model(name: &str, base: &Path) -> Result<Model> {
    crate::prepare::resolve_model(name, base)
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        0.5 * (values[n / 2 - 1] + values[n / 2])
    }
}

/// Run `body` (one call over `width` points) in `blocks` timed blocks of at
/// least `min_block` each; returns ns per point statistics.
pub(crate) fn measure(
    blocks: usize,
    min_block: Duration,
    width: usize,
    mut body: impl FnMut() -> Result<()>,
) -> Result<Value> {
    for _ in 0..3 {
        body()?;
    }
    let mut calls = 1usize;
    loop {
        let start = Instant::now();
        for _ in 0..calls {
            body()?;
        }
        if start.elapsed() >= min_block {
            break;
        }
        calls *= 2;
    }
    let mut per_point = Vec::with_capacity(blocks);
    let mut block_ms = Vec::with_capacity(blocks);
    for _ in 0..blocks {
        let start = Instant::now();
        for _ in 0..calls {
            body()?;
        }
        let elapsed = start.elapsed();
        block_ms.push(elapsed.as_secs_f64() * 1e3);
        per_point.push(ns(elapsed) / (calls * width) as f64);
    }
    let raw = per_point.clone();
    let median = median(&mut per_point);
    Ok(json!({
        "calls_per_block": calls,
        "blocks": blocks,
        "median_ns_per_point": median,
        "min_ns_per_point": per_point[0],
        "max_ns_per_point": per_point[per_point.len() - 1],
        "block_ns_per_point": raw,
        "block_ms": block_ms,
    }))
}

pub(crate) fn affinity() -> Value {
    fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status
                .lines()
                .find(|line| line.starts_with("Cpus_allowed_list"))
                .map(|line| line.split(':').nth(1).unwrap_or("").trim().to_string())
        })
        .map_or(Value::Null, Value::String)
}

fn check_case(
    numerator: &mut PhysicalTensorNumerator,
    pool: &Pool,
    case: &Path,
    threshold: f64,
) -> Result<Value> {
    let case: Value = serde_json::from_str(&fs::read_to_string(case)?)?;
    let mu_powers = match case["format"].as_str() {
        Some("ninjars-bench-case-v1") => 0,
        Some("ninjars-bench-case-v2") => case["mu_powers"].as_u64().unwrap_or(0) as usize,
        _ => return Err("--check-case expects a ninjars-bench-case-v1/v2 file".into()),
    };
    if mu_powers != numerator.mu_powers() {
        return Err("bench case mu^2 powers differ from the physical numerator".into());
    }
    let lanes = case["lanes"].as_array().ok_or("bench case has no lanes")?;
    if lanes.len() != pool.points.len() {
        return Err(format!(
            "bench case has {} lanes; the pool has {} points",
            lanes.len(),
            pool.points.len()
        )
        .into());
    }
    let n = numerator.propagators();
    if case["n"].as_u64() != Some(n as u64)
        || case["rank"].as_u64() != Some(numerator.rank() as u64)
    {
        return Err("bench case n/rank differ from the physical numerator".into());
    }
    let rows = pool.points.len();
    let momenta: Vec<_> = pool
        .points
        .iter()
        .flat_map(|p| p.momenta.iter().copied())
        .collect();
    let helicities: Vec<_> = pool
        .points
        .iter()
        .flat_map(|p| p.helicities.iter().copied())
        .collect();
    let model: Vec<_> = pool
        .points
        .iter()
        .flat_map(|p| p.model.iter().copied())
        .collect();
    let ncoeff = numerator.ncoeff();
    let mut tensor = vec![Complex::new(0., 0.); rows * ncoeff];
    numerator.prepare(&momenta, &helicities, &model, rows, &mut tensor)?;
    let mut shifts = vec![[0.; 4]; rows * n];
    let mut masses = vec![Complex::new(0., 0.); rows * n];
    let mut s_matrix = vec![0.; rows * n * n];
    numerator.kinematics(
        &momenta,
        &model,
        rows,
        &mut shifts,
        &mut masses,
        Some(&mut s_matrix),
        threshold,
    )?;
    let number = |v: &Value| v.as_f64().ok_or("non-numeric bench-case entry");
    let mut worst = [0_f64; 4];
    let mut ids = Vec::new();
    for (row, lane) in lanes.iter().enumerate() {
        ids.push(json!([pool.points[row].id, lane["id"]]));
        let expected = lane["tensor"].as_array().ok_or("missing tensor")?;
        if expected.len() != ncoeff {
            return Err("bench-case tensor length differs".into());
        }
        let expected = expected
            .iter()
            .map(|c| Ok(Complex::new(number(&c[0])?, number(&c[1])?)))
            .collect::<Result<Vec<_>>>()?;
        let norm = expected.iter().fold(0_f64, |m, c| m.max(c.norm().re));
        for (actual, expected) in tensor[row * ncoeff..(row + 1) * ncoeff]
            .iter()
            .zip(&expected)
        {
            worst[0] = worst[0].max((*actual - *expected).norm().re / norm);
        }
        let energy = shifts[row * n..(row + 1) * n]
            .iter()
            .fold(1_f64, |m, v| m.max(v[0].abs()));
        for (i, shift) in shifts[row * n..(row + 1) * n].iter().enumerate() {
            for (k, value) in shift.iter().enumerate() {
                worst[1] = worst[1].max((value - number(&lane["shifts"][i][k])?).abs() / energy);
            }
            let mass = Complex::new(
                number(&lane["masses_squared"][i][0])?,
                number(&lane["masses_squared"][i][1])?,
            );
            worst[2] =
                worst[2].max((masses[row * n + i] - mass).norm().re / mass.norm().re.max(1.));
            if !lane["s_matrix"].is_null() {
                for j in 0..n {
                    let expected = number(&lane["s_matrix"][i][j])?;
                    worst[3] = worst[3]
                        .max((s_matrix[row * n * n + i * n + j] - expected).abs() / energy.powi(2));
                }
            }
        }
    }
    Ok(json!({
        "case": case["name"],
        "pool_point_and_case_lane_ids": ids,
        "max_tensor_error_relative_to_lane_max": worst[0],
        "max_shift_error_relative_to_energy": worst[1],
        "max_mass_squared_error_relative": worst[2],
        "max_s_matrix_error_relative_to_energy_squared": worst[3],
    }))
}

pub fn run(args: Args) -> Result<()> {
    let fixture: Value = serde_json::from_str(&fs::read_to_string(&args.input)?)?;
    let base = args.input.parent().unwrap_or_else(|| Path::new("."));
    let graph = base.join(fixture["graph"].as_str().ok_or("fixture has no graph")?);
    let model_name = fixture["model"].as_str().unwrap_or("phi3");
    let pool: Pool = serde_json::from_str(&fs::read_to_string(&args.points)?)?;
    if pool.format != "ninjars-phase-space-pool-v1" {
        return Err("--points expects a ninjars-phase-space-pool-v1 file".into());
    }
    let fixed_parameters = pool
        .fixed_parameters
        .iter()
        .map(|(name, value)| {
            Ok((
                name.clone(),
                Atom::parse(value, "ninjars", ParseSettings::default())
                    .map_err(|e| format!("fixed parameter {name}: {e}"))?,
            ))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    let spec = PhysicalNumeratorSpec {
        model: Arc::new(model(model_name, base)?),
        dot: fs::read_to_string(&graph)?,
        numerator_mode: if args.feynman_rules {
            NumeratorMode::FeynmanRules
        } else {
            NumeratorMode::Explicit
        },
        color_dimension_invariants: fixture["color_dimension_invariants"]
            .as_bool()
            .unwrap_or(false),
        runtime_parameters: pool.model_parameters.clone(),
        fixed_parameters,
        factor_coefficients: args.factor,
        tensor_rank: args.rank,
        backend: match args.backend {
            Backend::SymjitO2 => CoefficientBackend::SymJitO2,
            Backend::Interpreter => CoefficientBackend::Interpreter,
        },
        r2: args.r2.scheme(),
    };
    let start = Instant::now();
    let mut numerator = PhysicalTensorNumerator::new(&spec)?;
    let construction = start.elapsed();
    let legs = numerator.legs_info().len();
    for point in &pool.points {
        if point.momenta.len() != legs
            || point.helicities.len() != legs
            || point.model.len() != pool.model_parameters.len()
        {
            return Err(format!(
                "pool point {} does not match the graph's legs/model",
                point.id
            )
            .into());
        }
    }
    let check = args
        .check_case
        .as_ref()
        .map(|case| check_case(&mut numerator, &pool, case, args.on_shell_threshold))
        .transpose()?;

    let mut timings = Vec::new();
    if !args.generation_only {
        let min_block = Duration::from_secs_f64(args.min_block_ms * 1e-3);
        let n = numerator.propagators();
        let ncoeff = numerator.ncoeff();
        for &width in &args.widths {
            if width == 0 {
                return Err("timing widths must be positive".into());
            }
            let points: Vec<_> = (0..width)
                .map(|i| &pool.points[i % pool.points.len()])
                .collect();
            let momenta: Vec<_> = points
                .iter()
                .flat_map(|p| p.momenta.iter().copied())
                .collect();
            let helicities: Vec<_> = points
                .iter()
                .flat_map(|p| p.helicities.iter().copied())
                .collect();
            let model: Vec<_> = points
                .iter()
                .flat_map(|p| p.model.iter().copied())
                .collect();
            let mut tensor = vec![Complex::new(0., 0.); width * ncoeff];
            let mut shifts = vec![[0.; 4]; width * n];
            let mut masses = vec![Complex::new(0., 0.); width * n];
            let mut s_matrix = vec![0.; width * n * n];
            let threshold = args.on_shell_threshold;
            let tensor_only = measure(args.blocks, min_block, width, || {
                numerator.prepare(&momenta, &helicities, &model, width, &mut tensor)?;
                black_box(&tensor);
                Ok(())
            })?;
            let kinematics_only = measure(args.blocks, min_block, width, || {
                numerator.kinematics(
                    &momenta,
                    &model,
                    width,
                    &mut shifts,
                    &mut masses,
                    Some(&mut s_matrix),
                    threshold,
                )?;
                black_box(&s_matrix);
                Ok(())
            })?;
            let combined = measure(args.blocks, min_block, width, || {
                numerator.prepare(&momenta, &helicities, &model, width, &mut tensor)?;
                numerator.kinematics(
                    &momenta,
                    &model,
                    width,
                    &mut shifts,
                    &mut masses,
                    Some(&mut s_matrix),
                    threshold,
                )?;
                black_box((&tensor, &s_matrix));
                Ok(())
            })?;
            timings.push(json!({
                "width": width,
                "tensor_coefficients": tensor_only,
                "propagator_data": kinematics_only,
                "tensor_and_propagator_data": combined,
            }));
        }
    }
    let plan = numerator.coefficient_plan();
    let report = json!({
        "schema_version": 1,
        "operation": "numerator-bench",
        "command": std::env::args().collect::<Vec<_>>(),
        "input": args.input,
        "graph": graph,
        "points": args.points,
        "point_ids": pool.points.iter().map(|p| p.id.clone()).collect::<Vec<_>>(),
        "scale_squared": pool.points.iter().map(|p| p.scale_squared).collect::<Vec<_>>(),
        "numerator_mode": if args.feynman_rules { "feynman_rules" } else { "explicit" },
        "backend": match args.backend { Backend::SymjitO2 => "symjit-o2", Backend::Interpreter => "symbolica-interpreter" },
        "coefficients_factored": args.factor,
        "legs": numerator.legs_info().iter().map(|leg| json!({"index":leg.index,"pdg":leg.pdg,"incoming":leg.incoming,"state":format!("{:?}",leg.state)})).collect::<Vec<_>>(),
        "runtime_model_parameters": numerator.runtime_parameters(),
        "propagators": numerator.propagators(),
        "rank": numerator.rank(),
        "numerator_degree": plan.rank(),
        "r2": args.r2.name(),
        "mu_powers": numerator.mu_powers(),
        "ncoeff": numerator.ncoeff(),
        "nonzero_tensor_slots": numerator.tensor_plan().entries().len(),
        "coefficient_heads": numerator.head_count(),
        "evaluator_inputs": numerator.input_columns(),
        "generation_ns": generation_json(numerator.generation_timings()),
        "construction_wall_ns": ns(construction),
        "check_case": check,
        "per_point_timing": timings,
        "cpu_affinity": affinity(),
        "boundary": "per point: raw binary64 momenta, helicities and model row -> external wavefunctions -> all coefficient heads -> dense Ninja tensor (lanes x ncoeff); propagator data (shifts, m^2, MadLoop-thresholded S matrix) measured separately; setup (graph, rules, contraction, planning, factoring, compilation) excluded",
    });
    let report = serde_json::to_string_pretty(&report)? + "\n";
    if let Some(path) = args.output {
        fs::write(path, report)?;
    } else {
        print!("{report}");
    }
    Ok(())
}
