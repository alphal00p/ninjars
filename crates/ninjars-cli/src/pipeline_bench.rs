//! End-to-end binary64 pipeline benchmark of one physical graph.
//!
//! The pipeline per batch of phase-space points is: external wavefunctions
//! and model couplings -> SymJIT O2 coefficient heads -> dense Ninja tensor
//! (`PhysicalTensorNumerator`) and propagator data -> `ninjars::fast`
//! reduction with the batched `TensorNumerator` -> oneloopmaster's native
//! master integrals. Besides the uninstrumented total, every component is
//! timed on its own through replay providers, mirroring the C++ reference
//! harness (`validation/benchmark/ninja_bench.cc`):
//!
//! - `numerator_preparation`: wavefunctions, coefficient heads, tensor
//!   scatter and propagator data;
//! - `reduction`: `FastAmplitude::evaluate` with recorded numerator outputs and
//!   master values replayed at O(1) cost (pure reduction);
//! - `reduction_and_tensor_numerator`: the live `TensorNumerator` with replayed
//!   masters; the difference to `reduction` is the tensor-numerator time;
//! - `masters`: the recorded master calls replayed through the live native
//!   provider, family by family, without any reduction.

use std::{
    error::Error,
    fs,
    hint::black_box,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

use clap::Args as ClapArgs;
use ninjars::fast::{
    C64, FastAmplitude, FastInput, FastMaster, FastMasters, FastOptions, FastResult, FastStatus,
    RMom, RecordingMasters, RecordingNumerator, TensorNumerator,
};
use ninjars_graph::{
    CoefficientBackend, NumeratorMode, PhysicalNumeratorSpec, PhysicalTensorNumerator,
};
use ninjars_symbolica::OneLoopNativeMasters;
use serde_json::{Value, json};
use symbolica::{atom::Atom, domains::float::Complex, parser::ParseSettings};

use crate::numerator_bench::{Pool, affinity, generation_json, measure, model, ns};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

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
    /// Factor coefficient heads exactly before compilation.
    #[arg(long)]
    pub factor: bool,
    /// Tensor rank (defaults to the numerator degree).
    #[arg(long)]
    pub rank: Option<usize>,
    #[arg(long, value_delimiter = ',', default_value = "1,8,32,128")]
    pub widths: Vec<usize>,
    /// Timed blocks per measurement; the median is reported.
    #[arg(long, default_value = "9")]
    pub blocks: usize,
    /// Minimum duration of one timed block in milliseconds.
    #[arg(long, default_value = "20")]
    pub min_block_ms: f64,
    /// MadLoop's OSThres for the kinematic matrix.
    #[arg(long, default_value = "1e-8")]
    pub on_shell_threshold: f64,
    /// Original-Ninja results ("ninjars-reference-ninja-results-v1") for the
    /// same pool points, used to check the integrated results.
    #[arg(long)]
    pub reference: Option<PathBuf>,
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

/// Records every master call with its arguments so the master work can be
/// replayed on its own.
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

/// Batched inputs of one timing width, cycling over the pool points.
struct Batch {
    width: usize,
    momenta: Vec<[f64; 4]>,
    helicities: Vec<i8>,
    model: Vec<f64>,
    scales: Vec<f64>,
    ids: Vec<String>,
}

impl Batch {
    fn new(pool: &Pool, width: usize) -> Self {
        let points: Vec<_> = (0..width)
            .map(|i| &pool.points[i % pool.points.len()])
            .collect();
        Self {
            width,
            momenta: points
                .iter()
                .flat_map(|p| p.momenta.iter().copied())
                .collect(),
            helicities: points
                .iter()
                .flat_map(|p| p.helicities.iter().copied())
                .collect(),
            model: points
                .iter()
                .flat_map(|p| p.model.iter().copied())
                .collect(),
            scales: points.iter().map(|p| p.scale_squared).collect(),
            ids: points.iter().map(|p| p.id.clone()).collect(),
        }
    }
}

/// Reusable per-width buffers of the whole pipeline.
struct Pipeline {
    tensor_raw: Vec<Complex<f64>>,
    masses_raw: Vec<Complex<f64>>,
    shifts: Vec<RMom>,
    masses: Vec<C64>,
    s_matrix: Vec<f64>,
    tensor: TensorNumerator,
    results: Vec<FastResult>,
    statuses: Vec<FastStatus>,
}

impl Pipeline {
    fn new(n: usize, rank: usize, ncoeff: usize, width: usize) -> Self {
        Self {
            tensor_raw: vec![Complex::new(0., 0.); width * ncoeff],
            masses_raw: vec![Complex::new(0., 0.); width * n],
            shifts: vec![[0.; 4]; width * n],
            masses: vec![C64::ZERO; width * n],
            s_matrix: vec![0.; width * n * n],
            tensor: TensorNumerator::new(rank, width),
            results: vec![FastResult::default(); width],
            statuses: vec![FastStatus::NotEvaluated; width],
        }
    }

    /// Numerator preparation: tensor coefficients and propagator data.
    fn prepare(
        &mut self,
        numerator: &mut PhysicalTensorNumerator,
        batch: &Batch,
        threshold: f64,
    ) -> Result<()> {
        let width = batch.width;
        numerator.prepare(
            &batch.momenta,
            &batch.helicities,
            &batch.model,
            width,
            &mut self.tensor_raw,
        )?;
        numerator.kinematics(
            &batch.momenta,
            &batch.model,
            width,
            &mut self.shifts,
            &mut self.masses_raw,
            Some(&mut self.s_matrix),
            threshold,
        )?;
        for (target, source) in self
            .tensor
            .coefficients_mut()
            .iter_mut()
            .zip(&self.tensor_raw)
        {
            *target = C64::from_symbolica(*source);
        }
        for (target, source) in self.masses.iter_mut().zip(&self.masses_raw) {
            *target = C64::from_symbolica(*source);
        }
        Ok(())
    }
}

fn check_statuses(statuses: &[FastStatus]) -> Result<()> {
    match statuses.iter().position(|s| *s != FastStatus::Success) {
        Some(lane) => Err(format!("lane {lane} failed: {:?}", statuses[lane]).into()),
        None => Ok(()),
    }
}

fn compare_reference(path: &Path, batch: &Batch, results: &[FastResult]) -> Result<Value> {
    let reference: Value = serde_json::from_str(&fs::read_to_string(path)?)?;
    let lanes = reference["lanes"]
        .as_array()
        .ok_or("reference has no lanes")?;
    let value = |v: &Value| -> Result<C64> {
        Ok(C64::new(
            v[0].as_f64().ok_or("reference entry")?,
            v[1].as_f64().ok_or("reference entry")?,
        ))
    };
    let mut worst = 0_f64;
    let mut checked = Vec::new();
    for (lane, id) in batch.ids.iter().enumerate() {
        let Some(expected) = lanes.iter().find(|l| l["id"] == id.as_str()) else {
            continue;
        };
        if checked.contains(id) {
            continue;
        }
        checked.push(id.clone());
        let expected = [
            value(&expected["eps0"])?,
            value(&expected["eps_m1"])?,
            value(&expected["eps_m2"])?,
        ];
        let scale = expected.iter().fold(0_f64, |m, z| m.max(z.abs()));
        for (a, b) in results[lane].laurent.iter().zip(&expected) {
            worst = worst.max((*a - *b).abs() / scale);
        }
    }
    Ok(json!({
        "reference": path,
        "points_checked": checked,
        "max_scaled_laurent_difference": worst,
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
        .collect::<Result<_>>()?;
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
        backend: CoefficientBackend::SymJitO2,
    };
    let start = Instant::now();
    let mut numerator = PhysicalTensorNumerator::new(&spec)?;
    let construction = start.elapsed();
    let (n, rank, ncoeff) = (
        numerator.propagators(),
        numerator.rank(),
        numerator.ncoeff(),
    );
    let mut amplitude = FastAmplitude::new(n, rank, FastOptions::default())?;
    let mut masters = OneLoopNativeMasters::new()?;
    let min_block = Duration::from_secs_f64(args.min_block_ms * 1e-3);
    let threshold = args.on_shell_threshold;

    let mut timings = Vec::new();
    for &width in &args.widths {
        if width == 0 {
            return Err("timing widths must be positive".into());
        }
        let batch = Batch::new(&pool, width);
        let mut pipe = Pipeline::new(n, rank, ncoeff, width);

        // Record one full evaluation: numerator outputs, master arguments
        // and values.
        pipe.prepare(&mut numerator, &batch, threshold)?;
        let mut recording_numerator = RecordingNumerator::new(TensorNumerator::new(rank, width));
        recording_numerator
            .inner_mut()
            .coefficients_mut()
            .copy_from_slice(pipe.tensor.coefficients());
        let mut recorder = ArgumentRecorder {
            inner: RecordingMasters::new(OneLoopNativeMasters::new()?),
            calls: Vec::new(),
        };
        let input = FastInput {
            lanes: width,
            shifts: &pipe.shifts,
            masses_squared: &pipe.masses,
            scales_squared: &batch.scales,
            s_matrix: Some(&pipe.s_matrix),
        };
        amplitude.evaluate(
            &input,
            &mut recording_numerator,
            &mut recorder,
            &mut pipe.results,
            &mut pipe.statuses,
        )?;
        check_statuses(&pipe.statuses)?;
        let check = args
            .reference
            .as_ref()
            .map(|path| compare_reference(path, &batch, &pipe.results))
            .transpose()?;
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
                &mut pipe.results,
                &mut pipe.statuses,
            )?;
            black_box(&pipe.results);
            Ok(())
        })?;
        let reduction_and_tensor = measure(args.blocks, min_block, width, || {
            replay_masters.rewind();
            amplitude.evaluate(
                &input,
                &mut pipe.tensor,
                &mut replay_masters,
                &mut pipe.results,
                &mut pipe.statuses,
            )?;
            black_box(&pipe.results);
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
        let preparation = measure(args.blocks, min_block, width, || {
            pipe.prepare(&mut numerator, &batch, threshold)?;
            black_box(&pipe.s_matrix);
            Ok(())
        })?;
        let total = measure(args.blocks, min_block, width, || {
            pipe.prepare(&mut numerator, &batch, threshold)?;
            let input = FastInput {
                lanes: width,
                shifts: &pipe.shifts,
                masses_squared: &pipe.masses,
                scales_squared: &batch.scales,
                s_matrix: Some(&pipe.s_matrix),
            };
            amplitude.evaluate(
                &input,
                &mut pipe.tensor,
                &mut masters,
                &mut pipe.results,
                &mut pipe.statuses,
            )?;
            black_box(&pipe.results);
            Ok(())
        })?;
        check_statuses(&pipe.statuses)?;

        let median = |v: &Value| v["median_ns_per_point"].as_f64().unwrap_or(f64::NAN);
        let tensor_ns = median(&reduction_and_tensor) - median(&reduction);
        let numerator_ns = median(&preparation) + tensor_ns;
        let sum = numerator_ns + median(&reduction) + median(&masters_only);
        let mut families = std::collections::BTreeMap::<String, usize>::new();
        for (family, _, rows) in &master_calls {
            *families.entry(format!("{family:?}")).or_default() += rows;
        }
        timings.push(json!({
            "width": width,
            "numerator_preparation": preparation,
            "reduction": reduction,
            "reduction_and_tensor_numerator": reduction_and_tensor,
            "masters": masters_only,
            "total": total,
            "summary_ns_per_point": {
                "numerator": numerator_ns,
                "numerator_preparation": median(&preparation),
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
    let report = json!({
        "schema_version": 1,
        "operation": "pipeline-bench",
        "command": std::env::args().collect::<Vec<_>>(),
        "input": args.input,
        "graph": graph,
        "points": args.points,
        "point_ids": pool.points.iter().map(|p| p.id.clone()).collect::<Vec<_>>(),
        "propagators": n,
        "rank": rank,
        "ncoeff": ncoeff,
        "coefficient_heads": numerator.head_count(),
        "coefficients_factored": args.factor,
        "numerator_mode": if args.feynman_rules { "feynman_rules" } else { "explicit" },
        "generation_ns": generation_json(numerator.generation_timings()),
        "construction_wall_ns": ns(construction),
        "per_point_timing": timings,
        "cpu_affinity": affinity(),
        "boundary": "per point: raw binary64 momenta, helicities and model row -> wavefunctions -> SymJIT O2 coefficient heads -> dense Ninja tensor + propagator data (numerator_preparation) -> ninjars::fast reduction with the batched TensorNumerator (tensor_numerator = reduction_and_tensor_numerator - reduction) -> oneloopmaster native masters; total is uninstrumented; setup excluded",
    });
    let report = serde_json::to_string_pretty(&report)? + "\n";
    if let Some(path) = args.output {
        fs::write(path, report)?;
    } else {
        print!("{report}");
    }
    Ok(())
}
