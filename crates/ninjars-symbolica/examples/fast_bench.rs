//! Pure-reduction benchmark of the binary64 fast engine on
//! "ninjars-bench-case-v1" JSON inputs.
//!
//! For every case and batch width the tensor-numerator outputs and the master
//! values (oneloopmaster's native generated Rust, `OneLoopNativeMasters`) are
//! recorded once, then three phases are timed:
//!
//! - `reduction`: `FastAmplitude::evaluate` with `ReplayNumerator` and
//!   `ReplayMasters` (pure reduction, O(1) providers);
//! - `reduction+tensor`: the live `TensorNumerator` with `ReplayMasters`;
//! - `reduction+tensor+masters` (with `--with-masters`): live numerator and
//!   live native masters.
//!
//! Lanes cycle over the case's phase-space points; when the width is not a
//! multiple of their number, every timed run evaluates one batch per rotation
//! of the point order (at width 1: every point in turn, like original Ninja's
//! `ninja_bench.cc`). Each phase reports the
//! median (and range) over at least `--blocks` blocks of at least
//! `--min-block-ms` milliseconds, in nanoseconds per phase-space point, and
//! the heap allocations counted inside the timed pure-reduction blocks. Pin
//! the process with `taskset -c <cpu>` and build with `--profile perf`:
//!
//! ```text
//! cargo run --profile perf -p ninjars-symbolica --features native-masters \
//!     --example fast_bench -- [--widths 1,8,32,128] [--blocks 7] \
//!     [--min-block-ms 20] [--with-masters] [--only-reduction] [--cut-stop k] \
//!     [--fake-masters] [--dump] cases/*.json
//! ```
//!
//! `--fake-masters` records unit master values instead of native ones (fast
//! setup for profiling the pure reduction; results are then meaningless).
//!
//! ```text
//! ```
//!
//! `--dump` instead prints one JSON line per point (all points in one batch)
//! with the Laurent coefficients, cut-constructible and rational parts.

use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
    time::Instant,
};

use ninjars::fast::{
    C64, FastAmplitude, FastInput, FastMaster, FastMasters, FastNumerator, FastOptions, FastResult,
    FastStatus, RMom, RecordingMasters, RecordingNumerator, TensorNumerator, tensor_layout::ncoeff,
};
use ninjars_symbolica::OneLoopNativeMasters;
use serde_json::{Value, json};

struct CountingAllocator;
static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

// Only this benchmark executable wraps allocation; every request is forwarded
// unchanged to System. The library crates forbid unsafe code.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.realloc(pointer, layout, size) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

type Error = Box<dyn std::error::Error>;

struct Point {
    id: String,
    shifts: Vec<RMom>,
    masses: Vec<C64>,
    mu2: f64,
    smat: Option<Vec<f64>>,
    tensor: Vec<C64>,
}

struct Case {
    name: String,
    n: usize,
    rank: usize,
    points: Vec<Point>,
}

fn number(value: &Value) -> Result<f64, Error> {
    value.as_f64().ok_or_else(|| "expected a number".into())
}

fn complex(value: &Value) -> Result<C64, Error> {
    Ok(C64::new(number(&value[0])?, number(&value[1])?))
}

fn load(path: &str) -> Result<Case, Error> {
    let value: Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    if value["format"] != "ninjars-bench-case-v1" {
        return Err(format!("{path}: not a ninjars-bench-case-v1 file").into());
    }
    let n = value["n"].as_u64().ok_or("n")? as usize;
    let rank = value["rank"].as_u64().ok_or("rank")? as usize;
    let mut points = Vec::new();
    for lane in value["lanes"].as_array().ok_or("lanes")? {
        let shifts = lane["shifts"]
            .as_array()
            .ok_or("shifts")?
            .iter()
            .map(|p| -> Result<RMom, Error> {
                Ok([
                    number(&p[0])?,
                    number(&p[1])?,
                    number(&p[2])?,
                    number(&p[3])?,
                ])
            })
            .collect::<Result<Vec<_>, _>>()?;
        let masses = lane["masses_squared"]
            .as_array()
            .ok_or("masses")?
            .iter()
            .map(complex)
            .collect::<Result<Vec<_>, _>>()?;
        let smat = match &lane["s_matrix"] {
            Value::Null => None,
            rows => Some(
                rows.as_array()
                    .ok_or("s_matrix")?
                    .iter()
                    .flat_map(|row| row.as_array().cloned().unwrap_or_default())
                    .map(|x| number(&x))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
        };
        let tensor = lane["tensor"]
            .as_array()
            .ok_or("tensor")?
            .iter()
            .map(complex)
            .collect::<Result<Vec<_>, _>>()?;
        if shifts.len() != n
            || masses.len() != n
            || tensor.len() != ncoeff(rank)
            || smat.as_ref().is_some_and(|s| s.len() != n * n)
        {
            return Err(format!("{path}: inconsistent lane shapes").into());
        }
        points.push(Point {
            id: lane["id"].as_str().unwrap_or("").to_string(),
            shifts,
            masses,
            mu2: number(&lane["mu_squared"])?,
            smat,
            tensor,
        });
    }
    Ok(Case {
        name: value["name"].as_str().unwrap_or(path).to_string(),
        n,
        rank,
        points,
    })
}

/// Lane-major inputs of a batch whose lanes cycle over the case points.
struct Batch {
    lanes: usize,
    shifts: Vec<RMom>,
    masses: Vec<C64>,
    scales: Vec<f64>,
    smat: Option<Vec<f64>>,
    tensor: TensorNumerator,
}

impl Batch {
    /// Lane `i` holds point `(offset + i) % points`.
    fn new(case: &Case, lanes: usize, offset: usize) -> Self {
        let n = case.n;
        let mut tensor = TensorNumerator::new(case.rank, lanes);
        let mut batch = Self {
            lanes,
            shifts: Vec::new(),
            masses: Vec::new(),
            scales: Vec::new(),
            smat: case.points.iter().any(|p| p.smat.is_some()).then(Vec::new),
            tensor: TensorNumerator::new(case.rank, 0),
        };
        for lane in 0..lanes {
            let point = &case.points[(offset + lane) % case.points.len()];
            batch.shifts.extend_from_slice(&point.shifts);
            batch.masses.extend_from_slice(&point.masses);
            batch.scales.push(point.mu2);
            if let Some(matrix) = &mut batch.smat {
                match &point.smat {
                    Some(s) => matrix.extend_from_slice(s),
                    None => {
                        for i in 0..n {
                            for j in 0..n {
                                let d: Vec<f64> = (0..4)
                                    .map(|k| point.shifts[i][k] - point.shifts[j][k])
                                    .collect();
                                matrix.push(d[0] * d[0] - d[1] * d[1] - d[2] * d[2] - d[3] * d[3]);
                            }
                        }
                    }
                }
            }
            tensor
                .lane_coefficients_mut(lane)
                .copy_from_slice(&point.tensor);
        }
        batch.tensor = tensor;
        batch
    }

    fn input(&self) -> FastInput<'_> {
        FastInput {
            lanes: self.lanes,
            shifts: &self.shifts,
            masses_squared: &self.masses,
            scales_squared: &self.scales,
            s_matrix: self.smat.as_deref(),
        }
    }
}

struct Timing {
    median: f64,
    min: f64,
    max: f64,
    blocks: usize,
    iterations: usize,
    allocations: usize,
}

/// Median nanoseconds per phase-space point over `blocks` blocks of at least
/// `min_block` seconds each.
fn time(
    lanes: usize,
    blocks: usize,
    min_block: f64,
    mut run: impl FnMut() -> Result<(), Error>,
) -> Result<Timing, Error> {
    // Warm up and calibrate.
    run()?;
    let mut iterations = 1usize;
    loop {
        let start = Instant::now();
        for _ in 0..iterations {
            run()?;
        }
        if start.elapsed().as_secs_f64() >= min_block {
            break;
        }
        iterations *= 2;
    }
    let mut samples = Vec::with_capacity(blocks);
    ALLOCATIONS.store(0, Ordering::Relaxed);
    for _ in 0..blocks {
        COUNTING.store(true, Ordering::Relaxed);
        let start = Instant::now();
        for _ in 0..iterations {
            run()?;
        }
        let elapsed = start.elapsed().as_secs_f64();
        COUNTING.store(false, Ordering::Relaxed);
        samples.push(elapsed * 1e9 / (iterations * lanes) as f64);
    }
    let allocations = ALLOCATIONS.load(Ordering::Relaxed);
    samples.sort_by(f64::total_cmp);
    Ok(Timing {
        median: samples[samples.len() / 2],
        min: samples[0],
        max: samples[samples.len() - 1],
        blocks,
        iterations,
        allocations,
    })
}

fn evaluate<N: FastNumerator + ?Sized, M: FastMasters + ?Sized>(
    amplitude: &mut FastAmplitude,
    batch: &Batch,
    numerator: &mut N,
    masters: &mut M,
    results: &mut [FastResult],
    statuses: &mut [FastStatus],
) -> Result<(), Error> {
    amplitude.evaluate(&batch.input(), numerator, masters, results, statuses)?;
    Ok(())
}

/// Identical statuses, and identical results on successful lanes (failed
/// lanes keep whatever result they held before the call).
fn same(
    results: &[FastResult],
    statuses: &[FastStatus],
    reference: &(Vec<FastResult>, Vec<FastStatus>),
) -> bool {
    statuses == reference.1.as_slice()
        && results
            .iter()
            .zip(&reference.0)
            .zip(statuses)
            .all(|((a, b), s)| *s != FastStatus::Success || a == b)
}

fn c(z: C64) -> Value {
    json!([z.re, z.im])
}

/// Live native masters, or (`--fake-masters`, profiling only) unit values.
enum MasterSource {
    Native(Box<OneLoopNativeMasters>),
    Fake,
}

impl FastMasters for MasterSource {
    fn evaluate(
        &mut self,
        family: FastMaster,
        args: &[C64],
        out: &mut [[C64; 3]],
        ok: &mut [bool],
    ) -> ninjars::Result<()> {
        match self {
            Self::Native(masters) => masters.evaluate(family, args, out, ok),
            Self::Fake => {
                out.fill([C64::ONE, C64::ONE, C64::ZERO]);
                ok.fill(true);
                Ok(())
            }
        }
    }
}

/// Records the arguments and values of every master row.
struct ArgumentRecorder<M> {
    inner: M,
    rows: Vec<(FastMaster, Vec<C64>, Vec<[C64; 3]>)>,
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
        let width = family.outputs();
        for (row, values) in args
            .chunks_exact(family.arity())
            .zip(out.chunks_exact(width))
        {
            self.rows.push((family, row.to_vec(), values.to_vec()));
        }
        Ok(())
    }
}

/// Evaluate every point separately (width 1) and print its results together
/// with the master rows used, in the cross-check table layout.
fn dump(case: &Case) -> Result<(), Error> {
    let mut amplitude = FastAmplitude::new(case.n, case.rank, FastOptions::default())?;
    let mut masters = ArgumentRecorder {
        inner: OneLoopNativeMasters::new()?,
        rows: Vec::new(),
    };
    for point in &case.points {
        let single = Case {
            name: case.name.clone(),
            n: case.n,
            rank: case.rank,
            points: vec![Point {
                id: point.id.clone(),
                shifts: point.shifts.clone(),
                masses: point.masses.clone(),
                mu2: point.mu2,
                smat: point.smat.clone(),
                tensor: point.tensor.clone(),
            }],
        };
        let mut batch = Batch::new(&single, 1, 0);
        let mut results = vec![FastResult::default(); 1];
        let mut statuses = vec![FastStatus::NotEvaluated; 1];
        let mut tensor = std::mem::replace(&mut batch.tensor, TensorNumerator::new(case.rank, 0));
        masters.rows.clear();
        evaluate(
            &mut amplitude,
            &batch,
            &mut tensor,
            &mut masters,
            &mut results,
            &mut statuses,
        )?;
        let mut table = serde_json::Map::new();
        for (family, key) in [
            (FastMaster::D0, "D0"),
            (FastMaster::C0, "C0"),
            (FastMaster::BubbleRank2, "B"),
            (FastMaster::A0, "A0"),
        ] {
            let rows: Vec<Value> = masters
                .rows
                .iter()
                .filter(|(f, _, _)| {
                    *f == family
                        || (family == FastMaster::BubbleRank2 && *f == FastMaster::BubbleRank3)
                })
                .map(|(_, args, values)| {
                    let mut flat: Vec<Value> = args.iter().map(|&z| c(z)).collect();
                    let mut values = values.clone();
                    if family == FastMaster::BubbleRank2 {
                        values.resize(4, [C64::ZERO; 3]);
                    }
                    flat.extend(values.iter().flatten().map(|&z| c(z)));
                    Value::Array(flat)
                })
                .collect();
            table.insert(key.into(), Value::Array(rows));
        }
        let r = &results[0];
        println!(
            "{}",
            json!({"case": case.name, "id": point.id, "status": format!("{:?}", statuses[0]),
                "laurent": r.laurent.map(c), "cut_constructible": c(r.cut_constructible),
                "rational": c(r.rational), "masters": table})
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn bench(
    case: &Case,
    widths: &[usize],
    blocks: usize,
    min_block: f64,
    options: FastOptions,
    with_masters: bool,
    only_reduction: bool,
    fake_masters: bool,
) -> Result<(), Error> {
    for &width in widths {
        // Like original Ninja's driver, every timed run cycles over all case
        // points: unless the width is a multiple of their number, one run
        // evaluates one batch per rotation of the point order.
        let points = case.points.len();
        let rotations = if width % points == 0 { 1 } else { points };
        let mut batches: Vec<Batch> = (0..rotations)
            .map(|offset| Batch::new(case, width, offset))
            .collect();
        let mut amplitude = FastAmplitude::new(case.n, case.rank, options)?;
        let mut results = vec![FastResult::default(); width];
        let mut statuses = vec![FastStatus::NotEvaluated; width];
        let mut recorded_numerators: Vec<_> = batches
            .iter_mut()
            .map(|b| {
                RecordingNumerator::new(std::mem::replace(
                    &mut b.tensor,
                    TensorNumerator::new(case.rank, 0),
                ))
            })
            .collect();
        let mut recorded_masters = RecordingMasters::new(if fake_masters {
            MasterSource::Fake
        } else {
            MasterSource::Native(Box::new(OneLoopNativeMasters::new()?))
        });
        let mut reference = Vec::new();
        let mut succeeded = 0;
        for (batch, numerator) in batches.iter().zip(&mut recorded_numerators) {
            evaluate(
                &mut amplitude,
                batch,
                numerator,
                &mut recorded_masters,
                &mut results,
                &mut statuses,
            )?;
            reference.push((results.clone(), statuses.clone()));
            succeeded += statuses
                .iter()
                .filter(|s| **s == FastStatus::Success)
                .count();
        }
        let mut replay_numerators: Vec<_> =
            recorded_numerators.iter().map(|r| r.replay()).collect();
        let mut replay_masters = recorded_masters.replay();
        let calls: usize = recorded_numerators.iter().map(|r| r.calls()).sum();
        let master_calls = recorded_masters.calls().len();
        let mut tensors: Vec<TensorNumerator> = recorded_numerators
            .into_iter()
            .map(RecordingNumerator::into_inner)
            .collect();
        let mut native = recorded_masters.into_inner();
        // Replays must reproduce the recorded evaluation bitwise.
        for (j, batch) in batches.iter().enumerate() {
            evaluate(
                &mut amplitude,
                batch,
                &mut replay_numerators[j],
                &mut replay_masters,
                &mut results,
                &mut statuses,
            )?;
            if !same(&results, &statuses, &reference[j]) {
                return Err(format!("{}: replay differs from recording", case.name).into());
            }
        }
        let lanes = width * rotations;
        let mut phases: Vec<(&str, Timing)> = vec![(
            "reduction",
            time(lanes, blocks, min_block, || {
                for (j, batch) in batches.iter().enumerate() {
                    evaluate(
                        &mut amplitude,
                        batch,
                        &mut replay_numerators[j],
                        &mut replay_masters,
                        &mut results,
                        &mut statuses,
                    )?;
                }
                Ok(())
            })?,
        )];
        if !only_reduction {
            phases.push((
                "reduction+tensor",
                time(lanes, blocks, min_block, || {
                    for (j, batch) in batches.iter().enumerate() {
                        evaluate(
                            &mut amplitude,
                            batch,
                            &mut tensors[j],
                            &mut replay_masters,
                            &mut results,
                            &mut statuses,
                        )?;
                    }
                    Ok(())
                })?,
            ));
        }
        if with_masters {
            phases.push((
                "reduction+tensor+masters",
                time(lanes, blocks, min_block, || {
                    for (j, batch) in batches.iter().enumerate() {
                        evaluate(
                            &mut amplitude,
                            batch,
                            &mut tensors[j],
                            &mut native,
                            &mut results,
                            &mut statuses,
                        )?;
                    }
                    Ok(())
                })?,
            ));
        }
        if !same(&results, &statuses, &reference[rotations - 1]) {
            return Err(format!("{}: timed results differ from recording", case.name).into());
        }
        for (phase, t) in phases {
            println!(
                "{}",
                json!({"case": case.name, "n": case.n, "rank": case.rank, "width": width,
                    "cut_stop": options.cut_stop,
                    "phase": phase, "median_ns_per_point": t.median, "min_ns_per_point": t.min,
                    "max_ns_per_point": t.max, "blocks": t.blocks, "iterations_per_block": t.iterations,
                    "allocations_in_timed_blocks": t.allocations, "successful_lanes": succeeded,
                    "rotations": rotations,
                    "numerator_calls": calls, "master_calls": master_calls})
            );
        }
    }
    Ok(())
}

fn main() -> Result<(), Error> {
    let mut widths = vec![1, 8, 32, 128];
    let mut blocks = 7;
    let mut min_block = 0.02;
    let mut dump_mode = false;
    let mut with_masters = false;
    let mut only_reduction = false;
    let mut fake_masters = false;
    let mut options = FastOptions::default();
    let mut files = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--widths" => {
                widths = args
                    .next()
                    .ok_or("--widths needs a list")?
                    .split(',')
                    .map(str::parse)
                    .collect::<Result<_, _>>()?
            }
            "--blocks" => blocks = args.next().ok_or("--blocks needs a value")?.parse()?,
            "--min-block-ms" => {
                min_block = args
                    .next()
                    .ok_or("--min-block-ms needs a value")?
                    .parse::<f64>()?
                    / 1e3
            }
            "--dump" => dump_mode = true,
            "--with-masters" => with_masters = true,
            "--only-reduction" => only_reduction = true,
            "--fake-masters" => fake_masters = true,
            "--cut-stop" => {
                options.cut_stop = args.next().ok_or("--cut-stop needs a value")?.parse()?
            }
            _ => files.push(arg),
        }
    }
    if blocks < 1 || widths.contains(&0) {
        return Err("blocks and widths must be positive".into());
    }
    for file in &files {
        let case = load(file)?;
        if dump_mode {
            dump(&case)?;
        } else {
            bench(
                &case,
                &widths,
                blocks,
                min_block,
                options,
                with_masters,
                only_reduction,
                fake_masters,
            )?;
        }
    }
    Ok(())
}
