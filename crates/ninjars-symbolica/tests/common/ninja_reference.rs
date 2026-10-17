//! Cross-check of the binary64 fast engine against original Ninja with
//! identical master integrals.
//!
//! Inputs are a "ninjars-bench-case-v1" case and original Ninja's results for
//! it as written by `validation/benchmark/ninja_bench.cc` (per lane: Laurent
//! coefficients, cut-constructible and rational parts, and the exact
//! `IntegralLibrary` calls with OneLOop's values, either inline as
//! `master_call_list` or in a separate `ninjars-reference-master-calls-v1`
//! file). The fast engine reduces every lane with `TensorNumerator` and a
//! master provider that answers each request with the OneLOop value of
//! original Ninja's call with the same arguments, so differences come from
//! the reductions alone.
//!
//! Shared by `tests/fast_ninja_reference.rs` and `examples/fast_crosscheck.rs`.

#![allow(dead_code)]

use ninjars::fast::{
    C64, FastAmplitude, FastInput, FastMaster, FastMasters, FastOptions, FastResult, FastStatus,
    RMom, TensorNumerator, tensor_layout::ncoeff,
};
use serde_json::Value;

pub type Error = Box<dyn std::error::Error>;

pub struct Point {
    pub id: String,
    pub shifts: Vec<RMom>,
    pub masses: Vec<C64>,
    pub mu2: f64,
    pub smat: Option<Vec<f64>>,
    pub tensor: Vec<C64>,
}

pub struct Case {
    pub name: String,
    pub n: usize,
    pub rank: usize,
    pub points: Vec<Point>,
}

fn number(value: &Value) -> Result<f64, Error> {
    value.as_f64().ok_or_else(|| "expected a number".into())
}

fn complex(value: &Value) -> Result<C64, Error> {
    Ok(C64::new(number(&value[0])?, number(&value[1])?))
}

pub fn parse_case(text: &str) -> Result<Case, Error> {
    let value: Value = serde_json::from_str(text)?;
    if value["format"] != "ninjars-bench-case-v1" {
        return Err("not a ninjars-bench-case-v1 file".into());
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
            return Err("inconsistent lane shapes".into());
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
        name: value["name"].as_str().unwrap_or("").to_string(),
        n,
        rank,
        points,
    })
}

/// One `IntegralLibrary` call of original Ninja with its OneLOop values,
/// converted to the fast engine's argument and output layout.
#[derive(Clone, Debug)]
pub struct MasterCall {
    pub family: FastMaster,
    pub args: Vec<C64>,
    pub values: Vec<[C64; 3]>,
}

/// Original Ninja's result for one lane.
pub struct NinjaLane {
    pub id: String,
    pub status: i64,
    /// `[eps0, eps_m1, eps_m2, cut_constructible, rational]`.
    pub values: [C64; 5],
    pub calls: Vec<MasterCall>,
}

fn triples(values: &[C64]) -> Vec<[C64; 3]> {
    values.as_chunks::<3>().0.to_vec()
}

fn master_call(call: &Value) -> Result<MasterCall, Error> {
    let kind = call["kind"].as_str().ok_or("kind")?;
    let mu2 = C64::real(number(&call["mu_squared"])?);
    let s = call["s"]
        .as_array()
        .ok_or("s")?
        .iter()
        .map(number)
        .collect::<Result<Vec<_>, _>>()?;
    let m = call["m2"]
        .as_array()
        .ok_or("m2")?
        .iter()
        .map(complex)
        .collect::<Result<Vec<_>, _>>()?;
    let result = call["result"]
        .as_array()
        .ok_or("result")?
        .iter()
        .map(complex)
        .collect::<Result<Vec<_>, _>>()?;
    let real = |x: f64| C64::real(x);
    let (family, args, mut values) = match kind {
        "box_cm" => {
            let mut args: Vec<C64> = s[..6].iter().copied().map(real).collect();
            args.extend_from_slice(&m[..4]);
            (FastMaster::D0, args, triples(&result))
        }
        "triangle_cm" => {
            let mut args: Vec<C64> = s[..3].iter().copied().map(real).collect();
            args.extend_from_slice(&m[..3]);
            (FastMaster::C0, args, triples(&result))
        }
        // Ninja's order is (B11, B1, B0) and (B111, B11, B1, B0); the fast
        // engine's (B0, B1, B11[, B111]).
        "rank2_bubble_cm" => (
            FastMaster::BubbleRank2,
            vec![real(s[0]), m[0], m[1]],
            triples(&result).into_iter().rev().collect(),
        ),
        "rank3_bubble_cm" => (
            FastMaster::BubbleRank3,
            vec![real(s[0]), m[0], m[1]],
            triples(&result).into_iter().rev().collect(),
        ),
        "tadpole_cm" => (FastMaster::A0, vec![m[0]], triples(&result)),
        other => return Err(format!("unsupported master call {other}").into()),
    };
    let mut args = args;
    args.push(mu2);
    values.truncate(family.outputs());
    Ok(MasterCall {
        family,
        args,
        values,
    })
}

/// Parse original Ninja's results. `masters` is the separate
/// `<case>.masters.json` when the results file has no inline call lists.
pub fn parse_ninja(results: &str, masters: Option<&str>) -> Result<Vec<NinjaLane>, Error> {
    let value: Value = serde_json::from_str(results)?;
    let separate: Option<Value> = masters.map(serde_json::from_str).transpose()?;
    let mut out = Vec::new();
    for (index, lane) in value["lanes"].as_array().ok_or("lanes")?.iter().enumerate() {
        let calls = match (&lane["master_call_list"], &separate) {
            (Value::Array(list), _) => list.clone(),
            (_, Some(file)) => file["lanes"][index]["calls"]
                .as_array()
                .ok_or("master calls")?
                .clone(),
            _ => return Err("no master calls".into()),
        };
        let field = |name: &str| complex(&lane[name]);
        out.push(NinjaLane {
            id: lane["id"].as_str().unwrap_or("").to_string(),
            status: lane["return_status"].as_i64().unwrap_or(-1),
            values: [
                field("eps0")?,
                field("eps_m1")?,
                field("eps_m2")?,
                field("cut_constructible")?,
                field("rational")?,
            ],
            calls: calls.iter().map(master_call).collect::<Result<_, _>>()?,
        });
    }
    Ok(out)
}

/// Answers every request with the OneLOop value of original Ninja's call with
/// the same family and arguments (relative agreement 1e-13 on each argument:
/// without an explicit S-matrix both codes round the invariants separately).
pub struct NinjaMasters {
    calls: Vec<MasterCall>,
    pub unmatched: Vec<(FastMaster, Vec<C64>)>,
}

impl NinjaMasters {
    pub fn new(lanes: &[NinjaLane]) -> Self {
        Self {
            calls: lanes.iter().flat_map(|l| l.calls.iter().cloned()).collect(),
            unmatched: Vec::new(),
        }
    }
}

fn same(a: C64, b: C64) -> bool {
    let scale = a.abs().max(b.abs());
    (a - b).abs() <= 1e-13 * scale
}

impl FastMasters for NinjaMasters {
    fn evaluate(
        &mut self,
        family: FastMaster,
        args: &[C64],
        out: &mut [[C64; 3]],
        ok: &mut [bool],
    ) -> ninjars::Result<()> {
        let arity = family.arity();
        let width = family.outputs();
        for (row, a) in args.chunks_exact(arity).enumerate() {
            let found = self
                .calls
                .iter()
                .find(|c| c.family == family && c.args.iter().zip(a).all(|(&x, &y)| same(x, y)));
            match found {
                Some(call) => {
                    out[row * width..(row + 1) * width].copy_from_slice(&call.values);
                    ok[row] = true;
                }
                None => {
                    self.unmatched.push((family, a.to_vec()));
                    ok[row] = false;
                }
            }
        }
        Ok(())
    }
}

/// Lane-major inputs of a batch made of the given case points.
pub struct Batch {
    pub lanes: usize,
    pub shifts: Vec<RMom>,
    pub masses: Vec<C64>,
    pub scales: Vec<f64>,
    pub smat: Option<Vec<f64>>,
    pub tensor: TensorNumerator,
}

impl Batch {
    pub fn new(case: &Case, points: &[usize]) -> Self {
        let n = case.n;
        let lanes = points.len();
        let mut tensor = TensorNumerator::new(case.rank, lanes);
        let mut smat = case.points.iter().any(|p| p.smat.is_some()).then(Vec::new);
        let (mut shifts, mut masses, mut scales) = (Vec::new(), Vec::new(), Vec::new());
        for (lane, &index) in points.iter().enumerate() {
            let point = &case.points[index];
            shifts.extend_from_slice(&point.shifts);
            masses.extend_from_slice(&point.masses);
            scales.push(point.mu2);
            if let Some(matrix) = &mut smat {
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
        Self {
            lanes,
            shifts,
            masses,
            scales,
            smat,
            tensor,
        }
    }

    pub fn input(&self) -> FastInput<'_> {
        FastInput {
            lanes: self.lanes,
            shifts: &self.shifts,
            masses_squared: &self.masses,
            scales_squared: &self.scales,
            s_matrix: self.smat.as_deref(),
        }
    }
}

pub fn values(r: &FastResult) -> [C64; 5] {
    [
        r.laurent[0],
        r.laurent[1],
        r.laurent[2],
        r.cut_constructible,
        r.rational,
    ]
}

/// Per-lane comparison with original Ninja. Errors are relative to the lane's
/// scale, the largest modulus among Ninja's five values (no absolute floor).
#[derive(Clone, Debug)]
pub struct LaneReport {
    pub id: String,
    pub status: FastStatus,
    /// Ninja's return value: 0 SUCCESS, 2 UNSTABLE_KINEMATICS.
    pub ninja_status: i64,
    pub scale: f64,
    /// Max scaled error of the three Laurent coefficients.
    pub laurent: f64,
    pub cut_constructible: f64,
    pub rational: f64,
}

pub struct CaseReport {
    pub name: String,
    pub n: usize,
    pub rank: usize,
    pub lanes: Vec<LaneReport>,
    pub unmatched_masters: usize,
    /// Every batched/permuted/repeated evaluation reproduced the width-1
    /// results bitwise (and the same statuses).
    pub batches_bitwise: bool,
    pub batch_failures: Vec<String>,
}

impl LaneReport {
    /// The fast status corresponds to Ninja's return value (an unstable Ninja
    /// point is `Unstable`; a successful one is `Success`).
    pub fn status_matches(&self) -> bool {
        match self.ninja_status {
            0 => self.status == FastStatus::Success,
            2 => self.status == FastStatus::Unstable,
            _ => false,
        }
    }
}

impl CaseReport {
    pub fn statuses_match(&self) -> bool {
        self.lanes.iter().all(LaneReport::status_matches)
    }
    pub fn max_laurent(&self) -> f64 {
        self.lanes.iter().map(|l| l.laurent).fold(0., f64::max)
    }
    pub fn max_split(&self) -> f64 {
        self.lanes
            .iter()
            .map(|l| l.cut_constructible.max(l.rational))
            .fold(0., f64::max)
    }
}

fn evaluate(
    amplitude: &mut FastAmplitude,
    case: &Case,
    points: &[usize],
    masters: &mut NinjaMasters,
) -> Result<(Vec<FastResult>, Vec<FastStatus>), Error> {
    let mut batch = Batch::new(case, points);
    let mut results = vec![FastResult::default(); points.len()];
    let mut statuses = vec![FastStatus::NotEvaluated; points.len()];
    let mut tensor = std::mem::replace(&mut batch.tensor, TensorNumerator::new(case.rank, 0));
    amplitude.evaluate(
        &batch.input(),
        &mut tensor,
        masters,
        &mut results,
        &mut statuses,
    )?;
    Ok((results, statuses))
}

/// Reduce every lane at width 1, compare with Ninja, then check that batches
/// (natural, reversed, odd and large cyclic widths, interleaved on one reused
/// engine) reproduce the width-1 results bitwise.
pub fn crosscheck(case: &Case, ninja: &[NinjaLane]) -> Result<CaseReport, Error> {
    if ninja.len() != case.points.len() {
        return Err("lane count mismatch".into());
    }
    let mut masters = NinjaMasters::new(ninja);
    let mut amplitude = FastAmplitude::new(case.n, case.rank, FastOptions::default())?;
    let count = case.points.len();
    let mut single = Vec::new();
    let mut lanes = Vec::new();
    for (index, reference) in ninja.iter().enumerate() {
        let (results, statuses) = evaluate(&mut amplitude, case, &[index], &mut masters)?;
        let got = values(&results[0]);
        let want = reference.values;
        let scale = want.iter().map(|z| z.abs()).fold(0., f64::max).max(1e-300);
        // Errors are measured only where both codes produced a result.
        let compared = reference.status == 0 && statuses[0] == FastStatus::Success;
        let err = |k: usize| {
            if compared {
                (got[k] - want[k]).abs() / scale
            } else {
                0.
            }
        };
        lanes.push(LaneReport {
            id: reference.id.clone(),
            status: statuses[0],
            ninja_status: reference.status,
            scale,
            laurent: err(0).max(err(1)).max(err(2)),
            cut_constructible: err(3),
            rational: err(4),
        });
        single.push((results[0], statuses[0]));
    }
    let unmatched_masters = masters.unmatched.len();
    let natural: Vec<usize> = (0..count).collect();
    let reversed: Vec<usize> = (0..count).rev().collect();
    let cyclic = |width: usize, step: usize| -> Vec<usize> {
        (0..width).map(|i| (i * step + i / count) % count).collect()
    };
    let mut batch_failures = Vec::new();
    for points in [
        natural,
        reversed,
        cyclic(37, 2),
        Vec::new(),
        cyclic(128, 1),
        cyclic(5, 3),
        cyclic(1, 1),
    ] {
        let (results, statuses) = evaluate(&mut amplitude, case, &points, &mut masters)?;
        for (lane, &index) in points.iter().enumerate() {
            let (want, want_status) = single[index];
            if statuses[lane] != want_status
                || (want_status == FastStatus::Success && results[lane] != want)
            {
                batch_failures.push(format!(
                    "width {} lane {lane} (point {index}): {:?} {:?} vs width-1 {:?} {:?}",
                    points.len(),
                    statuses[lane],
                    results[lane].laurent,
                    want_status,
                    want.laurent
                ));
            }
        }
    }
    Ok(CaseReport {
        name: case.name.clone(),
        n: case.n,
        rank: case.rank,
        lanes,
        unmatched_masters,
        batches_bitwise: batch_failures.is_empty(),
        batch_failures,
    })
}
