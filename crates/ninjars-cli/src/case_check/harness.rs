//! Engine-independent core of `ninjars case-check`: bench cases, reference
//! files, the precision-generic tensor numerator, both engines and the
//! per-tier tolerance policy.
//!
//! The rank n+2 tests of `ninjars-symbolica` compile this file as well
//! (`#[path = "../../ninjars-cli/src/case_check/harness.rs"]`), so it uses only
//! `ninjars`, `ninjars-symbolica`, `oneloop`, `symbolica` and `serde_json`, and
//! never `crate::` paths.
//!
//! - [`BenchCase`] reads `ninjars-bench-case-v1` (dense tensor) and `-v2`
//!   (explicit `mu^2` blocks, `N = sum_k (mu^2)^k T_k(q)`) files.
//! - [`Reference`] reads `ninjars-reference-collier-v1` files (the rank n+2
//!   suite of `validation/rank-n2`: 40-digit decimals, tier, uncertainty,
//!   condition) and original-Ninja result files
//!   (`ninjars-reference-ninja-results-v1`, `-ninja-bench-v1`).
//! - [`TensorNumerator`]-equivalent numerators for the generic engine in any
//!   precision: [`TensorGenericNumerator`] evaluates the dense tensor and its
//!   Laurent expansions numerically (truncated series in `1/t`), so a dense
//!   rank-9 numerator costs no Symbolica preparation.
//! - [`GenericEngine`] (any `NativeFloat` carrier) and [`run_fast`] (binary64
//!   fast engine with `TensorNumerator`/`MuTensorNumerator` and native masters,
//!   batched exactly as `case-bench`).
//! - [`Policy`] turns a scaled difference into a [`Verdict`].

#![allow(dead_code)]

use std::{
    collections::{BTreeMap, HashMap},
    error::Error,
    fs,
    path::{Path, PathBuf},
};

use ninjars::{
    AmplitudeInput, BatchShape, Complex, DoubleFloat, EvaluationStatus, ExpansionKind,
    ExpansionRequest, Float, FloatLike, FourMomentum, MasterProvider, Numerator, NumeratorRequest,
    Real, SingleFloat,
    fast::{
        C64, FastAmplitude, FastInput, FastMasters, FastOptions, FastResult, FastStatus,
        MuTensorNumerator, RMom, TensorNumerator,
        tensor_layout::{monomial_index, monomials, mu_block_offset, ncoeff, ncoeff_mu},
    },
};
use ninjars_symbolica::{
    OneLoopNativeMasters, SymbolicaAmplitudeKernels, SymbolicaCuts, SymbolicaHighCuts,
    SymbolicaMasters,
};
use serde_json::{Value, json};
use symbolica::{
    domains::{float::RealLike, integer::Integer, rational::Rational},
    evaluate::EvaluationDomain,
};

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn number(value: &Value) -> Result<f64> {
    value
        .as_f64()
        .ok_or_else(|| format!("expected a number, found {value}").into())
}

fn complex(value: &Value) -> Result<C64> {
    Ok(C64::new(number(&value[0])?, number(&value[1])?))
}

// ---------------------------------------------------------------------------
// Bench cases.

/// One phase-space point of a bench case.
#[derive(Clone, Debug)]
pub struct BenchLane {
    pub id: String,
    pub shifts: Vec<RMom>,
    pub masses: Vec<C64>,
    /// The renormalization scale `mu_R^2` (not the integrand `mu^2`).
    pub mu2: f64,
    /// Row-major `n x n` invariants, when the case supplies them.
    pub s_matrix: Option<Vec<f64>>,
    /// `T_0 ++ T_1 ++ ... ++ T_K` (`K = mu_powers`), `ncoeff_mu(rank, K)` entries.
    pub tensor: Vec<C64>,
}

/// A `ninjars-bench-case-v1` or `-v2` file.
#[derive(Clone, Debug)]
pub struct BenchCase {
    pub name: String,
    pub path: PathBuf,
    pub n: usize,
    pub rank: usize,
    pub mu_powers: usize,
    pub lanes: Vec<BenchLane>,
}

impl BenchCase {
    pub fn load(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::parse(&text, path).map_err(|e| format!("{}: {e}", path.display()).into())
    }

    pub fn parse(text: &str, path: &Path) -> Result<Self> {
        let value: Value = serde_json::from_str(text)?;
        let mu_powers = match value["format"].as_str() {
            Some("ninjars-bench-case-v1") => 0,
            Some("ninjars-bench-case-v2") => {
                value["mu_powers"].as_u64().ok_or("mu_powers")? as usize
            }
            _ => return Err("not a ninjars-bench-case-v1/v2 file".into()),
        };
        let n = value["n"].as_u64().ok_or("n")? as usize;
        let rank = value["rank"].as_u64().ok_or("rank")? as usize;
        if n == 0 || 2 * mu_powers > rank {
            return Err("invalid n, rank or mu_powers".into());
        }
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
            if shifts.len() != n
                || masses.len() != n
                || tensor.len() != ncoeff_mu(rank, mu_powers)
                || s_matrix.as_ref().is_some_and(|s| s.len() != n * n)
            {
                return Err("malformed lane".into());
            }
            lanes.push(BenchLane {
                id: lane["id"].as_str().unwrap_or("").to_string(),
                shifts,
                masses,
                mu2: number(&lane["mu_squared"])?,
                s_matrix,
                tensor,
            });
        }
        if lanes.is_empty() {
            return Err("no lanes".into());
        }
        Ok(Self {
            name: value["name"].as_str().unwrap_or("").to_string(),
            path: path.to_path_buf(),
            n,
            rank,
            mu_powers,
            lanes,
        })
    }

    /// Coefficients of one lane re-laid out for a declared rank `declared >=
    /// rank` (`mu_powers` unchanged): every block `T_k` of rank `rank - 2k`
    /// becomes the leading part of a rank `declared - 2k` block whose new
    /// top coefficients are zero.
    pub fn padded_tensor(&self, lane: usize, declared: usize) -> Vec<C64> {
        assert!(declared >= self.rank, "declared rank below the case rank");
        let source = &self.lanes[lane].tensor;
        let mut out = vec![C64::ZERO; ncoeff_mu(declared, self.mu_powers)];
        for k in 0..=self.mu_powers {
            let from = mu_block_offset(self.rank, k);
            let to = mu_block_offset(declared, k);
            let size = ncoeff(self.rank - 2 * k);
            out[to..to + size].copy_from_slice(&source[from..from + size]);
        }
        out
    }
}

// ---------------------------------------------------------------------------
// References.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReferenceKind {
    /// `ninjars-reference-collier-v1`: high-precision reference with tier,
    /// uncertainty and condition (`validation/rank-n2/README.md`).
    HighPrecision,
    /// Original Ninja's binary64 result (`validation/benchmark/ninja_bench.cc`).
    Ninja,
}

#[derive(Clone, Debug)]
pub struct ReferenceLane {
    pub id: String,
    /// `[eps0, eps_m1, eps_m2]` in binary64.
    pub values: [C64; 3],
    /// The same as decimal strings `[re, im]`, when the file has them.
    pub decimal: Option<[[String; 2]; 3]>,
    /// `max_k |ref_k|`, or 1 when the value vanishes exactly.
    pub scale: f64,
    pub tier: Option<String>,
    pub uncertainty: Option<f64>,
    /// COLLIER's contraction condition (sum of absolute terms over the result).
    pub condition: Option<f64>,
    /// The `input_ambiguity_parts` entry that sets `uncertainty`, if any.
    pub uncertainty_origin: Option<String>,
    /// Original Ninja's return status (0 success, 2 unstable).
    pub ninja_status: Option<i64>,
    /// Original Ninja's scaled error against this (high-precision) reference.
    pub ninja_error: Option<f64>,
    /// `rank - n` of the case.
    pub excess: Option<i64>,
}

#[derive(Clone, Debug)]
pub struct Reference {
    pub path: PathBuf,
    pub format: String,
    pub kind: ReferenceKind,
    pub lanes: Vec<ReferenceLane>,
}

fn decimal_pair(value: &Value) -> Option<[String; 2]> {
    Some([
        value.get(0)?.as_str()?.to_string(),
        value.get(1)?.as_str()?.to_string(),
    ])
}

impl Reference {
    /// `<dir>/<case>.json`, if it exists.
    pub fn find(dir: &Path, case: &str) -> Result<Option<Self>> {
        let path = dir.join(format!("{case}.json"));
        if !path.is_file() {
            return Ok(None);
        }
        Self::load(&path).map(Some)
    }

    pub fn load(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let value: Value = serde_json::from_str(&text)?;
        let format = value["format"].as_str().unwrap_or("").to_string();
        let kind = match format.as_str() {
            "ninjars-reference-collier-v1" => ReferenceKind::HighPrecision,
            "ninjars-reference-ninja-results-v1" | "ninjars-reference-ninja-bench-v1" => {
                ReferenceKind::Ninja
            }
            other => {
                return Err(
                    format!("{}: unknown reference format {other:?}", path.display()).into(),
                );
            }
        };
        let excess = match (value["n"].as_i64(), value["rank"].as_i64()) {
            (Some(n), Some(rank)) => Some(rank - n),
            _ => None,
        };
        let mut lanes = Vec::new();
        for lane in value["lanes"].as_array().ok_or("reference lanes")? {
            let id = lane["id"].as_str().unwrap_or("").to_string();
            // Lanes without any valid evaluation carry an "error" field only.
            let Ok(values) = (|| -> Result<[C64; 3]> {
                Ok([
                    complex(&lane["eps0"])?,
                    complex(&lane["eps_m1"])?,
                    complex(&lane["eps_m2"])?,
                ])
            })() else {
                continue;
            };
            let decimal = lane.get("laurent_decimal").and_then(|d| {
                Some([
                    decimal_pair(&d["eps0"])?,
                    decimal_pair(&d["eps_m1"])?,
                    decimal_pair(&d["eps_m2"])?,
                ])
            });
            let computed = values.iter().fold(0_f64, |m, z| m.max(z.abs()));
            let scale = lane["scale"]
                .as_f64()
                .unwrap_or(if computed > 0. { computed } else { 1. });
            let condition = lane["absum_over_scale"].as_f64().or_else(|| {
                lane["condition"]
                    .as_object()
                    .map(|c| c.values().filter_map(Value::as_f64).fold(0., f64::max))
            });
            let uncertainty = lane["uncertainty"].as_f64();
            let uncertainty_origin = lane["input_ambiguity_parts"].as_object().and_then(|parts| {
                let (name, largest) = parts
                    .iter()
                    .filter_map(|(k, v)| Some((k.clone(), v.as_f64()?)))
                    .max_by(|a, b| a.1.total_cmp(&b.1))?;
                (Some(largest) == uncertainty).then_some(name)
            });
            lanes.push(ReferenceLane {
                id,
                values,
                decimal,
                scale,
                tier: lane["tier"].as_str().map(str::to_string),
                uncertainty,
                condition,
                uncertainty_origin,
                ninja_status: lane["return_status"].as_i64(),
                ninja_error: lane["ninja"].as_f64(),
                excess,
            });
        }
        Ok(Self {
            path: path.to_path_buf(),
            format,
            kind,
            lanes,
        })
    }

    pub fn lane(&self, id: &str) -> Option<&ReferenceLane> {
        self.lanes.iter().find(|lane| lane.id == id)
    }
}

/// Exact value of a decimal string such as `-1.25E-9` or `0.000012`.
pub fn parse_decimal(text: &str) -> Result<Rational> {
    let text = text.trim();
    let (mantissa, exponent) = match text.find(['e', 'E']) {
        Some(at) => (&text[..at], text[at + 1..].parse::<i64>()?),
        None => (text, 0),
    };
    let (negative, mantissa) = match mantissa.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, mantissa.strip_prefix('+').unwrap_or(mantissa)),
    };
    let (whole, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let digits = format!("{whole}{fraction}");
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return Err(format!("not a decimal number: {text:?}").into());
    }
    let mut numerator: Integer = digits.parse()?;
    if negative {
        numerator = -numerator;
    }
    let power = exponent - fraction.len() as i64;
    let ten = Integer::from(10);
    Ok(if power >= 0 {
        Rational::from(numerator * ten.pow(power as u64))
    } else {
        Rational::from((numerator, ten.pow((-power) as u64)))
    })
}

/// The reference value in the carrier's precision: the decimals when present,
/// otherwise the binary64 values (exactly).
pub fn reference_values<T: SingleFloat>(
    lane: &ReferenceLane,
    prototype: &T,
) -> Result<[Complex<T>; 3]> {
    let exact = |x: f64| -> Result<T> {
        Ok(prototype.from_rational(&Rational::try_from(x).map_err(|_| "non-finite reference")?))
    };
    let mut out = Vec::with_capacity(3);
    for k in 0..3 {
        out.push(match &lane.decimal {
            Some(d) => Complex::new(
                prototype.from_rational(&parse_decimal(&d[k][0])?),
                prototype.from_rational(&parse_decimal(&d[k][1])?),
            ),
            None => Complex::new(exact(lane.values[k].re)?, exact(lane.values[k].im)?),
        });
    }
    Ok(out.try_into().unwrap_or_else(|_| unreachable!()))
}

/// `max_k |x_k - ref_k| / scale` over the three Laurent coefficients,
/// evaluated in the carrier's precision.
pub fn scaled_difference<T: SingleFloat + RealLike>(
    values: &[Complex<T>],
    reference: &[Complex<T>; 3],
    scale: f64,
) -> f64 {
    (0..3)
        .map(|k| {
            let re = (values[k].re.clone() - &reference[k].re).to_f64();
            let im = (values[k].im.clone() - &reference[k].im).to_f64();
            re.hypot(im)
        })
        .fold(0., f64::max)
        / scale
}

/// The same in binary64.
pub fn scaled_difference_f64(values: &[C64; 3], reference: &[C64; 3], scale: f64) -> f64 {
    (0..3)
        .map(|k| (values[k] - reference[k]).abs())
        .fold(0., f64::max)
        / scale
}

// ---------------------------------------------------------------------------
// Tolerances.

/// Working precision of a run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    F64,
    Double,
    Float(u32),
}

impl Precision {
    /// `f64`, `double` (DoubleFloat) or `float:<bits>`.
    pub fn parse(text: &str) -> Result<Self> {
        match text {
            "f64" | "binary64" => Ok(Self::F64),
            "double" | "doublefloat" | "dd" => Ok(Self::Double),
            other => match other.strip_prefix("float:") {
                Some(bits) => Ok(Self::Float(bits.parse()?)),
                None => {
                    Err(format!("unknown precision {other:?} (f64, double, float:<bits>)").into())
                }
            },
        }
    }

    pub fn label(self) -> String {
        match self {
            Self::F64 => "f64".into(),
            Self::Double => "double".into(),
            Self::Float(bits) => format!("float:{bits}"),
        }
    }

    /// Unit roundoff of the carrier.
    pub fn epsilon(self) -> f64 {
        match self {
            Self::F64 => f64::EPSILON / 2.,
            Self::Double => 2_f64.powi(-104),
            Self::Float(bits) => 2_f64.powi(-(bits as i32)),
        }
    }

    /// Arbitrary precision of at least 160 bits: the V4 tolerances apply.
    pub fn is_high(self) -> bool {
        matches!(self, Self::Float(bits) if bits >= 160)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Verdict {
    Pass,
    Fail,
    /// Compared and reported, but never a failure (tier C, ill-conditioned
    /// binary64 lanes, non-conservative uncertainties, unstable Ninja lanes).
    ReportOnly,
    /// Listed in the known-failure file and still failing.
    KnownFailure,
    /// No reference lane.
    Unchecked,
    /// The engine could not run the case.
    Error,
}

impl Verdict {
    pub fn label(self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::Fail => "fail",
            Self::ReportOnly => "report-only",
            Self::KnownFailure => "known-failure",
            Self::Unchecked => "unchecked",
            Self::Error => "error",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Judgement {
    pub verdict: Verdict,
    pub tolerance: Option<f64>,
    pub within_tolerance: Option<bool>,
    pub reason: Option<String>,
}

/// Uncertainty origins of `assemble.py` that come from the generic engine's
/// own rank <= n+1 twins. They estimate the input ambiguity of snapped
/// S-matrices and are not a conservative bound (S5a verifier).
pub const GENERIC_DERIVED_ORIGINS: [&str; 2] =
    ["generic_engine_extrapolated", "generic_engine_on_twins"];

/// Which engine produced a value: the binary64 error budget differs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Engine {
    Fast,
    Generic,
}

/// The kinematics of a lane id: the rank and `mu^2` tags removed
/// (`rn2-int-real-n4-r6-mu2-lane1` -> `rn2-int-real-n4-lane1`), so that the
/// rank twins of `validation/rank-n2` share it (`assemble.py`'s `kin_key`).
pub fn kinematics_key(id: &str) -> String {
    let Some(at) = id.rfind("-lane") else {
        return id.to_string();
    };
    let (head, tail) = id.split_at(at);
    let head = head
        .strip_suffix("-mu2")
        .or_else(|| head.strip_suffix("-muall"))
        .unwrap_or(head);
    match head.rfind("-r") {
        Some(pos)
            if pos + 2 < head.len() && head[pos + 2..].bytes().all(|b| b.is_ascii_digit()) =>
        {
            format!("{}{tail}", &head[..pos])
        }
        _ => id.to_string(),
    }
}

/// The tolerance policy (`validation/rank-n2/README.md`, design section 7).
///
/// High-precision references (`ninjars-reference-collier-v1`):
/// - Float of at least 160 bits (V4): tiers A and B pass at
///   `max(3 unc, 1e-30)`; tier-C lanes and tier-B lanes whose uncertainty
///   comes from the generic engine's own twins are report-only.
/// - Binary64 and other fixed precisions (V4f):
///   `3 unc' + eps/eps_64 max(binary64_factor max(1, cond), A e_ninja)`, where
///   `e_ninja` is original Ninja's binary64 error on the lane (the reference's
///   `ninja` field) or, without one, on its rank twins (`--twins`: the largest
///   twin error at the lane's rank excess, else the error at the two highest
///   twin excesses extrapolated geometrically), and `A` is
///   `fast_ninja_factor` or `generic_ninja_factor`. COLLIER's contraction
///   condition `cond` alone does not predict the reduction's binary64 loss
///   (calibration on the controls: up to 1e5 x `1e-14 cond` at n = 6), while
///   the fast engine stays within 12x Ninja's error on all 505 control lanes
///   with a Ninja result (the generic engine within 575x). `unc' = 100 unc`
///   for generic-engine-derived uncertainties. Tier C and `cond > 1e6` are
///   report-only.
///
/// Original-Ninja references: `ninja_tolerance` (scaled); lanes Ninja flagged
/// unstable are report-only. Lanes listed in `known_failures` that fail are
/// `known-failure`.
#[derive(Clone, Debug)]
pub struct Policy {
    pub binary64_factor: f64,
    pub fast_ninja_factor: f64,
    pub generic_ninja_factor: f64,
    pub ill_conditioned: f64,
    pub high_floor: f64,
    pub uncertainty_factor: f64,
    pub generic_derived_factor: f64,
    /// Largest binary64 budget taken from original Ninja's errors.
    pub ninja_budget_cap: f64,
    pub ninja_tolerance: f64,
    /// Lane id -> reason.
    pub known_failures: BTreeMap<String, String>,
    /// Kinematics key -> rank excess -> largest original-Ninja error.
    pub twins: BTreeMap<String, BTreeMap<i64, f64>>,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            binary64_factor: 1e-14,
            fast_ninja_factor: 30.,
            generic_ninja_factor: 1e3,
            ill_conditioned: 1e6,
            high_floor: 1e-30,
            uncertainty_factor: 3.,
            generic_derived_factor: 100.,
            ninja_budget_cap: 1e-4,
            ninja_tolerance: 1e-8,
            known_failures: BTreeMap::new(),
            twins: BTreeMap::new(),
        }
    }
}

impl Policy {
    /// Read a `ninjars-known-failures-v1` file (`lanes: [{id, reason}]`).
    pub fn load_known_failures(&mut self, path: &Path) -> Result<()> {
        let value: Value = serde_json::from_str(&fs::read_to_string(path)?)?;
        if value["format"] != "ninjars-known-failures-v1" {
            return Err(format!("{}: not a ninjars-known-failures-v1 file", path.display()).into());
        }
        for lane in value["lanes"].as_array().ok_or("known-failure lanes")? {
            let id = lane["id"].as_str().ok_or("known-failure id")?;
            let reason = lane["reason"]
                .as_str()
                .unwrap_or("listed as a known failure");
            self.known_failures
                .insert(id.to_string(), reason.to_string());
        }
        Ok(())
    }

    /// Read original-Ninja errors of rank twins from an `assemble.py` summary
    /// (`validation/rank-n2/summary/{controls,synthetic-twins}.json`: rows
    /// with `id`, `n`, `rank`, `ninja`).
    pub fn load_twins(&mut self, path: &Path) -> Result<()> {
        let value: Value = serde_json::from_str(&fs::read_to_string(path)?)?;
        for row in value
            .as_array()
            .ok_or("a summary is a JSON array of lane rows")?
        {
            let (Some(id), Some(n), Some(rank), Some(error)) = (
                row["id"].as_str(),
                row["n"].as_i64(),
                row["rank"].as_i64(),
                row["ninja"].as_f64(),
            ) else {
                continue;
            };
            let slot = self
                .twins
                .entry(kinematics_key(id))
                .or_default()
                .entry(rank - n)
                .or_insert(0.);
            *slot = slot.max(error);
        }
        Ok(())
    }

    /// Original Ninja's binary64 error on this lane, measured or estimated
    /// from the rank twins, and where it comes from.
    pub fn ninja_error(&self, lane: &ReferenceLane) -> Option<(f64, String)> {
        // A known failure shared with original Ninja (the oneloopmaster D0 bug)
        // is not a conditioning measure: no Ninja budget on its kinematics.
        let key = kinematics_key(&lane.id);
        if self
            .known_failures
            .keys()
            .any(|id| kinematics_key(id) == key)
        {
            return None;
        }
        if let Some(error) = lane.ninja_error {
            return Some((error, "original Ninja on this lane".into()));
        }
        let twins = self.twins.get(&key)?;
        let excess = lane.excess?;
        if let Some(&error) = twins.get(&excess) {
            return Some((
                error,
                format!("original Ninja on the rank twins at excess {excess}"),
            ));
        }
        let below: Vec<(i64, f64)> = twins.range(..excess).map(|(&k, &v)| (k, v)).collect();
        match below.as_slice() {
            [.., (x0, e0), (x1, e1)] if *e0 > 0. => {
                let growth = (e1 / e0).max(1.).powf(1. / (x1 - x0) as f64);
                Some((
                    e1 * growth.powi((excess - x1) as i32),
                    format!("original Ninja on the rank twins, extrapolated from excess {x1}"),
                ))
            }
            [.., (x1, e1)] => Some((
                *e1,
                format!("original Ninja on the rank twins at excess {x1}"),
            )),
            [] => None,
        }
    }

    pub fn describe(&self) -> Value {
        json!({
            "high_precision": format!(
                "Float >= 160 bits: tiers A, B: max({} unc, {:e}); tier C and tier-B lanes with generic-engine-derived uncertainty ({}) report-only",
                self.uncertainty_factor, self.high_floor, GENERIC_DERIVED_ORIGINS.join(", ")),
            "fixed_precision": format!(
                "{} unc + eps/eps_f64 max({:e} max(1, cond), A e_ninja), A = {} (fast) or {} (generic); e_ninja = original Ninja's error on the lane or its rank twins ({} kinematics loaded); {}x unc for generic-engine-derived uncertainties; cond > {:e} and tier C report-only",
                self.uncertainty_factor, self.binary64_factor, self.fast_ninja_factor,
                self.generic_ninja_factor, self.twins.len(), self.generic_derived_factor,
                self.ill_conditioned),
            "ninja": format!("original-Ninja references: {:e} scaled; Ninja-unstable lanes report-only", self.ninja_tolerance),
            "known_failures": self.known_failures,
            "scaled_difference": "max over (eps0, eps_m1, eps_m2) of |x - ref| / max_k |ref_k|",
        })
    }

    /// Judge one lane. `difference` is `None` when the engine did not succeed.
    pub fn judge(
        &self,
        engine: Engine,
        precision: Precision,
        reference: Option<(&ReferenceLane, ReferenceKind)>,
        succeeded: bool,
        difference: Option<f64>,
    ) -> Judgement {
        let Some((lane, kind)) = reference else {
            return Judgement {
                verdict: Verdict::Unchecked,
                tolerance: None,
                within_tolerance: None,
                reason: Some("no reference lane".into()),
            };
        };
        let mut report_only: Option<String> = None;
        let mut basis: Option<String> = None;
        let tolerance = match kind {
            ReferenceKind::Ninja => {
                if lane.ninja_status.is_some_and(|s| s != 0) {
                    report_only = Some("original Ninja flagged the lane unstable".into());
                }
                Some(self.ninja_tolerance)
            }
            ReferenceKind::HighPrecision => {
                let tier = lane.tier.as_deref().unwrap_or("C");
                let generic_derived = tier == "B"
                    && lane
                        .uncertainty_origin
                        .as_deref()
                        .is_some_and(|o| GENERIC_DERIVED_ORIGINS.contains(&o));
                if tier != "A" && tier != "B" {
                    report_only = Some(format!("tier {tier} reference"));
                }
                match lane.uncertainty {
                    None => {
                        report_only.get_or_insert_with(|| "reference without uncertainty".into());
                        None
                    }
                    Some(u) if precision.is_high() => {
                        if generic_derived {
                            report_only.get_or_insert_with(|| {
                                format!(
                                    "uncertainty from {} is not conservative",
                                    lane.uncertainty_origin.as_deref().unwrap_or("")
                                )
                            });
                        }
                        Some((self.uncertainty_factor * u).max(self.high_floor))
                    }
                    Some(u) => {
                        let condition = lane.condition.unwrap_or(1.).max(1.);
                        if condition > self.ill_conditioned {
                            report_only.get_or_insert_with(|| {
                                format!("ill-conditioned lane (cond {condition:.1e})")
                            });
                        }
                        let u = if generic_derived {
                            self.generic_derived_factor * u
                        } else {
                            u
                        };
                        let factor = match engine {
                            Engine::Fast => self.fast_ninja_factor,
                            Engine::Generic => self.generic_ninja_factor,
                        };
                        let mut budget = self.binary64_factor * condition;
                        if let Some((error, origin)) = self.ninja_error(lane) {
                            budget = budget.max((factor * error).min(self.ninja_budget_cap));
                            basis = Some(origin);
                        }
                        Some(
                            self.uncertainty_factor * u
                                + precision.epsilon() / Precision::F64.epsilon() * budget,
                        )
                    }
                }
            }
        };
        let known = self.known_failures.get(&lane.id);
        let within = match (succeeded, difference, tolerance) {
            (true, Some(d), Some(t)) => Some(d <= t),
            (true, _, None) => None,
            _ => Some(false),
        };
        let (verdict, reason) = match (within, known, report_only) {
            (Some(true), Some(k), _) => (
                Verdict::Pass,
                Some(format!("listed as a known failure ({k}) but passes")),
            ),
            (_, Some(k), _) => (Verdict::KnownFailure, Some(k.clone())),
            (_, None, Some(reason)) => (Verdict::ReportOnly, Some(reason)),
            (Some(true), None, None) => (Verdict::Pass, None),
            (None, None, None) => (Verdict::ReportOnly, Some("no tolerance".into())),
            (Some(false), None, None) => (
                Verdict::Fail,
                Some(if succeeded {
                    "outside the tolerance".into()
                } else {
                    "the engine did not succeed".into()
                }),
            ),
        };
        let reason = match (reason, basis) {
            (Some(r), Some(b)) => Some(format!("{r}; binary64 budget from {b}")),
            (None, Some(b)) => Some(format!("binary64 budget from {b}")),
            (r, None) => r,
        };
        Judgement {
            verdict,
            tolerance,
            within_tolerance: within,
            reason,
        }
    }
}

// ---------------------------------------------------------------------------
// Precision-generic tensor numerator.

/// Shape of one Laurent expansion: `q = t F(u, x, mu^2)` with `u = 1/t`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Shape {
    /// `Mu` (`beta = 0`) and `MuShifted`: `F = v0 + v1 u`, `mu^2 = t^2 (v0^2 - beta u^2)`.
    Mu,
    /// `F = e3 + a u + (beta + mu^2) e4 u^2`.
    T3,
    /// `F = e3 + (a0 + x a1) u + (b0 + b1 x + b2 x^2 + mu^2) e4 u^2`.
    T2,
}

/// A dense tensor numerator `N(q, mu^2) = sum_k (mu^2)^k T_k(q)` in the
/// `tensor_layout` (and `MuTensorNumerator` block) order for the generic
/// engine in any precision.
///
/// Expansions are computed numerically: each monomial `q^m = t^|m| prod_c
/// F_c^(m_c)` is a truncated series in `u = 1/t` (and `x`, `mu^2` for the
/// triangle and bubble parametrisations), built from its parent monomial with
/// one multiplication by `F_c`; the requested Laurent coefficients are then
/// contracted with the lane's coefficients. Exact arithmetic would reproduce
/// [`ninjars_symbolica::PolynomialNumerator`] exactly; in floating point both
/// round differently.
pub struct TensorGenericNumerator<T> {
    rank: usize,
    mu_powers: usize,
    width: usize,
    degrees: Vec<usize>,
    /// `(parent monomial, component)` of every monomial but the constant.
    parents: Vec<(usize, usize)>,
    coefficients: Vec<Complex<T>>,
    lanes: usize,
    zero: Complex<T>,
    one: Complex<T>,
}

impl<T: Real + SingleFloat> TensorGenericNumerator<T>
where
    Complex<T>: Real,
{
    pub fn new(rank: usize, mu_powers: usize, prototype: &T) -> Result<Self> {
        if 2 * mu_powers > rank {
            return Err("mu_powers exceeds rank / 2".into());
        }
        let list = monomials(rank);
        let degrees = list
            .iter()
            .map(|m| m.iter().sum::<u32>() as usize)
            .collect();
        let parents = list
            .iter()
            .map(|m| {
                let Some(c) = (0..4).rev().find(|&c| m[c] > 0) else {
                    return (0, 0);
                };
                let mut parent = *m;
                parent[c] -= 1;
                (monomial_index(parent), c)
            })
            .collect();
        let zero = Complex::new(prototype.zero(), prototype.zero());
        let one = Complex::new(prototype.one(), prototype.zero());
        Ok(Self {
            rank,
            mu_powers,
            width: ncoeff_mu(rank, mu_powers),
            degrees,
            parents,
            coefficients: Vec::new(),
            lanes: 0,
            zero,
            one,
        })
    }

    /// Coefficients per lane.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Bind lane-major coefficients (`width()` per lane).
    pub fn set_coefficients(&mut self, coefficients: Vec<Complex<T>>, lanes: usize) -> Result<()> {
        if coefficients.len() != lanes * self.width {
            return Err(format!(
                "{} coefficients for {lanes} lanes of width {}",
                coefficients.len(),
                self.width
            )
            .into());
        }
        self.coefficients = coefficients;
        self.lanes = lanes;
        Ok(())
    }

    /// `(offset, size)` of the `mu^(2k)` block.
    fn block(&self, k: usize) -> (usize, usize) {
        (mu_block_offset(self.rank, k), ncoeff(self.rank - 2 * k))
    }

    fn lane_coefficients(&self, lane: usize) -> ninjars::Result<&[Complex<T>]> {
        if lane >= self.lanes {
            return Err(ninjars::Error::Lane {
                lane,
                lanes: self.lanes,
            });
        }
        Ok(&self.coefficients[lane * self.width..(lane + 1) * self.width])
    }
}

fn minkowski<T: Real>(a: &[Complex<T>; 4], b: &[Complex<T>; 4]) -> Complex<T>
where
    Complex<T>: Real,
{
    a[0].clone() * &b[0] - a[1].clone() * &b[1] - a[2].clone() * &b[2] - a[3].clone() * &b[3]
}

fn components<T: Clone>(v: &FourMomentum<T>) -> [T; 4] {
    v.components().map(|x| x.clone())
}

impl<T: Real + SingleFloat> Numerator<Complex<T>> for TensorGenericNumerator<T>
where
    Complex<T>: Real,
{
    fn rank(&self) -> usize {
        self.rank
    }

    fn evaluate_batch(
        &mut self,
        requests: &[NumeratorRequest<'_, Complex<T>>],
        output: &mut [Complex<T>],
    ) -> ninjars::Result<()> {
        BatchShape {
            rows: requests.len(),
            columns: 1,
        }
        .validate("numerator output", output.len())?;
        let count = self.degrees.len();
        let mut values = vec![self.zero.clone(); count];
        for (request, out) in requests.iter().zip(output.iter_mut()) {
            let coefficients = self.lane_coefficients(request.id.lane)?;
            let q = components(request.q);
            values[0] = self.one.clone();
            for i in 1..count {
                let (parent, c) = self.parents[i];
                values[i] = values[parent].clone() * &q[c];
            }
            let mut total = self.zero.clone();
            let mut mu_power = self.one.clone();
            for k in 0..=self.mu_powers {
                let (offset, size) = self.block(k);
                let mut block = self.zero.clone();
                for (c, v) in coefficients[offset..offset + size].iter().zip(&values) {
                    block += c.clone() * v;
                }
                total += block * &mu_power;
                mu_power *= request.mu_squared;
            }
            *out = total;
        }
        Ok(())
    }

    fn expansion_batch(
        &mut self,
        kind: &ExpansionKind,
        requests: &[ExpansionRequest<'_, Complex<T>>],
        output: &mut [Complex<T>],
    ) -> ninjars::Result<()> {
        let columns = kind.len();
        BatchShape {
            rows: requests.len(),
            columns,
        }
        .validate("expansion output", output.len())?;
        let (shape, vectors, scalars, shifted) = match kind {
            ExpansionKind::Mu { .. } => (Shape::Mu, 2, 0, false),
            ExpansionKind::MuShifted { .. } => (Shape::Mu, 2, 1, true),
            ExpansionKind::T3 { .. } => (Shape::T3, 3, 1, false),
            ExpansionKind::T2 { .. } => (Shape::T2, 4, 3, false),
        };
        for request in requests {
            self.lane_coefficients(request.id.lane)?;
            if request.vectors.len() != vectors
                && !(matches!(kind, ExpansionKind::Mu { .. }) && request.vectors.len() == 1)
            {
                return Err(ninjars::Error::Shape {
                    name: "cut vectors",
                    expected: vectors,
                    actual: request.vectors.len(),
                });
            }
            BatchShape {
                rows: 1,
                columns: scalars,
            }
            .validate("cut parameters", request.parameters.len())?;
        }
        if requests.is_empty() || columns == 0 {
            return Ok(());
        }
        // Requested (t power, x power, mu^2 power) per column.
        let powers: Vec<(i64, usize, usize)> = match kind {
            ExpansionKind::Mu { powers } | ExpansionKind::MuShifted { powers } => {
                powers.iter().map(|&p| (i64::from(p), 0, 0)).collect()
            }
            ExpansionKind::T3 { powers } => powers
                .iter()
                .map(|&(p, l)| (i64::from(p), 0, usize::from(l)))
                .collect(),
            ExpansionKind::T2 { powers } => powers
                .iter()
                .map(|&(p, a, l)| (i64::from(p), usize::from(a), usize::from(l)))
                .collect(),
        };
        let lowest = powers.iter().map(|p| p.0).min().unwrap_or(0);
        // Deepest u order any term reaches at the requested powers.
        let depth = (self.rank as i64 - lowest).max(0) as usize;
        let (xs, mus) = match shape {
            Shape::Mu => (1, 1),
            Shape::T3 => (1, depth / 2 + 1),
            Shape::T2 => (depth + 1, depth / 2 + 1),
        };
        let size = (depth + 1) * xs * mus;
        let at = |j: usize, a: usize, l: usize| (j * xs + a) * mus + l;
        let count = self.degrees.len();
        let mut series = vec![self.zero.clone(); count * size];
        for (row, request) in requests.iter().enumerate() {
            let coefficients = self.lane_coefficients(request.id.lane)?;
            let v: Vec<[Complex<T>; 4]> = request.vectors.iter().map(components).collect();
            let zero4 = || std::array::from_fn::<_, 4, _>(|_| self.zero.clone());
            // Factor coefficients per component: [1, u, u x, u^2, u^2 x, u^2 x^2, u^2 mu^2].
            let factors: Vec<[Complex<T>; 7]> = (0..4)
                .map(|c| {
                    let z = || self.zero.clone();
                    match shape {
                        Shape::Mu => {
                            let v1 = v.get(1).cloned().unwrap_or_else(zero4);
                            [v[0][c].clone(), v1[c].clone(), z(), z(), z(), z(), z()]
                        }
                        Shape::T3 => {
                            let e4 = &v[2][c];
                            [
                                v[1][c].clone(),
                                v[0][c].clone(),
                                z(),
                                request.parameters[0].clone() * e4,
                                z(),
                                z(),
                                e4.clone(),
                            ]
                        }
                        Shape::T2 => {
                            let e4 = &v[3][c];
                            [
                                v[2][c].clone(),
                                v[0][c].clone(),
                                v[1][c].clone(),
                                request.parameters[0].clone() * e4,
                                request.parameters[1].clone() * e4,
                                request.parameters[2].clone() * e4,
                                e4.clone(),
                            ]
                        }
                    }
                })
                .collect();
            for value in series.iter_mut() {
                *value = self.zero.clone();
            }
            series[at(0, 0, 0)] = self.one.clone();
            for i in 1..count {
                let (parent, c) = self.parents[i];
                let f = &factors[c];
                let (head, tail) = series.split_at_mut(i * size);
                let source = &head[parent * size..(parent + 1) * size];
                let target = &mut tail[..size];
                for j in 0..=depth {
                    for a in 0..xs.min(j + 1) {
                        for l in 0..mus {
                            if a + 2 * l > j {
                                break;
                            }
                            let mut acc = f[0].clone() * &source[at(j, a, l)];
                            if j >= 1 {
                                acc += f[1].clone() * &source[at(j - 1, a, l)];
                                if shape == Shape::T2 && a >= 1 {
                                    acc += f[2].clone() * &source[at(j - 1, a - 1, l)];
                                }
                            }
                            if j >= 2 && shape != Shape::Mu {
                                acc += f[3].clone() * &source[at(j - 2, a, l)];
                                if shape == Shape::T2 {
                                    if a >= 1 {
                                        acc += f[4].clone() * &source[at(j - 2, a - 1, l)];
                                    }
                                    if a >= 2 {
                                        acc += f[5].clone() * &source[at(j - 2, a - 2, l)];
                                    }
                                }
                                if l >= 1 {
                                    acc += f[6].clone() * &source[at(j - 2, a, l - 1)];
                                }
                            }
                            target[at(j, a, l)] = acc;
                        }
                    }
                }
            }
            let out = &mut output[row * columns..(row + 1) * columns];
            match shape {
                Shape::Mu => {
                    // mu^2 = t^2 G(u), G = v0^2 - beta u^2; powers G^k truncated at depth.
                    let g0 = minkowski(&v[0], &v[0]);
                    let beta = if shifted {
                        request.parameters[0].clone()
                    } else {
                        self.zero.clone()
                    };
                    let mut g_powers = vec![{
                        let mut s = vec![self.zero.clone(); depth + 1];
                        s[0] = self.one.clone();
                        s
                    }];
                    for k in 1..=self.mu_powers {
                        let previous = &g_powers[k - 1];
                        let mut s = vec![self.zero.clone(); depth + 1];
                        for j in 0..=depth {
                            let mut acc = g0.clone() * &previous[j];
                            if j >= 2 {
                                acc -= beta.clone() * &previous[j - 2];
                            }
                            s[j] = acc;
                        }
                        g_powers.push(s);
                    }
                    for (column, &(p, _, _)) in powers.iter().enumerate() {
                        let mut acc = self.zero.clone();
                        for (k, g) in g_powers.iter().enumerate() {
                            let (offset, block) = self.block(k);
                            for i in 0..block {
                                let c = &coefficients[offset + i];
                                if c.is_fully_zero() {
                                    continue;
                                }
                                let j = (self.degrees[i] + 2 * k) as i64 - p;
                                if j < 0 || j as usize > depth {
                                    continue;
                                }
                                let j = j as usize;
                                let s = &series[i * size..(i + 1) * size];
                                let mut term = self.zero.clone();
                                for j1 in 0..=j {
                                    term += s[at(j1, 0, 0)].clone() * &g[j - j1];
                                }
                                acc += term * c;
                            }
                        }
                        out[column] = acc;
                    }
                }
                Shape::T3 | Shape::T2 => {
                    for (column, &(p, a, total_mu)) in powers.iter().enumerate() {
                        let mut acc = self.zero.clone();
                        for k in 0..=self.mu_powers.min(total_mu) {
                            let l = total_mu - k;
                            if l >= mus || a >= xs {
                                continue;
                            }
                            let (offset, block) = self.block(k);
                            for i in 0..block {
                                let c = &coefficients[offset + i];
                                if c.is_fully_zero() {
                                    continue;
                                }
                                let j = self.degrees[i] as i64 - p;
                                if j < 0 || j as usize > depth || a + 2 * l > j as usize {
                                    continue;
                                }
                                acc += series[i * size + at(j as usize, a, l)].clone() * c;
                            }
                        }
                        out[column] = acc;
                    }
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Generic engine.

/// One generic-engine lane: status and `[eps0, eps_m1, eps_m2,
/// cut_constructible, rational]`.
pub struct GenericLane<T> {
    pub status: EvaluationStatus,
    pub values: [Complex<T>; 5],
}

/// The generic engine (`SymbolicaAmplitudeKernels`) in one precision, with
/// plans cached per `(n, declared rank)`.
pub struct GenericEngine<T: oneloop::NativeFloat>
where
    Complex<T>: Real + EvaluationDomain,
{
    prototype: T,
    /// Opt-in cancellation test ([`SymbolicaAmplitudeKernels::with_stability_test`]).
    stability_test: bool,
    low: SymbolicaCuts<T>,
    high: SymbolicaHighCuts<T>,
    amplitudes:
        HashMap<(usize, usize), ninjars::Amplitude<Complex<T>, SymbolicaAmplitudeKernels<T>>>,
}

impl<T: oneloop::NativeFloat> GenericEngine<T>
where
    Complex<T>: Real + EvaluationDomain,
{
    pub fn new(prototype: T) -> Result<Self> {
        Ok(Self {
            low: SymbolicaCuts::new(prototype.clone())?,
            high: SymbolicaHighCuts::new(prototype.clone())?,
            prototype,
            stability_test: false,
            amplitudes: HashMap::new(),
        })
    }

    /// Report lanes whose cut contributions cancel to fewer than half of the
    /// carrier's bits as [`EvaluationStatus::Unstable`] (`--stability-test`).
    pub fn with_stability_test(mut self, enabled: bool) -> Self {
        self.stability_test = enabled;
        self.amplitudes.clear();
        self
    }

    pub fn prototype(&self) -> &T {
        &self.prototype
    }

    pub fn exact(&self, x: f64) -> T {
        self.prototype
            .from_rational(&Rational::try_from(x).expect("finite input"))
    }

    pub fn complex(&self, z: C64) -> Complex<T> {
        Complex::new(self.exact(z.re), self.exact(z.im))
    }

    /// Build (or reuse) the plans for `n` propagators at `rank`.
    pub fn prepare(&mut self, n: usize, rank: usize) -> Result<()> {
        if !self.amplitudes.contains_key(&(n, rank)) {
            let kernels = SymbolicaAmplitudeKernels::new(n, rank, self.prototype.clone())?;
            let kernels = if self.stability_test {
                kernels.with_stability_test()
            } else {
                kernels
            };
            let amplitude = kernels.into_amplitude(1)?;
            self.amplitudes.insert((n, rank), amplitude);
        }
        Ok(())
    }

    /// Reduce every lane of `case` with the numerator declared at
    /// `declared_rank` (>= the case rank; the topology uses the declared rank,
    /// the numerator keeps its own).
    pub fn evaluate<M: MasterProvider<Complex<T>>>(
        &mut self,
        case: &BenchCase,
        declared_rank: usize,
        masters: &mut M,
    ) -> Result<Vec<GenericLane<T>>> {
        let n = case.n;
        let count = case.lanes.len();
        let mut numerator =
            TensorGenericNumerator::new(case.rank, case.mu_powers, &self.prototype)?;
        let coefficients = case
            .lanes
            .iter()
            .flat_map(|lane| lane.tensor.iter().map(|&z| self.complex(z)))
            .collect();
        numerator.set_coefficients(coefficients, count)?;
        let real = |x: f64| Complex::new(self.exact(x), self.prototype.zero());
        let shifts: Vec<FourMomentum<Complex<T>>> = case
            .lanes
            .iter()
            .flat_map(|lane| lane.shifts.iter().map(|v| FourMomentum::from(v.map(real))))
            .collect();
        let masses: Vec<Complex<T>> = case
            .lanes
            .iter()
            .flat_map(|lane| lane.masses.iter().map(|&m| self.complex(m)))
            .collect();
        let scales: Vec<Complex<T>> = case.lanes.iter().map(|lane| real(lane.mu2)).collect();
        // Ninja's auxiliary box mu^2 sampling scale (as the S5a generic runner).
        let sampling: Vec<Complex<T>> = case
            .lanes
            .iter()
            .map(|lane| {
                let invariants = lane.s_matrix.iter().flatten().map(|x| x.abs());
                let masses = lane.masses.iter().map(|m| m.re.abs());
                real(invariants.chain(masses).fold(1., f64::max))
            })
            .collect();
        // Supplied S-matrices; missing ones (when another lane has one) are
        // the exact invariants of the shifts in the carrier's precision.
        let s_matrix: Option<Vec<Complex<T>>> = case
            .lanes
            .iter()
            .any(|lane| lane.s_matrix.is_some())
            .then(|| {
                let mut matrix = Vec::with_capacity(count * n * n);
                for (index, lane) in case.lanes.iter().enumerate() {
                    match &lane.s_matrix {
                        Some(s) => matrix.extend(s.iter().map(|&x| real(x))),
                        None => {
                            let v = &shifts[index * n..(index + 1) * n];
                            for i in 0..n {
                                for j in 0..n {
                                    let a = components(&v[i]);
                                    let b = components(&v[j]);
                                    let d: [Complex<T>; 4] =
                                        std::array::from_fn(|k| a[k].clone() - &b[k]);
                                    matrix.push(minkowski(&d, &d));
                                }
                            }
                        }
                    }
                }
                matrix
            });
        self.prepare(n, declared_rank)?;
        let amplitude = self.amplitudes.get_mut(&(n, declared_rank)).unwrap();
        amplitude.reset_batch(count);
        amplitude.evaluate_batch_with_inputs(
            AmplitudeInput {
                shifts: &shifts,
                masses_squared: &masses,
                scales_squared: &scales,
                sampling_scales: &sampling,
                s_matrix: s_matrix.as_deref(),
            },
            &mut numerator,
            &mut self.low,
            &mut self.high,
            masters,
        )?;
        Ok(amplitude
            .results()
            .iter()
            .zip(amplitude.statuses())
            .map(|(values, &status)| GenericLane {
                status,
                values: values.clone(),
            })
            .collect())
    }
}

/// Decimal strings `[re, im]` of a carrier value (45 significant digits for
/// arbitrary precision, shortest round trip in binary64).
pub fn decimal<T: SingleFloat + std::fmt::LowerExp>(z: &Complex<T>) -> [String; 2] {
    [format!("{:.45e}", z.re), format!("{:.45e}", z.im)]
}

// ---------------------------------------------------------------------------
// Fast engine.

/// The fast engine's batch of `width` lanes (the case's lanes repeated
/// cyclically, as `case-bench` builds it) and its results.
pub struct FastRun {
    pub ids: Vec<String>,
    /// Index of the case lane of every batch lane.
    pub lanes: Vec<usize>,
    pub results: Vec<FastResult>,
    pub statuses: Vec<FastStatus>,
}

/// Reduce `case` with the binary64 fast engine: `TensorNumerator` (v1) or
/// `MuTensorNumerator` (v2) declared at `declared_rank`, native masters, the
/// lane S-matrices only when every lane has one (as `case-bench`).
pub fn run_fast<M: FastMasters>(
    case: &BenchCase,
    declared_rank: usize,
    width: usize,
    masters: &mut M,
) -> Result<FastRun> {
    if width == 0 {
        return Err("the fast-engine width must be positive".into());
    }
    let lanes: Vec<usize> = (0..width).map(|i| i % case.lanes.len()).collect();
    let data: Vec<&BenchLane> = lanes.iter().map(|&i| &case.lanes[i]).collect();
    let shifts: Vec<RMom> = data.iter().flat_map(|l| l.shifts.iter().copied()).collect();
    let masses: Vec<C64> = data.iter().flat_map(|l| l.masses.iter().copied()).collect();
    let scales: Vec<f64> = data.iter().map(|l| l.mu2).collect();
    let s_matrix: Option<Vec<f64>> = data
        .iter()
        .map(|l| l.s_matrix.clone())
        .collect::<Option<Vec<_>>>()
        .map(|rows| rows.concat());
    let input = FastInput {
        lanes: width,
        shifts: &shifts,
        masses_squared: &masses,
        scales_squared: &scales,
        s_matrix: s_matrix.as_deref(),
    };
    let mut amplitude = FastAmplitude::new(case.n, declared_rank, FastOptions::default())?;
    let mut results = vec![FastResult::default(); width];
    let mut statuses = vec![FastStatus::NotEvaluated; width];
    let size = ncoeff_mu(declared_rank, case.mu_powers);
    if case.mu_powers == 0 {
        let mut tensor = TensorNumerator::new(declared_rank, width);
        for (lane, &index) in lanes.iter().enumerate() {
            tensor.coefficients_mut()[lane * size..(lane + 1) * size]
                .copy_from_slice(&case.padded_tensor(index, declared_rank));
        }
        amplitude.evaluate(&input, &mut tensor, masters, &mut results, &mut statuses)?;
    } else {
        let mut tensor = MuTensorNumerator::new(declared_rank, case.mu_powers, width)?;
        for (lane, &index) in lanes.iter().enumerate() {
            tensor.coefficients_mut()[lane * size..(lane + 1) * size]
                .copy_from_slice(&case.padded_tensor(index, declared_rank));
        }
        amplitude.evaluate(&input, &mut tensor, masters, &mut results, &mut statuses)?;
    }
    Ok(FastRun {
        ids: data.iter().map(|l| l.id.clone()).collect(),
        lanes,
        results,
        statuses,
    })
}

pub fn fast_values(result: &FastResult) -> [C64; 5] {
    [
        result.laurent[0],
        result.laurent[1],
        result.laurent[2],
        result.cut_constructible,
        result.rational,
    ]
}

/// Shortest round-trip decimals of a binary64 value.
pub fn decimal_f64(z: C64) -> [String; 2] {
    [format!("{:e}", z.re), format!("{:e}", z.im)]
}

// ---------------------------------------------------------------------------
// Checking lanes against references.

/// One lane of an engine run, compared and judged.
#[derive(Clone, Debug)]
pub struct CheckedLane {
    pub id: String,
    pub status: String,
    pub succeeded: bool,
    /// Decimal `[re, im]` of eps0, eps_m1, eps_m2, cut-constructible, rational.
    pub values: [[String; 2]; 5],
    /// Scaled difference to the reference (when both exist).
    pub difference: Option<f64>,
    pub judgement: Judgement,
    pub reference: Option<(ReferenceLane, ReferenceKind)>,
}

fn reference_lane<'a>(
    reference: Option<&'a Reference>,
    id: &str,
) -> Option<(&'a ReferenceLane, ReferenceKind)> {
    let reference = reference?;
    reference.lane(id).map(|lane| (lane, reference.kind))
}

/// A fast-engine run of one case, judged per case lane (the first
/// occurrence of each lane in the batch).
pub struct FastCheck {
    pub run: FastRun,
    pub lanes: Vec<CheckedLane>,
    /// Every repeated batch lane equals its first occurrence bitwise.
    pub repeats_bitwise: bool,
}

pub fn check_fast(
    case: &BenchCase,
    declared_rank: usize,
    width: usize,
    reference: Option<&Reference>,
    policy: &Policy,
    masters: &mut OneLoopNativeMasters,
) -> Result<FastCheck> {
    let count = case.lanes.len();
    if width < count {
        return Err(format!("{}: width {width} below the lane count {count}", case.name).into());
    }
    let run = run_fast(case, declared_rank, width, masters)?;
    let bits = |r: &FastResult| fast_values(r).map(|z| (z.re.to_bits(), z.im.to_bits()));
    let repeats_bitwise = (count..width).all(|lane| {
        let first = run.lanes[lane];
        run.statuses[lane] == run.statuses[first]
            && bits(&run.results[lane]) == bits(&run.results[first])
    });
    let mut lanes = Vec::new();
    for (index, lane) in case.lanes.iter().enumerate() {
        let status = run.statuses[index];
        let succeeded = status == FastStatus::Success;
        let values = fast_values(&run.results[index]);
        let found = reference_lane(reference, &lane.id);
        let difference = match found {
            Some((r, _)) if succeeded => Some(scaled_difference_f64(
                &[values[0], values[1], values[2]],
                &r.values,
                r.scale,
            )),
            _ => None,
        };
        let judgement = policy.judge(Engine::Fast, Precision::F64, found, succeeded, difference);
        lanes.push(CheckedLane {
            id: lane.id.clone(),
            status: format!("{status:?}"),
            succeeded,
            values: values.map(decimal_f64),
            difference,
            judgement,
            reference: found.map(|(r, k)| (r.clone(), k)),
        });
    }
    Ok(FastCheck {
        run,
        lanes,
        repeats_bitwise,
    })
}

/// Judge generic-engine lanes (in the carrier's precision).
pub fn check_generic_lanes<T: oneloop::NativeFloat>(
    case: &BenchCase,
    results: &[GenericLane<T>],
    reference: Option<&Reference>,
    policy: &Policy,
    precision: Precision,
    prototype: &T,
) -> Result<Vec<CheckedLane>> {
    let mut lanes = Vec::new();
    for (lane, result) in case.lanes.iter().zip(results) {
        let succeeded = result.status == EvaluationStatus::Success;
        let found = reference_lane(reference, &lane.id);
        let difference = match found {
            Some((r, _)) if succeeded => Some(scaled_difference(
                &result.values,
                &reference_values(r, prototype)?,
                r.scale,
            )),
            _ => None,
        };
        let judgement = policy.judge(Engine::Generic, precision, found, succeeded, difference);
        lanes.push(CheckedLane {
            id: lane.id.clone(),
            status: format!("{:?}", result.status),
            succeeded,
            values: std::array::from_fn(|k| decimal(&result.values[k])),
            difference,
            judgement,
            reference: found.map(|(r, k)| (r.clone(), k)),
        });
    }
    Ok(lanes)
}

/// Generic engines and their master providers, one per precision: `f64`
/// with oneloopmaster's native masters, `DoubleFloat` and `Float` with
/// `SymbolicaMasters` (its exact expressions; B1111 from its native tensor
/// family).
#[derive(Default)]
pub struct Engines {
    f64: Option<(GenericEngine<f64>, OneLoopNativeMasters)>,
    double: Option<(GenericEngine<DoubleFloat>, SymbolicaMasters<DoubleFloat>)>,
    float: BTreeMap<u32, (GenericEngine<Float>, SymbolicaMasters<Float>)>,
    /// Engines created from now on run the opt-in cancellation test.
    pub stability_test: bool,
}

impl Engines {
    pub fn f64(&mut self) -> Result<(&mut GenericEngine<f64>, &mut OneLoopNativeMasters)> {
        if self.f64.is_none() {
            self.f64 = Some((
                GenericEngine::new(0.)?.with_stability_test(self.stability_test),
                OneLoopNativeMasters::new()?,
            ));
        }
        let (engine, masters) = self.f64.as_mut().unwrap();
        Ok((engine, masters))
    }

    pub fn double(
        &mut self,
    ) -> Result<(
        &mut GenericEngine<DoubleFloat>,
        &mut SymbolicaMasters<DoubleFloat>,
    )> {
        if self.double.is_none() {
            let prototype = DoubleFloat::default();
            self.double = Some((
                GenericEngine::new(prototype)?.with_stability_test(self.stability_test),
                SymbolicaMasters::new(prototype),
            ));
        }
        let (engine, masters) = self.double.as_mut().unwrap();
        Ok((engine, masters))
    }

    pub fn float(
        &mut self,
        bits: u32,
    ) -> Result<(&mut GenericEngine<Float>, &mut SymbolicaMasters<Float>)> {
        let (engine, masters) = match self.float.entry(bits) {
            std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::btree_map::Entry::Vacant(entry) => {
                let prototype = Float::new(bits);
                let engine =
                    GenericEngine::new(prototype.clone())?.with_stability_test(self.stability_test);
                entry.insert((engine, SymbolicaMasters::new(prototype)))
            }
        };
        Ok((engine, masters))
    }

    /// Reduce and judge every lane of `case` at `precision`.
    pub fn check(
        &mut self,
        precision: Precision,
        case: &BenchCase,
        declared_rank: usize,
        reference: Option<&Reference>,
        policy: &Policy,
    ) -> Result<Vec<CheckedLane>> {
        match precision {
            Precision::F64 => {
                let (engine, masters) = self.f64()?;
                let results = engine.evaluate(case, declared_rank, masters)?;
                check_generic_lanes(case, &results, reference, policy, precision, &0.)
            }
            Precision::Double => {
                let (engine, masters) = self.double()?;
                let results = engine.evaluate(case, declared_rank, masters)?;
                let prototype = *engine.prototype();
                check_generic_lanes(case, &results, reference, policy, precision, &prototype)
            }
            Precision::Float(bits) => {
                let (engine, masters) = self.float(bits)?;
                let results = engine.evaluate(case, declared_rank, masters)?;
                let prototype = engine.prototype().clone();
                check_generic_lanes(case, &results, reference, policy, precision, &prototype)
            }
        }
    }
}
