//! Export shared tensor-numerator benchmark cases ("ninjars-bench-case-v1").
//!
//! The fixture is prepared exactly as `ninjars prepare` does. The complete
//! numerator (graph factor included once) is split into coefficient heads by
//! `PolynomialCoefficientPlan` and laid out by `TensorCoefficientPlan` in
//! original Ninja's/MadLoop's tensor ordering. Tensor coefficients, shifts,
//! squared masses, mu^2 and the kinematic matrix are evaluated from the exact
//! fixture inputs at high working precision (default 256 bits) and rounded
//! once to binary64. Every lane is checked against the existing
//! `PolynomialNumerator` (binary64 and high precision) at random complex loop
//! momenta; the report goes to stdout (or `--report`), the case file holds only
//! the shared format.
//!
//! A numerator with explicit `mu^2` terms (`--r2 fdh`) cannot be written in
//! original Ninja's tensor layout. It is exported as "ninjars-bench-case-v2":
//! the same fields plus `mu_powers` `K` and `r2`, with each lane's `tensor`
//! holding the blocks `T_0, ..., T_K` of `N = sum_k mu^(2k) T_k(q)` (`T_k` of
//! rank `rank - 2k`, `ninjars::fast::MuTensorNumerator`). The C++ reference
//! driver accepts only v1; `case-bench` reads both. Mu-free numerators keep v1.

use std::{
    error::Error,
    fs,
    path::PathBuf,
    time::{Duration, Instant},
};

use clap::{Args as ClapArgs, ValueEnum};
use ninjars::{FourMomentum, Numerator, NumeratorRequest, SampleId};
use ninjars_graph::{PolynomialCoefficientPlan, TensorCoefficientPlan};
use ninjars_symbolica::PolynomialNumerator;
use serde_json::{Value, json};
use symbolica::{
    atom::{Atom, AtomCore},
    domains::float::{Complex, Float, FloatLike, Real, RealLike},
    symbol,
};

use crate::{
    numerator_coefficients::CoefficientScalar,
    prepare::{self, PreparedDocument},
    runtime::numerical,
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Clone, Copy, ValueEnum)]
pub enum SMatrix {
    /// Sample-supplied matrix, else exact `(V_i - V_j)^2` from the exact shifts.
    Exact,
    /// Write `null`; consumers derive the matrix from the rounded shifts.
    None,
}

#[derive(ClapArgs)]
pub struct Args {
    /// Schema-version-1 JSON fixture, or a DOT graph with --samples.
    pub input: PathBuf,
    #[arg(long)]
    pub samples: Option<PathBuf>,
    #[arg(long)]
    pub model: Option<String>,
    #[arg(long)]
    pub feynman_rules: bool,
    /// R2 from the numerator: `off` (four-dimensional loop momentum, R1 only)
    /// or `fdh` (K.K -> q.q - mu^2 in FDH; the mu^2 terms give R2).
    #[arg(long, value_enum, default_value = "off")]
    pub r2: prepare::R2,
    /// Bench-case JSON to write.
    #[arg(short, long)]
    pub output: PathBuf,
    /// Comma-separated fixture sample indices to export (default: all, in order).
    #[arg(long, value_delimiter = ',')]
    pub select: Vec<usize>,
    /// Case name (default: the input file stem).
    #[arg(long)]
    pub name: Option<String>,
    /// Tensor rank (default: the numerator's degree in the loop momentum).
    #[arg(long)]
    pub rank: Option<usize>,
    #[arg(long, value_enum, default_value = "exact")]
    pub s_matrix: SMatrix,
    /// Working precision in bits of the exact-input evaluation.
    #[arg(long, default_value = "256")]
    pub bits: u32,
    /// Random complex loop momenta per lane checked against PolynomialNumerator.
    #[arg(long, default_value = "16")]
    pub check_points: usize,
    /// Factor the coefficient heads exactly before evaluation/compilation.
    #[arg(long)]
    pub factor: bool,
    /// Also compile the SymJIT O2 heads (requires --features numerator-jit) and
    /// compare their binary64 tensor with the high-precision export.
    #[arg(long)]
    pub symjit: bool,
    /// Write the JSON report here instead of standard output.
    #[arg(long)]
    pub report: Option<PathBuf>,
}

fn ns(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e9
}

fn to_f64(value: &Complex<Float>) -> Complex<f64> {
    Complex::new(value.re.to_f64(), value.im.to_f64())
}

fn real(value: &Complex<Float>, label: &str) -> Result<f64> {
    if value.im.to_f64() != 0. {
        return Err(format!("{label} must be real").into());
    }
    Ok(value.re.to_f64())
}

/// Deterministic xorshift64* stream in [-1, 1).
struct Stream(u64);
impl Stream {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let bits = self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11;
        bits as f64 / (1u64 << 52) as f64 - 1.
    }
}

/// `N(q, mu^2) = sum_k tensor[k] q^a(k) (mu^2)^b(k)`, plus the absolute sum
/// of the terms.
fn tensor_value(
    plan: &TensorCoefficientPlan,
    tensor: &[Complex<f64>],
    q: &[Complex<f64>; 4],
    mu_squared: Complex<f64>,
) -> (Complex<f64>, f64) {
    let mut value = Complex::new(0., 0.);
    let mut scale = 0.;
    for entry in plan.entries() {
        let mut term = tensor[entry.slot];
        for (component, power) in q.iter().zip(entry.exponents) {
            for _ in 0..power {
                term *= *component;
            }
        }
        for _ in 0..entry.mu_power {
            term *= mu_squared;
        }
        scale += term.norm().re;
        value += term;
    }
    (value, scale)
}

fn polynomial_values<T>(
    numerator: &mut PolynomialNumerator<T>,
    parameters: &[Complex<T>],
    lanes: usize,
    points: &[(usize, [Complex<T>; 4])],
    mu_squared: &[Complex<T>],
    zero: Complex<T>,
) -> Result<Vec<Complex<T>>>
where
    T: Real + symbolica::domains::float::SingleFloat,
    Complex<T>: Real + symbolica::evaluate::EvaluationDomain,
{
    numerator.set_parameter_batch(parameters, lanes)?;
    let momenta: Vec<_> = points
        .iter()
        .map(|(_, q)| {
            FourMomentum::from_args(q[0].clone(), q[1].clone(), q[2].clone(), q[3].clone())
        })
        .collect();
    let requests: Vec<_> = points
        .iter()
        .zip(&momenta)
        .zip(mu_squared)
        .enumerate()
        .map(|(sample, (((lane, _), q), mu_squared))| NumeratorRequest {
            id: SampleId {
                lane: *lane,
                sample,
            },
            q,
            mu_squared,
            partition: &[],
        })
        .collect();
    let mut output = vec![zero.clone(); requests.len()];
    numerator.evaluate_batch(&requests, &mut output)?;
    Ok(output)
}

pub fn run(args: Args) -> Result<()> {
    let start = Instant::now();
    let prepare_args = prepare::Args {
        input: args.input.clone(),
        samples: args.samples.clone(),
        model: args.model.clone(),
        feynman_rules: args.feynman_rules,
        r2: args.r2,
        output: None,
        #[cfg(feature = "reference-codegen")]
        reference_dir: None,
        #[cfg(feature = "reference-codegen")]
        reference_bench_dir: None,
    };
    let prepared: PreparedDocument = prepare::prepare(&prepare_args)?;
    let prepare_time = start.elapsed();
    let selection: Vec<usize> = if args.select.is_empty() {
        (0..prepared.samples.len()).collect()
    } else {
        args.select.clone()
    };
    if selection.is_empty() {
        return Err("export-bench needs at least one fixture sample".into());
    }
    if let Some(index) = selection.iter().find(|i| **i >= prepared.samples.len()) {
        return Err(format!(
            "selected sample {index} is outside 0..{}",
            prepared.samples.len()
        )
        .into());
    }
    let lanes = selection.len();
    let n = prepared.propagators;
    let variables =
        ["q0", "q1", "q2", "q3", "mu2"].map(|name| Atom::var(symbol!(format!("ninjars::{name}"))));

    let start = Instant::now();
    let plan = PolynomialCoefficientPlan::new(
        prepared.numerator_template.clone(),
        variables.clone(),
        prepared.parameters.clone(),
    )?;
    let planning_time = start.elapsed();
    let start = Instant::now();
    let plan = if args.factor {
        plan.factor_coefficients()?
    } else {
        plan
    };
    let factoring_time = start.elapsed();
    let start = Instant::now();
    // Ninja's plain tensor layout unless the numerator has mu^2 terms.
    let tensor_plan = TensorCoefficientPlan::with_mu_squared(&plan, args.rank)?;
    let mu_powers = tensor_plan.mu_powers();
    let tensor_plan_time = start.elapsed();
    let ncoeff = tensor_plan.ncoeff();
    let heads = tensor_plan.head_count();

    // Exact inputs at high precision, rounded once.
    let start = Instant::now();
    let proto = Float::new(args.bits);
    let high_parameters = selection
        .iter()
        .flat_map(|&index| {
            prepared.samples[index]
                .parameters
                .iter()
                .enumerate()
                .map(move |(column, atom)| (index, column, atom))
        })
        .map(|(index, column, atom)| {
            numerical(atom, &proto, &format!("sample {index} parameter {column}"))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut high_heads = vec![Complex::new(proto.zero(), proto.zero()); lanes * heads];
    plan.compile(proto.clone())?
        .evaluate_batch(&high_parameters, &mut high_heads, lanes)?;
    let rounded_heads: Vec<_> = high_heads.iter().map(to_f64).collect();
    let mut tensor = vec![Complex::new(0., 0.); lanes * ncoeff];
    tensor_plan.scatter(&rounded_heads, &mut tensor, lanes, &Complex::new(0., 0.))?;
    let evaluation_time = start.elapsed();

    let scale_squared = real(
        &numerical(&prepared.scale_squared, &proto, "scale_squared")?,
        "scale_squared",
    )?;
    let mut case_lanes = Vec::new();
    let mut energy_scale = 0_f64;
    let mut snapped = 0usize;
    for (row, &index) in selection.iter().enumerate() {
        let sample = &prepared.samples[index];
        let label = |what: &str, i: usize| format!("sample {index} {what} {i}");
        let shifts = sample
            .shifts
            .iter()
            .enumerate()
            .map(|(i, shift)| {
                shift
                    .components()
                    .into_iter()
                    .map(|atom| {
                        real(
                            &numerical(atom, &proto, &label("shift", i))?,
                            &label("shift", i),
                        )
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        energy_scale = shifts.iter().fold(energy_scale, |m, v| m.max(v[0].abs()));
        let masses = sample
            .mass_squared
            .iter()
            .enumerate()
            .map(|(i, atom)| numerical(atom, &proto, &label("mass_squared", i)).map(|v| to_f64(&v)))
            .collect::<Result<Vec<_>>>()?;
        let s_matrix = match args.s_matrix {
            SMatrix::None => Value::Null,
            SMatrix::Exact => {
                let derived;
                let exact = if let Some(supplied) = &sample.s_matrix {
                    supplied
                } else {
                    derived = sample
                        .shifts
                        .iter()
                        .flat_map(|first| {
                            sample.shifts.iter().map(move |second| {
                                let difference = second.clone() - first.clone();
                                difference.dot(&difference).expand()
                            })
                        })
                        .collect::<Vec<_>>();
                    &derived
                };
                let reference = sample
                    .shifts
                    .iter()
                    .map(|v| real(&numerical(v.components()[0], &proto, "energy")?, "energy"))
                    .collect::<Result<Vec<_>>>()?
                    .into_iter()
                    .fold(1_f64, |m, e| m.max(e.abs()))
                    .powi(2);
                let values = exact
                    .iter()
                    .enumerate()
                    .map(|(i, atom)| {
                        let value = real(
                            &numerical(atom, &proto, &label("s_matrix", i))?,
                            &label("s_matrix", i),
                        )?;
                        // Symbolica may not canonicalize every radical identity to
                        // zero; a 256-bit value below 2^-200 of the scale is zero.
                        if value != 0. && value.abs() < reference * 2_f64.powi(-200) {
                            snapped += 1;
                            Ok(0.)
                        } else {
                            Ok(value)
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;
                json!(values.chunks(n).collect::<Vec<_>>())
            }
        };
        let id = sample
            .id
            .clone()
            .unwrap_or_else(|| format!("sample-{index}"));
        case_lanes.push(json!({
            "id": id,
            "shifts": shifts,
            "masses_squared": masses.iter().map(|m| [m.re, m.im]).collect::<Vec<_>>(),
            "mu_squared": scale_squared,
            "s_matrix": s_matrix,
            "tensor": tensor[row * ncoeff..(row + 1) * ncoeff].iter().map(|c| [c.re, c.im]).collect::<Vec<_>>(),
        }));
    }
    let name = args.name.clone().unwrap_or_else(|| {
        args.input
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "case".into())
    });
    let case = if mu_powers == 0 {
        let mut case = json!({
            "format": "ninjars-bench-case-v1",
            "name": name,
            "n": n,
            "rank": tensor_plan.rank(),
            "lanes": case_lanes,
        });
        if args.r2 != prepare::R2::Off {
            case["r2"] = args.r2.name().into();
        }
        case
    } else {
        json!({
            "format": "ninjars-bench-case-v2",
            "name": name,
            "n": n,
            "rank": tensor_plan.rank(),
            "mu_powers": mu_powers,
            "r2": args.r2.name(),
            "tensor_layout": "blocks k = 0..=mu_powers of mu^(2k) T_k(q), T_k a rank-(rank-2k) tensor in ninjars::fast::tensor_layout order",
            "lanes": case_lanes,
        })
    };

    // Tensor reconstruction versus the existing polynomial numerator.
    let verification = if args.check_points > 0 {
        let scale = if energy_scale > 0. { energy_scale } else { 1. };
        let mut stream = Stream(0x9e37_79b9_7f4a_7c15);
        let points: Vec<(usize, [Complex<f64>; 4])> = (0..lanes)
            .flat_map(|lane| (0..args.check_points).map(move |_| lane))
            .map(|lane| {
                (
                    lane,
                    std::array::from_fn(|_| {
                        Complex::new(scale * stream.next(), scale * stream.next())
                    }),
                )
            })
            .collect();
        // mu^2 stays zero for mu-free numerators, keeping their checks unchanged.
        let mu_points: Vec<Complex<f64>> = points
            .iter()
            .map(|_| {
                if mu_powers == 0 {
                    Complex::new(0., 0.)
                } else {
                    Complex::new(scale * scale * stream.next(), scale * scale * stream.next())
                }
            })
            .collect();
        let f64_parameters: Vec<_> = high_parameters.iter().map(to_f64).collect();
        let mut direct = PolynomialNumerator::new(
            prepared.numerator_template.clone(),
            variables.clone(),
            prepared.parameters.clone(),
            0.,
        )?;
        let f64_values = polynomial_values(
            &mut direct,
            &f64_parameters,
            lanes,
            &points,
            &mu_points,
            Complex::new(0., 0.),
        )?;
        let mut high = PolynomialNumerator::new(
            prepared.numerator_template.clone(),
            variables.clone(),
            prepared.parameters.clone(),
            proto.clone(),
        )?;
        let high_points: Vec<_> = points
            .iter()
            .map(|(lane, q)| {
                (
                    *lane,
                    q.map(|c| {
                        Complex::new(
                            Float::with_val(args.bits, c.re),
                            Float::with_val(args.bits, c.im),
                        )
                    }),
                )
            })
            .collect();
        let high_mu: Vec<_> = mu_points
            .iter()
            .map(|c| {
                Complex::new(
                    Float::with_val(args.bits, c.re),
                    Float::with_val(args.bits, c.im),
                )
            })
            .collect();
        let high_values = polynomial_values(
            &mut high,
            &high_parameters,
            lanes,
            &high_points,
            &high_mu,
            Complex::new(proto.zero(), proto.zero()),
        )?;
        let mut worst = [0_f64; 3];
        for ((((lane, q), direct), high), mu_squared) in points
            .iter()
            .zip(&f64_values)
            .zip(&high_values)
            .zip(&mu_points)
        {
            let (value, terms) = tensor_value(
                &tensor_plan,
                &tensor[lane * ncoeff..(lane + 1) * ncoeff],
                q,
                *mu_squared,
            );
            let high = to_f64(high);
            let norm = high.norm().re;
            if !value.re.is_finite() || !value.im.is_finite() || norm == 0. {
                return Err("tensor verification produced a nonfinite or zero reference".into());
            }
            worst[0] = worst[0].max((value - high).norm().re / norm);
            worst[1] = worst[1].max((value - *direct).norm().re / direct.norm().re);
            worst[2] = worst[2].max(terms / norm);
        }
        json!({
            "points_per_lane": args.check_points,
            "q_component_scale": scale,
            "max_relative_error_vs_high_precision_polynomial": worst[0],
            "max_relative_error_vs_f64_polynomial": worst[1],
            "max_term_cancellation": worst[2],
        })
    } else {
        Value::Null
    };

    let jit = if args.symjit {
        let start = Instant::now();
        let mut evaluator = f64::compile_symjit_coefficients(&plan)?;
        let compile_time = start.elapsed();
        let inputs: Vec<_> = selection
            .iter()
            .flat_map(|&index| prepared.samples[index].parameters.iter())
            .map(|atom| numerical(atom, &0_f64, "binary64 parameter"))
            .collect::<Result<Vec<_>>>()?;
        let values = evaluator.evaluate_batch(&inputs, lanes)?.to_vec();
        let mut jit_tensor = vec![Complex::new(0., 0.); lanes * ncoeff];
        tensor_plan.scatter(&values, &mut jit_tensor, lanes, &Complex::new(0., 0.))?;
        let mut worst = 0_f64;
        for (jit_row, row) in jit_tensor.chunks(ncoeff).zip(tensor.chunks(ncoeff)) {
            let norm = row.iter().fold(0_f64, |m, c| m.max(c.norm().re));
            for (a, e) in jit_row.iter().zip(row) {
                worst = worst.max((*a - *e).norm().re / norm);
            }
        }
        json!({"compilation_ns": ns(compile_time), "optimization_level": 2,
            "max_tensor_error_relative_to_lane_max": worst})
    } else {
        Value::Null
    };

    fs::write(&args.output, serde_json::to_string(&case)? + "\n")?;
    let report = json!({
        "schema_version": 1,
        "operation": "export-bench",
        "format": case["format"],
        "r2": args.r2.name(),
        "input": args.input,
        "output": args.output,
        "graph": prepared.document["graph"],
        "selected_samples": selection,
        "lanes": lanes,
        "propagators": n,
        "tensor": {
            "rank": tensor_plan.rank(),
            "numerator_degree": plan.rank(),
            "mu_powers": mu_powers,
            "ncoeff": ncoeff,
            "nonzero_slots": tensor_plan.entries().len(),
            "coefficient_heads": heads,
            "external_parameters": prepared.parameters.len(),
            "factored": args.factor,
            "ordering": "original Ninja/MadLoop tensor_layout (ninjars::fast::tensor_layout)",
            "evaluation": format!("exact inputs at {} bits, rounded once to binary64", args.bits),
        },
        "s_matrix": match args.s_matrix { SMatrix::Exact => "exact", SMatrix::None => "none" },
        "s_matrix_entries_snapped_to_zero": snapped,
        "timings_ns": {
            "prepare": ns(prepare_time),
            "coefficient_planning": ns(planning_time),
            "coefficient_factoring": ns(factoring_time),
            "tensor_plan": ns(tensor_plan_time),
            "high_precision_evaluation": ns(evaluation_time),
        },
        "verification": verification,
        "symjit": jit,
    });
    let report = serde_json::to_string_pretty(&report)? + "\n";
    if let Some(path) = args.report {
        fs::write(path, report)?;
    } else {
        print!("{report}");
    }
    Ok(())
}
