//! Development controls for compiling existing reduction instruction graphs.
//! This does not install a production backend or modify amplitude selection.

use super::ExactKernel;
use crate::{
    BubbleSourceProjection, DenominatorExpansion, LaurentDivision, PolynomialNumerator,
    TadpoleGeometryProbe, TadpoleSourceProjection, TriangleBubbleSubtraction,
};
use ninjars::{BatchShape, Complex, Error, Result};
use std::{hint::black_box, time::Instant};
use symbolica::evaluate::{Instruction, JITCompilationSettings, JITCompiledEvaluator};

fn cases() -> Vec<(String, ExactKernel)> {
    let mut cases = vec![
        (
            "t2_denominator".into(),
            DenominatorExpansion::new(2).unwrap().exact,
        ),
        (
            "t3_denominator".into(),
            DenominatorExpansion::new(3).unwrap().exact,
        ),
        ("bubble_source".into(), BubbleSourceProjection::new().exact),
        (
            "tadpole_triangle_source".into(),
            TadpoleSourceProjection::new(3).unwrap().exact,
        ),
        (
            "tadpole_bubble_source".into(),
            TadpoleSourceProjection::new(2).unwrap().exact,
        ),
        (
            "tadpole_geometry_probe".into(),
            TadpoleGeometryProbe::new().exact,
        ),
    ];
    for rank in 1..=4 {
        let division = LaurentDivision::new(3, rank, 2).unwrap();
        cases.push((
            format!("bubble_correction_degree{}", rank - 1),
            TriangleBubbleSubtraction::for_quotient(&division)
                .unwrap()
                .exact,
        ));
    }
    for (n, rank, cut) in [
        (5, 4, 3),
        (5, 4, 2),
        (5, 4, 1),
        (6, 7, 3),
        (6, 7, 2),
        (6, 7, 1),
        (5, 0, 2),
    ] {
        cases.push((
            format!("division_n{n}_r{rank}_cut{cut}"),
            LaurentDivision::new(n, rank, cut).unwrap().exact,
        ));
    }
    cases
}

struct Probe {
    evaluator: Option<JITCompiledEvaluator<Complex<f64>>>,
    inputs: usize,
    outputs: usize,
}

impl Probe {
    fn new(plan: &ExactKernel) -> Result<Self> {
        if std::env::var_os("SYMJIT_TOML").is_some()
            || std::fs::exists("symjit.toml").map_err(|e| Error::Symbolic(e.to_string()))?
        {
            return Err(Error::Unsupported(
                "probe requires no external SymJIT configuration".into(),
            ));
        }
        let evaluator = if plan.expressions.is_empty() {
            None
        } else {
            let exact = plan.evaluator()?;
            // The probe only covers the existing straight-line algebraic plans.
            // Do not silently extrapolate to conditionals or helper bodies.
            let graph = exact.export_instructions();
            if !graph.sub_evaluators.is_empty()
                || !graph.constant_functions.is_empty()
                || graph.instructions.iter().any(|instruction| {
                    !matches!(
                        instruction,
                        Instruction::Add(..)
                            | Instruction::Mul(..)
                            | Instruction::Pow(..)
                            | Instruction::Assign(..)
                    )
                })
            {
                return Err(Error::Unsupported(
                    "probe requires a straight-line reduction graph".into(),
                ));
            }
            Some(
                exact
                    .jit_compile(
                        JITCompilationSettings::new()
                            .optimization_level(2)
                            .direct_translation(false)
                            .with_option("fastmath", "false")
                            .with_option("fast_complex", "true")
                            .with_option("use_threads", "false")
                            .with_option("use_simd", "false")
                            .with_option("simd_branch", "false")
                            .with_option("enable_simd512", "false"),
                    )
                    .map_err(Error::Symbolic)?,
            )
        };
        Ok(Self {
            evaluator,
            inputs: plan.parameters.len(),
            outputs: plan.expressions.len(),
        })
    }

    fn evaluate(
        &mut self,
        input: &[Complex<f64>],
        output: &mut [Complex<f64>],
        rows: usize,
    ) -> Result<()> {
        BatchShape {
            rows,
            columns: self.inputs,
        }
        .validate("probe inputs", input.len())?;
        BatchShape {
            rows,
            columns: self.outputs,
        }
        .validate("probe outputs", output.len())?;
        // Symbolica's generic BatchEvaluator wrapper divides by rows; use the
        // supported direct API only after validating the full dimensions.
        if rows == 0 {
            return Ok(());
        }
        if let Some(evaluator) = &mut self.evaluator {
            if rows == 1 {
                evaluator.evaluate(input, output);
            } else {
                evaluator.batch_evaluate(input, output, rows);
            }
        }
        Ok(())
    }
}

fn inputs(columns: usize, rows: usize) -> Vec<Complex<f64>> {
    (0..rows * columns)
        .map(|i| {
            let lane = i / columns;
            let column = i % columns;
            Complex::new(
                1. + ((column * 3 + lane * 11) % 37) as f64 / 64.,
                ((column * 7 + lane * 5) % 23) as f64 / 128. - 0.125,
            )
        })
        .collect()
}

fn assert_close(name: &str, actual: &[Complex<f64>], expected: &[Complex<f64>]) -> f64 {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .map(|(a, b)| {
            assert!(
                a.re.is_finite() && a.im.is_finite() && b.re.is_finite() && b.im.is_finite(),
                "{name}: nonfinite output"
            );
            let error = (a.re - b.re).hypot(a.im - b.im) / (1. + b.re.hypot(b.im));
            assert!(error <= 1e-12, "{name}: scaled error {error}");
            error
        })
        .fold(0., f64::max)
}

#[test]
fn complex_reduction_jit_matches_interpreter() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Probe>();
    let mut checked = 0;
    let mut maximum: f64 = 0.;
    for (name, plan) in cases() {
        let mut interpreter = plan.compile(&0f64).unwrap();
        let mut probe = Probe::new(&plan).unwrap();
        for rows in [0, 1, 3, 8, 31, 32, 128] {
            let input = inputs(plan.parameters.len(), rows);
            let mut expected = vec![Complex::new(0., 0.); rows * plan.expressions.len()];
            let mut actual = expected.clone();
            interpreter
                .evaluate_batch(&input, &mut expected, rows)
                .unwrap();
            probe.evaluate(&input, &mut actual, rows).unwrap();
            maximum = maximum.max(assert_close(&name, &actual, &expected));
            // Reuse the same compiled program with a lane permutation.
            let permuted: Vec<_> = input
                .chunks(plan.parameters.len())
                .rev()
                .flatten()
                .copied()
                .collect();
            interpreter
                .evaluate_batch(&permuted, &mut expected, rows)
                .unwrap();
            probe.evaluate(&permuted, &mut actual, rows).unwrap();
            maximum = maximum.max(assert_close(&name, &actual, &expected));
            checked += actual.len() * 2;
        }
        let mut sentinel = vec![Complex::new(13., -7.); plan.expressions.len()];
        let before = sentinel.clone();
        assert!(probe.evaluate(&[], &mut sentinel, 1).is_err());
        assert_eq!(sentinel, before);
    }
    println!("JIT_REDUCTION_CONTROL components={checked} max_scaled_error={maximum}");
}

#[test]
#[ignore = "diagnostic microbenchmark; run optimized, alone, with --nocapture"]
#[allow(
    clippy::assertions_on_constants,
    reason = "ignored diagnostic must compile in debug but refuse debug timing at runtime"
)]
fn reduction_jit_microbenchmark() {
    assert!(
        !cfg!(debug_assertions),
        "run this diagnostic with --release"
    );
    let mut records = Vec::new();
    for (name, plan) in cases() {
        if plan.expressions.is_empty() {
            continue;
        }
        let evaluator = plan.evaluator().unwrap();
        let instructions = evaluator.export_instructions().instructions.len();
        let mut interpreter = PolynomialNumerator::map_evaluator(evaluator, &0f64);
        let setup = Instant::now();
        let mut probe = Probe::new(&plan).unwrap();
        let setup_ns = setup.elapsed().as_nanos();
        for rows in [1, 8, 32, 128, 1024] {
            let input = inputs(plan.parameters.len(), rows);
            let mut interpreted = vec![Complex::new(0., 0.); rows * plan.expressions.len()];
            let mut compiled = interpreted.clone();
            interpreter
                .evaluate_batch(&input, &mut interpreted, rows)
                .unwrap();
            probe.evaluate(&input, &mut compiled, rows).unwrap();
            let error = assert_close(&name, &compiled, &interpreted);
            let rounds = 32768usize.div_ceil(rows);
            let mut blocks = Vec::new();
            for block in 0..6 {
                let mut times = [0u128; 2];
                for mode in if block % 2 == 0 { [0, 1] } else { [1, 0] } {
                    let start = Instant::now();
                    for _ in 0..rounds {
                        if mode == 0 {
                            interpreter
                                .evaluate_batch(
                                    black_box(&input),
                                    black_box(&mut interpreted),
                                    rows,
                                )
                                .unwrap();
                        } else {
                            probe
                                .evaluate(black_box(&input), black_box(&mut compiled), rows)
                                .unwrap();
                        }
                    }
                    times[mode] = start.elapsed().as_nanos();
                }
                assert_close(&name, &compiled, &interpreted);
                blocks.push(serde_json::json!({"order": if block % 2 == 0 { "interpreter_then_jit" } else { "jit_then_interpreter" }, "interpreter_ns": times[0], "jit_ns": times[1]}));
            }
            let minimum_ns = blocks
                .iter()
                .flat_map(|b| {
                    [
                        b["interpreter_ns"].as_u64().unwrap(),
                        b["jit_ns"].as_u64().unwrap(),
                    ]
                })
                .min()
                .unwrap();
            records.push(serde_json::json!({"plan":name,"rows":rows,"rounds":rounds,"inputs":plan.parameters.len(),"outputs":plan.expressions.len(),"instructions":instructions,"exact_evaluator_export_and_jit_setup_ns":setup_ns,"minimum_block_ns":minimum_ns,"block_below_20ms":minimum_ns < 20_000_000,"max_scaled_error":error,"blocks":blocks}));
        }
    }
    println!(
        "JIT_REDUCTION_PROBE {}",
        serde_json::json!({"performance_acceptance":false,"execution":"scalar_jit_per_row","workload":"synthetic_repeated_matrix; rows differ within a matrix","scope":"individual finite well-conditioned reduction plans; no numerator or masters", "settings":{"optimization_level":2,"direct_translation":false,"fastmath":false,"fast_complex":true,"use_threads":false,"use_simd":false,"simd_branch":false,"enable_simd512":false}, "records":records})
    );
}
