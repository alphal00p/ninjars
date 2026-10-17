//! CLI-owned coefficient binding. JIT code never enters the reduction engine.
//!
//! The caller keeps raw phase-space parameters, calls `evaluate_batch` inside
//! each timed end-to-end evaluation, then binds its returned coefficient rows to
//! the existing polynomial numerator. Reduction-only timings may prepare the
//! same rows outside timing for both implementations.

use ninjars_graph::PolynomialCoefficientPlan;
use oneloop::NativeFloat;
use symbolica::domains::float::{Complex, DoubleFloat, Float};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub(crate) trait CoefficientBatch<T> {
    fn evaluate_batch(&mut self, input: &[Complex<T>], rows: usize) -> Result<&[Complex<T>]>;
}

/// CLI precision dispatch, leaving every public reduction bound unchanged.
/// The only JIT implementation is typed binary64; other carriers are rejected.
pub(crate) trait CoefficientScalar: NativeFloat {
    fn compile_symjit_coefficients(
        plan: &PolynomialCoefficientPlan,
    ) -> Result<Box<dyn CoefficientBatch<Self>>>;
}

#[cfg(feature = "numerator-jit")]
struct F64Coefficients {
    evaluator: ninjars_graph::SymJitCoefficientEvaluator,
    columns: usize,
    values: Vec<Complex<f64>>,
}

#[cfg(feature = "numerator-jit")]
impl CoefficientBatch<f64> for F64Coefficients {
    fn evaluate_batch(&mut self, input: &[Complex<f64>], rows: usize) -> Result<&[Complex<f64>]> {
        let entries = rows
            .checked_mul(self.columns)
            .ok_or("coefficient batch size overflow")?;
        self.values.resize(entries, Complex::new(0., 0.));
        self.evaluator
            .evaluate_batch(input, &mut self.values, rows)?;
        Ok(&self.values)
    }
}

/// Symbolica's interpreted evaluator of every coefficient head, for the fast
/// engine's tensor numerator under `--numerator-evaluator symbolica`.
pub(crate) struct InterpretedCoefficients {
    evaluator: ninjars_graph::CoefficientEvaluator<f64>,
    columns: usize,
    values: Vec<Complex<f64>>,
}

impl InterpretedCoefficients {
    pub(crate) fn new(plan: &PolynomialCoefficientPlan) -> Result<Self> {
        Ok(Self {
            evaluator: plan.compile(0_f64)?,
            columns: plan.coefficient_expressions().len(),
            values: Vec::new(),
        })
    }
}

impl CoefficientBatch<f64> for InterpretedCoefficients {
    fn evaluate_batch(&mut self, input: &[Complex<f64>], rows: usize) -> Result<&[Complex<f64>]> {
        let entries = rows
            .checked_mul(self.columns)
            .ok_or("coefficient batch size overflow")?;
        self.values.resize(entries, Complex::new(0., 0.));
        self.evaluator
            .evaluate_batch(input, &mut self.values, rows)?;
        Ok(&self.values)
    }
}

impl CoefficientScalar for f64 {
    fn compile_symjit_coefficients(
        plan: &PolynomialCoefficientPlan,
    ) -> Result<Box<dyn CoefficientBatch<Self>>> {
        #[cfg(feature = "numerator-jit")]
        {
            Ok(Box::new(F64Coefficients {
                evaluator: plan.compile_symjit_o2()?,
                columns: plan.coefficient_expressions().len(),
                values: Vec::new(),
            }))
        }
        #[cfg(not(feature = "numerator-jit"))]
        {
            let _ = plan;
            Err("symjit-o2 requires rebuilding the CLI with --features numerator-jit".into())
        }
    }
}

macro_rules! unsupported_jit {
    ($scalar:ty, $name:literal) => {
        impl CoefficientScalar for $scalar {
            fn compile_symjit_coefficients(
                _: &PolynomialCoefficientPlan,
            ) -> Result<Box<dyn CoefficientBatch<Self>>> {
                Err(concat!(
                    "symjit-o2 supports f64 only; ",
                    $name,
                    " requires the symbolica numerator evaluator"
                )
                .into())
            }
        }
    };
}
unsupported_jit!(DoubleFloat, "DoubleFloat");
unsupported_jit!(Float, "arbitrary precision");

#[cfg(test)]
mod tests {
    use super::*;
    use symbolica::parse;

    fn plan() -> PolynomialCoefficientPlan {
        PolynomialCoefficientPlan::new(
            parse!("3*a*q0^2+mu2"),
            [
                parse!("q0"),
                parse!("q1"),
                parse!("q2"),
                parse!("q3"),
                parse!("mu2"),
            ],
            vec![parse!("a")],
        )
        .unwrap()
    }

    #[test]
    fn higher_precisions_are_explicitly_rejected() {
        let plan = plan();
        assert!(
            DoubleFloat::compile_symjit_coefficients(&plan)
                .err()
                .unwrap()
                .to_string()
                .contains("f64 only")
        );
        assert!(
            Float::compile_symjit_coefficients(&plan)
                .err()
                .unwrap()
                .to_string()
                .contains("f64 only")
        );
    }

    #[cfg(feature = "numerator-jit")]
    #[test]
    fn jit_binding_reuses_lane_coefficients_and_handles_empty_batches() {
        let plan = plan();
        let mut evaluator = f64::compile_symjit_coefficients(&plan).unwrap();
        let input = [
            Complex::new(2., 1.),
            Complex::new(0., 0.),
            Complex::new(-1., 2.),
        ];
        let output = evaluator.evaluate_batch(&input, 3).unwrap().to_vec();
        let mut control = plan.compile(0.).unwrap();
        let mut expected = vec![Complex::new(0., 0.); output.len()];
        control.evaluate_batch(&input, &mut expected, 3).unwrap();
        assert_eq!(output, expected);
        assert!(evaluator.evaluate_batch(&[], 0).unwrap().is_empty());
        assert!(evaluator.evaluate_batch(&[], 1).is_err());
        assert_eq!(evaluator.evaluate_batch(&input, 3).unwrap(), expected);
    }
    #[cfg(feature = "numerator-jit")]
    #[test]
    fn bound_coefficients_match_all_original_polynomial_callbacks() {
        use ninjars::{
            ExpansionKind, ExpansionRequest, FourMomentum, Numerator, NumeratorRequest, SampleId,
        };
        use ninjars_symbolica::PolynomialNumerator;
        use symbolica::domains::float::Real;
        let variables = [
            parse!("q0"),
            parse!("q1"),
            parse!("q2"),
            parse!("q3"),
            parse!("mu2"),
        ];
        let parameters = vec![parse!("a"), parse!("b")];
        let expression = parse!("3*(a+b)^4*(q0^2+q1*q2)+(a-b)*mu2^2+b*q3+5");
        let plan = PolynomialCoefficientPlan::new(
            expression.clone(),
            variables.clone(),
            parameters.clone(),
        )
        .unwrap();
        let mut evaluator = f64::compile_symjit_coefficients(&plan).unwrap();
        let mut direct =
            PolynomialNumerator::new(expression, variables.clone(), parameters, 0.).unwrap();
        let mut bound = PolynomialNumerator::new(
            plan.template().clone(),
            variables,
            plan.coefficient_symbols().to_vec(),
            0.,
        )
        .unwrap();
        let c = |x| Complex::new(x, 0.);
        let vector = |[a, b, c_, d]: [f64; 4]| FourMomentum::from_args(c(a), c(b), c(c_), c(d));
        let vectors = [
            vector([1., 2., 3., 4.]),
            vector([2., -1., 3., -2.]),
            vector([3., 1., -2., 4.]),
            vector([1., 2., -3., -1.]),
        ];
        let beta = [c(2.), c(3.), c(5.)];
        let mu_squared = c(7.);
        // Repeat the same cut identities after a new phase-space binding. The
        // coefficient plan must not turn them into cached physical numerators.
        for input in [
            [c(2.), c(3.), Complex::new(1., 1.), c(-2.), c(-4.), c(4.)],
            [c(1.), c(2.), c(-3.), Complex::new(0., 1.), c(2.), c(-2.)],
        ] {
            direct.set_parameter_batch(&input, 3).unwrap();
            bound
                .set_parameter_batch(evaluator.evaluate_batch(&input, 3).unwrap(), 3)
                .unwrap();
            let requests = [2, 0, 1, 2].map(|lane| NumeratorRequest {
                id: SampleId { lane, sample: 0 },
                q: &vectors[0],
                mu_squared: &mu_squared,
                partition: &[0, 1, 2],
            });
            let mut actual = vec![c(0.); requests.len()];
            let mut expected = actual.clone();
            direct.evaluate_batch(&requests, &mut expected).unwrap();
            bound.evaluate_batch(&requests, &mut actual).unwrap();
            let check = |actual: &[Complex<f64>], expected: &[Complex<f64>]| {
                for (a, e) in actual.iter().zip(expected) {
                    assert!(
                        (*a - *e).norm().re < 1e-10 * (1. + e.norm().re),
                        "{a} != {e}"
                    );
                }
            };
            check(&actual, &expected);
            for (kind, vector_count, parameter_count) in [
                (
                    ExpansionKind::Mu {
                        powers: (0..=4).collect(),
                    },
                    2,
                    0,
                ),
                (
                    ExpansionKind::T3 {
                        powers: (-2..=2)
                            .flat_map(|t| (0..=2).map(move |mu| (t, mu)))
                            .collect(),
                    },
                    3,
                    1,
                ),
                (
                    ExpansionKind::T2 {
                        powers: (-2..=2)
                            .flat_map(|t| {
                                (0..=4).flat_map(move |x| (0..=2).map(move |mu| (t, x, mu)))
                            })
                            .collect(),
                    },
                    4,
                    3,
                ),
            ] {
                let requests = [2, 0, 1, 2].map(|lane| ExpansionRequest {
                    id: SampleId { lane, sample: 0 },
                    partition: &[0, 1],
                    vectors: &vectors[..vector_count],
                    parameters: &beta[..parameter_count],
                });
                let mut actual = vec![c(0.); kind.len() * requests.len()];
                let mut expected = actual.clone();
                direct
                    .expansion_batch(&kind, &requests, &mut expected)
                    .unwrap();
                bound
                    .expansion_batch(&kind, &requests, &mut actual)
                    .unwrap();
                check(&actual, &expected);
            }
        }
    }
}
