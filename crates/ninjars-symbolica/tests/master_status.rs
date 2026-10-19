use ninjars::{
    Complex, DoubleFloat, Float, FloatLike, MasterFailure, MasterIntegral, MasterProvider,
    MasterRequest, MasterStatus,
};
use ninjars_symbolica::{OneLoopNativeMasters, SymbolicaDualMasters, SymbolicaMasters};
use oneloop::{NativeFloat, NativeTensorEvaluator, TensorIntegral};
use symbolica::{
    domains::{
        dual::{DualNumber, DualNumberStructure, HyperDual},
        float::{Real, RealLike, SingleFloat},
        rational::Rational,
    },
    evaluate::EvaluationDomain,
};

#[allow(clippy::suspicious_arithmetic_impl, clippy::suspicious_op_assign_impl)]
mod static_dual {
    symbolica::create_hyperdual_from_components!(
        StatusDual,
        [[0, 0], [1, 0], [0, 1], [2, 0], [1, 1]]
    );
}

fn c(value: f64) -> Complex<f64> {
    Complex::new(value, 0.)
}

fn close(actual: Complex<f64>, expected: Complex<f64>) {
    let tolerance = 1e-9 * (1. + expected.re.abs() + expected.im.abs());
    assert!(
        (actual.re - expected.re).abs() < tolerance,
        "{actual} != {expected}"
    );
    assert!(
        (actual.im - expected.im).abs() < tolerance,
        "{actual} != {expected}"
    );
}

#[test]
fn compacted_bubble_batches_preserve_branch_results_and_failed_rows() {
    let mut provider = SymbolicaMasters::new(0.);
    let success = MasterStatus::Success;
    let invalid_mass = MasterStatus::Failed(MasterFailure::InvalidMass);
    let nonfinite = MasterStatus::Failed(MasterFailure::NonFiniteInput);
    let invalid_scale = MasterStatus::Failed(MasterFailure::InvalidScale);
    let nonreal = MasterStatus::Failed(MasterFailure::NonRealInvariant);
    let points = [
        (c(-1.), c(1.), c(1.), success, -0.152_044_704_820_020_3),
        (c(-1.), c(0.), c(1.), success, 2.),
        (c(0.), c(1.), c(1.), success, 0.),
        (c(-1.), Complex::new(1., 0.01), c(1.), invalid_mass, 0.),
        (c(f64::NAN), c(1.), c(1.), nonfinite, 0.),
        (
            c(-1.),
            c(1.),
            Complex::new(1., f64::from_bits(1)),
            invalid_scale,
            0.,
        ),
        (
            Complex::new(-1., f64::from_bits(1)),
            c(1.),
            c(1.),
            nonreal,
            0.,
        ),
    ];
    let tail: Vec<_> = (0..31).map(|i| (3 * i + 2) % points.len()).collect();
    for order in [
        &[3, 0, 4, 2, 5, 1, 6][..],
        tail.as_slice(),
        &[5][..],
        &[][..],
        &[1][..],
        &[2, 0, 1][..],
    ] {
        let invariants: Vec<_> = order.iter().map(|&i| points[i].0).collect();
        let masses: Vec<_> = order.iter().flat_map(|&i| [points[i].1; 2]).collect();
        let scales: Vec<_> = order.iter().map(|&i| points[i].2).collect();
        let mut output: Vec<_> = (0..order.len()).map(|i| [c(101. + i as f64); 3]).collect();
        let before = output.clone();
        let mut status = vec![MasterStatus::Failed(MasterFailure::DerivativeShape); order.len()];
        provider
            .evaluate_batch(
                MasterRequest {
                    integral: MasterIntegral::B0,
                    invariants: &invariants,
                    masses_squared: &masses,
                    scales_squared: &scales,
                },
                &mut output,
                &mut status,
            )
            .unwrap();
        for (row, &source) in order.iter().enumerate() {
            assert_eq!(status[row], points[source].3);
            if status[row] == success {
                for (actual, expected) in output[row].iter().zip([points[source].4, 1., 0.]) {
                    close(*actual, c(expected));
                }
            } else {
                assert_eq!(output[row], before[row]);
            }
        }
    }
}

#[test]
fn global_errors_and_strict_failures_leave_all_buffers_unchanged() {
    let mut provider = SymbolicaMasters::new(0.);
    let masses = [c(2.), c(3.)];
    let scales = [c(2.), c(3.)];
    let initial_output = [[Complex::new(71., -19.); 3]; 2];
    let initial_status = [MasterStatus::Failed(MasterFailure::DerivativeShape); 2];
    for (integral, supplied_masses, status_rows, output_rows) in [
        (MasterIntegral::A0, &masses[..1], 2, 2),
        (MasterIntegral::A0, &masses[..], 1, 2),
        (MasterIntegral::A0, &masses[..], 2, 1),
        (MasterIntegral::An { rank: 5 }, &masses[..], 2, 2),
    ] {
        let mut output = initial_output;
        let mut status = initial_status;
        assert!(
            provider
                .evaluate_batch(
                    MasterRequest {
                        integral,
                        invariants: &[],
                        masses_squared: supplied_masses,
                        scales_squared: &scales
                    },
                    &mut output[..output_rows],
                    &mut status[..status_rows],
                )
                .is_err()
        );
        assert_eq!(output, initial_output);
        assert_eq!(status, initial_status);
    }
    // One domain-invalid row and one finite-input overflow, respectively.
    for (masses, scales) in [
        ([c(2.), Complex::new(3., 1.)], [c(2.), c(1.)]),
        ([c(2.), c(1e308)], [c(2.), c(1e-308)]),
    ] {
        let request = MasterRequest {
            integral: MasterIntegral::A0,
            invariants: &[],
            masses_squared: &masses,
            scales_squared: &scales,
        };
        let mut output = initial_output;
        assert!(provider.evaluate(request, &mut output).is_err());
        assert_eq!(output, initial_output);
        let mut statuses = initial_status;
        provider
            .evaluate_batch(request, &mut output, &mut statuses)
            .unwrap();
        assert_eq!(statuses[0], MasterStatus::Success);
        assert_eq!(
            statuses[1],
            MasterStatus::Failed(if masses[1].im > 0. {
                MasterFailure::InvalidMass
            } else {
                MasterFailure::NonFiniteResult
            })
        );
        assert_eq!(output[0], [c(2.), c(2.), c(0.)]);
        assert_eq!(output[1], initial_output[1]);
    }
}

fn dual_components<D: FloatLike + DualNumber<Complex<f64>>>(zero: D) {
    let dual = |values: [f64; 5]| {
        let mut value = zero.zero();
        value.values_mut().clone_from_slice(&values.map(c));
        value
    };
    let mut provider = SymbolicaDualMasters::new(0., &zero);
    let masses = [
        dual([2., 1., 0., 0., 0.]),
        dual([3., 1., 0., 0., f64::NAN]),
        dual([4., 1., 0., 0., 0.]),
        dual([0., 1., 0., 0., 0.]),
        dual([1e-320, 1., 0., 0., 0.]),
    ];
    let scales = [
        dual([3., 0., 1., 0., 0.]),
        dual([1., 0., 0., 0., 0.]),
        dual([5., 0., 1., 0., 0.]),
        dual([1., 0., 0., 0., 0.]),
        dual([1., 0., 0., 0., 0.]),
    ];
    let statuses = [
        MasterStatus::Success,
        MasterStatus::Failed(MasterFailure::NonFiniteInput),
        MasterStatus::Success,
        MasterStatus::Failed(MasterFailure::UnsupportedDerivativeDomain),
        MasterStatus::Failed(MasterFailure::NonFiniteResult),
    ];
    for order in [&[3, 0, 1, 2, 4][..], &[2][..], &[][..], &[0, 2, 0][..]] {
        let selected_masses: Vec<_> = order.iter().map(|&i| masses[i].clone()).collect();
        let selected_scales: Vec<_> = order.iter().map(|&i| scales[i].clone()).collect();
        let sentinel = dual([91., 92., 93., 94., 95.]);
        let mut output = vec![std::array::from_fn(|_| sentinel.clone()); order.len()];
        let mut status = vec![MasterStatus::Success; order.len()];
        provider
            .evaluate_batch(
                MasterRequest {
                    integral: MasterIntegral::A0,
                    invariants: &[],
                    masses_squared: &selected_masses,
                    scales_squared: &selected_scales,
                },
                &mut output,
                &mut status,
            )
            .unwrap();
        for (row, &source) in order.iter().enumerate() {
            assert_eq!(status[row], statuses[source]);
            if status[row] != MasterStatus::Success {
                for coefficient in &output[row] {
                    assert_eq!(coefficient.values(), sentinel.values());
                }
                continue;
            }
            let mass = masses[source].values()[0].re;
            let scale = scales[source].values()[0].re;
            let expected = [
                mass * (1. - (mass / scale).ln()),
                -(mass / scale).ln(),
                mass / scale,
                -1. / (2. * mass),
                1. / scale,
            ];
            for (actual, expected) in output[row][0].values().iter().zip(expected) {
                close(*actual, c(expected));
            }
            assert_eq!(
                output[row][1].values(),
                &[c(mass), c(1.), c(0.), c(0.), c(0.)]
            );
            assert!(output[row][2].values().iter().all(FloatLike::is_fully_zero));
        }
    }
}

#[test]
fn static_and_dynamic_dual_statuses_check_every_higher_mixed_component() {
    dual_components(static_dual::StatusDual { values: [c(0.); 5] });
    dual_components(HyperDual::from_values(
        vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![2, 0], vec![1, 1]],
        vec![c(0.); 5],
    ));
}

#[test]
fn dynamic_dual_shape_and_domain_failures_are_local_but_buffer_errors_are_global() {
    let shape = vec![vec![0, 0], vec![1, 0], vec![0, 1]];
    let dual = |value| HyperDual::from_values(shape.clone(), vec![c(value), c(0.), c(0.)]);
    let mut provider = SymbolicaDualMasters::new(0., &dual(0.));
    let malformed = HyperDual::from_values(
        vec![vec![0, 0], vec![0, 1], vec![1, 0]],
        vec![c(2.), c(0.), c(0.)],
    );
    let mut truncated = dual(2.);
    truncated.values.pop();
    let masses = [malformed, dual(2.), dual(2.), dual(2.), truncated, dual(2.)];
    let momenta = [dual(-1.), dual(-1.), dual(0.)];
    let scales = [dual(1.), dual(1.), dual(1.)];
    let sentinel = HyperDual::from_values(vec![vec![0]], vec![c(71.)]);
    let mut output = vec![std::array::from_fn(|_| sentinel.clone()); 3];
    let mut statuses = [MasterStatus::Success; 3];
    provider
        .evaluate_batch(
            MasterRequest {
                integral: MasterIntegral::B0,
                invariants: &momenta,
                masses_squared: &masses,
                scales_squared: &scales,
            },
            &mut output,
            &mut statuses,
        )
        .unwrap();
    assert_eq!(
        statuses,
        [
            MasterStatus::Failed(MasterFailure::DerivativeShape),
            MasterStatus::Success,
            MasterStatus::Failed(MasterFailure::DerivativeShape),
        ]
    );
    assert_eq!(
        output[1][0].get_shape(),
        shape.iter().map(Vec::as_slice).collect::<Vec<_>>()
    );
    for row in [0, 2] {
        for value in &output[row] {
            assert_eq!(value.get_shape(), sentinel.get_shape());
            assert_eq!(value.values(), sentinel.values());
        }
    }
    let before: Vec<_> = output
        .iter()
        .flatten()
        .map(|d| {
            (
                d.get_shape()
                    .into_iter()
                    .map(<[usize]>::to_vec)
                    .collect::<Vec<_>>(),
                d.values().to_vec(),
            )
        })
        .collect();
    let previous_status = statuses;
    assert!(
        provider
            .evaluate_batch(
                MasterRequest {
                    integral: MasterIntegral::B0,
                    invariants: &momenta[..2],
                    masses_squared: &masses,
                    scales_squared: &scales
                },
                &mut output,
                &mut statuses,
            )
            .is_err()
    );
    assert_eq!(statuses, previous_status);
    for ((shape, values), current) in before.iter().zip(output.iter().flatten()) {
        assert_eq!(
            current.get_shape(),
            shape.iter().map(Vec::as_slice).collect::<Vec<_>>()
        );
        assert_eq!(current.values(), values);
    }
}

fn precise_statuses<T: NativeFloat>(prototype: T)
where
    Complex<T>: EvaluationDomain,
{
    let tiny = prototype.from_rational(&Rational::from((1, 10)).pow(30));
    let mass = prototype.one() + tiny;
    let complex = |x| Complex::new(x, prototype.zero());
    let masses = [
        complex(mass.clone()),
        complex(prototype.from_i64(2)),
        complex(mass.clone()),
    ];
    let scales = [
        masses[0].clone(),
        complex(prototype.zero()),
        masses[2].clone(),
    ];
    let sentinel = complex(prototype.from_i64(17));
    let mut output = vec![std::array::from_fn(|_| sentinel.clone()); 3];
    let mut statuses = [MasterStatus::Success; 3];
    let mut provider = SymbolicaMasters::new(prototype.clone());
    provider
        .evaluate_batch(
            MasterRequest {
                integral: MasterIntegral::A0,
                invariants: &[],
                masses_squared: &masses,
                scales_squared: &scales,
            },
            &mut output,
            &mut statuses,
        )
        .unwrap();
    assert_eq!(
        statuses,
        [
            MasterStatus::Success,
            MasterStatus::Failed(MasterFailure::InvalidScale),
            MasterStatus::Success
        ]
    );
    for row in [0, 2] {
        assert_eq!(output[row][0], masses[row]);
        assert_eq!(output[row][1], masses[row]);
        assert_ne!(output[row][1].re, prototype.one());
        assert!(output[row][2].is_fully_zero());
    }
    assert!(output[1].iter().all(|value| value == &sentinel));

    let shape = vec![vec![0], vec![1], vec![2]];
    let lift = |value: &Complex<T>| {
        HyperDual::from_values(
            shape.clone(),
            vec![
                value.clone(),
                complex(prototype.one()),
                complex(prototype.zero()),
            ],
        )
    };
    let dual_masses: Vec<_> = masses.iter().map(lift).collect();
    let dual_scales: Vec<_> = scales.iter().map(lift).collect();
    let mut provider = SymbolicaDualMasters::new(prototype.clone(), &dual_masses[0]);
    let mut output = vec![std::array::from_fn(|_| dual_masses[1].clone()); 3];
    provider
        .evaluate_batch(
            MasterRequest {
                integral: MasterIntegral::A0,
                invariants: &[],
                masses_squared: &dual_masses,
                scales_squared: &dual_scales,
            },
            &mut output,
            &mut statuses,
        )
        .unwrap();
    assert_eq!(
        statuses[1],
        MasterStatus::Failed(MasterFailure::InvalidScale)
    );
    for row in [0, 2] {
        for coefficient in &output[row][..2] {
            assert_eq!(coefficient.values(), dual_masses[row].values());
        }
        assert!(output[row][2].values().iter().all(FloatLike::is_fully_zero));
    }
    for coefficient in &output[1] {
        assert_eq!(coefficient.values(), dual_masses[1].values());
    }
}

#[test]
fn double_double_and_arbitrary_precision_statuses_keep_input_digits() {
    precise_statuses(DoubleFloat::default());
    precise_statuses(Float::new(192));
}

#[test]
fn tracked_component_precision_is_retained_in_scalar_and_dual_rows() {
    let value = |bits| Complex::new(Float::with_val(bits, 2), Float::new(bits));
    // Real cancellation deliberately lowers significant precision in Numerica.
    // The absolute error still reflects the original 192-bit operands.
    let prototype = Float::new(192);
    let cancellation = prototype.from_i64(1)
        - prototype.from_rational(&Rational::from((
            999_999_999_999_999_999i64,
            1_000_000_000_000_000_000i64,
        )));
    assert!(cancellation.get_precision() < 192);
    let masses = [
        Complex::new(cancellation, Float::new(2)),
        Complex::new(
            Float::new(53).from_rational(&Rational::from((1, 7))),
            Float::new(17),
        ),
        value(256),
    ];
    let scales = masses.clone();
    let mut provider = SymbolicaMasters::new(Float::new(192));
    let mut output = vec![std::array::from_fn(|_| value(192)); 3];
    let mut statuses = [MasterStatus::Success; 3];
    provider
        .evaluate_batch(
            MasterRequest {
                integral: MasterIntegral::A0,
                invariants: &[],
                masses_squared: &masses,
                scales_squared: &scales,
            },
            &mut output,
            &mut statuses,
        )
        .unwrap();
    assert_eq!(statuses, [MasterStatus::Success; 3]);
    // The A0 pole equals the supplied squared mass exactly, independently of
    // working precision. Check stored precision too: Float equality omits it.
    for (result, mass) in output.iter().zip(&masses) {
        assert_eq!(&result[1], mass);
        assert_eq!(result[1].re.get_precision(), mass.re.get_precision());
        assert_eq!(result[1].im.get_precision(), mass.im.get_precision());
        assert!(result[2].is_fully_zero());
    }
    assert_eq!(output[1][1].re.get_precision(), 53);
    assert_eq!(output[2][1].re.get_precision(), 256);
    let lift = |value: Complex<Float>| {
        HyperDual::from_values(
            vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![2, 0], vec![1, 1]],
            vec![
                value,
                masses[0].clone(),
                masses[1].clone(),
                masses[2].clone(),
                Complex::new(
                    Float::new(31).from_rational(&Rational::from((1, 11))),
                    Float::new(3),
                ),
            ],
        )
    };
    let dual_masses: Vec<_> = masses.iter().cloned().map(lift).collect();
    let dual_scales = dual_masses.clone();
    let mut provider = SymbolicaDualMasters::new(Float::new(192), &dual_masses[0]);
    let mut output = vec![std::array::from_fn(|_| dual_masses[0].clone()); 3];
    provider
        .evaluate_batch(
            MasterRequest {
                integral: MasterIntegral::A0,
                invariants: &[],
                masses_squared: &dual_masses,
                scales_squared: &dual_scales,
            },
            &mut output,
            &mut statuses,
        )
        .unwrap();
    assert_eq!(statuses, [MasterStatus::Success; 3]);
    for (result, mass) in output.iter().zip(&dual_masses) {
        for (actual, expected) in result[1].values().iter().zip(mass.values()) {
            assert_eq!(actual, expected);
            assert_eq!(actual.re.get_precision(), expected.re.get_precision());
            assert_eq!(actual.im.get_precision(), expected.im.get_precision());
        }
    }
}

#[test]
fn tracked_precision_acceptance_does_not_claim_recovered_digits() {
    let rational = Rational::from((1, 7));
    let reference = Float::new(512).from_rational(&rational);
    // Independent analytic A0 finite term at mu_R²=1.
    let expected = reference.clone() * (reference.one() - reference.log());
    let evaluate = |working_bits, input_bits| {
        let mass = Complex::new(
            Float::new(input_bits).from_rational(&rational),
            Float::new(2),
        );
        let scale = Complex::new(Float::with_val(working_bits, 1), Float::new(working_bits));
        let mut provider = SymbolicaMasters::new(Float::new(working_bits));
        let mut output = vec![std::array::from_fn(|_| scale.zero())];
        let mut statuses = [MasterStatus::Failed(MasterFailure::NonFiniteResult)];
        provider
            .evaluate_batch(
                MasterRequest {
                    integral: MasterIntegral::A0,
                    invariants: &[],
                    masses_squared: &[mass],
                    scales_squared: &[scale],
                },
                &mut output,
                &mut statuses,
            )
            .unwrap();
        assert_eq!(statuses, [MasterStatus::Success]);
        output.remove(0)[0].clone()
    };
    let lower = evaluate(320, 53);
    let medium = evaluate(160, 160);
    let higher = evaluate(320, 320);
    let error = |value: &Complex<Float>| (value.re.clone() - &expected).norm().to_f64();
    assert!(
        lower.re.get_precision() < 100,
        "low-precision input must not gain asserted digits"
    );
    assert!(
        error(&lower) > 1e-20,
        "rounded input cannot recover the exact 1/7"
    );
    assert!(error(&medium) < 1e-45);
    assert!(error(&higher) < 1e-90);
    assert!(error(&higher) < error(&medium) * 1e-30);
}

// ---------------------------------------------------------------------------
// B1111 (rank n+2 bubbles) on the native route of fixed-precision carriers.

/// `(p², m0², m1², mu²)` rows of a B1111 batch with their expected statuses:
/// valid rows in several branches (spacelike, zero momentum after a mass swap,
/// complex masses, small momentum) and one row per failure class.
fn b1111_status_points() -> Vec<([Complex<f64>; 4], MasterStatus)> {
    let success = MasterStatus::Success;
    let tiny = f64::from_bits(1);
    vec![
        ([c(-1.), c(1.), c(1.5), c(1.)], success),
        ([c(0.), c(5.), c(2.), c(1.)], success),
        (
            [
                c(3.),
                Complex::new(0.7, -0.03),
                Complex::new(1.4, -0.08),
                c(2.),
            ],
            success,
        ),
        ([c(0.5f64.powi(16)), c(1.), c(1.5), c(1.)], success),
        (
            [c(-1.), Complex::new(1., 0.01), c(1.), c(1.)],
            MasterStatus::Failed(MasterFailure::InvalidMass),
        ),
        (
            [c(f64::NAN), c(1.), c(1.), c(1.)],
            MasterStatus::Failed(MasterFailure::NonFiniteInput),
        ),
        (
            [c(-1.), c(1.), c(1.), Complex::new(1., tiny)],
            MasterStatus::Failed(MasterFailure::InvalidScale),
        ),
        (
            [c(-1.), c(1.), c(1.), c(0.)],
            MasterStatus::Failed(MasterFailure::InvalidScale),
        ),
        (
            [Complex::new(-1., tiny), c(1.), c(1.), c(1.)],
            MasterStatus::Failed(MasterFailure::NonRealInvariant),
        ),
    ]
}

/// Invariants, squared masses and squared scales of one request.
type RequestRows<T> = (Vec<Complex<T>>, Vec<Complex<T>>, Vec<Complex<T>>);

/// Batches of B1111 rows in permuted, repeated and empty orders through
/// `provider`: failed rows keep their output and status class, and every
/// successful row is bitwise the single-row value (the native family is
/// evaluated once per compacted batch). Returns the single-row values.
fn b1111_batches<T, M>(provider: &mut M, lift: impl Fn(f64) -> T) -> Vec<Option<[Complex<T>; 3]>>
where
    T: NativeFloat,
    M: MasterProvider<Complex<T>>,
{
    let points = b1111_status_points();
    let complex = |z: Complex<f64>| Complex::new(lift(z.re), lift(z.im));
    let sentinel = complex(c(101.));
    let request = |rows: &[usize], buffers: &mut RequestRows<T>| {
        buffers.0 = rows.iter().map(|&i| complex(points[i].0[0])).collect();
        buffers.1 = rows
            .iter()
            .flat_map(|&i| [complex(points[i].0[1]), complex(points[i].0[2])])
            .collect();
        buffers.2 = rows.iter().map(|&i| complex(points[i].0[3])).collect();
    };
    let mut buffers: RequestRows<T> = (Vec::new(), Vec::new(), Vec::new());
    let single: Vec<_> = (0..points.len())
        .map(|i| {
            request(&[i], &mut buffers);
            let mut output = vec![std::array::from_fn(|_| sentinel.clone())];
            let result = provider.evaluate(
                MasterRequest {
                    integral: MasterIntegral::B1111,
                    invariants: &buffers.0,
                    masses_squared: &buffers.1,
                    scales_squared: &buffers.2,
                },
                &mut output,
            );
            // Strict evaluation fails exactly for the failing rows and then
            // leaves the output untouched.
            assert_eq!(result.is_ok(), points[i].1 == MasterStatus::Success);
            if result.is_err() {
                assert!(output[0].iter().all(|v| v == &sentinel));
            }
            result.ok().map(|()| output.pop().unwrap())
        })
        .collect();
    let tail: Vec<_> = (0..31).map(|i| (3 * i + 2) % points.len()).collect();
    for order in [
        &[4, 0, 5, 2, 6, 1, 8, 3, 7][..],
        tail.as_slice(),
        &[6][..],
        &[][..],
        &[3][..],
        &[2, 0, 1, 3, 3, 0][..],
    ] {
        request(order, &mut buffers);
        let mut output: Vec<_> = (0..order.len())
            .map(|i| std::array::from_fn(|_| complex(c(101. + i as f64))))
            .collect();
        let before = output.clone();
        let mut status = vec![MasterStatus::Failed(MasterFailure::DerivativeShape); order.len()];
        provider
            .evaluate_batch(
                MasterRequest {
                    integral: MasterIntegral::B1111,
                    invariants: &buffers.0,
                    masses_squared: &buffers.1,
                    scales_squared: &buffers.2,
                },
                &mut output,
                &mut status,
            )
            .unwrap();
        for (row, &source) in order.iter().enumerate() {
            assert_eq!(status[row], points[source].1, "row {row} = point {source}");
            match &single[source] {
                Some(value) => assert_eq!(&output[row], value, "point {source}"),
                None => assert_eq!(output[row], before[row]),
            }
        }
    }
    single
}

#[test]
fn rank_four_bubble_batches_keep_failed_rows_and_bitwise_lane_values() {
    let binary64 = b1111_batches(&mut SymbolicaMasters::new(0f64), |x| x);
    // OneLoopNativeMasters evaluates the same native family row by row: the
    // same statuses and bits.
    let native = b1111_batches(&mut OneLoopNativeMasters::new().unwrap(), |x| x);
    assert_eq!(binary64, native);
    let double = b1111_batches(
        &mut SymbolicaMasters::new(DoubleFloat::default()),
        DoubleFloat::from,
    );
    for (binary64, double) in binary64.iter().zip(&double) {
        assert_eq!(binary64.is_some(), double.is_some());
        if let (Some(binary64), Some(double)) = (binary64, double) {
            let to_f64 = |z: &Complex<DoubleFloat>| Complex::new(z.re.to_f64(), z.im.to_f64());
            close(binary64[0], to_f64(&double[0]));
            assert_eq!(binary64[1], c(0.2));
            assert_eq!(to_f64(&double[1]), c(0.2));
        }
    }
}

/// `SymbolicaMasters<Float>` serves B1111 from the native family as well:
/// the same statuses, untouched failed rows and bitwise batch/single-row
/// values, and the binary64 values to binary64 accuracy.
#[test]
fn arbitrary_precision_rank_four_bubble_batches_keep_failed_rows() {
    let binary64 = b1111_batches(&mut SymbolicaMasters::new(0f64), |x| x);
    let float = b1111_batches(&mut SymbolicaMasters::new(Float::new(256)), |x| {
        Float::with_val(256, x)
    });
    for (binary64, float) in binary64.iter().zip(&float) {
        assert_eq!(binary64.is_some(), float.is_some());
        if let (Some(binary64), Some(float)) = (binary64, float) {
            let to_f64 = |z: &Complex<Float>| Complex::new(z.re.to_f64(), z.im.to_f64());
            close(binary64[0], to_f64(&float[0]));
            assert_eq!(to_f64(&float[1]), c(0.2));
            assert!(float[2].is_fully_zero());
        }
    }
}

/// The native B1111 of `[p², m0², m1², mu²]` at `bits` working bits.
fn native_b1111(point: &[Complex<Float>; 4], bits: u32) -> [Complex<Float>; 3] {
    let mut native =
        NativeTensorEvaluator::<Float>::with_binary_precision(TensorIntegral::B1111, bits).unwrap();
    let zero = Complex::new(Float::new(bits), Float::new(bits));
    let mut family = vec![zero; TensorIntegral::B1111.outputs()];
    native.evaluate(point, &mut family).unwrap();
    std::array::from_fn(|k| family[12 + k].clone())
}

fn assert_bitwise(actual: &[Complex<Float>; 3], expected: &[Complex<Float>; 3], what: &str) {
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual, expected, "{what}");
        assert_eq!(
            actual.re.get_precision(),
            expected.re.get_precision(),
            "{what}"
        );
        assert_eq!(
            actual.im.get_precision(),
            expected.im.get_precision(),
            "{what}"
        );
    }
}

/// B1111 rows of a `Float(256)` provider through one strict batch.
fn float_b1111(
    provider: &mut SymbolicaMasters<Float>,
    rows: &[[Complex<Float>; 4]],
) -> Vec<[Complex<Float>; 3]> {
    let zero = Complex::new(Float::new(256), Float::new(256));
    let mut output = vec![std::array::from_fn(|_| zero.clone()); rows.len()];
    let masses: Vec<_> = rows
        .iter()
        .flat_map(|row| [row[1].clone(), row[2].clone()])
        .collect();
    provider
        .evaluate(
            MasterRequest {
                integral: MasterIntegral::B1111,
                invariants: &rows.iter().map(|row| row[0].clone()).collect::<Vec<_>>(),
                masses_squared: &masses,
                scales_squared: &rows.iter().map(|row| row[3].clone()).collect::<Vec<_>>(),
            },
            &mut output,
        )
        .unwrap();
    output
}

/// The native family rounds every input to one working precision, so
/// `SymbolicaMasters<Float>` picks it per row: never above the working
/// precision, and never above the bits that an input determines relative to
/// the row's scale (its tracked precision plus `log2(S/|x|)` for invariants
/// and masses, its own tracked precision for `mu²`). Each row is bitwise the
/// native family at that precision, in a batch or alone, also after the
/// evaluator cache has been cleared for too many precisions.
#[test]
fn float_b1111_rows_take_the_precision_their_inputs_determine() {
    let at = |bits: u32, x: f64| Float::with_val(bits, x);
    let real = |x: Float| {
        let zero = Float::new(x.get_precision());
        Complex::new(x, zero)
    };
    let row =
        |p2: Complex<Float>, mu2: Float| [p2, real(at(256, 1.)), real(at(256, 1.5)), real(mu2)];
    let seventh = Float::new(53).from_rational(&Rational::from((-1, 7)));
    let rows = [
        // Exact inputs at the working precision.
        (row(real(at(256, -1.)), at(256, 1.)), 256),
        // A 53-bit invariant of the row's size: 53 + floor(log2(1.5 * 7)).
        (row(real(seventh.clone()), at(256, 1.)), 56),
        // A tiny 8-bit invariant determines 8 + floor(log2(1.5 * 2^100)).
        (row(real(at(8, 0.5f64.powi(100))), at(256, 1.)), 108),
        // Inputs above the working precision are rounded to it.
        (
            [
                real(at(512, -1.)),
                real(at(512, 1.)),
                real(at(512, 1.5)),
                real(at(512, 1.)),
            ],
            256,
        ),
        // mu² limits the row by its own tracked precision.
        (row(real(at(256, -1.)), at(64, 1.)), 64),
        // A tiny, weakly determined width: 4 + floor(log2(1.5 * 2^200)).
        (
            [
                real(at(256, -1.)),
                real(at(256, 1.)),
                Complex::new(at(256, 1.5), at(4, -(0.5f64.powi(200)))),
                real(at(256, 1.)),
            ],
            204,
        ),
    ];
    let points: Vec<_> = rows.iter().map(|(point, _)| point.clone()).collect();
    let mut provider = SymbolicaMasters::new(Float::new(256));
    let batch = float_b1111(&mut provider, &points);
    for (index, ((point, bits), value)) in rows.iter().zip(&batch).enumerate() {
        let expected = native_b1111(point, *bits);
        assert_bitwise(value, &expected, &format!("row {index} at {bits} bits"));
        assert_bitwise(
            &float_b1111(&mut provider, std::slice::from_ref(point))[0],
            &expected,
            &format!("single row {index}"),
        );
        assert!(value[0].re.get_precision() <= *bits, "row {index}");
    }
    // The rounded 1/7 cannot recover the exact invariant's value.
    let exact = native_b1111(
        &row(
            real(Float::new(256).from_rational(&Rational::from((-1, 7)))),
            at(256, 1.),
        ),
        256,
    );
    let difference = (batch[1][0].re.clone() - &exact[0].re).norm().to_f64();
    assert!(difference > 1e-20, "{difference:e}");
    assert!(batch[1][0].re.get_precision() < 100);
    // Equal values give equal results at equal precision.
    assert_bitwise(&batch[3], &batch[0], "rounded 512-bit inputs");

    // More row precisions than the evaluator cache keeps, twice.
    for _ in 0..2 {
        let points: Vec<_> = (40..64)
            .map(|bits| row(real(at(bits, -1.)), at(256, 1.)))
            .collect();
        for ((point, bits), value) in points
            .iter()
            .zip(40..)
            .zip(float_b1111(&mut provider, &points))
        {
            assert_bitwise(&value, &native_b1111(point, bits), &format!("{bits} bits"));
        }
    }
}
