//! Independent on-shell constraints for four/five cuts and their derivatives.

use ninjars::{
    BoxCut, BoxCutRequest, BoxSampleRequest, Complex, CutFailure, CutOutcome, DoubleFloat, Float,
    FloatLike, FourMomentum, HighCutProvider, PentagonCutRequest, Real, SampleId, SingleFloat,
};
use ninjars_symbolica::{SymbolicaDualHighCuts, SymbolicaHighCuts};
use symbolica::{
    domains::{
        dual::{DualNumber, HyperDual},
        float::RealLike,
        rational::Rational,
    },
    evaluate::EvaluationDomain,
};

#[allow(clippy::suspicious_arithmetic_impl, clippy::suspicious_op_assign_impl)]
mod static_dual {
    symbolica::create_hyperdual_from_components!(
        HighCutDual,
        [[0, 0], [1, 0], [0, 1], [2, 0], [1, 1]]
    );
}
fn c(v: f64) -> Complex<f64> {
    Complex::new(v, 0.)
}
fn vector<T: FloatLike>(p: &T, values: [i64; 4]) -> FourMomentum<T> {
    FourMomentum::from_args(
        p.from_i64(values[0]),
        p.from_i64(values[1]),
        p.from_i64(values[2]),
        p.from_i64(values[3]),
    )
}
fn check_box<T: FloatLike>(
    cut: &BoxCut<T>,
    shifts: &[FourMomentum<T>],
    masses: &[T],
    check: &impl Fn(T),
) {
    let centered = cut.a.clone() + shifts[0].clone();
    check(centered.dot(&cut.v_orth));
    check(cut.v_orth.dot(&cut.v_orth) - &cut.v_orth_squared);
    check(masses[0].clone() - centered.dot(&centered) - &cut.beta);
    for shift in &shifts[1..] {
        check((shift.clone() - shifts[0].clone()).dot(&cut.v_orth));
    }
}
fn check_on_shell<T: FloatLike>(
    q: &FourMomentum<T>,
    mu: &T,
    shifts: &[FourMomentum<T>],
    masses: &[T],
    check: &impl Fn(T),
) {
    for (shift, mass) in shifts.iter().zip(masses) {
        let p = q.clone() + shift.clone();
        check(p.dot(&p) - mass - mu);
    }
}

#[test]
fn all_box_coordinate_charts_complex_roots_and_lane_failures() {
    let mut provider = SymbolicaHighCuts::new(0.).unwrap();
    let origin = vector(&c(0.), [1, -2, 3, 4]);
    let axes = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
    let mut points: Vec<[FourMomentum<Complex<f64>>; 4]> = (0..4)
        .map(|free| {
            let mut values = vec![origin];
            values.extend(
                (0..4)
                    .filter(|&axis| axis != free)
                    .map(|axis| origin + vector(&c(0.), axes[axis])),
            );
            values.try_into().unwrap()
        })
        .collect();
    points.push([
        origin,
        origin + vector(&c(0.), [1, 0, 0, 1]),
        origin + vector(&c(0.), [0, 1, 0, 0]),
        origin + vector(&c(0.), [0, 0, 1, 0]),
    ]); // null normal
    points.push([origin; 4]); // rank deficient
    let mut complex = points[0];
    complex[2].spatial.px = Complex::new(-2., 0.125);
    points.push(complex);
    let mut nonfinite = points[1];
    nonfinite[2].spatial.py = c(f64::NAN);
    points.push(nonfinite);
    let masses = [c(1.), Complex::new(2., -0.125), c(3.), c(4.)];
    let check = |value: Complex<f64>| assert!(value.norm().re < 1e-9, "residual {value}");
    let tail: Vec<_> = (0..19).map(|i| (3 * i + 1) % points.len()).collect();
    for order in [
        &[7, 2, 5, 0, 6, 1, 4, 3][..],
        tail.as_slice(),
        &[0][..],
        &[][..],
        &[3, 6][..],
    ] {
        let requests: Vec<_> = order
            .iter()
            .enumerate()
            .map(|(lane, &source)| BoxCutRequest {
                id: SampleId { lane, sample: 23 },
                shifts: std::array::from_fn(|i| &points[source][i]),
                masses_squared: std::array::from_fn(|i| &masses[i]),
            })
            .collect();
        let mut output = vec![CutOutcome::Unstable(CutFailure::NonFiniteResult); requests.len()];
        provider.box_batch(&requests, &mut output).unwrap();
        for (lane, (&source, out)) in order.iter().zip(&output).enumerate() {
            if matches!(source, 4 | 5) {
                assert!(matches!(
                    out,
                    CutOutcome::Unstable(CutFailure::DegenerateGram)
                ));
                continue;
            }
            if source == 7 {
                assert!(matches!(
                    out,
                    CutOutcome::Unstable(CutFailure::NonFiniteInput)
                ));
                continue;
            }
            let CutOutcome::Stable(cut) = out else {
                panic!("source{source}: {out:?}")
            };
            check_box(cut, &points[source], &masses, &check);
            let mus = [c(0.), c(-20.), Complex::new(2., 0.25), -cut.beta];
            let samples: Vec<_> = mus
                .iter()
                .enumerate()
                .map(|(sample, mu_squared)| BoxSampleRequest {
                    id: SampleId { lane, sample },
                    cut,
                    mu_squared,
                })
                .collect();
            let mut pairs = vec![CutOutcome::Unstable(CutFailure::NonFiniteResult); samples.len()];
            provider.sample_box_batch(&samples, &mut pairs).unwrap();
            for (sample, result) in pairs.iter().enumerate() {
                if sample == 3 {
                    assert!(matches!(
                        result,
                        CutOutcome::Unstable(CutFailure::SingularParameter)
                    ));
                    continue;
                }
                let CutOutcome::Stable(pair) = result else {
                    panic!("sample failed: {result:?}")
                };
                for q in pair {
                    check_on_shell(q, &mus[sample], &points[source], &masses, &check);
                }
                check((pair[0] + pair[1] - cut.a * c(2.)).temporal.value);
            }
        }
    }
    let request = BoxCutRequest {
        id: SampleId { lane: 0, sample: 0 },
        shifts: std::array::from_fn(|i| &points[0][i]),
        masses_squared: std::array::from_fn(|i| &masses[i]),
    };
    let mut sentinel = vec![CutOutcome::Unstable(CutFailure::SingularParameter); 2];
    assert!(provider.box_batch(&[request], &mut sentinel).is_err());
    assert!(
        sentinel
            .iter()
            .all(|v| matches!(v, CutOutcome::Unstable(CutFailure::SingularParameter)))
    );
    // A strictly negative real discriminant has complex roots; it is not a
    // geometry failure. Here the exact center gives beta=33/4 and v²=1.
    let real_masses = [c(1.), c(2.), c(3.), c(4.)];
    let mut box_out = [CutOutcome::Unstable(CutFailure::NonFiniteResult)];
    provider
        .box_batch(
            &[BoxCutRequest {
                id: SampleId { lane: 0, sample: 0 },
                shifts: std::array::from_fn(|i| &points[0][i]),
                masses_squared: std::array::from_fn(|i| &real_masses[i]),
            }],
            &mut box_out,
        )
        .unwrap();
    let CutOutcome::Stable(cut) = &box_out[0] else {
        panic!("real box")
    };
    assert_eq!(cut.beta, c(33. / 4.));
    assert_eq!(cut.v_orth_squared, c(1.));
    let mu = c(-9.);
    let mut sample = [CutOutcome::Unstable(CutFailure::NonFiniteResult)];
    provider
        .sample_box_batch(
            &[BoxSampleRequest {
                id: SampleId { lane: 0, sample: 0 },
                cut,
                mu_squared: &mu,
            }],
            &mut sample,
        )
        .unwrap();
    let CutOutcome::Stable(pair) = &sample[0] else {
        panic!("negative real discriminant")
    };
    assert!(pair[0].temporal.value.im > 0.);
    for q in pair {
        check_on_shell(q, &mu, &points[0], &real_masses, &check);
    }
}

#[test]
fn pentagon_exact_linear_solution_and_zero_mu_failure() {
    let mut provider = SymbolicaHighCuts::new(0.).unwrap();
    let shifts = [
        [1, -2, 3, 4],
        [2, -2, 3, 4],
        [1, -1, 3, 4],
        [1, -2, 4, 4],
        [1, -2, 3, 5],
    ]
    .map(|v| vector(&c(0.), v));
    let mut complex = shifts;
    complex[3].spatial.px = Complex::new(-2., 0.25);
    let mut singular = shifts;
    singular[4] = singular[3];
    let points = [shifts, complex, singular, shifts];
    let masses = [
        [c(1.), c(2.), c(3.), c(4.), c(5.)],
        [Complex::new(1., -0.25), c(2.), c(3.), c(4.), c(5.)],
        [c(1.), c(2.), c(3.), c(4.), c(5.)],
        [c(1.), c(4.), c(0.), c(0.), c(0.)],
    ];
    for order in [&[3, 1, 2, 0][..], &[0][..], &[][..], &[1, 0, 1][..]] {
        let requests: Vec<_> = order
            .iter()
            .enumerate()
            .map(|(lane, &source)| PentagonCutRequest {
                id: SampleId { lane, sample: 41 },
                shifts: std::array::from_fn(|i| &points[source][i]),
                masses_squared: std::array::from_fn(|i| &masses[source][i]),
            })
            .collect();
        let mut output = vec![CutOutcome::Unstable(CutFailure::NonFiniteResult); requests.len()];
        provider.pentagon_batch(&requests, &mut output).unwrap();
        for (&source, result) in order.iter().zip(&output) {
            if source == 2 {
                assert!(matches!(
                    result,
                    CutOutcome::Unstable(CutFailure::DegenerateGram)
                ));
                continue;
            }
            if source == 3 {
                assert!(matches!(
                    result,
                    CutOutcome::Unstable(CutFailure::SingularParameter)
                ));
                continue;
            }
            let CutOutcome::Stable(cut) = result else {
                panic!("pentagon{source}: {result:?}")
            };
            check_on_shell(
                &cut.q,
                &cut.mu_squared,
                &points[source],
                &masses[source],
                &|value| assert!(value.norm().re < 1e-10, "{value}"),
            );
            if source == 0 {
                assert_eq!(
                    cut.q,
                    vector(&c(0.), [-1, 0, -5, -6])
                        + FourMomentum::from_args(c(0.), c(0.5), c(0.), c(-0.5))
                );
                assert_eq!(cut.mu_squared, c(-13.5));
            }
        }
    }
}

fn dual_control<D: Real + DualNumber<Complex<f64>>>(prototype: D) {
    let mut provider = SymbolicaDualHighCuts::new(0., prototype.clone()).unwrap();
    let mut shifts: Vec<_> = [
        [1, -2, 3, 4],
        [1, -1, 3, 4],
        [1, -2, 4, 4],
        [1, -2, 3, 5],
        [2, -2, 3, 4],
    ]
    .into_iter()
    .map(|v| vector(&prototype, v))
    .collect();
    shifts[1].temporal.value.values_mut()[1] = c(0.25);
    shifts[2].spatial.pz.values_mut()[2] = c(0.125);
    let mut masses: Vec<_> = (1..=5).map(|v| prototype.from_i64(v)).collect();
    masses[1].values_mut()[1] = c(0.5);
    masses[2].values_mut()[4] = c(0.25);
    let check = |value: D| {
        for (i, v) in value.values().iter().enumerate() {
            assert!(v.norm().re < 1e-8, "component{i}: {v}");
        }
    };
    let mut output = [CutOutcome::Unstable(CutFailure::NonFiniteResult)];
    provider
        .box_batch(
            &[BoxCutRequest {
                id: SampleId { lane: 0, sample: 9 },
                shifts: std::array::from_fn(|i| &shifts[i]),
                masses_squared: std::array::from_fn(|i| &masses[i]),
            }],
            &mut output,
        )
        .unwrap();
    let CutOutcome::Stable(cut) = &output[0] else {
        panic!("Dual box: {output:?}")
    };
    check_box(cut, &shifts[..4], &masses[..4], &check);
    let mut mu = prototype.from_i64(2);
    mu.values_mut()[1] = c(0.25);
    mu.values_mut()[2] = c(0.5);
    let mut sampled = [CutOutcome::Unstable(CutFailure::NonFiniteResult)];
    provider
        .sample_box_batch(
            &[BoxSampleRequest {
                id: SampleId { lane: 0, sample: 0 },
                cut,
                mu_squared: &mu,
            }],
            &mut sampled,
        )
        .unwrap();
    let CutOutcome::Stable(pair) = &sampled[0] else {
        panic!("Dual samples: {sampled:?}")
    };
    for q in pair {
        check_on_shell(q, &mu, &shifts[..4], &masses[..4], &check);
    }
    let mut pentagon = [CutOutcome::Unstable(CutFailure::NonFiniteResult)];
    provider
        .pentagon_batch(
            &[PentagonCutRequest {
                id: SampleId { lane: 0, sample: 1 },
                shifts: std::array::from_fn(|i| &shifts[i]),
                masses_squared: std::array::from_fn(|i| &masses[i]),
            }],
            &mut pentagon,
        )
        .unwrap();
    let CutOutcome::Stable(cut) = &pentagon[0] else {
        panic!("Dual pentagon: {pentagon:?}")
    };
    check_on_shell(&cut.q, &cut.mu_squared, &shifts, &masses, &check);
    // A nonfinite mixed component must be checked even with a finite primal.
    masses[2].values_mut()[4] = c(f64::NAN);
    provider
        .box_batch(
            &[BoxCutRequest {
                id: SampleId { lane: 0, sample: 9 },
                shifts: std::array::from_fn(|i| &shifts[i]),
                masses_squared: std::array::from_fn(|i| &masses[i]),
            }],
            &mut output,
        )
        .unwrap();
    assert!(matches!(
        output[0],
        CutOutcome::Unstable(CutFailure::NonFiniteInput)
    ));
}

#[test]
fn static_and_dynamic_higher_mixed_dual_cut_equations() {
    dual_control(static_dual::HighCutDual { values: [c(0.); 5] });
    dual_control(HyperDual::from_values(
        vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![2, 0], vec![1, 1]],
        vec![c(0.); 5],
    ));
}

fn precision_control<T: Real + RealLike + SingleFloat>(prototype: T, tolerance: f64)
where
    Complex<T>: Real + EvaluationDomain,
{
    let zero = Complex::new(prototype.zero(), prototype.zero());
    let mut provider = SymbolicaHighCuts::new(prototype.clone()).unwrap();
    let mut shifts: Vec<_> = [
        [1, -2, 3, 4],
        [3, -1, 3, 4],
        [4, -2, 4, 4],
        [5, -2, 3, 5],
        [2, -2, 3, 4],
    ]
    .into_iter()
    .map(|v| vector(&zero, v))
    .collect();
    shifts[2].spatial.px.re = prototype.from_rational(&Rational::from((-11, 7)));
    let masses: Vec<_> = (1..=5)
        .map(|n| {
            Complex::new(
                prototype.from_rational(&Rational::from((n, 7))),
                prototype.zero(),
            )
        })
        .collect();
    let check = |v: Complex<T>| assert!(v.norm().re.to_f64() < tolerance, "residual {v}");
    let mut output = [CutOutcome::Unstable(CutFailure::NonFiniteResult)];
    provider
        .box_batch(
            &[BoxCutRequest {
                id: SampleId { lane: 0, sample: 0 },
                shifts: std::array::from_fn(|i| &shifts[i]),
                masses_squared: std::array::from_fn(|i| &masses[i]),
            }],
            &mut output,
        )
        .unwrap();
    let CutOutcome::Stable(cut) = &output[0] else {
        panic!("precision box: {output:?}")
    };
    check_box(cut, &shifts[..4], &masses[..4], &check);
    let mu = Complex::new(
        prototype.from_rational(&Rational::from((2, 7))),
        prototype.from_rational(&Rational::from((1, 11))),
    );
    let mut sampled = [CutOutcome::Unstable(CutFailure::NonFiniteResult)];
    provider
        .sample_box_batch(
            &[BoxSampleRequest {
                id: SampleId { lane: 0, sample: 0 },
                cut,
                mu_squared: &mu,
            }],
            &mut sampled,
        )
        .unwrap();
    let CutOutcome::Stable(pair) = &sampled[0] else {
        panic!("precision samples: {sampled:?}")
    };
    for q in pair {
        check_on_shell(q, &mu, &shifts[..4], &masses[..4], &check);
    }
    let mut output = [CutOutcome::Unstable(CutFailure::NonFiniteResult)];
    provider
        .pentagon_batch(
            &[PentagonCutRequest {
                id: SampleId { lane: 0, sample: 0 },
                shifts: std::array::from_fn(|i| &shifts[i]),
                masses_squared: std::array::from_fn(|i| &masses[i]),
            }],
            &mut output,
        )
        .unwrap();
    let CutOutcome::Stable(cut) = &output[0] else {
        panic!("precision pentagon: {output:?}")
    };
    check_on_shell(&cut.q, &cut.mu_squared, &shifts, &masses, &check);
    // Repeat the full geometry and root equations with higher/mixed components
    // at the same precision; every derivative, not just its primal, is checked.
    let shape = vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![2, 0], vec![1, 1]];
    let dual_zero = HyperDual::from_values(shape.clone(), vec![zero.clone(); 5]);
    let lift = |v: Complex<T>| {
        let mut d = dual_zero.zero();
        d.values[0] = v;
        d
    };
    let mut dual_shifts: Vec<_> = shifts.iter().cloned().map(|p| p.map(lift)).collect();
    let mut dual_masses: Vec<_> = masses.iter().cloned().map(lift).collect();
    dual_shifts[1].temporal.value.values[1] = zero.from_i64(1);
    dual_shifts[2].spatial.pz.values[2] = zero.from_i64(1);
    dual_masses[2].values[4] = zero.from_i64(1);
    let mut provider = SymbolicaDualHighCuts::new(prototype, dual_zero.clone()).unwrap();
    let mut output = [CutOutcome::Unstable(CutFailure::NonFiniteResult)];
    provider
        .box_batch(
            &[BoxCutRequest {
                id: SampleId { lane: 0, sample: 0 },
                shifts: std::array::from_fn(|i| &dual_shifts[i]),
                masses_squared: std::array::from_fn(|i| &dual_masses[i]),
            }],
            &mut output,
        )
        .unwrap();
    let CutOutcome::Stable(cut) = &output[0] else {
        panic!("precision Dual box")
    };
    let check_dual = |d: HyperDual<Complex<T>>| {
        for component in d.values {
            check(component);
        }
    };
    check_box(cut, &dual_shifts[..4], &dual_masses[..4], &check_dual);
    let mut dual_mu = lift(mu);
    dual_mu.values[1] = zero.from_i64(1);
    let mut sampled = [CutOutcome::Unstable(CutFailure::NonFiniteResult)];
    provider
        .sample_box_batch(
            &[BoxSampleRequest {
                id: SampleId { lane: 0, sample: 0 },
                cut,
                mu_squared: &dual_mu,
            }],
            &mut sampled,
        )
        .unwrap();
    let CutOutcome::Stable(pair) = &sampled[0] else {
        panic!("precision Dual sample")
    };
    for q in pair {
        check_on_shell(
            q,
            &dual_mu,
            &dual_shifts[..4],
            &dual_masses[..4],
            &check_dual,
        );
    }
    let mut pentagon = [CutOutcome::Unstable(CutFailure::NonFiniteResult)];
    provider
        .pentagon_batch(
            &[PentagonCutRequest {
                id: SampleId { lane: 0, sample: 0 },
                shifts: std::array::from_fn(|i| &dual_shifts[i]),
                masses_squared: std::array::from_fn(|i| &dual_masses[i]),
            }],
            &mut pentagon,
        )
        .unwrap();
    let CutOutcome::Stable(cut) = &pentagon[0] else {
        panic!("precision Dual pentagon")
    };
    check_on_shell(
        &cut.q,
        &cut.mu_squared,
        &dual_shifts,
        &dual_masses,
        &check_dual,
    );
    let malformed = HyperDual::from_values(vec![vec![0], vec![1]], vec![zero.clone(); 2]);
    let mut output = vec![CutOutcome::Unstable(CutFailure::NonFiniteResult); 2];
    provider
        .box_batch(
            &[
                BoxCutRequest {
                    id: SampleId { lane: 0, sample: 0 },
                    shifts: std::array::from_fn(|i| &dual_shifts[i]),
                    masses_squared: [
                        &dual_masses[0],
                        &malformed,
                        &dual_masses[2],
                        &dual_masses[3],
                    ],
                },
                BoxCutRequest {
                    id: SampleId { lane: 1, sample: 0 },
                    shifts: std::array::from_fn(|i| &dual_shifts[i]),
                    masses_squared: std::array::from_fn(|i| &dual_masses[i]),
                },
            ],
            &mut output,
        )
        .unwrap();
    assert!(matches!(
        output[0],
        CutOutcome::Unstable(CutFailure::DerivativeShape)
    ));
    assert!(matches!(output[1], CutOutcome::Stable(_)));
}
#[test]
fn double_double_and_arbitrary_precision_high_cuts() {
    precision_control(DoubleFloat::from(0.), 1e-25);
    precision_control(Float::new(256), 1e-65);
}
