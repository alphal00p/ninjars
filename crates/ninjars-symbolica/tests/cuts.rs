// Symbolica's generated Dual division implementations multiply by an inverse.
#![allow(clippy::suspicious_arithmetic_impl, clippy::suspicious_op_assign_impl)]

use ninjars::{
    BubbleCut, BubbleCutRequest, BubbleSampleRequest, Complex, CutBasis, CutFailure, CutOutcome,
    CutProvider, CutSampler, FourMomentum, SampleId, TadpoleCutRequest, TriangleCut,
    TriangleCutRequest, TriangleSampleRequest,
};
use ninjars_symbolica::{SymbolicaCuts, SymbolicaDualCuts};
use symbolica::{
    create_hyperdual_from_components,
    domains::{
        dual::{DualNumber, HyperDual},
        float::{DoubleFloat, Float, FloatLike, Real, RealLike, SingleFloat},
    },
    evaluate::EvaluationDomain,
};

create_hyperdual_from_components!(MixedCutDual, [[0, 0], [1, 0], [0, 1], [2, 0], [1, 1]]);

fn c(x: f64) -> Complex<f64> {
    Complex::new(x, 0.)
}

fn momentum<T: Real>(prototype: &T, values: [i64; 4]) -> FourMomentum<T> {
    FourMomentum::from_args(
        prototype.from_i64(values[0]),
        prototype.from_i64(values[1]),
        prototype.from_i64(values[2]),
        prototype.from_i64(values[3]),
    )
}

fn check_basis<T: Real>(basis: &CutBasis<T>, zero: &impl Fn(T)) {
    for vector in [&basis.e1, &basis.e2, &basis.e3, &basis.e4] {
        zero(vector.mass_squared());
    }
    for longitudinal in [&basis.e1, &basis.e2] {
        for transverse in [&basis.e3, &basis.e4] {
            zero(longitudinal.dot(transverse));
        }
    }
    zero(basis.e1.dot(&basis.e2) - &basis.e12);
    zero(basis.e3.dot(&basis.e4) + &basis.e12);
}

fn check_triangle<T: Real>(
    cut: &TriangleCut<T>,
    shifts: &[FourMomentum<T>; 3],
    masses: &[T; 3],
    zero: &impl Fn(T),
) {
    check_basis(&cut.basis, zero);
    zero(cut.basis.e3.dot(&cut.e4_effective) * cut.beta.from_i64(2) - cut.beta.one());
    for t in [-3, 2] {
        let t = cut.beta.from_i64(t);
        for mu in [0, 2] {
            let mu = cut.beta.from_i64(mu);
            let q = cut.a.clone()
                + cut.basis.e3.clone() * t.clone()
                + cut.e4_effective.clone() * ((cut.beta.clone() + &mu) / t.clone());
            for (shift, mass) in shifts.iter().zip(masses) {
                zero((q.clone() + shift).mass_squared() - mass - &mu);
            }
        }
    }
}

fn check_bubble<T: Real>(
    cut: &BubbleCut<T>,
    shifts: &[FourMomentum<T>; 2],
    masses: &[T; 2],
    zero: &impl Fn(T),
) {
    check_basis(&cut.basis, zero);
    for (t, x, mu) in [(2, -1, 0), (-3, 2, 1)] {
        let t = cut.parameters[0].from_i64(t);
        let x = cut.parameters[0].from_i64(x);
        let mu = cut.parameters[0].from_i64(mu);
        let beta = cut.parameters[0].clone()
            + cut.parameters[1].clone() * &x
            + cut.parameters[2].clone() * &x * &x;
        let q = cut.a0.clone()
            + cut.a1.clone() * x
            + cut.basis.e3.clone() * t.clone()
            + cut.e4_effective.clone() * ((beta + &mu) / t);
        for (shift, mass) in shifts.iter().zip(masses) {
            zero((q.clone() + shift).mass_squared() - mass - &mu);
        }
    }
}

#[test]
fn mixed_cut_batches_preserve_lane_order_and_isolate_failures() {
    let mut provider = SymbolicaCuts::new(0.).unwrap();
    let zero = c(0.);
    let origin = momentum(&zero, [0, 0, 0, 0]);
    let shifts = [
        [
            origin,
            momentum(&zero, [2, 1, 0, 0]),
            momentum(&zero, [3, 0, 1, 0]),
        ],
        [
            origin,
            momentum(&zero, [0, 1, 0, 0]),
            momentum(&zero, [0, 0, 1, 0]),
        ],
        [
            origin,
            momentum(&zero, [1, 0, 0, 1]),
            momentum(&zero, [2, 0, 0, 0]),
        ],
        [origin, origin, origin],
        [
            origin,
            FourMomentum::from_args(c(f64::NAN), zero, zero, zero),
            origin,
        ],
        [
            origin,
            FourMomentum::from_args(Complex::new(2., 0.25), c(1.), zero, zero),
            momentum(&zero, [3, 0, 1, 0]),
        ],
        [
            origin,
            momentum(&zero, [1, 1, 0, 0]),
            momentum(&zero, [2, 2, 0, 0]),
        ],
    ];
    let masses = [c(1.), Complex::new(2., -0.25), Complex::new(3., -0.5)];
    let check = |value: Complex<f64>| assert!(value.norm().re < 2e-10, "{value}");
    for order in [
        &[2, 3, 1, 4, 0, 5, 6][..],
        &[1][..],
        &[][..],
        &[0, 2, 1][..],
    ] {
        let requests: Vec<_> = order
            .iter()
            .map(|&lane| TriangleCutRequest {
                id: SampleId { lane, sample: 0 },
                shifts: std::array::from_fn(|i| &shifts[lane][i]),
                masses_squared: std::array::from_fn(|i| &masses[i]),
            })
            .collect();
        let mut out = vec![CutOutcome::Unstable(CutFailure::NonFiniteResult); order.len()];
        provider.triangle_batch(&requests, &mut out).unwrap();
        for (&lane, result) in order.iter().zip(out) {
            match result {
                CutOutcome::Stable(cut) => {
                    assert!(lane < 3 || lane == 5);
                    check_triangle(&cut, &shifts[lane], &masses, &check);
                    let ts = [Complex::new(2., 0.5), c(0.), c(-3.)];
                    let mu = Complex::new(1., -0.2);
                    let samples: Vec<_> = ts
                        .iter()
                        .enumerate()
                        .map(|(sample, t)| TriangleSampleRequest {
                            id: SampleId { lane, sample },
                            cut: &cut,
                            t,
                            mu_squared: &mu,
                        })
                        .collect();
                    let mut sampled = vec![CutOutcome::Unstable(CutFailure::NonFiniteResult); 3];
                    provider
                        .sample_triangle_batch(&samples, &mut sampled)
                        .unwrap();
                    for (sample, outcome) in sampled.iter().enumerate() {
                        if sample == 1 {
                            assert!(matches!(
                                outcome,
                                CutOutcome::Unstable(CutFailure::SingularParameter)
                            ));
                            continue;
                        }
                        let CutOutcome::Stable(q) = outcome else {
                            panic!("stable triangle sample failed")
                        };
                        for (v, m) in shifts[lane].iter().zip(masses) {
                            check((*q + v).mass_squared() - m - mu);
                        }
                    }
                    let mut single = [CutOutcome::Unstable(CutFailure::NonFiniteResult)];
                    provider
                        .sample_triangle_batch(&samples[..1], &mut single)
                        .unwrap();
                    let (CutOutcome::Stable(a), CutOutcome::Stable(b)) = (&single[0], &sampled[0])
                    else {
                        panic!("scalar/batch mismatch")
                    };
                    for (a, b) in a.components().into_iter().zip(b.components()) {
                        check(*a - *b);
                    }
                }
                CutOutcome::Unstable(failure) => assert_eq!(
                    failure,
                    match lane {
                        3 => CutFailure::DegenerateReference,
                        4 => CutFailure::NonFiniteInput,
                        6 => CutFailure::DegenerateGram,
                        _ => panic!("unexpected unstable lane {lane}"),
                    }
                ),
            }
        }
    }
    let bubble_shifts = [
        [origin, momentum(&zero, [2, 1, 0, 0])],
        [origin, momentum(&zero, [0, 0, 1, 0])],
        [origin, momentum(&zero, [1, 0, 0, 1])],
        [origin, origin],
    ];
    for order in [&[2, 3, 1, 0][..], &[1][..], &[][..], &[0, 2, 1][..]] {
        let requests: Vec<_> = order
            .iter()
            .map(|&lane| BubbleCutRequest {
                id: SampleId { lane, sample: 0 },
                shifts: std::array::from_fn(|i| &bubble_shifts[lane][i]),
                masses_squared: [&masses[0], &masses[1]],
            })
            .collect();
        let mut out = vec![CutOutcome::Unstable(CutFailure::NonFiniteResult); order.len()];
        provider.bubble_batch(&requests, &mut out).unwrap();
        for (&lane, result) in order.iter().zip(out) {
            match result {
                CutOutcome::Stable(cut) => {
                    assert!(lane < 3);
                    check_bubble(&cut, &bubble_shifts[lane], &[masses[0], masses[1]], &check);
                    let x = Complex::new(-1., 0.2);
                    let t = Complex::new(2., 0.5);
                    let mu = Complex::new(1., -0.2);
                    let request = BubbleSampleRequest {
                        id: SampleId { lane, sample: 0 },
                        cut: &cut,
                        x: &x,
                        t: &t,
                        mu_squared: &mu,
                    };
                    let mut out = [CutOutcome::Unstable(CutFailure::NonFiniteResult)];
                    provider.sample_bubble_batch(&[request], &mut out).unwrap();
                    let CutOutcome::Stable(q) = &out[0] else {
                        panic!("stable bubble sample failed")
                    };
                    for (v, m) in bubble_shifts[lane].iter().zip(masses) {
                        check((*q + v).mass_squared() - m - mu);
                    }
                }
                CutOutcome::Unstable(failure) => {
                    assert_eq!(lane, 3);
                    assert_eq!(failure, CutFailure::DegenerateReference);
                }
            }
        }
    }
    let request = TriangleCutRequest {
        id: SampleId { lane: 0, sample: 0 },
        shifts: std::array::from_fn(|i| &shifts[0][i]),
        masses_squared: std::array::from_fn(|i| &masses[i]),
    };
    assert!(provider.triangle_batch(&[request], &mut []).is_err());

    // Tadpole bases must avoid all uncut leading denominators, including a
    // pure spatial difference that would kill a fixed longitudinal axis.
    let complements = [
        momentum(&zero, [0, 1, 0, 0]),
        momentum(&zero, [0, 0, 1, 0]),
        momentum(&zero, [1, 0, 0, 1]),
    ];
    let repeated = [origin];
    let requests = [
        TadpoleCutRequest {
            id: SampleId { lane: 7, sample: 0 },
            shift: &origin,
            mass_squared: &masses[0],
            complementary_shifts: &complements,
        },
        TadpoleCutRequest {
            id: SampleId { lane: 3, sample: 0 },
            shift: &origin,
            mass_squared: &masses[1],
            complementary_shifts: &repeated,
        },
        TadpoleCutRequest {
            id: SampleId { lane: 1, sample: 0 },
            shift: &shifts[0][1],
            mass_squared: &masses[2],
            complementary_shifts: &[],
        },
    ];
    let mut output = vec![CutOutcome::Unstable(CutFailure::NonFiniteResult); 3];
    provider.tadpole_batch(&requests, &mut output).unwrap();
    for (row, result) in output.iter().enumerate() {
        if row == 1 {
            assert!(matches!(
                result,
                CutOutcome::Unstable(CutFailure::DegenerateReference)
            ));
            continue;
        }
        let CutOutcome::Stable(cut) = result else {
            panic!("unexpected unstable tadpole")
        };
        check_basis(&cut.basis, &check);
        for (direction, effective) in [
            (&cut.basis.e1, &cut.e2_effective),
            (&cut.basis.e3, &cut.e4_effective),
        ] {
            let q = cut.a + *direction * c(2.) + *effective * ((cut.beta + c(3.)) / c(2.));
            check((q + requests[row].shift).mass_squared() - requests[row].mass_squared - c(3.));
        }
        for shift in requests[row].complementary_shifts {
            for e in [&cut.basis.e1, &cut.basis.e2, &cut.basis.e3, &cut.basis.e4] {
                assert!((*shift - *requests[row].shift).dot(e).norm().re > 1e-5);
            }
        }
    }
    provider.tadpole_batch(&[], &mut []).unwrap();
    provider.sample_triangle_batch(&[], &mut []).unwrap();
    provider.sample_bubble_batch(&[], &mut []).unwrap();
}

fn precision_check<T: Real + RealLike + SingleFloat>(prototype: T, tolerance: f64)
where
    Complex<T>: Real + EvaluationDomain,
{
    let mut provider = SymbolicaCuts::new(prototype.clone()).unwrap();
    let c = |v| Complex::new(prototype.from_i64(v), prototype.zero());
    let shifts = [
        momentum(&c(0), [0, 0, 0, 0]),
        momentum(&c(0), [2, 1, 0, 0]),
        momentum(&c(0), [3, 0, 1, 0]),
    ];
    let masses = [c(1), c(2), c(3)];
    let request = TriangleCutRequest {
        id: SampleId { lane: 0, sample: 0 },
        shifts: std::array::from_fn(|i| &shifts[i]),
        masses_squared: std::array::from_fn(|i| &masses[i]),
    };
    let mut output = [CutOutcome::Unstable(CutFailure::NonFiniteResult)];
    provider.triangle_batch(&[request], &mut output).unwrap();
    let CutOutcome::Stable(cut) = &output[0] else {
        panic!("unexpected unstable cut")
    };
    check_triangle(cut, &shifts, &masses, &|v| {
        assert!(v.norm().re.to_f64() < tolerance, "{v}")
    });
    let dual = HyperDual::from_values(vec![vec![0], vec![1], vec![2]], vec![c(0); 3]);
    let mut dual_provider = SymbolicaDualCuts::new(prototype.clone(), dual.clone()).unwrap();
    let lift = |value: Complex<T>| {
        let mut result = dual.zero();
        result.values_mut()[0] = value;
        result
    };
    let mut dual_shifts = shifts.map(|shift| shift.map(&lift));
    dual_shifts[1].temporal.value.values_mut()[1] = c(1);
    let dual_masses = masses.map(lift);
    let request = TriangleCutRequest {
        id: SampleId { lane: 0, sample: 0 },
        shifts: std::array::from_fn(|i| &dual_shifts[i]),
        masses_squared: std::array::from_fn(|i| &dual_masses[i]),
    };
    let mut output = [CutOutcome::Unstable(CutFailure::NonFiniteResult)];
    dual_provider
        .triangle_batch(&[request], &mut output)
        .unwrap();
    let CutOutcome::Stable(cut) = &output[0] else {
        panic!("unexpected unstable high-precision Dual cut")
    };
    check_triangle(cut, &dual_shifts, &dual_masses, &|value| {
        for value in value.values() {
            assert!(value.norm().re.to_f64() < tolerance, "{value}");
        }
    });
}

#[test]
fn cut_geometry_keeps_double_double_and_arbitrary_precision() {
    precision_check(DoubleFloat::from(0.), 1e-27);
    precision_check(Float::new(256), 1e-70);
}

fn dual_check<D: Real + DualNumber<Complex<f64>>>(prototype: D) {
    let mut provider = SymbolicaDualCuts::new(0., prototype.clone()).unwrap();
    let mut shifts = [
        momentum(&prototype, [0, 0, 0, 0]),
        momentum(&prototype, [1, 0, 0, 1]),
        momentum(&prototype, [2, 0, 0, 0]),
    ];
    // The x perturbation leaves the null shell transversely. Clipping a small
    // invariant to zero loses these components even at an exactly null primal.
    shifts[1].temporal.value.values_mut()[1] = c(1.);
    let mut masses = [
        prototype.from_i64(1),
        prototype.from_i64(2),
        prototype.from_i64(3),
    ];
    masses[1].values_mut()[2] = c(1.);
    let request = TriangleCutRequest {
        id: SampleId { lane: 0, sample: 0 },
        shifts: std::array::from_fn(|i| &shifts[i]),
        masses_squared: std::array::from_fn(|i| &masses[i]),
    };
    let mut output = [CutOutcome::Unstable(CutFailure::NonFiniteResult)];
    provider.triangle_batch(&[request], &mut output).unwrap();
    let CutOutcome::Stable(cut) = &output[0] else {
        panic!("unexpected unstable Dual triangle")
    };
    let check = |value: D| {
        for (component, value) in value.values().iter().enumerate() {
            assert!(value.norm().re < 2e-10, "component {component}: {value}");
        }
    };
    check_triangle(cut, &shifts, &masses, &check);
    // The cut equations independently fix a0=-1/2 and
    // az=-1+(u+u²-v)/2. Thus beta=7/4-u+v-3u²/4-uv/2+... .
    // This also checks the Taylor-coefficient convention for second order.
    for (actual, expected) in cut.beta.values().iter().zip([1.75, -1., 1., -0.75, -0.5]) {
        assert!((*actual - c(expected)).norm().re < 1e-12);
    }
    let mut t = prototype.from_i64(2);
    t.values_mut()[1] = c(1.);
    let mut mu = prototype.from_i64(3);
    mu.values_mut()[2] = c(1.);
    let sample = TriangleSampleRequest {
        id: SampleId { lane: 0, sample: 0 },
        cut,
        t: &t,
        mu_squared: &mu,
    };
    let mut sampled = [CutOutcome::Unstable(CutFailure::NonFiniteResult)];
    provider
        .sample_triangle_batch(&[sample], &mut sampled)
        .unwrap();
    let CutOutcome::Stable(q) = &sampled[0] else {
        panic!("Dual triangle sampling failed")
    };
    for (v, m) in shifts.iter().zip(&masses) {
        check((q.clone() + v).mass_squared() - m - &mu);
    }
    let bubble_shifts = [shifts[0].clone(), shifts[1].clone()];
    let request = BubbleCutRequest {
        id: SampleId { lane: 0, sample: 0 },
        shifts: [&bubble_shifts[0], &bubble_shifts[1]],
        masses_squared: [&masses[0], &masses[1]],
    };
    let mut output = [CutOutcome::Unstable(CutFailure::NonFiniteResult)];
    provider.bubble_batch(&[request], &mut output).unwrap();
    let CutOutcome::Stable(cut) = &output[0] else {
        panic!("unexpected unstable Dual bubble")
    };
    check_bubble(
        cut,
        &bubble_shifts,
        &[masses[0].clone(), masses[1].clone()],
        &check,
    );
    let x = prototype.from_i64(-1);
    let sample = BubbleSampleRequest {
        id: SampleId { lane: 0, sample: 0 },
        cut,
        x: &x,
        t: &t,
        mu_squared: &mu,
    };
    let mut sampled = [CutOutcome::Unstable(CutFailure::NonFiniteResult)];
    provider
        .sample_bubble_batch(&[sample], &mut sampled)
        .unwrap();
    let CutOutcome::Stable(q) = &sampled[0] else {
        panic!("Dual bubble sampling failed")
    };
    for (v, m) in bubble_shifts.iter().zip(&masses) {
        check((q.clone() + v).mass_squared() - m - &mu);
    }
    let request = TadpoleCutRequest {
        id: SampleId { lane: 0, sample: 0 },
        shift: &shifts[1],
        mass_squared: &masses[1],
        complementary_shifts: &shifts[2..],
    };
    let mut output = [CutOutcome::Unstable(CutFailure::NonFiniteResult)];
    provider.tadpole_batch(&[request], &mut output).unwrap();
    let CutOutcome::Stable(cut) = &output[0] else {
        panic!("unexpected unstable Dual tadpole")
    };
    check_basis(&cut.basis, &check);
    for (direction, effective) in [
        (&cut.basis.e1, &cut.e2_effective),
        (&cut.basis.e3, &cut.e4_effective),
    ] {
        let q = cut.a.clone()
            + direction.clone() * prototype.from_i64(2)
            + effective.clone()
                * ((cut.beta.clone() + prototype.from_i64(3)) / prototype.from_i64(2));
        check((q + &shifts[1]).mass_squared() - &masses[1] - prototype.from_i64(3));
    }
    let mut invalid = masses[0].clone();
    invalid.values_mut()[4] = c(f64::NAN);
    let requests = [
        TriangleCutRequest {
            id: SampleId { lane: 3, sample: 0 },
            shifts: std::array::from_fn(|i| &shifts[i]),
            masses_squared: [&invalid, &masses[1], &masses[2]],
        },
        TriangleCutRequest {
            id: SampleId { lane: 0, sample: 0 },
            shifts: std::array::from_fn(|i| &shifts[i]),
            masses_squared: std::array::from_fn(|i| &masses[i]),
        },
    ];
    let mut output = vec![CutOutcome::Unstable(CutFailure::NonFiniteResult); 2];
    provider.triangle_batch(&requests, &mut output).unwrap();
    assert!(matches!(
        output[0],
        CutOutcome::Unstable(CutFailure::NonFiniteInput)
    ));
    assert!(matches!(output[1], CutOutcome::Stable(_)));
}

#[test]
fn higher_mixed_duals_preserve_null_shell_transverse_derivatives() {
    dual_check(MixedCutDual { values: [c(0.); 5] });
    dual_check(HyperDual::from_values(
        vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![2, 0], vec![1, 1]],
        vec![c(0.); 5],
    ));
}

#[test]
fn reordered_dual_components_are_rejected_per_lane() {
    let good = HyperDual::from_values(vec![vec![0, 0], vec![1, 0], vec![0, 1]], vec![c(0.); 3]);
    let bad = HyperDual::from_values(vec![vec![0, 0], vec![0, 1], vec![1, 0]], vec![c(0.); 3]);
    let mut provider = SymbolicaDualCuts::new(0., good.clone()).unwrap();
    let shifts = [momentum(&good, [0, 0, 0, 0]), momentum(&good, [2, 1, 0, 0])];
    let mass = good.from_i64(1);
    let requests = [
        BubbleCutRequest {
            id: SampleId { lane: 0, sample: 0 },
            shifts: [&shifts[0], &shifts[1]],
            masses_squared: [&mass, &bad],
        },
        BubbleCutRequest {
            id: SampleId { lane: 1, sample: 0 },
            shifts: [&shifts[0], &shifts[1]],
            masses_squared: [&mass, &mass],
        },
    ];
    let mut output = vec![CutOutcome::Unstable(CutFailure::NonFiniteResult); 2];
    provider.bubble_batch(&requests, &mut output).unwrap();
    assert!(matches!(
        output[0],
        CutOutcome::Unstable(CutFailure::DerivativeShape)
    ));
    assert!(matches!(output[1], CutOutcome::Stable(_)));
}
