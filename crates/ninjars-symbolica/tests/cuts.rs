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
                complementary_shifts: &[],
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

/// Largest component modulus of a momentum.
fn component_scale<T: Real + RealLike>(v: &FourMomentum<Complex<T>>) -> f64 {
    v.components()
        .into_iter()
        .map(|c| c.norm().re.to_f64())
        .fold(0., f64::max)
}

/// The transverse directions of a bubble cut must not be orthogonal to an
/// uncut shift difference: that uncut denominator would lose its leading
/// Laurent coefficient 2 e3.(V_uncut - V_first), leaving only round-off to
/// divide by. At the integer pentagon point below (examples/graphs/
/// pentagon-rank4.dot), V1 = 2 V3 - 3 r for the rational reference
/// r = (1, -2/3, -2/3, -1/3), the largest |k.r| on the {0, 3} double cut
/// among the first eight references. Among all 24 the largest is
/// r' = (1, -1/3, -2/3, -2/3), and an added uncut shift V5 = -k - 3 r' =
/// (-5, 0, 0, 0) lies exactly in its plane, so choosing by |k.r| alone hits
/// this plane.
fn coplanar_bubble_check<T: Real + RealLike + SingleFloat>(prototype: T, tolerance: f64)
where
    Complex<T>: Real + EvaluationDomain,
{
    let mut provider = SymbolicaCuts::new(prototype.clone()).unwrap();
    let zero = Complex::new(prototype.zero(), prototype.zero());
    let shifts = [
        momentum(&zero, [0, 0, 0, 0]),
        momentum(&zero, [-2, -1, -2, -2]),
    ];
    let uncut = [
        momentum(&zero, [-7, 0, -2, -3]),
        momentum(&zero, [-5, -1, -1, -3]),
        momentum(&zero, [5, -1, -2, -1]),
        momentum(&zero, [-5, 0, 0, 0]),
    ];
    let one = Complex::new(prototype.one(), prototype.zero());
    let masses = [one.clone(), one];
    let collinear = [shifts[1].clone() * Complex::new(prototype.from_i64(2), prototype.zero())];
    let request = |complementary_shifts| BubbleCutRequest {
        id: SampleId { lane: 0, sample: 0 },
        shifts: [&shifts[0], &shifts[1]],
        masses_squared: [&masses[0], &masses[1]],
        complementary_shifts,
    };
    let requests = [request(&[]), request(&uncut), request(&collinear)];
    let mut output = vec![CutOutcome::Unstable(CutFailure::NonFiniteResult); 3];
    provider.bubble_batch(&requests, &mut output).unwrap();
    // Leading coefficients relative to the vector sizes, as in the selection.
    let leading = |cut: &BubbleCut<Complex<T>>, v: &FourMomentum<Complex<T>>| {
        let v = v.clone() - shifts[0].clone();
        [&cut.basis.e3, &cut.basis.e4]
            .map(|e| v.dot(e).norm().re.to_f64() / (component_scale(e) * component_scale(&v)))
    };
    let check = |v: Complex<T>| assert!(v.norm().re.to_f64() < tolerance, "{v}");
    // Without the uncut lines the provider keeps the |k.r| choice, which is
    // exactly degenerate for V5 here (only round-off survives).
    let CutOutcome::Stable(blind) = &output[0] else {
        panic!("bubble without uncut lines failed")
    };
    check_bubble(blind, &shifts, &masses, &check);
    for value in leading(blind, &uncut[3]) {
        assert!(
            value < tolerance,
            "the fixture is no longer degenerate: {value}"
        );
    }
    // With them, every uncut line keeps an O(1) leading coefficient.
    let CutOutcome::Stable(cut) = &output[1] else {
        panic!("bubble with uncut lines failed: {:?}", output[1])
    };
    check_bubble(cut, &shifts, &masses, &check);
    for v in &uncut {
        for value in leading(cut, v) {
            assert!(value > 1e-2, "uncut leading coefficient {value}");
        }
    }
    // An uncut shift collinear with the cut momentum lies in the plane of
    // every reference; the provider reports it instead of dividing by zero.
    assert!(matches!(
        output[2],
        CutOutcome::Unstable(CutFailure::DegenerateReference)
    ));
}

#[test]
fn bubble_reference_avoids_uncut_lines_in_its_plane() {
    coplanar_bubble_check(0., 1e-12);
    coplanar_bubble_check(DoubleFloat::from(0.), 1e-27);
    coplanar_bubble_check(Float::new(256), 1e-70);
}

/// The selected reference (the basis e2) of the {D0, D3} double cut of
/// examples/graphs/physical/four-photon/top-box-1.json: cut shifts p2 and p0,
/// uncut shifts 0 and p0 + p1, all in the x-z plane. The references related
/// by y -> -y therefore score exactly alike (at least two of the 24 tie for
/// the best).
fn tied_bubble_reference<T: Real + RealLike + SingleFloat>(prototype: T) -> Vec<f64>
where
    Complex<T>: Real + EvaluationDomain,
{
    let mut provider = SymbolicaCuts::new(prototype.clone()).unwrap();
    let zero = Complex::new(prototype.zero(), prototype.zero());
    let shifts = [
        momentum(&zero, [100, 60, 0, 80]),
        momentum(&zero, [100, 0, 0, 100]),
    ];
    let uncut = [
        momentum(&zero, [0, 0, 0, 0]),
        momentum(&zero, [200, 0, 0, 0]),
    ];
    let mass = Complex::new(prototype.from_i64(173 * 173), prototype.zero());
    let request = BubbleCutRequest {
        id: SampleId { lane: 0, sample: 0 },
        shifts: [&shifts[0], &shifts[1]],
        masses_squared: [&mass, &mass],
        complementary_shifts: &uncut,
    };
    let mut output = [CutOutcome::Unstable(CutFailure::NonFiniteResult)];
    provider.bubble_batch(&[request], &mut output).unwrap();
    let CutOutcome::Stable(cut) = &output[0] else {
        panic!("tied bubble cut failed: {:?}", output[0])
    };
    let check = |v: Complex<T>| assert!(v.norm().re.to_f64() < 1e-6, "{v}");
    check_bubble(cut, &shifts, &[mass.clone(), mass.clone()], &check);
    cut.basis
        .e2
        .components()
        .into_iter()
        .flat_map(|c| [c.re.to_f64(), c.im.to_f64()])
        .collect()
}

#[test]
fn tied_bubble_references_do_not_depend_on_the_precision() {
    // The reference fixes how the bubble residue splits between B1/B11, its
    // mu^2 rational term and the tadpoles' A0 subtraction. Round-off used to
    // break this tie, so lane 3 of the four-photon top-box-1/2 fixtures moved
    // ~1375 between cut_constructible_finite and rational at 68, 80, 88, 92,
    // 100, 104, 116, ... bits. Ties now go to the lowest-index reference.
    let expected = tied_bubble_reference(Float::new(400));
    let mut chosen = vec![
        ("f64".to_string(), tied_bubble_reference(0.)),
        (
            "double-float".into(),
            tied_bubble_reference(DoubleFloat::from(0.)),
        ),
    ];
    for bits in [
        64, 68, 72, 80, 88, 92, 100, 104, 108, 116, 128, 140, 148, 256,
    ] {
        chosen.push((
            format!("{bits} bits"),
            tied_bubble_reference(Float::new(bits)),
        ));
    }
    for (label, e2) in chosen {
        assert!(
            e2.iter().zip(&expected).all(|(a, b)| (a - b).abs() < 1e-12),
            "{label}: reference {e2:?} != 400-bit {expected:?}"
        );
    }
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
        complementary_shifts: &shifts[2..],
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
            complementary_shifts: &[],
        },
        BubbleCutRequest {
            id: SampleId { lane: 1, sample: 0 },
            shifts: [&shifts[0], &shifts[1]],
            masses_squared: [&mass, &mass],
            complementary_shifts: &[],
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

/// The {0, 2} double cut of lane 2 of the final review's
/// `fr-light-zero-n5-r7` (massless, light-like legs): `k = (2, 14, -2, 6)`
/// with uncut shifts that include two null vectors. The fast engine's score
/// over all 24 signed permutations of `(1, 2/3, 2/3, 1/3)` picks
/// `(1, -2/3, -1/3, -2/3)`; the previous score over the first eight,
/// `(1, ±2/3, ±2/3, ±1/3)`, picked `(1, -2/3, -2/3, -1/3)`.
fn light_like_bubble<T: Real + RealLike + SingleFloat>(prototype: T) -> Vec<f64>
where
    Complex<T>: Real + EvaluationDomain,
{
    let mut provider = SymbolicaCuts::new(prototype.clone()).unwrap();
    let zero = Complex::new(prototype.zero(), prototype.zero());
    let shifts = [
        momentum(&zero, [0, 0, 0, 0]),
        momentum(&zero, [-2, -14, 2, -6]),
    ];
    let uncut = [
        momentum(&zero, [21, -11, -16, 8]),
        momentum(&zero, [-33, -28, -19, 12]),
        momentum(&zero, [9, 4, 7, 4]),
    ];
    let request = BubbleCutRequest {
        id: SampleId { lane: 0, sample: 0 },
        shifts: [&shifts[0], &shifts[1]],
        masses_squared: [&zero, &zero],
        complementary_shifts: &uncut,
    };
    let mut output = [CutOutcome::Unstable(CutFailure::NonFiniteResult)];
    provider.bubble_batch(&[request], &mut output).unwrap();
    let CutOutcome::Stable(cut) = &output[0] else {
        panic!("light-like bubble cut failed: {:?}", output[0])
    };
    let check = |v: Complex<T>| assert!(v.norm().re.to_f64() < 1e-9, "{v}");
    check_bubble(cut, &shifts, &[zero.clone(), zero.clone()], &check);
    cut.basis
        .e2
        .components()
        .into_iter()
        .flat_map(|c| [c.re.to_f64(), c.im.to_f64()])
        .collect()
}

#[test]
fn bubble_reference_scores_all_24_candidates() {
    let third = 1. / 3.;
    let want = [1., -2. * third, -third, -2. * third];
    let close = |e2: &[f64]| {
        e2.chunks(2)
            .zip(want)
            .all(|(c, w)| (c[0] - w).abs() < 1e-12 && c[1].abs() < 1e-12)
    };
    for (label, e2) in [
        ("f64", light_like_bubble(0.)),
        ("double-float", light_like_bubble(DoubleFloat::from(0.))),
        ("256 bits", light_like_bubble(Float::new(256))),
    ] {
        assert!(close(&e2), "{label}: {e2:?}");
    }
}

/// A tadpole basis scores only the growing expansion vectors,
/// `min_v min(|e1.v|/|e1|, |e3.v|/|e3|)/|v|` (Euclidean norms), as in the
/// fast engine. For the uncut difference `v = (-3, -3, -3, -1)` the previous
/// rule, the smallest projection on all four vectors relative to the largest
/// component of `v` (capped at 1), chose a different basis.
#[test]
fn tadpole_basis_scores_the_growing_vectors() {
    // The (a, b, c, d) of the six bases: e1 and e2 have the Weyl components
    // (1, a + i b) and (1, c + i d); e3 mixes the angle of e1 with the square
    // of e2, e4 the angle of e2 with the square of e1 (`tadpole_plans.rs`).
    let pairs: [(f64, f64, f64, f64); 6] = [
        (1., 2., -2., 1.),
        (2., -1., 1., 3.),
        (-1., 3., 2., 1.),
        (3., 1., -1., -2.),
        (1., -3., 3., -2.),
        (2., 3., -3., 1.),
    ];
    let v = [-3., -3., -3., -1.];
    let dot =
        |e: [Complex<f64>; 4]| e[0] * c(v[0]) - e[1] * c(v[1]) - e[2] * c(v[2]) - e[3] * c(v[3]);
    let norm2 = |e: &[Complex<f64>; 4]| e.iter().map(|x| x.norm_squared()).sum::<f64>();
    let from_spinors = |angle: [Complex<f64>; 2], square: [Complex<f64>; 2]| {
        let (d0, d1) = (angle[0] * square[0], angle[1] * square[1]);
        let (o0, o1) = (angle[0] * square[1], angle[1] * square[0]);
        let half = c(0.5);
        [
            (d0 + d1) * half,
            (o0 + o1) * half,
            Complex::new(0., 1.) * (o0 - o1) * half,
            (d0 - d1) * half,
        ]
    };
    let bases: Vec<[[Complex<f64>; 4]; 4]> = pairs
        .iter()
        .map(|&(a, b, cc, d)| {
            let one = c(1.);
            let (angle1, square1) = ([one, Complex::new(a, b)], [one, Complex::new(a, -b)]);
            let (angle2, square2) = ([one, Complex::new(cc, d)], [one, Complex::new(cc, -d)]);
            [
                from_spinors(angle1, square1),
                from_spinors(angle2, square2),
                from_spinors(angle1, square2),
                from_spinors(angle2, square1),
            ]
        })
        .collect();
    let vv: f64 = v.iter().map(|x| x * x).sum();
    let scores: Vec<f64> = bases
        .iter()
        .map(|[e1, _, e3, _]| {
            (dot(*e1).norm_squared() / norm2(e1)).min(dot(*e3).norm_squared() / norm2(e3)) / vv
        })
        .collect();
    let v_max = v.iter().fold(0., |m: f64, x| m.max(x.abs()));
    // The previous rule started every score at 1 (so projections above |v|
    // tie) and took the first best.
    let previous: Vec<f64> = bases
        .iter()
        .map(|basis| {
            basis
                .iter()
                .map(|e| dot(*e).norm().re / v_max)
                .fold(1., f64::min)
        })
        .collect();
    let argmax = |s: &[f64]| (0..6).fold(0, |best, i| if s[i] > s[best] { i } else { best });
    let best = argmax(&scores);
    assert_ne!(
        argmax(&previous),
        best,
        "the fixture no longer separates the two rules"
    );

    let mut provider = SymbolicaCuts::new(0.).unwrap();
    let zero = c(0.);
    let origin = momentum(&zero, [0, 0, 0, 0]);
    let uncut = [momentum(&zero, [-3, -3, -3, -1])];
    let mass = c(2.);
    let request = TadpoleCutRequest {
        id: SampleId { lane: 0, sample: 0 },
        shift: &origin,
        mass_squared: &mass,
        complementary_shifts: &uncut,
    };
    let mut output = [CutOutcome::Unstable(CutFailure::NonFiniteResult)];
    provider.tadpole_batch(&[request], &mut output).unwrap();
    let CutOutcome::Stable(cut) = &output[0] else {
        panic!("tadpole failed: {:?}", output[0])
    };
    let check = |value: Complex<f64>| assert!(value.norm().re < 1e-10, "{value}");
    check_basis(&cut.basis, &check);
    // The basis is identified by e1 = ((1 + a² + b²)/2, a, b, (1 - a² - b²)/2).
    let e1 = cut.basis.e1.components().map(|x| x.re);
    let index = pairs
        .iter()
        .position(|&(a, b, _, _)| (e1[1] - a).abs() < 1e-12 && (e1[2] - b).abs() < 1e-12)
        .expect("one of the six bases");
    assert_eq!(index, best, "scores {scores:?}");
}

/// Bubble rows are probed with all 24 references row by row: in a batch every
/// row equals its single-row result bitwise, in any order.
#[test]
fn bubble_batches_equal_single_rows() {
    let mut provider = SymbolicaCuts::new(0.).unwrap();
    let zero = c(0.);
    let shifts = [
        [
            momentum(&zero, [0, 0, 0, 0]),
            momentum(&zero, [-2, -14, 2, -6]),
        ],
        [
            momentum(&zero, [100, 60, 0, 80]),
            momentum(&zero, [100, 0, 0, 100]),
        ],
        [
            momentum(&zero, [0, 0, 0, 0]),
            momentum(&zero, [-2, -1, -2, -2]),
        ],
    ];
    let uncut = [
        vec![
            momentum(&zero, [21, -11, -16, 8]),
            momentum(&zero, [-33, -28, -19, 12]),
            momentum(&zero, [9, 4, 7, 4]),
        ],
        vec![
            momentum(&zero, [0, 0, 0, 0]),
            momentum(&zero, [200, 0, 0, 0]),
        ],
        Vec::new(),
    ];
    let masses = [c(0.), c(173. * 173.), c(1.)];
    let request = |g: usize| BubbleCutRequest {
        id: SampleId { lane: g, sample: 0 },
        shifts: [&shifts[g][0], &shifts[g][1]],
        masses_squared: [&masses[g], &masses[g]],
        complementary_shifts: &uncut[g],
    };
    let bits = |outcome: &CutOutcome<BubbleCut<Complex<f64>>>| {
        let CutOutcome::Stable(cut) = outcome else {
            panic!("bubble failed: {outcome:?}")
        };
        [
            &cut.basis.e1,
            &cut.basis.e2,
            &cut.basis.e3,
            &cut.basis.e4,
            &cut.a0,
            &cut.a1,
        ]
        .into_iter()
        .flat_map(|v| v.components())
        .chain(&cut.parameters)
        .flat_map(|x| [x.re.to_bits(), x.im.to_bits()])
        .collect::<Vec<_>>()
    };
    let single: Vec<_> = (0..3)
        .map(|g| {
            let mut output = [CutOutcome::Unstable(CutFailure::NonFiniteResult)];
            provider.bubble_batch(&[request(g)], &mut output).unwrap();
            bits(&output[0])
        })
        .collect();
    for order in [&[0, 1, 2][..], &[2, 1, 0], &[1, 2, 0, 1], &[2, 0, 2, 1, 0]] {
        let requests: Vec<_> = order.iter().map(|&g| request(g)).collect();
        let mut output = vec![CutOutcome::Unstable(CutFailure::NonFiniteResult); requests.len()];
        provider.bubble_batch(&requests, &mut output).unwrap();
        for (position, &g) in order.iter().enumerate() {
            assert_eq!(
                bits(&output[position]),
                single[g],
                "row {g} in order {order:?}"
            );
        }
    }
}
