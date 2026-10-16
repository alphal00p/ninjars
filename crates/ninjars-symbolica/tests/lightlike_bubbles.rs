//! Lightlike external legs must not force uncut Laurent denominators to zero.

use ninjars::{
    BubbleReducer, Complex, CutFailure, CutOutcome, DoubleFloat, EvaluationStatus, Float,
    FourMomentum, Real, Topology, TriangleReducer,
};
use ninjars_symbolica::{
    DualPolynomialNumerator, PolynomialNumerator, SymbolicaAmplitudeKernels, SymbolicaCuts,
    SymbolicaDualBubbleKernels, SymbolicaDualCuts, SymbolicaDualTriangleKernels, SymbolicaMasters,
};
use oneloop::NativeFloat;
use symbolica::{
    atom::Atom,
    domains::{dual::HyperDual, rational::Rational},
    evaluate::EvaluationDomain,
    parse,
};

fn variables() -> [Atom; 5] {
    [
        parse!("q0"),
        parse!("q1"),
        parse!("q2"),
        parse!("q3"),
        parse!("mu2"),
    ]
}

fn pure_lightlike_bubble<T: NativeFloat>(prototype: T, digits: u32)
where
    Complex<T>: Real + EvaluationDomain,
{
    let value = |n| Complex::new(prototype.from_i64(n), prototype.zero());
    let momentum = |p: [i64; 4]| FourMomentum::from(p.map(value));
    // The numerator cancels D2 exactly. D0 and D1 are separated by a null
    // momentum, so the independent Feynman-parameter integral is
    // B0(0;9,9)=[-log(9),1,0] at scale²=1.
    let mut numerator = PolynomialNumerator::new(
        parse!("(q0-2)^2-q1^2-q2^2-q3^2-9-mu2"),
        variables(),
        vec![],
        prototype.clone(),
    )
    .unwrap();
    numerator.set_parameter_batch(&[], 1).unwrap();
    let shifts = [
        momentum([-1, 0, 0, 1]),
        momentum([0; 4]),
        momentum([-2, 0, 0, 0]),
    ];
    let mut geometry = SymbolicaCuts::new(prototype.clone()).unwrap();
    let mut masters = SymbolicaMasters::new(prototype.clone());
    let mut amplitude = SymbolicaAmplitudeKernels::new(3, 2, prototype.clone())
        .unwrap()
        .into_amplitude(1)
        .unwrap();
    amplitude
        .evaluate_batch(
            &shifts,
            &[value(9), value(9), value(9)],
            &[value(1)],
            &mut numerator,
            &mut geometry,
            &mut masters,
        )
        .unwrap();
    assert_eq!(amplitude.statuses(), &[EvaluationStatus::Success]);
    let tolerance = prototype.from_rational(&Rational::from((1, 10)).pow(digits.into()));
    let check = |actual: &Complex<T>, expected: &Complex<T>| {
        let error = actual - expected;
        assert!(
            error.re.norm().real_cmp(&tolerance).unwrap().is_lt()
                && error.im.norm().real_cmp(&tolerance).unwrap().is_lt(),
            "{actual} != {expected}"
        );
    };
    for outcome in amplitude.bubbles() {
        let CutOutcome::Stable(residue) = outcome else {
            panic!("unstable bubble")
        };
        let cut = &amplitude.topology().cuts()[residue.cut_index];
        for (i, coefficient) in residue.coefficients.iter().enumerate() {
            check(
                coefficient,
                &value(i64::from(i == 0 && cut.partition == [0, 1])),
            );
        }
    }
    let expected = -prototype.from_i64(9).log();
    for (actual, expected) in amplitude.results()[0].iter().zip([
        Complex::new(expected.clone(), prototype.zero()),
        value(1),
        value(0),
        Complex::new(expected, prototype.zero()),
        value(0),
    ]) {
        check(actual, &expected);
    }
}

#[test]
fn canceled_triangle_denominator_leaves_equal_mass_lightlike_bubble() {
    pure_lightlike_bubble(0_f64, 10);
    pure_lightlike_bubble(DoubleFloat::default(), 27);
    pure_lightlike_bubble(Float::new(192), 48);
}

#[test]
fn lightlike_bubble_keeps_mixed_dual_coefficients_and_isolates_failed_lanes() {
    let shape = vec![
        vec![0, 0],
        vec![1, 0],
        vec![0, 1],
        vec![2, 0],
        vec![1, 1],
        vec![2, 1],
    ];
    let c = |v| Complex::new(v, 0.);
    let dual = |values: [f64; 6]| HyperDual::from_values(shape.clone(), values.map(c).to_vec());
    let number = |v| dual([v, 0., 0., 0., 0., 0.]);
    let prototype = number(0.);
    let coefficient = dual([1., 2., 3., 5., 7., 11.]);
    let translation = dual([0., 3., -2., 1., 4., 5.]);
    let source = PolynomialNumerator::new(
        parse!("a*((q0-2+h)^2-q1^2-q2^2-q3^2-9-mu2)"),
        variables(),
        vec![parse!("a"), parse!("h")],
        0.,
    )
    .unwrap();
    let mut numerator = DualPolynomialNumerator::new(source, prototype.clone()).unwrap();
    numerator
        .set_parameter_batch(
            &[
                coefficient.clone(),
                translation.clone(),
                coefficient.clone(),
                translation.clone(),
            ],
            2,
        )
        .unwrap();
    let routing = [[-1., 0., 0., 1.], [0.; 4], [-2., 0., 0., 0.]];
    let shifts: Vec<_> = (0..2)
        .flat_map(|_| {
            routing.map(|p| {
                FourMomentum::from_args(
                    number(p[0]) + &translation,
                    number(p[1]),
                    number(p[2]),
                    number(p[3]),
                )
            })
        })
        .collect();
    let mut masses = vec![number(9.); 6];
    let mut geometry = SymbolicaDualCuts::new(0., prototype.clone()).unwrap();
    let mut triangle = TriangleReducer::new(
        SymbolicaDualTriangleKernels::new(3, 2, 0., &prototype).unwrap(),
        prototype.clone(),
    )
    .unwrap();
    let mut bubble = BubbleReducer::new(
        SymbolicaDualBubbleKernels::new(3, 2, 0., &prototype).unwrap(),
        prototype,
    )
    .unwrap();
    let topology = Topology::new(3, 2).unwrap();
    let mut triangles = vec![CutOutcome::Unstable(CutFailure::NonFiniteResult); 2];
    let mut bubbles = vec![CutOutcome::Unstable(CutFailure::NonFiniteResult); 6];
    for failed_lane in [false, true] {
        if failed_lane {
            masses[5].values[5] = c(f64::NAN);
        }
        triangle
            .evaluate_batch(
                &shifts,
                &masses,
                2,
                &mut numerator,
                &mut geometry,
                &mut triangles,
            )
            .unwrap();
        bubble
            .evaluate_batch(
                &shifts,
                &masses,
                2,
                &topology,
                &triangles,
                &mut numerator,
                &mut geometry,
                &mut bubbles,
            )
            .unwrap();
        for (row, outcome) in bubbles.iter().enumerate() {
            if failed_lane && row % 2 == 1 {
                assert!(matches!(
                    outcome,
                    CutOutcome::Unstable(CutFailure::NonFiniteInput)
                ));
                continue;
            }
            let CutOutcome::Stable(residue) = outcome else {
                panic!("unstable good lane: {outcome:?}")
            };
            let partition = &topology.cuts()[residue.cut_index].partition;
            for (i, actual) in residue.coefficients.iter().enumerate() {
                let expected = if i == 0 && partition == &[0, 1] {
                    coefficient.clone()
                } else {
                    number(0.)
                };
                for (actual, expected) in actual.values.iter().zip(&expected.values) {
                    assert!(
                        (actual - expected).re.abs() < 1e-9,
                        "{actual} != {expected}"
                    );
                    assert!(
                        (actual - expected).im.abs() < 1e-9,
                        "{actual} != {expected}"
                    );
                }
            }
        }
    }
}
