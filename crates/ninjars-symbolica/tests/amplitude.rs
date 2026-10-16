use ninjars::{Complex, EvaluationStatus, FourMomentum, Numerator, SingleFloat};
use ninjars_symbolica::{
    PolynomialNumerator, SymbolicaAmplitudeKernels, SymbolicaCuts, SymbolicaMasters,
};
use symbolica::{atom::Atom, parse};

fn c(v: f64) -> Complex<f64> {
    Complex::new(v, 0.)
}
fn q(v: [f64; 4]) -> FourMomentum<Complex<f64>> {
    FourMomentum::from(v.map(c))
}
fn vars() -> [Atom; 5] {
    [
        parse!("q0"),
        parse!("q1"),
        parse!("q2"),
        parse!("q3"),
        parse!("mu2"),
    ]
}
fn near(actual: Complex<f64>, expected: f64) {
    assert!(
        (actual.re - expected).abs() < 1e-9 * (1. + expected.abs()) && actual.im.abs() < 1e-9,
        "{actual} != {expected}"
    );
}

#[test]
fn shifted_tadpoles_match_analytic_tensor_and_mu_integrals() {
    let mut masters = SymbolicaMasters::new(0.);
    let mut geometry = SymbolicaCuts::new(0.).unwrap();
    let mass = 2_f64;
    let finite = mass * (1. - mass.ln());
    for (expression, expected) in [
        (parse!("1"), [finite, mass, 0.]),
        (parse!("q0"), [-3. * finite, -3. * mass, 0.]),
        (
            parse!("q0^2"),
            [
                (9. + mass / 4.) * finite + mass * mass / 8.,
                (9. + mass / 4.) * mass,
                0.,
            ],
        ),
        (parse!("mu2"), [mass * mass / 2., 0., 0.]),
    ] {
        let mut numerator = PolynomialNumerator::new(expression, vars(), vec![], 0.).unwrap();
        numerator.set_parameter_batch(&[], 1).unwrap();
        let mut amplitude = SymbolicaAmplitudeKernels::new(1, numerator.rank(), 0.)
            .unwrap()
            .into_amplitude(1)
            .unwrap();
        amplitude
            .evaluate_batch(
                &[q([3., 1., 2., 1.])],
                &[c(mass)],
                &[c(1.)],
                &mut numerator,
                &mut geometry,
                &mut masters,
            )
            .unwrap();
        assert_eq!(amplitude.statuses(), &[EvaluationStatus::Success]);
        for (actual, expected) in amplitude.results()[0][..3].iter().zip(expected) {
            near(*actual, expected);
        }
        assert!(amplitude.tadpoles().len() == 1);
    }
}

#[test]
fn integrated_bubble_and_triangle_polynomials_match_original_ninja() {
    let mut masters = SymbolicaMasters::new(0.);
    let mut geometry = SymbolicaCuts::new(0.).unwrap();
    let cases = [
        (
            vec![q([0.; 4]), q([0., 1., 0., 0.])],
            parse!("q0-q1"),
            [-0.07602235241001004, 0.5, 0.],
        ),
        (
            vec![q([0.; 4]), q([0., 1., 0., 0.])],
            parse!("(q0-q1)^2"),
            [-0.04580758099112625, 1. / 3., 0.],
        ),
        (
            vec![q([0.; 4]), q([0., 1., 0., 0.])],
            parse!("(q0-q1)^3"),
            [-0.03070019528000567, 0.25, 0.],
        ),
        (
            vec![q([0.; 4]), q([2., 1., 0., 0.]), q([3., 0., 1., 0.])],
            parse!("mu2"),
            [0.5, 0., 0.],
        ),
        (
            vec![q([0.; 4]), q([2., 1., 0., 0.]), q([3., 0., 1., 0.])],
            parse!("mu2^2"),
            [-1. / 12., 0., 0.],
        ),
    ];
    for (shifts, expression, expected) in cases {
        let mut numerator =
            PolynomialNumerator::new(expression.clone(), vars(), vec![], 0.).unwrap();
        numerator.set_parameter_batch(&[], 1).unwrap();
        let mut amplitude = SymbolicaAmplitudeKernels::new(shifts.len(), numerator.rank(), 0.)
            .unwrap()
            .into_amplitude(1)
            .unwrap();
        amplitude
            .evaluate_batch(
                &shifts,
                &vec![c(1.); shifts.len()],
                &[c(1.)],
                &mut numerator,
                &mut geometry,
                &mut masters,
            )
            .unwrap();
        assert_eq!(
            amplitude.statuses(),
            &[EvaluationStatus::Success],
            "{expression}"
        );
        for (actual, expected) in amplitude.results()[0][..3].iter().zip(expected) {
            near(*actual, expected);
        }
    }
}

#[test]
fn accumulation_reset_and_failed_lane_isolation() {
    let mut masters = SymbolicaMasters::new(0.);
    let mut geometry = SymbolicaCuts::new(0.).unwrap();
    let mut numerator = PolynomialNumerator::new(parse!("1"), vars(), vec![], 0.).unwrap();
    numerator.set_parameter_batch(&[], 3).unwrap();
    let mut amplitude = SymbolicaAmplitudeKernels::new(1, 0, 0.)
        .unwrap()
        .into_amplitude(3)
        .unwrap();
    let shifts = vec![q([0.; 4]); 3];
    let masses = [c(1.), c(2.), c(3.)];
    let good = [c(1.); 3];
    amplitude
        .evaluate_batch(
            &shifts,
            &masses,
            &good,
            &mut numerator,
            &mut geometry,
            &mut masters,
        )
        .unwrap();
    let initial = amplitude.results().to_vec();
    amplitude
        .evaluate_batch(
            &shifts,
            &masses,
            &[c(1.), c(0.), c(1.)],
            &mut numerator,
            &mut geometry,
            &mut masters,
        )
        .unwrap();
    assert_eq!(amplitude.results()[1], initial[1]);
    assert!(matches!(
        amplitude.statuses()[1],
        EvaluationStatus::IntegrationFailed { .. }
    ));
    for lane in [0, 2] {
        for (actual, initial) in amplitude.results()[lane].iter().zip(&initial[lane]) {
            near(*actual, 2. * initial.re);
        }
    }
    let saved = amplitude.results().to_vec();
    let statuses = amplitude.statuses().to_vec();
    assert!(
        amplitude
            .evaluate_batch(
                &shifts,
                &masses,
                &good[..2],
                &mut numerator,
                &mut geometry,
                &mut masters
            )
            .is_err()
    );
    assert_eq!(amplitude.results(), saved);
    assert_eq!(amplitude.statuses(), statuses);
    amplitude.reset();
    assert!(
        amplitude
            .results()
            .iter()
            .flatten()
            .all(SingleFloat::is_zero)
    );
    assert_eq!(amplitude.statuses(), &[EvaluationStatus::NotEvaluated; 3]);
    amplitude.reset_batch(0);
    numerator.set_parameter_batch(&[], 0).unwrap();
    amplitude
        .evaluate_batch(&[], &[], &[], &mut numerator, &mut geometry, &mut masters)
        .unwrap();
    assert!(amplitude.results().is_empty());
}

#[test]
fn scalar_scaleless_tadpole_and_null_bubble_vanish() {
    let mut masters = SymbolicaMasters::new(0.);
    let mut geometry = SymbolicaCuts::new(0.).unwrap();
    let mut numerator = PolynomialNumerator::new(parse!("1"), vars(), vec![], 0.).unwrap();
    numerator.set_parameter_batch(&[], 1).unwrap();
    for shifts in [vec![q([0.; 4])], vec![q([0.; 4]), q([1., 0., 0., 1.])]] {
        let mut amplitude = SymbolicaAmplitudeKernels::new(shifts.len(), 0, 0.)
            .unwrap()
            .into_amplitude(1)
            .unwrap();
        amplitude
            .evaluate_batch(
                &shifts,
                &vec![c(0.); shifts.len()],
                &[c(1.)],
                &mut numerator,
                &mut geometry,
                &mut masters,
            )
            .unwrap();
        assert_eq!(amplitude.statuses(), &[EvaluationStatus::Success]);
        for value in &amplitude.results()[0] {
            near(*value, 0.);
        }
    }
}
