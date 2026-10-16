//! Integrated four/six-line controls from independently executed original Ninja.
use ninjars::{
    BoxCoefficientMode, BoxMuMode, BoxReductionOptions, Complex, EvaluationStatus, FourMomentum,
    Numerator,
};
use ninjars_symbolica::{
    PolynomialNumerator, SymbolicaAmplitudeKernels, SymbolicaCuts, SymbolicaHighCuts,
    SymbolicaMasters,
};
use symbolica::{atom::Atom, parser::ParseSettings};

fn c(x: f64) -> Complex<f64> {
    Complex::new(x, 0.)
}
fn vars() -> [Atom; 5] {
    ["q0", "q1", "q2", "q3", "mu2"]
        .map(|name| Atom::parse(name, "ninjars_high_validation", ParseSettings::default()).unwrap())
}

#[test]
fn complete_four_to_six_line_amplitudes_match_original_ninja_both_mu_modes() {
    let evidence: serde_json::Value = serde_json::from_str(include_str!(
        "../../../validation/evidence/high-reduction.json"
    ))
    .unwrap();
    let cases = evidence["cases"].as_array().unwrap();
    let mut low = SymbolicaCuts::new(0.).unwrap();
    let mut high = SymbolicaHighCuts::new(0.).unwrap();
    let mut masters = SymbolicaMasters::new(0.);
    let mut checks = Vec::new();
    let mut max_scaled_error = 0_f64;
    for pair in cases.as_chunks::<2>().0 {
        let n = pair[0]["shifts"].as_array().unwrap().len();
        let expression = Atom::parse(
            pair[0]["expression"].as_str().unwrap(),
            "ninjars_high_validation",
            ParseSettings::default(),
        )
        .unwrap();
        let mut numerator = PolynomialNumerator::new(expression, vars(), vec![], 0.).unwrap();
        numerator.set_parameter_batch(&[], 2).unwrap();
        let shifts: Vec<_> = pair
            .iter()
            .flat_map(|case| {
                case["shifts"].as_array().unwrap().iter().map(|p| {
                    FourMomentum::from(std::array::from_fn::<_, 4, _>(|i| {
                        c(p[i].as_f64().unwrap())
                    }))
                })
            })
            .collect();
        let masses: Vec<_> = pair
            .iter()
            .flat_map(|case| {
                case["masses_squared"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|m| Complex::new(m[0].as_f64().unwrap(), m[1].as_f64().unwrap()))
            })
            .collect();
        // Follow Ninja's dimensional scale rule instead of sampling mu² near
        // unity when the invariants reach hundreds. Tiny scales at such points
        // amplify cancellation when extracting the mu⁴ coefficient.
        let sampling_scales: Vec<_> = (0..2)
            .map(|lane| {
                let p = &shifts[lane * n..(lane + 1) * n];
                let mut scale = 1_f64;
                for first in 0..n {
                    for second in first + 1..n {
                        let difference = p[second] - p[first];
                        scale = scale.max(difference.dot(&difference).re.abs());
                    }
                }
                for mass in &masses[lane * n..(lane + 1) * n] {
                    scale = scale.max(mass.re.abs()).max(mass.im.abs());
                }
                c(scale)
            })
            .collect();
        for (coefficients, mu_extraction) in [
            (BoxCoefficientMode::Integrated, BoxMuMode::Expansion),
            (BoxCoefficientMode::Full, BoxMuMode::Sampling),
        ] {
            let mut amplitude = SymbolicaAmplitudeKernels::new(n, numerator.rank(), 0.)
                .unwrap()
                .with_box_options(BoxReductionOptions {
                    coefficients,
                    mu_extraction,
                })
                .into_amplitude(2)
                .unwrap();
            amplitude
                .evaluate_batch_with_high_cuts(
                    &shifts,
                    &masses,
                    &[c(1.); 2],
                    &sampling_scales,
                    &mut numerator,
                    &mut low,
                    &mut high,
                    &mut masters,
                )
                .unwrap();
            for (lane, case) in pair.iter().enumerate() {
                let label = case["name"].as_str().unwrap();
                assert_eq!(
                    amplitude.statuses()[lane],
                    EvaluationStatus::Success,
                    "{label} {mu_extraction:?}"
                );
                for coefficient in 0..3 {
                    for part in 0..2 {
                        let value = amplitude.results()[lane][coefficient];
                        let actual = if part == 0 { value.re } else { value.im };
                        let expected = case["reference"]["epsilon"][coefficient][part]
                            .as_f64()
                            .unwrap();
                        max_scaled_error =
                            max_scaled_error.max((actual - expected).abs() / (1. + expected.abs()));
                        assert!(
                            (actual - expected).abs() <= 1e-9 * (1. + expected.abs()),
                            "{label} {mu_extraction:?} coefficient={coefficient},part={part}: actual={actual:e},expected={expected:e}"
                        );
                    }
                }
                let value = amplitude.results()[lane];
                assert!(
                    (value[0].re - value[3].re - value[4].re).abs()
                        < 1e-9 * (1. + value[0].re.abs())
                );
                checks.push(serde_json::json!({"case":label,"mu_extraction":format!("{mu_extraction:?}"),"sampling_scale":sampling_scales[lane].re,"actual":value.map(|v|[v.re,v.im])}));
            }
        }
    }
    eprintln!(
        "NINJARS_HIGH_REDUCTION {}",
        serde_json::json!({"checks":checks,"max_scaled_error":max_scaled_error,"scaled_tolerance":1e-9,"performance_acceptance":false,"physical_validation":false,"full_ninja_acceptance":false})
    );
}
