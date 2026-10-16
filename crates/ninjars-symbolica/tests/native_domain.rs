#![cfg(feature = "native-kernels")]

use ninjars::{Complex, DoubleFloat, Float, FloatLike, Real, SingleFloat};
use ninjars_symbolica::{LaurentDivision, TriangleBubbleSubtraction};
use oneloop::NativeFloat;
use symbolica::evaluate::EvaluationDomain;

fn correction<T: NativeFloat + Into<Float>>(prototype: T, from_f64: impl Fn(f64) -> T)
where
    Complex<T>: Real + EvaluationDomain,
{
    let plan =
        TriangleBubbleSubtraction::for_quotient(&LaurentDivision::new(5, 3, 2).unwrap()).unwrap();
    let layout = plan.input_layout();
    let mut native = plan.compile_native(&prototype).unwrap();
    let mut oracle = plan.compile::<Float>(&Float::new(256)).unwrap();
    let precision = Float::new(256);
    let tolerance = Float::parse(
        if T::FIXED_BITS.is_none() {
            "1e-65"
        } else {
            "1e-12"
        },
        Some(256),
    )
    .unwrap();
    for scale in [4e102, 8e102, 1e110, -1e110, 1e155, 1e-110] {
        for rows in [1, 3] {
            let lane_scales = if rows == 1 {
                vec![scale]
            } else {
                vec![scale, 1., scale]
            };
            let input: Vec<_> = lane_scales
                .iter()
                .flat_map(|&scale| {
                    let mut row = vec![
                        Complex::new(prototype.zero(), prototype.zero());
                        plan.input_columns()
                    ];
                    row[layout.residue.start + 3].re = from_f64(scale);
                    row[layout.projection_e3.start + 2].re = prototype.one();
                    row[layout.denominator.start].re = from_f64(scale);
                    row[layout.denominator.start + 1].re = from_f64(scale);
                    row
                })
                .collect();
            // Preserve every input's represented value when raising the oracle
            // precision; no binary64 conversion of DoubleFloat/Float values.
            let exact = |value: &T| {
                let value: Float = value.clone().into();
                precision.from_rational(&value.to_rational())
            };
            let high: Vec<_> = input
                .iter()
                .map(|value| Complex::new(exact(&value.re), exact(&value.im)))
                .collect();
            let mut actual = vec![Complex::new(prototype.zero(), prototype.zero()); rows];
            let mut expected = vec![Complex::new(precision.zero(), precision.zero()); rows];
            native.evaluate_batch(&input, &mut actual, rows).unwrap();
            oracle.evaluate_batch(&high, &mut expected, rows).unwrap();
            for (lane, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                assert!(expected.is_finite());
                assert!(
                    (expected.re.clone() - precision.one()).norm()
                        < Float::parse("1e-70", Some(256)).unwrap()
                );
                if actual.is_finite() {
                    let actual = Complex::new(exact(&actual.re), exact(&actual.im));
                    assert!(
                        (actual - expected).norm().re < tolerance,
                        "incorrect finite native correction: scale={} lane={lane}",
                        lane_scales[lane]
                    );
                } else {
                    // Low-level kernels report buffer validity; the owning
                    // reduction engine rejects these nonfinite output lanes.
                    assert!(T::FIXED_BITS.is_some());
                    assert!(lane_scales[lane] != 1. && lane_scales[lane] != 4e102);
                }
            }
        }
    }
}

#[test]
fn native_triangle_bubble_overflow_remains_visible() {
    correction(0., |x| x);
    correction(DoubleFloat::from(0.), DoubleFloat::from);
    correction(Float::new(256), |x| Float::with_val(256, x));
}
