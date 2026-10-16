//! Cached completed conditionals versus existing generated scalar masters and original fixtures.
#![cfg(feature = "generated-evaluators")]
#[path = "support/fixtures.rs"]
mod fixtures;

use oneloop::{
    A0, B0, C0, D0, NativeEvaluator, NativeFloat, OneLoopExpressions, ScalarIntegral, dB0,
};
use symbolica::{
    atom::{Atom, AtomCore},
    domains::{
        float::{Complex, DoubleFloat, Float, RealLike},
        rational::Rational,
    },
    evaluate::{EvaluationDomain, ExpressionEvaluator},
};

fn compare<T: NativeFloat + RealLike>(
    exact: &ExpressionEvaluator<Complex<Rational>>,
    rows: &[&fixtures::Fixture],
    prototype: T,
    from_f64: impl Fn(f64) -> T,
    tolerance: f64,
) where
    Complex<T>: EvaluationDomain,
{
    let map = |value: &Complex<Rational>| {
        Complex::new(
            prototype.from_rational(&value.re),
            prototype.from_rational(&value.im),
        )
    };
    let points = rows
        .iter()
        .flat_map(|r| {
            r.args
                .iter()
                .map(|v| Complex::new(from_f64(v.re), from_f64(v.im)))
        })
        .collect::<Vec<_>>();
    let arity = rows[0].family.arity();
    let zero = Complex::new(prototype.zero(), prototype.zero());
    let mut baseline =
        NativeEvaluator::<T>::with_binary_precision(rows[0].family, prototype.get_precision())
            .unwrap();
    let mut expected = vec![zero.clone(); rows.len() * 3];
    for (input, output) in points.chunks(arity).zip(expected.chunks_mut(3)) {
        baseline.evaluate(input, output).unwrap();
    }
    let mut failures = Vec::new();
    let mut max_error = 0.0_f64;
    {
        let mode = 0;
        let graph = exact;
        let scalar = graph
            .clone()
            .map_coeff_with_prec(&map, prototype.get_precision());
        let mut batch = scalar.clone().into_batch_evaluator();
        let mut scalar = scalar;
        let mut scalar_values = vec![zero.clone(); rows.len() * 3];
        for (input, output) in points.chunks(arity).zip(scalar_values.chunks_mut(3)) {
            scalar.evaluate(input, output);
        }
        for width in [0, 1, 3, 31, 256] {
            let mut values = vec![zero.clone(); rows.len() * 3];
            if width == 0 {
                batch.evaluate_batch(&[], &mut [], 0).unwrap();
                continue;
            }
            for (input, output) in points
                .chunks(arity * width)
                .zip(values.chunks_mut(3 * width))
            {
                batch
                    .evaluate_batch(input, output, input.len() / arity)
                    .unwrap();
            }
            for (i, ((actual, expected), scalar)) in
                values.iter().zip(&expected).zip(&scalar_values).enumerate()
            {
                for (part, a, e, s) in [
                    ("real", &actual.re, &expected.re, &scalar.re),
                    ("imag", &actual.im, &expected.im, &scalar.im),
                ] {
                    let norm = e.norm().to_f64().max(s.norm().to_f64()).max(1.0);
                    let error = (a.clone() - e).norm().to_f64() / norm;
                    let batch_error = (a.clone() - s).norm().to_f64() / norm;
                    max_error = max_error.max(error);
                    if !a.is_finite()
                        || !e.is_finite()
                        || !s.is_finite()
                        || !error.is_finite()
                        || error > tolerance
                        || batch_error > tolerance
                    {
                        failures.push(format!("mode{mode} width{width} {} coeff{} {part}: error={error:e} batch_error={batch_error:e} actual={a} baseline={e} scalar={s}", rows[i/3].name, i%3));
                    }
                }
            }
            for (row, value) in rows.iter().zip(values.chunks(3)) {
                let converted = value
                    .iter()
                    .map(|v| Complex::new(v.re.to_f64(), v.im.to_f64()))
                    .collect::<Vec<_>>();
                failures.extend(
                    row.failures(&converted)
                        .into_iter()
                        .map(|s| format!("mode{mode} width{width}: {s}")),
                );
            }
        }
    }
    eprintln!(
        "{} {} rows precision={} bits max_scaled_native_error={max_error:e}",
        rows[0].family.name(),
        rows.len(),
        prototype.get_precision()
    );
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn cached_conditionals_match_scalar_native_and_batches_in_three_precisions() {
    std::thread::Builder::new()
        .stack_size(128 * 1024 * 1024)
        .spawn(|| {
            let context = OneLoopExpressions::new();
            let fixtures = fixtures::parse(include_str!("data/parity.txt"));
            assert_eq!(fixtures.len(), 293);
            for (family, symbol) in [
                (ScalarIntegral::A0, A0()),
                (ScalarIntegral::B0, B0()),
                (ScalarIntegral::DB0, dB0()),
                (ScalarIntegral::C0, C0()),
                (ScalarIntegral::D0, D0()),
            ] {
                let parameters = (0..family.arity())
                    .map(|j| {
                        symbolica::symbol!(format!("opt_numeric_{}_{}", family.name(), j)).to_atom()
                    })
                    .collect::<Vec<_>>();
                let calls = [0, -1, -2].map(|tag| {
                    let mut args = vec![Atom::num(tag)];
                    args.extend(parameters.iter().cloned());
                    symbol.call(&args)
                });
                let exact = {
                    Atom::evaluator_multiple(&calls, &parameters)
                        .function_map(context.function_map().clone())
                        .direct_translation(true)
                        .horner_iterations(0)
                        .build()
                        .unwrap()
                };
                let rows = fixtures
                    .iter()
                    .filter(|r| r.family == family)
                    .collect::<Vec<_>>();
                compare(&exact, &rows, 0.0, |x| x, 1e-9);
                compare(
                    &exact,
                    &rows,
                    DoubleFloat::from(0.0),
                    DoubleFloat::from,
                    1e-22,
                );
                compare(
                    &exact,
                    &rows,
                    Float::new(256),
                    |x| Float::with_val(256, x),
                    1e-60,
                );
            }
        })
        .unwrap()
        .join()
        .unwrap();
}
