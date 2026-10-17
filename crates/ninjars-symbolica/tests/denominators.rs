use ninjars::{
    BubbleCutRequest, Complex, CutFailure, CutOutcome, CutProvider, ExpansionKind, FloatLike,
    FourMomentum, Real, SampleId, TadpoleCutRequest, TriangleCutRequest,
};
use ninjars_symbolica::{DenominatorExpansion, SymbolicaCuts, SymbolicaDualCuts};
use symbolica::domains::dual::{DualNumber, HyperDual};

fn c(value: f64) -> Complex<f64> {
    Complex::new(value, 0.)
}

fn momentum<T: Real>(prototype: &T, components: [i64; 4]) -> FourMomentum<T> {
    FourMomentum::from_args(
        prototype.from_i64(components[0]),
        prototype.from_i64(components[1]),
        prototype.from_i64(components[2]),
        prototype.from_i64(components[3]),
    )
}

fn append_vector<T: Clone>(row: &mut Vec<T>, vector: &FourMomentum<T>) {
    row.extend(vector.components().into_iter().cloned());
}

/// Assemble the documented callback input rows using independently generated
/// on-shell geometry. Return each row with its directly sampled loop momentum.
fn rows_on_cut<T: Real>(
    provider: &mut impl CutProvider<T>,
    cut_size: usize,
    shifts: &[FourMomentum<T>; 4],
    masses: &[T; 4],
    t: &T,
    x: &T,
    mu: &T,
) -> Vec<(Vec<T>, FourMomentum<T>)> {
    let id = SampleId { lane: 0, sample: 0 };
    let mut rows = Vec::new();
    let mut add = |vectors: &[&FourMomentum<T>], parameters: &[T]| {
        let q = if cut_size == 2 {
            vectors[0].clone()
                + vectors[1].clone() * x.clone()
                + vectors[2].clone() * t.clone()
                + vectors[3].clone()
                    * ((parameters[0].clone()
                        + parameters[1].clone() * x
                        + parameters[2].clone() * x * x
                        + mu)
                        / t)
        } else {
            vectors[0].clone()
                + vectors[1].clone() * t.clone()
                + vectors[2].clone() * ((parameters[0].clone() + mu) / t)
        };
        let mut row = Vec::new();
        for vector in vectors {
            append_vector(&mut row, vector);
        }
        row.extend_from_slice(parameters);
        append_vector(&mut row, &shifts[0]);
        row.push(masses[0].clone());
        append_vector(&mut row, &shifts[3]);
        row.push(masses[3].clone());
        rows.push((row, q));
    };
    match cut_size {
        1 => {
            let request = TadpoleCutRequest {
                id,
                shift: &shifts[0],
                mass_squared: &masses[0],
                complementary_shifts: &shifts[1..],
            };
            let mut out = [CutOutcome::Unstable(CutFailure::NonFiniteResult)];
            provider.tadpole_batch(&[request], &mut out).unwrap();
            let CutOutcome::Stable(cut) = &out[0] else {
                panic!("unstable tadpole fixture")
            };
            for (direction, effective) in [
                (&cut.basis.e1, &cut.e2_effective),
                (&cut.basis.e3, &cut.e4_effective),
            ] {
                add(
                    &[&cut.a, direction, effective],
                    std::slice::from_ref(&cut.beta),
                );
            }
        }
        2 => {
            let request = BubbleCutRequest {
                id,
                shifts: [&shifts[0], &shifts[1]],
                masses_squared: [&masses[0], &masses[1]],
                complementary_shifts: &shifts[2..],
            };
            let mut out = [CutOutcome::Unstable(CutFailure::NonFiniteResult)];
            provider.bubble_batch(&[request], &mut out).unwrap();
            let CutOutcome::Stable(cut) = &out[0] else {
                panic!("unstable bubble fixture")
            };
            let swapped = cut.basis.e3.clone()
                * (-cut.basis.e12.from_i64(1)
                    / (cut.basis.e12.clone() * cut.basis.e12.from_i64(2)));
            for (direction, effective) in [
                (&cut.basis.e3, &cut.e4_effective),
                (&cut.basis.e4, &swapped),
            ] {
                add(&[&cut.a0, &cut.a1, direction, effective], &cut.parameters);
            }
        }
        3 => {
            let request = TriangleCutRequest {
                id,
                shifts: std::array::from_fn(|i| &shifts[i]),
                masses_squared: std::array::from_fn(|i| &masses[i]),
            };
            let mut out = [CutOutcome::Unstable(CutFailure::NonFiniteResult)];
            provider.triangle_batch(&[request], &mut out).unwrap();
            let CutOutcome::Stable(cut) = &out[0] else {
                panic!("unstable triangle fixture")
            };
            let swapped = cut.basis.e3.clone()
                * (-cut.basis.e12.from_i64(1)
                    / (cut.basis.e12.clone() * cut.basis.e12.from_i64(2)));
            for (direction, effective) in [
                (&cut.basis.e3, &cut.e4_effective),
                (&cut.basis.e4, &swapped),
            ] {
                add(
                    &[&cut.a, direction, effective],
                    std::slice::from_ref(&cut.beta),
                );
            }
        }
        _ => unreachable!(),
    }
    rows
}

fn reconstruct<T: Real>(kind: &ExpansionKind, coefficients: &[T], t: &T, x: &T, mu: &T) -> T {
    // Keep this direct arithmetic oracle independent of upstream pow(0), whose
    // dynamic/static Dual regression was exposed by this comparison.
    let positive_power = |value: &T, exponent| {
        if exponent == 0 {
            value.one()
        } else {
            value.pow(exponent)
        }
    };
    let powers: Vec<_> = match kind {
        ExpansionKind::T3 { powers } => powers.iter().map(|&(t, mu)| (t, 0, mu)).collect(),
        ExpansionKind::T2 { powers } => powers.clone(),
        _ => unreachable!(),
    };
    coefficients
        .iter()
        .zip(powers)
        .fold(t.zero(), |total, (coefficient, (tp, xp, mp))| {
            let t_power = if tp < 0 {
                t.inv().pow(tp.unsigned_abs().into())
            } else {
                positive_power(t, tp as u64)
            };
            total
                + coefficient.clone()
                    * t_power
                    * positive_power(x, xp.into())
                    * positive_power(mu, mp.into())
        })
}

fn check(actual: Complex<f64>, expected: Complex<f64>) {
    assert!(
        (actual - expected).norm().re < 1e-9 * (1. + expected.norm().re),
        "actual {actual}, expected {expected}"
    );
}

#[test]
fn denominator_coefficients_reconstruct_on_all_cut_types_and_orientations() {
    let mut provider = SymbolicaCuts::new(0.).unwrap();
    let masses = [
        Complex::new(1., -0.1),
        Complex::new(2., -0.25),
        Complex::new(3., -0.5),
        Complex::new(5., -0.2),
    ];
    for cut_size in 1..=3 {
        let plan = DenominatorExpansion::new(cut_size).unwrap();
        let mut evaluator = plan.compile(&0.).unwrap();
        let mut inputs = Vec::new();
        let mut expectations = Vec::new();
        for spatial in [false, true] {
            let origin = momentum(&c(0.), [1, -2, 3, -1]);
            let shifts = [
                origin,
                origin + momentum(&c(0.), if spatial { [0, 1, 0, 0] } else { [2, 1, 0, 0] }),
                origin + momentum(&c(0.), if spatial { [0, 0, 1, 0] } else { [3, 0, 1, 0] }),
                origin + FourMomentum::from_args(Complex::new(4., 0.2), c(1.), c(-2.), c(3.)),
            ];
            for (t, x, mu) in [
                (Complex::new(2., 0.3), Complex::new(-1., 0.2), c(0.)),
                (Complex::new(-3., 0.7), c(2.), Complex::new(1., -0.4)),
            ] {
                for (row, q) in rows_on_cut(&mut provider, cut_size, &shifts, &masses, &t, &x, &mu)
                {
                    assert_eq!(row.len(), plan.input_columns());
                    for line in 0..cut_size {
                        check((q + shifts[line]).mass_squared() - masses[line] - mu, c(0.));
                    }
                    inputs.extend(row);
                    expectations.push((t, x, mu, (q + shifts[3]).mass_squared() - masses[3] - mu));
                }
            }
        }
        let width = plan.expansion().len();
        let mut output = vec![c(0.); expectations.len() * width];
        evaluator
            .evaluate_batch(&inputs, &mut output, expectations.len())
            .unwrap();
        for ((t, x, mu, expected), coefficients) in
            expectations.iter().zip(output.chunks_exact(width))
        {
            check(
                reconstruct(plan.expansion(), coefficients, t, x, mu),
                *expected,
            );
        }
        for order in [&[7, 0, 3][..], &[1][..], &[][..]] {
            let reordered: Vec<_> = order
                .iter()
                .flat_map(|&lane| {
                    let start = lane * plan.input_columns();
                    inputs[start..start + plan.input_columns()].iter().copied()
                })
                .collect();
            let mut result = vec![c(0.); order.len() * width];
            evaluator
                .evaluate_batch(&reordered, &mut result, order.len())
                .unwrap();
            for (&lane, row) in order.iter().zip(result.chunks_exact(width)) {
                assert_eq!(row, &output[lane * width..(lane + 1) * width]);
            }
        }
        let mut unchanged = output[..width].to_vec();
        let before = unchanged.clone();
        assert!(
            evaluator
                .evaluate_batch(&inputs[..plan.input_columns() - 1], &mut unchanged, 1)
                .is_err()
        );
        assert_eq!(unchanged, before);
    }
}

#[test]
fn dual_denominators_match_direct_higher_and_mixed_cut_derivatives() {
    let prototype = HyperDual::from_values(
        vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![2, 0], vec![1, 1]],
        vec![c(0.); 5],
    );
    let mut provider = SymbolicaDualCuts::new(0., prototype.clone()).unwrap();
    let origin = momentum(&prototype, [1, -2, 3, -1]);
    let mut shifts = [
        origin.clone(),
        origin.clone() + momentum(&prototype, [2, 1, 0, 0]),
        origin.clone() + momentum(&prototype, [3, 0, 1, 0]),
        origin + momentum(&prototype, [4, 1, -2, 3]),
    ];
    shifts[1].temporal.value.values_mut()[1] = Complex::new(1., 0.2);
    shifts[3].spatial.px.values_mut()[2] = Complex::new(1., -0.1);
    let mut masses = std::array::from_fn(|i| prototype.from_i64(i as i64 + 1));
    masses[1].values_mut()[2] = c(1.);
    let mut t = prototype.from_i64(2);
    t.values_mut()[1] = c(1.);
    let x = prototype.from_i64(-1);
    let mut mu = prototype.from_i64(1);
    mu.values_mut()[2] = c(1.);
    for cut_size in 1..=3 {
        let plan = DenominatorExpansion::new(cut_size).unwrap();
        let mut evaluator = plan.compile_dual(&0., &prototype).unwrap();
        let rows = rows_on_cut(&mut provider, cut_size, &shifts, &masses, &t, &x, &mu);
        let inputs: Vec<_> = rows
            .iter()
            .flat_map(|(row, _)| row.iter().cloned())
            .collect();
        let mut outputs = vec![prototype.clone(); rows.len() * plan.expansion().len()];
        evaluator
            .evaluate_batch(&inputs, &mut outputs, rows.len())
            .unwrap();
        let mut primal_evaluator = plan.compile(&0.).unwrap();
        let primal_inputs: Vec<_> = inputs.iter().map(|v| v.values()[0]).collect();
        let mut primal_outputs = vec![c(0.); outputs.len()];
        primal_evaluator
            .evaluate_batch(&primal_inputs, &mut primal_outputs, rows.len())
            .unwrap();
        for (index, (value, expected)) in outputs.iter().zip(&primal_outputs).enumerate() {
            assert!(
                (value.values()[0] - *expected).norm().re < 1e-10,
                "cut {cut_size} coefficient {index}, Dual {:?}, primal {expected}",
                value.values()
            );
        }
        for ((_, q), coefficients) in rows
            .iter()
            .zip(outputs.chunks_exact(plan.expansion().len()))
        {
            for line in 0..cut_size {
                let residual = (q.clone() + &shifts[line]).mass_squared() - &masses[line] - &mu;
                for component in residual.values() {
                    check(*component, c(0.));
                }
            }
            let expected = (q.clone() + &shifts[3]).mass_squared() - &masses[3] - &mu;
            let result = reconstruct(plan.expansion(), coefficients, &t, &x, &mu);
            for (component, (actual, expected)) in
                result.values().iter().zip(expected.values()).enumerate()
            {
                assert!(
                    (*actual - *expected).norm().re < 1e-9 * (1. + expected.norm().re),
                    "cut {cut_size}, component {component}, actual {actual}, expected {expected}"
                );
            }
            assert!(expected.values()[4].norm().re > 1e-5);
        }
        evaluator.evaluate_batch(&[], &mut [], 0).unwrap();
    }
}
