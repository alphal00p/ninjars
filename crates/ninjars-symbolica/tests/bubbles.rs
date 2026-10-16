//! Coefficient-stage double-cut controls, independent of master contraction.

use ninjars::{
    BubbleDivisionRequest, BubbleReducer, BubbleReductionKernels, BubbleSampleRequest, Complex,
    CutFailure, CutOutcome, CutSampler, DoubleFloat, Float, FloatLike, FourMomentum, Numerator,
    NumeratorRequest, Real, SampleId, SingleFloat, Topology, TriangleReducer,
};
use ninjars_symbolica::{
    DualPolynomialNumerator, PolynomialNumerator, SymbolicaBubbleKernels, SymbolicaCuts,
    SymbolicaDualBubbleKernels, SymbolicaDualCuts, SymbolicaDualTriangleKernels,
    SymbolicaTriangleKernels,
};
use symbolica::{
    atom::Atom,
    domains::{dual::HyperDual, rational::Rational},
    evaluate::EvaluationDomain,
    parse, symbol,
};

fn c(v: f64) -> Complex<f64> {
    Complex::new(v, 0.)
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
fn shift(p: [i64; 4]) -> FourMomentum<Complex<f64>> {
    FourMomentum::from_args(
        c(p[0] as f64),
        c(p[1] as f64),
        c(p[2] as f64),
        c(p[3] as f64),
    )
}
fn near(a: Complex<f64>, b: Complex<f64>) {
    assert!(
        (a.re - b.re).abs() <= 1e-9 * (1. + b.re.abs()),
        "{a} != {b}"
    );
    assert!(
        (a.im - b.im).abs() <= 1e-9 * (1. + b.im.abs()),
        "{a} != {b}"
    );
}
fn denominator(p: [i64; 4], mass: i64) -> Atom {
    let q = vars();
    let v = FourMomentum::from_args(&q[0] + p[0], &q[1] + p[1], &q[2] + p[2], &q[3] + p[3]);
    v.dot(&v) - mass - &q[4]
}
fn bubble_ansatz() -> PolynomialNumerator<f64> {
    PolynomialNumerator::new(parse!("c0+c9*mu2+c1*q0+c2*q0^2+c3*q1+c4*q1^2+c5*q2+c6*q2^2+c7*q0*q1+c8*q0*q2+mu2*(c10*q0+c11*q1+c12*q2)+c13*q0^3+c14*q1^3+c15*q2^3+c16*q0^2*q1+c17*q0^2*q2+c18*q0*q1^2+c19*q0*q2^2"),vars(),(0..20).map(|i|Atom::var(symbol!(format!("c{i}")))).collect(),0.).unwrap()
}
fn triangle_ansatz() -> PolynomialNumerator<f64> {
    PolynomialNumerator::new(parse!("c0+c7*mu2+(c1+c8*mu2)*q0+c2*q0^2+c3*q0^3+(c4+c9*mu2)*q1+c5*q1^2+c6*q1^3+c14*mu2^2+c10*mu2*q0^2+c12*q0^4+c11*mu2*q1^2+c13*q1^4"),vars(),(0..15).map(|i|Atom::var(symbol!(format!("c{i}")))).collect(),0.).unwrap()
}

fn numeric_requests<'a>(
    points: &'a [FourMomentum<Complex<f64>>],
    samples: &'a [(Complex<f64>, Complex<f64>, Complex<f64>)],
    partition: &'a [usize],
) -> Vec<NumeratorRequest<'a, Complex<f64>>> {
    points
        .iter()
        .zip(samples)
        .enumerate()
        .map(|(sample, (q, (_, _, mu)))| NumeratorRequest {
            id: SampleId { lane: 0, sample },
            q,
            mu_squared: mu,
            partition,
        })
        .collect()
}

#[test]
fn polynomial_residues_reconstruct_with_zero_one_and_two_triangle_parents() {
    let routing = [[1, -1, 2, 0], [3, 0, 2, 0], [4, -1, 3, 0], [5, 0, 2, 1]];
    let samples = [
        (c(2.), c(1.), c(0.)),
        (Complex::new(-1., 0.5), c(-2.), c(1.)),
        (c(3.), Complex::new(1., 0.25), Complex::new(2., -0.5)),
    ];
    for (n, expression) in [
        (2, parse!("1+q0+2*q1^2+3*q2^3+q0*q1*q3+mu2*(q0+2*q1+3)")),
        (
            3,
            parse!("1+q0+2*q1^2+3*q2^3+q3^4+2*q0*q1*q2*q3+mu2*(q0^2+2*q1*q2+q3)+3*mu2^2"),
        ),
        // The quadruple-cut residue vanishes, but two distinct triangle
        // parents contribute to target bubble [0,1]. This tests their sum.
        (
            4,
            denominator(routing[2], 3) * denominator(routing[3], 4) * 3
                + parse!("q0^3") * denominator(routing[3], 4)
                + parse!("q1^3") * denominator(routing[2], 3),
        ),
    ] {
        let mut numerator = PolynomialNumerator::new(expression, vars(), vec![], 0.).unwrap();
        numerator.set_parameter_batch(&[], 1).unwrap();
        let rank = numerator.rank();
        assert_eq!(rank, n + 1);
        let topology = Topology::new(n, rank).unwrap();
        let shifts: Vec<_> = routing[..n].iter().copied().map(shift).collect();
        let masses: Vec<_> = (1..=n).map(|i| c(i as f64)).collect();
        let mut geometry = SymbolicaCuts::new(0.).unwrap();
        let mut triangles = Vec::new();
        if n >= 3 {
            let mut reducer =
                TriangleReducer::new(SymbolicaTriangleKernels::new(n, rank, 0.).unwrap(), c(0.))
                    .unwrap();
            triangles.resize(
                reducer.cut_count(),
                CutOutcome::Unstable(CutFailure::NonFiniteResult),
            );
            reducer
                .evaluate_batch(
                    &shifts,
                    &masses,
                    1,
                    &mut numerator,
                    &mut geometry,
                    &mut triangles,
                )
                .unwrap();
            assert!(triangles.iter().all(|r| matches!(r, CutOutcome::Stable(_))));
        }
        let mut reducer =
            BubbleReducer::new(SymbolicaBubbleKernels::new(n, rank, 0.).unwrap(), c(0.)).unwrap();
        let mut residues =
            vec![CutOutcome::Unstable(CutFailure::NonFiniteResult); reducer.cut_count()];
        reducer
            .evaluate_batch(
                &shifts,
                &masses,
                1,
                &topology,
                &triangles,
                &mut numerator,
                &mut geometry,
                &mut residues,
            )
            .unwrap();
        for result in residues {
            let CutOutcome::Stable(residue) = result else {
                panic!("unexpected unstable n={n} bubble")
            };
            assert_eq!(residue.coefficients.len(), 20);
            let partition = &topology.cuts()[residue.cut_index].partition;
            let requests: Vec<_> = samples
                .iter()
                .enumerate()
                .map(|(sample, (t, x, mu))| BubbleSampleRequest {
                    id: SampleId { lane: 0, sample },
                    cut: &residue.geometry,
                    t,
                    x,
                    mu_squared: mu,
                })
                .collect();
            let mut sampled =
                vec![CutOutcome::Unstable(CutFailure::NonFiniteResult); samples.len()];
            geometry
                .sample_bubble_batch(&requests, &mut sampled)
                .unwrap();
            let q: Vec<_> = sampled
                .into_iter()
                .map(|r| match r {
                    CutOutcome::Stable(q) => q,
                    _ => panic!("unstable sample"),
                })
                .collect();
            let coordinates: Vec<_> = q
                .iter()
                .map(|q| {
                    let r = *q + shifts[partition[0]];
                    FourMomentum::from_args(
                        r.dot(&residue.geometry.basis.e2),
                        r.dot(&residue.geometry.basis.e3),
                        r.dot(&residue.geometry.basis.e4),
                        c(0.),
                    )
                })
                .collect();
            let mut bubble = bubble_ansatz();
            bubble
                .set_parameter_batch(&residue.coefficients, 1)
                .unwrap();
            let mut got = vec![c(0.); samples.len()];
            bubble
                .evaluate_batch(
                    &numeric_requests(&coordinates, &samples, partition),
                    &mut got,
                )
                .unwrap();
            let complement = &topology.cuts()[residue.cut_index].complement;
            let mut denominator_values = vec![vec![c(1.); samples.len()]; n];
            for &extra in complement {
                let mut denominator = PolynomialNumerator::new(
                    denominator(routing[extra], extra as i64 + 1),
                    vars(),
                    vec![],
                    0.,
                )
                .unwrap();
                denominator.set_parameter_batch(&[], 1).unwrap();
                denominator
                    .evaluate_batch(
                        &numeric_requests(&q, &samples, partition),
                        &mut denominator_values[extra],
                    )
                    .unwrap();
                for (value, d) in got.iter_mut().zip(&denominator_values[extra]) {
                    *value *= d;
                }
            }
            for source in &triangles {
                let CutOutcome::Stable(source) = source else {
                    unreachable!()
                };
                let source_partition = &topology.cuts()[source.cut_index].partition;
                if !partition.iter().all(|i| source_partition.contains(i)) {
                    continue;
                }
                let origin = source_partition[0];
                let coordinates: Vec<_> = q
                    .iter()
                    .map(|q| {
                        let r = *q + shifts[origin];
                        FourMomentum::from_args(
                            r.dot(&source.geometry.basis.e3),
                            r.dot(&source.geometry.basis.e4),
                            c(0.),
                            c(0.),
                        )
                    })
                    .collect();
                let mut triangle = triangle_ansatz();
                triangle
                    .set_parameter_batch(&source.coefficients, 1)
                    .unwrap();
                let mut triangle_values = vec![c(0.); samples.len()];
                triangle
                    .evaluate_batch(
                        &numeric_requests(&coordinates, &samples, partition),
                        &mut triangle_values,
                    )
                    .unwrap();
                for &extra in complement {
                    if !source_partition.contains(&extra) {
                        for (value, d) in triangle_values.iter_mut().zip(&denominator_values[extra])
                        {
                            *value *= d;
                        }
                    }
                }
                // Independent on-cut identity: N=sum residues*uncut denominators.
                for (value, triangle) in got.iter_mut().zip(triangle_values) {
                    *value += triangle;
                }
            }
            let mut expected = vec![c(0.); samples.len()];
            numerator
                .evaluate_batch(&numeric_requests(&q, &samples, partition), &mut expected)
                .unwrap();
            for (a, b) in got.into_iter().zip(expected) {
                near(a, b);
            }
        }
    }
}

#[test]
fn bubble_batches_preserve_parameter_lanes_and_explicit_mu_squared() {
    let mut numerator = PolynomialNumerator::new(
        parse!("a*mu2+b"),
        vars(),
        vec![parse!("a"), parse!("b")],
        0.,
    )
    .unwrap();
    let mut reducer =
        BubbleReducer::new(SymbolicaBubbleKernels::new(2, 2, 0.).unwrap(), c(0.)).unwrap();
    let topology = Topology::new(2, 2).unwrap();
    let mut geometry = SymbolicaCuts::new(0.).unwrap();
    let points = [
        [shift([0, 0, 0, 0]), shift([2, 1, 0, 0])],
        [shift([1, -1, 2, 0]), shift([3, 0, 2, 0])],
        [shift([0, 0, 0, 0]); 2],
    ];
    let parameters = [
        [c(2.), c(3.)],
        [Complex::new(-1., 0.25), c(7.)],
        [c(13.), c(17.)],
    ];
    for order in [&[2, 0, 1][..], &[1][..], &[][..], &[1, 0, 1, 0, 2][..]] {
        let shifts: Vec<_> = order.iter().flat_map(|&i| points[i]).collect();
        let masses = vec![c(1.); shifts.len()];
        let values: Vec<_> = order.iter().flat_map(|&i| parameters[i]).collect();
        numerator.set_parameter_batch(&values, order.len()).unwrap();
        let mut output = vec![CutOutcome::Unstable(CutFailure::NonFiniteResult); order.len()];
        reducer
            .evaluate_batch(
                &shifts,
                &masses,
                order.len(),
                &topology,
                &[],
                &mut numerator,
                &mut geometry,
                &mut output,
            )
            .unwrap();
        for (lane, (&source, result)) in order.iter().zip(output).enumerate() {
            if source == 2 {
                assert!(matches!(result, CutOutcome::Unstable(_)));
                continue;
            }
            let CutOutcome::Stable(residue) = result else {
                panic!("stable bubble lane failed")
            };
            assert_eq!(residue.id.lane, lane);
            for (i, value) in residue.coefficients.into_iter().enumerate() {
                near(
                    value,
                    match i {
                        0 => parameters[source][1],
                        9 => parameters[source][0],
                        _ => c(0.),
                    },
                );
            }
        }
    }
}

#[test]
fn unstable_or_misidentified_parent_cannot_be_treated_as_zero() {
    let mut numerator = PolynomialNumerator::new(parse!("1+q0"), vars(), vec![], 0.).unwrap();
    numerator.set_parameter_batch(&[], 2).unwrap();
    let shifts: Vec<_> = [[0, 0, 0, 0], [2, 1, 0, 0], [3, 0, 1, 0]]
        .into_iter()
        .map(shift)
        .cycle()
        .take(6)
        .collect();
    let masses = vec![c(1.); 6];
    let mut geometry = SymbolicaCuts::new(0.).unwrap();
    let mut triangle =
        TriangleReducer::new(SymbolicaTriangleKernels::new(3, 1, 0.).unwrap(), c(0.)).unwrap();
    let mut parents = vec![CutOutcome::Unstable(CutFailure::NonFiniteResult); 2];
    triangle
        .evaluate_batch(
            &shifts,
            &masses,
            2,
            &mut numerator,
            &mut geometry,
            &mut parents,
        )
        .unwrap();
    assert!(parents.iter().all(|p| matches!(p, CutOutcome::Stable(_))));
    parents[1] = CutOutcome::Unstable(CutFailure::DegenerateGram);
    let mut reducer =
        BubbleReducer::new(SymbolicaBubbleKernels::new(3, 1, 0.).unwrap(), c(0.)).unwrap();
    let mut output =
        vec![CutOutcome::Unstable(CutFailure::SingularParameter); reducer.cut_count() * 2];
    reducer
        .evaluate_batch(
            &shifts,
            &masses,
            2,
            triangle.topology(),
            &parents,
            &mut numerator,
            &mut geometry,
            &mut output,
        )
        .unwrap();
    for rows in output.as_chunks::<2>().0 {
        assert!(matches!(rows[0], CutOutcome::Stable(_)));
        assert!(matches!(
            rows[1],
            CutOutcome::Unstable(CutFailure::DegenerateGram)
        ));
    }
    let CutOutcome::Stable(parent) = &mut parents[0] else {
        unreachable!()
    };
    parent.id.lane = 1;
    output.fill(CutOutcome::Unstable(CutFailure::SingularParameter));
    assert!(
        reducer
            .evaluate_batch(
                &shifts,
                &masses,
                2,
                triangle.topology(),
                &parents,
                &mut numerator,
                &mut geometry,
                &mut output
            )
            .is_err()
    );
    assert!(
        output
            .iter()
            .all(|r| matches!(r, CutOutcome::Unstable(CutFailure::SingularParameter)))
    );
    assert!(
        reducer
            .evaluate_batch(
                &shifts,
                &masses,
                2,
                &Topology::new(3, 2).unwrap(),
                &parents,
                &mut numerator,
                &mut geometry,
                &mut output
            )
            .is_err()
    );
    assert!(
        output
            .iter()
            .all(|r| matches!(r, CutOutcome::Unstable(CutFailure::SingularParameter)))
    );
}

#[test]
fn higher_and_mixed_dual_parent_subtraction_preserves_the_bubble_coefficient() {
    let shape = vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![2, 0], vec![1, 1]];
    let prototype = HyperDual::from_values(shape.clone(), vec![c(0.); 5]);
    let dual = |values: [f64; 5]| HyperDual::from_values(shape.clone(), values.map(c).to_vec());
    let lift = |value: Complex<f64>| {
        let mut d = prototype.zero();
        d.values[0] = value;
        d
    };
    let routing = [[1, -1, 2, 0], [3, 0, 2, 0], [4, -1, 3, 0]];
    let expression = parse!("a") * denominator(routing[2], 3) + parse!("b*mu2");
    let source =
        PolynomialNumerator::new(expression, vars(), vec![parse!("a"), parse!("b")], 0.).unwrap();
    let mut numerator = DualPolynomialNumerator::new(source, prototype.clone()).unwrap();
    let a = dual([3., 2., -1., 5., 7.]);
    numerator
        .set_parameter_batch(&[a.clone(), dual([2., -3., 4., -2., 1.])], 1)
        .unwrap();
    let mut shifts: Vec<_> = routing.into_iter().map(|p| shift(p).map(lift)).collect();
    shifts[0].temporal.value.values[1] = c(1.);
    let mut masses = vec![
        dual([1., 0., 0., 0., 0.]),
        dual([2., 0., 0., 0., 0.]),
        dual([3., 0., 0., 0., 0.]),
    ];
    masses[0].values[2] = c(1.);
    let mut geometry = SymbolicaDualCuts::new(0., prototype.clone()).unwrap();
    let mut triangle = TriangleReducer::new(
        SymbolicaDualTriangleKernels::new(3, 2, 0., &prototype).unwrap(),
        prototype.clone(),
    )
    .unwrap();
    let mut parents = vec![CutOutcome::Unstable(CutFailure::NonFiniteResult); 1];
    triangle
        .evaluate_batch(
            &shifts,
            &masses,
            1,
            &mut numerator,
            &mut geometry,
            &mut parents,
        )
        .unwrap();
    let mut reducer = BubbleReducer::new(
        SymbolicaDualBubbleKernels::new(3, 2, 0., &prototype).unwrap(),
        prototype,
    )
    .unwrap();
    let mut output = vec![CutOutcome::Unstable(CutFailure::NonFiniteResult); reducer.cut_count()];
    reducer
        .evaluate_batch(
            &shifts,
            &masses,
            1,
            triangle.topology(),
            &parents,
            &mut numerator,
            &mut geometry,
            &mut output,
        )
        .unwrap();
    for result in output {
        let CutOutcome::Stable(residue) = result else {
            panic!("unexpected unstable dual bubble")
        };
        let target = reducer.topology().cuts()[residue.cut_index].partition == [0, 1];
        for (i, value) in residue.coefficients.iter().enumerate() {
            for (actual, expected) in value.values.iter().zip(&a.values) {
                near(*actual, if target && i == 0 { *expected } else { c(0.) });
            }
        }
    }
}

fn precise_bubble<T: Real + SingleFloat>(prototype: T)
where
    Complex<T>: Real + EvaluationDomain,
{
    let zero = Complex::new(prototype.zero(), prototype.zero());
    let lift = |value: i64| Complex::new(prototype.from_i64(value), prototype.zero());
    let a = Complex::new(
        prototype.from_rational(&Rational::from((1, 3))),
        prototype.zero(),
    );
    let b = Complex::new(
        prototype.one() + prototype.from_rational(&Rational::from((1, 10)).pow(30)),
        prototype.zero(),
    );
    assert!(b.re != prototype.one());
    let mut numerator = PolynomialNumerator::new(
        parse!("a*mu2+b"),
        vars(),
        vec![parse!("a"), parse!("b")],
        prototype.clone(),
    )
    .unwrap();
    numerator
        .set_parameter_batch(&[a.clone(), b.clone()], 1)
        .unwrap();
    let shifts = [[1, -1, 2, 0], [3, 0, 2, 0]]
        .map(|v| FourMomentum::from_args(lift(v[0]), lift(v[1]), lift(v[2]), lift(v[3])));
    let masses = [
        Complex::new(
            prototype.one(),
            prototype.from_rational(&Rational::from((-1, 10))),
        ),
        Complex::new(
            prototype.from_i64(2),
            prototype.from_rational(&Rational::from((-1, 5))),
        ),
    ];
    let mut geometry = SymbolicaCuts::new(prototype.clone()).unwrap();
    let mut reducer =
        BubbleReducer::new(SymbolicaBubbleKernels::new(2, 2, prototype).unwrap(), zero).unwrap();
    let mut output = [CutOutcome::Unstable(CutFailure::NonFiniteResult)];
    reducer
        .evaluate_batch(
            &shifts,
            &masses,
            1,
            &Topology::new(2, 2).unwrap(),
            &[],
            &mut numerator,
            &mut geometry,
            &mut output,
        )
        .unwrap();
    let CutOutcome::Stable(residue) = &output[0] else {
        panic!("high-precision complex-mass bubble unstable")
    };
    assert_eq!(residue.coefficients[0], b);
    assert_eq!(residue.coefficients[9], a);
}

#[test]
fn doublefloat_and_arbitrary_precision_complex_mass_bubbles_keep_inputs() {
    precise_bubble(DoubleFloat::default());
    precise_bubble(Float::new(192));
}

#[test]
fn division_isolates_uncut_dual_shapes_and_rejects_duplicate_indices() {
    let prototype = HyperDual::from_values(vec![vec![0], vec![1]], vec![c(0.); 2]);
    let lift = |value| {
        let mut d = prototype.zero();
        d.values[0] = c(value);
        d
    };
    let zero = FourMomentum::from_args(lift(0.), lift(0.), lift(0.), lift(0.));
    let shifts = [
        zero.clone(),
        FourMomentum::from_args(lift(0.), lift(1.), lift(0.), lift(0.)),
        FourMomentum::from_args(lift(0.), lift(0.), lift(0.), lift(1.)),
    ];
    let masses = [lift(1.), lift(1.), lift(1.)];
    let mut malformed = masses.clone();
    malformed[2] = HyperDual::from_values(vec![vec![0]], vec![c(1.)]);
    let vectors = [zero.clone(), zero.clone(), shifts[2].clone(), zero];
    let parameters = [lift(0.), lift(0.), lift(0.)];
    let mut kernels = SymbolicaDualBubbleKernels::new(3, 2, 0., &prototype).unwrap();
    let columns = kernels.numerator_expansion().len();
    let width = kernels.quotient_columns();
    let mut numerator = vec![lift(0.); columns * 2];
    numerator[0] = lift(1.);
    numerator[columns] = lift(1.);
    let sentinel = HyperDual::from_values(vec![vec![0], vec![1]], vec![c(17.), c(19.)]);
    let mut output = vec![sentinel.clone(); width * 2];
    let mut stable = [true, true];
    let requests = [
        BubbleDivisionRequest {
            partition: &[0, 1],
            complement: &[2],
            shifts: &shifts,
            masses_squared: &masses,
            vectors: &vectors,
            parameters: &parameters,
        },
        BubbleDivisionRequest {
            partition: &[0, 1],
            complement: &[2],
            shifts: &shifts,
            masses_squared: &malformed,
            vectors: &vectors,
            parameters: &parameters,
        },
    ];
    kernels
        .divide_batch(&requests, &numerator, &mut output, &mut stable)
        .unwrap();
    assert_eq!(stable, [true, false]);
    assert!(output[width..].iter().all(|v| v.values == sentinel.values));
    let before: Vec<_> = output[..width].iter().map(|v| v.values.clone()).collect();
    let request = BubbleDivisionRequest {
        partition: &[0, 0],
        complement: &[2],
        shifts: &shifts,
        masses_squared: &masses,
        vectors: &vectors,
        parameters: &parameters,
    };
    let mut status = [true];
    assert!(
        kernels
            .divide_batch(
                &[request],
                &numerator[..columns],
                &mut output[..width],
                &mut status
            )
            .is_err()
    );
    assert_eq!(status, [true]);
    assert_eq!(
        before,
        output[..width]
            .iter()
            .map(|v| v.values.clone())
            .collect::<Vec<_>>()
    );
}
