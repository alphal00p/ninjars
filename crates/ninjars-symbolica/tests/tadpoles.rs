//! Minimal integrated tadpole coefficients, including both parent families.
//! Uncomputed spurious coefficients are explicitly masked; this is not a
//! complete single-cut reconstruction test.

use ninjars::{
    BubbleCutRequest, BubbleReducer, BubbleResidue, Complex, CutFailure, CutOutcome, CutProvider,
    DoubleFloat, Float, FourMomentum, IntegratedTadpoleResidue, Numerator, Real, SampleId,
    SingleFloat, TadpoleReducer, Topology, TriangleCutRequest, TriangleReducer, TriangleResidue,
};
use ninjars_symbolica::{
    DualPolynomialNumerator, PolynomialNumerator, SymbolicaBubbleKernels, SymbolicaCuts,
    SymbolicaDualBubbleKernels, SymbolicaDualCuts, SymbolicaDualTadpoleKernels,
    SymbolicaTadpoleKernels, SymbolicaTriangleKernels,
};
use symbolica::{
    atom::{Atom, AtomCore},
    domains::{dual::HyperDual, float::RealLike, rational::Rational},
    evaluate::EvaluationDomain,
    parse, symbol,
};

type C = Complex<f64>;
type Parents = (
    Vec<CutOutcome<TriangleResidue<C>>>,
    Vec<CutOutcome<BubbleResidue<C>>>,
);
fn c(x: f64) -> C {
    Complex::new(x, 0.)
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
fn shift(p: [i64; 4]) -> FourMomentum<C> {
    FourMomentum::from_args(
        c(p[0] as f64),
        c(p[1] as f64),
        c(p[2] as f64),
        c(p[3] as f64),
    )
}
fn symbolic_shift(p: [i64; 4]) -> FourMomentum<Atom> {
    let q = vars();
    FourMomentum::from_args(&q[0] + p[0], &q[1] + p[1], &q[2] + p[2], &q[3] + p[3])
}
fn denominator(p: [i64; 4], mass: i64) -> Atom {
    let p = symbolic_shift(p);
    p.dot(&p) - mass - vars()[4].clone()
}
fn near(a: C, b: C) {
    assert!((a - b).norm().re < 1e-9 * (1. + b.norm().re), "{a} != {b}");
}
fn parents(
    topology: &Topology,
    shifts: &[FourMomentum<C>],
    masses: &[C],
    lanes: usize,
    numerator: &mut PolynomialNumerator<f64>,
    geometry: &mut SymbolicaCuts<f64>,
) -> Parents {
    let n = topology.propagators();
    let rank = topology.rank();
    let mut triangles = vec![];
    if n >= 3 && rank + 3 >= n {
        let mut reducer =
            TriangleReducer::new(SymbolicaTriangleKernels::new(n, rank, 0.).unwrap(), c(0.))
                .unwrap();
        triangles.resize(
            reducer.cut_count() * lanes,
            CutOutcome::Unstable(CutFailure::NonFiniteResult),
        );
        reducer
            .evaluate_batch(shifts, masses, lanes, numerator, geometry, &mut triangles)
            .unwrap();
    }
    let mut bubbles = vec![];
    if n >= 2 && rank + 2 >= n {
        let mut reducer =
            BubbleReducer::new(SymbolicaBubbleKernels::new(n, rank, 0.).unwrap(), c(0.)).unwrap();
        bubbles.resize(
            reducer.cut_count() * lanes,
            CutOutcome::Unstable(CutFailure::NonFiniteResult),
        );
        reducer
            .evaluate_batch(
                shifts,
                masses,
                lanes,
                topology,
                &triangles,
                numerator,
                geometry,
                &mut bubbles,
            )
            .unwrap();
    }
    (triangles, bubbles)
}
fn check(result: &CutOutcome<IntegratedTadpoleResidue<C>>, expected: [C; 3], higher: bool) {
    let CutOutcome::Stable(r) = result else {
        panic!("unexpected tadpole failure: {result:?}")
    };
    assert_eq!(r.coefficients.len(), if higher { 16 } else { 5 });
    for (i, (&value, &computed)) in r.coefficients.iter().zip(&r.computed).enumerate() {
        assert_eq!(computed, i == 0 || (higher && matches!(i, 14 | 15)));
        near(
            value,
            match i {
                0 => expected[0],
                14 => expected[1],
                15 => expected[2],
                _ => c(0.),
            },
        );
    }
}

#[test]
fn single_propagator_projection_and_lane_permutations() {
    let mut geometry = SymbolicaCuts::new(0.).unwrap();
    let routing = [[1, -1, 2, 0], [3, 0, -1, 2], [-2, 2, 1, 1]];
    for (expression, rank, higher) in [(parse!("a+q0"), 1, false), (parse!("a+b*mu2"), 2, true)] {
        let params = if higher {
            vec![parse!("a"), parse!("b")]
        } else {
            vec![parse!("a")]
        };
        let mut numerator = PolynomialNumerator::new(expression, vars(), params, 0.).unwrap();
        let topology = Topology::new(1, rank).unwrap();
        let mut reducer =
            TadpoleReducer::new(SymbolicaTadpoleKernels::new(1, rank, 0.).unwrap(), c(0.)).unwrap();
        for order in [&[2, 0, 1][..], &[1][..], &[][..], &[0, 2][..]] {
            let shifts: Vec<_> = order.iter().map(|&i| shift(routing[i])).collect();
            let masses = vec![c(2.); order.len()];
            let parameters: Vec<_> = order
                .iter()
                .flat_map(|&i| {
                    if higher {
                        vec![c(2. + i as f64), Complex::new(3., i as f64)]
                    } else {
                        vec![c(2. + i as f64)]
                    }
                })
                .collect();
            numerator
                .set_parameter_batch(&parameters, order.len())
                .unwrap();
            let mut output = vec![CutOutcome::Unstable(CutFailure::NonFiniteResult); order.len()];
            reducer
                .evaluate_batch(
                    &shifts,
                    &masses,
                    order.len(),
                    &topology,
                    &[],
                    &topology,
                    &[],
                    &mut numerator,
                    &mut geometry,
                    &mut output,
                )
                .unwrap();
            for (lane, (&i, result)) in order.iter().zip(&output).enumerate() {
                check(
                    result,
                    [
                        c(2. + i as f64 - if higher { 0. } else { routing[i][0] as f64 }),
                        if higher {
                            Complex::new(3., i as f64)
                        } else {
                            c(0.)
                        },
                        c(0.),
                    ],
                    higher,
                );
                let CutOutcome::Stable(r) = result else {
                    unreachable!()
                };
                assert_eq!(r.id.lane, lane);
            }
        }
    }
    // Independent original higher-rank ansatz contains c15*(p.e3)*(p.e4).
    // With no complementary lines geometry selects this first exact Weyl chart.
    let a1 = [Atom::num(1), Atom::num(1) + Atom::i() * 2];
    let s1 = [Atom::num(1), Atom::num(1) - Atom::i() * 2];
    let a2 = [Atom::num(1), Atom::num(-2) + Atom::i()];
    let s2 = [Atom::num(1), Atom::num(-2) - Atom::i()];
    let e3 = FourMomentum::from_weyl_spinors(&a1, &s2);
    let e4 = FourMomentum::from_weyl_spinors(&a2, &s1);
    let p = symbolic_shift(routing[0]);
    let mut numerator =
        PolynomialNumerator::new(p.dot(&e3) * p.dot(&e4), vars(), vec![], 0.).unwrap();
    numerator.set_parameter_batch(&[], 1).unwrap();
    let topology = Topology::new(1, 2).unwrap();
    let mut reducer =
        TadpoleReducer::new(SymbolicaTadpoleKernels::new(1, 2, 0.).unwrap(), c(0.)).unwrap();
    let mut output = vec![CutOutcome::Unstable(CutFailure::NonFiniteResult)];
    reducer
        .evaluate_batch(
            &[shift(routing[0])],
            &[c(2.)],
            1,
            &topology,
            &[],
            &topology,
            &[],
            &mut numerator,
            &mut geometry,
            &mut output,
        )
        .unwrap();
    check(&output[0], [c(0.), c(0.), c(1.)], true);
}

#[test]
fn exact_propagator_cancellation_and_both_nonzero_parent_families() {
    let routing = [[1, -1, 2, 0], [3, 0, 2, 0], [4, -1, 3, 0]];
    let shifts: Vec<_> = routing.into_iter().map(shift).collect();
    let masses = [c(1.), c(2.), c(3.)];
    let mut geometry = SymbolicaCuts::new(0.).unwrap();
    for (n, expression) in [
        (2, denominator(routing[1], 2) * 7),
        (
            3,
            denominator(routing[1], 2) * denominator(routing[2], 3) * 7,
        ),
    ] {
        let mut numerator = PolynomialNumerator::new(expression, vars(), vec![], 0.).unwrap();
        numerator.set_parameter_batch(&[], 1).unwrap();
        let rank = numerator.rank();
        let topology = Topology::new(n, rank).unwrap();
        let (triangles, bubbles) = parents(
            &topology,
            &shifts[..n],
            &masses[..n],
            1,
            &mut numerator,
            &mut geometry,
        );
        let mut reducer =
            TadpoleReducer::new(SymbolicaTadpoleKernels::new(n, rank, 0.).unwrap(), c(0.)).unwrap();
        let mut output = vec![CutOutcome::Unstable(CutFailure::NonFiniteResult); n];
        reducer
            .evaluate_batch(
                &shifts[..n],
                &masses[..n],
                1,
                &topology,
                &triangles,
                &topology,
                &bubbles,
                &mut numerator,
                &mut geometry,
                &mut output,
            )
            .unwrap();
        for (line, result) in output.iter().enumerate() {
            check(
                result,
                [if line == 0 { c(7.) } else { c(0.) }, c(0.), c(0.)],
                rank > n,
            );
        }
    }
    // Construct genuine canonical triangle and bubble residues using the
    // independently obtained shared cut bases. Their source origins differ.
    let mut triangle = [CutOutcome::Unstable(CutFailure::NonFiniteResult)];
    geometry
        .triangle_batch(
            &[TriangleCutRequest {
                id: SampleId { lane: 0, sample: 0 },
                shifts: [&shifts[0], &shifts[1], &shifts[2]],
                masses_squared: [&masses[0], &masses[1], &masses[2]],
            }],
            &mut triangle,
        )
        .unwrap();
    let mut bubble = [CutOutcome::Unstable(CutFailure::NonFiniteResult)];
    geometry
        .bubble_batch(
            &[BubbleCutRequest {
                id: SampleId { lane: 0, sample: 0 },
                shifts: [&shifts[1], &shifts[2]],
                masses_squared: [&masses[1], &masses[2]],
            }],
            &mut bubble,
        )
        .unwrap();
    let CutOutcome::Stable(triangle) = &triangle[0] else {
        panic!("triangle basis")
    };
    let CutOutcome::Stable(bubble) = &bubble[0] else {
        panic!("bubble basis")
    };
    let params: Vec<_> = (0..8)
        .map(|i| Atom::var(symbol!(format!("tadpole_parent_e{i}"))))
        .collect();
    let axis = |offset: usize| {
        FourMomentum::from_args(
            params[offset].clone(),
            params[offset + 1].clone(),
            params[offset + 2].clone(),
            params[offset + 3].clone(),
        )
    };
    let tri = symbolic_shift(routing[0]).dot(&axis(0));
    let bub = symbolic_shift(routing[1]).dot(&axis(4));
    for higher in [false, true] {
        let expression = if higher {
            tri.pow(4)
                + bub.pow(2) * denominator(routing[0], 1)
                + denominator(routing[0], 1) * denominator(routing[2], 3) * 7
        } else {
            tri.pow(3) + &bub * denominator(routing[0], 1)
        };
        let mut numerator =
            PolynomialNumerator::new(expression, vars(), params.clone(), 0.).unwrap();
        let values: Vec<_> = triangle
            .basis
            .e3
            .components()
            .into_iter()
            .chain(bubble.basis.e2.components())
            .copied()
            .collect();
        numerator.set_parameter_batch(&values, 1).unwrap();
        let rank = numerator.rank();
        let topology = Topology::new(3, rank).unwrap();
        let (triangles, bubbles) = parents(
            &topology,
            &shifts,
            &masses,
            1,
            &mut numerator,
            &mut geometry,
        );
        let mut reducer =
            TadpoleReducer::new(SymbolicaTadpoleKernels::new(3, rank, 0.).unwrap(), c(0.)).unwrap();
        let mut output = vec![CutOutcome::Unstable(CutFailure::NonFiniteResult); 3];
        reducer
            .evaluate_batch(
                &shifts,
                &masses,
                1,
                &topology,
                &triangles,
                &topology,
                &bubbles,
                &mut numerator,
                &mut geometry,
                &mut output,
            )
            .unwrap();
        for (line, result) in output.iter().enumerate() {
            check(
                result,
                [
                    if higher && line == 1 { c(7.) } else { c(0.) },
                    c(0.),
                    c(0.),
                ],
                higher,
            );
        }
    }
}

#[test]
fn parent_failures_provenance_and_unchanged_outputs() {
    let mut numerator =
        PolynomialNumerator::new(parse!("q0^3+q1^2+mu2"), vars(), vec![], 0.).unwrap();
    numerator.set_parameter_batch(&[], 2).unwrap();
    let topology = Topology::new(3, 3).unwrap();
    let shifts: Vec<_> = (0..2)
        .flat_map(|_| [[1, -1, 2, 0], [3, 0, 2, 0], [4, -1, 3, 0]].map(shift))
        .collect();
    let masses = vec![c(1.); 6];
    let mut geometry = SymbolicaCuts::new(0.).unwrap();
    let (triangles, mut bubbles) = parents(
        &topology,
        &shifts,
        &masses,
        2,
        &mut numerator,
        &mut geometry,
    );
    let mut reducer =
        TadpoleReducer::new(SymbolicaTadpoleKernels::new(3, 3, 0.).unwrap(), c(0.)).unwrap();
    let mut output = vec![CutOutcome::Unstable(CutFailure::NonFiniteResult); 6];
    // Bubble [0,1], lane1 fails. Only target lines0/1 in that lane fail.
    bubbles[1] = CutOutcome::Unstable(CutFailure::DegenerateGram);
    reducer
        .evaluate_batch(
            &shifts,
            &masses,
            2,
            &topology,
            &triangles,
            &topology,
            &bubbles,
            &mut numerator,
            &mut geometry,
            &mut output,
        )
        .unwrap();
    for (row, value) in output.iter().enumerate() {
        if row == 1 || row == 3 {
            assert!(matches!(
                value,
                CutOutcome::Unstable(CutFailure::DegenerateGram)
            ));
        } else {
            assert!(
                matches!(value, CutOutcome::Stable(_)),
                "row{row}: {value:?}"
            );
        }
    }
    let CutOutcome::Stable(bubble) = &mut bubbles[0] else {
        panic!("stable parent")
    };
    bubble.id.sample += 1;
    let mut sentinel = vec![CutOutcome::Unstable(CutFailure::SingularParameter); 6];
    assert!(
        reducer
            .evaluate_batch(
                &shifts,
                &masses,
                2,
                &topology,
                &triangles,
                &topology,
                &bubbles,
                &mut numerator,
                &mut geometry,
                &mut sentinel
            )
            .is_err()
    );
    assert!(
        sentinel
            .iter()
            .all(|r| matches!(r, CutOutcome::Unstable(CutFailure::SingularParameter)))
    );
    let wrong_rank = Topology::new(3, 2).unwrap();
    assert!(
        reducer
            .evaluate_batch(
                &shifts,
                &masses,
                2,
                &wrong_rank,
                &triangles,
                &topology,
                &bubbles,
                &mut numerator,
                &mut geometry,
                &mut sentinel
            )
            .is_err()
    );
    assert!(
        sentinel
            .iter()
            .all(|r| matches!(r, CutOutcome::Unstable(CutFailure::SingularParameter)))
    );
    let CutOutcome::Stable(bubble) = &mut bubbles[0] else {
        unreachable!()
    };
    bubble.id.sample -= 1;
    numerator.set_parameter_batch(&[], 0).unwrap();
    assert!(
        reducer
            .evaluate_batch(
                &shifts,
                &masses,
                2,
                &topology,
                &triangles,
                &topology,
                &bubbles,
                &mut numerator,
                &mut geometry,
                &mut sentinel
            )
            .is_err()
    );
    assert!(
        sentinel
            .iter()
            .all(|r| matches!(r, CutOutcome::Unstable(CutFailure::SingularParameter)))
    );
}

#[test]
fn higher_mixed_dual_mu_coefficients_and_nonfinite_components() {
    let shape = vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![2, 0], vec![1, 1]];
    let zero = HyperDual::from_values(shape, vec![c(0.); 5]);
    let mut a = zero.clone();
    a.values = vec![c(2.), c(3.), c(5.), c(7.), c(11.)];
    let mut b = zero.clone();
    b.values = vec![c(-3.), c(2.), c(-5.), c(13.), c(17.)];
    let source = PolynomialNumerator::new(
        parse!("a+b*mu2"),
        vars(),
        vec![parse!("a"), parse!("b")],
        0.,
    )
    .unwrap();
    let mut numerator = DualPolynomialNumerator::new(source, zero.clone()).unwrap();
    numerator
        .set_parameter_batch(&[a.clone(), b.clone()], 1)
        .unwrap();
    let mut geometry = SymbolicaDualCuts::new(0., zero.clone()).unwrap();
    let mut reducer = TadpoleReducer::new(
        SymbolicaDualTadpoleKernels::new(1, 2, 0., &zero).unwrap(),
        zero.clone(),
    )
    .unwrap();
    let topology = Topology::new(1, 2).unwrap();
    let mut shift_value = zero.clone();
    shift_value.values = vec![c(1.), c(2.), c(3.), c(5.), c(7.)];
    let mut shifts = vec![FourMomentum::from_args(
        shift_value.clone(),
        zero.clone(),
        zero.clone(),
        zero.clone(),
    )];
    let mut mass = shift_value;
    mass.values[0] = c(2.);
    let mut output = vec![CutOutcome::Unstable(CutFailure::NonFiniteResult)];
    reducer
        .evaluate_batch(
            &shifts,
            &[mass.clone()],
            1,
            &topology,
            &[],
            &topology,
            &[],
            &mut numerator,
            &mut geometry,
            &mut output,
        )
        .unwrap();
    let CutOutcome::Stable(r) = &output[0] else {
        panic!("Dual tadpole failed")
    };
    for (i, value) in r.coefficients.iter().enumerate() {
        for (component, &actual) in value.values.iter().enumerate() {
            near(
                actual,
                match i {
                    0 => a.values[component],
                    14 => b.values[component],
                    _ => c(0.),
                },
            );
        }
    }
    // Primal equality cannot detect this invalid mixed derivative.
    shifts[0].temporal.value.values[4] = c(f64::NAN);
    reducer
        .evaluate_batch(
            &shifts,
            &[mass],
            1,
            &topology,
            &[],
            &topology,
            &[],
            &mut numerator,
            &mut geometry,
            &mut output,
        )
        .unwrap();
    assert!(matches!(
        output[0],
        CutOutcome::Unstable(CutFailure::NonFiniteInput)
    ));
}

#[test]
fn higher_mixed_dual_parent_residue_is_removed_componentwise() {
    let zero = HyperDual::from_values(
        vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![2, 0], vec![1, 1]],
        vec![c(0.); 5],
    );
    let lift = |v: C| {
        let mut d = zero.clone();
        d.values[0] = v;
        d
    };
    let mut shifts: Vec<_> = [[1, -1, 2, 0], [3, 0, 2, 0]]
        .into_iter()
        .map(|p| shift(p).map(lift))
        .collect();
    shifts[0].temporal.value.values[1] = c(0.25);
    shifts[1].spatial.py.values[2] = c(0.125);
    let mut masses = vec![lift(c(1.)), lift(c(2.))];
    masses[1].values[1] = c(0.5);
    let mut geometry = SymbolicaDualCuts::new(0., zero.clone()).unwrap();
    let mut cut = [CutOutcome::Unstable(CutFailure::NonFiniteResult)];
    geometry
        .bubble_batch(
            &[BubbleCutRequest {
                id: SampleId { lane: 0, sample: 0 },
                shifts: [&shifts[0], &shifts[1]],
                masses_squared: [&masses[0], &masses[1]],
            }],
            &mut cut,
        )
        .unwrap();
    let CutOutcome::Stable(cut) = &cut[0] else {
        panic!("Dual bubble basis")
    };
    let p: Vec<_> = (0..8)
        .map(|i| Atom::var(symbol!(format!("tadpole_dual_parent_p{i}"))))
        .collect();
    let q = vars();
    let momentum =
        FourMomentum::from_args(&q[0] + &p[4], &q[1] + &p[5], &q[2] + &p[6], &q[3] + &p[7]);
    let axis = FourMomentum::from_args(p[0].clone(), p[1].clone(), p[2].clone(), p[3].clone());
    let source = PolynomialNumerator::new(momentum.dot(&axis).pow(3), q, p, 0.).unwrap();
    let mut numerator = DualPolynomialNumerator::new(source, zero.clone()).unwrap();
    let parameters: Vec<_> = cut
        .basis
        .e2
        .components()
        .into_iter()
        .chain(shifts[0].components())
        .cloned()
        .collect();
    numerator.set_parameter_batch(&parameters, 1).unwrap();
    let topology = Topology::new(2, 3).unwrap();
    let mut bubbles = vec![CutOutcome::Unstable(CutFailure::NonFiniteResult)];
    let mut bubble_reducer = BubbleReducer::new(
        SymbolicaDualBubbleKernels::new(2, 3, 0., &zero).unwrap(),
        zero.clone(),
    )
    .unwrap();
    bubble_reducer
        .evaluate_batch(
            &shifts,
            &masses,
            1,
            &topology,
            &[],
            &mut numerator,
            &mut geometry,
            &mut bubbles,
        )
        .unwrap();
    let CutOutcome::Stable(parent) = &bubbles[0] else {
        panic!("Dual bubble reduction")
    };
    for (index, value) in parent.coefficients.iter().enumerate() {
        for (component, &actual) in value.values.iter().enumerate() {
            near(
                actual,
                if index == 13 && component == 0 {
                    c(1.)
                } else {
                    c(0.)
                },
            );
        }
    }
    let mut reducer = TadpoleReducer::new(
        SymbolicaDualTadpoleKernels::new(2, 3, 0., &zero).unwrap(),
        zero.clone(),
    )
    .unwrap();
    let mut output = vec![CutOutcome::Unstable(CutFailure::NonFiniteResult); 2];
    reducer
        .evaluate_batch(
            &shifts,
            &masses,
            1,
            &topology,
            &[],
            &topology,
            &bubbles,
            &mut numerator,
            &mut geometry,
            &mut output,
        )
        .unwrap();
    for result in &output {
        let CutOutcome::Stable(r) = result else {
            panic!("Dual tadpole correction failed: {result:?}")
        };
        for value in &r.coefficients {
            for &component in &value.values {
                near(component, c(0.));
            }
        }
    }
    let CutOutcome::Stable(parent) = &mut bubbles[0] else {
        unreachable!()
    };
    parent.coefficients[13].values[4] = c(f64::NAN);
    reducer
        .evaluate_batch(
            &shifts,
            &masses,
            1,
            &topology,
            &[],
            &topology,
            &bubbles,
            &mut numerator,
            &mut geometry,
            &mut output,
        )
        .unwrap();
    assert!(
        output
            .iter()
            .all(|v| matches!(v, CutOutcome::Unstable(CutFailure::NonFiniteInput)))
    );
    let CutOutcome::Stable(parent) = &mut bubbles[0] else {
        unreachable!()
    };
    parent.coefficients[13] = HyperDual::from_values(vec![vec![0], vec![1]], vec![c(1.), c(0.)]);
    reducer
        .evaluate_batch(
            &shifts,
            &masses,
            1,
            &topology,
            &[],
            &topology,
            &bubbles,
            &mut numerator,
            &mut geometry,
            &mut output,
        )
        .unwrap();
    assert!(
        output
            .iter()
            .all(|v| matches!(v, CutOutcome::Unstable(CutFailure::DerivativeShape)))
    );
}

fn precision_control<T: Real + RealLike + SingleFloat>(prototype: T, tolerance: f64)
where
    Complex<T>: Real + EvaluationDomain,
{
    let value = |a: i64, b: i64| {
        Complex::new(
            prototype.from_rational(&Rational::from((a, b))),
            prototype.zero(),
        )
    };
    let mut numerator =
        PolynomialNumerator::new(parse!("q0^2"), vars(), vec![], prototype.clone()).unwrap();
    numerator.set_parameter_batch(&[], 1).unwrap();
    let mut geometry = SymbolicaCuts::new(prototype.clone()).unwrap();
    let mut reducer = TadpoleReducer::new(
        SymbolicaTadpoleKernels::new(1, 2, prototype.clone()).unwrap(),
        value(0, 1),
    )
    .unwrap();
    let topology = Topology::new(1, 2).unwrap();
    let shifts = [FourMomentum::from_args(
        value(1, 7),
        value(0, 1),
        value(0, 1),
        value(0, 1),
    )];
    let masses = [value(2, 7)];
    let mut output = vec![CutOutcome::Unstable(CutFailure::NonFiniteResult)];
    reducer
        .evaluate_batch(
            &shifts,
            &masses,
            1,
            &topology,
            &[],
            &topology,
            &[],
            &mut numerator,
            &mut geometry,
            &mut output,
        )
        .unwrap();
    let CutOutcome::Stable(r) = &output[0] else {
        panic!("precision tadpole")
    };
    // First exact Weyl chart: e1^0=e2^0=3, e1.e2=5 and
    // e3^0*e4^0=13/2. Direct T3 expansion of q0² fixes these
    // three coefficients independently of the implemented projection.
    for (i, (a, b)) in [(0, (131, 245)), (14, (9, 5)), (15, (31, 25))] {
        let error = (r.coefficients[i].clone() - value(a, b)).norm().re.to_f64();
        assert!(error < tolerance, "c{i} error {error} >= {tolerance}");
    }
}

#[test]
fn double_double_and_arbitrary_precision_tadpole_coefficients() {
    precision_control(DoubleFloat::from(0.), 1e-28);
    precision_control(Float::new(256), 1e-70);
}
