//! Complete residue and callback controls; no master provider is involved.
// Symbolica's generated Dual division multiplies by an inverse.
#![allow(clippy::suspicious_arithmetic_impl, clippy::suspicious_op_assign_impl)]
use ninjars::*;
use ninjars_symbolica::*;
use oneloop::NativeFloat;
use symbolica::{
    atom::Atom,
    create_hyperdual_from_components,
    domains::dual::{DualNumber, DualNumberStructure, HyperDual},
    evaluate::EvaluationDomain,
    parse,
};

create_hyperdual_from_components!(MixedBubbleDual, [[0, 0], [1, 0], [0, 1], [2, 0], [1, 1]]);

type RequestIdentity = (SampleId, Vec<usize>);
struct Trace<N> {
    inner: N,
    expansions: Vec<(ExpansionKind, Vec<RequestIdentity>)>,
}
impl<V, N: Numerator<V>> Numerator<V> for Trace<N> {
    fn rank(&self) -> usize {
        self.inner.rank()
    }
    fn evaluate_batch(
        &mut self,
        requests: &[NumeratorRequest<'_, V>],
        output: &mut [V],
    ) -> Result<()> {
        self.inner.evaluate_batch(requests, output)
    }
    fn expansion_batch(
        &mut self,
        kind: &ExpansionKind,
        requests: &[ExpansionRequest<'_, V>],
        output: &mut [V],
    ) -> Result<()> {
        self.expansions.push((
            kind.clone(),
            requests
                .iter()
                .map(|r| (r.id, r.partition.to_vec()))
                .collect(),
        ));
        self.inner.expansion_batch(kind, requests, output)
    }
}
fn variables() -> [Atom; 5] {
    [
        parse!("q0"),
        parse!("q1"),
        parse!("q2"),
        parse!("q3"),
        parse!("mu2"),
    ]
}
fn polynomial(rank: usize) -> Atom {
    match rank {
        0 => parse!("a"),
        1 => parse!("a*(q0+2*q1+1)"),
        2 => parse!("a*(q0^2+q1+3*mu2+2)"),
        3 => parse!("a*(q0^3+q1*mu2+q2*q3+2)"),
        4 => parse!("a*(q0^4+q1*mu2+3*mu2^2+q2*q3+2)"),
        _ => unreachable!(),
    }
}
fn kinematics<V: FloatLike>(zero: &V, n: usize, lanes: usize) -> (Vec<FourMomentum<V>>, Vec<V>) {
    let routing = [
        [0, 0, 0, 0],
        [2, 1, 0, 0],
        [3, 0, 1, 0],
        [4, 0, 0, 1],
        [5, 2, 1, -1],
    ];
    let shifts = (0..lanes)
        .flat_map(|lane| {
            routing[..n].iter().map(move |q| {
                let scale = zero.from_i64(17 + lane as i64) / zero.from_i64(17);
                FourMomentum::from(std::array::from_fn(|i| {
                    zero.from_i64(q[i]) * &scale + zero.from_i64(i as i64 + 1) / zero.from_i64(7)
                }))
            })
        })
        .collect();
    let masses = (0..lanes)
        .flat_map(|lane| (0..n).map(move |line| zero.from_i64(100 + lane as i64 + line as i64)))
        .collect();
    (shifts, masses)
}

#[allow(clippy::too_many_arguments)]
fn compare<V: FloatLike, K: BubbleReductionKernels<V>, N: Numerator<V>, C: CutProvider<V>>(
    full: &mut BubbleReducer<V, K>,
    pruned: &mut BubbleReducer<V, K>,
    numerator: &mut Trace<N>,
    cuts: &mut C,
    topology: &Topology,
    parents: &[CutOutcome<TriangleResidue<V>>],
    shifts: &[FourMomentum<V>],
    masses: &[V],
    lanes: usize,
    check: impl Fn(&V, &V),
) -> usize {
    let mut expected =
        vec![CutOutcome::Unstable(CutFailure::SingularParameter); full.cut_count() * lanes];
    let mut actual = expected.clone();
    numerator.expansions.clear();
    full.evaluate_batch(
        shifts,
        masses,
        lanes,
        topology,
        parents,
        numerator,
        cuts,
        &mut expected,
    )
    .unwrap();
    let requests = std::mem::take(&mut numerator.expansions);
    pruned
        .evaluate_batch(
            shifts,
            masses,
            lanes,
            topology,
            parents,
            numerator,
            cuts,
            &mut actual,
        )
        .unwrap();
    assert_eq!(
        numerator.expansions, requests,
        "arithmetic-only mode changed callback heads or identity"
    );
    let mut stable = 0;
    for (actual, expected) in actual.iter().zip(&expected) {
        match (actual, expected) {
            (CutOutcome::Stable(a), CutOutcome::Stable(b)) => {
                stable += 1;
                assert_eq!(a.id, b.id);
                assert_eq!(a.cut_index, b.cut_index);
                assert_eq!(a.coefficients.len(), b.coefficients.len());
                for (a, b) in a.coefficients.iter().zip(&b.coefficients) {
                    check(a, b);
                }
            }
            (CutOutcome::Unstable(a), CutOutcome::Unstable(b)) => assert_eq!(a, b),
            _ => panic!("complete bubble status differs"),
        }
    }
    stable
}

fn ordinary<T: NativeFloat>(p: T)
where
    Complex<T>: Real + EvaluationDomain,
{
    let mut tolerance = p.one();
    for _ in 0..match T::FIXED_BITS {
        Some(bits) if bits < 60 => 9,
        Some(_) => 24,
        None => 40,
    } {
        tolerance /= p.from_i64(10);
    }
    for (n, rank) in [(2, 0), (2, 1), (2, 2), (2, 3), (3, 2), (5, 4)] {
        let zero = Complex::new(p.zero(), p.zero());
        let mut numerator = Trace {
            inner: PolynomialNumerator::new(
                polynomial(rank),
                variables(),
                vec![parse!("a")],
                p.clone(),
            )
            .unwrap(),
            expansions: Vec::new(),
        };
        let mut cuts = SymbolicaCuts::new(p.clone()).unwrap();
        let mut triangle = (n >= 3).then(|| {
            TriangleReducer::new(
                SymbolicaTriangleKernels::new(n, rank, p.clone()).unwrap(),
                zero.clone(),
            )
            .unwrap()
        });
        let topology = Topology::new(n, rank).unwrap();
        let mut full = BubbleReducer::new(
            SymbolicaBubbleKernels::new(n, rank, p.clone()).unwrap(),
            zero.clone(),
        )
        .unwrap();
        let selected = SymbolicaBubbleKernels::new(n, rank, p.clone())
            .unwrap()
            .with_pruned_swapped_arithmetic()
            .unwrap();
        let mut pruned = BubbleReducer::new(selected, zero.clone()).unwrap();
        for (lanes, bad, reverse) in [
            (1, false, false),
            (3, false, true),
            (3, true, false),
            (0, false, false),
            (1, false, false),
        ] {
            let (mut shifts, mut masses) = kinematics(&zero, n, lanes);
            let mut parameters: Vec<_> = (0..lanes)
                .map(|lane| {
                    Complex::new(
                        p.from_i64(3 + lane as i64) / p.from_i64(7),
                        -p.one() / p.from_i64(11),
                    )
                })
                .collect();
            if reverse {
                shifts = shifts.chunks(n).rev().flatten().cloned().collect();
                masses = masses.chunks(n).rev().flatten().cloned().collect();
                parameters.reverse();
            }
            if bad {
                masses[n].re = p.nan().unwrap();
            }
            numerator
                .inner
                .set_parameter_batch(&parameters, lanes)
                .unwrap();
            let mut parents = Vec::new();
            if let Some(triangle) = &mut triangle {
                parents.resize(
                    triangle.cut_count() * lanes,
                    CutOutcome::Unstable(CutFailure::NonFiniteResult),
                );
                triangle
                    .evaluate_batch(
                        &shifts,
                        &masses,
                        lanes,
                        &mut numerator,
                        &mut cuts,
                        &mut parents,
                    )
                    .unwrap();
            }
            let stable = compare(
                &mut full,
                &mut pruned,
                &mut numerator,
                &mut cuts,
                &topology,
                &parents,
                &shifts,
                &masses,
                lanes,
                |a, b| {
                    assert!(
                        (a - b).norm().re
                            <= tolerance.clone() * (p.one() + a.norm().re + b.norm().re),
                        "{a:?} != {b:?}"
                    );
                },
            );
            assert_eq!(stable, pruned.cut_count() * (lanes - usize::from(bad)));
        }
    }
}

#[test]
fn full_residues_callbacks_and_failed_lanes_match_in_three_precisions() {
    ordinary(0.0f64);
    ordinary(DoubleFloat::from(0.));
    ordinary(Float::new(192));
}

#[test]
fn full_asymmetric_higher_and_mixed_dual_residues_match() {
    dual_residues(MixedBubbleDual {
        values: [Complex::new(0., 0.); 5],
    });
    dual_residues(HyperDual::from_values(
        vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![2, 0], vec![1, 1]],
        vec![Complex::new(0., 0.); 5],
    ));
}

fn dual_residues<D: Real + DualNumber<Complex<f64>>>(zero: D) {
    let (n, rank) = (3, 4);
    let source =
        PolynomialNumerator::new(polynomial(rank), variables(), vec![parse!("a")], 0.).unwrap();
    let mut numerator = Trace {
        inner: DualPolynomialNumerator::new(source, zero.clone()).unwrap(),
        expansions: Vec::new(),
    };
    let mut cuts = SymbolicaDualCuts::new(0., zero.clone()).unwrap();
    let mut triangle = TriangleReducer::new(
        SymbolicaDualTriangleKernels::new(n, rank, 0., &zero).unwrap(),
        zero.clone(),
    )
    .unwrap();
    let mut full = BubbleReducer::new(
        SymbolicaDualBubbleKernels::new(n, rank, 0., &zero).unwrap(),
        zero.clone(),
    )
    .unwrap();
    let mut pruned = BubbleReducer::new(
        SymbolicaDualBubbleKernels::new(n, rank, 0., &zero)
            .unwrap()
            .with_pruned_swapped_arithmetic()
            .unwrap(),
        zero.clone(),
    )
    .unwrap();
    for lanes in [1, 3, 0, 1] {
        let (mut shifts, mut masses) = kinematics(&zero, n, lanes);
        for lane in 0..lanes {
            shifts[lane * n].temporal.value.values_mut()[1] = Complex::new(0.25, -0.125);
            shifts[lane * n + 1].spatial.px.values_mut()[2] = Complex::new(-0.0625, 0.125);
            masses[lane * n].values_mut()[2] = Complex::new(0.375, -0.25);
        }
        let parameters: Vec<_> = (0..lanes)
            .map(|lane| {
                let mut value = zero.clone();
                value.values_mut().clone_from_slice(&[
                    Complex::new(3. + lane as f64, -0.5),
                    Complex::new(2., 0.75),
                    Complex::new(-1., -0.25),
                    Complex::new(5., 1.25),
                    Complex::new(7., -2.),
                ]);
                value
            })
            .collect();
        numerator
            .inner
            .set_parameter_batch(&parameters, lanes)
            .unwrap();
        let mut parents =
            vec![CutOutcome::Unstable(CutFailure::NonFiniteResult); triangle.cut_count() * lanes];
        triangle
            .evaluate_batch(
                &shifts,
                &masses,
                lanes,
                &mut numerator,
                &mut cuts,
                &mut parents,
            )
            .unwrap();
        let stable = compare(
            &mut full,
            &mut pruned,
            &mut numerator,
            &mut cuts,
            triangle.topology(),
            &parents,
            &shifts,
            &masses,
            lanes,
            |a, b| {
                assert_eq!(a.get_shape(), b.get_shape());
                assert_eq!(a.values().len(), 5);
                for (a, b) in a.values().iter().zip(b.values()) {
                    assert!(
                        (a - b).norm().re <= 1e-9 * (1. + a.norm().re + b.norm().re),
                        "{a} != {b}"
                    );
                }
            },
        );
        assert_eq!(stable, pruned.cut_count() * lanes);
    }
}
