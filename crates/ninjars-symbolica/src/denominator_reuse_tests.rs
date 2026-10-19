//! Compare the combined path with unchanged standalone kernel calls. No master
//! provider is involved: every lower-cut coefficient is compared directly.
use crate::*;
use ninjars::*;
use oneloop::NativeFloat;
use symbolica::{atom::Atom, domains::dual::HyperDual, evaluate::EvaluationDomain, parse};

struct Separate<K>(K);
impl<V, K: BubbleReductionKernels<V>> BubbleReductionKernels<V> for Separate<K> {
    fn propagators(&self) -> usize {
        self.0.propagators()
    }
    fn rank(&self) -> usize {
        self.0.rank()
    }
    fn numerator_expansion(&self) -> &ExpansionKind {
        self.0.numerator_expansion()
    }
    fn quotient_columns(&self) -> usize {
        self.0.quotient_columns()
    }
    fn coefficient_count(&self) -> usize {
        self.0.coefficient_count()
    }
    fn swapped_effective_batch(
        &mut self,
        cuts: &[&BubbleCut<V>],
        out: &mut [FourMomentum<V>],
    ) -> Result<()> {
        self.0.swapped_effective_batch(cuts, out)
    }
    fn divide_batch(
        &mut self,
        r: &[BubbleDivisionRequest<'_, V>],
        n: &[V],
        o: &mut [V],
        s: &mut [bool],
    ) -> Result<()> {
        self.0.divide_batch(r, n, o, s)
    }
    fn triangle_corrections_batch(
        &mut self,
        r: &[BubbleTriangleCorrectionRequest<'_, V>],
        o: &mut [V],
        s: &mut [bool],
    ) -> Result<()> {
        self.0.triangle_corrections_batch(r, o, s)
    }
    fn subtract_batch(&mut self, q: &[V], c: &[V], o: &mut [V], r: usize) -> Result<()> {
        self.0.subtract_batch(q, c, o, r)
    }
    fn project_batch(&mut self, q: &[V], e: &[V], o: &mut [V]) -> Result<()> {
        self.0.project_batch(q, e, o)
    }
    fn values_are_finite(&self, v: &[V]) -> bool {
        self.0.values_are_finite(v)
    }
}
impl<V, K: TadpoleReductionKernels<V>> TadpoleReductionKernels<V> for Separate<K> {
    fn propagators(&self) -> usize {
        self.0.propagators()
    }
    fn rank(&self) -> usize {
        self.0.rank()
    }
    fn numerator_expansion(&self) -> &ExpansionKind {
        self.0.numerator_expansion()
    }
    fn quotient_columns(&self) -> usize {
        self.0.quotient_columns()
    }
    fn coefficient_count(&self) -> usize {
        self.0.coefficient_count()
    }
    fn computed_coefficients(&self) -> &[bool] {
        self.0.computed_coefficients()
    }
    fn value_status(&self, v: &[V]) -> Option<CutFailure> {
        self.0.value_status(v)
    }
    fn divide_batch(
        &mut self,
        r: &[TadpoleDivisionRequest<'_, V>],
        n: &[V],
        o: &mut [V],
        s: &mut [bool],
    ) -> Result<()> {
        self.0.divide_batch(r, n, o, s)
    }
    fn correction_batch(
        &mut self,
        r: &[TadpoleCorrectionRequest<'_, V>],
        o: &mut [V],
        s: &mut [bool],
    ) -> Result<()> {
        self.0.correction_batch(r, o, s)
    }
    fn combine_batch(&mut self, r: &[V], c: &[V], n: usize, o: &mut [V]) -> Result<()> {
        self.0.combine_batch(r, c, n, o)
    }
    fn project_batch(&mut self, r: &[V], e: &[V], o: &mut [V]) -> Result<()> {
        self.0.project_batch(r, e, o)
    }
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

#[test]
fn rank_specialization_full_lower_stage_coefficient_parity() {
    rank_specialization_lower_stages(false);
}

#[cfg(feature = "native-kernels")]
#[test]
fn native_rank_specialization_full_lower_stage_coefficient_parity() {
    rank_specialization_lower_stages(true);
}

fn rank_specialization_lower_stages(native_rank: bool) {
    let zero = Complex::new(0., 0.);
    let (n, rank) = (5, 4);
    let mut numerator = PolynomialNumerator::new(expression(rank), vars(), vec![], 0.).unwrap();
    let mut cuts = SymbolicaCuts::new(0.).unwrap();
    let mut triangle =
        TriangleReducer::new(SymbolicaTriangleKernels::new(n, rank, 0.).unwrap(), zero).unwrap();
    let make_b = || SymbolicaBubbleKernels::new(n, rank, 0.).unwrap();
    let make_t = || SymbolicaTadpoleKernels::new(n, rank, 0.).unwrap();
    let b = make_b().with_rank_specialized_corrections().unwrap();
    let t = make_t().with_rank_specialized_corrections().unwrap();
    #[cfg(feature = "native-kernels")]
    let (b, t) = if native_rank {
        (
            b.with_native_rank_specialized_corrections().unwrap(),
            t.with_native_rank_specialized_corrections().unwrap(),
        )
    } else {
        (b, t)
    };
    #[cfg(not(feature = "native-kernels"))]
    assert!(!native_rank);
    let mut b = BubbleReducer::new(b, zero).unwrap();
    let mut old_b = BubbleReducer::new(Separate(make_b()), zero).unwrap();
    let mut t = TadpoleReducer::new(t, zero).unwrap();
    let mut old_t = TadpoleReducer::new(Separate(make_t()), zero).unwrap();
    let topology = Topology::new(n, rank).unwrap();
    for (lanes, bad, reverse) in [
        (1, false, false),
        (3, false, true),
        (3, true, false),
        (0, false, false),
        (1, false, false),
    ] {
        numerator.set_parameter_batch(&[], lanes).unwrap();
        let (mut shifts, mut masses) = kinematics(&zero, n, lanes);
        if reverse {
            shifts = shifts.chunks(n).rev().flatten().cloned().collect();
            masses = masses.chunks(n).rev().flatten().cloned().collect();
        }
        if bad {
            masses[n].re = f64::NAN;
        }
        let mut triangles =
            vec![CutOutcome::Unstable(CutFailure::NonFiniteResult); triangle.cut_count() * lanes];
        triangle
            .evaluate_batch(
                &shifts,
                &masses,
                lanes,
                &mut numerator,
                &mut cuts,
                &mut triangles,
            )
            .unwrap();
        crate::rank_specialization::ROWS.with(|c| c.set([0; 6]));
        let (_, _, stable) = compare(
            &mut b,
            &mut old_b,
            &mut t,
            &mut old_t,
            &topology,
            &triangles,
            &shifts,
            &masses,
            lanes,
            &mut numerator,
            &mut cuts,
            |a, b| {
                assert!(
                    (a - b).norm().re <= 1e-9 * (1. + a.norm().re + b.norm().re),
                    "{a} != {b}"
                );
            },
        );
        let good = lanes - usize::from(bad);
        assert_eq!(stable, 15 * good);
        let dispatched = crate::rank_specialization::ROWS.with(|c| c.get());
        // Both bubble orientations and all parent source kinds remain present.
        assert_eq!(
            dispatched,
            [
                60 * good,
                60 * good,
                30 * good,
                30 * good,
                20 * good,
                20 * good
            ]
        );
    }
}
fn expression(rank: usize) -> Atom {
    match rank {
        2 => parse!("q0^2+q1+3*mu2+2"),
        3 => parse!("q0^3+q1*mu2+q2*q3+2"),
        4 => parse!("q0^4+q1*mu2+3*mu2^2+q2*q3+2"),
        _ => unreachable!(),
    }
}
fn counts() -> (usize, usize) {
    (
        super::DENOMINATOR_ROWS.with(|v| v.replace(0)),
        crate::tadpole_kernels::DENOMINATOR_ROWS.with(|v| v.replace(0)),
    )
}
fn kinematics<V: FloatLike>(p: &V, n: usize, lanes: usize) -> (Vec<FourMomentum<V>>, Vec<V>) {
    let routing = [
        [0, 0, 0, 0],
        [2, 1, 0, 0],
        [3, 0, 1, 0],
        [4, 0, 0, 1],
        [5, 2, 1, -1],
    ];
    let shifts = (0..lanes)
        .flat_map(|lane| {
            routing[..n].iter().map(move |r| {
                let scale = p.from_i64(17 + lane as i64) / p.from_i64(17);
                FourMomentum::from(std::array::from_fn(|i| {
                    p.from_i64(r[i]) * &scale + p.from_i64(i as i64 + 1) / p.from_i64(7)
                }))
            })
        })
        .collect();
    let masses = (0..lanes)
        .flat_map(|lane| (0..n).map(move |line| p.from_i64(100 + lane as i64 + line as i64)))
        .collect();
    (shifts, masses)
}
#[allow(clippy::too_many_arguments)]
fn compare<
    V: FloatLike,
    N: Numerator<V>,
    C: CutProvider<V>,
    B: BubbleReductionKernels<V>,
    T: TadpoleReductionKernels<V>,
>(
    bubble: &mut BubbleReducer<V, B>,
    old_bubble: &mut BubbleReducer<V, Separate<B>>,
    tadpole: &mut TadpoleReducer<V, T>,
    old_tadpole: &mut TadpoleReducer<V, Separate<T>>,
    topology: &Topology,
    triangles: &[CutOutcome<TriangleResidue<V>>],
    shifts: &[FourMomentum<V>],
    masses: &[V],
    lanes: usize,
    numerator: &mut N,
    cuts: &mut C,
    check: impl Fn(&V, &V),
) -> ((usize, usize), (usize, usize), usize) {
    let mut a = vec![CutOutcome::Unstable(CutFailure::NonFiniteResult); bubble.cut_count() * lanes];
    let mut b = a.clone();
    counts();
    bubble
        .evaluate_batch(
            shifts, masses, lanes, topology, triangles, numerator, cuts, &mut a,
        )
        .unwrap();
    let mut ta =
        vec![CutOutcome::Unstable(CutFailure::NonFiniteResult); tadpole.cut_count() * lanes];
    tadpole
        .evaluate_batch(
            shifts,
            masses,
            lanes,
            topology,
            triangles,
            bubble.topology(),
            &a,
            numerator,
            cuts,
            &mut ta,
        )
        .unwrap();
    let actual = counts();
    old_bubble
        .evaluate_batch(
            shifts, masses, lanes, topology, triangles, numerator, cuts, &mut b,
        )
        .unwrap();
    let mut tb = ta.clone();
    old_tadpole
        .evaluate_batch(
            shifts,
            masses,
            lanes,
            topology,
            triangles,
            old_bubble.topology(),
            &b,
            numerator,
            cuts,
            &mut tb,
        )
        .unwrap();
    let old = counts();
    let mut stable = 0;
    for (a, b) in a.iter().zip(&b) {
        match (a, b) {
            (CutOutcome::Stable(a), CutOutcome::Stable(b)) => {
                stable += 1;
                assert_eq!(a.id, b.id);
                for (x, y) in a.coefficients.iter().zip(&b.coefficients) {
                    check(x, y)
                }
            }
            (CutOutcome::Unstable(a), CutOutcome::Unstable(b)) => assert_eq!(a, b),
            _ => panic!("bubble status differs"),
        }
    }
    for (a, b) in ta.iter().zip(&tb) {
        match (a, b) {
            (CutOutcome::Stable(a), CutOutcome::Stable(b)) => {
                stable += 1;
                assert_eq!(a.id, b.id);
                assert_eq!(a.computed, b.computed);
                for (x, y) in a.coefficients.iter().zip(&b.coefficients) {
                    check(x, y)
                }
            }
            (CutOutcome::Unstable(a), CutOutcome::Unstable(b)) => assert_eq!(a, b),
            _ => panic!("tadpole status differs"),
        }
    }
    (actual, old, stable)
}
fn ordinary<T: NativeFloat>(p: T, cases: &[(usize, usize)], native: bool)
where
    Complex<T>: Real + EvaluationDomain,
{
    for &(n, rank) in cases {
        let zero = Complex::new(p.zero(), p.zero());
        let mut numerator =
            PolynomialNumerator::new(expression(rank), vars(), vec![], p.clone()).unwrap();
        let mut cuts = SymbolicaCuts::new(p.clone()).unwrap();
        let mut triangle = (n >= 3).then(|| {
            TriangleReducer::new(
                SymbolicaTriangleKernels::new(n, rank, p.clone()).unwrap(),
                zero.clone(),
            )
            .unwrap()
        });
        let make_b = || {
            #[cfg(feature = "native-kernels")]
            if native {
                return SymbolicaBubbleKernels::new_with_native_kernels(n, rank, p.clone())
                    .unwrap();
            }
            assert!(!native);
            SymbolicaBubbleKernels::new(n, rank, p.clone()).unwrap()
        };
        let make_t = || {
            #[cfg(feature = "native-kernels")]
            if native {
                return SymbolicaTadpoleKernels::new_with_native_kernels(n, rank, p.clone())
                    .unwrap();
            }
            assert!(!native);
            SymbolicaTadpoleKernels::new(n, rank, p.clone()).unwrap()
        };
        let mut b = BubbleReducer::new(make_b(), zero.clone()).unwrap();
        let mut old_b = BubbleReducer::new(Separate(make_b()), zero.clone()).unwrap();
        let mut t = TadpoleReducer::new(make_t(), zero.clone()).unwrap();
        let mut old_t = TadpoleReducer::new(Separate(make_t()), zero.clone()).unwrap();
        let topology = Topology::new(n, rank).unwrap();
        for (lanes, bad, reverse) in [
            (1, false, false),
            (3, false, true),
            (3, true, false),
            (0, false, false),
            (1, false, false),
        ] {
            numerator.set_parameter_batch(&[], lanes).unwrap();
            let (mut shifts, mut masses) = kinematics(&zero, n, lanes);
            if reverse {
                shifts = shifts.chunks(n).rev().flatten().cloned().collect();
                masses = masses.chunks(n).rev().flatten().cloned().collect();
            }
            if bad {
                masses[n].re = p.nan().unwrap();
            }
            let mut triangles = Vec::new();
            if let Some(triangle) = &mut triangle {
                triangles.resize(
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
                        &mut triangles,
                    )
                    .unwrap();
            }
            let (actual, old, stable) = compare(
                &mut b,
                &mut old_b,
                &mut t,
                &mut old_t,
                &topology,
                &triangles,
                &shifts,
                &masses,
                lanes,
                &mut numerator,
                &mut cuts,
                |a, b| {
                    assert_eq!(a, b);
                    assert_eq!(a.re.get_precision(), b.re.get_precision());
                    assert_eq!(a.im.get_precision(), b.im.get_precision());
                },
            );
            let good = lanes - usize::from(bad);
            assert_eq!(
                stable,
                (n * (n - 1) / 2 + n) * good,
                "n={n},r={rank},native={native}"
            );
            let unique_b = n * (n - 1) * (n - 2) * good;
            let unique_t = (if rank > n { 2 } else { 1 }) * n * (n - 1) * good;
            assert_eq!(actual, (unique_b, unique_t));
            assert_eq!(old, (unique_b * 2, unique_t * n));
        }
    }
}
#[test]
fn ordinary_full_coefficients_and_exact_executed_denominator_counts() {
    ordinary(0., &[(5, 4), (2, 2), (2, 3)], false);
}
#[cfg(feature = "native-kernels")]
#[test]
fn native_full_coefficients_and_exact_executed_denominator_counts() {
    ordinary(0., &[(5, 4), (2, 2), (2, 3)], true);
}
#[test]
fn precision_and_changed_batch_metadata_match_standalone() {
    ordinary(DoubleFloat::default(), &[(3, 4)], false);
    ordinary(Float::new(192), &[(3, 4)], false);
}
fn mixed_dual(native: bool) {
    let shape = vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![2, 0], vec![1, 1]];
    let zero = HyperDual::from_values(shape, vec![Complex::new(0., 0.); 5]);
    let (n, rank) = (3, 4);
    let topology = Topology::new(n, rank).unwrap();
    let source = PolynomialNumerator::new(
        expression(rank) * parse!("a"),
        vars(),
        vec![parse!("a")],
        0.,
    )
    .unwrap();
    let mut numerator = DualPolynomialNumerator::new(source, zero.clone()).unwrap();
    let mut cuts = SymbolicaDualCuts::new(0., zero.clone()).unwrap();
    let mut triangle = TriangleReducer::new(
        SymbolicaDualTriangleKernels::new(n, rank, 0., &zero).unwrap(),
        zero.clone(),
    )
    .unwrap();
    let make_b = || {
        #[cfg(feature = "native-kernels")]
        if native {
            return SymbolicaDualBubbleKernels::new_with_native_denominators(n, rank, 0., &zero)
                .unwrap();
        }
        assert!(!native);
        SymbolicaDualBubbleKernels::new(n, rank, 0., &zero).unwrap()
    };
    let make_t = || {
        #[cfg(feature = "native-kernels")]
        if native {
            return SymbolicaDualTadpoleKernels::new_with_native_kernels(n, rank, 0., &zero)
                .unwrap();
        }
        assert!(!native);
        SymbolicaDualTadpoleKernels::new(n, rank, 0., &zero).unwrap()
    };
    let mut b = BubbleReducer::new(make_b(), zero.clone()).unwrap();
    let mut old_b = BubbleReducer::new(Separate(make_b()), zero.clone()).unwrap();
    let mut t = TadpoleReducer::new(make_t(), zero.clone()).unwrap();
    let mut old_t = TadpoleReducer::new(Separate(make_t()), zero.clone()).unwrap();
    for (lanes, bad) in [(3, false), (1, false), (3, true), (0, false), (3, false)] {
        let mut a = zero.one();
        a.values[1] = Complex::new(1., 0.);
        a.values[2] = Complex::new(0., 1.);
        numerator
            .set_parameter_batch(&vec![a; lanes], lanes)
            .unwrap();
        let (shifts, mut masses) = kinematics(&zero, n, lanes);
        for mass in &mut masses {
            mass.values[1] = Complex::new(1., 0.);
            mass.values[4] = Complex::new(0.25, 0.);
        }
        if bad {
            masses[n].values[4].im = f64::NAN;
        }
        let mut triangles =
            vec![CutOutcome::Unstable(CutFailure::NonFiniteResult); triangle.cut_count() * lanes];
        triangle
            .evaluate_batch(
                &shifts,
                &masses,
                lanes,
                &mut numerator,
                &mut cuts,
                &mut triangles,
            )
            .unwrap();
        let (_, _, stable) = compare(
            &mut b,
            &mut old_b,
            &mut t,
            &mut old_t,
            &topology,
            &triangles,
            &shifts,
            &masses,
            lanes,
            &mut numerator,
            &mut cuts,
            |a, b| assert_eq!(a.values, b.values),
        );
        assert_eq!(stable, 6 * (lanes - usize::from(bad)));
    }
}

#[test]
fn full_higher_mixed_dual_components_and_failed_retry() {
    mixed_dual(false);
    #[cfg(feature = "native-kernels")]
    mixed_dual(true);
}

#[test]
fn selected_parent_errors_are_skipped_only_for_failed_bubble_targets_and_retry_is_clean() {
    type C = Complex<f64>;
    let c = |v| C::new(v, 0.);
    let zero = c(0.);
    let (n, rank) = (3, 2);
    let topology = Topology::new(n, rank).unwrap();
    let (shifts, masses) = kinematics(&zero, n, 1);
    let mut cuts = SymbolicaCuts::new(0.).unwrap();
    let triangle_index = topology
        .cuts()
        .iter()
        .position(|c| c.partition.len() == 3)
        .unwrap();
    let bubble_index = topology
        .cuts()
        .iter()
        .position(|c| c.partition == [1, 2])
        .unwrap();
    let mut triangle = [CutOutcome::Unstable(CutFailure::NonFiniteResult)];
    cuts.triangle_batch(
        &[TriangleCutRequest {
            id: SampleId {
                lane: 0,
                sample: triangle_index,
            },
            shifts: [&shifts[0], &shifts[1], &shifts[2]],
            masses_squared: [&masses[0], &masses[1], &masses[2]],
        }],
        &mut triangle,
    )
    .unwrap();
    let CutOutcome::Stable(triangle) = triangle[0].clone() else {
        panic!("triangle geometry")
    };
    let source = TriangleResidue {
        cut_index: triangle_index,
        id: SampleId {
            lane: 0,
            sample: triangle_index,
        },
        geometry: triangle,
        coefficients: (1..=10).map(|v| c(v as f64)).collect(),
    };
    let mut bubble = [CutOutcome::Unstable(CutFailure::NonFiniteResult)];
    cuts.bubble_batch(
        &[BubbleCutRequest {
            id: SampleId {
                lane: 0,
                sample: bubble_index,
            },
            shifts: [&shifts[1], &shifts[2]],
            masses_squared: [&masses[1], &masses[2]],
            complementary_shifts: &[],
            rank_class: RankClass::UpToN,
        }],
        &mut bubble,
    )
    .unwrap();
    let CutOutcome::Stable(bubble) = bubble[0].clone() else {
        panic!("bubble geometry")
    };
    let mut kernel = SymbolicaBubbleKernels::new(n, rank, 0.).unwrap();
    let mut swapped = [shifts[0]];
    kernel
        .swapped_effective_batch(&[&bubble], &mut swapped)
        .unwrap();
    let vectors = [
        [bubble.a0, bubble.a1, bubble.basis.e3, bubble.e4_effective],
        [bubble.a0, bubble.a1, bubble.basis.e4, swapped[0]],
    ];
    let divisions: Vec<_> = vectors
        .iter()
        .map(|vectors| BubbleDivisionRequest {
            partition: &[1, 2],
            complement: &[0],
            shifts: &shifts,
            masses_squared: &masses,
            vectors,
            parameters: &bubble.parameters,
        })
        .collect();
    let requests: Vec<_> = vectors
        .iter()
        .enumerate()
        .map(|(orientation, vectors)| BubbleTriangleCorrectionRequest {
            id: SampleId {
                lane: 0,
                sample: 2 * bubble_index + orientation,
            },
            partition: &[1, 2],
            source_partition: &[0, 1, 2],
            source: &source,
            shifts: &shifts,
            masses_squared: &masses,
            vectors,
            parameters: &bubble.parameters,
        })
        .collect();
    let width = kernel.quotient_columns();
    let columns = kernel.numerator_expansion().len();
    let numerator = vec![c(1.); columns * 2];
    let mut q = vec![c(17.); 2 * width];
    let mut qs = [false; 2];
    let mut out = q.clone();
    let mut status = [false; 2];
    counts();
    kernel
        .divide_with_corrections_batch(
            &divisions,
            &numerator,
            &mut q,
            &mut qs,
            &requests,
            &mut out,
            &mut status,
        )
        .unwrap();
    assert_eq!(qs, [true; 2]);
    assert_eq!(status, [true; 2]);
    assert_eq!(counts().0, 2);
    let expected = out.clone();
    let expected_q = q.clone();
    // A standalone correction call needs no preceding division and recomputes.
    counts();
    out.fill(c(17.));
    kernel
        .triangle_corrections_batch(&requests, &mut out, &mut status)
        .unwrap();
    assert_eq!(out, expected);
    assert_eq!(counts().0, 2);
    // Distinct allocations are not assumed equal, even though these inputs are.
    let cloned_shifts = shifts.clone();
    let mut separate = requests.clone();
    for r in &mut separate {
        r.shifts = &cloned_shifts;
    }
    counts();
    kernel
        .divide_with_corrections_batch(
            &divisions,
            &numerator,
            &mut q,
            &mut qs,
            &separate,
            &mut out,
            &mut status,
        )
        .unwrap();
    assert_eq!(out, expected);
    assert_eq!(counts().0, 4);
    // Different values behind distinct live references must likewise recompute.
    let mut changed = vectors;
    changed[0][0].temporal.value += c(1.0 / 17.0);
    let mut distinct = requests.clone();
    distinct[0].vectors = &changed[0];
    kernel
        .triangle_corrections_batch(&distinct, &mut out, &mut status)
        .unwrap();
    let distinct_expected = out.clone();
    counts();
    kernel
        .divide_with_corrections_batch(
            &divisions,
            &numerator,
            &mut q,
            &mut qs,
            &distinct,
            &mut out,
            &mut status,
        )
        .unwrap();
    assert_eq!(out, distinct_expected);
    assert_eq!(counts().0, 4);
    let mut malformed = source.clone();
    malformed.coefficients.pop();
    let mut bad = requests.clone();
    bad[0].source = &malformed;
    // Selected malformed parent errors after successful scratch quotients.
    out.fill(c(17.));
    assert!(
        kernel
            .divide_with_corrections_batch(
                &divisions,
                &numerator,
                &mut q,
                &mut qs,
                &bad,
                &mut out,
                &mut status
            )
            .is_err()
    );
    assert_eq!(q, expected_q);
    assert!(out.iter().all(|v| *v == c(17.)));
    let mut nonfinite = numerator.clone();
    nonfinite[0] = c(f64::NAN);
    kernel
        .divide_with_corrections_batch(
            &divisions,
            &nonfinite,
            &mut q,
            &mut qs,
            &bad,
            &mut out,
            &mut status,
        )
        .unwrap();
    assert_eq!(qs, [false, true]);
    assert_eq!(status, [false, false]);
    assert!(out.iter().all(|v| *v == c(17.)));
    counts();
    kernel
        .divide_with_corrections_batch(
            &divisions,
            &numerator,
            &mut q,
            &mut qs,
            &requests,
            &mut out,
            &mut status,
        )
        .unwrap();
    assert_eq!(out, expected);
    assert_eq!(counts().0, 2);
    q.fill(c(19.));
    qs.fill(false);
    assert!(
        kernel
            .divide_with_corrections_batch(
                &divisions,
                &numerator,
                &mut q,
                &mut qs,
                &requests,
                &mut out[..width],
                &mut status
            )
            .is_err()
    );
    assert!(q.iter().all(|v| *v == c(19.)));
    assert_eq!(qs, [false; 2]);
}

#[test]
fn tadpole_parent_failures_standalone_and_distinct_borrows_remain_correct() {
    type C = Complex<f64>;
    let c = |v| C::new(v, 0.);
    let zero = c(0.);
    let (n, rank) = (2, 2);
    let topology = Topology::new(n, rank).unwrap();
    let (shifts, masses) = kinematics(&zero, n, 1);
    let mut cuts = SymbolicaCuts::new(0.).unwrap();
    let bubble_index = topology
        .cuts()
        .iter()
        .position(|c| c.partition.len() == 2)
        .unwrap();
    let mut bubble = [CutOutcome::Unstable(CutFailure::NonFiniteResult)];
    cuts.bubble_batch(
        &[BubbleCutRequest {
            id: SampleId {
                lane: 0,
                sample: bubble_index,
            },
            shifts: [&shifts[0], &shifts[1]],
            masses_squared: [&masses[0], &masses[1]],
            complementary_shifts: &[],
            rank_class: RankClass::UpToN,
        }],
        &mut bubble,
    )
    .unwrap();
    let CutOutcome::Stable(bubble) = bubble[0].clone() else {
        panic!("bubble geometry")
    };
    let source = BubbleResidue {
        cut_index: bubble_index,
        id: SampleId {
            lane: 0,
            sample: bubble_index,
        },
        geometry: bubble,
        coefficients: (1..=10).map(|v| c(v as f64)).collect(),
    };
    let targets: Vec<_> = topology
        .cuts()
        .iter()
        .enumerate()
        .filter_map(|(i, c)| (c.partition.len() == 1).then_some(i))
        .collect();
    let complementary = [vec![shifts[1]], vec![shifts[0]]];
    let requests: Vec<_> = targets
        .iter()
        .enumerate()
        .map(|(line, &sample)| TadpoleCutRequest {
            id: SampleId { lane: 0, sample },
            shift: &shifts[line],
            mass_squared: &masses[line],
            complementary_shifts: &complementary[line],
            rank_class: RankClass::UpToN,
        })
        .collect();
    let mut geometry = vec![CutOutcome::Unstable(CutFailure::NonFiniteResult); 2];
    cuts.tadpole_batch(&requests, &mut geometry).unwrap();
    let geometry: Vec<_> = geometry
        .into_iter()
        .map(|c| match c {
            CutOutcome::Stable(c) => c,
            _ => panic!("tadpole geometry"),
        })
        .collect();
    let vectors: Vec<_> = geometry
        .iter()
        .map(|c| [c.a, c.basis.e3, c.e4_effective])
        .collect();
    let divisions: Vec<_> = targets
        .iter()
        .enumerate()
        .map(|(line, &target)| TadpoleDivisionRequest {
            partition: &topology.cuts()[target].partition,
            complement: &topology.cuts()[target].complement,
            shifts: &shifts,
            masses_squared: &masses,
            vectors: &vectors[line],
            beta: &geometry[line].beta,
        })
        .collect();
    let requests: Vec<_> = targets
        .iter()
        .enumerate()
        .map(|(line, &sample)| TadpoleCorrectionRequest {
            id: SampleId { lane: 0, sample },
            source: TadpoleSource::Bubble(&source),
            shifts: &shifts,
            masses_squared: &masses,
            vectors: &vectors[line],
            beta: &geometry[line].beta,
        })
        .collect();
    let mut kernel = SymbolicaTadpoleKernels::new(n, rank, 0.).unwrap();
    let width = kernel.quotient_columns();
    let columns = kernel.numerator_expansion().len();
    let numerator = vec![c(1.); columns * 2];
    let mut q = vec![c(17.); 2 * width];
    let mut qs = [false; 2];
    let mut out = q.clone();
    let mut status = [false; 2];
    counts();
    kernel
        .divide_with_corrections_batch(
            &divisions,
            &numerator,
            &mut q,
            &mut qs,
            &requests,
            &mut out,
            &mut status,
        )
        .unwrap();
    assert_eq!(qs, [true; 2]);
    assert_eq!(status, [true; 2]);
    assert_eq!(counts().1, 2);
    let expected = out.clone();
    counts();
    kernel
        .correction_batch(&requests, &mut out, &mut status)
        .unwrap();
    assert_eq!(out, expected);
    assert_eq!(counts().1, 2);
    let copy = masses.clone();
    let mut separate = requests.clone();
    for r in &mut separate {
        r.masses_squared = &copy;
    }
    counts();
    kernel
        .divide_with_corrections_batch(
            &divisions,
            &numerator,
            &mut q,
            &mut qs,
            &separate,
            &mut out,
            &mut status,
        )
        .unwrap();
    assert_eq!(out, expected);
    assert_eq!(counts().1, 4);
    let mut malformed = source.clone();
    malformed.coefficients.pop();
    let mut bad = requests.clone();
    bad[0].source = TadpoleSource::Bubble(&malformed);
    out.fill(c(17.));
    assert!(
        kernel
            .divide_with_corrections_batch(
                &divisions,
                &numerator,
                &mut q,
                &mut qs,
                &bad,
                &mut out,
                &mut status
            )
            .is_err()
    );
    assert!(out.iter().all(|v| *v == c(17.)));
    let mut nonfinite = numerator.clone();
    nonfinite[0] = c(f64::NAN);
    kernel
        .divide_with_corrections_batch(
            &divisions,
            &nonfinite,
            &mut q,
            &mut qs,
            &bad,
            &mut out,
            &mut status,
        )
        .unwrap();
    assert_eq!(qs, [false, true]);
    assert_eq!(status, [false, true]);
    assert!(out[..width].iter().all(|v| *v == c(17.)));
    assert_eq!(&out[width..], &expected[width..]);
    counts();
    kernel
        .divide_with_corrections_batch(
            &divisions,
            &numerator,
            &mut q,
            &mut qs,
            &requests,
            &mut out,
            &mut status,
        )
        .unwrap();
    assert_eq!(out, expected);
    assert_eq!(counts().1, 2);
    q.fill(c(19.));
    qs.fill(false);
    assert!(
        kernel
            .divide_with_corrections_batch(
                &divisions,
                &numerator,
                &mut q,
                &mut qs,
                &requests,
                &mut out[..width],
                &mut status
            )
            .is_err()
    );
    assert!(q.iter().all(|v| *v == c(19.)));
    assert_eq!(qs, [false; 2]);
}
#[test]
fn zero_parent_and_empty_combined_calls_use_no_denominator_rows() {
    let zero = Complex::new(0., 0.);
    let one = Complex::new(1., 0.);
    let shifts = [FourMomentum::from([zero; 4])];
    let masses = [one];
    let vectors = [shifts[0]; 3];
    let request = [TadpoleDivisionRequest {
        partition: &[0],
        complement: &[],
        shifts: &shifts,
        masses_squared: &masses,
        vectors: &vectors,
        beta: &one,
    }];
    let mut combined = SymbolicaTadpoleKernels::new(1, 0, 0.).unwrap();
    let mut separate = Separate(SymbolicaTadpoleKernels::new(1, 0, 0.).unwrap());
    for kernel in [
        &mut combined as &mut dyn TadpoleReductionKernels<_>,
        &mut separate,
    ] {
        counts();
        let mut q = [zero];
        let mut stable = [false];
        kernel
            .divide_with_corrections_batch(
                &request,
                &[one],
                &mut q,
                &mut stable,
                &[],
                &mut [],
                &mut [],
            )
            .unwrap();
        assert_eq!(q, [one]);
        assert_eq!(stable, [true]);
        assert_eq!(counts(), (0, 0));
        kernel
            .divide_with_corrections_batch(&[], &[], &mut [], &mut [], &[], &mut [], &mut [])
            .unwrap();
    }
}
