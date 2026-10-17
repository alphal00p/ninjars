//! Exact liveness, raw API compatibility and numerical failure controls.
use super::*;
use ninjars::{CutBasis, SampleId, Topology, TriangleCut, TriangleResidue};
use symbolica::{
    atom::AtomCore,
    domains::dual::{DualNumberStructure, HyperDual},
    id::{Pattern, Replacement},
};

fn c(x: f64) -> Complex<f64> {
    Complex::new(x, 0.)
}
fn close(a: &Complex<f64>, b: &Complex<f64>) {
    assert!(
        (a - b).norm().re <= 1e-9 * (1. + a.norm().re + b.norm().re),
        "{a} != {b}"
    );
}

struct Fixture<V> {
    shifts: Vec<FourMomentum<V>>,
    masses: Vec<V>,
    vectors: [[FourMomentum<V>; 4]; 2],
    parameters: [V; 3],
    complement: Vec<usize>,
    source_partitions: Vec<[usize; 3]>,
    sources: Vec<TriangleResidue<V>>,
}

fn fixture<V: FloatLike>(n: usize, rank: usize, make: impl Fn(i64, i64) -> V) -> Fixture<V> {
    let q = |r: [i64; 4]| FourMomentum::from(r.map(|x| make(x, 0)));
    let topology = Topology::new(n, rank).unwrap();
    let basis = CutBasis {
        e1: q([1, 0, 0, 1]),
        e2: q([1, 0, 0, -1]),
        e3: q([2, 1, 0, 0]),
        e4: q([1, 0, 1, 0]),
        e12: make(2, 0),
    };
    let source_partitions: Vec<_> = (2..n).map(|i| [0, 1, i]).collect();
    let sources = source_partitions
        .iter()
        .map(|partition| {
            let index = topology
                .cuts()
                .iter()
                .position(|cut| cut.partition == *partition)
                .unwrap();
            TriangleResidue {
                cut_index: index,
                id: SampleId {
                    lane: 0,
                    sample: index,
                },
                geometry: TriangleCut {
                    basis: basis.clone(),
                    a: q([1, 2, -1, 0]),
                    e4_effective: basis.e4.clone(),
                    beta: make(2, -1),
                },
                coefficients: (0..if rank > n { 15 } else { 10 })
                    .map(|i| {
                        if (n, rank) == (5, 4) && [3, 6, 8, 9].contains(&i) {
                            make(0, 0)
                        } else {
                            make(i as i64 + 1, -1)
                        }
                    })
                    .collect(),
            }
        })
        .collect();
    Fixture {
        shifts: (0..n).map(|i| q([3 * i as i64, i as i64, 0, 0])).collect(),
        masses: (0..n).map(|i| make(30 + i as i64, -1 - i as i64)).collect(),
        vectors: [
            [
                q([4, 1, 2, 3]),
                q([1, 2, 0, 0]),
                basis.e3.clone(),
                basis.e4.clone(),
            ],
            [
                q([4, 1, 2, 3]),
                q([1, 2, 0, 0]),
                basis.e4.clone(),
                basis.e3.clone(),
            ],
        ],
        parameters: [make(3, 1), make(1, -1), make(2, 0)],
        complement: (2..n).collect(),
        source_partitions,
        sources,
    }
}

fn requests<V>(
    f: &Fixture<V>,
) -> (
    Vec<BubbleDivisionRequest<'_, V>>,
    Vec<BubbleTriangleCorrectionRequest<'_, V>>,
) {
    let divisions = f
        .vectors
        .iter()
        .map(|vectors| BubbleDivisionRequest {
            partition: &[0, 1],
            complement: &f.complement,
            shifts: &f.shifts,
            masses_squared: &f.masses,
            vectors,
            parameters: &f.parameters,
        })
        .collect();
    let corrections = f
        .vectors
        .iter()
        .flat_map(|vectors| {
            f.sources
                .iter()
                .zip(&f.source_partitions)
                .map(move |(source, partition)| BubbleTriangleCorrectionRequest {
                    // Deliberately the same odd sample ID in both orientations. Only the
                    // validated combined row ordering identifies the swapped graph.
                    id: SampleId {
                        lane: 0,
                        sample: 777,
                    },
                    partition: &[0, 1],
                    source_partition: partition,
                    source,
                    shifts: &f.shifts,
                    masses_squared: &f.masses,
                    vectors,
                    parameters: &f.parameters,
                })
        })
        .collect();
    (divisions, corrections)
}

#[test]
fn swapped_head_liveness_preserves_every_exact_residue_component() {
    for (rank, count) in [(1, 0), (2, 1), (3, 3), (4, 7)] {
        let division = LaurentDivision::new(3, rank, 2).unwrap();
        let projection = ResidueProjection::new(&division).unwrap();
        let heads = positive_heads(&division);
        assert_eq!(heads.len(), count);
        let width = division.quotient_expansion().len();
        let replacements: Vec<_> = (0..width)
            .filter(|i| !heads.contains(i))
            .map(|i| {
                Replacement::new(
                    Pattern::Literal(projection.exact.parameters[width + i].clone()),
                    Pattern::Literal(Atom::Zero),
                )
            })
            .collect();
        for expression in &projection.exact.expressions {
            let pruned = expression.replace_multiple(&replacements);
            assert_eq!((expression - pruned).expand(), Atom::Zero);
        }
        // Selecting positive quotient heads must not shift the original
        // numerator powers, original rank, or higher-rank triangle source ABI.
        let ExpansionKind::T2 { powers } = division.numerator_expansion() else {
            unreachable!()
        };
        let removed: Vec<_> = powers
            .iter()
            .enumerate()
            .filter_map(|(i, &(t, _, _))| (t <= 1).then_some(i))
            .collect();
        for &head in &heads {
            for &input in &removed {
                assert!(
                    !division.exact.expressions[head].contains(&division.exact.parameters[input])
                );
            }
        }
    }
}

#[test]
fn projection_pruning_preserves_raw_apis_reuse_and_retry_for_all_degrees() {
    for rank in 1..=4 {
        let f = fixture(3, rank, |re, im| Complex::new(re as f64, im as f64));
        let (divisions, corrections) = requests(&f);
        let mut full = SymbolicaBubbleKernels::new(3, rank, 0.).unwrap();
        let mut pruned = SymbolicaBubbleKernels::new(3, rank, 0.)
            .unwrap()
            .with_pruned_swapped_arithmetic()
            .unwrap();
        if rank == 1 {
            let selected = pruned.engine.pruned_swapped.as_ref().unwrap();
            assert!(selected.heads.is_empty());
            assert!(selected.division.is_none() && selected.correction.is_none());
        }
        let width = full.quotient_columns();
        let numerator: Vec<_> = (0..2 * width)
            .map(|i| Complex::new(i as f64 + 1., -0.25))
            .collect();
        let mut expected_q = vec![c(87.); 2 * width];
        let mut expected_c = expected_q.clone();
        let mut qs = [false; 2];
        let mut cs = [false; 2];
        full.divide_with_corrections_batch(
            &divisions,
            &numerator,
            &mut expected_q,
            &mut qs,
            &corrections,
            &mut expected_c,
            &mut cs,
        )
        .unwrap();
        assert_eq!((qs, cs), ([true; 2], [true; 2]));
        assert!(expected_q[width..].iter().any(|v| !v.is_fully_zero()));
        let mut q = vec![c(f64::NAN); 2 * width];
        let mut corr = q.clone();
        for projection_only in [false, true, false, true] {
            q.fill(c(f64::NAN));
            corr.fill(c(f64::NAN));
            if let Some(plan) = &mut pruned.engine.pruned_swapped {
                plan.input.fill(c(f64::NAN));
                plan.values.fill(c(f64::NAN));
            }
            DENOMINATOR_ROWS.with(|v| v.set(0));
            if projection_only {
                pruned
                    .divide_with_corrections_for_projection_batch(
                        &divisions,
                        &numerator,
                        &mut q,
                        &mut qs,
                        &corrections,
                        &mut corr,
                        &mut cs,
                    )
                    .unwrap();
            } else {
                pruned
                    .divide_with_corrections_batch(
                        &divisions,
                        &numerator,
                        &mut q,
                        &mut qs,
                        &corrections,
                        &mut corr,
                        &mut cs,
                    )
                    .unwrap();
            }
            assert_eq!(DENOMINATOR_ROWS.with(|v| v.get()), 2);
            assert_eq!((qs, cs), ([true; 2], [true; 2]));
            let live = positive_heads(&full.engine.layout);
            for i in 0..2 * width {
                if projection_only && i >= width && !live.contains(&(i % width)) {
                    assert!(q[i].is_fully_zero() && corr[i].is_fully_zero());
                } else {
                    close(&q[i], &expected_q[i]);
                    close(&corr[i], &expected_c[i]);
                }
            }
        }
        pruned
            .divide_batch(&divisions, &numerator, &mut q, &mut qs)
            .unwrap();
        pruned
            .triangle_corrections_batch(&corrections, &mut corr, &mut cs)
            .unwrap();
        for i in 0..2 * width {
            close(&q[i], &expected_q[i]);
            close(&corr[i], &expected_c[i]);
        }
        // Equal values with a distinct borrow must fall back to denominator
        // recomputation, without changing the retained arithmetic heads.
        let independent_vectors = f.vectors;
        let mut independent = corrections.clone();
        for (row, request) in independent.iter_mut().enumerate() {
            request.vectors = &independent_vectors[row];
        }
        DENOMINATOR_ROWS.with(|v| v.set(0));
        pruned
            .divide_with_corrections_for_projection_batch(
                &divisions,
                &numerator,
                &mut q,
                &mut qs,
                &independent,
                &mut corr,
                &mut cs,
            )
            .unwrap();
        assert_eq!(DENOMINATOR_ROWS.with(|v| v.get()), 4);
        assert_eq!((qs, cs), ([true; 2], [true; 2]));
        let live = positive_heads(&full.engine.layout);
        for i in 0..2 * width {
            if i < width || live.contains(&(i % width)) {
                close(&q[i], &expected_q[i]);
                close(&corr[i], &expected_c[i]);
            }
        }
        // Even d0's empty selected graph retains raw source and generated
        // source-projection checks for its swapped parent.
        for projection_overflow in [false, true] {
            let mut bad_source = f.sources[0].clone();
            if projection_overflow {
                bad_source.geometry.basis.e3.temporal.value = c(1e308);
            } else {
                bad_source.coefficients[0] = c(f64::NAN);
            }
            let mut bad_parents = corrections.clone();
            bad_parents[1].source = &bad_source;
            pruned
                .divide_with_corrections_for_projection_batch(
                    &divisions,
                    &numerator,
                    &mut q,
                    &mut qs,
                    &bad_parents,
                    &mut corr,
                    &mut cs,
                )
                .unwrap();
            assert_eq!((qs, cs), ([true; 2], [true, false]));
        }
        // Every raw input is still checked even when its arithmetic head is dead.
        let mut bad = numerator.clone();
        bad[2 * width - 1] = c(f64::NAN);
        pruned
            .divide_with_corrections_for_projection_batch(
                &divisions,
                &bad,
                &mut q,
                &mut qs,
                &corrections,
                &mut corr,
                &mut cs,
            )
            .unwrap();
        assert_eq!(qs, [true, false]);
        assert_eq!(cs, [false; 2]);
        let previous = q.clone();
        assert!(
            pruned
                .divide_with_corrections_for_projection_batch(
                    &divisions,
                    &numerator[..numerator.len() - 1],
                    &mut q,
                    &mut qs,
                    &corrections,
                    &mut corr,
                    &mut cs
                )
                .is_err()
        );
        assert_eq!(q, previous);
        let mut malformed = corrections.clone();
        malformed[1].source_partition = &[0, 1, 9];
        assert!(
            pruned
                .divide_with_corrections_for_projection_batch(
                    &divisions, &numerator, &mut q, &mut qs, &malformed, &mut corr, &mut cs
                )
                .is_err()
        );
        pruned
            .divide_with_corrections_for_projection_batch(
                &divisions,
                &numerator,
                &mut q,
                &mut qs,
                &corrections,
                &mut corr,
                &mut cs,
            )
            .unwrap();
        assert_eq!((qs, cs), ([true; 2], [true; 2]));
    }
}

#[test]
fn discarded_division_overflow_is_distinct_from_bad_inputs_or_live_overflow() {
    // For n3/r2, N=(t^2*a+t*b), D=t+d gives live q1=a and
    // discarded swapped q0=b-a*d. All denominator heads are finite here.
    let mut f = fixture(3, 2, |re, im| Complex::new(re as f64, im as f64));
    f.shifts = vec![
        FourMomentum::from([c(0.); 4]),
        FourMomentum::from([c(0.); 4]),
        FourMomentum::from([c(1.), c(0.), c(0.), c(0.)]),
    ];
    f.masses = vec![c(1e200), c(0.), c(0.)];
    f.parameters = [c(0.); 3];
    f.vectors = std::array::from_fn(|_| {
        [
            FourMomentum::from([c(0.); 4]),
            FourMomentum::from([c(0.); 4]),
            FourMomentum::from([c(0.5), c(0.), c(0.), c(0.)]),
            FourMomentum::from([c(0.); 4]),
        ]
    });
    let mut full = SymbolicaBubbleKernels::new(3, 2, 0.).unwrap();
    let mut pruned = SymbolicaBubbleKernels::new(3, 2, 0.)
        .unwrap()
        .with_pruned_swapped_arithmetic()
        .unwrap();
    let (divisions, _) = requests(&f);
    let numerator = [c(0.), c(1.), c(0.), c(1e200), c(1.), c(0.)];
    let mut a = vec![c(9.); 6];
    let mut b = a.clone();
    let mut sa = [false; 2];
    let mut sb = sa;
    full.engine
        .divide(&divisions, &numerator, &mut a, &mut sa, false)
        .unwrap();
    pruned
        .engine
        .divide(&divisions, &numerator, &mut b, &mut sb, true)
        .unwrap();
    assert_eq!(sa, [true, false]);
    assert_eq!(sb, [true, true]);
    close(&b[3], &c(1e200));
    assert!(b[4].is_fully_zero() && b[5].is_fully_zero());
    // Reduce the finite leading denominator so the retained q1 itself overflows.
    drop(divisions);
    for vectors in &mut f.vectors {
        vectors[2].temporal.value = c(0.25);
    }
    let (divisions, _) = requests(&f);
    let mut numerator = numerator;
    numerator[3] = c(1e308);
    full.engine
        .divide(&divisions, &numerator, &mut a, &mut sa, false)
        .unwrap();
    pruned
        .engine
        .divide(&divisions, &numerator, &mut b, &mut sb, true)
        .unwrap();
    assert_eq!(sa, [true, false]);
    assert_eq!(sb, sa);
}

#[test]
fn rank_guard_covers_all_dual_components_before_orientation_grouping() {
    rank_guard_dual_control(false);
}

#[cfg(feature = "native-kernels")]
#[test]
fn native_rank_guard_and_pruning_orders_keep_all_mixed_dual_components() {
    rank_guard_dual_control(true);
}

fn rank_guard_dual_control(native: bool) {
    let shape = vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![2, 0], vec![1, 1]];
    let prototype = HyperDual::from_values(shape.clone(), vec![c(0.); 5]);
    let make = |re, im| {
        HyperDual::from_values(
            shape.clone(),
            (0..5)
                .map(|component| {
                    if re == 0 && im == 0 {
                        c(0.)
                    } else {
                        Complex::new(
                            (re * (component as i64 + 1)) as f64,
                            (im * (2 * component as i64 + 1)) as f64,
                        )
                    }
                })
                .collect(),
        )
    };
    for prune_first in [false, true] {
        let mut f = fixture(5, 4, make);
        let stage = || {
            #[cfg(feature = "native-kernels")]
            if native {
                return SymbolicaDualBubbleKernels::new_with_native_denominators(
                    5, 4, 0., &prototype,
                )
                .unwrap();
            }
            assert!(!native);
            SymbolicaDualBubbleKernels::new(5, 4, 0., &prototype).unwrap()
        };
        let mut selected = stage();
        if prune_first {
            selected = selected.with_pruned_swapped_arithmetic().unwrap();
        }
        #[cfg(feature = "native-kernels")]
        let mut selected = if native {
            selected.with_native_rank_specialized_corrections().unwrap()
        } else {
            selected.with_rank_specialized_corrections().unwrap()
        };
        #[cfg(not(feature = "native-kernels"))]
        let mut selected = selected.with_rank_specialized_corrections().unwrap();
        if !prune_first {
            selected = selected.with_pruned_swapped_arithmetic().unwrap();
        }
        let mut generic = stage().with_pruned_swapped_arithmetic().unwrap();
        let numerator: Vec<_> = (0..6).map(|i| make(i + 1, -1)).collect();
        for broken in [false, true] {
            let mut bad_source = f.sources[2].clone();
            bad_source.coefficients[3].values[4] = Complex::new(1., 2.);
            let (divisions, mut corrections) = requests(&f);
            if broken {
                // Only one swapped request violates the rank subspace. The
                // first-orientation group must also take the generic path.
                corrections[5].source = &bad_source;
            }
            let mut q = vec![prototype.clone(); 6];
            let mut expected_q = q.clone();
            let mut corr = vec![prototype.clone(); 18];
            let mut expected_c = corr.clone();
            let mut qs = [false; 2];
            let mut eqs = qs;
            let mut cs = [false; 6];
            let mut ecs = cs;
            rank_specialization::ROWS.with(|v| v.set([0; 6]));
            selected
                .divide_with_corrections_for_projection_batch(
                    &divisions,
                    &numerator,
                    &mut q,
                    &mut qs,
                    &corrections,
                    &mut corr,
                    &mut cs,
                )
                .unwrap();
            let dispatched = rank_specialization::ROWS.with(|v| v.get());
            assert_eq!(&dispatched[..2], if broken { &[0, 6] } else { &[6, 0] });
            generic
                .divide_with_corrections_for_projection_batch(
                    &divisions,
                    &numerator,
                    &mut expected_q,
                    &mut eqs,
                    &corrections,
                    &mut expected_c,
                    &mut ecs,
                )
                .unwrap();
            assert_eq!(qs, [true; 2]);
            assert_eq!(cs, [true; 6]);
            assert_eq!((qs, cs), (eqs, ecs));
            for (a, b) in q
                .iter()
                .chain(&corr)
                .zip(expected_q.iter().chain(&expected_c))
            {
                assert_eq!(a.get_shape(), b.get_shape());
                for (x, y) in a.values.iter().zip(&b.values) {
                    close(x, y);
                }
            }
        }
        // A finite but incompatible ordered shape must still invalidate the source
        // in both orientations, before any guarded or restricted arithmetic.
        f.sources[0].coefficients[0] =
            HyperDual::from_values(vec![vec![0], vec![1]], vec![c(1.), c(2.)]);
        let (divisions, corrections) = requests(&f);
        let mut q = vec![prototype.clone(); 6];
        let mut corr = vec![prototype.clone(); 18];
        let mut qs = [false; 2];
        let mut cs = [false; 6];
        selected
            .divide_with_corrections_for_projection_batch(
                &divisions,
                &numerator,
                &mut q,
                &mut qs,
                &corrections,
                &mut corr,
                &mut cs,
            )
            .unwrap();
        assert_eq!(qs, [true; 2]);
        assert_eq!(cs, [false, true, true, false, true, true]);
    }
}

#[test]
fn pruned_dual_graphs_keep_explicit_precision_instead_of_carrier_precision() {
    use ninjars::Float;
    let p = Float::new(256);
    let carrier = Float::new(128);
    let dual = HyperDual::from_values(
        vec![vec![0], vec![1], vec![2]],
        vec![Complex::new(carrier.zero(), carrier.zero()); 3],
    );
    let stage = SymbolicaDualBubbleKernels::new(3, 2, p, &dual).unwrap();
    crate::kernel::DUAL_COMPILATION_PRECISIONS.with(|records| records.borrow_mut().clear());
    stage.with_pruned_swapped_arithmetic().unwrap();
    let records = crate::kernel::DUAL_COMPILATION_PRECISIONS
        .with(|records| std::mem::take(&mut *records.borrow_mut()));
    assert_eq!(records.len(), 2);
    assert!(records.iter().any(|(_, constants)| !constants.is_empty()));
    for (precision, constants) in records {
        assert_eq!(precision, 256);
        assert!(constants.iter().all(|&bits| bits == 256));
    }
}

#[cfg(feature = "native-kernels")]
#[test]
fn native_rank_and_row_policy_do_not_change_selected_interpreter_backend() {
    use crate::{BatchExecutionPolicy, KernelBackend, SymbolicaAmplitudeKernels};
    use ninjars::AmplitudeKernels;
    use std::num::NonZeroUsize;
    let policy = BatchExecutionPolicy::FixedRows(NonZeroUsize::new(3).unwrap());
    for reverse in [false, true] {
        let factory = SymbolicaAmplitudeKernels::new(5, 4, 0.).unwrap();
        let factory = if reverse {
            factory
                .with_pruned_swapped_arithmetic()
                .with_native_batch_execution(policy)
                .with_native_rank_specialized_corrections()
                .unwrap()
                .with_native_kernels()
        } else {
            factory
                .with_native_kernels()
                .with_native_rank_specialized_corrections()
                .unwrap()
                .with_native_batch_execution(policy)
                .with_pruned_swapped_arithmetic()
        };
        let stage = factory.bubble_kernels().unwrap();
        let report = stage.pruned_swapped_arithmetic().unwrap();
        assert_eq!((report.full_columns, report.swapped_columns), (3, 1));
        assert!(report.guarded_rank_correction);
        assert_eq!(report.backend(), "symbolica-interpreter");
        assert_eq!(factory.pruned_swapped_arithmetic(), Some(report));
        assert_eq!(factory.native_batch_execution(), policy);
        assert_eq!(
            stage.rank_specialized_correction_backend(),
            Some(KernelBackend::NativeRust)
        );
        let selected = stage.engine.pruned_swapped.unwrap();
        assert_eq!(
            selected.division.unwrap().backend(),
            KernelBackend::SymbolicaInterpreter
        );
        assert_eq!(
            selected.correction.unwrap().backend(),
            KernelBackend::SymbolicaInterpreter
        );
        assert_eq!(
            selected.specialized_correction.unwrap().backend(),
            KernelBackend::SymbolicaInterpreter
        );
    }
}

#[test]
fn opting_in_does_not_activate_tadpole_only_or_rank_inactive_bubbles() {
    use crate::SymbolicaAmplitudeKernels;
    for (n, rank) in [(1, 0), (1, 2), (5, 0)] {
        let factory = SymbolicaAmplitudeKernels::new(n, rank, 0.)
            .unwrap()
            .with_pruned_swapped_arithmetic();
        let report = factory.pruned_swapped_arithmetic().unwrap();
        assert_eq!((report.full_columns, report.swapped_columns), (0, 0));
        factory.into_amplitude(1).unwrap();
    }
}

#[test]
fn selected_group_error_does_not_publish_partial_residues() {
    use crate::{PolynomialNumerator, SymbolicaCuts};
    use ninjars::{BubbleReducer, CutFailure, CutOutcome};
    use symbolica::parse;
    let mut stage = SymbolicaBubbleKernels::new(2, 2, 0.)
        .unwrap()
        .with_pruned_swapped_arithmetic()
        .unwrap();
    // Deliberately inject an incompatible selected evaluator. Its shape error
    // occurs after the full first-orientation group has already executed.
    stage.engine.pruned_swapped.as_mut().unwrap().division = Some(
        ExactKernel {
            parameters: vec![],
            expressions: vec![Atom::Zero],
        }
        .compile(&0.)
        .unwrap(),
    );
    let mut reducer = BubbleReducer::new(stage, c(0.)).unwrap();
    let variables = [
        parse!("q0"),
        parse!("q1"),
        parse!("q2"),
        parse!("q3"),
        parse!("mu2"),
    ];
    let mut numerator =
        PolynomialNumerator::new(parse!("q0^2+mu2+1"), variables, vec![], 0.).unwrap();
    numerator.set_parameter_batch(&[], 1).unwrap();
    let shifts = [
        FourMomentum::from([c(0.); 4]),
        FourMomentum::from([c(2.), c(1.), c(0.), c(0.)]),
    ];
    let mut output = vec![CutOutcome::Unstable(CutFailure::SingularParameter)];
    let mut cuts = SymbolicaCuts::new(0.).unwrap();
    assert!(
        reducer
            .evaluate_batch(
                &shifts,
                &[c(100.), c(101.)],
                1,
                &Topology::new(2, 2).unwrap(),
                &[],
                &mut numerator,
                &mut cuts,
                &mut output
            )
            .is_err()
    );
    assert!(matches!(
        output[0],
        CutOutcome::Unstable(CutFailure::SingularParameter)
    ));
}

struct LegacyCombined {
    kind: ExpansionKind,
    called: bool,
}
impl BubbleReductionKernels<Complex<f64>> for LegacyCombined {
    fn propagators(&self) -> usize {
        3
    }
    fn rank(&self) -> usize {
        2
    }
    fn numerator_expansion(&self) -> &ExpansionKind {
        &self.kind
    }
    fn quotient_columns(&self) -> usize {
        3
    }
    fn coefficient_count(&self) -> usize {
        10
    }
    fn swapped_effective_batch(
        &mut self,
        _: &[&BubbleCut<Complex<f64>>],
        _: &mut [FourMomentum<Complex<f64>>],
    ) -> Result<()> {
        unreachable!()
    }
    fn divide_batch(
        &mut self,
        _: &[BubbleDivisionRequest<'_, Complex<f64>>],
        _: &[Complex<f64>],
        _: &mut [Complex<f64>],
        _: &mut [bool],
    ) -> Result<()> {
        unreachable!()
    }
    fn triangle_corrections_batch(
        &mut self,
        _: &[BubbleTriangleCorrectionRequest<'_, Complex<f64>>],
        _: &mut [Complex<f64>],
        _: &mut [bool],
    ) -> Result<()> {
        unreachable!()
    }
    fn divide_with_corrections_batch(
        &mut self,
        divisions: &[BubbleDivisionRequest<'_, Complex<f64>>],
        numerator: &[Complex<f64>],
        q: &mut [Complex<f64>],
        qs: &mut [bool],
        parents: &[BubbleTriangleCorrectionRequest<'_, Complex<f64>>],
        corrections: &mut [Complex<f64>],
        cs: &mut [bool],
    ) -> Result<()> {
        assert_eq!((divisions.len(), numerator.len(), parents.len()), (2, 6, 2));
        self.called = true;
        q.fill(c(42.));
        corrections.fill(c(43.));
        qs.fill(true);
        cs.fill(true);
        Ok(())
    }
    fn subtract_batch(
        &mut self,
        _: &[Complex<f64>],
        _: &[Complex<f64>],
        _: &mut [Complex<f64>],
        _: usize,
    ) -> Result<()> {
        unreachable!()
    }
    fn project_batch(
        &mut self,
        _: &[Complex<f64>],
        _: &[Complex<f64>],
        _: &mut [Complex<f64>],
    ) -> Result<()> {
        unreachable!()
    }
    fn values_are_finite(&self, _: &[Complex<f64>]) -> bool {
        unreachable!()
    }
}

#[test]
fn legacy_implementers_keep_the_existing_full_combined_dispatch() {
    let f = fixture(3, 2, |re, im| Complex::new(re as f64, im as f64));
    let (divisions, parents) = requests(&f);
    let mut legacy = LegacyCombined {
        kind: LaurentDivision::new(3, 2, 2)
            .unwrap()
            .numerator_expansion()
            .clone(),
        called: false,
    };
    let mut q = [c(0.); 6];
    let mut corrections = q;
    let mut qs = [false; 2];
    let mut cs = qs;
    legacy
        .divide_with_corrections_for_projection_batch(
            &divisions,
            &[c(1.); 6],
            &mut q,
            &mut qs,
            &parents,
            &mut corrections,
            &mut cs,
        )
        .unwrap();
    assert!(legacy.called);
    assert_eq!(q, [c(42.); 6]);
    assert_eq!(corrections, [c(43.); 6]);
    assert_eq!((qs, cs), ([true; 2], [true; 2]));
}
