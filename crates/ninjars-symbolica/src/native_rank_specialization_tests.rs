//! Controls for generated alternates, sharing the existing exact guard fixtures.
use super::*;
use crate::native_kernels::rank_specialization::{CONSTANT_PRECISIONS, ROWS as NATIVE_ROWS};

fn native_counts() -> [usize; 6] {
    NATIVE_ROWS.with(|counts| counts.replace([0; 6]))
}

fn shape() -> Vec<Vec<usize>> {
    vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![2, 0], vec![1, 1]]
}

#[test]
fn ordinary_native_alternates_preserve_all_guard_controls() {
    ordinary(0., false, true);
    ordinary(0., true, true);
    ordinary(DoubleFloat::default(), true, true);
    ordinary(Float::new(192), true, true);
}

#[test]
fn native_dual_alternates_preserve_every_derivative_only_guard() {
    dual_guards(true);
}

#[test]
fn native_dispatch_tracks_the_whole_active_source_kind() {
    let c = |r, i| Complex::new(r as f64, i as f64);
    let mut b = SymbolicaBubbleKernels::new_with_native_kernels(5, 4, 0.)
        .unwrap()
        .with_native_rank_specialized_corrections()
        .unwrap();
    let mut t = SymbolicaTadpoleKernels::new_with_native_kernels(5, 4, 0.)
        .unwrap()
        .with_native_rank_specialized_corrections()
        .unwrap();
    let mut reference_b = SymbolicaBubbleKernels::new(5, 4, 0.).unwrap();
    let mut reference_t = SymbolicaTadpoleKernels::new(5, 4, 0.).unwrap();
    for kind in [None, Some(3), Some(2)] {
        let mut fixtures = vec![fixture(c), fixture(c), fixture(c)];
        if let Some(size) = kind {
            let slot = zero_slots(5, 4, size).unwrap()[0];
            if size == 3 {
                fixtures[1].triangle.coefficients[slot] = c(1, -1);
            } else {
                fixtures[1].bubble.coefficients[slot] = c(1, -1);
            }
        }
        native_counts();
        let dispatch = compare(
            &fixtures,
            &mut b,
            &mut reference_b,
            &mut t,
            &mut reference_t,
            &c(0, 0),
            |a, b| close(a, b, &0.),
        );
        assert_eq!(
            native_counts(),
            [dispatch[0], 0, dispatch[2], 0, dispatch[4], 0]
        );
    }
    native_counts();
    assert_eq!(
        compare(
            &[],
            &mut b,
            &mut reference_b,
            &mut t,
            &mut reference_t,
            &c(0, 0),
            |a, b| close(a, b, &0.)
        ),
        [0; 6]
    );
    assert_eq!(native_counts(), [0; 6]);
}

fn dual_precision<T: NativeFloat>(p: T, dual_shape: Vec<Vec<usize>>, native: bool)
where
    Complex<T>: Real + EvaluationDomain,
{
    let zero = HyperDual::from_values(
        dual_shape.clone(),
        vec![Complex::new(p.zero(), p.zero()); dual_shape.len()],
    );
    let make_b = || SymbolicaDualBubbleKernels::new(5, 4, p.clone(), &zero).unwrap();
    let make_t = || SymbolicaDualTadpoleKernels::new(5, 4, p.clone(), &zero).unwrap();
    let mut b = make_b().with_native_rank_specialized_corrections().unwrap();
    let mut t = make_t().with_native_rank_specialized_corrections().unwrap();
    let mut reference_b = make_b().with_rank_specialized_corrections().unwrap();
    let mut reference_t = make_t().with_rank_specialized_corrections().unwrap();
    let expected_backend = if native {
        KernelBackend::NativeRust
    } else {
        KernelBackend::SymbolicaInterpreter
    };
    assert_eq!(
        b.rank_specialized_correction_backend(),
        Some(expected_backend)
    );
    assert_eq!(
        t.rank_specialized_correction_backend(),
        Some(expected_backend)
    );
    for rows in [0, 1, 3, 1] {
        let mut fixtures: Vec<_> = (0..rows)
            .map(|lane| {
                let mut f = fixture(|r, i| {
                    let mut value = zero.clone();
                    value.values[0] = Complex::new(p.from_i64(r), p.from_i64(i));
                    value
                });
                // Break both derivative-component and lane symmetries across
                // independent input axes. Structural-mask coefficients stay zero.
                for component in 1..dual_shape.len() {
                    let c = component as i64;
                    let l = lane as i64 + 1;
                    f.masses[0].values[component] =
                        Complex::new(p.from_i64(2 * c + l), p.from_i64(c * c - 2 * l));
                    f.shifts[0].temporal.value.values[component] =
                        Complex::new(p.from_i64(3 * c - 2 * l), p.from_i64(-c - 3 * l));
                    f.triangle.coefficients[2].values[component] =
                        Complex::new(p.from_i64(c * c + 2 * l), p.from_i64(2 * c - l));
                    f.bubble.coefficients[1].values[component] =
                        Complex::new(p.from_i64(-2 * c - l), p.from_i64(c * c + 3 * l));
                }
                for (size, coefficients) in
                    [(3, &f.triangle.coefficients), (2, &f.bubble.coefficients)]
                {
                    for slot in zero_slots(5, 4, size).unwrap() {
                        assert!(
                            coefficients[slot]
                                .values
                                .iter()
                                .all(|v| v.re.is_zero() && v.im.is_zero())
                        );
                    }
                }
                f
            })
            .collect();
        fixtures.reverse();
        native_counts();
        let first_derivatives = dual_shape
            .iter()
            .position(|d| d.as_slice() == [1, 0])
            .zip(dual_shape.iter().position(|d| d.as_slice() == [0, 1]));
        let distinguishable = std::cell::Cell::new(false);
        assert_eq!(
            compare(
                &fixtures,
                &mut b,
                &mut reference_b,
                &mut t,
                &mut reference_t,
                &zero,
                |a, b| {
                    assert_eq!(a.get_shape(), b.get_shape());
                    for (a, b) in a.values.iter().zip(&b.values) {
                        close(a, b, &p);
                    }
                    if let Some((left, right)) = first_derivatives {
                        let difference = (&b.values[left] - &b.values[right]).norm().re;
                        let scale = b.values[left].norm().re + b.values[right].norm().re + p.one();
                        if difference > scale / p.from_i64(1_000_000) {
                            distinguishable.set(true);
                        }
                    }
                }
            ),
            [rows, 0, rows, 0, rows, 0]
        );
        if rows > 0 && first_derivatives.is_some() {
            assert!(
                distinguishable.get(),
                "fixture must distinguish the two first derivatives in at least one output head"
            );
        }
        assert_eq!(
            native_counts(),
            if native {
                [0, rows, 0, rows, 0, rows]
            } else {
                [0; 6]
            }
        );
    }
}

#[test]
fn native_higher_mixed_dual_matches_the_interpreted_alternate_at_each_precision() {
    dual_precision(0., shape(), true);
    dual_precision(DoubleFloat::default(), shape(), true);
    dual_precision(Float::new(192), shape(), true);
}

#[test]
fn uncatalogued_and_reordered_dual_shapes_keep_interpreted_alternates() {
    for unsupported in [
        vec![vec![0], vec![1], vec![2]],
        vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![2, 0], vec![1, 1]],
    ] {
        dual_precision(0., unsupported.clone(), false);
        let dual = HyperDual::from_values(
            unsupported.clone(),
            vec![Complex::new(0., 0.); unsupported.len()],
        );
        for first in [false, true] {
            let kernels = SymbolicaDualAmplitudeKernels::new(5, 4, 0., &dual).unwrap();
            let kernels = if first {
                kernels
                    .with_native_rank_specialized_corrections()
                    .unwrap()
                    .with_native_kernels()
            } else {
                kernels
                    .with_native_kernels()
                    .with_native_rank_specialized_corrections()
                    .unwrap()
            };
            assert_eq!(
                kernels.rank_specialized_correction_backend(),
                Some(KernelBackend::SymbolicaInterpreter)
            );
            assert_eq!(
                kernels
                    .bubble_kernels()
                    .unwrap()
                    .rank_specialized_correction_backend(),
                Some(KernelBackend::SymbolicaInterpreter)
            );
            assert_eq!(
                kernels
                    .tadpole_kernels()
                    .unwrap()
                    .rank_specialized_correction_backend(),
                Some(KernelBackend::SymbolicaInterpreter)
            );
        }
    }
}

#[test]
fn native_amplitude_builder_orders_and_explicit_interpreted_override() {
    for denominators_only in [false, true] {
        for first in [false, true] {
            let select = |k: SymbolicaAmplitudeKernels<f64>| {
                if denominators_only {
                    k.with_native_denominators()
                } else {
                    k.with_native_kernels()
                }
            };
            let k = SymbolicaAmplitudeKernels::new(5, 4, 0.).unwrap();
            let k = if first {
                select(k.with_native_rank_specialized_corrections().unwrap())
            } else {
                select(k)
                    .with_native_rank_specialized_corrections()
                    .unwrap()
            };
            assert_eq!(
                k.rank_specialized_correction_backend(),
                Some(KernelBackend::NativeRust)
            );
            assert_eq!(k.kernel_backends()[0].backend, KernelBackend::NativeRust);
            assert_eq!(
                k.bubble_kernels()
                    .unwrap()
                    .rank_specialized_correction_backend(),
                Some(KernelBackend::NativeRust)
            );
            assert_eq!(
                k.tadpole_kernels()
                    .unwrap()
                    .rank_specialized_correction_backend(),
                Some(KernelBackend::NativeRust)
            );
            let k = k.with_rank_specialized_corrections().unwrap();
            assert_eq!(
                k.rank_specialized_correction_backend(),
                Some(KernelBackend::SymbolicaInterpreter)
            );
            assert_eq!(
                k.bubble_kernels()
                    .unwrap()
                    .rank_specialized_correction_backend(),
                Some(KernelBackend::SymbolicaInterpreter)
            );
            let k = k.with_native_rank_specialized_corrections().unwrap();
            assert_eq!(
                k.tadpole_kernels()
                    .unwrap()
                    .rank_specialized_correction_backend(),
                Some(KernelBackend::NativeRust)
            );

            let dual = HyperDual::from_values(shape(), vec![Complex::new(0., 0.); 5]);
            let select = |k: SymbolicaDualAmplitudeKernels<f64, HyperDual<Complex<f64>>>| {
                if denominators_only {
                    k.with_native_denominators()
                } else {
                    k.with_native_kernels()
                }
            };
            let k = SymbolicaDualAmplitudeKernels::new(5, 4, 0., &dual).unwrap();
            let k = if first {
                select(k.with_native_rank_specialized_corrections().unwrap())
            } else {
                select(k)
                    .with_native_rank_specialized_corrections()
                    .unwrap()
            };
            assert_eq!(
                k.rank_specialized_correction_backend(),
                Some(KernelBackend::NativeRust)
            );
            assert_eq!(k.kernel_backends()[0].backend, KernelBackend::NativeRust);
            assert_eq!(
                k.bubble_kernels()
                    .unwrap()
                    .rank_specialized_correction_backend(),
                Some(KernelBackend::NativeRust)
            );
            assert_eq!(
                k.tadpole_kernels()
                    .unwrap()
                    .rank_specialized_correction_backend(),
                Some(KernelBackend::NativeRust)
            );
            let k = k.with_rank_specialized_corrections().unwrap();
            assert_eq!(
                k.rank_specialized_correction_backend(),
                Some(KernelBackend::SymbolicaInterpreter)
            );
            assert_eq!(
                k.tadpole_kernels()
                    .unwrap()
                    .rank_specialized_correction_backend(),
                Some(KernelBackend::SymbolicaInterpreter)
            );
            let k = k.with_native_rank_specialized_corrections().unwrap();
            assert_eq!(
                k.bubble_kernels()
                    .unwrap()
                    .rank_specialized_correction_backend(),
                Some(KernelBackend::NativeRust)
            );
        }
    }
}

#[test]
fn native_constants_preserve_float256_scalar_with_float128_dual_carrier() {
    let p = Float::new(256);
    let dual = HyperDual::from_values(
        shape(),
        vec![Complex::new(Float::new(128), Float::new(128)); 5],
    );
    let take = || CONSTANT_PRECISIONS.with(|r| std::mem::take(&mut *r.borrow_mut()));
    let check = |records: Vec<Vec<u32>>, expected: usize| {
        assert_eq!(records.len(), expected);
        for constants in records {
            assert!(!constants.is_empty());
            assert!(
                constants.iter().all(|&bits| bits == 256),
                "actual native constants: {constants:?}"
            );
        }
    };
    take();
    SymbolicaDualBubbleKernels::new(5, 4, p.clone(), &dual)
        .unwrap()
        .with_native_rank_specialized_corrections()
        .unwrap();
    check(take(), 1);
    SymbolicaDualTadpoleKernels::new(5, 4, p.clone(), &dual)
        .unwrap()
        .with_native_rank_specialized_corrections()
        .unwrap();
    check(take(), 2);
    let k = SymbolicaDualAmplitudeKernels::new(5, 4, p, &dual)
        .unwrap()
        .with_native_rank_specialized_corrections()
        .unwrap();
    k.bubble_kernels().unwrap();
    k.tadpole_kernels().unwrap();
    check(take(), 3);
    assert!(
        dual.values
            .iter()
            .all(|v| v.re.get_precision() == 128 && v.im.get_precision() == 128)
    );
}

#[test]
fn unsupported_topologies_return_errors_before_native_installation() {
    let dual = HyperDual::from_values(shape(), vec![Complex::new(0., 0.); 5]);
    for (n, r) in [(1, 0), (4, 3), (5, 5)] {
        assert!(
            SymbolicaBubbleKernels::new(n, r, 0.)
                .and_then(|k| k.with_native_rank_specialized_corrections())
                .is_err()
        );
        assert!(
            SymbolicaTadpoleKernels::new(n, r, 0.)
                .and_then(|k| k.with_native_rank_specialized_corrections())
                .is_err()
        );
        assert!(
            SymbolicaDualBubbleKernels::new(n, r, 0., &dual)
                .and_then(|k| k.with_native_rank_specialized_corrections())
                .is_err()
        );
        assert!(
            SymbolicaDualTadpoleKernels::new(n, r, 0., &dual)
                .and_then(|k| k.with_native_rank_specialized_corrections())
                .is_err()
        );
    }
}

#[test]
fn native_failed_active_batches_keep_sentinels_and_retry() {
    failed_batches(true);
}
