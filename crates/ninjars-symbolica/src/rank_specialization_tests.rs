use super::*;
use crate::*;
use ninjars::*;
use oneloop::NativeFloat;
use symbolica::{
    domains::dual::{DualNumberStructure, HyperDual},
    evaluate::EvaluationDomain,
};

struct Fixture<V> {
    shifts: Vec<FourMomentum<V>>,
    masses: Vec<V>,
    triangle: TriangleResidue<V>,
    bubble: BubbleResidue<V>,
    bubble_vectors: [FourMomentum<V>; 4],
    tadpole_vectors: [FourMomentum<V>; 3],
    parameters: [V; 3],
    target: usize,
}

fn fixture<V: FloatLike>(v: impl Fn(i64, i64) -> V) -> Fixture<V> {
    let momentum = |r: [i64; 4]| FourMomentum::from(r.map(|x| v(x, 1)));
    let shifts: Vec<_> = (0..5)
        .map(|i| momentum([3 * i + 1, i, 2 * i, -i]))
        .collect();
    let masses = (0..5).map(|i| v(30 + i, -1 - i)).collect();
    let basis = CutBasis {
        e1: momentum([1, 0, 0, 1]),
        e2: momentum([1, 0, 0, -1]),
        e3: momentum([2, 1, 0, 0]),
        e4: momentum([1, 0, 1, 0]),
        e12: v(2, 0),
    };
    let topology = Topology::new(5, 4).unwrap();
    let index = |p: &[usize]| {
        topology
            .cuts()
            .iter()
            .position(|c| c.partition == p)
            .unwrap()
    };
    let coefficients = |size| {
        let zeros = zero_slots(5, 4, size).unwrap();
        (0..10)
            .map(|i| {
                if zeros.contains(&i) {
                    v(0, 0)
                } else {
                    v(i as i64 + 1, -1)
                }
            })
            .collect()
    };
    let triangle_index = index(&[0, 1, 2]);
    let triangle = TriangleResidue {
        cut_index: triangle_index,
        id: SampleId {
            lane: 0,
            sample: triangle_index,
        },
        geometry: TriangleCut {
            basis: basis.clone(),
            a: momentum([3, 1, 2, 0]),
            e4_effective: basis.e4.clone(),
            beta: v(2, -1),
        },
        coefficients: coefficients(3),
    };
    let bubble_index = index(&[0, 1]);
    let bubble = BubbleResidue {
        cut_index: bubble_index,
        id: SampleId {
            lane: 0,
            sample: bubble_index,
        },
        geometry: BubbleCut {
            basis: basis.clone(),
            a0: momentum([4, 2, 1, 1]),
            a1: momentum([1, 0, 0, 1]),
            e4_effective: basis.e4.clone(),
            parameters: [v(3, 1), v(1, -1), v(2, 0)],
        },
        coefficients: coefficients(2),
    };
    Fixture {
        shifts,
        masses,
        triangle,
        bubble,
        bubble_vectors: [
            momentum([4, 1, 2, 3]),
            momentum([1, 2, 0, 0]),
            basis.e3.clone(),
            basis.e4.clone(),
        ],
        tadpole_vectors: [momentum([3, 2, 1, 0]), basis.e3, basis.e4],
        parameters: [v(3, 1), v(1, -1), v(2, 0)],
        target: index(&[1]),
    }
}

fn counts() -> [usize; 6] {
    ROWS.with(|c| c.replace([0; 6]))
}

/// Compare both correction stages, including their dispatch decisions. Requests
/// have complex masses/origins and target origin 1 differs from source origin 0.
fn compare<V: FloatLike, B: BubbleReductionKernels<V>, T: TadpoleReductionKernels<V>>(
    fixtures: &[Fixture<V>],
    b: &mut B,
    generic_b: &mut B,
    t: &mut T,
    generic_t: &mut T,
    zero: &V,
    check: impl Fn(&V, &V),
) -> [usize; 6] {
    let requests: Vec<_> = fixtures
        .iter()
        .map(|f| BubbleTriangleCorrectionRequest {
            id: SampleId { lane: 0, sample: 0 },
            partition: &[1, 2],
            source_partition: &[0, 1, 2],
            source: &f.triangle,
            shifts: &f.shifts,
            masses_squared: &f.masses,
            vectors: &f.bubble_vectors,
            parameters: &f.parameters,
        })
        .collect();
    let sentinel = zero.from_i64(987);
    let mut actual = vec![sentinel.clone(); requests.len() * b.quotient_columns()];
    let mut expected = actual.clone();
    let mut status = vec![false; requests.len()];
    let mut expected_status = status.clone();
    counts();
    b.triangle_corrections_batch(&requests, &mut actual, &mut status)
        .unwrap();
    let mut dispatched = counts();
    generic_b
        .triangle_corrections_batch(&requests, &mut expected, &mut expected_status)
        .unwrap();
    assert_eq!(status, expected_status);
    if !fixtures.is_empty() {
        assert!(expected_status.iter().any(|&stable| stable));
    }
    for (a, e) in actual.iter().zip(&expected) {
        check(a, e);
    }
    let requests: Vec<_> = fixtures
        .iter()
        .flat_map(|f| {
            [
                TadpoleSource::Triangle(&f.triangle),
                TadpoleSource::Bubble(&f.bubble),
            ]
            .map(|source| TadpoleCorrectionRequest {
                id: SampleId {
                    lane: 0,
                    sample: f.target,
                },
                source,
                shifts: &f.shifts,
                masses_squared: &f.masses,
                vectors: &f.tadpole_vectors,
                beta: &f.parameters[0],
            })
        })
        .collect();
    actual = vec![sentinel; requests.len()];
    expected = actual.clone();
    status = vec![false; requests.len()];
    expected_status = status.clone();
    counts();
    t.correction_batch(&requests, &mut actual, &mut status)
        .unwrap();
    let td = counts();
    for i in 0..6 {
        dispatched[i] += td[i];
    }
    generic_t
        .correction_batch(&requests, &mut expected, &mut expected_status)
        .unwrap();
    assert_eq!(status, expected_status);
    if !fixtures.is_empty() {
        assert!(expected_status.iter().any(|&stable| stable));
    }
    for (a, e) in actual.iter().zip(&expected) {
        check(a, e);
    }
    dispatched
}

fn close<T: NativeFloat>(a: &Complex<T>, b: &Complex<T>, p: &T)
where
    Complex<T>: Real + EvaluationDomain,
{
    let exponent = match T::FIXED_BITS {
        Some(bits) if bits < 60 => 10,
        Some(_) => 25,
        None => 45,
    };
    let mut tolerance = p.one();
    for _ in 0..exponent {
        tolerance /= p.from_i64(10);
    }
    let difference = (a - b).norm().re;
    let scale = a.norm().re + b.norm().re + p.one();
    assert!(
        difference <= tolerance * scale,
        "actual={a:?}, expected={b:?}"
    );
    if T::FIXED_BITS.is_none() {
        // Enabling the alternate derives its prototype from the stage's zero.
        // Track the output precision as well as its numerical agreement.
        for component in [&a.re, &a.im] {
            if !component.is_zero() {
                assert!(component.get_precision() >= p.get_precision().saturating_sub(32));
            }
        }
    }
}

fn ordinary<T: NativeFloat>(p: T, native: bool)
where
    Complex<T>: Real + EvaluationDomain,
{
    let zero = Complex::new(p.zero(), p.zero());
    let make_b = || {
        #[cfg(feature = "native-kernels")]
        if native {
            return SymbolicaBubbleKernels::new_with_native_kernels(5, 4, p.clone()).unwrap();
        }
        assert!(!native);
        SymbolicaBubbleKernels::new(5, 4, p.clone()).unwrap()
    };
    let make_t = || {
        #[cfg(feature = "native-kernels")]
        if native {
            return SymbolicaTadpoleKernels::new_with_native_kernels(5, 4, p.clone()).unwrap();
        }
        assert!(!native);
        SymbolicaTadpoleKernels::new(5, 4, p.clone()).unwrap()
    };
    let mut b = make_b().with_rank_specialized_corrections().unwrap();
    let mut t = make_t().with_rank_specialized_corrections().unwrap();
    let mut generic_b = make_b();
    let mut generic_t = make_t();
    let make = || fixture(|r, i| Complex::new(p.from_i64(r), p.from_i64(i)));
    for rows in [0, 1, 3, 1] {
        let fixtures: Vec<_> = (0..rows).map(|_| make()).collect();
        assert_eq!(
            compare(
                &fixtures,
                &mut b,
                &mut generic_b,
                &mut t,
                &mut generic_t,
                &zero,
                |a, b| close(a, b, &p)
            ),
            [rows, 0, rows, 0, rows, 0]
        );
    }
    // Every masked source slot independently forces the whole source-kind
    // batch back to generic, including the other otherwise eligible rows.
    for size in [3, 2] {
        for slot in zero_slots(5, 4, size).unwrap() {
            let mut fixtures = vec![make(), make(), make()];
            if size == 3 {
                fixtures[0].triangle.coefficients[slot] = zero.one();
            } else {
                fixtures[0].bubble.coefficients[slot] = zero.one();
            }
            fixtures.reverse();
            let expected = if size == 3 {
                [0, 3, 0, 3, 3, 0]
            } else {
                [3, 0, 3, 0, 0, 3]
            };
            assert_eq!(
                compare(
                    &fixtures,
                    &mut b,
                    &mut generic_b,
                    &mut t,
                    &mut generic_t,
                    &zero,
                    |a, b| close(a, b, &p)
                ),
                expected
            );
        }
    }
    let mut fixtures = vec![make(), make(), make()];
    fixtures[1].masses[0].im = p.nan().unwrap();
    assert_eq!(
        compare(
            &fixtures,
            &mut b,
            &mut generic_b,
            &mut t,
            &mut generic_t,
            &zero,
            |a, b| close(a, b, &p)
        ),
        [2, 0, 2, 0, 2, 0]
    );
    assert_eq!(
        compare(
            &[make()],
            &mut b,
            &mut generic_b,
            &mut t,
            &mut generic_t,
            &zero,
            |a, b| close(a, b, &p)
        ),
        [1, 0, 1, 0, 1, 0]
    );
}

#[test]
fn masks_are_exact_projection_zeros() {
    assert_eq!(zero_slots(5, 4, 3).unwrap(), [3, 6, 8, 9]);
    assert_eq!(zero_slots(5, 4, 2).unwrap(), [2, 4, 6, 7, 8, 9]);
    assert!(require_supported(5, 5).is_err());
    assert!(require_supported(4, 3).is_err());
}

#[test]
fn nonempty_failed_batches_keep_sentinels_and_retry_cleanly() {
    let c = |r, i| Complex::new(r as f64, i as f64);
    let mut f = fixture(c);
    let mut bubble = SymbolicaBubbleKernels::new(5, 4, 0.)
        .unwrap()
        .with_rank_specialized_corrections()
        .unwrap();
    let mut tadpole = SymbolicaTadpoleKernels::new(5, 4, 0.)
        .unwrap()
        .with_rank_specialized_corrections()
        .unwrap();
    let sentinel = c(987, 0);
    for bad in [true, false] {
        f.masses[0].im = if bad { f64::NAN } else { -1. };
        let request = BubbleTriangleCorrectionRequest {
            id: SampleId { lane: 0, sample: 0 },
            partition: &[1, 2],
            source_partition: &[0, 1, 2],
            source: &f.triangle,
            shifts: &f.shifts,
            masses_squared: &f.masses,
            vectors: &f.bubble_vectors,
            parameters: &f.parameters,
        };
        let mut output = vec![sentinel; 3];
        let mut stable = [false];
        counts();
        bubble
            .triangle_corrections_batch(&[request], &mut output, &mut stable)
            .unwrap();
        assert_eq!(stable, [!bad]);
        assert_eq!(counts(), if bad { [0; 6] } else { [1, 0, 0, 0, 0, 0] });
        if bad {
            assert_eq!(output, vec![sentinel; 3]);
        }
        let requests = [
            TadpoleSource::Triangle(&f.triangle),
            TadpoleSource::Bubble(&f.bubble),
        ]
        .map(|source| TadpoleCorrectionRequest {
            id: SampleId {
                lane: 0,
                sample: f.target,
            },
            source,
            shifts: &f.shifts,
            masses_squared: &f.masses,
            vectors: &f.tadpole_vectors,
            beta: &f.parameters[0],
        });
        let mut output = [sentinel; 2];
        let mut stable = [false; 2];
        tadpole
            .correction_batch(&requests, &mut output, &mut stable)
            .unwrap();
        assert_eq!(stable, [!bad; 2]);
        assert_eq!(counts(), if bad { [0; 6] } else { [0, 0, 1, 0, 1, 0] });
        if bad {
            assert_eq!(output, [sentinel; 2]);
        }
    }
}

#[test]
fn ordinary_specialized_and_all_masked_slot_fallbacks() {
    ordinary(0., false);
    ordinary(DoubleFloat::default(), false);
    ordinary(Float::new(192), false);
}

#[cfg(feature = "native-kernels")]
#[test]
fn native_generic_fallback_is_preserved() {
    ordinary(0., true);
}

#[test]
fn higher_mixed_derivative_only_coefficients_force_fallback() {
    let zero = HyperDual::from_values(
        vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![2, 0], vec![1, 1]],
        vec![Complex::new(0., 0.); 5],
    );
    let make = || {
        fixture(|r, i| {
            let mut v = zero.clone();
            v.values[0] = Complex::new(r as f64, i as f64);
            v
        })
    };
    let make_b = || SymbolicaDualBubbleKernels::new(5, 4, 0., &zero).unwrap();
    let make_t = || SymbolicaDualTadpoleKernels::new(5, 4, 0., &zero).unwrap();
    let mut b = make_b().with_rank_specialized_corrections().unwrap();
    let mut t = make_t().with_rank_specialized_corrections().unwrap();
    let mut generic_b = make_b();
    let mut generic_t = make_t();
    let check = |a: &HyperDual<Complex<f64>>, b: &HyperDual<Complex<f64>>| {
        assert_eq!(a.get_shape(), b.get_shape());
        assert_eq!(a.values.len(), b.values.len());
        for (a, b) in a.values.iter().zip(&b.values) {
            close(a, b, &0.);
        }
    };
    for rows in [0, 1, 3] {
        let mut fixtures: Vec<_> = (0..rows).map(|_| make()).collect();
        for f in &mut fixtures {
            for component in 1..5 {
                f.triangle.coefficients[2].values[component] = Complex::new(0.25, -0.5);
                f.bubble.coefficients[1].values[component] = Complex::new(-0.25, 0.5);
                f.masses[0].values[component] = Complex::new(0.125, 0.25);
                f.shifts[0].temporal.value.values[component] = Complex::new(0.125, -0.25);
            }
        }
        assert_eq!(
            compare(
                &fixtures,
                &mut b,
                &mut generic_b,
                &mut t,
                &mut generic_t,
                &zero,
                check
            ),
            [rows, 0, rows, 0, rows, 0]
        );
    }
    for component in 1..5 {
        for size in [3, 2] {
            for slot in zero_slots(5, 4, size).unwrap() {
                let mut fixtures = vec![make(), make(), make()];
                let coefficients = if size == 3 {
                    &mut fixtures[1].triangle.coefficients
                } else {
                    &mut fixtures[1].bubble.coefficients
                };
                coefficients[slot].values[component] = Complex::new(1., -1.);
                let expected = if size == 3 {
                    [0, 3, 0, 3, 3, 0]
                } else {
                    [3, 0, 3, 0, 0, 3]
                };
                assert_eq!(
                    compare(
                        &fixtures,
                        &mut b,
                        &mut generic_b,
                        &mut t,
                        &mut generic_t,
                        &zero,
                        check
                    ),
                    expected
                );
            }
        }
    }
    let mut fixtures = vec![make(), make(), make()];
    fixtures[1].triangle.coefficients[3].values[4].im = f64::NAN;
    assert_eq!(
        compare(
            &fixtures,
            &mut b,
            &mut generic_b,
            &mut t,
            &mut generic_t,
            &zero,
            check
        ),
        [2, 0, 2, 0, 3, 0]
    );
    assert_eq!(
        compare(
            &[make()],
            &mut b,
            &mut generic_b,
            &mut t,
            &mut generic_t,
            &zero,
            check
        ),
        [1, 0, 1, 0, 1, 0]
    );
}

#[test]
fn exceptional_overflow_is_classified_by_independent_float_control() {
    let plan =
        TriangleBubbleSubtraction::for_quotient(&LaurentDivision::new(5, 4, 2).unwrap()).unwrap();
    let zeros = zero_slots(5, 4, 3).unwrap();
    let exact = specialize(
        &plan.exact,
        &[0, 1, 2],
        plan.input_layout().residue.clone(),
        &zeros,
    );
    let mut ordinary = plan.exact.compile(&0.).unwrap();
    let mut specialized = exact.compile(&0.).unwrap();
    let mut oracle = plan.exact.compile(&Float::new(256)).unwrap();
    let layout = plan.input_layout();
    let mut row = vec![Complex::new(0., 0.); plan.input_columns()];
    // c2*t²/(s*t+s) has coefficients [1,-1,0] when c2=s.
    // The generic graph retains cubic inverse terms multiplied by zero c3.
    row[layout.residue.start + 2].re = 1e110;
    row[layout.projection_e3.start + 2].re = 1.;
    row[layout.denominator.start].re = 1e110;
    row[layout.denominator.start + 1].re = 1e110;
    let high: Vec<_> = row
        .iter()
        .map(|v| Complex::new(Float::with_val(256, v.re), Float::with_val(256, v.im)))
        .collect();
    let mut generic = vec![Complex::new(0., 0.); 3];
    let mut actual = generic.clone();
    let mut expected = vec![Complex::new(Float::new(256), Float::new(256)); 3];
    ordinary.evaluate_batch(&row, &mut generic, 1).unwrap();
    specialized.evaluate_batch(&row, &mut actual, 1).unwrap();
    oracle.evaluate_batch(&high, &mut expected, 1).unwrap();
    assert!(generic.iter().any(|v| !v.is_finite()));
    for (i, value) in [1., -1., 0.].into_iter().enumerate() {
        close(&actual[i], &Complex::new(value, 0.), &0.);
        close(
            &expected[i],
            &Complex::new(Float::with_val(256, value), Float::new(256)),
            &Float::new(256),
        );
    }
}

#[cfg(feature = "native-kernels")]
#[test]
fn amplitude_builder_orders_report_both_backends() {
    for first in [false, true] {
        let kernels = SymbolicaAmplitudeKernels::new(5, 4, 0.).unwrap();
        let kernels = if first {
            kernels
                .with_rank_specialized_corrections()
                .unwrap()
                .with_native_kernels()
        } else {
            kernels
                .with_native_kernels()
                .with_rank_specialized_corrections()
                .unwrap()
        };
        assert!(kernels.rank_specialized_corrections_enabled());
        assert_eq!(
            kernels.rank_specialized_correction_backend(),
            Some(KernelBackend::SymbolicaInterpreter)
        );
        assert_eq!(
            kernels.kernel_backends()[3].backend,
            KernelBackend::NativeRust
        );
        assert!(
            kernels
                .bubble_kernels()
                .unwrap()
                .rank_specialized_corrections_enabled()
        );
        assert!(
            kernels
                .tadpole_kernels()
                .unwrap()
                .rank_specialized_corrections_enabled()
        );
        let dual = HyperDual::from_values(
            vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![2, 0], vec![1, 1]],
            vec![Complex::new(0., 0.); 5],
        );
        let kernels = SymbolicaDualAmplitudeKernels::new(5, 4, 0., &dual).unwrap();
        let kernels = if first {
            kernels
                .with_rank_specialized_corrections()
                .unwrap()
                .with_native_kernels()
        } else {
            kernels
                .with_native_kernels()
                .with_rank_specialized_corrections()
                .unwrap()
        };
        assert_eq!(
            kernels.rank_specialized_correction_backend(),
            Some(KernelBackend::SymbolicaInterpreter)
        );
        assert_eq!(
            kernels.kernel_backends()[3].backend,
            KernelBackend::SymbolicaInterpreter
        );
        assert_eq!(
            kernels.kernel_backends()[4].backend,
            KernelBackend::NativeRust
        );
        assert!(
            kernels
                .bubble_kernels()
                .unwrap()
                .rank_specialized_corrections_enabled()
        );
        assert!(
            kernels
                .tadpole_kernels()
                .unwrap()
                .rank_specialized_corrections_enabled()
        );
    }
}
