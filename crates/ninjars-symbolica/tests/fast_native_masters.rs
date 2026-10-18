//! Accuracy of the binary64 bubble masters of `OneLoopNativeMasters`.
//!
//! B0, B1, B11 and B111 are a port of OneLOop's direct bubble formulas
//! (`src/oneloop_native.rs`). They are checked against oneloopmaster's
//! symbolic masters evaluated in 300-bit `Float` arithmetic (where the
//! Passarino-Veltman relations they encode are harmless) on a grid that
//! includes tiny `|p²| << m²` (where binary64 Passarino-Veltman reductions lose
//! `(m²/p²)^rank` relative accuracy), zero momentum, thresholds, on-shell
//! points, massless and complex masses; and against the OneLOop values
//! original Ninja requested on every reference case.
//!
//! B1111 (rank n+2 bubbles, oneloopmaster's native `TensorIntegral::B1111`)
//! is checked against the exact 300-bit expression on the same grid, through
//! the mass-swap identity on every bubble of the reference cases (OneLOop's own
//! `bub1111` has the wrong swap sign, so it is never a reference), and bitwise
//! between the batched `FastMasters` call, its row-wise fallback, the
//! `MasterProvider` interface and `SymbolicaMasters<f64>`. Run with
//! `--features native-masters --test-threads=1`.

mod common;

use common::ninja_reference::parse_ninja;
use ninjars::fast::{C64, FastMaster, FastMasters};
use ninjars::{
    Complex, Float, MasterFailure, MasterIntegral, MasterProvider, MasterRequest, MasterStatus,
};
use ninjars_symbolica::{OneLoopNativeMasters, SymbolicaMasters};
use std::path::PathBuf;

fn relative(native: &[C64; 3], reference: &[C64; 3]) -> f64 {
    let scale = reference.iter().map(|z| z.abs()).fold(1e-300, f64::max);
    (0..3)
        .map(|j| (native[j] - reference[j]).abs())
        .fold(0., f64::max)
        / scale
}

#[test]
fn bubbles_match_exact_masters_including_tiny_momenta() {
    let bits = 300;
    let mut exact = SymbolicaMasters::new(Float::new(bits));
    let mut native = OneLoopNativeMasters::new().unwrap();
    let cf = |z: C64| Complex::new(Float::with_val(bits, z.re), Float::with_val(bits, z.im));
    let (top, higgs) = (C64::real(29929.), C64::real(15625.));
    let complex_top = C64::new(29929., -259.5);
    let masses = [
        (top, higgs),
        (higgs, top),
        (top, C64::ZERO),
        (C64::ZERO, complex_top),
        (complex_top, complex_top),
        (top, top),
        (C64::ZERO, C64::ZERO),
        (complex_top, higgs),
    ];
    let momenta = [
        0., 1e-6, -1e-4, 1e-2, -1e-2, 1., -1., 100., 15625., 29929., 59858., 119716., -2.5e5,
        2.5e5, 1e7,
    ];
    let mut worst = [0f64; 4];
    for &(m0, m1) in &masses {
        for &p in &momenta {
            // The 300-bit symbolic reference itself fails above the
            // threshold of a massive-massless pair in this ordering.
            if p == 2.5e5 && m1.is_zero() && !m0.is_zero() {
                continue;
            }
            let value = native.bubble(C64::real(p), m0, m1, C64::real(1e4)).unwrap();
            for (k, integral) in [
                MasterIntegral::B0,
                MasterIntegral::B1,
                MasterIntegral::B11,
                MasterIntegral::B111,
            ]
            .into_iter()
            .enumerate()
            {
                let zero = cf(C64::ZERO);
                let mut out = vec![[zero.clone(), zero.clone(), zero]];
                exact
                    .evaluate(
                        MasterRequest {
                            integral,
                            invariants: &[cf(C64::real(p))],
                            masses_squared: &[cf(m0), cf(m1)],
                            scales_squared: &[cf(C64::real(1e4))],
                        },
                        &mut out,
                    )
                    .unwrap();
                let reference = out[0].clone().map(|v| {
                    let f = v.to_f64();
                    C64::new(f.re, f.im)
                });
                let error = relative(&value[k], &reference);
                assert!(
                    error <= 1e-13,
                    "B{k}(p²={p:e}, {m0:?}, {m1:?}): native {:?} exact {:?} ({error:e})",
                    value[k],
                    reference
                );
                worst[k] = worst[k].max(error);
            }
        }
    }
    eprintln!("FAST_NATIVE_BUBBLES max relative error B0/B1/B11/B111 vs 300-bit: {worst:?}");
}

/// OneLOop's B0, B1, B11 (and B111 where original Ninja's own rank-3 recursion,
/// which divides by `p⁶`, is well conditioned) on every bubble original Ninja
/// requested in the reference cases.
#[test]
fn bubbles_match_oneloop_on_reference_calls() {
    let root =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../validation/fast/ninja_reference");
    let mut native = OneLoopNativeMasters::new().unwrap();
    let mut checked = 0;
    let mut worst = 0f64;
    for dir in [root.join("ninja"), root.join("conditioning/ninja")] {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            let lanes = parse_ninja(&std::fs::read_to_string(&path).unwrap(), None).unwrap();
            for call in lanes.iter().flat_map(|l| &l.calls) {
                if !matches!(
                    call.family,
                    FastMaster::BubbleRank2 | FastMaster::BubbleRank3
                ) {
                    continue;
                }
                let mut out = vec![[C64::ZERO; 3]; call.family.outputs()];
                let mut ok = [false];
                FastMasters::evaluate(&mut native, call.family, &call.args, &mut out, &mut ok)
                    .unwrap();
                assert!(ok[0], "{}: {:?}", path.display(), call.args);
                let (p, m0, m1) = (call.args[0].re, call.args[1].abs(), call.args[2].abs());
                for (k, (value, oneloop)) in out.iter().zip(&call.values).enumerate() {
                    if k == 3 && p.abs() > 1e-8 && p.abs() < 0.1 * m0.max(m1) {
                        continue;
                    }
                    let error = relative(value, oneloop);
                    assert!(
                        error <= 1e-12,
                        "{}: B{k}{:?}: native {value:?} OneLOop {oneloop:?} ({error:e})",
                        path.display(),
                        call.args
                    );
                    worst = worst.max(error);
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 200, "only {checked} bubble values checked");
    eprintln!("FAST_NATIVE_BUBBLES {checked} OneLOop values, max relative difference {worst:e}");
}

// ---------------------------------------------------------------------------
// Rank n+2 bubbles: B1111 (`FastMaster::BubbleRank4`, `MasterIntegral::B1111`)
// from oneloopmaster's native `TensorIntegral::B1111`.

/// Binary64 target for B1111 against 300 bits, relative to the largest
/// coefficient of its row (oneloopmaster's `native_tensor_b1111.rs` uses 5e-14
/// of the same scale plus 1e-3): after a mass swap `B1111 = B0' + 4B1' +
/// 6B11' + 4B111' + B1111'` cancels below its terms, so its rounding error is
/// a few ulps of `max_k |B_k|`, not of `|B1111|`.
const B1111_TOLERANCE: f64 = 5e-14;

fn norm3(value: &[C64; 3]) -> f64 {
    value.iter().map(|z| z.abs()).fold(0., f64::max)
}

fn difference(a: &[C64; 3], b: &[C64; 3]) -> f64 {
    (0..3).map(|j| (a[j] - b[j]).abs()).fold(0., f64::max)
}

/// B1111 of the exact oneloopmaster expression in `Float(bits)`.
fn exact_b1111(exact: &mut SymbolicaMasters<Float>, bits: u32, args: [C64; 4]) -> [C64; 3] {
    let cf = |z: C64| Complex::new(Float::with_val(bits, z.re), Float::with_val(bits, z.im));
    let zero = cf(C64::ZERO);
    let mut out = vec![[zero.clone(), zero.clone(), zero]];
    exact
        .evaluate(
            MasterRequest {
                integral: MasterIntegral::B1111,
                invariants: &[cf(args[0])],
                masses_squared: &[cf(args[1]), cf(args[2])],
                scales_squared: &[cf(args[3])],
            },
            &mut out,
        )
        .unwrap();
    out[0].clone().map(|v| {
        let f = v.to_f64();
        C64::new(f.re, f.im)
    })
}

/// One `MasterProvider` row of `provider` for `integral` at `[p², m0², m1², mu²]`.
fn provider_row<M: MasterProvider<Complex<f64>>>(
    provider: &mut M,
    integral: MasterIntegral,
    args: [C64; 4],
) -> [C64; 3] {
    let mut out = vec![[Complex::new(0., 0.); 3]];
    provider
        .evaluate(
            MasterRequest {
                integral,
                invariants: &[args[0].to_symbolica()],
                masses_squared: &[args[1].to_symbolica(), args[2].to_symbolica()],
                scales_squared: &[args[3].to_symbolica()],
            },
            &mut out,
        )
        .unwrap();
    out[0].map(C64::from_symbolica)
}

fn bit_patterns(values: &[[C64; 3]]) -> Vec<[u64; 6]> {
    values
        .iter()
        .map(|v| {
            [
                v[0].re.to_bits(),
                v[0].im.to_bits(),
                v[1].re.to_bits(),
                v[1].im.to_bits(),
                v[2].re.to_bits(),
                v[2].im.to_bits(),
            ]
        })
        .collect()
}

#[test]
fn rank_four_bubbles_match_exact_masters_including_tiny_momenta() {
    let bits = 300;
    let mut exact = SymbolicaMasters::new(Float::new(bits));
    let mut native = OneLoopNativeMasters::new().unwrap();
    let (top, higgs) = (C64::real(29929.), C64::real(15625.));
    let complex_top = C64::new(29929., -259.5);
    let masses = [
        (top, higgs),
        (higgs, top),
        (top, C64::ZERO),
        (C64::ZERO, complex_top),
        (complex_top, complex_top),
        (top, top),
        (C64::ZERO, C64::ZERO),
        (complex_top, higgs),
    ];
    let momenta = [
        0., 1e-6, -1e-4, 1e-2, -1e-2, 1., -1., 100., 15625., 29929., 59858., 119716., -2.5e5,
        2.5e5, 1e7,
    ];
    let mut worst = 0f64;
    let mut checked = 0;
    for &(m0, m1) in &masses {
        for &p in &momenta {
            // As above: the 300-bit symbolic reference fails here.
            if p == 2.5e5 && m1.is_zero() && !m0.is_zero() {
                continue;
            }
            let mu2 = C64::real(1e4);
            let args = [C64::real(p), m0, m1, mu2];
            let value = native.bubble_rank4(args[0], m0, m1, mu2).unwrap();
            // B0..B111 of the rank-four family are bitwise the rank-three ones.
            let rank3 = native.bubble(args[0], m0, m1, mu2).unwrap();
            assert_eq!(bit_patterns(&value[..4]), bit_patterns(&rank3), "{args:?}");
            let reference = exact_b1111(&mut exact, bits, args);
            let scale = value.iter().map(norm3).fold(norm3(&reference), f64::max);
            let error = difference(&value[4], &reference) / scale.max(1e-300);
            assert!(
                error <= B1111_TOLERANCE,
                "B1111{args:?}: native {:?} exact {reference:?} ({error:e})",
                value[4]
            );
            // The 1/eps pole is exactly 1/5 and there is no double pole,
            // except for the scaleless bubble (zero in OneLOop's convention).
            let pole = if args[..3].iter().all(|z| z.is_zero()) {
                C64::ZERO
            } else {
                C64::real(0.2)
            };
            assert_eq!(value[4][1], pole, "{args:?}");
            assert_eq!(value[4][2], C64::ZERO, "{args:?}");
            worst = worst.max(error);
            checked += 1;
        }
    }
    assert!(checked > 100);
    eprintln!("FAST_NATIVE_B1111 {checked} points, max error vs 300-bit / max_k |B_k|: {worst:e}");
}

/// `B_n(p; m0, m1) = (-1)^n sum_k C(n,k) B_k(p; m1, m0)` at rank four, the
/// identity OneLOop's `bub1111` violates by `2 |B1111|` (wrong swap sign), on
/// every bubble original Ninja requested in the reference cases; there
/// `BubbleRank4` also reproduces the native rank-2/3 families bitwise.
#[test]
fn rank_four_bubbles_on_reference_calls_satisfy_the_swap_identity() {
    const BINOMIAL: [f64; 5] = [1., 4., 6., 4., 1.];
    let root =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../validation/fast/ninja_reference");
    let mut native = OneLoopNativeMasters::new().unwrap();
    let mut checked = 0;
    let mut worst = 0f64;
    for dir in [root.join("ninja"), root.join("conditioning/ninja")] {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            let lanes = parse_ninja(&std::fs::read_to_string(&path).unwrap(), None).unwrap();
            for call in lanes.iter().flat_map(|l| &l.calls) {
                if !matches!(
                    call.family,
                    FastMaster::BubbleRank2 | FastMaster::BubbleRank3
                ) {
                    continue;
                }
                let mut lower = vec![[C64::ZERO; 3]; call.family.outputs()];
                let mut rank4 = vec![[C64::ZERO; 3]; 5];
                let mut ok = [false];
                FastMasters::evaluate(&mut native, call.family, &call.args, &mut lower, &mut ok)
                    .unwrap();
                assert!(ok[0]);
                FastMasters::evaluate(
                    &mut native,
                    FastMaster::BubbleRank4,
                    &call.args,
                    &mut rank4,
                    &mut ok,
                )
                .unwrap();
                assert!(ok[0], "{}: {:?}", path.display(), call.args);
                assert_eq!(
                    bit_patterns(&rank4[..lower.len()]),
                    bit_patterns(&lower),
                    "{:?}",
                    call.args
                );
                let swapped = [call.args[0], call.args[2], call.args[1], call.args[3]];
                let mut other = vec![[C64::ZERO; 3]; 5];
                FastMasters::evaluate(
                    &mut native,
                    FastMaster::BubbleRank4,
                    &swapped,
                    &mut other,
                    &mut ok,
                )
                .unwrap();
                assert!(ok[0]);
                let mut sum = [C64::ZERO; 3];
                let mut terms = norm3(&rank4[4]);
                for (k, value) in other.iter().enumerate() {
                    for j in 0..3 {
                        sum[j] += value[j] * BINOMIAL[k];
                    }
                    terms += BINOMIAL[k] * norm3(value);
                }
                let error = difference(&rank4[4], &sum) / terms;
                assert!(
                    error <= 1e-14,
                    "{}: B1111{:?} = {:?}, swapped sum {sum:?} ({error:e})",
                    path.display(),
                    call.args,
                    rank4[4]
                );
                worst = worst.max(error);
                checked += 1;
            }
        }
    }
    assert!(checked > 200, "only {checked} bubble calls checked");
    eprintln!("FAST_NATIVE_B1111 swap identity on {checked} Ninja bubble calls: max {worst:e}");
}

/// Rank-four rows with every input class of the batched interfaces: the
/// fixtures, both mass orders, complex masses, zero momentum and on-shell
/// rows that OneLOop's threshold snaps (`|m²| < 1e-9`, `|p² - m²| < 1e-9`).
fn rank_four_rows() -> Vec<[C64; 4]> {
    let one = C64::real(1.);
    vec![
        [
            C64::real(3.),
            C64::new(0.7, -0.03),
            C64::new(1.4, -0.08),
            one,
        ],
        [C64::ZERO, C64::real(2.), C64::real(5.), one],
        [C64::ZERO, C64::real(5.), C64::real(2.), one],
        [C64::ZERO, C64::real(3.), C64::real(3.), C64::real(7.)],
        [C64::real(-2.5), C64::real(1.), C64::real(1.5), one],
        [C64::real(10.), C64::real(1.), C64::real(2.), C64::real(4.)],
        [C64::real(2.), C64::ZERO, C64::real(2.), one],
        [C64::real(2. + 3e-10), C64::real(1e-12), C64::real(2.), one],
        [
            C64::real(4e-10),
            C64::real(173. * 173.),
            C64::new(80.4, -2.1),
            one,
        ],
        [C64::real(1e-3), C64::real(1.), C64::real(1.25), one],
        [
            C64::real(1e6),
            C64::new(29929., -259.5),
            C64::ZERO,
            C64::real(1e4),
        ],
    ]
}

/// `OneLoopNativeMasters` serves B1111 bitwise identically through the batched
/// `FastMasters` call (one native call per family), its row-wise fallback, the
/// generic `MasterProvider` (strict and batch), and `SymbolicaMasters<f64>`
/// (the same native family); `[B0..B111]` are bitwise those of the rank-two and
/// rank-three families and of the `MasterProvider` B0/B1/B11/B111 requests.
#[test]
fn rank_four_bubble_interfaces_are_bitwise_identical() {
    let rows = rank_four_rows();
    let args: Vec<C64> = rows.iter().flatten().copied().collect();
    let mut native = OneLoopNativeMasters::new().unwrap();
    let mut symbolica = SymbolicaMasters::new(0f64);
    let mut rank4 = vec![[C64::ZERO; 3]; rows.len() * 5];
    let mut rank3 = vec![[C64::ZERO; 3]; rows.len() * 4];
    let mut rank2 = vec![[C64::ZERO; 3]; rows.len() * 3];
    let mut ok = vec![false; rows.len()];
    FastMasters::evaluate(
        &mut native,
        FastMaster::BubbleRank4,
        &args,
        &mut rank4,
        &mut ok,
    )
    .unwrap();
    assert!(ok.iter().all(|&ok| ok));
    FastMasters::evaluate(
        &mut native,
        FastMaster::BubbleRank3,
        &args,
        &mut rank3,
        &mut ok,
    )
    .unwrap();
    FastMasters::evaluate(
        &mut native,
        FastMaster::BubbleRank2,
        &args,
        &mut rank2,
        &mut ok,
    )
    .unwrap();
    for (row, &point) in rows.iter().enumerate() {
        let family = &rank4[row * 5..row * 5 + 5];
        assert_eq!(
            bit_patterns(&family[..4]),
            bit_patterns(&rank3[row * 4..row * 4 + 4])
        );
        assert_eq!(
            bit_patterns(&family[..3]),
            bit_patterns(&rank2[row * 3..row * 3 + 3])
        );
        // Single-row FastMasters call and the convenience accessor.
        let mut single = vec![[C64::ZERO; 3]; 5];
        let mut single_ok = [false];
        FastMasters::evaluate(
            &mut native,
            FastMaster::BubbleRank4,
            &point,
            &mut single,
            &mut single_ok,
        )
        .unwrap();
        assert!(single_ok[0]);
        assert_eq!(bit_patterns(&single), bit_patterns(family));
        let accessor = native
            .bubble_rank4(point[0], point[1], point[2], point[3])
            .unwrap();
        assert_eq!(bit_patterns(&accessor), bit_patterns(family));
        // The generic MasterProvider interfaces, one integral per request.
        for (k, integral) in [
            MasterIntegral::B0,
            MasterIntegral::B1,
            MasterIntegral::B11,
            MasterIntegral::B111,
            MasterIntegral::B1111,
        ]
        .into_iter()
        .enumerate()
        {
            let value = provider_row(&mut native, integral, point);
            assert_eq!(
                bit_patterns(&[value]),
                bit_patterns(&family[k..k + 1]),
                "{integral:?}{point:?}"
            );
        }
        let value = provider_row(&mut symbolica, MasterIntegral::B1111, point);
        assert_eq!(
            bit_patterns(&[value]),
            bit_patterns(&family[4..5]),
            "SymbolicaMasters{point:?}"
        );
    }
    // A batch MasterProvider request over all rows at once.
    let invariants: Vec<_> = rows.iter().map(|r| r[0].to_symbolica()).collect();
    let masses: Vec<_> = rows
        .iter()
        .flat_map(|r| [r[1].to_symbolica(), r[2].to_symbolica()])
        .collect();
    let scales: Vec<_> = rows.iter().map(|r| r[3].to_symbolica()).collect();
    for provider in [
        &mut native as &mut dyn MasterProvider<Complex<f64>>,
        &mut symbolica,
    ] {
        let mut output = vec![[Complex::new(0., 0.); 3]; rows.len()];
        let mut statuses = vec![MasterStatus::Failed(MasterFailure::DerivativeShape); rows.len()];
        provider
            .evaluate_batch(
                MasterRequest {
                    integral: MasterIntegral::B1111,
                    invariants: &invariants,
                    masses_squared: &masses,
                    scales_squared: &scales,
                },
                &mut output,
                &mut statuses,
            )
            .unwrap();
        assert!(statuses.iter().all(|s| *s == MasterStatus::Success));
        for (row, value) in output.iter().enumerate() {
            assert_eq!(
                bit_patterns(&[value.map(C64::from_symbolica)]),
                bit_patterns(&rank4[row * 5 + 4..row * 5 + 5])
            );
        }
    }
    // A row outside the domain contract (Im m² > 0) sends the family to the
    // row-wise fallback: it fails alone, the others keep their bits.
    let mut mixed = args.clone();
    mixed.splice(
        4..4,
        [
            C64::real(1.),
            C64::new(1., 0.5),
            C64::real(2.),
            C64::real(1.),
        ],
    );
    let mut out = vec![[C64::new(7., 7.); 3]; (rows.len() + 1) * 5];
    let mut ok = vec![true; rows.len() + 1];
    FastMasters::evaluate(
        &mut native,
        FastMaster::BubbleRank4,
        &mixed,
        &mut out,
        &mut ok,
    )
    .unwrap();
    assert!(!ok[1]);
    for (row, source) in [0usize]
        .into_iter()
        .chain(1..rows.len())
        .zip([0usize].into_iter().chain(2..rows.len() + 1))
    {
        assert!(ok[source]);
        assert_eq!(
            bit_patterns(&out[source * 5..source * 5 + 5]),
            bit_patterns(&rank4[row * 5..row * 5 + 5])
        );
    }
}

/// The corrected-sign fixtures through `OneLoopNativeMasters`:
/// oneloopmaster's reference complex-mass row and B1111(0; 2, 5)
/// = int x⁴ [1/eps - log(2 + 3x)] dx.
#[test]
fn rank_four_bubble_fixtures() {
    let mut native = OneLoopNativeMasters::new().unwrap();
    for (args, finite) in [
        (
            rank_four_rows()[0],
            C64::new(0.029_939_862_301_517_683, 0.017_531_313_325_083_266),
        ),
        (rank_four_rows()[1], C64::real(-0.299_847_502_997_085_9)),
    ] {
        let value = provider_row(&mut native, MasterIntegral::B1111, args);
        assert!(
            (value[0] - finite).abs() <= 64. * f64::EPSILON * finite.abs(),
            "{args:?}: {value:?}"
        );
        assert_eq!(value[1], C64::real(0.2));
        assert_eq!(value[2], C64::ZERO);
    }
}
