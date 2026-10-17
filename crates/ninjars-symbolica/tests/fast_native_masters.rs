//! Accuracy of the binary64 bubble masters of `OneLoopNativeMasters`.
//!
//! B0, B1, B11 and B111 are a port of OneLOop's direct bubble formulas
//! (`src/oneloop_native.rs`). They are checked against oneloopmaster's
//! symbolic masters evaluated in 300-bit `Float` arithmetic (where the
//! Passarino-Veltman relations they encode are harmless) on a grid that
//! includes tiny `|p²| << m²` (where binary64 Passarino-Veltman reductions lose
//! `(m²/p²)^rank` relative accuracy), zero momentum, thresholds, on-shell
//! points, massless and complex masses; and against the OneLOop values
//! original Ninja requested on every reference case. Run with
//! `--features native-masters --test-threads=1`.

mod common;

use common::ninja_reference::parse_ninja;
use ninjars::fast::{C64, FastMaster, FastMasters};
use ninjars::{Complex, Float, MasterIntegral, MasterProvider, MasterRequest};
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
