//! Ill-conditioned points: the fast engine is as accurate as original Ninja.
//!
//! On points whose result is a tiny remainder of large cancellations, the
//! binary64 fast engine and original Ninja can differ well above 1e-9 (up to
//! 5e-6 relative on `syn-gghhh-ggHHH-n5-r5`). This test settles which one is
//! right: the generic NinjaRS engine reduces the same tensor numerator in
//! 256-bit `Float` arithmetic with exactly the OneLOop masters original Ninja
//! used (`tests/common/precision.rs`), and both binary64 results are compared
//! with that exact reduction. Cases are copied from the main checkout's
//! `validation/benchmark/synthetic` (gg->HHH-like rank-5 pentagons with
//! complex top masses; dense random n=6 rank-5 numerators), with original
//! Ninja's results and master calls from `validation/benchmark/ninja_bench.cc`
//! (o3-fortranrules). Run with `--test-threads=1`.

mod common;

use common::ninja_reference::{parse_case, parse_ninja};
use common::precision::precision;
use ninjars::fast::FastStatus;
use std::path::PathBuf;

#[test]
fn fast_engine_is_as_accurate_as_original_ninja_on_ill_conditioned_points() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../validation/fast/ninja_reference/conditioning");
    let mut worst_ratio = 0f64;
    let mut worst_ninja = 0f64;
    for name in ["syn-gghhh-ggHHH-n5-r5", "syn-n6-r5"] {
        let file = format!("{name}.json");
        let case =
            parse_case(&std::fs::read_to_string(root.join("cases").join(&file)).unwrap()).unwrap();
        let ninja = parse_ninja(
            &std::fs::read_to_string(root.join("ninja").join(&file)).unwrap(),
            None,
        )
        .unwrap();
        for lane in precision(&case, &ninja, 256, None).unwrap() {
            assert_eq!(lane.fast_status, FastStatus::Success, "{name} {}", lane.id);
            eprintln!(
                "FAST_CONDITIONING {name} {}: scale {:e}, fast error {:e}, Ninja error {:e}",
                lane.id, lane.scale, lane.fast_laurent, lane.ninja_laurent
            );
            // Errors relative to the exact result's largest modulus; a factor
            // of four covers the observed spread (0.7-2x) of two independent
            // binary64 roundings of the same cancellation.
            assert!(
                lane.fast_laurent <= 4. * lane.ninja_laurent + 1e-13,
                "{name} {}: fast error {:e} vs original Ninja's {:e}",
                lane.id,
                lane.fast_laurent,
                lane.ninja_laurent
            );
            // The cut-constructible/rational split follows Ninja's (the
            // generic engine moves some bubble terms between the two parts).
            assert!(
                (lane.fast_split - lane.ninja_split).abs() <= 4. * lane.ninja_laurent + 1e-12,
                "{name} {}: split {:e} vs Ninja {:e}",
                lane.id,
                lane.fast_split,
                lane.ninja_split
            );
            worst_ratio = worst_ratio.max(lane.fast_laurent / lane.ninja_laurent.max(1e-16));
            worst_ninja = worst_ninja.max(lane.ninja_laurent);
        }
    }
    // The cases must exercise genuinely ill-conditioned points.
    assert!(
        worst_ninja > 1e-6,
        "no ill-conditioned point ({worst_ninja:e})"
    );
    eprintln!("FAST_CONDITIONING max fast/Ninja error ratio {worst_ratio:.2}");
}
