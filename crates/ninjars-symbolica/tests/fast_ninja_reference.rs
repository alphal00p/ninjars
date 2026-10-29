//! The binary64 fast engine against ORIGINAL Ninja with identical masters.
//!
//! `validation/fast/ninja_reference/` holds the seven physical benchmark cases
//! (hgg/four-photon/gg->HH/gg->HHH top loops, copied from
//! `validation/benchmark/cases`) and edge cases from
//! `validation/fast/make_edge_cases.py` (lightlike and vanishing invariants,
//! massless/complex internal masses, infrared-divergent massless triangles and
//! boxes, the second and third tadpole reference bases, coincident shifts and
//! collinear momenta next to stable points), each with original Ninja's
//! results (`validation/benchmark/ninja_bench.cc`, build o3-fortranrules)
//! including every OneLOop master call it made. The fast engine reduces every
//! lane with `TensorNumerator` and those exact master values; see
//! `tests/common/ninja_reference.rs`.
//!
//! Where the two differ by more than [`TOLERANCE`], the exact reduction of the
//! same binary64 inputs with the same masters (the generic engine in
//! `Float(256)`, `tests/common/precision.rs`) settles it: the fast engine must
//! be the more accurate one, and its cut-constructible/rational split must
//! follow Ninja's (as in `fast_conditioning.rs`). This happens on 20 lanes,
//! since the fast engine's light-like spinors (`fast/geometry.rs` `Spinor`)
//! differ from Ninja's:
//! - the 18 `p² = ±0.01` lanes of `edge-small-p2-n2-r{1,2,3}`: their bubble
//!   basis has a light-like `e1` with `p⁰ - p³ = 4e-13`, inside Ninja's
//!   `INFRARED_EPS` window, which drops its transverse momentum (Ninja 5.6e-9
//!   to 6.3e-8 off the exact reduction, the fast engine at most 6.6e-11);
//! - two rank n+1 tadpole-basis lanes, where both lose digits and the
//!   difference moved just above the tolerance: `edge-tadpole-bases-n2-r3`
//!   `second-basis-complex` (Ninja 1.9e-11 off, the fast engine 5.8e-12) and
//!   `edge-tadpole-bases-n3-r4` `third-basis-mixed` (2.7e-11 and 1.1e-11).

mod common;

use common::ninja_reference::{crosscheck, parse_case, parse_ninja};
use common::precision::precision;
use std::path::PathBuf;

/// Scaled (relative to each lane's largest Ninja value, no absolute floor)
/// agreement required of every Laurent coefficient and of the
/// cut-constructible/rational split. Observed: physical cases <= 4.8e-13;
/// edge cases <= 4.6e-12 (rank n+1, where original Ninja itself loses
/// digits; see validation/fast/evidence/); the 20 lanes beyond it are settled
/// by the exact reduction (module documentation).
const TOLERANCE: f64 = 1e-11;

#[test]
fn fast_engine_matches_original_ninja_with_identical_masters() {
    let root =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../validation/fast/ninja_reference");
    let mut files: Vec<PathBuf> = std::fs::read_dir(root.join("cases"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    files.sort();
    assert!(
        files.len() >= 20,
        "missing reference cases in {}",
        root.display()
    );
    let (mut unstable, mut lanes, mut worst, mut departed) = (0, 0, 0f64, 0);
    for path in &files {
        let case = parse_case(&std::fs::read_to_string(path).unwrap()).unwrap();
        let reference = root.join("ninja").join(path.file_name().unwrap());
        let ninja = parse_ninja(&std::fs::read_to_string(&reference).unwrap(), None).unwrap();
        let report = crosscheck(&case, &ninja).unwrap();
        let name = &report.name;
        assert_eq!(
            report.unmatched_masters, 0,
            "{name}: unmatched master requests"
        );
        assert!(
            report.batches_bitwise,
            "{name}: {:?}",
            report.batch_failures
        );
        let mut departures = Vec::new();
        for (index, lane) in report.lanes.iter().enumerate() {
            assert!(
                lane.status_matches(),
                "{name} {}: fast {:?}, Ninja return value {}",
                lane.id,
                lane.status,
                lane.ninja_status
            );
            let error = lane.laurent.max(lane.cut_constructible).max(lane.rational);
            if error > TOLERANCE {
                eprintln!(
                    "FAST_NINJA_REFERENCE {name} {}: scaled difference {error:e} (laurent {:e}, \
                     cut-constructible {:e}, rational {:e})",
                    lane.id, lane.laurent, lane.cut_constructible, lane.rational
                );
                departures.push(index);
            } else {
                worst = worst.max(error);
            }
            unstable += usize::from(lane.ninja_status != 0);
            lanes += 1;
        }
        if !departures.is_empty() {
            for lane in precision(&case, &ninja, 256, Some(departures.as_slice())).unwrap() {
                eprintln!(
                    "FAST_NINJA_REFERENCE {name} {}: against the exact reduction, fast {:e}, Ninja {:e}",
                    lane.id, lane.fast_laurent, lane.ninja_laurent
                );
                assert!(
                    lane.fast_laurent < lane.ninja_laurent,
                    "{name} {}: the fast engine departs from original Ninja but the exact \
                     reduction does not side with it (fast error {:e}, Ninja error {:e})",
                    lane.id,
                    lane.fast_laurent,
                    lane.ninja_laurent
                );
                // The cut-constructible/rational split follows Ninja's (as in
                // `fast_conditioning.rs`).
                assert!(
                    (lane.fast_split - lane.ninja_split).abs() <= 4. * lane.ninja_laurent + 1e-12,
                    "{name} {}: split {:e} vs Ninja {:e}",
                    lane.id,
                    lane.fast_split,
                    lane.ninja_split
                );
                departed += 1;
            }
        }
    }
    assert!(unstable >= 8, "edge cases must include unstable points");
    eprintln!(
        "FAST_NINJA_REFERENCE cases={} lanes={lanes} unstable={unstable} departed={departed} \
         max_scaled_error={worst:e}",
        files.len()
    );
}
