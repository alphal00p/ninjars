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

mod common;

use common::ninja_reference::{crosscheck, parse_case, parse_ninja};
use std::path::PathBuf;

/// Scaled (relative to each lane's largest Ninja value, no absolute floor)
/// agreement required of every Laurent coefficient and of the
/// cut-constructible/rational split. Observed: physical cases <= 7.1e-13;
/// edge cases <= 3.9e-12 (rank n+1, where original Ninja itself loses
/// digits; see validation/fast/evidence/).
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
    let (mut unstable, mut lanes, mut worst) = (0, 0, 0f64);
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
        for lane in &report.lanes {
            assert!(
                lane.status_matches(),
                "{name} {}: fast {:?}, Ninja return value {}",
                lane.id,
                lane.status,
                lane.ninja_status
            );
            let error = lane.laurent.max(lane.cut_constructible).max(lane.rational);
            assert!(
                error <= TOLERANCE,
                "{name} {}: scaled error {error:e} (laurent {:e}, cut-constructible {:e}, rational {:e})",
                lane.id,
                lane.laurent,
                lane.cut_constructible,
                lane.rational
            );
            unstable += usize::from(lane.ninja_status != 0);
            lanes += 1;
            worst = worst.max(error);
        }
    }
    assert!(unstable >= 8, "edge cases must include unstable points");
    eprintln!(
        "FAST_NINJA_REFERENCE cases={} lanes={lanes} unstable={unstable} max_scaled_error={worst:e}",
        files.len()
    );
}
