//! The box-Gram conditioning bound (`ninjars::conditioning`,
//! `docs/box-gram-conditioning.md`) on lanes with a nearly degenerate box.
//!
//! The fixtures of `validation/conditioning/box-gram` hold one pentagon or
//! box lane each, with random tensor numerators: the two points of NinjaRS
//! issue #15 (HC05 `g g > x0 g g`, final state collinear to binary64
//! rounding; served `1e15 .. 1e34` off with status success before the
//! bound), 2 -> 3 points tilted out of a collinear final state by `1e-1 ..
//! 1e-16` rad, and planar points (event plane containing the beam, regular
//! boxes) as controls. Each lane's reference is the generic engine at 1800
//! and 2000 bits (recorded with their agreement).
//!
//! - Every carrier either refuses a lane (a typed status) or agrees with the
//!   reference within `TOLERANCE` of the fixture's scale.
//! - The bound acts as its model says (double-double and MPFR; binary64 is
//!   never refused with `DegenerateGram`): a lane whose largest predicted loss
//!   `(r + 1) / 2 · log2 λ²` (the fixture's `box_gram_log2_lambda2`, computed
//!   at 120 digits) exceeds `p - BOX_GRAM_MARGIN_BITS` by more than one bit
//!   is refused; one below it by more than one bit is never refused with
//!   `DegenerateGram`.
//! - The issue #15 lanes are refused (`DegenerateGram`) at 200, 256 and 266
//!   bits and served from 400 bits up.

#[path = "common/conditioning.rs"]
mod conditioning;

use conditioning::{Fixture, LaneRef, Outcome, generic_lanes, parse_fixture};
use ninjars::Float;
use ninjars::conditioning::{BOX_GRAM_MARGIN_BITS, box_gram_applies};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::OnceLock;
use symbolica::domains::float::DoubleFloat;

const DIRECTORY: &str = "validation/conditioning/box-gram";
const TOLERANCE: f64 = 1e-5;

/// The generic engine in binary64, double-double and MPFR.
const CARRIERS: [(&str, u32); 8] = [
    ("f64", 53),
    ("dd", 106),
    ("float:128", 128),
    ("float:200", 200),
    ("float:256", 256),
    ("float:266", 266),
    ("float:400", 400),
    ("float:512", 512),
];

struct Loaded {
    fixture: Fixture,
    raw: Value,
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixtures() -> Vec<Loaded> {
    let mut files: Vec<_> = std::fs::read_dir(root().join(DIRECTORY))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|e| e == "json"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no fixtures in {DIRECTORY}");
    files
        .iter()
        .map(|path| {
            let text = std::fs::read_to_string(path).unwrap();
            Loaded {
                fixture: parse_fixture(&text).unwrap_or_else(|e| panic!("{path:?}: {e}")),
                raw: serde_json::from_str(&text).unwrap(),
            }
        })
        .collect()
}

fn run(carrier: &str, lanes: &[LaneRef<'_>]) -> Vec<Outcome> {
    match carrier {
        "f64" => generic_lanes(lanes, 0f64, 53).unwrap(),
        "dd" => generic_lanes(lanes, DoubleFloat::from(0.), 106).unwrap(),
        other => {
            let bits: u32 = other.strip_prefix("float:").unwrap().parse().unwrap();
            generic_lanes(lanes, Float::with_val(bits, 0), bits).unwrap()
        }
    }
}

type Outcomes = BTreeMap<(&'static str, usize, usize), Outcome>;

/// Every carrier on every fixture lane, batched per `(n, rank)`, once for
/// all tests: `(carrier, fixture index, lane) -> outcome`.
fn outcomes() -> &'static (Vec<Loaded>, Outcomes) {
    static CACHE: OnceLock<(Vec<Loaded>, Outcomes)> = OnceLock::new();
    CACHE.get_or_init(|| {
        let loaded = fixtures();
        let mut groups: BTreeMap<(usize, usize), Vec<(usize, usize)>> = BTreeMap::new();
        for (index, l) in loaded.iter().enumerate() {
            for lane in 0..l.fixture.tensors.len() {
                groups
                    .entry((l.fixture.n, l.fixture.rank))
                    .or_default()
                    .push((index, lane));
            }
        }
        let mut out = BTreeMap::new();
        for members in groups.values() {
            let lanes: Vec<LaneRef<'_>> = members
                .iter()
                .map(|&(index, lane)| (&loaded[index].fixture, lane))
                .collect();
            for (carrier, _) in CARRIERS {
                for (&(index, lane), outcome) in members.iter().zip(run(carrier, &lanes)) {
                    out.insert((carrier, index, lane), outcome);
                }
            }
        }
        (loaded, out)
    })
}

fn bits(carrier: &str) -> u32 {
    CARRIERS.iter().find(|(c, _)| *c == carrier).unwrap().1
}

/// The fixture's scale: its largest reference modulus.
fn scale(fixture: &Fixture) -> f64 {
    fixture
        .references
        .iter()
        .flatten()
        .map(|r| {
            let re: f64 = r[0].parse().unwrap();
            let im: f64 = r[1].parse().unwrap();
            re.hypot(im)
        })
        .fold(0., f64::max)
}

/// The largest predicted loss `(r + 1) / 2 · log2 λ²` over the lane's boxes.
fn predicted_loss(l: &Loaded) -> f64 {
    let lambda = l.raw["box_gram_log2_lambda2"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .fold(f64::NEG_INFINITY, f64::max);
    (l.fixture.rank as f64 + 1.) / 2. * lambda
}

fn degenerate_gram(outcome: &Outcome) -> bool {
    outcome.status.contains("DegenerateGram")
}

#[test]
fn every_carrier_refuses_or_agrees_with_the_reference() {
    let (loaded, outcomes) = outcomes();
    let mut failures = Vec::new();
    let (mut served, mut refused) = (0, 0);
    for ((carrier, index, lane), outcome) in outcomes {
        let fixture = &loaded[*index].fixture;
        if !outcome.success {
            refused += 1;
            continue;
        }
        served += 1;
        let reference = fixture.references[*lane].as_ref().unwrap();
        let value = outcome.laurent.unwrap()[0];
        let r =
            ninjars::fast::C64::new(reference[0].parse().unwrap(), reference[1].parse().unwrap());
        let error = (value - r).abs() / scale(fixture);
        if error.is_nan() || error > TOLERANCE {
            failures.push(format!(
                "{} lane {lane} {carrier}: served {error:.3e} off",
                fixture.name
            ));
        }
    }
    assert!(
        served > 0 && refused > 0,
        "served {served}, refused {refused}"
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn the_bound_refuses_as_its_model_predicts() {
    let (loaded, outcomes) = outcomes();
    let mut failures = Vec::new();
    let mut acted = 0;
    for ((carrier, index, lane), outcome) in outcomes {
        let l = &loaded[*index];
        let budget = f64::from(bits(carrier) - BOX_GRAM_MARGIN_BITS);
        let loss = predicted_loss(l);
        if !box_gram_applies(bits(carrier)) {
            if degenerate_gram(outcome) {
                failures.push(format!(
                    "{} lane {lane} {carrier}: {} although the bound does not apply",
                    l.fixture.name, outcome.status
                ));
            }
        } else if loss > budget + 1. {
            acted += 1;
            if outcome.success {
                failures.push(format!(
                    "{} lane {lane} {carrier}: predicted loss {loss:.1} > {budget} but served",
                    l.fixture.name
                ));
            }
        } else if loss < budget - 1. && degenerate_gram(outcome) {
            failures.push(format!(
                "{} lane {lane} {carrier}: predicted loss {loss:.1} < {budget} but {}",
                l.fixture.name, outcome.status
            ));
        }
    }
    assert!(acted > 0);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn issue_15_points_are_refused_below_400_bits_and_served_above() {
    let (loaded, outcomes) = outcomes();
    let mut checked = 0;
    for ((carrier, index, lane), outcome) in outcomes {
        let l = &loaded[*index];
        if !l.fixture.name.starts_with("issue15-") {
            continue;
        }
        match bits(carrier) {
            200 | 256 | 266 => assert!(
                degenerate_gram(outcome),
                "{} lane {lane} {carrier}: {}",
                l.fixture.name,
                outcome.status
            ),
            400 | 512 => assert!(
                outcome.success,
                "{} lane {lane} {carrier}: {}",
                l.fixture.name, outcome.status
            ),
            _ => assert!(!outcome.success, "{} lane {lane} {carrier}", l.fixture.name),
        }
        checked += 1;
    }
    assert_eq!(checked, 2 * 2 * CARRIERS.len());
}

/// Regular boxes (planar points, the least tilted collinear point) are never
/// refused by the bound (binary64 is not tested by it).
#[test]
fn regular_and_planar_controls_are_never_refused_by_the_bound() {
    let (loaded, outcomes) = outcomes();
    let mut checked = 0;
    for ((carrier, index, lane), outcome) in outcomes {
        let l = &loaded[*index];
        if !(l.fixture.name.starts_with("planar-") || l.fixture.name.ends_with("-t1e-1")) {
            continue;
        }
        assert!(
            !degenerate_gram(outcome),
            "{} lane {lane} {carrier}: {}",
            l.fixture.name,
            outcome.status
        );
        checked += 1;
    }
    assert!(checked > 0);
}
