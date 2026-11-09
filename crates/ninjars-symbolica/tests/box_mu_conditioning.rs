//! The box mu-expansion's conditioning bound (`ninjars::conditioning`,
//! `docs/box-mu-conditioning.md`) on near-planar 2 -> 3 lanes.
//!
//! The fixtures of `validation/conditioning/box-mu` hold single families of
//! loop-induced and Born-interfered processes at points whose event plane
//! contains the beam, tilted out of it by a small angle, plus synthetic rank
//! n+1 and n+2 numerators on the same geometry and regular controls. Each
//! lane's reference is the generic engine at two working precisions of at
//! least 100 digits that agree (recorded in the fixture with the agreement).
//!
//! - Every engine and carrier either refuses a lane (a typed status) or agrees
//!   with the reference within `TOLERANCE` of the fixture's scale, the largest
//!   reference modulus among its lanes (one family at one point: a suppressed
//!   helicity lane is measured on the scale of the family, the scale of its
//!   contribution to the squared amplitude).
//! - The fast engine refuses a fixture lane with both noise thresholds at zero
//!   (`FastOptions::stability_threshold` and `relative_tolerance`): the bound
//!   does not read them.
//! - Controls keep their recorded statuses and binary64 bits (the values of
//!   NinjaRS main 98e223da, before the bound); a lane refused there stays
//!   refused with the same kind of failure.
//! - A fixture lane that loses more than the target for a reason outside the
//!   box mu-expansion records it (`outside_bound`: the localisation and, per
//!   carrier, the error) and is held to twice that error; the bound is not
//!   widened to refuse it (`docs/box-mu-conditioning.md`, section 4).

#[path = "common/conditioning.rs"]
mod conditioning;

use conditioning::{
    Fixture, LaneRef, Outcome, fast_lanes, generic_lanes, parse_fixture, relative_error,
};
use ninjars::Float;
use ninjars::fast::FastOptions;
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::PathBuf;
use symbolica::domains::float::DoubleFloat;

const DIRECTORY: &str = "validation/conditioning/box-mu";
const TOLERANCE: f64 = 1e-5;

/// Engines and carriers: `fast` (default options), `fast-conditioned`,
/// `fast-unchecked` (zero noise thresholds), the generic engine in binary64,
/// double-double and MPFR at 107, 128 and 200 bits. A rank n+2 fixture takes
/// the fast engine's rank n+2 stages (`engine_x2.rs`).
const CARRIERS: [&str; 8] = [
    "fast",
    "fast-conditioned",
    "fast-unchecked",
    "f64",
    "dd",
    "float:107",
    "float:128",
    "float:200",
];

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

struct Loaded {
    fixture: Fixture,
    raw: Value,
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

fn options(carrier: &str) -> FastOptions {
    match carrier {
        "fast" => FastOptions::default(),
        "fast-conditioned" => FastOptions {
            conditioned_cuts: true,
            ..FastOptions::default()
        },
        "fast-unchecked" => FastOptions {
            stability_threshold: 0.,
            relative_tolerance: 0.,
            ..FastOptions::default()
        },
        _ => unreachable!(),
    }
}

/// One carrier on lanes of equal `(n, rank)`.
fn run(carrier: &str, lanes: &[LaneRef<'_>]) -> Vec<Outcome> {
    match carrier {
        "fast" | "fast-conditioned" | "fast-unchecked" => {
            fast_lanes(lanes, options(carrier)).unwrap()
        }
        "f64" => generic_lanes(lanes, 0f64, 53).unwrap(),
        "dd" => generic_lanes(lanes, DoubleFloat::from(0.), 106).unwrap(),
        other => {
            let bits: u32 = other.strip_prefix("float:").unwrap().parse().unwrap();
            generic_lanes(lanes, Float::with_val(bits, 0), bits).unwrap()
        }
    }
}

/// Every carrier on every fixture lane, batched per `(n, rank)`:
/// `(carrier, fixture index, lane) -> outcome`.
fn outcomes(loaded: &[Loaded]) -> BTreeMap<(&'static str, usize, usize), Outcome> {
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
        for carrier in CARRIERS {
            for (&(index, lane), outcome) in members.iter().zip(run(carrier, &lanes)) {
                out.insert((carrier, index, lane), outcome);
            }
        }
    }
    out
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

#[test]
fn every_carrier_refuses_or_agrees_with_the_reference() {
    let loaded = fixtures();
    let outcomes = outcomes(&loaded);
    let mut failures = Vec::new();
    let mut served = 0;
    let mut refused = 0;
    for ((carrier, index, lane), outcome) in &outcomes {
        let fixture = &loaded[*index].fixture;
        if !outcome.success {
            refused += 1;
            continue;
        }
        served += 1;
        let reference = fixture.references[*lane]
            .as_ref()
            .unwrap_or_else(|| panic!("{}: lane {lane} has no reference", fixture.name));
        let value = outcome.laurent.unwrap()[0];
        let r =
            ninjars::fast::C64::new(reference[0].parse().unwrap(), reference[1].parse().unwrap());
        let error = (value - r).abs() / scale(fixture);
        // A loss outside the box mu-expansion, localised in the fixture
        // (`outside_bound`, per carrier the recorded error): kept as a
        // regression record at that error, not as within the target.
        if let Some(recorded) = loaded[*index].raw["outside_bound"]["errors"][*carrier].as_f64() {
            if error.is_nan() || error > 2. * recorded {
                failures.push(format!(
                    "{} lane {lane} {carrier}: {error:.3e} off, above twice the recorded {recorded:.3e}",
                    fixture.name
                ));
            }
            continue;
        }
        if error.is_nan() || error > TOLERANCE {
            failures.push(format!(
                "{} lane {lane} {carrier}: served {:.3e} off (lane-relative {:.3e})",
                fixture.name,
                error,
                relative_error(value, reference)
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
fn zero_noise_thresholds_still_refuse() {
    let loaded = fixtures();
    let marked: Vec<_> = loaded
        .iter()
        .filter(|l| l.raw["refused_with_zero_thresholds"] == Value::Bool(true))
        .collect();
    assert!(
        !marked.is_empty(),
        "no fixture marked refused_with_zero_thresholds"
    );
    for l in marked {
        let lanes: Vec<LaneRef<'_>> = (0..l.fixture.tensors.len())
            .map(|lane| (&l.fixture, lane))
            .collect();
        for outcome in fast_lanes(&lanes, options("fast-unchecked")).unwrap() {
            assert_eq!(outcome.status, "Unstable", "{}", l.fixture.name);
        }
    }
}

#[test]
fn controls_keep_their_statuses_and_bits() {
    let loaded = fixtures();
    let mut checked = 0;
    for l in &loaded {
        let Some(controls) = l.raw["controls"].as_object() else {
            continue;
        };
        let lanes: Vec<LaneRef<'_>> = (0..l.fixture.tensors.len())
            .map(|lane| (&l.fixture, lane))
            .collect();
        for (carrier, expected) in controls {
            let got = run(carrier, &lanes);
            for (lane, (outcome, expected)) in
                got.iter().zip(expected.as_array().unwrap()).enumerate()
            {
                // A lane refused before stays refused with the same kind of
                // failure (the bound may refuse it at an earlier cut); a
                // served lane keeps its status and bits.
                let status = expected["status"].as_str().unwrap();
                let kind = |s: &str| s.split("failure: ").nth(1).map(str::to_owned);
                if status == "Success" {
                    assert_eq!(
                        outcome.status, status,
                        "{} lane {lane} {carrier}",
                        l.fixture.name
                    );
                } else {
                    assert!(!outcome.success, "{} lane {lane} {carrier}", l.fixture.name);
                    assert_eq!(
                        kind(&outcome.status),
                        kind(status),
                        "{} lane {lane} {carrier}: {} vs {status}",
                        l.fixture.name,
                        outcome.status
                    );
                }
                if let Some(bits) = expected["finite_bits"].as_array() {
                    let value = outcome.laurent.unwrap()[0];
                    assert_eq!(
                        [value.re.to_bits(), value.im.to_bits()],
                        [bits[0].as_u64().unwrap(), bits[1].as_u64().unwrap()],
                        "{} lane {lane} {carrier}",
                        l.fixture.name
                    );
                }
                checked += 1;
            }
        }
    }
    assert!(checked > 0, "no control recorded");
}
