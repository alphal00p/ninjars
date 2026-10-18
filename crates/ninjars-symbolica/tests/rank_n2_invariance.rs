//! V7 (design section 7): routing, relabelling and batching invariance.
//!
//! On the exact dyadic cases of `validation/rank-n2/identity`
//! (`manifest.json`):
//! - V7a: a global loop-momentum shift `V_i -> V_i - P` with
//!   `N'(q) = N(q - P)` leaves the integral unchanged;
//! - V7b: relabelling the propagators (all dihedral orders for n <= 4, seven
//!   random permutations for n = 5, 6) leaves it unchanged;
//! - V7c: fast-engine results do not depend on the batch width or the lane
//!   order (bitwise).
//!
//! Tolerances: generic Float(256) 1e-60 scaled, fast binary64
//! `max(1e-11, 10 noise)` scaled, with `noise` the base lane's change when its
//! propagators are listed in reverse order (as `tests/fast_parity.rs`).
//! The rank n+2 cases run at integration. Now, the same identities run on
//! their rank n+1 truncations (relabelling commutes with truncation; the
//! shifted numerator of the truncation is re-expanded exactly here), and the
//! exact re-expansion is checked against the generator's own shifted cases.

mod rank_n2_support;

use std::collections::HashMap;

use ninjars::fast::{C64, FastStatus};
use ninjars::{Complex, Float};
use ninjars_symbolica::OneLoopNativeMasters;
use rank_n2_support::{
    cases_in, controls, fast_noise,
    harness::{
        BenchCase, Engines, fast_values, run_fast, scaled_difference, scaled_difference_f64,
    },
    rank_n2, shifted, truncated,
};
use serde_json::Value;

const BITS: u32 = 256;
const FLOAT_TOLERANCE: f64 = 1e-60;
const FAST_TOLERANCE: f64 = 1e-11;

fn manifest() -> Value {
    serde_json::from_str(
        &std::fs::read_to_string(rank_n2().join("identity/manifest.json")).unwrap(),
    )
    .unwrap()
}

fn load(name: &str) -> BenchCase {
    BenchCase::load(&rank_n2().join(format!("identity/cases/{name}.json"))).unwrap()
}

/// `(case, lane) == (base, lane)` pairs of one manifest key, with the shift
/// `P` for V7a.
struct Pair {
    case: String,
    lane: usize,
    base: String,
    base_lane: usize,
    shift: Option<[i64; 4]>,
}

fn pairs(key: &str) -> Vec<Pair> {
    manifest()[key]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| Pair {
            case: e["case"].as_str().unwrap().into(),
            lane: e["lane"].as_u64().unwrap() as usize,
            base: e["equals"]["case"].as_str().unwrap().into(),
            base_lane: e["equals"]["lane"].as_u64().unwrap() as usize,
            shift: e["P"].as_array().map(|p| {
                let v: Vec<i64> = p.iter().map(|x| x.as_i64().unwrap()).collect();
                [v[0], v[1], v[2], v[3]]
            }),
        })
        .collect()
}

/// One case reduced by both engines: Float(256) Laurent coefficients and
/// scales, binary64 Laurent coefficients and noise, per lane.
struct Reduced {
    float: Vec<[Complex<Float>; 3]>,
    float_scale: Vec<f64>,
    fast: Vec<[C64; 3]>,
    noise: Vec<f64>,
}

fn reduce(case: &BenchCase, engines: &mut Engines, masters: &mut OneLoopNativeMasters) -> Reduced {
    use symbolica::domains::float::RealLike;
    let (engine, float_masters) = engines.float(BITS).unwrap();
    let lanes = engine.evaluate(case, case.rank, float_masters).unwrap();
    let mut out = Reduced {
        float: Vec::new(),
        float_scale: Vec::new(),
        fast: Vec::new(),
        noise: fast_noise(case, case.rank, masters),
    };
    for (index, lane) in lanes.into_iter().enumerate() {
        assert_eq!(
            lane.status,
            ninjars::EvaluationStatus::Success,
            "{} lane {index}",
            case.name
        );
        let [a, b, c, ..] = lane.values;
        out.float_scale.push(
            [&a, &b, &c]
                .iter()
                .map(|z| z.re.to_f64().hypot(z.im.to_f64()))
                .fold(0., f64::max)
                .max(1e-300),
        );
        out.float.push([a, b, c]);
    }
    let run = run_fast(case, case.rank, case.lanes.len(), masters).unwrap();
    for lane in 0..case.lanes.len() {
        assert_eq!(
            run.statuses[lane],
            FastStatus::Success,
            "{} lane {lane}",
            case.name
        );
        let v = fast_values(&run.results[lane]);
        out.fast.push([v[0], v[1], v[2]]);
    }
    out
}

/// Compare every pair, reducing each distinct `case(pair)` and
/// `base(pair)` once with both engines (Float(256) and the binary64 fast
/// engine).
fn check_pairs(
    label: &str,
    pairs: &[Pair],
    case: impl Fn(&Pair) -> BenchCase,
    base: impl Fn(&Pair) -> BenchCase,
) {
    let mut engines = Engines::default();
    let mut masters = OneLoopNativeMasters::new().unwrap();
    let mut cases: HashMap<String, Reduced> = HashMap::new();
    let mut bases: HashMap<String, Reduced> = HashMap::new();
    let (mut worst_float, mut worst_fast) = (0_f64, 0_f64);
    for pair in pairs {
        if !cases.contains_key(&pair.case) {
            let reduced = reduce(&case(pair), &mut engines, &mut masters);
            cases.insert(pair.case.clone(), reduced);
        }
        if !bases.contains_key(&pair.base) {
            let reduced = reduce(&base(pair), &mut engines, &mut masters);
            bases.insert(pair.base.clone(), reduced);
        }
        let (x, y) = (&cases[&pair.case], &bases[&pair.base]);
        let name = format!(
            "{label} {} lane {} = {} lane {}",
            pair.case, pair.lane, pair.base, pair.base_lane
        );
        let d = scaled_difference(
            &x.float[pair.lane],
            &y.float[pair.base_lane],
            y.float_scale[pair.base_lane],
        );
        assert!(d <= FLOAT_TOLERANCE, "{name}: Float(256) {d:e}");
        worst_float = worst_float.max(d);
        let (u, v) = (&x.fast[pair.lane], &y.fast[pair.base_lane]);
        let scale = v.iter().map(|z| z.abs()).fold(0., f64::max).max(1e-300);
        let d = scaled_difference_f64(u, v, scale);
        let noise = y.noise[pair.base_lane];
        assert!(
            d <= FAST_TOLERANCE.max(10. * noise),
            "{name}: binary64 {d:e} (noise {noise:e})"
        );
        worst_fast = worst_fast.max(d);
    }
    eprintln!(
        "{label}: {} pairs, max scaled difference Float(256) {worst_float:e}, binary64 {worst_fast:e}",
        pairs.len()
    );
}

/// The base case of every V7a pair, shifted by the manifest's `P` per lane.
fn shift_of(base: &BenchCase, pairs: &[Pair], name: &str) -> Vec<[i64; 4]> {
    (0..base.lanes.len())
        .map(|lane| {
            pairs
                .iter()
                .find(|p| p.case == name && p.lane == lane)
                .and_then(|p| p.shift)
                .unwrap_or([0; 4])
        })
        .collect()
}

/// The exact re-expansion `N(q - P)` reproduces the generator's shifted
/// cases (numerators and shifts) bitwise.
#[test]
fn exact_shift_reexpansion_matches_the_generator() {
    let pairs = pairs("V7a");
    let mut checked = 0;
    for name in pairs
        .iter()
        .map(|p| p.case.clone())
        .collect::<std::collections::BTreeSet<_>>()
    {
        let case = load(&name);
        let base = load(&pairs.iter().find(|p| p.case == name).unwrap().base);
        let ours = shifted(&base, &shift_of(&base, &pairs, &name));
        for (a, b) in ours.lanes.iter().zip(&case.lanes) {
            assert_eq!(a.shifts, b.shifts, "{name}");
            let bits = |t: &[C64]| {
                t.iter()
                    .map(|z| (z.re.to_bits(), z.im.to_bits()))
                    .collect::<Vec<_>>()
            };
            assert_eq!(bits(&a.tensor), bits(&b.tensor), "{name} {}", a.id);
            checked += 1;
        }
    }
    assert!(checked >= 24, "{checked} shifted lanes");
}

/// V7b on the rank n+1 truncations of the relabelling cases (n = 2..6).
#[test]
fn relabelling_invariance_below_rank_n2() {
    let pairs = pairs("V7b");
    assert!(pairs.len() >= 60);
    check_pairs(
        "V7b rank n+1",
        &pairs,
        |p| {
            let case = load(&p.case);
            truncated(&case, case.n + 1)
        },
        |p| {
            let case = load(&p.base);
            truncated(&case, case.n + 1)
        },
    );
}

/// V7a on the rank n+1 truncations of the shift cases: the truncated base
/// numerator, re-expanded exactly at `q - P`, on the shifted kinematics.
#[test]
fn shift_invariance_below_rank_n2() {
    let pairs = pairs("V7a");
    assert!(pairs.len() >= 20);
    check_pairs(
        "V7a rank n+1",
        &pairs,
        |p| {
            let base = load(&p.base);
            let base = truncated(&base, base.n + 1);
            shifted(&base, &shift_of(&base, &pairs, &p.case))
        },
        |p| {
            let base = load(&p.base);
            truncated(&base, base.n + 1)
        },
    );
}

/// V7c: per lane, a batch of all lanes, the lanes in reverse order, every
/// lane alone and a 37-lane cyclic batch agree bitwise.
fn check_batches(cases: &[BenchCase]) -> usize {
    let mut masters = OneLoopNativeMasters::new().unwrap();
    let bits =
        |r: &ninjars::fast::FastResult| fast_values(r).map(|z| (z.re.to_bits(), z.im.to_bits()));
    let mut checked = 0;
    for case in cases {
        let count = case.lanes.len();
        let all = run_fast(case, case.rank, count, &mut masters).unwrap();
        let mut reversed = case.clone();
        reversed.lanes.reverse();
        let back = run_fast(&reversed, case.rank, count, &mut masters).unwrap();
        let cyclic = run_fast(case, case.rank, 37.max(count), &mut masters).unwrap();
        for lane in 0..count {
            let mut single = case.clone();
            single.lanes = vec![case.lanes[lane].clone()];
            let alone = run_fast(&single, case.rank, 1, &mut masters).unwrap();
            let want = (all.statuses[lane], bits(&all.results[lane]));
            let label = format!("{} lane {lane}", case.name);
            assert_eq!(
                (alone.statuses[0], bits(&alone.results[0])),
                want,
                "{label}: alone"
            );
            let r = count - 1 - lane;
            assert_eq!(
                (back.statuses[r], bits(&back.results[r])),
                want,
                "{label}: reversed"
            );
            for (i, &index) in cyclic.lanes.iter().enumerate() {
                if index == lane {
                    assert_eq!(
                        (cyclic.statuses[i], bits(&cyclic.results[i])),
                        want,
                        "{label}: cyclic position {i}"
                    );
                }
            }
            checked += 1;
        }
    }
    checked
}

/// V7c on rank <= n+1: the controls (n = 1..7) and the truncated identity
/// cases.
#[test]
fn batch_width_and_lane_order_are_bitwise_below_rank_n2() {
    let mut cases: Vec<BenchCase> = controls(7)
        .iter()
        .map(|f| BenchCase::load(f).unwrap())
        .collect();
    cases.extend(
        cases_in(&rank_n2().join("identity/cases"), |name| {
            name.starts_with("rn2-exact-")
        })
        .iter()
        .map(|f| {
            let case = BenchCase::load(f).unwrap();
            truncated(&case, case.n + 1)
        }),
    );
    let checked = check_batches(&cases);
    eprintln!("V7c rank <= n+1: {checked} lanes bitwise");
}

/// V7a at rank n+2: the generator's shifted cases equal their base lanes.
#[test]
#[ignore = "enable at integration"]
fn shift_invariance_rank_n2() {
    check_pairs("V7a", &pairs("V7a"), |p| load(&p.case), |p| load(&p.base));
}

/// V7b at rank n+2.
#[test]
#[ignore = "enable at integration"]
fn relabelling_invariance_rank_n2() {
    check_pairs("V7b", &pairs("V7b"), |p| load(&p.case), |p| load(&p.base));
}

/// V7c at rank n+2: the exact cases and the reference suite.
#[test]
#[ignore = "enable at integration"]
fn batch_width_and_lane_order_are_bitwise_rank_n2() {
    let mut files = cases_in(&rank_n2().join("identity/cases"), |name| {
        name.starts_with("rn2-exact-")
    });
    files.extend(cases_in(&rank_n2().join("cases"), |_| true));
    let cases: Vec<BenchCase> = files.iter().map(|f| BenchCase::load(f).unwrap()).collect();
    let checked = check_batches(&cases);
    eprintln!("V7c rank n+2: {checked} lanes bitwise");
}
