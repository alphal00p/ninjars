//! Parity of the binary64 fast engine with the validated generic engine.
//!
//! Both engines receive identical kinematics, an identical exact S-matrix and
//! master values from the same oneloopmaster native backend
//! (`OneLoopNativeMasters`, ahead-of-time generated Rust), so differences come
//! from the reductions alone. Numerators are random dense polynomials with
//! per-lane complex coefficients: with explicit `mu²` terms through
//! `GenericNumerator(PolynomialNumerator)`, and q-only through
//! `TensorNumerator`. Run with `--features native-masters --test-threads=1`.

use ninjars::fast::{
    C64, FastAmplitude, FastInput, FastOptions, FastResult, FastStatus, GenericNumerator, RMom,
    TensorNumerator,
    tensor_layout::{monomials, ncoeff},
};
use ninjars::{AmplitudeInput, Complex, EvaluationStatus, FourMomentum, Numerator};
use ninjars_symbolica::{
    OneLoopNativeMasters, PolynomialNumerator, SymbolicaAmplitudeKernels, SymbolicaCuts,
    SymbolicaHighCuts,
};
use symbolica::{atom::Atom, parser::ParseSettings};

const NAMESPACE: &str = "ninjars_fast_parity";

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64 * 2. - 1.
    }
}

fn parse(text: &str) -> Atom {
    Atom::parse(text, NAMESPACE, ParseSettings::default()).unwrap()
}

fn mp(a: &RMom, b: &RMom) -> f64 {
    a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
}

fn diff(a: &RMom, b: &RMom) -> RMom {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
}

/// Real cut bases need non-spacelike planes for every box/triangle pair; the
/// planes and invariants are kept well away from degenerate limits so that
/// both engines stay well conditioned in binary64.
fn planes_ok(v: &[RMom]) -> bool {
    let n = v.len();
    let ok = |k1: RMom, k2: RMom| {
        let k12 = mp(&k1, &k2);
        let product = mp(&k1, &k1) * mp(&k2, &k2);
        k12 * k12 - product > 0.1 * (k12 * k12).max(product.abs()).max(0.01)
    };
    for i in 0..n {
        for j in i + 1..n {
            let d = diff(&v[i], &v[j]);
            if mp(&d, &d).abs() < 0.05 {
                return false;
            }
        }
    }
    for a in 0..n {
        for b in a + 1..n {
            for c in b + 1..n {
                if !ok(diff(&v[a], &v[c]), diff(&v[b], &v[a])) {
                    return false;
                }
                for d in c + 1..n {
                    if !ok(diff(&v[a], &v[d]), diff(&v[b], &v[a])) {
                        return false;
                    }
                }
            }
        }
    }
    // Mixed timelike and spacelike invariants.
    let s: Vec<f64> = (0..n)
        .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
        .map(|(i, j)| mp(&diff(&v[i], &v[j]), &diff(&v[i], &v[j])))
        .collect();
    n < 3 || (s.iter().any(|&x| x > 0.1) && s.iter().any(|&x| x < -0.1))
}

fn kinematics(rng: &mut Rng, n: usize) -> Vec<RMom> {
    loop {
        let v: Vec<RMom> = (0..n)
            .map(|i| {
                if i == 0 {
                    [0.; 4]
                } else {
                    [
                        2. * rng.uniform(),
                        rng.uniform(),
                        rng.uniform(),
                        rng.uniform(),
                    ]
                }
            })
            .collect();
        if planes_ok(&v) {
            return v;
        }
    }
}

struct Lane {
    shifts: Vec<RMom>,
    masses: Vec<C64>,
}

/// Lanes: real masses, complex masses, a permutation of the complex lane
/// (identical integral), and mixed massless/massive propagators.
fn lanes(rng: &mut Rng, n: usize) -> Vec<Lane> {
    let v = kinematics(rng, n);
    let real: Vec<C64> = (0..n)
        .map(|_| C64::real(0.3 + rng.uniform().abs()))
        .collect();
    let complex: Vec<C64> = (0..n)
        .map(|_| C64::new(0.3 + rng.uniform().abs(), -0.2 * rng.uniform().abs()))
        .collect();
    let permutation: Vec<usize> = (0..n).map(|i| (i * 3 + 1) % n).collect::<Vec<_>>();
    let permutation = if n > 1 && {
        let mut sorted = permutation.clone();
        sorted.sort();
        sorted == (0..n).collect::<Vec<_>>()
    } {
        permutation
    } else {
        (0..n).rev().collect()
    };
    let mixed: Vec<C64> = (0..n)
        .map(|i| {
            if i % 2 == 0 {
                C64::ZERO
            } else {
                C64::real(0.5 + rng.uniform().abs())
            }
        })
        .collect();
    let w = kinematics(rng, n);
    vec![
        Lane {
            shifts: v.clone(),
            masses: real,
        },
        Lane {
            shifts: v.clone(),
            masses: complex.clone(),
        },
        Lane {
            shifts: permutation.iter().map(|&i| v[i]).collect(),
            masses: permutation.iter().map(|&i| complex[i]).collect(),
        },
        Lane {
            shifts: w,
            masses: mixed,
        },
    ]
}

/// Monomials `(q exponents, mu² power)` of a rank-`rank` numerator: the
/// q-only tensor block first (tensor_layout order), then mu² terms.
fn numerator_terms(rank: usize) -> Vec<([u32; 4], u32)> {
    let mut terms: Vec<_> = monomials(rank).into_iter().map(|m| (m, 0)).collect();
    for j in 1..=rank as u32 / 2 {
        terms.extend(monomials(rank - 2 * j as usize).into_iter().map(|m| (m, j)));
    }
    terms
}

/// Dense numerators through rank 5; above, a fixed pseudo-random quarter of
/// the monomials (always including the leading `q0^rank`) keeps Symbolica's
/// symbolic preparation of the generic engine affordable.
fn active(rank: usize, k: usize) -> bool {
    rank <= 5 || k == ncoeff(rank - 1) || (k.wrapping_mul(2_654_435_761) >> 7) % 4 == 0
}

fn polynomial(rank: usize) -> (Atom, Vec<Atom>) {
    let names = ["q0", "q1", "q2", "q3"];
    let terms = numerator_terms(rank);
    let mut text = String::new();
    for (k, (m, j)) in terms.iter().enumerate() {
        if !active(rank, k) {
            continue;
        }
        if !text.is_empty() {
            text.push('+');
        }
        text.push_str(&format!("p{k}"));
        for (mu, &e) in m.iter().enumerate() {
            if e > 0 {
                text.push_str(&format!("*{}^{e}", names[mu]));
            }
        }
        if *j > 0 {
            text.push_str(&format!("*mu2^{j}"));
        }
    }
    let parameters = (0..terms.len()).map(|k| parse(&format!("p{k}"))).collect();
    (parse(&text), parameters)
}

fn s_matrix(lane: &Lane) -> Vec<f64> {
    let n = lane.shifts.len();
    let mut s = vec![0.; n * n];
    for i in 0..n {
        for j in 0..n {
            let d = diff(&lane.shifts[i], &lane.shifts[j]);
            s[i * n + j] = mp(&d, &d);
        }
    }
    s
}

fn scale(values: &[C64]) -> f64 {
    1. + values[..3].iter().map(|z| z.abs()).fold(0., f64::max)
}

fn generic_values(generic: &[Complex<f64>; 5]) -> [C64; 5] {
    generic.map(C64::from_symbolica)
}

fn fast_values(fast: &FastResult) -> [C64; 5] {
    [
        fast.laurent[0],
        fast.laurent[1],
        fast.laurent[2],
        fast.cut_constructible,
        fast.rational,
    ]
}

/// Largest scaled difference of the three Laurent coefficients.
fn laurent_difference(a: &[C64; 5], b: &[C64; 5]) -> f64 {
    (0..3).map(|k| (a[k] - b[k]).abs()).fold(0., f64::max) / scale(b)
}

#[derive(Default)]
struct Summary {
    laurent: f64,
    split: f64,
    noise: f64,
    within_noise: usize,
    checks: usize,
}

/// Scaled 1e-9 parity of the three Laurent coefficients. At numerically
/// ill-conditioned points, where both engines lose digits, the difference
/// may instead stay within ten times `noise`: the larger change of either
/// engine's result when the propagators of the same point are listed in
/// reverse order (an identical integral). The cut-constructible/rational
/// split is only recorded: the fast engine reproduces original Ninja's split
/// (validation/fast/crosscheck_ninja.py), while the generic engine
/// distributes some rank >= n bubble terms differently (identical totals).
fn check(label: &str, fast: &[C64; 5], generic: &[C64; 5], noise: f64, summary: &mut Summary) {
    let error = laurent_difference(fast, generic);
    summary.laurent = summary.laurent.max(error);
    summary.noise = summary.noise.max(noise);
    summary.checks += 1;
    if error > 1e-9 {
        assert!(
            error <= 10. * noise,
            "{label}: fast {:?} generic {:?} (scaled error {error:e}, noise {noise:e})",
            &fast[..3],
            &generic[..3]
        );
        summary.within_noise += 1;
        eprintln!("FAST_PARITY_NOISE {label}: scaled error {error:e} within 10x noise {noise:e}");
    }
    summary.split = summary
        .split
        .max((fast[4] - generic[4]).abs() / scale(generic));
    let consistency = (fast[3] + fast[4] - fast[0]).abs() / scale(fast);
    assert!(
        consistency <= 1e-14,
        "{label}: finite part is not cut-constructible + rational"
    );
}

fn parity(n: usize) {
    let mut rng = Rng(0x5eed + n as u64);
    let mut masters = OneLoopNativeMasters::new().unwrap();
    let mut fast_masters = OneLoopNativeMasters::new().unwrap();
    let mut low = SymbolicaCuts::new(0.).unwrap();
    let mut high = SymbolicaHighCuts::new(0.).unwrap();
    let variables = ["q0", "q1", "q2", "q3", "mu2"].map(parse);
    let mut summary = Summary::default();
    for rank in 0..=n + 1 {
        // Lanes 4..8 repeat lanes 0..4 with reversed propagator order.
        let mut lanes = lanes(&mut rng, n);
        let reversed: Vec<Lane> = lanes
            .iter()
            .map(|lane| Lane {
                shifts: lane.shifts.iter().rev().copied().collect(),
                masses: lane.masses.iter().rev().copied().collect(),
            })
            .collect();
        lanes.extend(reversed);
        let count = lanes.len();
        let (expression, parameters) = polynomial(rank);
        let terms = parameters.len();
        let mut numerator =
            PolynomialNumerator::new(expression, variables.clone(), parameters, 0.).unwrap();
        let mut amplitude = SymbolicaAmplitudeKernels::new(n, rank, 0.)
            .unwrap()
            .into_amplitude(count)
            .unwrap();
        let mut fast = FastAmplitude::new(n, rank, FastOptions::default()).unwrap();
        let shifts: Vec<RMom> = lanes.iter().flat_map(|l| l.shifts.clone()).collect();
        let masses: Vec<C64> = lanes.iter().flat_map(|l| l.masses.clone()).collect();
        let matrix: Vec<f64> = lanes.iter().flat_map(s_matrix).collect();
        let scales: Vec<f64> = (0..count).map(|i| [1.0, 2.5, 2.5, 0.7][i % 4]).collect();
        let to_c = |x: f64| Complex::new(x, 0.);
        let generic_shifts: Vec<FourMomentum<Complex<f64>>> = shifts
            .iter()
            .map(|v| FourMomentum::from(v.map(to_c)))
            .collect();
        let generic_masses: Vec<Complex<f64>> = masses.iter().map(|m| m.to_symbolica()).collect();
        let generic_matrix: Vec<Complex<f64>> = matrix.iter().map(|&x| to_c(x)).collect();
        let generic_scales: Vec<Complex<f64>> = scales.iter().map(|&x| to_c(x)).collect();
        let sampling: Vec<Complex<f64>> = (0..count)
            .map(|lane| {
                let s = &matrix[lane * n * n..(lane + 1) * n * n];
                let m = &masses[lane * n..(lane + 1) * n];
                to_c(
                    s.iter()
                        .map(|x| x.abs())
                        .chain(m.iter().map(|z| z.re.abs()))
                        .fold(1., f64::max),
                )
            })
            .collect();
        for with_mu in [true, false] {
            // Lanes 1 and 2 share coefficients (lane 2 permutes propagators),
            // and lanes 4..8 those of lanes 0..4.
            let mut values = vec![C64::ZERO; count * terms];
            for lane in 0..count {
                for k in 0..terms {
                    values[lane * terms + k] = if lane >= 4 {
                        values[(lane - 4) * terms + k]
                    } else if lane == 2 {
                        values[terms + k]
                    } else if !active(rank, k) || (k >= ncoeff(rank) && !with_mu) {
                        C64::ZERO
                    } else {
                        C64::new(rng.uniform(), rng.uniform())
                    };
                }
            }
            let symbolica_values: Vec<Complex<f64>> =
                values.iter().map(|z| z.to_symbolica()).collect();
            numerator
                .set_parameter_batch(&symbolica_values, count)
                .unwrap();
            amplitude.reset_batch(count);
            amplitude
                .evaluate_batch_with_inputs(
                    AmplitudeInput {
                        shifts: &generic_shifts,
                        masses_squared: &generic_masses,
                        scales_squared: &generic_scales,
                        sampling_scales: &sampling,
                        s_matrix: Some(&generic_matrix),
                    },
                    &mut numerator,
                    &mut low,
                    &mut high,
                    &mut masters,
                )
                .unwrap();
            let input = FastInput {
                lanes: count,
                shifts: &shifts,
                masses_squared: &masses,
                scales_squared: &scales,
                s_matrix: Some(&matrix),
            };
            let mut results = vec![FastResult::default(); count];
            let mut statuses = vec![FastStatus::NotEvaluated; count];
            if with_mu {
                let mut generic = GenericNumerator::with_rank(numerator, rank);
                fast.evaluate(
                    &input,
                    &mut generic,
                    &mut fast_masters,
                    &mut results,
                    &mut statuses,
                )
                .unwrap();
                numerator = generic.into_inner();
            } else {
                let mut tensor = TensorNumerator::new(rank, count);
                for lane in 0..count {
                    tensor
                        .lane_coefficients_mut(lane)
                        .copy_from_slice(&values[lane * terms..lane * terms + ncoeff(rank)]);
                }
                fast.evaluate(
                    &input,
                    &mut tensor,
                    &mut fast_masters,
                    &mut results,
                    &mut statuses,
                )
                .unwrap();
            }
            let fast: Vec<[C64; 5]> = results.iter().map(fast_values).collect();
            let generic: Vec<[C64; 5]> = amplitude.results().iter().map(generic_values).collect();
            for lane in 0..count {
                let label = format!("n={n} rank={rank} mu={with_mu} lane={lane}");
                assert_eq!(
                    amplitude.statuses()[lane],
                    EvaluationStatus::Success,
                    "{label}"
                );
                assert_eq!(statuses[lane], FastStatus::Success, "{label}");
            }
            let noise: Vec<f64> = (0..4)
                .map(|lane| {
                    laurent_difference(&fast[lane], &fast[lane + 4])
                        .max(laurent_difference(&generic[lane], &generic[lane + 4]))
                })
                .collect();
            for lane in 0..count {
                let label = format!("n={n} rank={rank} mu={with_mu} lane={lane}");
                check(
                    &label,
                    &fast[lane],
                    &generic[lane],
                    noise[lane % 4],
                    &mut summary,
                );
            }
            // The permuted lane computes the same integral.
            check(
                &format!("n={n} rank={rank} mu={with_mu} permutation"),
                &fast[2],
                &generic[1],
                noise[1].max(noise[2]),
                &mut summary,
            );
        }
        let _ = numerator.rank();
    }
    eprintln!(
        "FAST_PARITY n={n} checks={} max_scaled_laurent_error={:e} \
         checks_above_1e-9_within_noise={} max_noise={:e} max_rational_split_difference={:e}",
        summary.checks, summary.laurent, summary.within_noise, summary.noise, summary.split
    );
}

#[test]
fn parity_n1() {
    parity(1);
}
#[test]
fn parity_n2() {
    parity(2);
}
#[test]
fn parity_n3() {
    parity(3);
}
#[test]
fn parity_n4() {
    parity(4);
}
#[test]
fn parity_n5() {
    parity(5);
}
#[test]
fn parity_n6() {
    parity(6);
}
