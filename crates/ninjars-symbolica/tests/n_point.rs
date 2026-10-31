//! Both reduction engines at eight and nine propagators, on dense tensors.
//!
//! Neither engine has code sized by the number n of propagators: n enters
//! through the C(n, k) cut lists and the n - k uncut denominators of every
//! cut, and every fixed-size table is indexed by the cut size or by the rank
//! excess max(rank - n, 0). These tests check n = 8 and 9 (pinch identities
//! up to n = 10) on dense numerators, with every one of the
//! binomial(rank + 4, 4) tensor coefficients non-zero, and without an
//! external reference:
//!
//! - (a) pinch identities: the n-point numerator `D_{n-1}(q, mu²) T(q)`, with
//!   `D_{n-1}` the last denominator, reduces to the (n-1)-point reduction of
//!   `T` in every Laurent order (fast engine);
//! - (b) the fast engine against the exact reduction: the generic DoubleFloat
//!   engine on the same polynomial with the binary64 native master values;
//! - (c) the reversed propagator order (the same integral, other cut bases)
//!   in the fast engine;
//! - (d) DoubleFloat against Float(128), both with the binary64 master values,
//!   so that only the reduction is compared.
//!
//! Ranks up to n+1 are asserted. Rank n+2 is only reported (e), in a test
//! ignored by default (`--ignored`): some points are so ill-conditioned there
//! that binary64 keeps no digit, and the fast engine has no cancellation
//! test, so `FastStatus::Success` is no accuracy claim at rank n+2.
//!
//! The kinematics, masses and Gram screening are those of `any_n.rs`.
//! Without a Symbolica licence, run with `--test-threads=1`.

use ninjars::fast::tensor_layout::{monomial_index, monomials, ncoeff};
use ninjars::fast::{
    C64, FastAmplitude, FastInput, FastNumerator, FastOptions, FastResult, FastStatus,
    MuTensorNumerator, RMom, TensorNumerator,
};
use ninjars::{
    AmplitudeInput, Complex, DoubleFloat, EvaluationStatus, Float, FourMomentum, Laurent,
    MasterProvider, MasterRequest, MasterStatus, Real, SingleFloat,
};
use ninjars_symbolica::{OneLoopNativeMasters, PolynomialNumerator, SymbolicaAmplitudeKernels};
use ninjars_symbolica::{SymbolicaCuts, SymbolicaHighCuts};
use oneloop::NativeFloat;
use std::time::Instant;
use symbolica::{
    atom::Atom,
    domains::{float::RealLike, rational::Rational},
    evaluate::EvaluationDomain,
    parser::ParseSettings,
};

const NAMESPACE: &str = "ninjars_n_point";
const SEED: u64 = 0xa11_7e57;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform on [-1, 1], rounded to 2^-20: every input is exact in binary64
    /// and therefore identical in every carrier.
    fn uniform(&mut self) -> f64 {
        let x = (self.next() >> 11) as f64 / (1u64 << 53) as f64 * 2. - 1.;
        (x * 1048576.).round() / 1048576.
    }
    fn complex(&mut self) -> C64 {
        C64::new(self.uniform(), self.uniform())
    }
}

fn mp(a: &RMom, b: &RMom) -> f64 {
    a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
}

fn diff(a: &RMom, b: &RMom) -> RMom {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
}

/// Real cut bases need non-spacelike planes for every box/triangle pair; keep
/// the planes and invariants away from degenerate limits (as in `any_n.rs`).
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
    true
}

/// Energies in [-3, 3] and spatial components in [-1, 1], Gram-screened.
fn kinematics(rng: &mut Rng, n: usize) -> Vec<RMom> {
    loop {
        let v: Vec<RMom> = (0..n)
            .map(|i| {
                if i == 0 {
                    [0.; 4]
                } else {
                    [
                        3. * rng.uniform(),
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

#[derive(Clone)]
struct Lane {
    shifts: Vec<RMom>,
    masses: Vec<C64>,
}

/// Real masses, complex masses (same shifts) and mixed massless/massive
/// propagators (other shifts).
fn lanes(rng: &mut Rng, n: usize) -> Vec<Lane> {
    let v = kinematics(rng, n);
    let real = (0..n)
        .map(|_| C64::real(0.3 + rng.uniform().abs()))
        .collect();
    let complex = (0..n)
        .map(|_| C64::new(0.3 + rng.uniform().abs(), -0.2 * rng.uniform().abs()))
        .collect();
    let mixed = (0..n)
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
            shifts: v,
            masses: complex,
        },
        Lane {
            shifts: w,
            masses: mixed,
        },
    ]
}

fn shifts(lanes: &[Lane]) -> Vec<RMom> {
    lanes.iter().flat_map(|l| l.shifts.clone()).collect()
}

fn masses(lanes: &[Lane]) -> Vec<C64> {
    lanes.iter().flat_map(|l| l.masses.clone()).collect()
}

fn s_matrix(lanes: &[Lane]) -> Vec<f64> {
    let mut s = Vec::new();
    for lane in lanes {
        for a in &lane.shifts {
            for b in &lane.shifts {
                let d = diff(a, b);
                s.push(mp(&d, &d));
            }
        }
    }
    s
}

fn scales(lanes: &[Lane]) -> Vec<f64> {
    (0..lanes.len()).map(|i| [1.0, 2.5, 0.7][i % 3]).collect()
}

/// Ninja's dimensional rule for the auxiliary box mu² sampling scale.
fn sampling(lanes: &[Lane]) -> Vec<f64> {
    let s = s_matrix(lanes);
    let mut offset = 0;
    lanes
        .iter()
        .map(|lane| {
            let block = &s[offset..offset + lane.shifts.len().pow(2)];
            offset += block.len();
            block
                .iter()
                .map(|x| x.abs())
                .chain(lane.masses.iter().map(|m| m.re.abs()))
                .fold(1., f64::max)
        })
        .collect()
}

fn fast_values(f: &FastResult) -> [Complex<f64>; 5] {
    [
        f.laurent[0],
        f.laurent[1],
        f.laurent[2],
        f.cut_constructible,
        f.rational,
    ]
    .map(C64::to_symbolica)
}

/// The binary64 fast engine with oneloopmaster native masters.
fn fast<N: FastNumerator>(
    n: usize,
    rank: usize,
    lanes: &[Lane],
    numerator: &mut N,
) -> (Vec<[Complex<f64>; 5]>, Vec<FastStatus>) {
    let count = lanes.len();
    let (shifts, masses, matrix, scales) =
        (shifts(lanes), masses(lanes), s_matrix(lanes), scales(lanes));
    let input = FastInput {
        lanes: count,
        shifts: &shifts,
        masses_squared: &masses,
        scales_squared: &scales,
        s_matrix: Some(&matrix),
    };
    let mut engine = FastAmplitude::new(n, rank, FastOptions::default()).unwrap();
    let mut masters = OneLoopNativeMasters::new().unwrap();
    let mut results = vec![FastResult::default(); count];
    let mut statuses = vec![FastStatus::NotEvaluated; count];
    engine
        .evaluate(&input, numerator, &mut masters, &mut results, &mut statuses)
        .unwrap();
    (results.iter().map(fast_values).collect(), statuses)
}

fn exact<T: SingleFloat>(prototype: &T, x: f64) -> T {
    prototype.from_rational(&Rational::try_from(x).unwrap())
}

fn complex<T: SingleFloat>(prototype: &T, z: C64) -> Complex<T> {
    Complex::new(exact(prototype, z.re), exact(prototype, z.im))
}

fn to_c64<T: RealLike>(z: &Complex<T>) -> C64 {
    C64::new(z.re.to_f64(), z.im.to_f64())
}

/// Exact widening to a common 512-bit Float.
fn wide<T: Clone + Into<Float>>(x: &T) -> Float {
    let mut x: Float = x.clone().into();
    x.set_prec(512);
    x
}

/// Modulus of a complex difference, computed exactly before rounding.
fn distance<T: Clone + Into<Float>, U: Clone + Into<Float>>(a: &Complex<T>, b: &Complex<U>) -> f64 {
    let re = wide(&a.re) - &wide(&b.re);
    let im = wide(&a.im) - &wide(&b.im);
    re.to_f64().hypot(im.to_f64())
}

/// Largest Laurent difference relative to the reference's largest Laurent
/// modulus in the same lane.
fn laurent_error<T, U>(a: &[Complex<T>; 5], reference: &[Complex<U>; 5]) -> f64
where
    T: Clone + Into<Float>,
    U: Clone + Into<Float> + RealLike,
{
    let error = (0..3)
        .map(|k| distance(&a[k], &reference[k]))
        .fold(0., f64::max);
    let scale = (0..3)
        .map(|k| to_c64(&reference[k]).abs())
        .fold(f64::MIN_POSITIVE, f64::max);
    error / scale
}

/// `((q + v)² - m²) T(q)` as a dense rank-(r+2) tensor, for a dense rank-r `T`.
fn times_denominator(t: &[C64], r: usize, v: &RMom, m2: C64) -> Vec<C64> {
    let metric = [1., -1., -1., -1.];
    let mut quadric = vec![([0; 4], C64::real(mp(v, v)) - m2)];
    for mu in 0..4 {
        let mut e = [0; 4];
        e[mu] = 2;
        quadric.push((e, C64::real(metric[mu])));
        e[mu] = 1;
        quadric.push((e, C64::real(2. * metric[mu] * v[mu])));
    }
    let mut out = vec![C64::ZERO; ncoeff(r + 2)];
    for (a, &c) in monomials(r).iter().zip(t) {
        for (b, d) in &quadric {
            let i = monomial_index([a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]);
            out[i] += c * *d;
        }
    }
    out
}

/// (a) The rank-`rank` n-point numerator `D_{n-1} T`, with
/// `D_{n-1} = (q + V_{n-1})² - m²_{n-1} - mu²` (a `MuTensorNumerator` with
/// one mu² block), against the (n-1)-point reduction of the dense rank-(rank-2)
/// `T`. Returns the statuses and errors of the three lanes.
fn pinch(n: usize, rank: usize) -> Vec<(FastStatus, FastStatus, f64)> {
    let r = rank - 2;
    let mut rng = Rng((SEED + 11) ^ ((n as u64) << 32) ^ ((r as u64) << 40));
    let full = lanes(&mut rng, n);
    let count = full.len();
    let (mut big, mut small) = (Vec::new(), Vec::new());
    for lane in &full {
        let t: Vec<C64> = (0..ncoeff(r)).map(|_| rng.complex()).collect();
        big.extend(times_denominator(
            &t,
            r,
            &lane.shifts[n - 1],
            lane.masses[n - 1],
        ));
        big.extend(t.iter().map(|&c| -c));
        small.extend(t);
    }
    let mut numerator = MuTensorNumerator::new(rank, 1, count).unwrap();
    numerator.set_coefficients(count, &big).unwrap();
    let mut pinched_numerator = TensorNumerator::new(r, count);
    pinched_numerator.set_coefficients(count, &small).unwrap();
    let pinched: Vec<Lane> = full
        .iter()
        .map(|l| Lane {
            shifts: l.shifts[..n - 1].to_vec(),
            masses: l.masses[..n - 1].to_vec(),
        })
        .collect();
    let (values, statuses) = fast(n, rank, &full, &mut numerator);
    let (reference, reference_statuses) = fast(n - 1, r, &pinched, &mut pinched_numerator);
    (0..count)
        .map(|lane| {
            let error = laurent_error(&values[lane], &reference[lane]);
            eprintln!(
                "N_POINT_PINCH n={n} rank={rank} lane={lane} statuses={:?}/{:?} \
                 vs_pinched={error:.3e}",
                statuses[lane], reference_statuses[lane]
            );
            (statuses[lane], reference_statuses[lane], error)
        })
        .collect()
}

/// Asserted up to rank n+1, where the largest error is 4.5e-8 (n = 9, rank
/// 10). Rank n+2 is printed only (up to 2.9e-5 at n = 8).
#[test]
fn pinch_identities_n8_to_n10() {
    for n in [8, 9, 10] {
        for rank in [2, n, n + 1, n + 2] {
            for (status, reference_status, error) in pinch(n, rank) {
                if rank <= n + 1 {
                    assert_eq!(status, FastStatus::Success, "n={n} rank={rank}");
                    assert_eq!(reference_status, FastStatus::Success, "n={n} rank={rank}");
                    assert!(error <= 1e-6, "n={n} rank={rank}: {error:e}");
                }
            }
        }
    }
}

/// A dense random rank-`rank` tensor on three lanes of n propagators.
struct Problem {
    n: usize,
    rank: usize,
    lanes: Vec<Lane>,
    /// Lane-major, `ncoeff(rank)` per lane.
    coefficients: Vec<C64>,
}

impl Problem {
    fn new(n: usize, rank: usize) -> Self {
        let mut rng = Rng(SEED ^ 0xde75e ^ ((n as u64) << 8) ^ rank as u64);
        let lanes = lanes(&mut rng, n);
        let coefficients = (0..lanes.len() * ncoeff(rank))
            .map(|_| rng.complex())
            .collect();
        Self {
            n,
            rank,
            lanes,
            coefficients,
        }
    }

    /// The same integrals with the propagators in reversed order.
    fn reversed(&self) -> Vec<Lane> {
        self.lanes
            .iter()
            .map(|l| Lane {
                shifts: l.shifts.iter().rev().copied().collect(),
                masses: l.masses.iter().rev().copied().collect(),
            })
            .collect()
    }

    fn fast(&self, lanes: &[Lane]) -> (Vec<[Complex<f64>; 5]>, Vec<FastStatus>) {
        let mut numerator = TensorNumerator::new(self.rank, lanes.len());
        numerator
            .set_coefficients(lanes.len(), &self.coefficients)
            .unwrap();
        fast(self.n, self.rank, lanes, &mut numerator)
    }

    /// The same tensor as a Symbolica polynomial, one parameter per
    /// coefficient (no mu² terms), in the generic engine in any carrier.
    fn generic<T, M>(&self, prototype: T, masters: &mut M) -> Vec<[Complex<T>; 5]>
    where
        T: NativeFloat,
        Complex<T>: Real + EvaluationDomain,
        M: MasterProvider<Complex<T>>,
    {
        let parse = |text: &str| Atom::parse(text, NAMESPACE, ParseSettings::default()).unwrap();
        let names = ["q0", "q1", "q2", "q3"];
        let mut text = String::from("0");
        let mut parameters = Vec::new();
        for (k, m) in monomials(self.rank).iter().enumerate() {
            text.push_str(&format!("+c{k}"));
            for (mu, &e) in m.iter().enumerate() {
                if e > 0 {
                    text.push_str(&format!("*{}^{e}", names[mu]));
                }
            }
            parameters.push(parse(&format!("c{k}")));
        }
        let p = &prototype;
        let mut numerator = PolynomialNumerator::new(
            parse(&text),
            ["q0", "q1", "q2", "q3", "mu2"].map(parse),
            parameters,
            p.clone(),
        )
        .unwrap();
        let count = self.lanes.len();
        let values: Vec<_> = self.coefficients.iter().map(|&z| complex(p, z)).collect();
        numerator.set_parameter_batch(&values, count).unwrap();
        let mut amplitude = SymbolicaAmplitudeKernels::new(self.n, self.rank, p.clone())
            .unwrap()
            .into_amplitude(count)
            .unwrap();
        let real = |x: f64| complex(p, C64::real(x));
        let shifts: Vec<FourMomentum<Complex<T>>> = shifts(&self.lanes)
            .iter()
            .map(|v| FourMomentum::from(v.map(real)))
            .collect();
        let masses: Vec<_> = masses(&self.lanes).iter().map(|&m| complex(p, m)).collect();
        let matrix: Vec<_> = s_matrix(&self.lanes).into_iter().map(real).collect();
        let scales: Vec<_> = scales(&self.lanes).into_iter().map(real).collect();
        let sampling: Vec<_> = sampling(&self.lanes).into_iter().map(real).collect();
        let mut low = SymbolicaCuts::new(p.clone()).unwrap();
        let mut high = SymbolicaHighCuts::new(p.clone()).unwrap();
        amplitude
            .evaluate_batch_with_inputs(
                AmplitudeInput {
                    shifts: &shifts,
                    masses_squared: &masses,
                    scales_squared: &scales,
                    sampling_scales: &sampling,
                    s_matrix: Some(&matrix),
                },
                &mut numerator,
                &mut low,
                &mut high,
                masters,
            )
            .unwrap();
        for (lane, status) in amplitude.statuses().iter().enumerate() {
            assert_eq!(
                *status,
                EvaluationStatus::Success,
                "n={} rank={} lane={lane} bits={}",
                self.n,
                self.rank,
                p.get_precision()
            );
        }
        amplitude.results().to_vec()
    }
}

/// The binary64 native master values, widened exactly to the carrier of the
/// prototype. A reduction with these is the exact reduction of the binary64
/// inputs with exactly the master values of the binary64 engines.
struct Binary64Masters<T>(OneLoopNativeMasters, T);

impl<T: SingleFloat + RealLike> MasterProvider<Complex<T>> for Binary64Masters<T> {
    fn evaluate(
        &mut self,
        request: MasterRequest<'_, Complex<T>>,
        output: &mut [Laurent<Complex<T>>],
    ) -> ninjars::Result<()> {
        let mut statuses = vec![MasterStatus::Success; request.rows()];
        self.evaluate_batch(request, output, &mut statuses)
    }

    fn evaluate_batch(
        &mut self,
        request: MasterRequest<'_, Complex<T>>,
        output: &mut [Laurent<Complex<T>>],
        statuses: &mut [MasterStatus],
    ) -> ninjars::Result<()> {
        // Every argument is an exact binary64 value (exact inputs and S-matrix).
        let round = |values: &[Complex<T>]| -> Vec<Complex<f64>> {
            values
                .iter()
                .map(|z| Complex::new(z.re.to_f64(), z.im.to_f64()))
                .collect()
        };
        let (invariants, masses, scales) = (
            round(request.invariants),
            round(request.masses_squared),
            round(request.scales_squared),
        );
        let mut values = vec![[Complex::new(0., 0.); 3]; request.rows()];
        self.0.evaluate_batch(
            MasterRequest {
                integral: request.integral,
                invariants: &invariants,
                masses_squared: &masses,
                scales_squared: &scales,
            },
            &mut values,
            statuses,
        )?;
        for ((out, value), status) in output.iter_mut().zip(&values).zip(statuses.iter()) {
            if *status == MasterStatus::Success {
                *out = value.map(|z| complex(&self.1, C64::new(z.re, z.im)));
            }
        }
        Ok(())
    }
}

/// Per-lane errors of one dense problem.
struct Errors {
    statuses: (FastStatus, FastStatus),
    /// (b) fast against the exact (DoubleFloat) reduction.
    fast: f64,
    /// (c) fast with reversed propagators against the exact reduction.
    reversed: f64,
    /// (c) fast against fast with reversed propagators.
    fast_vs_reversed: f64,
    /// (d) Float(128) against DoubleFloat.
    float128_vs_double: f64,
}

/// (b)-(d) on the dense rank-`rank` tensor at n propagators.
fn dense(n: usize, rank: usize) -> Vec<Errors> {
    let problem = Problem::new(n, rank);
    let (values, statuses) = problem.fast(&problem.lanes);
    let (reversed, reversed_statuses) = problem.fast(&problem.reversed());
    let start = Instant::now();
    let dd = DoubleFloat::from(0.);
    let exact = problem.generic(
        dd,
        &mut Binary64Masters(OneLoopNativeMasters::new().unwrap(), dd),
    );
    let dd_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let f128 = Float::new(128);
    let float = problem.generic(
        f128.clone(),
        &mut Binary64Masters(OneLoopNativeMasters::new().unwrap(), f128),
    );
    let float_seconds = start.elapsed().as_secs_f64();
    (0..problem.lanes.len())
        .map(|lane| {
            let errors = Errors {
                statuses: (statuses[lane], reversed_statuses[lane]),
                fast: laurent_error(&values[lane], &exact[lane]),
                reversed: laurent_error(&reversed[lane], &exact[lane]),
                fast_vs_reversed: laurent_error(&values[lane], &reversed[lane]),
                float128_vs_double: laurent_error(&exact[lane], &float[lane]),
            };
            eprintln!(
                "N_POINT_DENSE n={n} rank={rank} ncoeff={} lane={lane} statuses={:?}/{:?} \
                 fast_vs_exact={:.3e} reversed_vs_exact={:.3e} fast_vs_reversed={:.3e} \
                 double_vs_float128={:.3e} double_seconds={dd_seconds:.2} \
                 float128_seconds={float_seconds:.2}",
                ncoeff(rank),
                errors.statuses.0,
                errors.statuses.1,
                errors.fast,
                errors.reversed,
                errors.fast_vs_reversed,
                errors.float128_vs_double
            );
            errors
        })
        .collect()
}

/// (b)-(d) at ranks n and n+1. Binary64 loses digits with the rank and the
/// number of uncut lines: on these points the largest errors are 1.0e-7 at
/// rank n and 3.3e-6 at rank n+1 (n = 8 and n = 9 respectively, reversed
/// order), three to ten times below the bounds. DoubleFloat agrees with Float(128) to about 1e-24.
fn dense_up_to_n_plus_1(n: usize) {
    for rank in [n, n + 1] {
        let bound = if rank == n { 1e-6 } else { 1e-5 };
        for (lane, e) in dense(n, rank).iter().enumerate() {
            let at = format!("n={n} rank={rank} lane={lane}");
            assert_eq!(e.statuses.0, FastStatus::Success, "{at}");
            assert_eq!(e.statuses.1, FastStatus::Success, "{at} reversed");
            let checks = [
                ("fast", e.fast, bound),
                ("reversed", e.reversed, bound),
                ("fast_vs_reversed", e.fast_vs_reversed, bound),
                ("float128_vs_double", e.float128_vs_double, 1e-20),
            ];
            for (name, value, bound) in checks {
                assert!(value <= bound, "{at}: {name} {value:e}");
            }
        }
    }
}

#[test]
fn dense_tensors_n8() {
    dense_up_to_n_plus_1(8);
}

#[test]
fn dense_tensors_n9() {
    dense_up_to_n_plus_1(9);
}

/// (e) Rank n+2, reported only. Some points are ill-conditioned: at n = 8,
/// two lanes (one phase-space point, two mass sets) of the fast engine are off
/// by 5e2 and 5e1 times the amplitude while reporting `FastStatus::Success`,
/// and DoubleFloat keeps only about 14 digits there (Float(128) against
/// DoubleFloat). The reversed propagator order flags those lanes (relative
/// difference about 2.5). Only the generic engine's
/// `EvaluationStatus::Success` is asserted.
#[test]
#[ignore = "report only; about three minutes, mostly Float(128)"]
fn dense_tensors_rank_n_plus_2_report() {
    for n in [8, 9] {
        dense(n, n + 2);
    }
}
