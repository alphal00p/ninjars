//! The generic engine beyond six propagators, in every supported precision.
//!
//! Five-cuts are the highest cuts at any n; seven or more propagators only add
//! uncut denominators to every pentagon, box, triangle, bubble and tadpole.
//! Random polynomial numerators of rank n-1, n and n+1 with explicit mu² terms
//! are reduced on random real kinematics with real, complex and mixed
//! massless/massive squared masses:
//!
//! - (a) generic binary64 against `ninjars::fast` with identical oneloopmaster
//!   native masters and S-matrix; the DoubleFloat reduction with the same
//!   binary64 master values is the exact reference;
//! - (b) DoubleFloat against arbitrary-precision Float;
//! - (c) Float at two working precisions;
//! - (d) Dual-number derivatives (binary64 and Float carriers) against finite
//!   differences of Float evaluations;
//! - n = 7 against original Ninja with its recorded OneLOop masters, and a
//!   ten-propagator smoke test.
//!
//! (b)-(d) use smooth polynomial stand-in masters, generic over every carrier,
//! so that they measure the reduction alone: Dual C0/D0 masters are not
//! available, and oneloopmaster's DoubleFloat and interpreted Float D0 can
//! select a different branch than OneLOop for some real-mass boxes, at any n.

mod common;

use common::ninja_reference::{parse_case, parse_ninja};
use common::precision::precision;
use ninjars::fast::{
    C64, FastAmplitude, FastInput, FastOptions, FastResult, FastStatus, GenericNumerator, RMom,
    tensor_layout::monomials,
};
use ninjars::{
    AmplitudeInput, Complex, DoubleFloat, EvaluationStatus, Float, FloatLike, FourMomentum,
    Laurent, MasterProvider, MasterRequest, MasterStatus, Real, SingleFloat,
};
use ninjars_symbolica::{
    DualPolynomialNumerator, OneLoopNativeMasters, PolynomialNumerator, SymbolicaAmplitudeKernels,
    SymbolicaCuts, SymbolicaDualAmplitudeKernels, SymbolicaDualCuts, SymbolicaDualHighCuts,
    SymbolicaHighCuts,
};
use oneloop::NativeFloat;
use std::path::PathBuf;
use symbolica::{
    atom::Atom,
    domains::{dual::HyperDual, float::RealLike, rational::Rational},
    evaluate::EvaluationDomain,
    parser::ParseSettings,
};

const NAMESPACE: &str = "ninjars_any_n";
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
}

fn parse(text: &str) -> Atom {
    Atom::parse(text, NAMESPACE, ParseSettings::default()).unwrap()
}

fn variables() -> [Atom; 5] {
    ["q0", "q1", "q2", "q3", "mu2"].map(parse)
}

fn mp(a: &RMom, b: &RMom) -> f64 {
    a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
}

fn diff(a: &RMom, b: &RMom) -> RMom {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
}

/// Real cut bases need non-spacelike planes for every box/triangle pair; keep
/// the planes and invariants away from degenerate limits (as in fast_parity).
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

/// Energies in [-3, 3] and spatial components in [-1, 1]: with fast_parity's
/// [-2, 2] energies, ten shifts essentially never satisfy `planes_ok`.
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

/// Real masses, complex masses and mixed massless/massive propagators.
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

/// Monomials `(q exponents, mu² power)` of a rank-`rank` numerator.
fn numerator_terms(rank: usize) -> Vec<([u32; 4], u32)> {
    let mut terms: Vec<_> = monomials(rank).into_iter().map(|m| (m, 0)).collect();
    for j in 1..=rank as u32 / 2 {
        terms.extend(monomials(rank - 2 * j as usize).into_iter().map(|m| (m, j)));
    }
    terms
}

/// Every term of rank weight at most three, every maximal-weight term in q0
/// and mu² alone, and fixed pseudo-random selections of the other monomials
/// (1/40 through weight six, 1/97 above): preparing the polynomial
/// numerator's Laurent expansions grows steeply with the number of high-rank
/// terms (independently of n; see `examples/generic_scaling.rs`).
fn active(rank: usize, terms: &[([u32; 4], u32)], k: usize) -> bool {
    let (m, j) = terms[k];
    let weight = (m.iter().sum::<u32>() + 2 * j) as usize;
    let hash = k.wrapping_mul(2_654_435_761) >> 7;
    weight <= 3
        || (weight == rank && m[1] + m[2] + m[3] == 0)
        || (weight <= 6 && hash.is_multiple_of(40))
        || hash.is_multiple_of(97)
}

/// Numerator expression and its parameter symbols, one per active term.
fn polynomial(rank: usize) -> (Atom, Vec<Atom>) {
    let names = ["q0", "q1", "q2", "q3"];
    let terms = numerator_terms(rank);
    let mut text = String::from("0");
    let mut parameters = Vec::new();
    for (k, (m, j)) in terms.iter().enumerate() {
        if !active(rank, &terms, k) {
            continue;
        }
        text.push_str(&format!("+p{k}"));
        for (mu, &e) in m.iter().enumerate() {
            if e > 0 {
                text.push_str(&format!("*{}^{e}", names[mu]));
            }
        }
        if *j > 0 {
            text.push_str(&format!("*mu2^{j}"));
        }
        parameters.push(parse(&format!("p{k}")));
    }
    (parse(&text), parameters)
}

/// One random numerator and three phase-space lanes.
struct Problem {
    n: usize,
    rank: usize,
    lanes: Vec<Lane>,
    expression: Atom,
    parameters: Vec<Atom>,
    /// Lane-major coefficients of the active terms.
    coefficients: Vec<C64>,
}

impl Problem {
    fn new(n: usize, rank: usize, seed: u64) -> Self {
        let mut rng = Rng(seed ^ ((n as u64) << 32) ^ ((rank as u64) << 40));
        let lanes = lanes(&mut rng, n);
        let (expression, parameters) = polynomial(rank);
        let coefficients = (0..lanes.len() * parameters.len())
            .map(|_| C64::new(rng.uniform(), rng.uniform()))
            .collect();
        Self {
            n,
            rank,
            lanes,
            expression,
            parameters,
            coefficients,
        }
    }

    fn count(&self) -> usize {
        self.lanes.len()
    }

    fn shifts(&self) -> Vec<RMom> {
        self.lanes.iter().flat_map(|l| l.shifts.clone()).collect()
    }

    fn masses(&self) -> Vec<C64> {
        self.lanes.iter().flat_map(|l| l.masses.clone()).collect()
    }

    fn s_matrix(&self) -> Vec<f64> {
        let n = self.n;
        let mut s = Vec::with_capacity(self.count() * n * n);
        for lane in &self.lanes {
            for i in 0..n {
                for j in 0..n {
                    let d = diff(&lane.shifts[i], &lane.shifts[j]);
                    s.push(mp(&d, &d));
                }
            }
        }
        s
    }

    /// Ninja's dimensional rule for the auxiliary box mu² sampling scale.
    fn sampling(&self) -> Vec<f64> {
        let n = self.n;
        let s = self.s_matrix();
        (0..self.count())
            .map(|lane| {
                s[lane * n * n..(lane + 1) * n * n]
                    .iter()
                    .map(|x| x.abs())
                    .chain(self.lanes[lane].masses.iter().map(|m| m.re.abs()))
                    .fold(1., f64::max)
            })
            .collect()
    }

    fn scales(&self) -> Vec<f64> {
        (0..self.count()).map(|i| [1.0, 2.5, 0.7][i % 3]).collect()
    }

    /// Prepared numerator with this problem's coefficients bound.
    fn numerator<T>(&self, prototype: &T) -> PolynomialNumerator<T>
    where
        T: NativeFloat,
        Complex<T>: Real + EvaluationDomain,
    {
        let mut numerator = PolynomialNumerator::new(
            self.expression.clone(),
            variables(),
            self.parameters.clone(),
            prototype.clone(),
        )
        .unwrap();
        let values: Vec<_> = self
            .coefficients
            .iter()
            .map(|&z| complex(prototype, z))
            .collect();
        numerator
            .set_parameter_batch(&values, self.count())
            .unwrap();
        numerator
    }
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

/// Generic-engine evaluation in any complex carrier with the given masters.
fn generic<T, M>(
    problem: &Problem,
    numerator: &mut PolynomialNumerator<T>,
    prototype: T,
    masters: &mut M,
) -> Vec<[Complex<T>; 5]>
where
    T: NativeFloat,
    Complex<T>: Real + EvaluationDomain,
    M: MasterProvider<Complex<T>>,
{
    let p = &prototype;
    let count = problem.count();
    let mut amplitude = SymbolicaAmplitudeKernels::new(problem.n, problem.rank, p.clone())
        .unwrap()
        .into_amplitude(count)
        .unwrap();
    let real = |x: f64| complex(p, C64::real(x));
    let shifts: Vec<FourMomentum<Complex<T>>> = problem
        .shifts()
        .iter()
        .map(|v| FourMomentum::from(v.map(real)))
        .collect();
    let masses: Vec<_> = problem.masses().iter().map(|&m| complex(p, m)).collect();
    let matrix: Vec<_> = problem.s_matrix().into_iter().map(real).collect();
    let scales: Vec<_> = problem.scales().into_iter().map(real).collect();
    let sampling: Vec<_> = problem.sampling().into_iter().map(real).collect();
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
            numerator,
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
            problem.n,
            problem.rank,
            p.get_precision()
        );
    }
    amplitude.results().to_vec()
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

/// The binary64 fast engine on a problem, with the numerator behind its
/// generic adapter and oneloopmaster native masters.
fn fast(problem: &Problem, numerator: PolynomialNumerator<f64>) -> Vec<[Complex<f64>; 5]> {
    let count = problem.count();
    let shifts = problem.shifts();
    let masses = problem.masses();
    let matrix = problem.s_matrix();
    let scales = problem.scales();
    let input = FastInput {
        lanes: count,
        shifts: &shifts,
        masses_squared: &masses,
        scales_squared: &scales,
        s_matrix: Some(&matrix),
    };
    let mut numerator = GenericNumerator::with_rank(numerator, problem.rank);
    let mut engine = FastAmplitude::new(problem.n, problem.rank, FastOptions::default()).unwrap();
    let mut masters = OneLoopNativeMasters::new().unwrap();
    let mut results = vec![FastResult::default(); count];
    let mut statuses = vec![FastStatus::NotEvaluated; count];
    engine
        .evaluate(
            &input,
            &mut numerator,
            &mut masters,
            &mut results,
            &mut statuses,
        )
        .unwrap();
    for status in &statuses {
        assert_eq!(*status, FastStatus::Success);
    }
    results.iter().map(fast_values).collect()
}

/// The binary64 native master values, widened exactly to DoubleFloat. A
/// DoubleFloat reduction with these is the exact reduction of the binary64
/// inputs with exactly the masters both binary64 engines use.
struct Binary64Masters(OneLoopNativeMasters);

impl MasterProvider<Complex<DoubleFloat>> for Binary64Masters {
    fn evaluate(
        &mut self,
        request: MasterRequest<'_, Complex<DoubleFloat>>,
        output: &mut [Laurent<Complex<DoubleFloat>>],
    ) -> ninjars::Result<()> {
        let mut statuses = vec![MasterStatus::Success; request.rows()];
        self.evaluate_batch(request, output, &mut statuses)
    }

    fn evaluate_batch(
        &mut self,
        request: MasterRequest<'_, Complex<DoubleFloat>>,
        output: &mut [Laurent<Complex<DoubleFloat>>],
        statuses: &mut [MasterStatus],
    ) -> ninjars::Result<()> {
        // Every argument is an exact binary64 value (exact inputs and S-matrix).
        let round = |values: &[Complex<DoubleFloat>]| -> Vec<Complex<f64>> {
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
                *out =
                    value.map(|z| Complex::new(DoubleFloat::from(z.re), DoubleFloat::from(z.im)));
            }
        }
        Ok(())
    }
}

/// Generic binary64, `ninjars::fast` and the exact (DoubleFloat) reduction,
/// all with identical oneloopmaster native binary64 master values.
#[allow(clippy::type_complexity)]
fn binary64_engines(
    problem: &Problem,
) -> (
    Vec<[Complex<f64>; 5]>,
    Vec<[Complex<f64>; 5]>,
    Vec<[Complex<DoubleFloat>; 5]>,
) {
    let mut numerator = problem.numerator(&0_f64);
    let mut masters = OneLoopNativeMasters::new().unwrap();
    let binary64 = generic(problem, &mut numerator, 0_f64, &mut masters);
    let fast = fast(problem, numerator);
    let dd = DoubleFloat::from(0.);
    let reference = generic(
        problem,
        &mut problem.numerator(&dd),
        dd,
        &mut Binary64Masters(OneLoopNativeMasters::new().unwrap()),
    );
    (binary64, fast, reference)
}

/// (a) Generic binary64 vs `ninjars::fast`, identical native masters. The
/// engines agree to 1e-10 of the largest Laurent coefficient at well-conditioned
/// points. Where they do not, both are measured against the exact reduction
/// with the same master values: high-rank points with many uncut lines lose
/// digits in both engines (with different cut bases, by different amounts),
/// and the generic engine must stay within ten times the fast engine's error.
fn fast_parity(n: usize) {
    let (mut worst, mut direct, mut checks) = (0_f64, 0, 0);
    for rank in [n - 1, n, n + 1] {
        let problem = Problem::new(n, rank, SEED);
        let (binary64, fast, reference) = binary64_engines(&problem);
        for lane in 0..problem.count() {
            let error = laurent_error(&binary64[lane], &fast[lane]);
            let generic_error = laurent_error(&binary64[lane], &reference[lane]);
            let fast_error = laurent_error(&fast[lane], &reference[lane]);
            eprintln!(
                "ANY_N_FAST n={n} rank={rank} lane={lane} generic_vs_fast={error:.3e} \
                 generic_vs_exact={generic_error:.3e} fast_vs_exact={fast_error:.3e}"
            );
            assert!(generic_error <= 1e-7, "n={n} rank={rank} lane={lane}");
            if error <= 1e-10 {
                direct += 1;
            } else {
                assert!(
                    generic_error <= 10. * fast_error.max(1e-11),
                    "n={n} rank={rank} lane={lane}: generic {:?} fast {:?} \
                     (difference {error:e}; errors generic {generic_error:e}, fast {fast_error:e})",
                    &binary64[lane][..3],
                    &fast[lane][..3]
                );
            }
            worst = worst.max(error);
            checks += 1;
        }
    }
    eprintln!(
        "ANY_N_FAST_SUMMARY n={n} checks={checks} max_generic_vs_fast={worst:.3e} \
         agreeing_to_1e-10={direct}"
    );
    assert!(2 * direct >= checks, "too few well-conditioned points");
}

#[test]
fn generic_binary64_matches_fast_engine_n7() {
    fast_parity(7);
}

#[test]
fn generic_binary64_matches_fast_engine_n8() {
    fast_parity(8);
}

/// Smooth polynomial stand-ins for every master family, generic over carriers.
///
/// Values depend on all invariants, masses and the scale, so the reduced
/// amplitude depends nonlinearly on every reduction coefficient's kinematics.
struct PolynomialMasters;

impl<V: FloatLike> MasterProvider<V> for PolynomialMasters {
    fn evaluate(
        &mut self,
        request: MasterRequest<'_, V>,
        output: &mut [Laurent<V>],
    ) -> ninjars::Result<()> {
        let mut statuses = vec![MasterStatus::Success; request.rows()];
        self.evaluate_batch(request, output, &mut statuses)
    }

    fn evaluate_batch(
        &mut self,
        request: MasterRequest<'_, V>,
        output: &mut [Laurent<V>],
        statuses: &mut [MasterStatus],
    ) -> ninjars::Result<()> {
        let ni = request.integral.invariant_count();
        let nm = request.integral.mass_count();
        let family = format!("{:?}", request.integral)
            .bytes()
            .fold(7_i64, |h, b| (h * 31 + i64::from(b)) % 97);
        for row in 0..request.rows() {
            let scale = &request.scales_squared[row];
            let args: Vec<&V> = request.invariants[row * ni..(row + 1) * ni]
                .iter()
                .chain(&request.masses_squared[row * nm..(row + 1) * nm])
                .collect();
            let c = |x: i64| scale.from_i64(x);
            let mut linear = c(family) / c(10);
            let mut quadratic = scale.clone() / c(3);
            for (k, &a) in args.iter().enumerate() {
                let w = k as i64;
                linear += a.clone() * &(c(w + 2) / c(7));
                quadratic += a.clone() * &(a.clone() * &(c(w % 3 - 1) / c(11)));
                if let Some(&next) = args.get(k + 1) {
                    quadratic += a.clone() * &(next.clone() / c(13));
                }
            }
            output[row] = [
                linear.clone() * &quadratic,
                linear + &c(1),
                c(family) / c(5),
            ];
            statuses[row] = MasterStatus::Success;
        }
        Ok(())
    }
}

/// (b) DoubleFloat and (c) Float(200) against Float(300).
fn precision_ladder(n: usize) {
    for rank in [n - 1, n, n + 1] {
        let problem = Problem::new(n, rank, SEED + 1);
        let dd = DoubleFloat::from(0.);
        let low = generic(
            &problem,
            &mut problem.numerator(&dd),
            dd,
            &mut PolynomialMasters,
        );
        let f200 = Float::new(200);
        let middle = generic(
            &problem,
            &mut problem.numerator(&f200),
            f200,
            &mut PolynomialMasters,
        );
        let f300 = Float::new(300);
        let high = generic(
            &problem,
            &mut problem.numerator(&f300),
            f300,
            &mut PolynomialMasters,
        );
        for lane in 0..problem.count() {
            let dd_error = laurent_error(&low[lane], &high[lane]);
            let float_error = laurent_error(&middle[lane], &high[lane]);
            eprintln!(
                "ANY_N_PRECISION n={n} rank={rank} lane={lane} \
                 double_float={dd_error:.3e} float200={float_error:.3e}"
            );
            assert!(
                dd_error <= 1e-28,
                "n={n} rank={rank} lane={lane}: {dd_error:e}"
            );
            assert!(
                float_error <= 1e-55,
                "n={n} rank={rank} lane={lane}: {float_error:e}"
            );
        }
    }
}

#[test]
fn double_float_and_float_precisions_agree_n7() {
    precision_ladder(7);
}

#[test]
fn double_float_and_float_precisions_agree_n8() {
    precision_ladder(8);
}

/// The derivative variable x enters one squared mass and (with weight 1/2)
/// one shift component of every lane, and the first numerator coefficient.
const MASS_LINE: usize = 3;
const SHIFT_LINE: usize = 2;

/// Float evaluations of the amplitude at each x with the stand-in masters.
fn float_at(problem: &Problem, bits: u32, points: &[Float]) -> Vec<Vec<[Complex<Float>; 5]>> {
    let p = Float::new(bits);
    let n = problem.n;
    let count = problem.count();
    let mut numerator = PolynomialNumerator::new(
        problem.expression.clone(),
        variables(),
        problem.parameters.clone(),
        p.clone(),
    )
    .unwrap();
    let mut amplitude = SymbolicaAmplitudeKernels::new(n, problem.rank, p.clone())
        .unwrap()
        .into_amplitude(count)
        .unwrap();
    let mut low = SymbolicaCuts::new(p.clone()).unwrap();
    let mut high = SymbolicaHighCuts::new(p.clone()).unwrap();
    let real = |s: f64| complex(&p, C64::real(s));
    let scales: Vec<_> = problem.scales().into_iter().map(real).collect();
    let sampling: Vec<_> = problem.sampling().into_iter().map(real).collect();
    let width = problem.parameters.len();
    let mut results = Vec::new();
    for x in points {
        let values: Vec<_> = problem
            .coefficients
            .iter()
            .enumerate()
            .map(|(i, &z)| {
                let mut v = complex(&p, z);
                if i % width == 0 {
                    v.re += x;
                }
                v
            })
            .collect();
        numerator.set_parameter_batch(&values, count).unwrap();
        let mut shifts: Vec<FourMomentum<Complex<Float>>> = problem
            .shifts()
            .iter()
            .map(|v| FourMomentum::from(v.map(real)))
            .collect();
        let mut masses: Vec<_> = problem.masses().iter().map(|&m| complex(&p, m)).collect();
        let half = x.clone() * &exact(&p, 0.5);
        for lane in 0..count {
            masses[lane * n + MASS_LINE].re += x;
            shifts[lane * n + SHIFT_LINE].spatial.px.re += &half;
        }
        amplitude.reset();
        amplitude
            .evaluate_batch_with_inputs(
                AmplitudeInput {
                    shifts: &shifts,
                    masses_squared: &masses,
                    scales_squared: &scales,
                    sampling_scales: &sampling,
                    s_matrix: None,
                },
                &mut numerator,
                &mut low,
                &mut high,
                &mut PolynomialMasters,
            )
            .unwrap();
        assert!(
            amplitude
                .statuses()
                .iter()
                .all(|s| *s == EvaluationStatus::Success)
        );
        results.push(amplitude.results().to_vec());
    }
    results
}

/// Dual evaluation at x = 0 with unit seeds in the same places. Components
/// follow Symbolica's Taylor convention: `[value, d/dx, (d/dx)²/2]`.
fn dual_at<T>(problem: &Problem, prototype: T) -> Vec<[HyperDual<Complex<T>>; 5]>
where
    T: NativeFloat,
    Complex<T>: Real + EvaluationDomain,
{
    let p = &prototype;
    let n = problem.n;
    let count = problem.count();
    let zero = Complex::new(p.zero(), p.zero());
    let d = HyperDual::from_values(vec![vec![0], vec![1], vec![2]], vec![zero; 3]);
    let constant = |z: Complex<T>| {
        let mut v = d.clone();
        v.values[0] = z;
        v
    };
    let seeded = |z: Complex<T>, slope: T| {
        let mut v = constant(z);
        v.values[1] = Complex::new(slope, p.zero());
        v
    };
    let one = p.one();
    let half = exact(p, 0.5);
    let source = PolynomialNumerator::new(
        problem.expression.clone(),
        variables(),
        problem.parameters.clone(),
        p.clone(),
    )
    .unwrap();
    let mut numerator = DualPolynomialNumerator::new(source, d.clone()).unwrap();
    let width = problem.parameters.len();
    let values: Vec<_> = problem
        .coefficients
        .iter()
        .enumerate()
        .map(|(i, &z)| {
            if i % width == 0 {
                seeded(complex(p, z), one.clone())
            } else {
                constant(complex(p, z))
            }
        })
        .collect();
    numerator.set_parameter_batch(&values, count).unwrap();
    let mut amplitude = SymbolicaDualAmplitudeKernels::new(n, problem.rank, p.clone(), &d)
        .unwrap()
        .into_amplitude(count)
        .unwrap();
    let real = |x: f64| constant(complex(p, C64::real(x)));
    let shifts: Vec<FourMomentum<HyperDual<Complex<T>>>> = problem
        .shifts()
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let px = if i % n == SHIFT_LINE {
                seeded(complex(p, C64::real(v[1])), half.clone())
            } else {
                real(v[1])
            };
            FourMomentum::from_args(real(v[0]), px, real(v[2]), real(v[3]))
        })
        .collect();
    let masses: Vec<_> = problem
        .masses()
        .iter()
        .enumerate()
        .map(|(i, &m)| {
            if i % n == MASS_LINE {
                seeded(complex(p, m), one.clone())
            } else {
                constant(complex(p, m))
            }
        })
        .collect();
    let scales: Vec<_> = problem.scales().into_iter().map(real).collect();
    let sampling: Vec<_> = problem.sampling().into_iter().map(real).collect();
    let mut low = SymbolicaDualCuts::new(p.clone(), d.clone()).unwrap();
    let mut high = SymbolicaDualHighCuts::new(p.clone(), d.clone()).unwrap();
    amplitude
        .evaluate_batch_with_inputs(
            AmplitudeInput {
                shifts: &shifts,
                masses_squared: &masses,
                scales_squared: &scales,
                sampling_scales: &sampling,
                s_matrix: None,
            },
            &mut numerator,
            &mut low,
            &mut high,
            &mut PolynomialMasters,
        )
        .unwrap();
    for (lane, status) in amplitude.statuses().iter().enumerate() {
        assert_eq!(*status, EvaluationStatus::Success, "lane {lane}");
    }
    amplitude.results().to_vec()
}

/// (d) Dual derivatives against central finite differences of Float(512)
/// evaluations. With step 2^-80, truncation is O(2^-160) and rounding at most
/// O(2^-352) in the second difference, both far below the 1e-40 tolerance.
fn derivatives(n: usize, rank: usize) {
    let problem = Problem::new(n, rank, SEED + 2);
    let bits = 512;
    let zero = Float::new(bits);
    let h = exact(&zero, 2_f64.powi(-80));
    let mut evaluations = float_at(
        &problem,
        bits,
        &[zero.clone() - &h, zero.clone(), zero.clone() + &h],
    )
    .into_iter();
    let (minus, center, plus) = (
        evaluations.next().unwrap(),
        evaluations.next().unwrap(),
        evaluations.next().unwrap(),
    );
    let dual64 = dual_at(&problem, 0_f64);
    let dual200 = dual_at(&problem, Float::new(200));
    let real = |x: Float| Complex::new(x, zero.clone());
    let first_step = real((h.clone() * &exact(&zero, 2.)).inv());
    let second_step = real((h.clone() * &h * &exact(&zero, 2.)).inv());
    let two = real(exact(&zero, 2.));
    let (mut worst64, mut worst200) = (0_f64, 0_f64);
    for lane in 0..problem.count() {
        for k in 0..5 {
            let first = (plus[lane][k].clone() - &minus[lane][k]) * &first_step;
            let second = (plus[lane][k].clone() - &(center[lane][k].clone() * &two)
                + &minus[lane][k])
                * &second_step;
            let expected = [&center[lane][k], &first, &second];
            let scale = expected
                .iter()
                .map(|z| to_c64(z).abs())
                .fold(f64::MIN_POSITIVE, f64::max);
            for (order, e) in expected.iter().enumerate() {
                let error64 = distance(&dual64[lane][k].values[order], e) / scale;
                let error200 = distance(&dual200[lane][k].values[order], e) / scale;
                assert!(
                    error64 <= 1e-9,
                    "n={n} rank={rank} lane={lane} component={k} order={order}: {error64:e}"
                );
                assert!(
                    error200 <= 1e-40,
                    "n={n} rank={rank} lane={lane} component={k} order={order}: {error200:e}"
                );
                worst64 = worst64.max(error64);
                worst200 = worst200.max(error200);
            }
        }
        // The derivative variable must actually move the finite part.
        let first = (plus[lane][0].clone() - &minus[lane][0]) * &first_step;
        assert!(to_c64(&first).abs() > 1e-6 * to_c64(&center[lane][0]).abs());
    }
    eprintln!(
        "ANY_N_DUAL n={n} rank={rank} max_error_dual_f64={worst64:.3e} \
         max_error_dual_float200={worst200:.3e}"
    );
}

#[test]
fn dual_derivatives_match_finite_differences_n7() {
    derivatives(7, 8);
}

/// Original Ninja's binary64 results for a seven-propagator top loop (2 -> 5,
/// `validation/fast/make_edge_cases.py`) against the exact reduction of the
/// generic engine in Float(256), both with the OneLOop master values original
/// Ninja used (`tests/common/precision.rs`).
fn original_ninja(name: &str) {
    let root =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../validation/fast/ninja_reference");
    let file = format!("{name}.json");
    let case =
        parse_case(&std::fs::read_to_string(root.join("cases").join(&file)).unwrap()).unwrap();
    let ninja = parse_ninja(
        &std::fs::read_to_string(root.join("ninja").join(&file)).unwrap(),
        None,
    )
    .unwrap();
    assert_eq!(case.n, 7);
    let lanes = precision(&case, &ninja, 256, None).unwrap();
    assert_eq!(lanes.len(), case.points.len());
    for lane in lanes {
        eprintln!(
            "ANY_N_NINJA {name} {}: scale {:.3e}, original Ninja error {:.3e}, fast error {:.3e}",
            lane.id, lane.scale, lane.ninja_laurent, lane.fast_laurent
        );
        assert_eq!(lane.fast_status, FastStatus::Success, "{name} {}", lane.id);
        assert!(
            lane.ninja_laurent <= 1e-9,
            "{name} {}: original Ninja differs from the exact generic reduction by {:e}",
            lane.id,
            lane.ninja_laurent
        );
    }
}

#[test]
fn seven_propagators_rank_six_match_original_ninja_with_identical_masters() {
    original_ninja("edge-many-n7-r6");
}

/// The dense rank-7 tensor numerator takes minutes of symbolic expansion
/// preparation (see `examples/generic_scaling.rs`). Observed: original Ninja
/// within 3.7e-13 of the exact reduction on all four lanes.
#[test]
#[ignore = "dense rank-7 numerator preparation takes several minutes"]
fn seven_propagators_rank_seven_match_original_ninja_with_identical_masters() {
    original_ninja("edge-many-n7-r7");
}

/// One smoke test at n = 10 and rank n-1 (every cut stage active): generic
/// binary64 and the fast engine against the exact reduction, all with
/// identical native masters.
#[test]
fn ten_propagator_smoke() {
    let n = 10;
    let problem = Problem::new(n, n - 1, SEED + 3);
    let (binary64, fast, reference) = binary64_engines(&problem);
    for lane in 0..problem.count() {
        let generic_error = laurent_error(&binary64[lane], &reference[lane]);
        let fast_error = laurent_error(&fast[lane], &reference[lane]);
        eprintln!(
            "ANY_N_SMOKE n={n} rank={} lane={lane} generic_vs_exact={generic_error:.3e} \
             fast_vs_exact={fast_error:.3e}",
            n - 1
        );
        assert!(generic_error <= 1e-8, "lane {lane}: {generic_error:e}");
        assert!(fast_error <= 1e-8, "lane {lane}: {fast_error:e}");
    }
}
