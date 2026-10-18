//! Shared helpers of the rank n+2 generic-engine tests (`tests/rank_n2_*.rs`).
//!
//! Kinematics and numerator coefficients are dyadic rationals (multiples of
//! 2^-20), so every carrier (binary64, DoubleFloat, Float) sees identical
//! inputs. Residue formulas are written out independently of the engine from
//! the canonical tables of `tools/rank_n2/theory/formulas_rank_n2.json`.

#![allow(dead_code)]

use ninjars::fast::{C64, RMom, tensor_layout::monomials};
use ninjars::{
    Amplitude, AmplitudeInput, BoxCoefficientMode, BoxMuMode, BoxReductionOptions, Complex,
    EvaluationStatus, Float, FourMomentum, Laurent, MasterProvider, MasterRequest, MasterStatus,
    Numerator, Real, SingleFloat,
};
use ninjars_symbolica::{
    PolynomialNumerator, SymbolicaAmplitudeKernels, SymbolicaCuts, SymbolicaHighCuts,
};
use oneloop::NativeFloat;
use symbolica::{
    atom::Atom,
    domains::{float::RealLike, rational::Rational},
    evaluate::EvaluationDomain,
    parser::ParseSettings,
};

pub const NAMESPACE: &str = "ninjars_rank_n2";

pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform on [-1, 1], rounded to 2^-20 (exact in every carrier).
    pub fn uniform(&mut self) -> f64 {
        let x = (self.next() >> 11) as f64 / (1u64 << 53) as f64 * 2. - 1.;
        (x * 1048576.).round() / 1048576.
    }
    pub fn complex(&mut self) -> C64 {
        C64::new(self.uniform(), self.uniform())
    }
}

pub fn parse(text: &str) -> Atom {
    Atom::parse(text, NAMESPACE, ParseSettings::default()).unwrap()
}

pub fn variables() -> [Atom; 5] {
    ["q0", "q1", "q2", "q3", "mu2"].map(parse)
}

pub fn exact<T: SingleFloat>(prototype: &T, x: f64) -> T {
    prototype.from_rational(&Rational::try_from(x).unwrap())
}

pub fn complex<T: SingleFloat>(prototype: &T, z: C64) -> Complex<T> {
    Complex::new(exact(prototype, z.re), exact(prototype, z.im))
}

pub fn to_c64<T: RealLike>(z: &Complex<T>) -> C64 {
    C64::new(z.re.to_f64(), z.im.to_f64())
}

/// Exact widening to a common 1024-bit Float.
pub fn wide<T: Clone + Into<Float>>(x: &T) -> Float {
    let mut x: Float = x.clone().into();
    x.set_prec(1024);
    x
}

/// Modulus of a complex difference, computed exactly before rounding.
pub fn distance<T: Clone + Into<Float>, U: Clone + Into<Float>>(
    a: &Complex<T>,
    b: &Complex<U>,
) -> f64 {
    let re = wide(&a.re) - &wide(&b.re);
    let im = wide(&a.im) - &wide(&b.im);
    re.to_f64().hypot(im.to_f64())
}

pub fn modulus<T: RealLike>(z: &Complex<T>) -> f64 {
    to_c64(z).abs()
}

/// Laurent difference `[finite, 1/eps, 1/eps²]` relative to the reference's
/// largest Laurent modulus (the scaled convention of `fast_ninja_reference`).
pub fn laurent_error<T, U>(a: &[Complex<T>], reference: &[Complex<U>]) -> f64
where
    T: Clone + Into<Float>,
    U: Clone + Into<Float> + RealLike,
{
    let error = (0..3)
        .map(|k| distance(&a[k], &reference[k]))
        .fold(0., f64::max);
    let scale = (0..3)
        .map(|k| modulus(&reference[k]))
        .fold(f64::MIN_POSITIVE, f64::max);
    error / scale
}

pub fn mp(a: &RMom, b: &RMom) -> f64 {
    a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
}

pub fn diff(a: &RMom, b: &RMom) -> RMom {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
}

/// Real cut bases need non-spacelike planes for every box/triangle pair; keep
/// the planes and invariants away from degenerate limits (as in `any_n`).
pub fn planes_ok(v: &[RMom]) -> bool {
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

/// Energies in [-3, 3] and spatial components in [-1, 1], first shift zero.
pub fn kinematics(rng: &mut Rng, n: usize) -> Vec<RMom> {
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
pub struct Lane {
    pub shifts: Vec<RMom>,
    pub masses: Vec<C64>,
}

/// A lane with real masses and one with complex masses on the same shifts,
/// and a lane with mixed massless/massive lines on other shifts.
pub fn lanes(rng: &mut Rng, n: usize) -> Vec<Lane> {
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

/// Monomials `(q exponents, mu² power)` of every term of weight <= `rank`.
pub fn numerator_terms(rank: usize) -> Vec<([u32; 4], u32)> {
    let mut terms: Vec<_> = monomials(rank).into_iter().map(|m| (m, 0)).collect();
    for j in 1..=rank as u32 / 2 {
        terms.extend(monomials(rank - 2 * j as usize).into_iter().map(|m| (m, j)));
    }
    terms
}

pub fn weight(term: &([u32; 4], u32)) -> usize {
    (term.0.iter().sum::<u32>() + 2 * term.1) as usize
}

/// `sum_k p_k * term_k` over the selected terms, one parameter per term.
pub fn polynomial(terms: &[([u32; 4], u32)]) -> (Atom, Vec<Atom>) {
    let names = ["q0", "q1", "q2", "q3"];
    let mut text = String::from("0");
    let mut parameters = Vec::new();
    for (k, (m, j)) in terms.iter().enumerate() {
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

/// One random rank-`rank` numerator and its phase-space lanes.
pub struct Problem {
    pub n: usize,
    pub rank: usize,
    pub lanes: Vec<Lane>,
    pub terms: Vec<([u32; 4], u32)>,
    pub expression: Atom,
    pub parameters: Vec<Atom>,
    /// Lane-major coefficients of `terms`.
    pub coefficients: Vec<C64>,
}

impl Problem {
    /// Every term of weight <= 3 and every top-weight term; lower weights
    /// keep a fixed pseudo-random selection (Symbolica's preparation of the
    /// numerator expansions grows steeply with the number of terms). With
    /// `mu_terms = false` the numerator is a pure tensor in q.
    pub fn new(n: usize, rank: usize, seed: u64, mu_terms: bool, top_stride: usize) -> Self {
        let mut rng = Rng(seed ^ ((n as u64) << 32) ^ ((rank as u64) << 40));
        let lanes = lanes(&mut rng, n);
        let terms: Vec<_> = numerator_terms(rank)
            .into_iter()
            .enumerate()
            .filter(|(k, term)| {
                let w = weight(term);
                let hash = k.wrapping_mul(2_654_435_761) >> 7;
                (mu_terms || term.1 == 0)
                    && (w <= 3
                        || (w == rank && hash.is_multiple_of(top_stride))
                        || term.0[1] + term.0[2] + term.0[3] == 0
                        || hash.is_multiple_of(5 * top_stride))
            })
            .map(|(_, term)| term)
            .collect();
        let (expression, parameters) = polynomial(&terms);
        let coefficients = (0..lanes.len() * terms.len())
            .map(|_| rng.complex())
            .collect();
        Self {
            n,
            rank,
            lanes,
            terms,
            expression,
            parameters,
            coefficients,
        }
    }

    pub fn count(&self) -> usize {
        self.lanes.len()
    }

    pub fn shifts(&self) -> Vec<RMom> {
        self.lanes.iter().flat_map(|l| l.shifts.clone()).collect()
    }

    pub fn masses(&self) -> Vec<C64> {
        self.lanes.iter().flat_map(|l| l.masses.clone()).collect()
    }

    pub fn s_matrix(&self) -> Vec<f64> {
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
    pub fn sampling(&self) -> Vec<f64> {
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

    pub fn scales(&self) -> Vec<f64> {
        (0..self.count()).map(|i| [1.0, 2.5, 0.7][i % 3]).collect()
    }

    /// Prepared numerator with this problem's coefficients bound.
    pub fn numerator<T>(&self, prototype: &T) -> PolynomialNumerator<T>
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

pub fn options(coefficients: BoxCoefficientMode, mu_extraction: BoxMuMode) -> BoxReductionOptions {
    BoxReductionOptions {
        coefficients,
        mu_extraction,
    }
}

/// Zero masters: residue-level tests do not need integrals.
pub struct ZeroMasters;
impl<T: NativeFloat> MasterProvider<Complex<T>> for ZeroMasters
where
    Complex<T>: Real,
{
    fn evaluate(
        &mut self,
        request: MasterRequest<'_, Complex<T>>,
        output: &mut [Laurent<Complex<T>>],
    ) -> ninjars::Result<()> {
        let mut statuses = vec![MasterStatus::Success; output.len()];
        self.evaluate_batch(request, output, &mut statuses)
    }
    fn evaluate_batch(
        &mut self,
        request: MasterRequest<'_, Complex<T>>,
        output: &mut [Laurent<Complex<T>>],
        statuses: &mut [MasterStatus],
    ) -> ninjars::Result<()> {
        request.validate(output.len())?;
        for (value, status) in output.iter_mut().zip(statuses.iter_mut()) {
            let zero = request.scales_squared[0].zero();
            *value = [zero.clone(), zero.clone(), zero];
            *status = MasterStatus::Success;
        }
        Ok(())
    }
}

/// Lane-major momenta and masses of a problem in carrier `T`.
pub fn carriers<T: SingleFloat>(
    problem: &Problem,
    prototype: &T,
) -> (Vec<FourMomentum<Complex<T>>>, Vec<Complex<T>>) {
    let real = |x: f64| complex(prototype, C64::real(x));
    let shifts = problem
        .shifts()
        .iter()
        .map(|v| FourMomentum::from(v.map(real)))
        .collect();
    let masses = problem
        .masses()
        .iter()
        .map(|&m| complex(prototype, m))
        .collect();
    (shifts, masses)
}

/// The generic engine on every lane of `problem`; all lanes must succeed.
pub fn reduce<T, M, N>(
    problem: &Problem,
    numerator: &mut N,
    prototype: &T,
    box_options: BoxReductionOptions,
    masters: &mut M,
) -> Amplitude<Complex<T>, SymbolicaAmplitudeKernels<T>>
where
    T: NativeFloat,
    Complex<T>: Real + EvaluationDomain,
    M: MasterProvider<Complex<T>>,
    N: Numerator<Complex<T>>,
{
    reduce_with(problem, numerator, prototype, box_options, masters, |k| k)
}

/// [`reduce`] with further kernel configuration.
pub fn reduce_with<T, M, N>(
    problem: &Problem,
    numerator: &mut N,
    prototype: &T,
    box_options: BoxReductionOptions,
    masters: &mut M,
    configure: impl FnOnce(SymbolicaAmplitudeKernels<T>) -> SymbolicaAmplitudeKernels<T>,
) -> Amplitude<Complex<T>, SymbolicaAmplitudeKernels<T>>
where
    T: NativeFloat,
    Complex<T>: Real + EvaluationDomain,
    M: MasterProvider<Complex<T>>,
    N: Numerator<Complex<T>>,
{
    let p = prototype;
    let count = problem.count();
    let kernels = SymbolicaAmplitudeKernels::new(problem.n, problem.rank, p.clone())
        .unwrap()
        .with_box_options(box_options);
    let mut amplitude = configure(kernels).into_amplitude(count).unwrap();
    let real = |x: f64| complex(p, C64::real(x));
    let (shifts, masses) = carriers(problem, p);
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
    amplitude
}

// ---------------------------------------------------------------------------
// Canonical residues as functions of (q, mu²), written out from the theory
// tables independently of the engine's ansatz code.

pub fn power<T: Real>(z: &Complex<T>, k: u32) -> Complex<T> {
    let mut result = z.one();
    for _ in 0..k {
        result = result * z;
    }
    result
}

/// Ninja order with u = q̄·e3, v = q̄·e4 (q̄ = q + V_first): c0..c14 as in
/// Ninja (`polys_implem.cxx`), then the rank n+2 terms c15..c20 = u⁵, v⁵,
/// mu²u³, mu²v³, mu⁴u, mu⁴v.
pub const TRIANGLE_MONOMIALS: [(u32, u32, u32); 21] = [
    (0, 0, 0),
    (1, 0, 0),
    (2, 0, 0),
    (3, 0, 0),
    (0, 1, 0),
    (0, 2, 0),
    (0, 3, 0),
    (0, 0, 1),
    (1, 0, 1),
    (0, 1, 1),
    (2, 0, 1),
    (0, 2, 1),
    (4, 0, 0),
    (0, 4, 0),
    (0, 0, 2),
    (5, 0, 0),
    (0, 5, 0),
    (3, 0, 1),
    (0, 3, 1),
    (1, 0, 2),
    (0, 1, 2),
];

/// Ninja order `(w, u, v, mu²)` exponents with w = q̄·e2, u = q̄·e3,
/// v = q̄·e4, then the rank n+2 terms c20..c34.
pub const BUBBLE_MONOMIALS: [(u32, u32, u32, u32); 35] = [
    (0, 0, 0, 0),
    (1, 0, 0, 0),
    (2, 0, 0, 0),
    (0, 1, 0, 0),
    (0, 2, 0, 0),
    (0, 0, 1, 0),
    (0, 0, 2, 0),
    (1, 1, 0, 0),
    (1, 0, 1, 0),
    (0, 0, 0, 1),
    (1, 0, 0, 1),
    (0, 1, 0, 1),
    (0, 0, 1, 1),
    (3, 0, 0, 0),
    (0, 3, 0, 0),
    (0, 0, 3, 0),
    (2, 1, 0, 0),
    (2, 0, 1, 0),
    (1, 2, 0, 0),
    (1, 0, 2, 0),
    (4, 0, 0, 0),
    (2, 0, 0, 1),
    (0, 0, 0, 2),
    (0, 4, 0, 0),
    (0, 0, 4, 0),
    (1, 3, 0, 0),
    (1, 0, 3, 0),
    (2, 2, 0, 0),
    (2, 0, 2, 0),
    (3, 1, 0, 0),
    (3, 0, 1, 0),
    (0, 2, 0, 1),
    (0, 0, 2, 1),
    (1, 1, 0, 1),
    (1, 0, 1, 1),
];

pub fn triangle_value<T: Real>(
    c: &[Complex<T>],
    u: &Complex<T>,
    v: &Complex<T>,
    mu: &Complex<T>,
) -> Complex<T> {
    c.iter()
        .zip(TRIANGLE_MONOMIALS)
        .fold(u.zero(), |sum, (c, (a, b, j))| {
            sum + c.clone() * &power(u, a) * &power(v, b) * &power(mu, j)
        })
}

pub fn bubble_value<T: Real>(
    c: &[Complex<T>],
    w: &Complex<T>,
    u: &Complex<T>,
    v: &Complex<T>,
    mu: &Complex<T>,
) -> Complex<T> {
    c.iter()
        .zip(BUBBLE_MONOMIALS)
        .fold(u.zero(), |sum, (c, (a, b, d, j))| {
            sum + c.clone() * &power(w, a) * &power(u, b) * &power(v, d) * &power(mu, j)
        })
}

/// `c0 + c2 mu² + c4 mu⁴ + c6 mu⁶ + (c1 + c3 mu² + c5 mu⁴) x`, c6 at rank n+2.
pub fn box_value<T: Real>(c: &[Complex<T>], x: &Complex<T>, mu: &Complex<T>) -> Complex<T> {
    let mut even = c[0].clone() + c[2].clone() * mu + c[4].clone() * &power(mu, 2);
    if c.len() > 6 {
        even = even + c[6].clone() * &power(mu, 3);
    }
    let mut odd = c[1].clone() + c[3].clone() * mu;
    if c.len() > 5 {
        odd = odd + c[5].clone() * &power(mu, 2);
    }
    even + odd * x
}

pub fn denominator<T: Real>(
    q: &FourMomentum<Complex<T>>,
    shift: &FourMomentum<Complex<T>>,
    mass: &Complex<T>,
    mu: &Complex<T>,
) -> Complex<T> {
    let r = q.clone() + shift.clone();
    r.dot(&r) - mass - mu
}

/// Smooth polynomial stand-ins for every master family (B1111 included),
/// generic over carriers (as in `any_n`): derivative tests of the reduction
/// alone, outside the masters' supported derivative domains.
pub struct PolynomialMasters;

impl<V: ninjars::FloatLike> MasterProvider<V> for PolynomialMasters {
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
