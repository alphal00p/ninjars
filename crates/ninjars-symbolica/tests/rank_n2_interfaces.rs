//! The rank n+2 interfaces frozen before the engines support r = n+2.
//!
//! - `ExpansionKind::MuShifted` (`q = t v0 + v1`, `mu² = t² v0² - beta`)
//!   against direct substitution, for `PolynomialNumerator` in binary64 and
//!   `Float`, and for `DualPolynomialNumerator` (every Dual component).
//! - `GenericNumerator`: the order-4 `T2` layout (and mindeg-5 `T3`) against
//!   direct evaluation of the numerator on the cut parametrisation, and
//!   `FastNumerator::mu_expansion_x2`.
//! - `MasterIntegral::B1111` through `SymbolicaMasters` (exact oneloopmaster
//!   expressions) against oneloopmaster's references, `FastMaster::BubbleRank4`
//!   through `GenericMasters`, and `OneLoopNativeMasters` rejecting both until
//!   the native B1111 family exists.
//! - Rank n+2 accepted by the generic engine only (the fast engine follows
//!   with its own rank n+2 engine); rank n+3 rejected by both.

// Symbolica's generated Dual division implementations multiply by an inverse.
#![allow(clippy::suspicious_arithmetic_impl, clippy::suspicious_op_assign_impl)]

use ninjars::fast::{
    C64, CMom, FastAmplitude, FastMaster, FastMasters, FastNumerator, FastOptions, GenericMasters,
    GenericNumerator, adapters::t2_powers, adapters::t3_powers, t2_len, t3_len,
};
use ninjars::{
    Complex, Error, ExpansionKind, ExpansionRequest, Float, FourMomentum, Laurent, MasterIntegral,
    MasterProvider, MasterRequest, MasterStatus, Numerator, NumeratorRequest, RankClass, SampleId,
    Topology,
};
use ninjars_symbolica::{
    DualPolynomialNumerator, LaurentDivision, OneLoopNativeMasters, PolynomialNumerator,
    SymbolicaAmplitudeKernels, SymbolicaMasters,
};
use symbolica::{
    atom::Atom,
    domains::{
        dual::{DualNumber, HyperDual},
        float::{Real, RealLike, SingleFloat},
        rational::Rational,
    },
    evaluate::EvaluationDomain,
    parser::ParseSettings,
};

const NAMESPACE: &str = "ninjars_rank_n2_interfaces";

fn parse(text: &str) -> Atom {
    Atom::parse(text, NAMESPACE, ParseSettings::default()).unwrap()
}

fn variables() -> [Atom; 5] {
    ["q0", "q1", "q2", "q3", "mu2"].map(parse)
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Dyadic values in [-1, 1) with 20 bits, exact in every carrier.
    fn uniform(&mut self) -> f64 {
        (self.next() >> 44) as f64 / (1u64 << 19) as f64 - 1.
    }
    fn complex(&mut self) -> C64 {
        C64::new(self.uniform(), self.uniform())
    }
    fn vector(&mut self) -> CMom {
        [
            self.complex(),
            self.complex(),
            self.complex(),
            self.complex(),
        ]
    }
}

/// Monomials `(q exponents, mu² power)`.
type Terms = Vec<([u32; 4], u32)>;

/// A dense rank-`rank` numerator with explicit `mu²` powers, one parameter
/// per monomial. Returns the expression, its parameters and the monomials.
fn dense(rank: u32) -> (Atom, Vec<Atom>, Terms) {
    let names = ["q0", "q1", "q2", "q3"];
    let mut terms = Vec::new();
    for j in 0..=rank / 2 {
        for e0 in 0..=rank {
            for e1 in 0..=rank {
                for e2 in 0..=rank {
                    for e3 in 0..=rank {
                        if e0 + e1 + e2 + e3 + 2 * j <= rank {
                            terms.push(([e0, e1, e2, e3], j));
                        }
                    }
                }
            }
        }
    }
    let mut text = String::new();
    for (k, (m, j)) in terms.iter().enumerate() {
        if k > 0 {
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
    (parse(&text), parameters, terms)
}

fn convert<T: SingleFloat>(prototype: &T, z: C64) -> Complex<T> {
    Complex::new(
        prototype.from_rational(&Rational::try_from(z.re).unwrap()),
        prototype.from_rational(&Rational::try_from(z.im).unwrap()),
    )
}

fn momentum<T: Clone>(v: [Complex<T>; 4]) -> FourMomentum<Complex<T>> {
    let [a, b, c, d] = v;
    FourMomentum::from_args(a, b, c, d)
}

fn dot<T: Real>(a: &[T; 4], b: &[T; 4]) -> T {
    a[0].clone() * &b[0] - a[1].clone() * &b[1] - a[2].clone() * &b[2] - a[3].clone() * &b[3]
}

/// `MuShifted` with every power `t^rank..t^0` reconstructs `N(t v0 + v1,
/// t² v0² - beta)` exactly, lane by lane; the leading three are also returned
/// alone and in that order.
fn mu_shifted_in<T>(prototype: T, tolerance: f64)
where
    T: Real + SingleFloat + RealLike,
    Complex<T>: Real + EvaluationDomain,
{
    let rank = 4;
    let (expression, parameters, terms) = dense(rank);
    let mut numerator =
        PolynomialNumerator::new(expression, variables(), parameters, prototype.clone()).unwrap();
    let mut rng = Rng(0x5ead);
    let lanes = 2;
    let values: Vec<_> = (0..lanes * terms.len())
        .map(|_| convert(&prototype, rng.complex()))
        .collect();
    numerator.set_parameter_batch(&values, lanes).unwrap();
    let all = ExpansionKind::MuShifted {
        powers: (0..=rank as i32).rev().collect(),
    };
    let leading = ExpansionKind::MuShifted {
        powers: vec![rank as i32, rank as i32 - 1, rank as i32 - 2],
    };
    assert_eq!(all.len(), rank as usize + 1);
    for row in 0..3 {
        let v0: [Complex<T>; 4] = rng.vector().map(|z| convert(&prototype, z));
        let v1: [Complex<T>; 4] = rng.vector().map(|z| convert(&prototype, z));
        let beta = convert(&prototype, rng.complex());
        let vectors = [momentum(v0.clone()), momentum(v1.clone())];
        let parameters = [beta.clone()];
        let lane = row % lanes;
        let request = [ExpansionRequest {
            id: SampleId { lane, sample: row },
            partition: &[0, 1, 2, 3],
            vectors: &vectors,
            parameters: &parameters,
        }];
        let zero = Complex::new(prototype.zero(), prototype.zero());
        let mut coefficients = vec![zero.clone(); all.len()];
        numerator
            .expansion_batch(&all, &request, &mut coefficients)
            .unwrap();
        let mut three = vec![zero.clone(); 3];
        numerator
            .expansion_batch(&leading, &request, &mut three)
            .unwrap();
        // Separately compiled evaluators may round differently.
        for (a, b) in three.iter().zip(&coefficients) {
            let difference = (a.clone() - b).norm().re.to_f64();
            let scale = b.norm().re.to_f64().max(1.);
            assert!(
                difference <= tolerance * scale,
                "leading MuShifted powers: {a} vs {b}"
            );
        }
        let v00 = dot(&v0, &v0);
        for t in [0.75, -1.25, 2.5] {
            let t = convert(&prototype, C64::real(t));
            let q: [Complex<T>; 4] = std::array::from_fn(|i| t.clone() * &v0[i] + &v1[i]);
            let mu2 = t.clone() * &t * &v00 - &beta;
            let q = momentum(q);
            let direct = [NumeratorRequest {
                id: SampleId { lane, sample: 0 },
                q: &q,
                mu_squared: &mu2,
                partition: &[0, 1, 2, 3],
            }];
            let mut value = vec![zero.clone()];
            numerator.evaluate_batch(&direct, &mut value).unwrap();
            let mut sum = zero.clone();
            let mut power = Complex::new(prototype.one(), prototype.zero());
            for coefficient in coefficients.iter().rev() {
                sum += coefficient.clone() * &power;
                power *= &t;
            }
            let difference = (sum - &value[0]).norm().re.to_f64();
            let scale = value[0].norm().re.to_f64().max(1.);
            assert!(
                difference <= tolerance * scale,
                "row {row}: MuShifted {difference:e} from direct substitution"
            );
        }
    }
}

#[test]
fn mu_shifted_matches_direct_substitution_in_binary64_and_float() {
    mu_shifted_in(0f64, 1e-13);
    mu_shifted_in(Float::new(200), 1e-50);
}

#[test]
fn mu_shifted_matches_direct_substitution_for_every_dual_component() {
    // Value and first derivatives with respect to beta and one component of v0.
    let shape = vec![vec![0, 0], vec![1, 0], vec![0, 1]];
    let prototype = HyperDual::from_values(shape.clone(), vec![Complex::new(0., 0.); 3]);
    let dual = |value: C64, d_beta: f64, d_v: f64| {
        HyperDual::from_values(
            shape.clone(),
            vec![
                value.to_symbolica(),
                Complex::new(d_beta, 0.),
                Complex::new(d_v, 0.),
            ],
        )
    };
    let constant = |value: C64| dual(value, 0., 0.);
    let rank = 4;
    let (expression, parameters, terms) = dense(rank);
    let source = PolynomialNumerator::new(expression, variables(), parameters, 0.).unwrap();
    let mut numerator = DualPolynomialNumerator::new(source, prototype.clone()).unwrap();
    let mut rng = Rng(0xd0a1);
    let values: Vec<_> = (0..terms.len()).map(|_| constant(rng.complex())).collect();
    numerator.set_parameter_batch(&values, 1).unwrap();
    let mut v0: [_; 4] = rng.vector().map(constant);
    v0[1] = dual(rng.complex(), 0., 1.);
    let v1: [_; 4] = rng.vector().map(constant);
    let beta = dual(rng.complex(), 1., 0.);
    let kind = ExpansionKind::MuShifted {
        powers: (0..=rank as i32).rev().collect(),
    };
    let vectors = [
        FourMomentum::from_args(v0[0].clone(), v0[1].clone(), v0[2].clone(), v0[3].clone()),
        FourMomentum::from_args(v1[0].clone(), v1[1].clone(), v1[2].clone(), v1[3].clone()),
    ];
    let parameters = [beta.clone()];
    let request = [ExpansionRequest {
        id: SampleId { lane: 0, sample: 0 },
        partition: &[0, 1, 2, 3],
        vectors: &vectors,
        parameters: &parameters,
    }];
    let mut coefficients = vec![prototype.clone(); kind.len()];
    numerator
        .expansion_batch(&kind, &request, &mut coefficients)
        .unwrap();
    let v00 = dot(&v0, &v0);
    for t in [0.5, -1.5] {
        let t = constant(C64::real(t));
        let q: [_; 4] = std::array::from_fn(|i| t.clone() * &v0[i] + &v1[i]);
        let mu2 = t.clone() * &t * &v00 - &beta;
        let q = FourMomentum::from_args(q[0].clone(), q[1].clone(), q[2].clone(), q[3].clone());
        let direct = [NumeratorRequest {
            id: SampleId { lane: 0, sample: 0 },
            q: &q,
            mu_squared: &mu2,
            partition: &[0, 1, 2, 3],
        }];
        let mut value = vec![prototype.clone()];
        numerator.evaluate_batch(&direct, &mut value).unwrap();
        let mut sum = prototype.clone();
        let mut power = constant(C64::real(1.));
        for coefficient in coefficients.iter().rev() {
            sum += coefficient.clone() * &power;
            power *= &t;
        }
        for (component, (a, b)) in sum.values().iter().zip(value[0].values()).enumerate() {
            let difference = (*a - *b).norm().re;
            assert!(
                difference <= 1e-12 * b.norm().re.max(1.),
                "Dual component {component}: {a} vs {b}"
            );
        }
        assert!(
            value[0].values()[1].norm().re > 1e-6 && value[0].values()[2].norm().re > 1e-6,
            "the seeds must reach the numerator"
        );
    }
}

/// The 1-vector `Mu` padding is not extended to `MuShifted`, which needs both
/// vectors and `beta`.
#[test]
fn mu_shifted_requires_two_vectors_and_beta() {
    let mut numerator =
        PolynomialNumerator::new(parse("q0^2+mu2"), variables(), vec![], 0.).unwrap();
    numerator.set_parameter_batch(&[], 1).unwrap();
    let c = |x: f64| Complex::new(x, 0.);
    let v = FourMomentum::from_args(c(2.), c(1.), c(0.), c(0.));
    let w = FourMomentum::from_args(c(3.), c(0.), c(0.), c(0.));
    let kind = ExpansionKind::MuShifted {
        powers: vec![2, 1, 0],
    };
    let mut out = [c(0.); 3];
    for (vectors, parameters) in [(&[v][..], &[c(0.5)][..]), (&[v, w][..], &[][..])] {
        let request = [ExpansionRequest {
            id: SampleId { lane: 0, sample: 0 },
            partition: &[],
            vectors,
            parameters,
        }];
        assert!(
            numerator
                .expansion_batch(&kind, &request, &mut out)
                .is_err()
        );
    }
    // q = t v + w, mu² = 3 t² - beta: q0² + mu² = (2t + 3)² + 3t² - 1/2.
    let vectors = [v, w];
    let request = [ExpansionRequest {
        id: SampleId { lane: 0, sample: 0 },
        partition: &[],
        vectors: &vectors,
        parameters: &[c(0.5)],
    }];
    numerator
        .expansion_batch(&kind, &request, &mut out)
        .unwrap();
    assert_eq!(out, [c(7.), c(12.), c(8.5)]);
}

// ---------------------------------------------------------------------------
// GenericNumerator layouts.

fn generic_numerator(rank: u32, seed: u64) -> GenericNumerator<PolynomialNumerator<f64>> {
    let (expression, parameters, terms) = dense(rank);
    let mut numerator = PolynomialNumerator::new(expression, variables(), parameters, 0.).unwrap();
    let mut rng = Rng(seed);
    let values: Vec<_> = (0..2 * terms.len())
        .map(|_| rng.complex().to_symbolica())
        .collect();
    numerator.set_parameter_batch(&values, 2).unwrap();
    GenericNumerator::with_rank(numerator, rank as usize)
}

fn direct(
    numerator: &mut GenericNumerator<PolynomialNumerator<f64>>,
    lane: usize,
    q: CMom,
    mu2: C64,
) -> C64 {
    let mut out = [C64::ZERO];
    numerator
        .evaluate(&[0, 1], &[lane], &[q], &[mu2], &mut out)
        .unwrap();
    out[0]
}

fn axpy(terms: &[(C64, &CMom)]) -> CMom {
    std::array::from_fn(|i| {
        terms
            .iter()
            .map(|(a, v)| *a * v[i])
            .fold(C64::ZERO, |s, z| s + z)
    })
}

#[test]
fn t2_order_four_layout_matches_original_ninja_extension() {
    assert_eq!(t2_len(4), 22);
    assert_eq!(
        t2_powers(7, 4)[13..],
        [
            (3, 0, 0),
            (3, 0, 1),
            (3, 0, 2),
            (3, 1, 0),
            (3, 1, 1),
            (3, 2, 0),
            (3, 2, 1),
            (3, 3, 0),
            (3, 4, 0)
        ]
    );
    for mindeg in 0..=3 {
        // Unchanged rank <= n+1 layouts.
        let expected: &[&[(u16, u16)]] = &[
            &[(0, 0)],
            &[(0, 0), (1, 0)],
            &[(0, 0), (0, 1), (1, 0), (2, 0)],
            &[(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (3, 0)],
        ];
        let flat: Vec<_> = expected[..=mindeg]
            .iter()
            .enumerate()
            .flat_map(|(i, order)| order.iter().map(move |&(x, m)| (5 - i as i32, x, m)))
            .collect();
        assert_eq!(t2_powers(5, mindeg), flat);
    }
    assert_eq!(t3_len(5), 12);
    assert_eq!(
        t3_powers(5, 5)[9..],
        [(0, 0), (0, 1), (0, 2)],
        "level 5 of the rank-5 T3 layout"
    );
}

/// At rank 2 the mindeg-4 `T2` expansion (levels `t^2..t^-2`) is the whole
/// Laurent polynomial, so the 22 entries must reproduce the numerator on the
/// bubble-cut parametrisation `q = a0 + x a1 + t e3 + (p0 + p1 x + p2 x² +
/// mu²)/t e4` exactly; likewise the mindeg-5 `T3` expansion at rank 2 with a
/// vanishing level 5.
#[test]
fn generic_numerator_order_four_t2_and_order_five_t3_match_direct_evaluation() {
    let mut numerator = generic_numerator(2, 0x7a1);
    let mut rng = Rng(0xbeef);
    let rows = 2;
    let lanes = [1, 0];
    let a0: Vec<CMom> = (0..rows).map(|_| rng.vector()).collect();
    let a1: Vec<CMom> = (0..rows).map(|_| rng.vector()).collect();
    let e3: Vec<CMom> = (0..rows).map(|_| rng.vector()).collect();
    let e4: Vec<CMom> = (0..rows).map(|_| rng.vector()).collect();
    let param: Vec<[C64; 3]> = (0..rows)
        .map(|_| [rng.complex(), rng.complex(), rng.complex()])
        .collect();
    let mut t2 = vec![C64::ZERO; rows * t2_len(4)];
    numerator
        .t2_expansion(&[0, 1], &lanes, &a0, &a1, &e3, &e4, &param, 4, &mut t2)
        .unwrap();
    let powers = t2_powers(2, 4);
    for row in 0..rows {
        for (t, x, mu) in [
            (0.7f64, 1.3f64, -0.4f64),
            (-1.1, 0.25, 0.9),
            (1.9, -0.6, 0.35),
        ] {
            let [p0, p1, p2] = param[row];
            let s = (p0 + p1 * x + p2 * (x * x) + mu) / t;
            let q = axpy(&[
                (C64::real(1.), &a0[row]),
                (C64::real(x), &a1[row]),
                (C64::real(t), &e3[row]),
                (s, &e4[row]),
            ]);
            let expected = direct(&mut numerator, lanes[row], q, C64::real(mu));
            let mut sum = C64::ZERO;
            for (k, &(pt, px, pm)) in powers.iter().enumerate() {
                sum += t2[row * t2_len(4) + k]
                    * (t.powi(pt) * x.powi(i32::from(px)) * mu.powi(i32::from(pm)));
            }
            assert!(
                (sum - expected).abs() <= 1e-12 * expected.abs().max(1.),
                "T2 order 4 row {row}: {sum:?} vs {expected:?}"
            );
        }
    }
    let beta: Vec<C64> = (0..rows).map(|_| rng.complex()).collect();
    let mut t3 = vec![C64::ZERO; rows * t3_len(5)];
    numerator
        .t3_expansion(&[0, 1, 2], &lanes, &a0, &e3, &e4, &beta, 5, &mut t3)
        .unwrap();
    let powers = t3_powers(2, 5);
    for row in 0..rows {
        assert!(
            t3[row * 12 + 9..row * 12 + 12]
                .iter()
                .all(|z| z.abs() < 1e-13)
        );
        for (t, mu) in [(0.8f64, -0.3f64), (-1.4, 0.6)] {
            let q = axpy(&[
                (C64::real(1.), &a0[row]),
                (C64::real(t), &e3[row]),
                ((beta[row] + mu) / t, &e4[row]),
            ]);
            let expected = direct(&mut numerator, lanes[row], q, C64::real(mu));
            let sum = powers
                .iter()
                .enumerate()
                .map(|(k, &(pt, pm))| t3[row * 12 + k] * (t.powi(pt) * mu.powi(i32::from(pm))))
                .fold(C64::ZERO, |s, z| s + z);
            assert!(
                (sum - expected).abs() <= 1e-12 * expected.abs().max(1.),
                "T3 order 5 row {row}: {sum:?} vs {expected:?}"
            );
        }
    }
    assert!(matches!(
        numerator.t2_expansion(&[0, 1], &lanes, &a0, &a1, &e3, &e4, &param, 5, &mut t2),
        Err(Error::Expansion(_))
    ));
}

/// `mu_expansion_x2` returns the `t^r, t^(r-1), t^(r-2)` coefficients of
/// `N(t v0 + v1, t² v0² - beta)`: checked against the polynomial in `t`
/// interpolated from direct evaluations (degree `r`).
#[test]
fn generic_numerator_mu_expansion_x2_matches_direct_evaluation() {
    let rank = 4usize;
    let mut numerator = generic_numerator(rank as u32, 0x3b);
    let mut rng = Rng(0xfeed);
    let lanes = [0, 1, 0];
    let v0: Vec<CMom> = lanes.iter().map(|_| rng.vector()).collect();
    let v1: Vec<CMom> = lanes.iter().map(|_| rng.vector()).collect();
    let beta: Vec<C64> = lanes.iter().map(|_| rng.complex()).collect();
    let mut out = vec![C64::ZERO; 3 * lanes.len()];
    numerator
        .mu_expansion_x2(&[0, 1, 2, 3], &lanes, &v0, &v1, &beta, &mut out)
        .unwrap();
    for (row, &lane) in lanes.iter().enumerate() {
        // Newton interpolation through t = 0..=rank, then the monomial
        // coefficients of the top three powers.
        let nodes: Vec<f64> = (0..=rank).map(|k| k as f64).collect();
        let values: Vec<C64> = nodes
            .iter()
            .map(|&t| {
                let t = C64::real(t);
                let q = axpy(&[(t, &v0[row]), (C64::real(1.), &v1[row])]);
                let v00 = ninjars::fast::mp_cc(&v0[row], &v0[row]);
                direct(&mut numerator, lane, q, t * t * v00 - beta[row])
            })
            .collect();
        let mut divided = values.clone();
        for level in 1..=rank {
            for k in (level..=rank).rev() {
                divided[k] = (divided[k] - divided[k - 1]) / C64::real(nodes[k] - nodes[k - level]);
            }
        }
        // Expand the Newton form into monomial coefficients.
        let mut monomial = vec![C64::ZERO; rank + 1];
        for k in (0..=rank).rev() {
            // monomial = monomial * (t - nodes[k]) + divided[k]
            let mut next = vec![C64::ZERO; rank + 1];
            for (power, &c) in monomial.iter().enumerate() {
                if power < rank {
                    next[power + 1] += c;
                }
                next[power] -= c * C64::real(nodes[k]);
            }
            next[0] += divided[k];
            monomial = next;
        }
        for (k, &expected) in [monomial[rank], monomial[rank - 1], monomial[rank - 2]]
            .iter()
            .enumerate()
        {
            let got = out[3 * row + k];
            assert!(
                (got - expected).abs() <= 1e-9 * expected.abs().max(1.),
                "row {row} t^{}: {got:?} vs {expected:?}",
                rank - k
            );
        }
    }
}

/// Implementors that predate rank n+2 keep compiling and report the missing
/// callback instead of silently returning zeros.
#[test]
fn mu_expansion_x2_defaults_to_unsupported() {
    struct Legacy;
    impl FastNumerator for Legacy {
        fn rank(&self) -> usize {
            0
        }
        fn evaluate(
            &mut self,
            _: &[usize],
            _: &[usize],
            _: &[CMom],
            _: &[C64],
            _: &mut [C64],
        ) -> ninjars::Result<()> {
            Ok(())
        }
        fn mu_expansion(
            &mut self,
            _: &[usize],
            _: &[usize],
            _: &[CMom],
            _: Option<&[CMom]>,
            _: &mut [C64],
        ) -> ninjars::Result<()> {
            Ok(())
        }
        fn t3_expansion(
            &mut self,
            _: &[usize],
            _: &[usize],
            _: &[CMom],
            _: &[CMom],
            _: &[CMom],
            _: &[C64],
            _: usize,
            _: &mut [C64],
        ) -> ninjars::Result<()> {
            Ok(())
        }
        fn t2_expansion(
            &mut self,
            _: &[usize],
            _: &[usize],
            _: &[CMom],
            _: &[CMom],
            _: &[CMom],
            _: &[CMom],
            _: &[[C64; 3]],
            _: usize,
            _: &mut [C64],
        ) -> ninjars::Result<()> {
            Ok(())
        }
    }
    let v = [[C64::ZERO; 4]];
    let mut out = [C64::ZERO; 3];
    assert!(matches!(
        Legacy.mu_expansion_x2(&[0], &[0], &v, &v, &[C64::ZERO], &mut out),
        Err(Error::Unsupported(_))
    ));
}

// ---------------------------------------------------------------------------
// Masters.

fn b1111_fixture_rows() -> [([C64; 4], C64); 2] {
    [
        // oneloopmaster src/tests.rs rank_four_bubbles_match_existing_reference.
        (
            [
                C64::real(3.),
                C64::new(0.7, -0.03),
                C64::new(1.4, -0.08),
                C64::real(1.),
            ],
            C64::new(0.029_939_862_301_517_683, 0.017_531_313_325_083_266),
        ),
        // B1111(0; 2, 5) = int x⁴ [1/eps - log(2 + 3x)] dx (the corrected
        // OneLOop mass-swap sign).
        (
            [C64::real(0.), C64::real(2.), C64::real(5.), C64::real(1.)],
            C64::real(-0.299_847_502_997_086_6),
        ),
    ]
}

#[test]
fn exact_b1111_matches_oneloopmaster_references() {
    assert_eq!(MasterIntegral::B1111.invariant_count(), 1);
    assert_eq!(MasterIntegral::B1111.mass_count(), 2);
    assert_eq!(MasterIntegral::B1111.argument_count(), 4);
    let mut masters = SymbolicaMasters::new(0f64);
    for (args, finite) in b1111_fixture_rows() {
        let invariants = [args[0].to_symbolica()];
        let masses = [args[1].to_symbolica(), args[2].to_symbolica()];
        let scales = [args[3].to_symbolica()];
        let mut output: Vec<Laurent<Complex<f64>>> = vec![[Complex::new(0., 0.); 3]];
        let mut statuses = [MasterStatus::Success];
        masters
            .evaluate_batch(
                MasterRequest {
                    integral: MasterIntegral::B1111,
                    invariants: &invariants,
                    masses_squared: &masses,
                    scales_squared: &scales,
                },
                &mut output,
                &mut statuses,
            )
            .unwrap();
        assert_eq!(statuses, [MasterStatus::Success]);
        let value = output[0].map(C64::from_symbolica);
        assert!((value[0] - finite).abs() < 1e-12, "{args:?}: {value:?}");
        assert!((value[1] - C64::real(0.2)).abs() < 1e-13);
        assert_eq!(value[2], C64::ZERO);
    }
    // Float carriers use the same exact expressions.
    let p = Float::new(200);
    let mut masters = SymbolicaMasters::new(p.clone());
    let (args, finite) = b1111_fixture_rows()[1];
    let mut output = vec![[
        Complex::new(p.clone(), p.clone()),
        Complex::new(p.clone(), p.clone()),
        Complex::new(p.clone(), p.clone()),
    ]];
    masters
        .evaluate(
            MasterRequest {
                integral: MasterIntegral::B1111,
                invariants: &[convert(&p, args[0])],
                masses_squared: &[convert(&p, args[1]), convert(&p, args[2])],
                scales_squared: &[convert(&p, args[3])],
            },
            &mut output,
        )
        .unwrap();
    assert!((output[0][0].re.to_f64() - finite.re).abs() < 1e-15);
}

#[test]
fn bubble_rank4_family_through_generic_masters() {
    assert_eq!(FastMaster::BubbleRank4.arity(), 4);
    assert_eq!(FastMaster::BubbleRank4.outputs(), 5);
    let mut generic = GenericMasters::new(SymbolicaMasters::new(0f64));
    let rows: Vec<[C64; 4]> = b1111_fixture_rows().iter().map(|r| r.0).collect();
    let args: Vec<C64> = rows.iter().flatten().copied().collect();
    let mut rank4 = vec![[C64::ZERO; 3]; rows.len() * 5];
    let mut ok = vec![false; rows.len()];
    generic
        .evaluate(FastMaster::BubbleRank4, &args, &mut rank4, &mut ok)
        .unwrap();
    assert!(ok.iter().all(|&ok| ok));
    let mut rank3 = vec![[C64::ZERO; 3]; rows.len() * 4];
    generic
        .evaluate(FastMaster::BubbleRank3, &args, &mut rank3, &mut ok)
        .unwrap();
    for (row, (_, finite)) in b1111_fixture_rows().iter().enumerate() {
        // [B0, B1, B11, B111] bitwise as the rank-3 family, then B1111.
        assert_eq!(rank4[row * 5..row * 5 + 4], rank3[row * 4..row * 4 + 4]);
        assert!((rank4[row * 5 + 4][0] - *finite).abs() < 1e-12);
    }
}

#[test]
fn native_masters_reject_b1111_until_the_native_family_exists() {
    let mut native = OneLoopNativeMasters::new().unwrap();
    let args = b1111_fixture_rows()[0].0;
    let mut out = vec![[C64::ZERO; 3]; 5];
    let mut ok = [false];
    assert!(matches!(
        FastMasters::evaluate(
            &mut native,
            FastMaster::BubbleRank4,
            &args,
            &mut out,
            &mut ok
        ),
        Err(Error::Unsupported(_))
    ));
    let mut output = vec![[Complex::new(0., 0.); 3]];
    let mut statuses = [MasterStatus::Success];
    let request = MasterRequest {
        integral: MasterIntegral::B1111,
        invariants: &[args[0].to_symbolica()],
        masses_squared: &[args[1].to_symbolica(), args[2].to_symbolica()],
        scales_squared: &[args[3].to_symbolica()],
    };
    assert!(matches!(
        MasterProvider::evaluate_batch(&mut native, request, &mut output, &mut statuses),
        Err(Error::Unsupported(_))
    ));
    assert_eq!(output[0], [Complex::new(0., 0.); 3], "outputs unchanged");
}

// ---------------------------------------------------------------------------

/// The generic engine accepts rank n+2 (S2); the fast engine still rejects it
/// until its rank n+2 engine lands. Rank n+3 is rejected everywhere.
#[test]
fn rank_n_plus_two_is_generic_only() {
    for n in 1..=7 {
        assert_eq!(RankClass::of(n, n + 1), Some(RankClass::NPlus1));
        assert_eq!(RankClass::of(n, n + 2), Some(RankClass::NPlus2));
        assert!(Topology::new(n, n + 1).is_ok());
        assert_eq!(
            Topology::new(n, n + 2).unwrap().rank_class(),
            RankClass::NPlus2
        );
        assert!(matches!(Topology::new(n, n + 3), Err(Error::Rank { .. })));
        assert!(FastAmplitude::new(n, n + 2, FastOptions::default()).is_err());
        assert!(SymbolicaAmplitudeKernels::new(n, n + 2, 0f64).is_ok());
        assert!(SymbolicaAmplitudeKernels::new(n, n + 3, 0f64).is_err());
        for cut in 1..=n.min(3) {
            assert!(LaurentDivision::new(n, n + 2, cut).is_ok());
            assert!(LaurentDivision::new(n, n + 3, cut).is_err());
        }
    }
}
