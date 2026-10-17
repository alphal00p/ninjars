//! Dense tensor-numerator coefficients in original Ninja's layout.
//!
//! The reference here is an independent literal port of `tensor_ninja.cc`
//! (`ten_build`, `ten_from_mom`, `ten_contract` and their coefficient-count
//! tables), i.e. exactly what `ninja::TensorNumerator::evaluate` computes.

use ninjars::fast::tensor_layout;
use ninjars_graph::{CoefficientError, PolynomialCoefficientPlan, TensorCoefficientPlan};
use symbolica::{
    atom::{Atom, AtomCore},
    domains::float::{Complex, Real},
    parse,
};

type C = Complex<f64>;

// tensor_ninja.cc tables, copied verbatim.
const TEN_NCOEFF: [usize; 22] = [
    0, 1, 5, 15, 35, 70, 126, 210, 330, 495, 715, 1001, 1365, 1820, 2380, 3060, 3876, 4845, 5985,
    7315, 8855, 10626,
];
const TEN_NCOEFF_R: [usize; 21] = [
    1, 4, 10, 20, 35, 56, 84, 120, 165, 220, 286, 364, 455, 560, 680, 816, 969, 1140, 1330, 1540,
    1771,
];
const COMP2: [usize; 21] = [
    0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136, 153, 171, 190, 210,
];
const COMP3: [usize; 21] = [
    0, 1, 4, 10, 20, 35, 56, 84, 120, 165, 220, 286, 364, 455, 560, 680, 816, 969, 1140, 1330, 1540,
];

fn ten_ncoeff(rank: usize) -> usize {
    TEN_NCOEFF[rank + 1]
}

// Literal ports keep tensor_ninja.cc's index loops.
#[allow(clippy::needless_range_loop)]
fn ten_build<T: Copy + std::ops::Mul<Output = T>>(rank: usize, t1: &[T], v: &[T; 4], t0: &mut [T]) {
    let mut j = 1;
    t0[0] = t1[0] * v[0];
    for i in 0..rank {
        t0[j] = v[1] * t1[i];
        j += 1;
    }
    for i in 0..COMP2[rank] {
        t0[j] = v[2] * t1[i];
        j += 1;
    }
    for i in 0..COMP3[rank] {
        t0[j] = v[3] * t1[i];
        j += 1;
    }
}

#[allow(clippy::needless_range_loop)]
fn ten_from_mom<T: Copy + std::ops::Mul<Output = T>>(rank: usize, v: &[T; 4], one: T) -> Vec<T> {
    let mut t = vec![one; ten_ncoeff(rank)];
    let mut lower = 0;
    let mut current = 1;
    for r in 1..=rank {
        let (head, tail) = t.split_at_mut(current);
        ten_build(r, &head[lower..], v, tail);
        lower = current;
        current += TEN_NCOEFF_R[r];
    }
    t
}

fn ten_contract(rank: usize, t1: &[C], t2: &[C]) -> C {
    (0..ten_ncoeff(rank)).fold(C::new(0., 0.), |sum, i| sum + t1[i] * t2[i])
}

/// Deterministic xorshift64* stream in [-1, 1).
struct Stream(u64);
impl Stream {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11) as f64 / (1u64 << 52) as f64 - 1.
    }
    fn integer(&mut self, bound: i64) -> i64 {
        (((self.next() + 1.) * 0.5 * (2 * bound + 1) as f64) as i64 - bound).clamp(-bound, bound)
    }
    fn complex(&mut self) -> C {
        C::new(self.next(), self.next())
    }
}

fn variables() -> [Atom; 5] {
    ["q0", "q1", "q2", "q3", "mu2"].map(|name| Atom::var(symbolica::symbol!(name)))
}

#[test]
fn layout_matches_ten_from_mom_exactly() {
    // Distinct primes identify every monomial by unique factorization, so the
    // exact binary64 products prove the slot ordering of ten_build.
    let primes: [f64; 4] = [2., 3., 5., 7.];
    for rank in 0..=8 {
        let tensor = ten_from_mom(rank, &primes, 1.);
        let monomials = tensor_layout::monomials(rank);
        assert_eq!(tensor.len(), tensor_layout::ncoeff(rank));
        assert_eq!(monomials.len(), tensor.len());
        for (slot, exponents) in monomials.iter().enumerate() {
            let value = exponents
                .iter()
                .zip(primes)
                .fold(1., |product, (power, prime)| {
                    product * prime.powi(*power as i32)
                });
            assert_eq!(tensor[slot], value, "rank {rank} slot {slot}");
            assert_eq!(tensor_layout::monomial_index(*exponents), slot);
        }
    }
}

/// Exponents and integer coefficients `(k_a, k_b, k_1)` of `k_a a + k_b b + k_1`.
type Terms = Vec<([u32; 4], [i64; 3])>;

/// Random rank-`rank` polynomial with parameter-dependent coefficients,
/// structural zeros and repeated coefficient heads.
fn random_polynomial(rank: usize, stream: &mut Stream) -> (Atom, Terms) {
    let [q0, q1, q2, q3, _] = variables();
    let (a, b) = (parse!("a"), parse!("b"));
    let mut terms = Vec::new();
    let mut expression = Atom::Zero;
    let mut previous = [1, 0, 0];
    for exponents in tensor_layout::monomials(rank) {
        let roll = stream.next();
        let coefficient = if roll < -0.6 {
            continue;
        } else if roll < -0.3 {
            previous
        } else {
            [stream.integer(9), stream.integer(9), stream.integer(9)]
        };
        if coefficient == [0, 0, 0] {
            continue;
        }
        previous = coefficient;
        let head = &a * Atom::num(coefficient[0])
            + &b * Atom::num(coefficient[1])
            + Atom::num(coefficient[2]);
        let monomial = [&q0, &q1, &q2, &q3]
            .iter()
            .zip(exponents)
            .fold(Atom::num(1), |product, (q, power)| {
                product * q.pow(Atom::num(power as i64))
            });
        expression += head * monomial;
        terms.push((exponents, coefficient));
    }
    // Keep the full degree so the tensor rank equals `rank`.
    let top = [rank as u32, 0, 0, 0];
    if !terms.iter().any(|(exponents, _)| *exponents == top) {
        expression += &a * q0.pow(Atom::num(rank as i64));
        terms.push((top, [1, 0, 0]));
    }
    (expression, terms)
}

#[test]
fn scattered_tensor_reproduces_the_polynomial_through_ninjas_contraction() {
    let mut stream = Stream(0x5eed_1234_abcd_0042);
    for rank in 0..=5 {
        let (expression, terms) = random_polynomial(rank, &mut stream);
        let plan =
            PolynomialCoefficientPlan::new(expression, variables(), vec![parse!("a"), parse!("b")])
                .unwrap();
        let tensor_plan = TensorCoefficientPlan::new(&plan).unwrap();
        assert_eq!(tensor_plan.rank(), rank);
        assert_eq!(tensor_plan.ncoeff(), ten_ncoeff(rank));
        assert_eq!(tensor_plan.entries().len(), terms.len());
        assert_eq!(
            tensor_plan.entries().len() + tensor_plan.zero_slots().len(),
            tensor_plan.ncoeff()
        );
        let lanes = 3;
        let parameters: Vec<C> = (0..2 * lanes).map(|_| stream.complex()).collect();
        let mut heads = vec![C::new(0., 0.); lanes * tensor_plan.head_count()];
        plan.compile(0.)
            .unwrap()
            .evaluate_batch(&parameters, &mut heads, lanes)
            .unwrap();
        let mut tensor = vec![C::new(f64::NAN, f64::NAN); lanes * tensor_plan.ncoeff()];
        tensor_plan
            .scatter(&heads, &mut tensor, lanes, &C::new(0., 0.))
            .unwrap();
        for lane in 0..lanes {
            let row = &tensor[lane * tensor_plan.ncoeff()..(lane + 1) * tensor_plan.ncoeff()];
            assert!(row.iter().all(|c| c.re.is_finite() && c.im.is_finite()));
            for &slot in tensor_plan.zero_slots() {
                assert_eq!(row[slot], C::new(0., 0.));
            }
            let (a, b) = (parameters[2 * lane], parameters[2 * lane + 1]);
            for _ in 0..8 {
                let q = [(); 4].map(|_| stream.complex() * C::new(3., 0.));
                let ninja = ten_contract(rank, row, &ten_from_mom(rank, &q, C::new(1., 0.)));
                let mut scale = 0_f64;
                let direct = terms.iter().fold(C::new(0., 0.), |sum, (exponents, k)| {
                    let coefficient = a * C::new(k[0] as f64, 0.)
                        + b * C::new(k[1] as f64, 0.)
                        + C::new(k[2] as f64, 0.);
                    let term = q
                        .iter()
                        .zip(exponents)
                        .fold(coefficient, |product, (q, power)| {
                            (0..*power).fold(product, |product, _| product * *q)
                        });
                    scale += term.norm().re;
                    sum + term
                });
                assert!(
                    (ninja - direct).norm().re <= 1e-13 * scale,
                    "rank {rank} lane {lane}: {ninja} != {direct}"
                );
            }
        }
    }
}

#[test]
fn explicit_mu_squared_and_insufficient_rank_are_rejected() {
    let plan = PolynomialCoefficientPlan::new(
        parse!("a*q0^2+b*q1*mu2+3"),
        variables(),
        vec![parse!("a"), parse!("b")],
    )
    .unwrap();
    let error = TensorCoefficientPlan::new(&plan).unwrap_err();
    assert!(matches!(
        error,
        CoefficientError::ExplicitMuSquared { power: 1 }
    ));
    assert!(error.to_string().contains("mu^2"));

    let plan = PolynomialCoefficientPlan::new(
        parse!("a*q0^2*q3+b*q1+3"),
        variables(),
        vec![parse!("a"), parse!("b")],
    )
    .unwrap();
    assert!(matches!(
        TensorCoefficientPlan::with_rank(&plan, 2).unwrap_err(),
        CoefficientError::TensorRank {
            requested: 2,
            required: 3
        }
    ));
    // A larger rank pads the higher homogeneous blocks with zeros.
    let padded = TensorCoefficientPlan::with_rank(&plan, 5).unwrap();
    assert_eq!(padded.ncoeff(), 126);
    assert_eq!(padded.entries().len(), 3);
    assert_eq!(padded.zero_slots().len(), 123);
    let slots: Vec<_> = padded.entries().iter().map(|entry| entry.slot).collect();
    let expected = [[0, 0, 0, 0], [0, 1, 0, 0], [2, 0, 0, 1]].map(tensor_layout::monomial_index);
    assert_eq!(slots, expected);
}

#[test]
fn scatter_validates_shapes_and_overwrites_every_slot() {
    let plan = PolynomialCoefficientPlan::new(
        parse!("a*q0+a*q2+b"),
        variables(),
        vec![parse!("a"), parse!("b")],
    )
    .unwrap();
    let tensor_plan = TensorCoefficientPlan::new(&plan).unwrap();
    // Two monomials share the head `a`.
    assert_eq!(tensor_plan.head_count(), 2);
    assert_eq!(tensor_plan.ncoeff(), 5);
    let heads = [10., 20., 30., 40.];
    let mut tensor = [f64::NAN; 10];
    tensor_plan.scatter(&heads, &mut tensor, 2, &0.).unwrap();
    let a = plan
        .coefficient_expressions()
        .iter()
        .position(|head| *head == parse!("a"))
        .unwrap();
    let (a0, b0) = (heads[a], heads[1 - a]);
    let (a1, b1) = (heads[2 + a], heads[3 - a]);
    assert_eq!(tensor, [b0, a0, 0., a0, 0., b1, a1, 0., a1, 0.]);
    for (heads_len, tensor_len) in [(3, 10), (4, 9)] {
        assert!(matches!(
            tensor_plan.scatter(&heads[..heads_len], &mut tensor[..tensor_len], 2, &0.),
            Err(CoefficientError::Shape { .. })
        ));
    }
    tensor_plan.scatter::<f64>(&[], &mut [], 0, &0.).unwrap();
}
