//! Ultraviolet residue of a one-loop tensor integral, read off its numerator.
//!
//! [`FastAmplitude`](super::FastAmplitude) drops scaleless-bubble coefficients
//! (their combined pole `1/ε_UV − 1/ε_IR` vanishes), so it cannot separate UV
//! and IR poles at the master boundary and refuses selected providers. The
//! ultraviolet residue, however, does not need the reduction at all: it is
//! fixed by the integrand's large-`q` behaviour. For
//! `μ^(2ε)/(iπ^(d/2) r_Γ) ∫ d^d q N(q) / Π_i D_i` with
//! `D_i = (q + V_i)² − m_i²`, `c_i = V_i² − m_i²` and a four-dimensional
//! tensor numerator `N(q) = Σ_α t_α q^α` ([`super::tensor_layout`]),
//!
//! ```text
//! 1/D_i = Σ_k (−1)^k (2 q·V_i + c_i)^k / (q²)^(k+1),
//! ```
//!
//! and the terms of total degree −4 give `1/ε_UV` times
//!
//! ```text
//! ⟨q^β⟩ = Π_μ (β_μ − 1)!! g_μμ^(β_μ/2) / (2^k (k+1)!)   (|β| = 2k, every β_μ even),
//! ```
//!
//! the residue of `∫ q^β / (q²)^(k+2)`. The degree-`r` block of `N`
//! contributes when `j = r − 2n + 4 ≥ 0`, through the terms with `a_i` factors
//! `2 q·V_i` and `b_i` factors `c_i`, `Σ_i (a_i + 2 b_i) = j`, weighted by
//! `(−1)^(a_i+b_i) C(a_i+b_i, a_i)`.
//!
//! The result equals the generic engine's `1/ε` coefficient with
//! oneloopmaster's `PoleSelection::UV` masters (A0 and its tensors, B0 and the
//! longitudinal bubble moments, scaleless bubbles included; C0, D0 and dB0
//! have none): `ninjars-symbolica/tests/tensor_uv_residue.rs`. With it a
//! consumer of the fast engine separates the poles of its combined result
//! without a second reduction and without any master evaluation:
//! `UV = tensor_uv_residue(..)` and `IR = combined − UV` (the double pole is
//! infrared; the dropped scaleless bubbles have `UV + IR = 0`, so their IR part
//! is `−UV` and is included). The finite part is untouched. Any carrier works
//! (binary64, double-double, arbitrary precision); the walk only multiplies
//! and adds, with exact integer weights.

use crate::{Complex, SingleFloat};

use super::tensor_layout::{block_offset, homogeneous_index, monomials, ncoeff};

/// Homogeneous polynomials of every degree up to `max` in the tensor layout:
/// monomials per degree, the index of `m · q_μ`, and the angular average
/// `⟨q^β⟩ = numerator / denominator` of the even degrees.
struct Layout {
    exponents: Vec<Vec<[u32; 4]>>,
    raise: Vec<Vec<[usize; 4]>>,
    average: Vec<(Vec<(usize, i64)>, i64)>,
}

impl Layout {
    fn new(max: usize) -> Self {
        let all = monomials(max);
        let exponents: Vec<Vec<[u32; 4]>> = (0..=max)
            .map(|d| all[block_offset(d)..ncoeff(d)].to_vec())
            .collect();
        let raise = (0..=max)
            .map(|d| {
                exponents[d]
                    .iter()
                    .map(|&m| {
                        std::array::from_fn(|mu| {
                            let mut up = m;
                            up[mu] += 1;
                            homogeneous_index(up)
                        })
                    })
                    .collect()
            })
            .collect();
        let average = (0..=max)
            .map(|d| {
                if d % 2 == 1 {
                    return (Vec::new(), 1);
                }
                let k = (d / 2) as i64;
                let denominator = (1..=k + 1).product::<i64>() << k;
                let terms = exponents[d]
                    .iter()
                    .enumerate()
                    .filter(|(_, m)| m.iter().all(|&e| e % 2 == 0))
                    .map(|(index, m)| {
                        let pairings: i64 = m
                            .iter()
                            .map(|&e| (1..e as i64).step_by(2).product::<i64>())
                            .product();
                        // g_00 = 1, g_ii = −1: (−1)^((β1 + β2 + β3)/2).
                        let sign = if ((m[1] + m[2] + m[3]) / 2) % 2 == 0 {
                            1
                        } else {
                            -1
                        };
                        (index, sign * pairings)
                    })
                    .collect();
                (terms, denominator)
            })
            .collect();
        Self {
            exponents,
            raise,
            average,
        }
    }
}

/// One integral's inputs to the residue walk.
struct Walk<'a, T: SingleFloat> {
    layout: &'a Layout,
    /// `w_μ = g_μμ V_i^μ` (`q·V_i = Σ_μ w_μ q^μ`) per propagator.
    covariant: Vec<[Complex<T>; 4]>,
    /// `c_i = V_i² − m_i²` per propagator.
    constants: Vec<Complex<T>>,
    zero: Complex<T>,
    total: Complex<T>,
}

fn is_zero<T: SingleFloat>(value: &Complex<T>) -> bool {
    value.re.is_zero() && value.im.is_zero()
}

impl<T: SingleFloat> Walk<'_, T> {
    fn integer(&self, value: i64) -> Complex<T> {
        Complex::new(self.zero.re.from_i64(value), self.zero.re.zero())
    }

    /// `polynomial · (q·V_i)` for a homogeneous polynomial of `degree`.
    fn raise(&self, polynomial: &[Complex<T>], degree: usize, i: usize) -> Vec<Complex<T>> {
        let mut out = vec![self.zero.clone(); self.layout.exponents[degree + 1].len()];
        for (value, targets) in polynomial.iter().zip(&self.layout.raise[degree]) {
            if is_zero(value) {
                continue;
            }
            for (target, w) in targets.iter().zip(&self.covariant[i]) {
                out[*target] += value.clone() * w;
            }
        }
        out
    }

    fn average(&self, polynomial: &[Complex<T>], degree: usize) -> Complex<T> {
        let (terms, denominator) = &self.layout.average[degree];
        let mut sum = self.zero.clone();
        for &(index, weight) in terms {
            sum += polynomial[index].clone() * &self.integer(weight);
        }
        sum * &(self.integer(1) / &self.integer(*denominator))
    }

    /// Distribute the remaining `j` over propagators `i..`.
    fn walk(
        &mut self,
        i: usize,
        remaining: usize,
        polynomial: &[Complex<T>],
        degree: usize,
        weight: Complex<T>,
    ) {
        if i == self.constants.len() {
            if remaining == 0 && degree.is_multiple_of(2) {
                let value = self.average(polynomial, degree);
                self.total += weight * &value;
            }
            return;
        }
        let shifted = self.covariant[i].iter().any(|w| !is_zero(w));
        let constant = self.constants[i].clone();
        let has_constant = !is_zero(&constant);
        let mut current = polynomial.to_vec();
        let mut current_degree = degree;
        for a in 0..=remaining {
            if a > 0 {
                if !shifted {
                    break;
                }
                current = self.raise(&current, current_degree, i);
                current_degree += 1;
            }
            let mut power = self.integer(1);
            for b in 0..=(remaining - a) / 2 {
                if b > 0 {
                    if !has_constant {
                        break;
                    }
                    power *= &constant;
                }
                // (−1)^(a+b) C(a+b, a) 2^a c_i^b.
                let binomial = (1..=a as i64).fold(1i64, |c, k| c * (b as i64 + k) / k);
                let sign = if (a + b) % 2 == 0 { 1 } else { -1 };
                let factor = self.integer(sign * (binomial << a)) * &power;
                self.walk(
                    i + 1,
                    remaining - a - 2 * b,
                    &current,
                    current_degree,
                    weight.clone() * &factor,
                );
            }
        }
    }
}

/// The ultraviolet residue (coefficient of `1/ε_UV`) of one integral: `n`
/// propagators with real or complex shifts `shifts` (`4n` entries, the four
/// components of each `V_i` in turn), squared masses `masses_squared` (`n`)
/// and the dense rank-`rank` tensor `coefficients` (`binomial(rank + 4, 4)`,
/// [`super::tensor_layout`]). Zero unless `rank + 4 ≥ 2n`.
pub fn tensor_uv_residue<T: SingleFloat>(
    n: usize,
    rank: usize,
    shifts: &[Complex<T>],
    masses_squared: &[Complex<T>],
    coefficients: &[Complex<T>],
) -> Complex<T> {
    TensorUvResidues::new(n, rank).residue(shifts, masses_squared, coefficients)
}

/// [`tensor_uv_residue`] for many integrals of one `(n, rank)` shape: the
/// monomial tables are built once.
pub struct TensorUvResidues {
    n: usize,
    rank: usize,
    layout: Option<Layout>,
}

impl TensorUvResidues {
    pub fn new(n: usize, rank: usize) -> Self {
        let divergent = rank + 4 >= 2 * n;
        Self {
            n,
            rank,
            // Degrees reached: r + Σ a_i <= rank + j = 2 rank − 2n + 4.
            layout: divergent.then(|| Layout::new(2 * rank + 4 - 2 * n)),
        }
    }

    /// The residue of one integral (inputs as [`tensor_uv_residue`]).
    pub fn residue<T: SingleFloat>(
        &self,
        shifts: &[Complex<T>],
        masses_squared: &[Complex<T>],
        coefficients: &[Complex<T>],
    ) -> Complex<T> {
        let (n, rank) = (self.n, self.rank);
        assert_eq!(shifts.len(), 4 * n, "4n shift components");
        assert_eq!(masses_squared.len(), n, "n squared masses");
        assert_eq!(
            coefficients.len(),
            ncoeff(rank),
            "binomial(rank + 4, 4) coefficients"
        );
        let zero = Complex::new(coefficients[0].re.zero(), coefficients[0].re.zero());
        let Some(layout) = &self.layout else {
            return zero;
        };
        let metric = [1, -1, -1, -1];
        let mut walk = Walk {
            layout,
            covariant: (0..n)
                .map(|i| {
                    std::array::from_fn(|mu| {
                        let v = shifts[4 * i + mu].clone();
                        if metric[mu] > 0 { v } else { -v }
                    })
                })
                .collect(),
            constants: (0..n)
                .map(|i| {
                    let v = &shifts[4 * i..4 * i + 4];
                    let square = v[0].clone() * &v[0]
                        - v[1].clone() * &v[1]
                        - v[2].clone() * &v[2]
                        - v[3].clone() * &v[3];
                    square - &masses_squared[i]
                })
                .collect(),
            zero: zero.clone(),
            total: zero.clone(),
        };
        for r in (2 * n).saturating_sub(4)..=rank {
            let block = &coefficients[block_offset(r)..ncoeff(r)];
            if block.iter().all(is_zero) {
                continue;
            }
            let one = walk.integer(1);
            walk.walk(0, r + 4 - 2 * n, block, r, one);
        }
        walk.total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fast::tensor_layout::monomial_index;

    fn c(x: f64) -> Complex<f64> {
        Complex::new(x, 0.)
    }

    fn tensor(rank: usize, terms: &[([u32; 4], f64)]) -> Vec<Complex<f64>> {
        let mut out = vec![c(0.); ncoeff(rank)];
        for &(m, value) in terms {
            out[monomial_index(m)] = c(value);
        }
        out
    }

    #[test]
    fn scalar_tadpole_and_bubble_residues() {
        // A0(m²): m², at any shift.
        let shift = [c(3.), c(1.), c(-2.), c(0.5)];
        let one = tensor(0, &[([0; 4], 1.)]);
        assert_eq!(tensor_uv_residue(1, 0, &[c(0.); 4], &[c(7.)], &one), c(7.));
        assert!((tensor_uv_residue(1, 0, &shift, &[c(7.)], &one).re - 7.).abs() < 1e-13);
        // B0: 1, also scaleless.
        let shifts = [c(0.), c(0.), c(0.), c(0.), c(2.), c(0.), c(0.), c(2.)];
        assert_eq!(
            tensor_uv_residue(2, 0, &shifts, &[c(0.), c(0.)], &one),
            c(1.)
        );
    }

    #[test]
    fn rank_two_bubble_matches_b00_and_b11() {
        // ∫ q^μ q^ν / (q² − m0²)((q + p)² − m1²): UV of
        // g^μν B00 + p^μ p^ν B11 is g^μν ((m0² + m1²)/4 − p²/12) + p^μ p^ν / 3.
        let p = [3., 1., -2., 0.5];
        let p2 = p[0] * p[0] - p[1] * p[1] - p[2] * p[2] - p[3] * p[3];
        let (m0, m1) = (5., 11.);
        let mut shifts = vec![c(0.); 4];
        shifts.extend(p.map(c));
        let metric = [1., -1., -1., -1.];
        for mu in 0..4 {
            for nu in mu..4 {
                let mut e = [0u32; 4];
                e[mu] += 1;
                e[nu] += 1;
                let numerator = tensor(2, &[(e, 1.)]);
                let got = tensor_uv_residue(2, 2, &shifts, &[c(m0), c(m1)], &numerator);
                let g = if mu == nu { metric[mu] } else { 0. };
                let expected = g * ((m0 + m1) / 4. - p2 / 12.) + p[mu] * p[nu] / 3.;
                assert!(
                    (got.re - expected).abs() < 1e-13,
                    "{mu}{nu}: {got:?} {expected}"
                );
            }
        }
    }

    #[test]
    fn scaleless_longitudinal_moments() {
        // B_r(0; 0, 0): N = (q·n)^r with n·p = 1 for a light-like p has the
        // UV residue (−1)^r/(r+1) (oneloopmaster's scaleless moments).
        let p = [2., 0., 0., 2.];
        let mut shifts = vec![c(0.); 4];
        shifts.extend(p.map(c));
        // n = (1/4, 0, 0, −1/4): n·p = 1, n² = 0.
        let n_vec: [f64; 4] = [0.25, 0., 0., -0.25];
        let covariant = [n_vec[0], -n_vec[1], -n_vec[2], -n_vec[3]];
        for rank in 0..=4usize {
            let mut numerator = vec![c(0.); ncoeff(rank)];
            for m in monomials(rank).into_iter().skip(block_offset(rank)) {
                let mut value = (1..=rank).product::<usize>() as f64;
                for (mu, &e) in m.iter().enumerate() {
                    value *=
                        covariant[mu].powi(e as i32) / (1..=e as usize).product::<usize>() as f64;
                }
                numerator[monomial_index(m)] = c(value);
            }
            let got = tensor_uv_residue(2, rank, &shifts, &[c(0.), c(0.)], &numerator);
            let expected = if rank % 2 == 0 { 1. } else { -1. } / (rank + 1) as f64;
            assert!((got.re - expected).abs() < 1e-14, "rank {rank}: {got:?}");
        }
    }

    #[test]
    fn finite_integrals_have_no_residue() {
        let numerator = tensor(1, &[([0, 1, 0, 0], 1.)]);
        let shifts = vec![c(0.); 12];
        assert_eq!(
            tensor_uv_residue(3, 1, &shifts, &[c(1.), c(2.), c(3.)], &numerator),
            c(0.)
        );
    }
}
