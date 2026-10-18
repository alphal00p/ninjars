//! Dense tensor-numerator coefficients in original Ninja's/MadLoop's layout.
//!
//! A [`PolynomialCoefficientPlan`] writes the numerator as
//! `sum_terms head[term] * q0^a0 q1^a1 q2^a2 q3^a3 mu2^a4`, where each term's
//! coefficient is exactly one evaluator output ("head") and distinct terms
//! have distinct monomials. Ninja's tensor interface instead stores
//! `N(q) = sum_k tensor[k] * monomial_k(q)` with the monomial order of
//! [`ninjars::fast::tensor_layout`]. Because the plan's polynomial is in
//! canonical form, every tensor slot is either structurally zero or a copy of
//! exactly one head with numerical factor one: no linear combination, symmetry
//! factor or rescaling is needed. The map is therefore a static scatter
//! computed once at setup; [`TensorCoefficientPlan::scatter`] performs no
//! allocation and no arithmetic.
//!
//! Ninja's tensor numerator has no `mu^2` slots, so [`TensorCoefficientPlan::new`]
//! and [`TensorCoefficientPlan::with_rank`] reject numerators with explicit
//! `mu^2` dependence. [`TensorCoefficientPlan::with_mu_squared`] instead lays
//! `N = sum_k mu^(2k) T_k(q)` out in the block layout of
//! [`ninjars::fast::MuTensorNumerator`] (block `k` a rank-`(R - 2k)` tensor);
//! without `mu^2` terms that layout is exactly the plain tensor.

use std::collections::HashMap;

use ninjars::fast::tensor_layout;

use crate::{CoefficientError, PolynomialCoefficientPlan};

type Result<T> = std::result::Result<T, CoefficientError>;

/// `tensor[slot] = heads[head]`, with numerical factor one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TensorEntry {
    /// Index in the dense tensor (`tensor_layout::monomial_index`).
    pub slot: usize,
    /// Column of the coefficient evaluator output.
    pub head: usize,
    /// Exponents `(q0, q1, q2, q3)` of the slot's monomial.
    pub exponents: [u32; 4],
    /// Power `k` of the slot's `mu^(2k)` block (zero without `mu^2` terms).
    pub mu_power: u32,
}

/// Static map from evaluated coefficient heads to a dense rank-`R` tensor of
/// `binomial(R + 4, 4)` complex coefficients per phase-space lane, followed
/// by the `mu^(2k)` blocks of [`Self::with_mu_squared`].
#[derive(Clone, Debug)]
pub struct TensorCoefficientPlan {
    rank: usize,
    mu_powers: usize,
    ncoeff: usize,
    heads: usize,
    entries: Vec<TensorEntry>,
    zero_slots: Vec<usize>,
}

impl TensorCoefficientPlan {
    /// Use the numerator's own degree as the tensor rank.
    pub fn new(plan: &PolynomialCoefficientPlan) -> Result<Self> {
        Self::with_rank(plan, plan.rank())
    }

    /// Lay out the numerator in a rank-`rank` tensor; slots of degree above
    /// the numerator's degree stay zero. Fails if the numerator depends on
    /// `mu^2` or exceeds `rank`.
    pub fn with_rank(plan: &PolynomialCoefficientPlan, rank: usize) -> Result<Self> {
        Self::build(plan, rank, false)
    }

    /// Lay out a numerator with explicit `mu^2` terms as
    /// `N = sum_{k=0}^{K} mu^(2k) T_k(q)`, `K` the highest `mu^2` power and
    /// `T_k` a dense rank-`(rank - 2k)` tensor (rank defaults to the numerator
    /// degree, `mu^2` weighing two); see
    /// [`ninjars::fast::tensor_layout::ncoeff_mu`]. Without `mu^2` terms the
    /// result is identical to [`Self::with_rank`].
    pub fn with_mu_squared(plan: &PolynomialCoefficientPlan, rank: Option<usize>) -> Result<Self> {
        Self::build(plan, rank.unwrap_or(plan.rank()), true)
    }

    fn build(plan: &PolynomialCoefficientPlan, rank: usize, allow_mu: bool) -> Result<Self> {
        let heads = plan.coefficient_expressions();
        let index: HashMap<_, _> = heads
            .iter()
            .enumerate()
            .map(|(column, head)| (head, column))
            .collect();
        let mut mu_powers = 0;
        for term in plan.polynomial() {
            let mu = term.exponents.get(4).copied().unwrap_or(0);
            if mu != 0 && !allow_mu {
                return Err(CoefficientError::ExplicitMuSquared { power: mu });
            }
            mu_powers = mu_powers.max(mu.max(0) as usize);
        }
        if 2 * mu_powers > rank {
            return Err(CoefficientError::TensorRank {
                requested: rank,
                required: plan.rank(),
            });
        }
        let ncoeff = tensor_layout::ncoeff_mu(rank, mu_powers);
        let mut entries = Vec::with_capacity(plan.polynomial().nterms());
        for term in plan.polynomial() {
            let [a0, a1, a2, a3, mu] = <[i32; 5]>::try_from(term.exponents)
                .map_err(|_| CoefficientError::NotPolynomial)?;
            if [a0, a1, a2, a3, mu].iter().any(|power| *power < 0) {
                return Err(CoefficientError::NotPolynomial);
            }
            let exponents = [a0, a1, a2, a3].map(|power| power as u32);
            let degree = exponents.iter().sum::<u32>() as usize + 2 * mu as usize;
            if degree > rank {
                return Err(CoefficientError::TensorRank {
                    requested: rank,
                    required: plan.rank(),
                });
            }
            let head = *index
                .get(term.coefficient)
                .ok_or_else(|| CoefficientError::Symbolic("unindexed coefficient head".into()))?;
            entries.push(TensorEntry {
                slot: tensor_layout::mu_block_offset(rank, mu as usize)
                    + tensor_layout::monomial_index(exponents),
                head,
                exponents,
                mu_power: mu as u32,
            });
        }
        entries.sort_by_key(|entry| entry.slot);
        if entries.windows(2).any(|pair| pair[0].slot == pair[1].slot) {
            return Err(CoefficientError::Symbolic(
                "repeated monomial in canonical polynomial".into(),
            ));
        }
        let mut occupied = vec![false; ncoeff];
        for entry in &entries {
            occupied[entry.slot] = true;
        }
        let zero_slots = (0..ncoeff).filter(|slot| !occupied[*slot]).collect();
        Ok(Self {
            rank,
            mu_powers,
            ncoeff,
            heads: heads.len(),
            entries,
            zero_slots,
        })
    }

    /// Tensor rank `R`.
    pub fn rank(&self) -> usize {
        self.rank
    }
    /// Highest power `K` of `mu^2` (zero for Ninja's plain tensor layout).
    pub fn mu_powers(&self) -> usize {
        self.mu_powers
    }
    /// Coefficients per lane, `binomial(R + 4, 4)` (plus the `mu^2` blocks).
    pub fn ncoeff(&self) -> usize {
        self.ncoeff
    }
    /// Evaluator outputs per lane consumed by [`Self::scatter`].
    pub fn head_count(&self) -> usize {
        self.heads
    }
    /// Nonzero slots, sorted by slot.
    pub fn entries(&self) -> &[TensorEntry] {
        &self.entries
    }
    /// Structurally zero slots, sorted.
    pub fn zero_slots(&self) -> &[usize] {
        &self.zero_slots
    }

    /// Scatter `lanes` row-major head rows into `lanes` dense tensor rows.
    /// Every output entry is written; no allocation is performed.
    pub fn scatter<T: Clone>(
        &self,
        heads: &[T],
        tensor: &mut [T],
        lanes: usize,
        zero: &T,
    ) -> Result<()> {
        for (buffer, columns, actual) in [
            ("tensor heads", self.heads, heads.len()),
            ("tensor coefficients", self.ncoeff, tensor.len()),
        ] {
            let expected = lanes
                .checked_mul(columns)
                .ok_or(CoefficientError::Overflow)?;
            if expected != actual {
                return Err(CoefficientError::Shape {
                    buffer,
                    expected,
                    actual,
                });
            }
        }
        if lanes == 0 {
            return Ok(());
        }
        for (row, values) in tensor.chunks_exact_mut(self.ncoeff).enumerate() {
            let source = &heads[row * self.heads..(row + 1) * self.heads];
            for entry in &self.entries {
                values[entry.slot].clone_from(&source[entry.head]);
            }
            for &slot in &self.zero_slots {
                values[slot].clone_from(zero);
            }
        }
        Ok(())
    }
}
