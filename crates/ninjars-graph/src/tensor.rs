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
//! Ninja's tensor numerator has no `mu^2` slots, so numerators with explicit
//! `mu^2` dependence are rejected by this path.

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
}

/// Static map from evaluated coefficient heads to a dense rank-`R` tensor of
/// `binomial(R + 4, 4)` complex coefficients per phase-space lane.
#[derive(Clone, Debug)]
pub struct TensorCoefficientPlan {
    rank: usize,
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
        let heads = plan.coefficient_expressions();
        let index: HashMap<_, _> = heads
            .iter()
            .enumerate()
            .map(|(column, head)| (head, column))
            .collect();
        let ncoeff = tensor_layout::ncoeff(rank);
        let mut entries = Vec::with_capacity(plan.polynomial().nterms());
        for term in plan.polynomial() {
            let [a0, a1, a2, a3, mu] = <[i32; 5]>::try_from(term.exponents)
                .map_err(|_| CoefficientError::NotPolynomial)?;
            if mu != 0 {
                return Err(CoefficientError::ExplicitMuSquared { power: mu });
            }
            if [a0, a1, a2, a3].iter().any(|power| *power < 0) {
                return Err(CoefficientError::NotPolynomial);
            }
            let exponents = [a0, a1, a2, a3].map(|power| power as u32);
            let degree = exponents.iter().sum::<u32>() as usize;
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
                slot: tensor_layout::monomial_index(exponents),
                head,
                exponents,
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
    /// Coefficients per lane, `binomial(R + 4, 4)`.
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
