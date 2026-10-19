//! Master weights and rational terms in original Ninja residue conventions.
//!
//! The physical formulas follow the MI sections of Ninja `ninja.cc` and
//! `Xninja.cc`. Symbolica owns expression arithmetic and numerical instruction
//! generation; the numerical wrappers only pack existing coefficient arrays.

use std::ops::Range;

use ninjars::{
    BatchShape, Complex, Error, Laurent, MasterIntegral, RankClass, Real, Result, SingleFloat,
    residue_coefficient_count,
};
use symbolica::{
    atom::{Atom, AtomCore, AtomView},
    domains::dual::DualNumber,
    evaluate::EvaluationDomain,
    symbol,
};

use crate::kernel::{CompiledDualKernel, CompiledKernel, ExactKernel};

/// Column ranges in one residue-to-master-weight input row.
#[derive(Clone, Debug)]
pub struct MasterContractionInputLayout {
    /// Original Ninja c-index order, including unused entries.
    pub coefficients: Range<usize>,
    /// Bubble: s21. Triangle: [s21, s32, s13]. Box:
    /// [s21, s32, s43, s14, s31, s42]. Tadpole: empty.
    pub invariants: Range<usize>,
    /// Squared masses in cut-partition order.
    pub masses_squared: Range<usize>,
    /// Tadpole basis scalar product e1.dot(e2), so e3.dot(e4) = -e12.
    pub e12: Option<usize>,
    /// Bubble (V_second - V_first).dot(e2).
    pub ke2: Option<usize>,
}

/// Exact contraction of a tadpole, bubble, triangle, or box residue.
///
/// Ordinary/higher-rank coefficient counts are 5/16, 10/20, 10/15, and 5/6. Higher
/// rank means numerator rank n+1. In particular, the higher-rank tadpole retains
/// original unused c[9]; c[14] and c[15] are never compacted or renumbered.
/// Rank n+2 ([`Self::new_class`]) has 16, 35, 21 and 7 coefficients: the bubble
/// adds B1111 with weight c20·ke2⁴ and the rational terms of c21 (mu²(q·e2)²)
/// and c22 (mu⁴), the box the rational term of c6 (mu⁶); tadpole and triangle
/// formulas are unchanged (`tools/rank_n2/README.md`).
///
/// Weight rows are `[weights in masters() order, rational]`. Contracted rows
/// are `[finite including rational, single pole, double pole,
/// cut-constructible finite, rational]`. The explicit rational term belongs
/// only to the finite coefficient; rational contributions already inside a
/// tensor master stay in the cut-constructible contribution, as in Ninja.
///
/// Bubble B1, B111 and B1111 use OneLOop's `(q+K)` denominator convention
/// directly: their UV poles are -1/2, -1/4 and 1/5. Callers supply compatible master Laurent
/// arrays and handle scaleless/singular lanes before accumulation.
pub struct MasterContraction {
    layout: MasterContractionInputLayout,
    masters: Vec<MasterIntegral>,
    pub(crate) weights: ExactKernel,
    /// The additive terms of the rational weight, with the weights' inputs.
    pub(crate) rational_terms: ExactKernel,
    pub(crate) contraction: ExactKernel,
}

impl MasterContraction {
    pub fn new(cut_size: usize, higher_rank: bool) -> Result<Self> {
        Self::new_class(
            cut_size,
            if higher_rank {
                RankClass::NPlus1
            } else {
                RankClass::UpToN
            },
        )
    }

    /// Contraction for a residue of the given rank class.
    pub fn new_class(cut_size: usize, class: RankClass) -> Result<Self> {
        if !(1..=4).contains(&cut_size) {
            return Err(Error::Unsupported(
                "master contraction currently supports cuts of size one through four".into(),
            ));
        }
        let higher_rank = class != RankClass::UpToN;
        let rank_n2 = class == RankClass::NPlus2;
        let coefficient_count = residue_coefficient_count(cut_size, class);
        let invariant_count = match cut_size {
            1 => 0,
            2 => 1,
            3 => 3,
            _ => 6,
        };
        let invariant_start = coefficient_count;
        let mass_start = invariant_start + invariant_count;
        let basis_start = mass_start + cut_size;
        let layout = MasterContractionInputLayout {
            coefficients: 0..coefficient_count,
            invariants: invariant_start..mass_start,
            masses_squared: mass_start..basis_start,
            e12: (cut_size == 1).then_some(basis_start),
            ke2: (cut_size == 2).then_some(basis_start),
        };
        let parameters = symbolic_parameters(
            "ninjars_master_weights",
            basis_start + usize::from(cut_size < 3),
        );
        let c = &parameters[layout.coefficients.clone()];
        let s = &parameters[layout.invariants.clone()];
        let m = &parameters[layout.masses_squared.clone()];
        let (masters, mut expressions, rational) = match cut_size {
            1 => {
                let mut weight = c[0].clone();
                let rational = if higher_rank {
                    let e12 = &parameters[layout.e12.unwrap()];
                    weight -= e12 * &m[0] * &c[15] / 4;
                    (&c[14] - e12 * &c[15] / 4) * m[0].pow(2) / 2
                } else {
                    Atom::Zero
                };
                (vec![MasterIntegral::A0], vec![weight], rational)
            }
            2 => {
                let ke2 = &parameters[layout.ke2.unwrap()];
                let mut masters = vec![MasterIntegral::B0, MasterIntegral::B1, MasterIntegral::B11];
                let mut expressions = vec![c[0].clone(), &c[1] * ke2, &c[2] * ke2.pow(2)];
                let mut rational = (&m[0] * 3 + &m[1] * 3 - &s[0]) * &c[9] / 6;
                if higher_rank {
                    masters.push(MasterIntegral::B111);
                    expressions.push(&c[13] * ke2.pow(3));
                    rational += ke2 * (&s[0] - &m[0] * 2 - &m[1] * 4) * &c[10] / 12;
                }
                if rank_n2 {
                    // I2[mu²(q·e2)²] and I2[mu⁴] by dimension shifting, with
                    // m[0] the origin line (`tools/rank_n2/README.md`).
                    masters.push(MasterIntegral::B1111);
                    expressions.push(&c[20] * ke2.pow(4));
                    rational += ke2.pow(2) * (&m[0] * 5 + &m[1] * 15 - &s[0] * 3) * &c[21] / 60
                        - (s[0].pow(2) - &s[0] * (&m[0] + &m[1]) * 5
                            + (m[0].pow(2) + &m[0] * &m[1] + m[1].pow(2)) * 10)
                            * &c[22]
                            / 60;
                }
                (masters, expressions, rational)
            }
            3 => {
                let mut rational = &c[7] / 2;
                if higher_rank {
                    rational += &c[14] * ((&s[0] + &s[1] + &s[2]) / 4 - &m[0] - &m[1] - &m[2]) / 6;
                }
                (vec![MasterIntegral::C0], vec![c[0].clone()], rational)
            }
            _ => {
                let mut rational = -&c[4] / 6;
                if rank_n2 {
                    // I4[mu⁶] = sum m/12 - sum s/60 over all six invariants.
                    let masses = m.iter().fold(Atom::Zero, |sum, v| sum + v);
                    let invariants = s.iter().fold(Atom::Zero, |sum, v| sum + v);
                    rational += &c[6] * (masses / 12 - invariants / 60);
                }
                (vec![MasterIntegral::D0], vec![c[0].clone()], rational)
            }
        };
        // The additive terms of the rational part as built above, for the
        // opt-in cancellation test only; the weights keep the rational part
        // as one expression.
        let rational_terms = ExactKernel {
            expressions: match rational.as_view() {
                AtomView::Add(add) => add.iter().map(|term| term.to_owned()).collect(),
                _ => vec![rational.clone()],
            },
            parameters: parameters.clone(),
        };
        expressions.push(rational);
        let weights = ExactKernel {
            expressions,
            parameters,
        };

        let n = masters.len();
        let parameters = symbolic_parameters("ninjars_master_contraction", n + 1 + n * 3);
        let coefficient = |epsilon: usize| {
            (0..n).fold(Atom::Zero, |sum, master| {
                sum + &parameters[master] * &parameters[n + 1 + 3 * master + epsilon]
            })
        };
        let cut_constructible = coefficient(0);
        let expressions = vec![
            &cut_constructible + &parameters[n],
            coefficient(1),
            coefficient(2),
            cut_constructible,
            parameters[n].clone(),
        ];
        Ok(Self {
            layout,
            masters,
            weights,
            rational_terms,
            contraction: ExactKernel {
                expressions,
                parameters,
            },
        })
    }

    pub fn input_layout(&self) -> &MasterContractionInputLayout {
        &self.layout
    }
    pub fn input_columns(&self) -> usize {
        self.weights.parameters.len()
    }
    pub fn masters(&self) -> &[MasterIntegral] {
        &self.masters
    }
    pub fn weight_columns(&self) -> usize {
        self.masters.len() + 1
    }

    pub fn compile<T: Real + SingleFloat>(
        &self,
        prototype: &T,
    ) -> Result<MasterContractionEvaluator<T>>
    where
        Complex<T>: Real + EvaluationDomain,
    {
        Ok(MasterContractionEvaluator {
            weights: self.weights.compile(prototype)?,
            contraction: self.contraction.compile(prototype)?,
            master_count: self.masters.len(),
            arguments: Vec::new(),
        })
    }

    pub fn compile_dual<T: Real + SingleFloat, D: Real + DualNumber<Complex<T>>>(
        &self,
        prototype: &T,
        dual: &D,
    ) -> Result<DualMasterContractionEvaluator<T, D>>
    where
        Complex<T>: Real + EvaluationDomain,
    {
        Ok(DualMasterContractionEvaluator {
            weights: self.weights.compile_dual(prototype, dual)?,
            contraction: self.contraction.compile_dual(prototype, dual)?,
            master_count: self.masters.len(),
            arguments: Vec::new(),
        })
    }
}

fn symbolic_parameters(namespace: &str, count: usize) -> Vec<Atom> {
    (0..count)
        .map(|i| Atom::var(symbol!(format!("{namespace}::p{i}"))))
        .collect()
}

fn pack<T: Clone>(
    master_count: usize,
    weights: &[T],
    masters: &[Laurent<T>],
    rows: usize,
    output: &mut Vec<T>,
) -> Result<()> {
    let columns = master_count + 1;
    BatchShape { rows, columns }.validate("master weights", weights.len())?;
    BatchShape {
        rows,
        columns: master_count,
    }
    .validate("master Laurent arrays", masters.len())?;
    output.clear();
    for row in 0..rows {
        output.extend_from_slice(&weights[row * columns..(row + 1) * columns]);
        for master in &masters[row * master_count..(row + 1) * master_count] {
            output.extend_from_slice(master);
        }
    }
    Ok(())
}

/// Retains numerical kernels and packing storage across scalar-domain batches.
pub struct MasterContractionEvaluator<T> {
    weights: CompiledKernel<T>,
    contraction: CompiledKernel<T>,
    master_count: usize,
    arguments: Vec<Complex<T>>,
}

impl<T: Real + SingleFloat> MasterContractionEvaluator<T>
where
    Complex<T>: Real + EvaluationDomain,
{
    pub fn weights_batch(
        &mut self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        rows: usize,
    ) -> Result<()> {
        self.weights.evaluate_batch(input, output, rows)
    }

    /// Masters are row-major, with one Laurent array per entry in `masters()`.
    pub fn contract_batch(
        &mut self,
        weights: &[Complex<T>],
        masters: &[Laurent<Complex<T>>],
        output: &mut [Complex<T>],
        rows: usize,
    ) -> Result<()> {
        pack(
            self.master_count,
            weights,
            masters,
            rows,
            &mut self.arguments,
        )?;
        self.contraction
            .evaluate_batch(&self.arguments, output, rows)
    }
}

/// The same two instruction graphs with all static or dynamic Dual components.
pub struct DualMasterContractionEvaluator<T, D> {
    weights: CompiledDualKernel<T, D>,
    contraction: CompiledDualKernel<T, D>,
    master_count: usize,
    arguments: Vec<D>,
}

impl<T: Real + SingleFloat, D: Real + DualNumber<Complex<T>>> DualMasterContractionEvaluator<T, D>
where
    Complex<T>: Real + EvaluationDomain,
{
    pub fn weights_batch(&mut self, input: &[D], output: &mut [D], rows: usize) -> Result<()> {
        self.weights.evaluate_batch(input, output, rows)
    }

    /// Masters are row-major, with one Laurent array per entry in `masters()`.
    pub fn contract_batch(
        &mut self,
        weights: &[D],
        masters: &[Laurent<D>],
        output: &mut [D],
        rows: usize,
    ) -> Result<()> {
        pack(
            self.master_count,
            weights,
            masters,
            rows,
            &mut self.arguments,
        )?;
        self.contraction
            .evaluate_batch(&self.arguments, output, rows)
    }
}

#[cfg(test)]
mod tests {
    use super::MasterContraction;
    use ninjars::RankClass;
    use symbolica::atom::Atom;

    /// The cancellation test's rational terms are the additive terms of the
    /// rational weight as built: they add up to it exactly and read the same
    /// inputs.
    #[test]
    fn rational_terms_add_up_to_the_rational_weight() {
        let mut counts = Vec::new();
        for size in 1..=4 {
            for class in [RankClass::UpToN, RankClass::NPlus1, RankClass::NPlus2] {
                let plan = MasterContraction::new_class(size, class).unwrap();
                let terms = &plan.rational_terms;
                assert_eq!(terms.parameters, plan.weights.parameters);
                assert_eq!(
                    &Atom::add_many(terms.expressions.iter()),
                    plan.weights.expressions.last().unwrap(),
                    "size {size} {class:?}"
                );
                counts.push(terms.expressions.len());
            }
        }
        // Tadpole (zero below rank n+1), bubble (c9, c10, c21, c22), triangle
        // (c7, c14) and box (c4, c6) by rank class.
        assert_eq!(counts, [1, 1, 1, 1, 2, 4, 1, 2, 2, 1, 1, 2]);
    }
}
