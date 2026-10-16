//! Original Ninja residue ordering, projected from corrected Laurent quotients.

use ninjars::{Complex, Error, ExpansionKind, Real, Result, SingleFloat};
use symbolica::{
    atom::{Atom, AtomCore},
    domains::dual::DualNumber,
    evaluate::EvaluationDomain,
    symbol,
};

use crate::{
    LaurentDivision,
    kernel::{CompiledDualKernel, CompiledKernel, ExactKernel},
};

/// Convert both transverse orientations into Ninja's indexed residue coefficients.
///
/// Each input row is `[first quotient, swapped quotient, e1.dot(e2)]`. Both
/// quotients use `quotient_expansion()` ordering. On the first orientation the
/// growing vector is e3; on the swapped orientation it is e4. The null basis
/// satisfies `e3.dot(e4) = -e1.dot(e2)`. Bubble inputs must already include
/// triangle subtractions. Coefficient order is original `polys_implem.cxx`.
/// The nonzero `g=e1.dot(e2)` and canonical offsets are required: for triangles,
/// `(a+V0).dot(e3/e4)=0`; for bubbles, `(a0+V0).dot(e2/e3/e4)=0`
/// and `a1.dot(e2)=g`. `SymbolicaCuts` supplies these conventions. Arbitrary
/// null vectors alone do not define a compatible projection.
pub struct ResidueProjection {
    quotient: ExpansionKind,
    pub(crate) exact: ExactKernel,
}

impl ResidueProjection {
    pub fn new(division: &LaurentDivision) -> Result<Self> {
        let cut = division.cut_size();
        if cut != 2 && cut != 3 {
            return Err(Error::Unsupported(
                "full residue projection currently requires a bubble or triangle".into(),
            ));
        }
        let quotient = division.quotient_expansion().clone();
        let width = quotient.len();
        let parameters: Vec<_> = (0..2 * width + 1)
            .map(|i| Atom::var(symbol!(format!("ninjars_projection::p{i}"))))
            .collect();
        let g = &parameters[2 * width];
        let transverse = -g;
        let coefficient = |swapped: bool, t: i32, x: u16, mu: u16| {
            let index = match &quotient {
                ExpansionKind::T3 { powers } => {
                    if x == 0 {
                        powers.iter().position(|&p| p == (t, mu))
                    } else {
                        None
                    }
                }
                ExpansionKind::T2 { powers } => powers.iter().position(|&p| p == (t, x, mu)),
                _ => unreachable!(),
            };
            index.map_or(Atom::Zero, |i| {
                parameters[usize::from(swapped) * width + i].clone()
            })
        };
        let mut expressions = vec![
            Atom::Zero;
            if division.higher_rank() {
                if cut == 3 { 15 } else { 20 }
            } else {
                10
            }
        ];
        if cut == 3 {
            expressions[0] = (coefficient(false, 0, 0, 0) + coefficient(true, 0, 0, 0)) / 2;
            expressions[7] = (coefficient(false, 0, 0, 1) + coefficient(true, 0, 0, 1)) / 2;
            for (swapped, indices) in [(true, [1, 2, 3, 8]), (false, [4, 5, 6, 9])] {
                for (power, &index) in indices[..3].iter().enumerate() {
                    expressions[index] = coefficient(swapped, power as i32 + 1, 0, 0)
                        / transverse.pow(power as i32 + 1);
                }
                expressions[indices[3]] = coefficient(swapped, 1, 0, 1) / &transverse;
            }
            if division.higher_rank() {
                expressions[10] = coefficient(true, 2, 0, 1) / transverse.pow(2);
                expressions[11] = coefficient(false, 2, 0, 1) / transverse.pow(2);
                expressions[12] = coefficient(true, 4, 0, 0) / transverse.pow(4);
                expressions[13] = coefficient(false, 4, 0, 0) / transverse.pow(4);
                expressions[14] = (coefficient(false, 0, 0, 2) + coefficient(true, 0, 0, 2)) / 2;
            }
        } else {
            expressions[0] = coefficient(false, 0, 0, 0);
            expressions[1] = coefficient(false, 0, 1, 0) / g;
            expressions[2] = coefficient(false, 0, 2, 0) / g.pow(2);
            expressions[9] = coefficient(false, 0, 0, 1);
            for (swapped, indices) in [(true, [3, 4, 7]), (false, [5, 6, 8])] {
                expressions[indices[0]] = coefficient(swapped, 1, 0, 0) / &transverse;
                expressions[indices[1]] = coefficient(swapped, 2, 0, 0) / transverse.pow(2);
                expressions[indices[2]] = coefficient(swapped, 1, 1, 0) / (g * &transverse);
            }
            if division.higher_rank() {
                expressions[10] = coefficient(false, 0, 1, 1) / g;
                expressions[13] = coefficient(false, 0, 3, 0) / g.pow(3);
                for (swapped, indices) in [(true, [11, 14, 16, 18]), (false, [12, 15, 17, 19])] {
                    expressions[indices[0]] = coefficient(swapped, 1, 0, 1) / &transverse;
                    expressions[indices[1]] = coefficient(swapped, 3, 0, 0) / transverse.pow(3);
                    expressions[indices[2]] =
                        coefficient(swapped, 1, 2, 0) / (&transverse * g.pow(2));
                    expressions[indices[3]] =
                        coefficient(swapped, 2, 1, 0) / (transverse.pow(2) * g);
                }
            }
        }
        Ok(Self {
            quotient,
            exact: ExactKernel {
                expressions,
                parameters,
            },
        })
    }

    pub fn quotient_expansion(&self) -> &ExpansionKind {
        &self.quotient
    }
    pub fn input_columns(&self) -> usize {
        self.exact.parameters.len()
    }
    pub fn coefficient_count(&self) -> usize {
        self.exact.expressions.len()
    }
    pub fn compile<T: Real + SingleFloat>(
        &self,
        prototype: &T,
    ) -> Result<ResidueProjectionEvaluator<T>>
    where
        Complex<T>: Real + EvaluationDomain,
    {
        Ok(ResidueProjectionEvaluator {
            kernel: self.exact.compile(prototype)?,
        })
    }
    pub fn compile_dual<T: Real + SingleFloat, D: Real + DualNumber<Complex<T>>>(
        &self,
        prototype: &T,
        dual: &D,
    ) -> Result<DualResidueProjectionEvaluator<T, D>>
    where
        Complex<T>: Real + EvaluationDomain,
    {
        Ok(DualResidueProjectionEvaluator {
            kernel: self.exact.compile_dual(prototype, dual)?,
        })
    }
}

pub struct ResidueProjectionEvaluator<T> {
    kernel: CompiledKernel<T>,
}
impl<T: Real + SingleFloat> ResidueProjectionEvaluator<T>
where
    Complex<T>: Real + EvaluationDomain,
{
    pub fn evaluate_batch(
        &mut self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        rows: usize,
    ) -> Result<()> {
        self.kernel.evaluate_batch(input, output, rows)
    }
}

pub struct DualResidueProjectionEvaluator<T, D> {
    kernel: CompiledDualKernel<T, D>,
}
impl<T: Real + SingleFloat, D: Real + DualNumber<Complex<T>>> DualResidueProjectionEvaluator<T, D>
where
    Complex<T>: Real + EvaluationDomain,
{
    pub fn evaluate_batch(&mut self, input: &[D], output: &mut [D], rows: usize) -> Result<()> {
        self.kernel.evaluate_batch(input, output, rows)
    }
}
