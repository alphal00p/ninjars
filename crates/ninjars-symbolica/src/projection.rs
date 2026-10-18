//! Original Ninja residue ordering, projected from corrected Laurent quotients.

use ninjars::{
    Complex, Error, ExpansionKind, FourMomentum, RankClass, Real, Result, SingleFloat,
    residue_coefficient_count,
};
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

/// Exact projections of a bubble sampling line on a parent triangle's axes.
///
/// Each input row is `[a0, a1, e3, e4_effective, V_triangle, E3, E4]`,
/// with four components per shared momentum. Outputs are the four projections
/// `[(a0+V_triangle)·E, a1·E, e3·E, e4_effective·E]` for E3 then E4.
/// The parent triangle's origin is independent of the target bubble's origin.
/// This is the existing bubble correction plan, also exposed for mechanical
/// code generation; all contractions use FeynKit's shared +--- momentum type.
pub struct BubbleSourceProjection {
    pub(crate) exact: ExactKernel,
}
impl Default for BubbleSourceProjection {
    fn default() -> Self {
        Self::new()
    }
}
impl BubbleSourceProjection {
    pub fn new() -> Self {
        let p: Vec<_> = (0..28)
            .map(|i| Atom::var(symbol!(format!("ninjars_bubble_source_projection::p{i}"))))
            .collect();
        let vector = |offset: usize| {
            FourMomentum::from_args(
                p[offset].clone(),
                p[offset + 1].clone(),
                p[offset + 2].clone(),
                p[offset + 3].clone(),
            )
        };
        let mut expressions = Vec::new();
        for axis in [20, 24] {
            let e = vector(axis);
            expressions.push((vector(0) + vector(16)).dot(&e));
            for v in [4, 8, 12] {
                expressions.push(vector(v).dot(&e));
            }
        }
        Self {
            exact: ExactKernel {
                expressions,
                parameters: p,
            },
        }
    }
    pub fn compile<T: Real + SingleFloat>(
        &self,
        prototype: &T,
    ) -> Result<BubbleSourceProjectionEvaluator<T>>
    where
        Complex<T>: Real + EvaluationDomain,
    {
        Ok(BubbleSourceProjectionEvaluator {
            kernel: self.exact.compile(prototype)?,
        })
    }
    pub fn compile_dual<T: Real + SingleFloat, D: Real + DualNumber<Complex<T>>>(
        &self,
        prototype: &T,
        dual: &D,
    ) -> Result<DualBubbleSourceProjectionEvaluator<T, D>>
    where
        Complex<T>: Real + EvaluationDomain,
    {
        Ok(DualBubbleSourceProjectionEvaluator {
            kernel: self.exact.compile_dual(prototype, dual)?,
        })
    }
}
/// Batched arithmetic for the documented 28→8 source-projection layout.
pub struct BubbleSourceProjectionEvaluator<T> {
    kernel: CompiledKernel<T>,
}
impl<T: Real + SingleFloat> BubbleSourceProjectionEvaluator<T>
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
/// All-component Symbolica Dual transformation of the same projection plan.
pub struct DualBubbleSourceProjectionEvaluator<T, D> {
    kernel: CompiledDualKernel<T, D>,
}
impl<T: Real + SingleFloat, D: Real + DualNumber<Complex<T>>>
    DualBubbleSourceProjectionEvaluator<T, D>
where
    Complex<T>: Real + EvaluationDomain,
{
    pub fn evaluate_batch(&mut self, input: &[D], output: &mut [D], rows: usize) -> Result<()> {
        self.kernel.evaluate_batch(input, output, rows)
    }
}

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
///
/// At rank n+2 the new coefficients follow the canonical order of
/// `tools/rank_n2/theory/formulas_rank_n2.json`, with u = q·e3, v = q·e4 and
/// w = q·e2: triangle c15..c20 = u⁵, v⁵, mu²u³, mu²v³, mu⁴u, mu⁴v; bubble
/// c20..c34 = w⁴, mu²w², mu⁴, u⁴, v⁴, wu³, wv³, w²u², w²v², w³u, w³v, mu²u²,
/// mu²v², mu²wu, mu²wv.
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
        let class = division.rank_class();
        let mut expressions = vec![Atom::Zero; residue_coefficient_count(cut, class)];
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
        if class == RankClass::NPlus2 {
            // Every new coefficient is Q[t^b x^a mu^(2j)] / (g^a T^b), read from
            // the orientation whose growing vector gives its transverse power.
            let read = |swapped: bool, t: i32, x: u16, mu: u16| {
                coefficient(swapped, t, x, mu) / (g.pow(i32::from(x)) * transverse.pow(t))
            };
            if cut == 3 {
                for (swapped, [fifth, mu_cubic, mu2_linear]) in
                    [(true, [15, 17, 19]), (false, [16, 18, 20])]
                {
                    expressions[fifth] = read(swapped, 5, 0, 0);
                    expressions[mu_cubic] = read(swapped, 3, 0, 1);
                    expressions[mu2_linear] = read(swapped, 1, 0, 2);
                }
            } else {
                expressions[20] = read(false, 0, 4, 0);
                expressions[21] = read(false, 0, 2, 1);
                expressions[22] = read(false, 0, 0, 2);
                for (swapped, offset) in [(true, 0), (false, 1)] {
                    expressions[23 + offset] = read(swapped, 4, 0, 0);
                    expressions[25 + offset] = read(swapped, 3, 1, 0);
                    expressions[27 + offset] = read(swapped, 2, 2, 0);
                    expressions[29 + offset] = read(swapped, 1, 3, 0);
                    expressions[31 + offset] = read(swapped, 2, 0, 1);
                    expressions[33 + offset] = read(swapped, 1, 1, 1);
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

#[cfg(feature = "native-kernels")]
impl BubbleSourceProjection {
    /// Select ordinary generic Rust arithmetic from this exact source plan.
    pub fn compile_native<T: oneloop::rust_codegen::runtime::NativeFloat>(
        &self,
        prototype: &T,
    ) -> Result<BubbleSourceProjectionEvaluator<T>>
    where
        Complex<T>: Real + EvaluationDomain,
    {
        let mut result = self.compile(prototype)?;
        crate::native_kernels::bubbles::install(
            &mut result.kernel,
            crate::native_kernels::bubbles::BubblePlan::SourceProjection,
            prototype,
        )?;
        Ok(result)
    }
}
#[cfg(feature = "native-kernels")]
impl<T: Real + SingleFloat> BubbleSourceProjectionEvaluator<T>
where
    Complex<T>: Real + EvaluationDomain,
{
    pub fn backend(&self) -> crate::KernelBackend {
        self.kernel.backend()
    }
}
#[cfg(feature = "native-kernels")]
impl<T: Real + SingleFloat, D: Real + DualNumber<Complex<T>>>
    DualBubbleSourceProjectionEvaluator<T, D>
where
    Complex<T>: Real + EvaluationDomain,
{
    /// No generated Dual projection is selected in the current catalog.
    pub fn backend(&self) -> crate::KernelBackend {
        self.kernel.backend()
    }
}
