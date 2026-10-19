//! Existing tadpole source projections and reference selection expressions.
//!
//! Shared FeynKit momenta supply every contraction and Weyl operation. These
//! plans expose the same expressions to the interpreter and mechanical emitter;
//! they do not select a numerical backend or alter chart selection.

use ninjars::{Complex, Error, FourMomentum, Real, Result, SingleFloat};
use symbolica::{atom::Atom, domains::dual::DualNumber, evaluate::EvaluationDomain, symbol};

use crate::kernel::{CompiledDualKernel, CompiledKernel, ExactKernel};

fn momentum(p: &[Atom], start: usize) -> FourMomentum<Atom> {
    FourMomentum::from_args(
        p[start].clone(),
        p[start + 1].clone(),
        p[start + 2].clone(),
        p[start + 3].clone(),
    )
}

/// Project the tadpole sampling line on a parent triangle or bubble's axes.
///
/// Rows contain `[a, e, f, V_parent, E3, E4]` for a triangle (24 columns),
/// or `[a, e, f, V_parent, E2, E3, E4]` for a bubble (28 columns). Every
/// momentum has the shared `[E,x,y,z]` component order. Output is
/// `[(a+V_parent)·E, e·E, f·E]` for each parent axis in the given order.
pub struct TadpoleSourceProjection {
    source_size: usize,
    pub(crate) exact: ExactKernel,
}

impl TadpoleSourceProjection {
    pub fn new(source_size: usize) -> Result<Self> {
        let (axes, scope) = match source_size {
            3 => (2, "triangle_axes"),
            2 => (3, "bubble_axes"),
            _ => {
                return Err(Error::Unsupported(
                    "tadpole source projection requires a triangle or bubble".into(),
                ));
            }
        };
        let parameters: Vec<_> = (0..16 + 4 * axes)
            .map(|i| Atom::var(symbol!(format!("ninjars_tadpole_{scope}::p{i}"))))
            .collect();
        let offset = momentum(&parameters, 0) + momentum(&parameters, 12);
        let mut expressions = Vec::new();
        for axis in 0..axes {
            let e = momentum(&parameters, 16 + axis * 4);
            expressions.extend([
                offset.dot(&e),
                momentum(&parameters, 4).dot(&e),
                momentum(&parameters, 8).dot(&e),
            ]);
        }
        Ok(Self {
            source_size,
            exact: ExactKernel {
                expressions,
                parameters,
            },
        })
    }

    pub fn source_size(&self) -> usize {
        self.source_size
    }
}

/// `(a, b, c, d)` of the six tadpole bases: the Weyl components
/// `(1, a + i b)` and `(1, c + i d)` of `e1` and `e2`.
pub(crate) const TADPOLE_REFERENCE_SPINORS: [(i64, i64, i64, i64); 6] = [
    (1, 2, -2, 1),
    (2, -1, 1, 3),
    (-1, 3, 2, 1),
    (3, 1, -1, -2),
    (1, -3, 3, -2),
    (2, 3, -3, 1),
];

// The existing six exact rational Weyl choices. Probe and finish construction
// use this one source so their candidate indices and normalizations stay equal.
pub(crate) fn tadpole_reference_bases() -> [[FourMomentum<Atom>; 4]; 6] {
    TADPOLE_REFERENCE_SPINORS.map(|(a, b, c, d)| {
        let angle1 = [Atom::num(1), Atom::num(a) + Atom::i() * b];
        let square1 = [Atom::num(1), Atom::num(a) - Atom::i() * b];
        let angle2 = [Atom::num(1), Atom::num(c) + Atom::i() * d];
        let square2 = [Atom::num(1), Atom::num(c) - Atom::i() * d];
        [
            FourMomentum::from_weyl_spinors(&angle1, &square1),
            FourMomentum::from_weyl_spinors(&angle2, &square2),
            FourMomentum::from_weyl_spinors(&angle1, &square2),
            FourMomentum::from_weyl_spinors(&angle2, &square1),
        ]
    })
}

/// Complementary-shift probe used to choose a tadpole's null reference basis.
///
/// Input is `[V_target, V_complement]` (eight components). Output is their
/// difference `V_complement - V_target` followed by its four axis projections
/// for each of the six existing reference bases (28 columns total). Numerical
/// conditioning and candidate selection remain in the cut provider. This is
/// the probe, not the separate five-input geometry finish.
pub struct TadpoleGeometryProbe {
    pub(crate) exact: ExactKernel,
}

impl Default for TadpoleGeometryProbe {
    fn default() -> Self {
        Self::new()
    }
}

impl TadpoleGeometryProbe {
    pub fn new() -> Self {
        let parameters: Vec<_> = (0..8)
            .map(|i| Atom::var(symbol!(format!("ninjars_cut_kernel::p{i}"))))
            .collect();
        let difference = momentum(&parameters, 4) - momentum(&parameters, 0);
        let mut expressions: Vec<_> = difference.components().into_iter().cloned().collect();
        for basis in tadpole_reference_bases() {
            expressions.extend(basis.iter().map(|axis| difference.dot(axis)));
        }
        Self {
            exact: ExactKernel {
                expressions,
                parameters,
            },
        }
    }
}

macro_rules! evaluator {
    ($plan:ty) => {
        impl $plan {
            pub fn input_columns(&self) -> usize {
                self.exact.parameters.len()
            }

            pub fn output_columns(&self) -> usize {
                self.exact.expressions.len()
            }

            pub fn compile<T: Real + SingleFloat>(
                &self,
                prototype: &T,
            ) -> Result<TadpolePlanEvaluator<T>>
            where
                Complex<T>: Real + EvaluationDomain,
            {
                Ok(TadpolePlanEvaluator {
                    kernel: self.exact.compile(prototype)?,
                })
            }

            pub fn compile_dual<T: Real + SingleFloat, D: Real + DualNumber<Complex<T>>>(
                &self,
                prototype: &T,
                dual: &D,
            ) -> Result<DualTadpolePlanEvaluator<T, D>>
            where
                Complex<T>: Real + EvaluationDomain,
            {
                Ok(DualTadpolePlanEvaluator {
                    kernel: self.exact.compile_dual(prototype, dual)?,
                })
            }
        }
    };
}
evaluator!(TadpoleSourceProjection);
evaluator!(TadpoleGeometryProbe);

/// Batched evaluation in the originating tadpole plan's documented row order.
pub struct TadpolePlanEvaluator<T> {
    kernel: CompiledKernel<T>,
}

impl<T: Real + SingleFloat> TadpolePlanEvaluator<T>
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

/// Symbolica's full Dual transform of the originating tadpole plan.
pub struct DualTadpolePlanEvaluator<T, D> {
    kernel: CompiledDualKernel<T, D>,
}

impl<T: Real + SingleFloat, D: Real + DualNumber<Complex<T>>> DualTadpolePlanEvaluator<T, D>
where
    Complex<T>: Real + EvaluationDomain,
{
    pub fn evaluate_batch(&mut self, input: &[D], output: &mut [D], rows: usize) -> Result<()> {
        self.kernel.evaluate_batch(input, output, rows)
    }
}

#[cfg(feature = "native-kernels")]
impl TadpoleSourceProjection {
    /// Select the generated ordinary source-axis plan. This does not select
    /// the independent geometry reference probe or its cut provider.
    pub fn compile_native<T: oneloop::rust_codegen::runtime::NativeFloat>(
        &self,
        prototype: &T,
    ) -> Result<TadpolePlanEvaluator<T>>
    where
        Complex<T>: Real + EvaluationDomain,
    {
        use crate::native_kernels::tadpoles::{TadpoleSourcePlan, install_projection};
        let mut result = self.compile(prototype)?;
        install_projection(
            &mut result.kernel,
            TadpoleSourcePlan::for_source(self.source_size)?,
            prototype,
        )?;
        Ok(result)
    }

    /// Select the exact ordered five-component Dual body when available.
    /// Other complete shapes retain the existing interpreter evaluator.
    pub fn compile_dual_native<
        T: oneloop::rust_codegen::runtime::NativeFloat,
        D: Real + DualNumber<Complex<T>>,
    >(
        &self,
        prototype: &T,
        dual: &D,
    ) -> Result<DualTadpolePlanEvaluator<T, D>>
    where
        Complex<T>: Real + EvaluationDomain,
    {
        use crate::native_kernels::tadpoles::{TadpoleSourcePlan, install_dual_projection};
        let mut result = self.compile_dual(prototype, dual)?;
        install_dual_projection(
            &mut result.kernel,
            TadpoleSourcePlan::for_source(self.source_size)?,
            prototype,
        )?;
        Ok(result)
    }
}

#[cfg(feature = "native-kernels")]
impl<T: Real + SingleFloat> TadpolePlanEvaluator<T>
where
    Complex<T>: Real + EvaluationDomain,
{
    pub fn backend(&self) -> crate::KernelBackend {
        self.kernel.backend()
    }
}

#[cfg(feature = "native-kernels")]
impl<T: Real + SingleFloat, D: Real + DualNumber<Complex<T>>> DualTadpolePlanEvaluator<T, D>
where
    Complex<T>: Real + EvaluationDomain,
{
    pub fn backend(&self) -> crate::KernelBackend {
        self.kernel.backend()
    }
}
