//! Explicit, partial native dispatch for named existing Symbolica plans.
//! No dimension-based or carrier-type inference selects generated arithmetic.

pub(crate) mod bubbles;

use ninjars::{Complex, Error, Real, Result};
use oneloop::rust_codegen::runtime::{
    BatchWorkspace, C, ConstantSpec, Context, NativeFloat, batch, primitives,
};
use symbolica::{domains::dual::DualNumber, evaluate::EvaluationDomain};

use crate::kernel::{CompiledDualKernel, CompiledKernel, KernelExecutor};

// Preserve the upstream emitter output byte-for-byte, including neutral
// slot-offset arithmetic and uniform borrow expressions. These allowances
// apply only to generated modules.
#[rustfmt::skip]
#[path = "generated/constants.rs"]
mod constants;
#[allow(clippy::deref_addrof, clippy::identity_op, clippy::erasing_op)]
#[rustfmt::skip]
#[path = "generated/t2.rs"]
mod t2;
#[allow(clippy::deref_addrof, clippy::identity_op, clippy::erasing_op)]
#[rustfmt::skip]
#[path = "generated/t2_dual.rs"]
mod t2_dual;
#[allow(clippy::deref_addrof, clippy::identity_op, clippy::erasing_op)]
#[rustfmt::skip]
#[path = "generated/t3.rs"]
mod t3;
#[allow(clippy::deref_addrof, clippy::identity_op, clippy::erasing_op)]
#[rustfmt::skip]
#[path = "generated/t3_dual.rs"]
mod t3_dual;

/// Execution selected for a kernel family, independently of numerical precision.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KernelBackend {
    SymbolicaInterpreter,
    NativeRust,
}

/// Families covered by the amplitude factory's execution report.
/// Numerator callbacks, geometry providers and masters are separate objects.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KernelFamily {
    T2Denominators,
    T3Denominators,
    BubbleSourceProjection,
    TriangleBubbleCorrection,
    OtherReduction,
    Integration,
    Accumulation,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KernelBackendSelection {
    pub family: KernelFamily,
    pub backend: KernelBackend,
}

pub(crate) fn coverage(native: bool, bubble_corrections: bool) -> [KernelBackendSelection; 7] {
    use KernelFamily::*;
    [
        T2Denominators,
        T3Denominators,
        BubbleSourceProjection,
        TriangleBubbleCorrection,
        OtherReduction,
        Integration,
        Accumulation,
    ]
    .map(|family| KernelBackendSelection {
        family,
        backend: if (native && matches!(family, T2Denominators | T3Denominators))
            || (bubble_corrections
                && matches!(family, BubbleSourceProjection | TriangleBubbleCorrection))
        {
            KernelBackend::NativeRust
        } else {
            KernelBackend::SymbolicaInterpreter
        },
    })
}

/// This identifies semantics and parameter order, not merely a buffer size.
#[derive(Clone, Copy)]
pub(crate) enum DenominatorPlan {
    T2,
    T3,
}
impl DenominatorPlan {
    fn dimensions(self) -> (usize, usize) {
        match self {
            Self::T2 => (29, 7),
            Self::T3 => (23, 4),
        }
    }
}

const DUAL_SHAPE: [[usize; 2]; 5] = [[0, 0], [1, 0], [0, 1], [2, 0], [1, 1]];

pub(crate) fn supported_shape(shape: &[&[usize]]) -> bool {
    shape.len() == DUAL_SHAPE.len() && shape.iter().zip(DUAL_SHAPE).all(|(a, b)| *a == b)
}

struct DenominatorExecutor<T: NativeFloat> {
    plan: DenominatorPlan,
    dual: bool,
    context: Context<T>,
    workspace: BatchWorkspace<T>,
}
impl<T: NativeFloat> KernelExecutor<Complex<T>> for DenominatorExecutor<T> {
    fn evaluate(&mut self, input: &[C<T>], output: &mut [C<T>], rows: usize) -> Result<()> {
        let evaluate = match (self.plan, self.dual) {
            (DenominatorPlan::T2, false) => t2::evaluate_batch,
            (DenominatorPlan::T3, false) => t3::evaluate_batch,
            (DenominatorPlan::T2, true) => t2_dual::evaluate_batch,
            (DenominatorPlan::T3, true) => t3_dual::evaluate_batch,
        };
        evaluate(input, output, rows, &self.context, &mut self.workspace).map_err(Error::Symbolic)
    }
}

fn install<T: NativeFloat>(
    kernel: &mut CompiledKernel<T>,
    plan: DenominatorPlan,
    dual: bool,
    prototype: &T,
) -> Result<()>
where
    Complex<T>: Real + EvaluationDomain,
{
    let components = if dual { DUAL_SHAPE.len() } else { 1 };
    let (inputs, outputs) = plan.dimensions();
    // Exact constants are built at working precision; inputs are never rounded.
    let context =
        Context::new(constants::CONSTANTS, prototype.get_precision()).map_err(Error::Symbolic)?;
    kernel.install_native(
        Box::new(DenominatorExecutor {
            plan,
            dual,
            context,
            workspace: BatchWorkspace::new(prototype.clone()),
        }),
        inputs * components,
        outputs * components,
    )
}

pub(crate) fn install_denominator<T: NativeFloat>(
    kernel: &mut CompiledKernel<T>,
    plan: DenominatorPlan,
    prototype: &T,
) -> Result<()>
where
    Complex<T>: Real + EvaluationDomain,
{
    install(kernel, plan, false, prototype)
}

pub(crate) fn install_dual_denominator<T: NativeFloat, D: Real + DualNumber<Complex<T>>>(
    kernel: &mut CompiledDualKernel<T, D>,
    plan: DenominatorPlan,
    prototype: &T,
) -> Result<()>
where
    Complex<T>: Real + EvaluationDomain,
{
    let (shape, kernel) = kernel.native_parts();
    if shape.len() == DUAL_SHAPE.len()
        && shape.iter().zip(DUAL_SHAPE).all(|(a, b)| a.as_slice() == b)
    {
        install(kernel, plan, true, prototype)?;
    }
    Ok(())
}
