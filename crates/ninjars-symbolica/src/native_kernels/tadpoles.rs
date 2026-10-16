//! Named tadpole source-axis projections; geometry selection is independent.
use super::{BatchWorkspace, C, ConstantSpec, Context, DUAL_SHAPE, NativeFloat, batch, primitives};
use crate::kernel::{CompiledDualKernel, CompiledKernel, KernelExecutor};
use ninjars::{Complex, Error, Real, Result};
use symbolica::{domains::dual::DualNumber, evaluate::EvaluationDomain};

#[rustfmt::skip]
#[path = "generated/tadpole_constants.rs"]
mod constants;
#[allow(clippy::deref_addrof, clippy::identity_op, clippy::erasing_op)]
#[rustfmt::skip]
#[path = "generated/tadpole_triangle_projection.rs"]
mod triangle;
#[allow(clippy::deref_addrof, clippy::identity_op, clippy::erasing_op)]
#[rustfmt::skip]
#[path = "generated/tadpole_triangle_projection_dual.rs"]
mod triangle_dual;
#[allow(clippy::deref_addrof, clippy::identity_op, clippy::erasing_op)]
#[rustfmt::skip]
#[path = "generated/tadpole_bubble_projection.rs"]
mod bubble;
#[allow(clippy::deref_addrof, clippy::identity_op, clippy::erasing_op)]
#[rustfmt::skip]
#[path = "generated/tadpole_bubble_projection_dual.rs"]
mod bubble_dual;

/// Parent identity fixes axis order, independently of numerical dimensions.
#[derive(Clone, Copy)]
pub(crate) enum TadpoleSourcePlan {
    Triangle,
    Bubble,
}

impl TadpoleSourcePlan {
    pub(crate) fn for_source(source_size: usize) -> Result<Self> {
        match source_size {
            3 => Ok(Self::Triangle),
            2 => Ok(Self::Bubble),
            _ => Err(Error::Unsupported(
                "native tadpole projection requires a triangle or bubble source".into(),
            )),
        }
    }

    fn dimensions(self) -> (usize, usize) {
        match self {
            Self::Triangle => (24, 6),
            Self::Bubble => (28, 9),
        }
    }
}

struct ProjectionExecutor<T: NativeFloat> {
    plan: TadpoleSourcePlan,
    dual: bool,
    context: Context<T>,
    workspace: BatchWorkspace<T>,
}

impl<T: NativeFloat> KernelExecutor<Complex<T>> for ProjectionExecutor<T> {
    fn evaluate(&mut self, input: &[C<T>], output: &mut [C<T>], rows: usize) -> Result<()> {
        let evaluate = match (self.plan, self.dual) {
            (TadpoleSourcePlan::Triangle, false) => triangle::evaluate_batch,
            (TadpoleSourcePlan::Triangle, true) => triangle_dual::evaluate_batch,
            (TadpoleSourcePlan::Bubble, false) => bubble::evaluate_batch,
            (TadpoleSourcePlan::Bubble, true) => bubble_dual::evaluate_batch,
        };
        evaluate(input, output, rows, &self.context, &mut self.workspace).map_err(Error::Symbolic)
    }
}

fn install<T: NativeFloat>(
    kernel: &mut CompiledKernel<T>,
    plan: TadpoleSourcePlan,
    dual: bool,
    prototype: &T,
) -> Result<()>
where
    Complex<T>: Real + EvaluationDomain,
{
    let (inputs, outputs) = plan.dimensions();
    let components = if dual { DUAL_SHAPE.len() } else { 1 };
    let context =
        Context::new(constants::CONSTANTS, prototype.get_precision()).map_err(Error::Symbolic)?;
    kernel.install_native(
        Box::new(ProjectionExecutor {
            plan,
            dual,
            context,
            workspace: BatchWorkspace::new(prototype.clone()),
        }),
        inputs * components,
        outputs * components,
    )
}

pub(crate) fn install_projection<T: NativeFloat>(
    kernel: &mut CompiledKernel<T>,
    plan: TadpoleSourcePlan,
    prototype: &T,
) -> Result<()>
where
    Complex<T>: Real + EvaluationDomain,
{
    install(kernel, plan, false, prototype)
}

pub(crate) fn install_dual_projection<T: NativeFloat, D: Real + DualNumber<Complex<T>>>(
    kernel: &mut CompiledDualKernel<T, D>,
    plan: TadpoleSourcePlan,
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
