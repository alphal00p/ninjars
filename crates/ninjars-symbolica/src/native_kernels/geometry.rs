//! Named tadpole reference probe; candidate scoring and finishes stay in Cuts.
use super::{
    BatchWorkspace, C, Context, NativeFloat, batch, primitives, tadpole_constants as constants,
};
use crate::kernel::{CompiledDualKernel, CompiledKernel, KernelExecutor};
use ninjars::{Complex, Error, Real, Result};
use symbolica::{domains::dual::DualNumber, evaluate::EvaluationDomain};

#[allow(clippy::deref_addrof, clippy::identity_op, clippy::erasing_op)]
#[rustfmt::skip]
#[path = "generated/tadpole_geometry_probe.rs"]
mod ordinary;
#[allow(clippy::deref_addrof, clippy::identity_op, clippy::erasing_op)]
#[rustfmt::skip]
#[path = "generated/tadpole_geometry_probe_dual.rs"]
mod dual;

struct ProbeExecutor<T: NativeFloat> {
    dual: bool,
    context: Context<T>,
    workspace: BatchWorkspace<T>,
}

impl<T: NativeFloat> KernelExecutor<Complex<T>> for ProbeExecutor<T> {
    fn evaluate(&mut self, input: &[C<T>], output: &mut [C<T>], rows: usize) -> Result<()> {
        let evaluate = if self.dual {
            dual::evaluate_batch
        } else {
            ordinary::evaluate_batch
        };
        evaluate(input, output, rows, &self.context, &mut self.workspace).map_err(Error::Symbolic)
    }
}

/// Only the named 8→28 reference probe may install this body. Dimensions are
/// checked at installation; they never select a different generated plan.
pub(crate) fn executor<T: NativeFloat>(
    prototype: &T,
    dual: bool,
) -> Result<Box<dyn KernelExecutor<Complex<T>> + Send + Sync>> {
    Ok(Box::new(ProbeExecutor {
        dual,
        context: Context::new(constants::CONSTANTS, prototype.get_precision())
            .map_err(Error::Symbolic)?,
        workspace: BatchWorkspace::new(prototype.clone()),
    }))
}

pub(crate) fn install<T: NativeFloat>(
    kernel: &mut CompiledKernel<T>,
    prototype: &T,
    dual: bool,
) -> Result<()>
where
    Complex<T>: Real + EvaluationDomain,
{
    let width = if dual { super::DUAL_SHAPE.len() } else { 1 };
    kernel.install_native(executor(prototype, dual)?, 8 * width, 28 * width)
}

pub(crate) fn install_dual<T: NativeFloat, D: Real + DualNumber<Complex<T>>>(
    kernel: &mut CompiledDualKernel<T, D>,
    prototype: &T,
) -> Result<()>
where
    Complex<T>: Real + EvaluationDomain,
{
    let (shape, kernel) = kernel.native_parts();
    if super::supported_shape(&shape.iter().map(Vec::as_slice).collect::<Vec<_>>()) {
        install(kernel, prototype, true)?;
    }
    Ok(())
}
