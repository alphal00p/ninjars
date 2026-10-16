//! Named ordinary bubble-correction kernels; Dual bodies remain interpreted.
use super::{BatchWorkspace, C, ConstantSpec, Context, NativeFloat, batch, primitives};
use crate::kernel::{CompiledKernel, KernelExecutor};
use ninjars::{Complex, Error, ExpansionKind, Real, Result};
use symbolica::evaluate::EvaluationDomain;

#[rustfmt::skip]
#[path = "generated/bubble_constants.rs"]
mod constants;
#[allow(clippy::deref_addrof, clippy::identity_op, clippy::erasing_op)]
#[rustfmt::skip]
#[path = "generated/bubble_source_projection.rs"]
mod source_projection;
#[allow(clippy::deref_addrof, clippy::identity_op, clippy::erasing_op)]
#[rustfmt::skip]
#[path = "generated/bubble_correction_degree0.rs"]
mod degree0;
#[allow(clippy::deref_addrof, clippy::identity_op, clippy::erasing_op)]
#[rustfmt::skip]
#[path = "generated/bubble_correction_degree1.rs"]
mod degree1;
#[allow(clippy::deref_addrof, clippy::identity_op, clippy::erasing_op)]
#[rustfmt::skip]
#[path = "generated/bubble_correction_degree2.rs"]
mod degree2;
#[allow(clippy::deref_addrof, clippy::identity_op, clippy::erasing_op)]
#[rustfmt::skip]
#[path = "generated/bubble_correction_higher.rs"]
mod higher;

/// Semantics originate in an existing, validated exact plan, never buffer sizes.
#[derive(Clone, Copy)]
pub(crate) enum BubblePlan {
    SourceProjection,
    OrdinaryDegree0,
    OrdinaryDegree1,
    OrdinaryDegree2,
    HigherDegree3,
}
impl BubblePlan {
    pub(crate) fn correction(higher_rank: bool, quotient: &ExpansionKind) -> Result<Option<Self>> {
        let ExpansionKind::T2 { powers } = quotient else {
            return Err(Error::Unsupported(
                "native bubble correction requires a T2 quotient".into(),
            ));
        };
        let Some(&(degree, _, _)) = powers.first() else {
            return Ok(None);
        };
        let plan = match (higher_rank, degree) {
            (false, 0) => Self::OrdinaryDegree0,
            (false, 1) => Self::OrdinaryDegree1,
            (false, 2) => Self::OrdinaryDegree2,
            (true, 3) => Self::HigherDegree3,
            _ => {
                return Err(Error::Unsupported(
                    "uncatalogued bubble correction degree".into(),
                ));
            }
        };
        Ok(Some(plan))
    }
    fn dimensions(self) -> (usize, usize) {
        match self {
            Self::SourceProjection => (28, 8),
            Self::OrdinaryDegree0 => (28, 1),
            Self::OrdinaryDegree1 => (28, 3),
            Self::OrdinaryDegree2 => (28, 7),
            Self::HigherDegree3 => (33, 13),
        }
    }
}
struct BubbleExecutor<T: NativeFloat> {
    plan: BubblePlan,
    context: Context<T>,
    workspace: BatchWorkspace<T>,
}
impl<T: NativeFloat> KernelExecutor<Complex<T>> for BubbleExecutor<T> {
    #[cfg(test)]
    fn probe_workspace(&self) -> Option<(usize, usize)> {
        Some((self.workspace.capacity(), self.workspace.storage_len()))
    }

    fn evaluate(&mut self, input: &[C<T>], output: &mut [C<T>], rows: usize) -> Result<()> {
        let evaluate = match self.plan {
            BubblePlan::SourceProjection => source_projection::evaluate_batch,
            BubblePlan::OrdinaryDegree0 => degree0::evaluate_batch,
            BubblePlan::OrdinaryDegree1 => degree1::evaluate_batch,
            BubblePlan::OrdinaryDegree2 => degree2::evaluate_batch,
            BubblePlan::HigherDegree3 => higher::evaluate_batch,
        };
        evaluate(input, output, rows, &self.context, &mut self.workspace).map_err(Error::Symbolic)
    }
}
pub(crate) fn install<T: NativeFloat>(
    kernel: &mut CompiledKernel<T>,
    plan: BubblePlan,
    prototype: &T,
) -> Result<()>
where
    Complex<T>: Real + EvaluationDomain,
{
    let (inputs, outputs) = plan.dimensions();
    let context =
        Context::new(constants::CONSTANTS, prototype.get_precision()).map_err(Error::Symbolic)?;
    kernel.install_native(
        Box::new(BubbleExecutor {
            plan,
            context,
            workspace: BatchWorkspace::new(prototype.clone()),
        }),
        inputs,
        outputs,
    )
}
