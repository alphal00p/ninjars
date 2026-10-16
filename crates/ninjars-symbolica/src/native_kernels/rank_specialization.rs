//! Explicit native dispatch for the same guarded n5/r4 exact correction plans.
use super::{BatchWorkspace, C, ConstantSpec, Context, DUAL_SHAPE, NativeFloat, batch, primitives};
use crate::{
    kernel::{CompiledDualKernel, CompiledKernel, KernelExecutor},
    rank_specialization::RankSpecializedCorrectionPlan,
};
use ninjars::{Complex, Error, Real, Result};
use symbolica::{domains::dual::DualNumber, evaluate::EvaluationDomain};

#[rustfmt::skip]
#[path = "generated/rank_specialization_constants.rs"]
mod constants;
#[allow(clippy::deref_addrof, clippy::identity_op, clippy::erasing_op)]
#[rustfmt::skip]
#[path = "generated/rank_triangle_to_bubble_n5_r4.rs"]
mod triangle_bubble;
#[allow(clippy::deref_addrof, clippy::identity_op, clippy::erasing_op)]
#[rustfmt::skip]
#[path = "generated/rank_triangle_to_bubble_n5_r4_dual.rs"]
mod triangle_bubble_dual;
#[allow(clippy::deref_addrof, clippy::identity_op, clippy::erasing_op)]
#[rustfmt::skip]
#[path = "generated/rank_triangle_to_tadpole_n5_r4.rs"]
mod triangle_tadpole;
#[allow(clippy::deref_addrof, clippy::identity_op, clippy::erasing_op)]
#[rustfmt::skip]
#[path = "generated/rank_triangle_to_tadpole_n5_r4_dual.rs"]
mod triangle_tadpole_dual;
#[allow(clippy::deref_addrof, clippy::identity_op, clippy::erasing_op)]
#[rustfmt::skip]
#[path = "generated/rank_bubble_to_tadpole_n5_r4.rs"]
mod bubble_tadpole;
#[allow(clippy::deref_addrof, clippy::identity_op, clippy::erasing_op)]
#[rustfmt::skip]
#[path = "generated/rank_bubble_to_tadpole_n5_r4_dual.rs"]
mod bubble_tadpole_dual;

#[cfg(test)]
thread_local! {
    pub(crate) static ROWS: std::cell::Cell<[usize; 6]> = const { std::cell::Cell::new([0; 6]) };
    pub(crate) static CONSTANT_PRECISIONS: std::cell::RefCell<Vec<Vec<u32>>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn dimensions(plan: RankSpecializedCorrectionPlan) -> (usize, usize) {
    use RankSpecializedCorrectionPlan::*;
    match plan {
        TriangleToBubbleN5R4 => (28, 3),
        TriangleToTadpoleN5R4 => (25, 1),
        BubbleToTadpoleN5R4 => (24, 1),
    }
}

struct CorrectionExecutor<T: NativeFloat> {
    plan: RankSpecializedCorrectionPlan,
    dual: bool,
    context: Context<T>,
    workspace: BatchWorkspace<T>,
}

impl<T: NativeFloat> KernelExecutor<Complex<T>> for CorrectionExecutor<T> {
    fn evaluate(&mut self, input: &[C<T>], output: &mut [C<T>], rows: usize) -> Result<()> {
        use RankSpecializedCorrectionPlan::*;
        #[cfg(test)]
        ROWS.with(|counter| {
            let kind = match self.plan {
                TriangleToBubbleN5R4 => 0,
                TriangleToTadpoleN5R4 => 1,
                BubbleToTadpoleN5R4 => 2,
            };
            let mut counts = counter.get();
            counts[2 * kind + usize::from(self.dual)] += rows;
            counter.set(counts);
        });
        let evaluate = match (self.plan, self.dual) {
            (TriangleToBubbleN5R4, false) => triangle_bubble::evaluate_batch,
            (TriangleToBubbleN5R4, true) => triangle_bubble_dual::evaluate_batch,
            (TriangleToTadpoleN5R4, false) => triangle_tadpole::evaluate_batch,
            (TriangleToTadpoleN5R4, true) => triangle_tadpole_dual::evaluate_batch,
            (BubbleToTadpoleN5R4, false) => bubble_tadpole::evaluate_batch,
            (BubbleToTadpoleN5R4, true) => bubble_tadpole_dual::evaluate_batch,
        };
        evaluate(input, output, rows, &self.context, &mut self.workspace).map_err(Error::Symbolic)
    }
}

fn install<T: NativeFloat>(
    kernel: &mut CompiledKernel<T>,
    plan: RankSpecializedCorrectionPlan,
    dual: bool,
    prototype: &T,
) -> Result<()>
where
    Complex<T>: Real + EvaluationDomain,
{
    let (inputs, outputs) = dimensions(plan);
    let components = if dual { DUAL_SHAPE.len() } else { 1 };
    let context: Context<T> =
        Context::new(constants::CONSTANTS, prototype.get_precision()).map_err(Error::Symbolic)?;
    #[cfg(test)]
    CONSTANT_PRECISIONS.with(|records| {
        records.borrow_mut().push(
            (0..constants::CONSTANTS.len())
                .flat_map(|index| {
                    let constant = context.constant(index);
                    [constant.re.get_precision(), constant.im.get_precision()]
                })
                .collect(),
        );
    });
    kernel.install_native(
        Box::new(CorrectionExecutor {
            plan,
            dual,
            context,
            workspace: BatchWorkspace::new(prototype.clone()),
        }),
        inputs * components,
        outputs * components,
    )
}

pub(crate) fn install_ordinary<T: NativeFloat>(
    kernel: &mut CompiledKernel<T>,
    plan: RankSpecializedCorrectionPlan,
    prototype: &T,
) -> Result<()>
where
    Complex<T>: Real + EvaluationDomain,
{
    install(kernel, plan, false, prototype)
}

pub(crate) fn install_dual<T: NativeFloat, D: Real + DualNumber<Complex<T>>>(
    kernel: &mut CompiledDualKernel<T, D>,
    plan: RankSpecializedCorrectionPlan,
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
