//! Test instrumentation around a mechanically emitted shared conditional body.
use super::{Blocked, ExactProbe, KernelExecutor, same};
use ninjars::{Error, FloatLike, Result};
use oneloop::rust_codegen::runtime::{
    BatchWorkspace, C, ConstantSpec, Context, NativeFloat, batch,
};
use std::cell::Cell;
thread_local! { static CALLS: Cell<usize> = const {Cell::new(0)}; }
mod primitives {
    use super::{C, CALLS, FloatLike, NativeFloat};
    pub(super) use oneloop::rust_codegen::runtime::primitives::truth;
    pub(super) fn powi<T: NativeFloat>(a: &C<T>, power: i64, real: bool) -> C<T> {
        CALLS.set(CALLS.get() + 1);
        assert!(!a.is_fully_zero(), "inactive reciprocal was evaluated");
        oneloop::rust_codegen::runtime::primitives::powi(a, power, real)
    }
}
#[rustfmt::skip]
mod constants;
#[allow(clippy::deref_addrof,clippy::identity_op,clippy::erasing_op,unused_assignments,unused_mut)]
#[rustfmt::skip]
mod body;
struct Executor<T: NativeFloat> {
    policy: crate::BatchExecutionPolicy,
    context: Context<T>,
    workspace: BatchWorkspace<T>,
}
impl<T: NativeFloat> KernelExecutor<C<T>> for Executor<T> {
    fn evaluate(&mut self, input: &[C<T>], output: &mut [C<T>], rows: usize) -> Result<()> {
        if self.policy != crate::BatchExecutionPolicy::Whole {
            return self
                .policy
                .evaluate(
                    input,
                    output,
                    rows,
                    (3, 2),
                    &mut self.workspace,
                    |input, output, rows, workspace| {
                        body::evaluate_batch(input, output, rows, &self.context, workspace)
                    },
                )
                .map_err(Error::Symbolic);
        }
        body::evaluate_batch(input, output, rows, &self.context, &mut self.workspace)
            .map_err(Error::Symbolic)
    }
}
pub(super) fn check<T: ExactProbe>(p: T) {
    check_policy(p, false);
}
pub(super) fn check_policy<T: ExactProbe>(p: T, supported: bool) {
    let make = || Executor {
        policy: crate::BatchExecutionPolicy::Whole,
        context: Context::new(constants::CONSTANTS, p.get_precision()).unwrap(),
        workspace: BatchWorkspace::new(p.clone()),
    };
    for block in [8, 16, 32, 64] {
        let mut whole = make();
        let mut chunked: Box<dyn KernelExecutor<C<T>>> = if supported {
            let mut executor = make();
            executor.workspace.prepare_controlled(129, 99, 13).unwrap();
            executor.policy =
                crate::BatchExecutionPolicy::FixedRows(std::num::NonZeroUsize::new(block).unwrap());
            Box::new(executor)
        } else {
            Box::new(Blocked {
                inner: Box::new(make()),
                inputs: 3,
                outputs: 2,
                block,
            })
        };
        for rows in [0, 1, 3, 8, 9, 17, 33, 65, 3, 1] {
            let input: Vec<_> = (0..rows)
                .flat_map(|i| {
                    let z = C::new(p.zero(), p.zero());
                    let value = C::new(p.from_i64(2 + (i % 3) as i64), p.zero());
                    if i % 2 == 0 {
                        [C::new(p.one(), p.zero()), value, z]
                    } else {
                        [z.clone(), z, value]
                    }
                })
                .collect();
            let mut expected = vec![C::new(p.zero(), p.zero()); rows * 2];
            let mut actual = expected.clone();
            CALLS.set(0);
            whole.evaluate(&input, &mut expected, rows).unwrap();
            assert_eq!(CALLS.get(), rows);
            CALLS.set(0);
            chunked.evaluate(&input, &mut actual, rows).unwrap();
            assert_eq!(CALLS.get(), rows);
            same(&actual, &expected);
        }
    }
}
