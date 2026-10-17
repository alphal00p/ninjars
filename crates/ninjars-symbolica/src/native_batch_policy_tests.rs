//! Supported owner policy through the actual generated reduction executors.
use super::*;
use crate::{BatchExecutionPolicy, rank_specialization::RankSpecializedCorrectionPlan};
use std::num::NonZeroUsize;

#[derive(Clone, Copy)]
enum Plan {
    Generic(Case),
    Rank(RankSpecializedCorrectionPlan),
}

impl Plan {
    fn make<T: NativeFloat>(self, p: &T, dual: bool) -> CompiledKernel<T>
    where
        Complex<T>: Real + EvaluationDomain,
    {
        match self {
            Self::Generic(case) => case.kernel(p, dual, None),
            Self::Rank(plan) => {
                let exact = plan.prepare().unwrap().exact;
                if dual {
                    let zero = Complex::new(p.zero(), p.zero());
                    let prototype = HyperDual::from_values(shape(), vec![zero; 5]);
                    let mut kernel = exact.compile_dual(p, &prototype).unwrap();
                    native_kernels::rank_specialization::install_dual(&mut kernel, plan, p)
                        .unwrap();
                    kernel.kernel
                } else {
                    let mut kernel = exact.compile(p).unwrap();
                    native_kernels::rank_specialization::install_ordinary(&mut kernel, plan, p)
                        .unwrap();
                    kernel
                }
            }
        }
    }
}

fn changing_policy<T: ExactProbe>(p: T)
where
    Complex<T>: Real + EvaluationDomain,
{
    use RankSpecializedCorrectionPlan::*;
    let plans = cases().into_iter().map(Plan::Generic).chain([
        Plan::Rank(TriangleToBubbleN5R4),
        Plan::Rank(TriangleToTadpoleN5R4),
        Plan::Rank(BubbleToTadpoleN5R4),
    ]);
    for plan in plans {
        for dual in [false, true] {
            if dual && matches!(plan, Plan::Generic(case) if !case.has_dual()) {
                continue;
            }
            let mut expected_kernel = plan.make(&p, dual);
            let mut actual_kernel = plan.make(&p, dual);
            for (block, rows) in [
                (None, 129),
                (Some(32), 65),
                (Some(8), 17),
                (Some(8), 3),
                (Some(8), 0),
                (Some(16), 33),
                (None, 129),
                (Some(4), 0),
                (Some(4), 1),
            ] {
                let policy = block.map_or(BatchExecutionPolicy::Whole, |rows| {
                    BatchExecutionPolicy::FixedRows(NonZeroUsize::new(rows).unwrap())
                });
                actual_kernel.set_native_batch_execution(policy);
                let input = inputs(&p, actual_kernel.inputs, rows);
                let mut expected =
                    vec![Complex::new(p.zero(), p.zero()); rows * actual_kernel.outputs];
                let mut actual = expected.clone();
                let before = actual_kernel.native.as_ref().unwrap().probe_workspace();
                assert!(
                    actual_kernel
                        .evaluate_batch(&input, &mut actual, usize::MAX)
                        .is_err()
                );
                same(&actual, &expected);
                assert_eq!(
                    actual_kernel.native.as_ref().unwrap().probe_workspace(),
                    before
                );
                expected_kernel
                    .evaluate_batch(&input, &mut expected, rows)
                    .unwrap();
                actual_kernel
                    .evaluate_batch(&input, &mut actual, rows)
                    .unwrap();
                same(&actual, &expected);
                if let Some(bound) = block {
                    let (capacity, cells) = actual_kernel
                        .native
                        .as_ref()
                        .unwrap()
                        .probe_workspace()
                        .unwrap();
                    assert!(capacity <= bound);
                    if rows == 0 && bound == 4 {
                        assert_eq!((capacity, cells), (0, 0));
                    }
                }
            }
        }
    }
}

#[test]
fn supported_policy_bounds_previously_grown_native_and_dual_rank_workspaces() {
    changing_policy(0f64);
    changing_policy(DoubleFloat::default());
    changing_policy(Float::new(192));
}

#[test]
fn supported_policy_factory_selection_is_explicit_and_order_independent() {
    use crate::{KernelBackend, SymbolicaAmplitudeKernels, SymbolicaDualAmplitudeKernels};
    let policy = BatchExecutionPolicy::FixedRows(NonZeroUsize::new(32).unwrap());
    for before in [false, true] {
        let mut kernels = SymbolicaAmplitudeKernels::new(5, 4, 0.).unwrap();
        assert_eq!(
            kernels.native_batch_execution(),
            BatchExecutionPolicy::Whole
        );
        if before {
            kernels = kernels.with_native_batch_execution(policy);
        }
        kernels = kernels
            .with_native_kernels()
            .with_native_rank_specialized_corrections()
            .unwrap();
        if !before {
            kernels = kernels.with_native_batch_execution(policy);
        }
        assert_eq!(kernels.native_batch_execution(), policy);
        assert_eq!(
            kernels.rank_specialized_correction_backend(),
            Some(KernelBackend::NativeRust)
        );
        assert_eq!(
            kernels.with_native_denominators().native_batch_execution(),
            policy
        );
    }
    for mut dual_shape in [
        shape(),
        vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![2, 0], vec![1, 1]],
    ] {
        let supported = dual_shape == shape();
        let prototype = HyperDual::from_values(
            std::mem::take(&mut dual_shape),
            vec![Complex::new(0., 0.); 5],
        );
        for before in [false, true] {
            let mut kernels = SymbolicaDualAmplitudeKernels::new(5, 4, 0., &prototype).unwrap();
            assert_eq!(
                kernels.native_batch_execution(),
                BatchExecutionPolicy::Whole
            );
            if before {
                kernels = kernels.with_native_batch_execution(policy);
            }
            kernels = kernels
                .with_native_kernels()
                .with_native_rank_specialized_corrections()
                .unwrap();
            if !before {
                kernels = kernels.with_native_batch_execution(policy);
            }
            assert_eq!(kernels.native_batch_execution(), policy);
            let expected = if supported {
                KernelBackend::NativeRust
            } else {
                KernelBackend::SymbolicaInterpreter
            };
            assert_eq!(
                kernels.rank_specialized_correction_backend(),
                Some(expected)
            );
            if !supported {
                assert!(
                    kernels
                        .kernel_backends()
                        .iter()
                        .all(|entry| entry.backend == expected)
                );
            }
        }
    }
}
