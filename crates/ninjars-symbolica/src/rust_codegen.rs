//! Export existing exact reduction plans through oneloopmaster's Rust emitter.
//!
//! This adapter introduces no algebra or intermediate representation. The
//! exported evaluator is the same Symbolica graph used by the interpreted
//! kernels. The upstream emitter preserves shared bodies and lazy branches.
//! Scalar exports evaluate one row. Batch exports additionally evaluate every
//! instruction across the batch using an upstream reusable workspace.

use ninjars::{Error, Result};
use symbolica::domains::dual::DualNumberStructure;

pub use crate::rank_specialization::RankSpecializedCorrectionPlan;
pub use oneloop::rust_codegen::{EmitOptions, GeneratedModule, RustEmitter, runtime};

use crate::{
    BubbleSourceProjection, DenominatorExpansion, LaurentDivision, TadpoleGeometryProbe,
    TadpoleSourceProjection, TriangleBubbleSubtraction,
};

/// Mechanical Rust export for a prepared reduction kernel.
///
/// Reuse one emitter across kernels, then export its constant table after the
/// last module. Construct the upstream `runtime::Context` once at working
/// precision. Input/output orders are unchanged from the kernel's existing
/// layouts. Callers retain responsibility for shape and nonsingularity checks.
pub trait RustKernelExport {
    fn export_rust(
        &self,
        emitter: &mut RustEmitter,
        options: EmitOptions,
    ) -> Result<GeneratedModule>;

    /// Export Symbolica's flattened Dual transformation of the exact graph.
    ///
    /// An already constructed shared Dual structure supplies derivative order
    /// and shape. For each ordinary input/output, all components occur
    /// contiguously in `DualNumberStructure::get_shape()` order. Runtime
    /// arithmetic uses the underlying complex scalar carrier.
    fn export_rust_dual<D: DualNumberStructure + Clone>(
        &self,
        structure: &D,
        emitter: &mut RustEmitter,
        options: EmitOptions,
    ) -> Result<GeneratedModule>;

    /// Export instruction-major batch arithmetic, including a scalar fast path.
    ///
    /// The generated `evaluate_batch` takes row-major inputs and outputs, a row
    /// count, the shared constant context, and a reusable `runtime::BatchWorkspace`.
    /// Buffer dimensions are checked before output writes. Export fails if the
    /// pinned upstream emitter does not support an instruction in the graph.
    fn export_rust_batch(
        &self,
        emitter: &mut RustEmitter,
        options: EmitOptions,
    ) -> Result<GeneratedModule>;

    /// Export the same batch execution after Symbolica's Dual transformation.
    ///
    /// Component ordering matches [`Self::export_rust_dual`]. Derivative
    /// components remain contiguous within each parameter and each batch row.
    fn export_rust_batch_dual<D: DualNumberStructure + Clone>(
        &self,
        structure: &D,
        emitter: &mut RustEmitter,
        options: EmitOptions,
    ) -> Result<GeneratedModule>;
}

macro_rules! export_kernel {
    ($kernel:ty) => {
        impl RustKernelExport for $kernel {
            fn export_rust(
                &self,
                emitter: &mut RustEmitter,
                options: EmitOptions,
            ) -> Result<GeneratedModule> {
                emitter
                    .emit(self.exact.evaluator()?, options)
                    .map_err(Error::Symbolic)
            }

            fn export_rust_dual<D: DualNumberStructure + Clone>(
                &self,
                structure: &D,
                emitter: &mut RustEmitter,
                options: EmitOptions,
            ) -> Result<GeneratedModule> {
                emitter
                    .emit(self.exact.dual_evaluator(structure)?, options)
                    .map_err(Error::Symbolic)
            }

            fn export_rust_batch(
                &self,
                emitter: &mut RustEmitter,
                options: EmitOptions,
            ) -> Result<GeneratedModule> {
                emitter
                    .emit_batch(self.exact.evaluator()?, options)
                    .map_err(Error::Symbolic)
            }

            fn export_rust_batch_dual<D: DualNumberStructure + Clone>(
                &self,
                structure: &D,
                emitter: &mut RustEmitter,
                options: EmitOptions,
            ) -> Result<GeneratedModule> {
                emitter
                    .emit_batch(self.exact.dual_evaluator(structure)?, options)
                    .map_err(Error::Symbolic)
            }
        }
    };
}

export_kernel!(LaurentDivision);
export_kernel!(TriangleBubbleSubtraction);
export_kernel!(DenominatorExpansion);
export_kernel!(BubbleSourceProjection);
export_kernel!(TadpoleSourceProjection);
export_kernel!(TadpoleGeometryProbe);

/// Export only the named guarded bodies; runtime guard enforcement remains in
/// the stage adapters. No public unguarded evaluator constructor is introduced.
impl RustKernelExport for RankSpecializedCorrectionPlan {
    fn export_rust(
        &self,
        emitter: &mut RustEmitter,
        options: EmitOptions,
    ) -> Result<GeneratedModule> {
        emitter
            .emit(self.prepare()?.exact.evaluator()?, options)
            .map_err(Error::Symbolic)
    }

    fn export_rust_dual<D: DualNumberStructure + Clone>(
        &self,
        structure: &D,
        emitter: &mut RustEmitter,
        options: EmitOptions,
    ) -> Result<GeneratedModule> {
        emitter
            .emit(self.prepare()?.exact.dual_evaluator(structure)?, options)
            .map_err(Error::Symbolic)
    }

    fn export_rust_batch(
        &self,
        emitter: &mut RustEmitter,
        options: EmitOptions,
    ) -> Result<GeneratedModule> {
        emitter
            .emit_batch(self.prepare()?.exact.evaluator()?, options)
            .map_err(Error::Symbolic)
    }

    fn export_rust_batch_dual<D: DualNumberStructure + Clone>(
        &self,
        structure: &D,
        emitter: &mut RustEmitter,
        options: EmitOptions,
    ) -> Result<GeneratedModule> {
        emitter
            .emit_batch(self.prepare()?.exact.dual_evaluator(structure)?, options)
            .map_err(Error::Symbolic)
    }
}
