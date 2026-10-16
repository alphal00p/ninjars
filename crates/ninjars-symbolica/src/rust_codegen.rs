//! Export existing exact reduction plans through oneloopmaster's Rust emitter.
//!
//! This adapter introduces no algebra or intermediate representation. The
//! exported evaluator is the same Symbolica graph used by the interpreted
//! kernels. The upstream emitter preserves shared bodies and lazy branches.
//! Emitted code evaluates one row; a loop around it is a scalar control, not a
//! replacement for NinjaRS's instruction-major batch execution.

use ninjars::{Error, Result};
use symbolica::domains::dual::DualNumberStructure;

pub use oneloop::rust_codegen::{EmitOptions, GeneratedModule, RustEmitter, runtime};

use crate::{DenominatorExpansion, LaurentDivision, TriangleBubbleSubtraction};

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
        }
    };
}

export_kernel!(LaurentDivision);
export_kernel!(TriangleBubbleSubtraction);
export_kernel!(DenominatorExpansion);
