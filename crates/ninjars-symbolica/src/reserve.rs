//! A process's symbols for history-independent exact evaluators.
//!
//! Symbolica orders the terms of an expression, and breaks ties while building
//! an evaluator, by the encoded bytes of its symbols, which hold their
//! registration indices (low byte first). An exact evaluator's instructions,
//! and so its rounding in every carrier, therefore depend on the IDs of the
//! symbols its expressions contain. NinjaRS keeps those IDs out of its
//! results in two ways:
//!
//! * the numerator, cut, kernel and graph evaluators are built on positional
//!   slots ([`ninjars_positional`]), which give the same instructions in every
//!   process, provided the slot pool holds two-byte variable slots;
//! * the master-integral evaluators ([`crate::SymbolicaMasters`]: every family
//!   in arbitrary precision, the tadpole tensors in double-double) are built
//!   from `ninjars::master_argument_*` and oneloopmaster's expressions and
//!   helper functions, which keep their own symbols.
//!
//! Both hold only if these symbols get the same IDs in every process: the
//! slot pool must be built, and the masters' symbols registered, before the
//! process registers symbols of its own. Otherwise a double-double or
//! arbitrary-precision result depends on what the process did before (in a
//! host that imported a saved Symbolica state of 20,000 symbols before its
//! first reduction, double-double values moved by 1e-28 relative; past about
//! 61,000 symbols the pool gets function slots only, and values moved by up
//! to 3e-16 at ill-conditioned points).
//!
//! **Contract.** A process that wants NinjaRS's double-double and
//! arbitrary-precision results independent of its history calls [`reserve`]
//! before it registers any symbol of its own (parsing, importing a state,
//! defining functions), and fails if it returns an error. [`reserve`] neither
//! compiles nor evaluates anything; binary64 reductions with the fast engine
//! never need it.

use ninjars_positional::{Error as PositionalError, Regime};
use std::sync::OnceLock;

/// Build the positional slot pool and register every master-integral symbol,
/// once per process (see the module documentation); later calls return the
/// first call's result. The error is [`ninjars_positional::Error::FunctionSlots`]
/// when the pool could not get two-byte variable slots (too many symbols had
/// been registered before the first call): NinjaRS's exact results then
/// depend on the process's history.
pub fn reserve() -> Result<Regime, PositionalError> {
    static RESERVED: OnceLock<Result<Regime, PositionalError>> = OnceLock::new();
    RESERVED
        .get_or_init(|| {
            let _license = ninjars::licensing::unlock_for_ninjars_operation();
            let regime = ninjars_positional::reserve();
            register_master_symbols()?;
            regime
        })
        .clone()
}

/// Register every symbol an exact master-integral evaluator refers to, in one
/// fixed order, without compiling one: oneloopmaster's shared definitions,
/// then each family's arguments and series ([`crate::masters::exact_families`]).
fn register_master_symbols() -> Result<(), PositionalError> {
    let _ = oneloop::OneLoopExpressions::new();
    for integral in crate::masters::exact_families() {
        crate::masters::master_series(integral)
            .map_err(|error| PositionalError::Symbolic(format!("{integral:?}: {error}")))?;
    }
    Ok(())
}
