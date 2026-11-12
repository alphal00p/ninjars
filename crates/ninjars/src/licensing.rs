//! Scoped Symbolica authorization for NinjaRS and its supported adapters.
//!
//! The supplied library key is registered by the `ninjars` crate, whose name
//! matches the package authorized by Symbolica. The persistent unlock stays
//! private. NinjaRS workspace crates and supported interfaces such as MadLoop
//! acquire a thread-bound guard only around their NinjaRS operations, including
//! preparation and oneloopmaster evaluation. This is not an application-wide
//! license for unrelated Symbolica computations.

use std::sync::LazyLock;

use symbolica::license::{LibraryUnlock, LibraryUnlockGuard};

// Issued by Symbolica for NinjaRS, including use through its supported
// interfaces. Keep registration in this crate: set_library_key! binds the
// signed package name to CARGO_CRATE_NAME.
const KEY: [&str; 3] = [
    "SU-465-ninjars-2028.03.01-",
    "R2VDDBO3ZCD5UDAUGUKXGX4ZEUZJTXWQPWKZJHVSND6MAK2EBVMQ2F3Q",
    "GIVISCZUA6AIYSUQEKDHG47SGY7OEQPIHXEHBT4JABO5SCQ",
];

static UNLOCK: LazyLock<Option<LibraryUnlock>> =
    LazyLock::new(|| symbolica::set_library_key!(&KEY.concat()).ok());

/// Authorize the current thread for a NinjaRS operation until the guard drops.
///
/// Internal integration API for NinjaRS workspace crates and supported
/// adapters. Acquire the guard before creating symbolic kernels or masters,
/// and acquire a separate guard inside each worker owned by the operation.
/// Synchronous callbacks and dependency calls share the current scope.
/// Do not retain this guard across unrelated caller computations.
///
/// Registration failures return `None`; expired or revoked licenses can yield
/// an inert guard. In either case Symbolica retains its normal personal-license
/// or restricted-mode behavior. A returned `Some` is not proof of authorization.
#[doc(hidden)]
#[inline]
pub fn unlock_for_ninjars_operation() -> Option<LibraryUnlockGuard> {
    UNLOCK.as_ref().map(LibraryUnlock::unlock)
}
