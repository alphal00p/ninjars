//! Batched binary64 master-integral boundary of the fast engine.
//!
//! Requests are grouped by family. Arguments are row-major and follow OneLOop
//! conventions with the squared renormalization scale last; every value is a
//! `C64` (invariants have zero imaginary parts). Each output entry is a
//! Laurent triple `[finite, 1/eps, 1/eps²]` in OneLOop normalization.

use super::complex::C64;
use crate::Result;

/// Master families requested by the fast engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FastMaster {
    /// `[m², mu²]` → `[A0]`.
    A0,
    /// `[p², m0², m1², mu²]` → `[B0, B1, B11]` (OneLOop `olo_b11` content).
    BubbleRank2,
    /// `[p², m0², m1², mu²]` → `[B0, B1, B11, B111]`.
    BubbleRank3,
    /// `[p1², p2², p3², m0², m1², m2², mu²]` → `[C0]`.
    C0,
    /// `[p1², p2², p3², p4², s12, s23, m0², m1², m2², m3², mu²]` → `[D0]`.
    D0,
    /// `[p², m0², m1², mu²]` → `[B0, B1, B11, B111, B1111]` (rank n+2
    /// bubbles; B1111 with the corrected OneLOop mass-swap sign).
    BubbleRank4,
}

impl FastMaster {
    pub const fn arity(self) -> usize {
        match self {
            Self::A0 => 2,
            Self::BubbleRank2 | Self::BubbleRank3 | Self::BubbleRank4 => 4,
            Self::C0 => 7,
            Self::D0 => 11,
        }
    }

    /// Number of Laurent triples written per row.
    pub const fn outputs(self) -> usize {
        match self {
            Self::A0 | Self::C0 | Self::D0 => 1,
            Self::BubbleRank2 => 3,
            Self::BubbleRank3 => 4,
            Self::BubbleRank4 => 5,
        }
    }
}

/// A batched provider of one-loop master integrals in binary64.
///
/// `args.len() == rows * family.arity()` and
/// `out.len() == rows * family.outputs()`. A row whose evaluation fails must
/// set `ok[row] = false`; its outputs are unspecified. Successful rows set
/// `ok[row] = true`.
pub trait FastMasters {
    fn evaluate(
        &mut self,
        family: FastMaster,
        args: &[C64],
        out: &mut [[C64; 3]],
        ok: &mut [bool],
    ) -> Result<()>;
}
