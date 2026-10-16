//! Batched scalar-master provider boundary, independent of an integration backend.

use crate::{BatchShape, Error, Laurent, Result};

/// Scalar and tensor coefficients needed by one-loop reduction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MasterIntegral {
    A0,
    B0,
    DB0,
    C0,
    D0,
    B1,
    B11,
    B111,
    /// The last coefficient returned by OneLOop `an(rank)`: A0 for rank 0/1,
    /// A00 for rank 2/3, and A0000 for rank 4.
    An {
        rank: usize,
    },
}

impl MasterIntegral {
    pub const fn invariant_count(self) -> usize {
        match self {
            Self::A0 | Self::An { .. } => 0,
            Self::B0 | Self::DB0 | Self::B1 | Self::B11 | Self::B111 => 1,
            Self::C0 => 3,
            Self::D0 => 6,
        }
    }

    pub const fn mass_count(self) -> usize {
        match self {
            Self::A0 | Self::An { .. } => 1,
            Self::B0 | Self::DB0 | Self::B1 | Self::B11 | Self::B111 => 2,
            Self::C0 => 3,
            Self::D0 => 4,
        }
    }

    pub const fn argument_count(self) -> usize {
        self.invariant_count() + self.mass_count() + 1
    }
}

/// One integral family evaluated at several independent points.
///
/// Invariants and squared masses are row-major; each scale supplies one row.
/// Triangle invariant order is `(p1², p2², p3²)`. Box order is
/// `(p1², p2², p3², p4², (p1+p2)², (p2+p3)²)`, as in OneLOop.
/// Squared masses follow denominator order. All entries carry the caller's
/// numerical domain, including its precision and derivative components.
#[derive(Clone, Copy, Debug)]
pub struct MasterRequest<'a, T> {
    pub integral: MasterIntegral,
    pub invariants: &'a [T],
    pub masses_squared: &'a [T],
    pub scales_squared: &'a [T],
}

impl<T> MasterRequest<'_, T> {
    pub fn rows(&self) -> usize {
        self.scales_squared.len()
    }

    pub fn validate(&self, output_rows: usize) -> Result<()> {
        for (name, columns, actual) in [
            (
                "master invariants",
                self.integral.invariant_count(),
                self.invariants.len(),
            ),
            (
                "master masses",
                self.integral.mass_count(),
                self.masses_squared.len(),
            ),
            ("master outputs", 1, output_rows),
        ] {
            BatchShape {
                rows: self.rows(),
                columns,
            }
            .validate(name, actual)?;
        }
        if matches!(self.integral, MasterIntegral::An { rank } if rank > 4) {
            return Err(Error::Unsupported(
                "tadpole tensor rank exceeds four".into(),
            ));
        }
        Ok(())
    }
}

/// Providers retain their numerical workspaces between evaluations.
pub trait MasterProvider<T> {
    /// Write one `[finite, simple pole, double pole]` result per input row.
    /// Invalid requests must leave the caller's output unchanged.
    fn evaluate(&mut self, request: MasterRequest<'_, T>, output: &mut [Laurent<T>]) -> Result<()>;
}
