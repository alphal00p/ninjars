use crate::{Error, Result};

/// Identity of a numerical request, independent of scheduling and batch packing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SampleId {
    pub lane: usize,
    pub sample: usize,
}

/// Validated matrix dimensions. Public numeric buffers are row-major.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchShape {
    pub rows: usize,
    pub columns: usize,
}

impl BatchShape {
    pub fn validate(self, name: &'static str, len: usize) -> Result<()> {
        let expected = self
            .rows
            .checked_mul(self.columns)
            .ok_or_else(|| Error::Unsupported("batch dimensions overflow usize".into()))?;
        if len != expected {
            return Err(Error::Shape {
                name,
                expected,
                actual: len,
            });
        }
        Ok(())
    }
}
