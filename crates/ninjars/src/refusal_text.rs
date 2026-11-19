//! The one wording of the refusal records' status texts
//! ([`crate::fast::Instability`], [`crate::conditioning::BoxGramRefusal`]):
//! `<test>: <quantity> <found>, <requirement> <bound>[ (<note>)]`, the found
//! number first and then what the test required, e.g.
//! `box normal: found 3.09e-11, required > 3.26e-8`. The text is derived from
//! the structured record; callers that need the numbers read the record.

use std::fmt;

pub(crate) struct RefusalText<'a> {
    /// The refusing test's name.
    pub(crate) test: &'a str,
    /// What was measured or predicted (`found`, `predicted error`, ...).
    pub(crate) quantity: &'a str,
    pub(crate) found: String,
    /// The comparison a served lane needs (`required >`, `allowed <=`).
    pub(crate) requirement: &'a str,
    pub(crate) bound: String,
    pub(crate) note: Option<String>,
}

impl fmt::Display for RefusalText<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {} {}, {} {}",
            self.test, self.quantity, self.found, self.requirement, self.bound
        )?;
        if let Some(note) = &self.note {
            write!(f, " ({note})")?;
        }
        Ok(())
    }
}

/// A record's number: `0`, `NaN`, else three significant digits.
pub(crate) fn number(x: f64) -> String {
    if x == 0. {
        "0".into()
    } else if x.is_nan() {
        "NaN".into()
    } else {
        format!("{x:.2e}")
    }
}
