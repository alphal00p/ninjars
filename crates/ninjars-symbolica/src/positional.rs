//! [`ninjars_positional::positional_evaluator`] with NinjaRS errors: exact
//! evaluators whose instructions do not depend on the process history.

use ninjars::{Complex, Error, Result};
use symbolica::{atom::Atom, domains::rational::Rational, evaluate::ExpressionEvaluator};

/// See [`ninjars_positional::positional_evaluator`].
pub(crate) fn positional_evaluator(
    expressions: &[Atom],
    parameters: &[Atom],
    aliases: &[(Atom, Atom)],
    common_pairs: bool,
) -> Result<ExpressionEvaluator<Complex<Rational>>> {
    ninjars_positional::positional_evaluator(expressions, parameters, aliases, common_pairs)
        .map_err(|error| match error {
            ninjars_positional::Error::Slots => Error::Unsupported(error.to_string()),
            ninjars_positional::Error::ForeignSlot => Error::Symbolic(error.to_string()),
            ninjars_positional::Error::Symbolic(message) => Error::Symbolic(message),
        })
}
