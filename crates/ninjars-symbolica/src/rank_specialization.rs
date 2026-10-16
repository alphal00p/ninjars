//! Guarded specialization of existing Symbolica correction expressions.
//!
//! Masks come from the exact residue projection, not numerical rank guesses.
//! Symbolica performs replacement and simplification; input ABIs stay intact.

use ninjars::{Error, Result};
use symbolica::{
    atom::{Atom, AtomCore},
    id::{Pattern, Replacement},
};

use crate::{LaurentDivision, ResidueProjection, kernel::ExactKernel};

pub(crate) fn require_supported(n: usize, rank: usize) -> Result<()> {
    if (n, rank) != (5, 4) {
        return Err(Error::Unsupported(
            "guarded correction specialization currently requires n=5 and rank=4".into(),
        ));
    }
    Ok(())
}

pub(crate) fn zero_slots(n: usize, rank: usize, source_size: usize) -> Result<Vec<usize>> {
    require_supported(n, rank)?;
    let projection = ResidueProjection::new(&LaurentDivision::new(n, rank, source_size)?)?;
    Ok(projection
        .exact
        .expressions
        .iter()
        .enumerate()
        .filter_map(|(index, expression)| (expression == &Atom::Zero).then_some(index))
        .collect())
}

pub(crate) fn specialize(
    plan: &ExactKernel,
    heads: &[usize],
    residue: std::ops::Range<usize>,
    zero_slots: &[usize],
) -> ExactKernel {
    let replacements: Vec<_> = zero_slots
        .iter()
        .map(|&slot| {
            assert!(slot < residue.len());
            Replacement::new(
                Pattern::Literal(plan.parameters[residue.start + slot].clone()),
                Pattern::Literal(Atom::Zero),
            )
        })
        .collect();
    ExactKernel {
        parameters: plan.parameters.clone(),
        expressions: heads
            .iter()
            .map(|&head| plan.expressions[head].replace_multiple(&replacements))
            .collect(),
    }
}

pub(crate) struct GuardedCorrection<K> {
    pub kernel: K,
    pub zero_slots: Vec<usize>,
}

#[cfg(test)]
thread_local! {
    // Pairs [specialized, generic] for triangle→bubble, triangle→tadpole,
    // bubble→tadpole. Count actual dispatched rows, not eligible requests.
    pub(crate) static ROWS: std::cell::Cell<[usize; 6]> = const {
        std::cell::Cell::new([0; 6])
    };
}

#[cfg(test)]
pub(crate) fn record(kind: usize, specialized: bool, rows: usize) {
    ROWS.with(|counts| {
        let mut values = counts.get();
        values[2 * kind + usize::from(!specialized)] += rows;
        counts.set(values);
    });
}

#[cfg(test)]
#[path = "rank_specialization_tests.rs"]
mod tests;
