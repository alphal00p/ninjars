//! Guarded specialization of existing Symbolica correction expressions.
//!
//! Masks come from the exact residue projection, not numerical rank guesses.
//! Symbolica performs replacement and simplification; input ABIs stay intact.

use ninjars::{Error, ExpansionKind, Result};
use symbolica::{
    atom::{Atom, AtomCore},
    id::{Pattern, Replacement},
};

use crate::{
    LaurentDivision, ResidueProjection, TadpoleSubtraction, TriangleBubbleSubtraction,
    kernel::ExactKernel,
};

/// Semantic identities for the three guarded n=5/r=4 correction bodies.
///
/// These identities do not imply that arbitrary residue inputs satisfy the
/// structural-zero assumptions. Runtime dispatch must retain the full guard.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RankSpecializedCorrectionPlan {
    TriangleToBubbleN5R4,
    TriangleToTadpoleN5R4,
    BubbleToTadpoleN5R4,
}

pub(crate) struct PreparedCorrection {
    pub exact: ExactKernel,
    pub zero_slots: Vec<usize>,
    pub output: ExpansionKind,
}

impl RankSpecializedCorrectionPlan {
    #[cfg(feature = "rust-codegen")]
    pub fn semantic_id(self) -> &'static str {
        match self {
            Self::TriangleToBubbleN5R4 => "triangle_to_bubble_n5_r4_full_quotient",
            Self::TriangleToTadpoleN5R4 => "triangle_to_tadpole_n5_r4_constant",
            Self::BubbleToTadpoleN5R4 => "bubble_to_tadpole_n5_r4_constant",
        }
    }

    /// One shared exact constructor supplies interpreted and emitted bodies.
    pub(crate) fn prepare(self) -> Result<PreparedCorrection> {
        let triangle = !matches!(self, Self::BubbleToTadpoleN5R4);
        let zero_slots = zero_slots(5, 4, if triangle { 3 } else { 2 })?;
        let (exact, output) = if matches!(self, Self::TriangleToBubbleN5R4) {
            let division = LaurentDivision::new(5, 4, 2)?;
            let plan = TriangleBubbleSubtraction::for_quotient(&division)?;
            let heads: Vec<_> = (0..plan.exact.expressions.len()).collect();
            (
                specialize(
                    &plan.exact,
                    &heads,
                    plan.input_layout().residue.clone(),
                    &zero_slots,
                ),
                division.quotient_expansion().clone(),
            )
        } else {
            let division = LaurentDivision::new(5, 4, 1)?;
            let plan = TadpoleSubtraction::new(if triangle { 3 } else { 2 }, false)?;
            let ExpansionKind::T3 { powers } = plan.output_expansion() else {
                unreachable!()
            };
            let constant = powers
                .iter()
                .position(|&power| power == (0, 0))
                .ok_or_else(|| {
                    Error::Expansion("tadpole correction has no constant head".into())
                })?;
            (
                specialize(
                    &plan.exact,
                    &[constant],
                    plan.input_layout().residue.clone(),
                    &zero_slots,
                ),
                division.quotient_expansion().clone(),
            )
        };
        Ok(PreparedCorrection {
            exact,
            zero_slots,
            output,
        })
    }
}

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
