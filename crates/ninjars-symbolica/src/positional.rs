//! Exact evaluators whose instructions do not depend on the process history.
//!
//! Symbolica sorts the terms of a sum and breaks ties in its Horner schemes by
//! the terms' encoded bytes, which contain the symbols' IDs (a symbol's ID
//! is its registration index, stored little-endian in one byte below 256 and
//! two bytes below 65536). An evaluator built from the caller's symbols thus
//! depends on which symbols the process registered before (other
//! topologies, numerators, threads), and its results round differently.
//!
//! [`positional_evaluator`] renames parameters and aliases to slots whose
//! encodings are functions of their positions only.

use std::{collections::HashMap, sync::LazyLock};

use symbolica::{
    atom::{Atom, AtomCore, AtomView, Symbol},
    domains::rational::Rational,
    evaluate::ExpressionEvaluator,
    function, symbol,
};

use ninjars::{Complex, Error, Result};

/// Evaluators with more parameters and aliases use [`Slots::function`].
const VARIABLE_SLOTS: usize = 1 << 12;

/// Positional stand-ins, registered together by the first evaluator of the
/// process so that later registrations (e.g. the 10^5 parameters of a
/// twenty-propagator accumulation) cannot exhaust their two-byte IDs. A
/// process that registered more than about 57000 symbols before gets only
/// function slots.
struct Slots {
    /// [`VARIABLE_SLOTS`] variables in blocks of 256 fresh symbols with
    /// consecutive IDs from a multiple of 256, or none if two-byte IDs ran
    /// out. Slot `i` is encoded as (`i % 256`, its block's high ID byte):
    /// every comparison between slots depends only on their positions, and
    /// all slots have the same encoded length.
    variables: Vec<Atom>,
    /// The first of `ninjars_slot::s0, s1, ...` with a two-byte ID.
    head: Symbol,
}

static SLOTS: LazyLock<Slots> = LazyLock::new(|| {
    let mut names = 0;
    let mut fresh = || {
        names += 1;
        symbol!(format!("ninjars_slot::s{}", names - 1))
    };
    let head = std::iter::repeat_with(&mut fresh)
        .find(|head| head.get_id() >= 256)
        .unwrap();
    let mut variables = Vec::with_capacity(VARIABLE_SLOTS);
    'blocks: while variables.len() < VARIABLE_SLOTS {
        // Pad to a multiple of 256; retry if another thread registered
        // symbols in between.
        loop {
            let id = fresh().get_id();
            if id >= u32::from(u16::MAX) - 256 {
                variables.clear();
                break 'blocks;
            }
            if id % 256 == 255 {
                break;
            }
        }
        let block: Vec<_> = (0..256).map(|_| fresh()).collect();
        let base = block[0].get_id();
        if base % 256 == 0 && block.iter().zip(base..).all(|(s, id)| s.get_id() == id) {
            variables.extend(block.into_iter().map(Atom::var));
        }
    }
    Slots { variables, head }
});

impl Slots {
    /// Positional stand-in for evaluators too large for the variables: the
    /// head of the three base-256 digits of `index` (slower to optimize).
    /// Slots differ only in their digits, so they sort by position, and all
    /// have the same encoded length. The head's ID is stored in every slot,
    /// and the lengths of nested sums and products depend on its width,
    /// hence a two-byte head in every process.
    fn function(&self, index: usize) -> Atom {
        let digit = |shift: u32| Atom::num(((index >> shift) & 0xff) as i64);
        function!(self.head, digit(16), digit(8), digit(0))
    }

    fn get(&self, count: usize) -> Vec<Atom> {
        if count <= self.variables.len() {
            self.variables[..count].to_vec()
        } else {
            (0..count).map(|index| self.function(index)).collect()
        }
    }
}

/// The exact evaluator of `expressions` in `parameters`, with argument-free
/// `aliases` (name, body) inlined, built after renaming every parameter and
/// alias to a positional slot: its instructions, and every result, depend
/// only on the expressions and the parameter positions.
///
/// Without `common_pairs`, Symbolica skips common-pair elimination.
pub(crate) fn positional_evaluator(
    expressions: &[Atom],
    parameters: &[Atom],
    aliases: &[(Atom, Atom)],
    common_pairs: bool,
) -> Result<ExpressionEvaluator<Complex<Rational>>> {
    let count = parameters.len() + aliases.len();
    if count > 1 << 24 {
        return Err(Error::Unsupported(
            "more than 2^24 evaluator parameters and aliases".into(),
        ));
    }
    let slots = SLOTS.get(count);
    let renamed: HashMap<AtomView<'_>, &Atom> = parameters
        .iter()
        .chain(aliases.iter().map(|(name, _)| name))
        .map(Atom::as_view)
        .zip(&slots)
        .collect();
    let rename = |atom: &Atom| {
        atom.replace_map(|view, _, out| {
            if let Some(&slot) = renamed.get(&view) {
                **out = slot.clone();
            }
        })
    };
    let expressions: Vec<_> = expressions.iter().map(rename).collect();
    let (positions, names) = slots.split_at(parameters.len());
    let aliases: Vec<_> = names
        .iter()
        .zip(aliases)
        .map(|(name, (_, body))| (name.clone(), rename(body)))
        .collect();
    let symbolic = |e: symbolica::evaluate::EvaluationError| Error::Symbolic(e.to_string());
    let mut builder = Atom::evaluator_multiple(&expressions, positions).cores(1);
    if !aliases.is_empty() {
        builder = builder.add_aliases(aliases).map_err(symbolic)?;
    }
    if !common_pairs {
        builder = builder.cpe_iterations(Some(0));
    }
    builder.build().map_err(symbolic)
}

#[cfg(test)]
mod tests {
    use super::{SLOTS, VARIABLE_SLOTS};
    use symbolica::atom::AtomCore;

    /// Slot `i` has an ID congruent to `i` modulo 256, in two-byte blocks of
    /// consecutive IDs that increase with `i`: Symbolica's byte comparisons
    /// between slots then depend only on their positions. (No variables if
    /// another test registered tens of thousands of symbols first.)
    #[test]
    fn variable_slots_are_aligned_two_byte_blocks() {
        let ids: Vec<u32> = SLOTS
            .variables
            .iter()
            .map(|slot| slot.get_symbol().unwrap().get_id())
            .collect();
        assert!(ids.is_empty() || ids.len() == VARIABLE_SLOTS);
        for (i, &id) in ids.iter().enumerate() {
            assert_eq!(id as usize % 256, i % 256, "slot {i}");
            assert!((256..=u32::from(u16::MAX)).contains(&id), "slot {i}");
            if i > 0 {
                assert!(id > ids[i - 1], "slot {i}");
            }
        }
        assert!((256..=u32::from(u16::MAX)).contains(&SLOTS.head.get_id()));
    }
}
