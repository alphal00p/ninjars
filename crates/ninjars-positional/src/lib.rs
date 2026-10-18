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
//! encodings are functions of their positions only. Every crate that builds
//! evaluators (the generic engine, graph coefficients and masses) shares this
//! crate's process-wide pool of slots.

#![forbid(unsafe_code)]

use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    sync::LazyLock,
};

use symbolica::{
    atom::{Atom, AtomCore, AtomView, Symbol},
    domains::{float::Complex, rational::Rational},
    evaluate::ExpressionEvaluator,
    function, symbol,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("more than 2^24 evaluator parameters and aliases")]
    Slots,
    #[error("expressions contain positional slots that are not parameters")]
    ForeignSlot,
    #[error("{0}")]
    Symbolic(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Evaluators with more parameters and aliases use [`Slots::function`].
const VARIABLE_SLOTS: usize = 1 << 12;

/// Positional stand-ins, registered together on first use (an evaluator or
/// [`slots`]) so that later registrations (e.g. the 10^5 parameters of a
/// twenty-propagator accumulation) cannot exhaust their two-byte IDs. A
/// process that registered more than about 61000 symbols before gets only
/// function slots.
struct Slots {
    /// [`VARIABLE_SLOTS`] variables in blocks of 256 fresh symbols with
    /// consecutive IDs from a multiple of 256, or none if two-byte IDs ran
    /// out. Slot `i` is encoded as (`i % 256`, its block's high ID byte):
    /// every comparison between slots depends only on their positions, and
    /// all slots have the same encoded length.
    variables: Vec<Atom>,
    /// The symbols of [`Slots::variables`].
    symbols: HashSet<Symbol>,
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
    let mut last = head.get_id();
    let mut variables = Vec::with_capacity(VARIABLE_SLOTS);
    'blocks: while variables.len() < VARIABLE_SLOTS {
        // Pad to a multiple of 256 (a block ends on one); retry if another
        // thread registered symbols in between.
        loop {
            if last >= u32::from(u16::MAX) - 256 {
                variables.clear();
                break 'blocks;
            }
            if last % 256 == 255 {
                break;
            }
            last = fresh().get_id();
        }
        let block: Vec<_> = (0..256).map(|_| fresh()).collect();
        let base = block[0].get_id();
        last = block[255].get_id();
        if base % 256 == 0 && block.iter().zip(base..).all(|(s, id)| s.get_id() == id) {
            variables.extend(block.into_iter().map(Atom::var));
        }
    }
    let symbols = variables
        .iter()
        .filter_map(|slot| slot.get_symbol())
        .collect();
    Slots {
        variables,
        symbols,
        head,
    }
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

    /// Whether `view` is a variable or function slot.
    fn contains(&self, view: AtomView<'_>) -> bool {
        match view {
            AtomView::Var(var) => self.symbols.contains(&var.get_symbol()),
            AtomView::Fun(fun) => fun.get_symbol() == self.head,
            _ => false,
        }
    }
}

/// The positional stand-ins of the first `count` parameters and aliases of
/// [`positional_evaluator`]: variables up to 4096 (unless two-byte IDs ran
/// out), else functions.
///
/// Renaming re-normalizes every expression, which takes about a tenth of
/// the evaluator's construction. Expressions built in these slots from the
/// start, and passed with exactly them as parameters and alias names, are
/// used as they are.
pub fn slots(count: usize) -> Result<Vec<Atom>> {
    if count > 1 << 24 {
        return Err(Error::Slots);
    }
    Ok(SLOTS.get(count))
}

/// The exact evaluator of `expressions` in `parameters`, with argument-free
/// `aliases` (name, body) inlined, built after renaming every parameter and
/// alias to a positional slot: its instructions, and every result, depend
/// only on the expressions and the parameter positions. Expressions written
/// in [`slots`], with exactly these as parameters and alias names, are not
/// renamed.
///
/// Without `common_pairs`, Symbolica skips common-pair elimination.
pub fn positional_evaluator(
    expressions: &[Atom],
    parameters: &[Atom],
    aliases: &[(Atom, Atom)],
    common_pairs: bool,
) -> Result<ExpressionEvaluator<Complex<Rational>>> {
    let slots = slots(parameters.len() + aliases.len())?;
    let names = || {
        parameters
            .iter()
            .chain(aliases.iter().map(|(name, _)| name))
    };
    let (expressions, aliases) = if names().eq(&slots) {
        (Cow::Borrowed(expressions), aliases.to_vec())
    } else {
        let renamed: HashMap<AtomView<'_>, &Atom> =
            names().map(Atom::as_view).zip(&slots).collect();
        // A slot that is not a parameter would be confused with the
        // parameter renamed to it.
        let mut foreign = false;
        let mut rename = |atom: &Atom| {
            atom.replace_map(|view, _, out| {
                if let Some(&slot) = renamed.get(&view) {
                    **out = slot.clone();
                } else if SLOTS.contains(view) {
                    foreign = true;
                }
            })
        };
        let expressions: Vec<_> = expressions.iter().map(&mut rename).collect();
        let aliases: Vec<_> = slots[parameters.len()..]
            .iter()
            .zip(aliases)
            .map(|(name, (_, body))| (name.clone(), rename(body)))
            .collect();
        if foreign {
            return Err(Error::ForeignSlot);
        }
        (Cow::Owned(expressions), aliases)
    };
    let symbolic = |e: symbolica::evaluate::EvaluationError| Error::Symbolic(e.to_string());
    let mut builder = Atom::evaluator_multiple(&expressions, &slots[..parameters.len()]).cores(1);
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
    use super::{Error, SLOTS, VARIABLE_SLOTS, positional_evaluator, slots};
    use symbolica::{
        atom::{Atom, AtomCore},
        domains::{float::Complex, rational::Rational},
        symbol,
    };

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

    fn value(expression: Atom, parameters: &[Atom]) -> Complex<f64> {
        let mut evaluator = positional_evaluator(&[expression], parameters, &[], true)
            .unwrap()
            .map_coeff_with_prec(
                &|v: &Complex<Rational>| Complex::new(v.re.to_f64(), v.im.to_f64()),
                53,
            );
        let mut out = [Complex::new(0., 0.)];
        evaluator.evaluate(&[Complex::new(0.3, -1.7), Complex::new(2.9, 0.1)], &mut out);
        out[0]
    }

    /// Expressions built in the slots are not renamed and give the evaluator
    /// of the same expressions in any other parameters.
    #[test]
    fn expressions_in_slots_match_renamed_expressions() {
        let named = ["x", "y"].map(|n| Atom::var(symbol!(format!("ninjars_positional_test::{n}"))));
        let positions = slots(2).unwrap();
        let expression = |x: &Atom, y: &Atom| (x + y) * (x - y) * (x * y + x);
        let renamed = value(expression(&named[0], &named[1]), &named);
        let direct = value(expression(&positions[0], &positions[1]), &positions);
        assert_eq!(
            (renamed.re.to_bits(), renamed.im.to_bits()),
            (direct.re.to_bits(), direct.im.to_bits())
        );
    }

    /// A slot that is not a parameter would alias the parameter renamed to it.
    #[test]
    fn foreign_slots_are_rejected() {
        let positions = slots(2).unwrap();
        assert!(matches!(
            positional_evaluator(&[&positions[0] + &positions[1]], &positions[1..], &[], true),
            Err(Error::ForeignSlot)
        ));
    }
}
