//! Explicit components for open spinor lines.
//!
//! Contraction leaves each open fermion line as the index contraction
//! `x_a (M1 ⋯ Mk)_ab y_b` of its two external spinor carriers `x`, `y` (`u`,
//! `ubar`, `v` or `vbar`): `chain(x, y, M1, ..., Mk)` for several Dirac
//! matrices, a bare `gamma(x, y, X)` (or `gamma5`, `projm`, `projp`,
//! `gamma0`) for one, and `dot(x, y)` for none. Lorentz indices left open on a
//! line are contracted with another line or with a vector `V(n, mink(4, i))`.
//! Given explicit spinor components, every line becomes an ordinary polynomial
//! in vector components.
//!
//! Matrices use the chiral (Weyl) basis of HELAS/MadGraph and of
//! `spenso-hep-lib`'s Weyl data: `γ⁰ = [[0, 1], [1, 0]]`,
//! `γᵏ = [[0, σᵏ], [−σᵏ, 0]]`, `γ⁵ = diag(−1, −1, 1, 1)`, and
//! `P∓ = (1 ∓ γ⁵)/2`. A slashed vector is `γ^μ g_μν V^ν` with the metric
//! `(+, −, −, −)` and contravariant components `[E, x, y, z]`.

use std::collections::BTreeMap;

use feynkit_kinematics::FourMomentum;
use idenso::dirac::AGS;
use spenso::network::tags::SPENSO_TAG;
use symbolica::atom::{Atom, AtomCore, AtomView, FunctionBuilder, Symbol};

use crate::{PreparationError, symbols};

type Matrix = [[Atom; 4]; 4];

/// Whether an argument is an external spinor carrier.
///
/// FeynKit registers each carrier symbol on first use. Names are compared
/// first so that inspecting a numerator registers no symbol: evaluators built
/// later order their operations by symbol identity.
fn is_spinor(argument: AtomView<'_>) -> bool {
    let AtomView::Fun(carrier) = argument else {
        return false;
    };
    let symbol = carrier.get_symbol();
    matches!(symbol.get_stripped_name(), "u" | "ubar" | "v" | "vbar")
        && [symbols::u(), symbols::ubar(), symbols::v(), symbols::vbar()].contains(&symbol)
}

fn is_dirac(symbol: Symbol) -> bool {
    [AGS.gamma, AGS.gamma5, AGS.projm, AGS.projp, AGS.gamma0].contains(&symbol)
}

/// Whether a factor is an open fermion line between two spinor carriers.
/// Color chains end on color slots and are left to color algebra.
fn is_spinor_line(part: AtomView<'_>) -> bool {
    let AtomView::Fun(function) = part else {
        return false;
    };
    let symbol = function.get_symbol();
    let mut arguments = function.iter();
    if symbol == SPENSO_TAG.chain {
        return arguments.next().is_some_and(is_spinor);
    }
    (symbol == SPENSO_TAG.dot || is_dirac(symbol))
        && arguments.next().is_some_and(is_spinor)
        && arguments.next().is_some_and(is_spinor)
}

/// Whether the expression contains an open spinor line.
pub fn has_spinor_chains(expression: &Atom) -> bool {
    let mut found = false;
    expression.visitor(&mut |part| {
        found |= is_spinor_line(part);
        !found
    });
    found
}

/// One Dirac matrix of a line: its head, its Lorentz or vector argument, and
/// whether its `(in, out)` markers are reversed.
struct Factor<'a> {
    head: Symbol,
    argument: Option<AtomView<'a>>,
    transposed: bool,
}

/// An open fermion line `x_a (M1 ⋯ Mk)_ab y_b`.
struct Line<'a> {
    source: AtomView<'a>,
    left: AtomView<'a>,
    right: AtomView<'a>,
    factors: Vec<Factor<'a>>,
}

fn unsupported(part: AtomView<'_>, message: &str) -> PreparationError {
    PreparationError::SpinorChain(format!("{message} in {}", part.to_canonical_string()))
}

impl<'a> Line<'a> {
    /// Read a factor accepted by [`is_spinor_line`].
    fn parse(part: AtomView<'a>) -> Result<Self, PreparationError> {
        let AtomView::Fun(function) = part else {
            unreachable!("Line::parse takes a spinor line");
        };
        let symbol = function.get_symbol();
        let arguments = function.iter().collect::<Vec<_>>();
        let (left, right) = match arguments.as_slice() {
            [left, right, ..] if is_spinor(*left) && is_spinor(*right) => (*left, *right),
            _ => return Err(unsupported(part, "a line endpoint is not a spinor carrier")),
        };
        let factors = if symbol == SPENSO_TAG.chain {
            arguments[2..]
                .iter()
                .map(|factor| Self::chain_factor(part, *factor))
                .collect::<Result<_, _>>()?
        } else if symbol == SPENSO_TAG.dot {
            Vec::new()
        } else {
            if arguments.len() > 3 {
                return Err(unsupported(part, "unsupported Dirac matrix"));
            }
            vec![Factor {
                head: symbol,
                argument: arguments.get(2).copied(),
                transposed: false,
            }]
        };
        Ok(Self {
            source: part,
            left,
            right,
            factors,
        })
    }

    /// A chain factor `M(in, out, ...)`, or its transpose `M(out, in, ...)`.
    fn chain_factor(
        part: AtomView<'a>,
        factor: AtomView<'a>,
    ) -> Result<Factor<'a>, PreparationError> {
        let AtomView::Fun(function) = factor else {
            return Err(unsupported(part, "unsupported Dirac factor"));
        };
        let marker = |argument: Option<AtomView<'_>>, symbol: Symbol| matches!(argument, Some(AtomView::Var(variable)) if variable.get_symbol() == symbol);
        let arguments = function.iter().collect::<Vec<_>>();
        let (first, second) = (arguments.first().copied(), arguments.get(1).copied());
        let transposed =
            if marker(first, SPENSO_TAG.chain_in) && marker(second, SPENSO_TAG.chain_out) {
                false
            } else if marker(first, SPENSO_TAG.chain_out) && marker(second, SPENSO_TAG.chain_in) {
                true
            } else {
                return Err(unsupported(
                    part,
                    "a chain factor without (in, out) markers",
                ));
            };
        if !is_dirac(function.get_symbol()) || arguments.len() > 3 {
            return Err(unsupported(part, "unsupported Dirac factor"));
        }
        Ok(Factor {
            head: function.get_symbol(),
            argument: arguments.get(2).copied(),
            transposed,
        })
    }

    /// Lorentz indices `mink(4, i)` left open on the line's gamma matrices.
    fn indices(&self) -> Vec<Atom> {
        self.factors
            .iter()
            .filter(|factor| factor.head == AGS.gamma)
            .filter_map(|factor| factor.argument.and_then(lorentz_index))
            .collect()
    }

    fn evaluate(
        &self,
        indices: &BTreeMap<Atom, usize>,
        vectors: &BTreeMap<Atom, FourMomentum<Atom>>,
        spinors: &BTreeMap<Atom, [Atom; 4]>,
    ) -> Result<Atom, PreparationError> {
        let spinor = |argument: AtomView<'_>| -> Result<[Atom; 4], PreparationError> {
            let key = carrier(argument).ok_or_else(|| {
                unsupported(self.source, "a line endpoint is not a spinor carrier")
            })?;
            spinors.get(&key).cloned().ok_or_else(|| {
                PreparationError::SpinorChain(format!(
                    "no components for external spinor {}",
                    key.to_canonical_string()
                ))
            })
        };
        let mut row = spinor(self.left)?;
        let right = spinor(self.right)?;
        for factor in &self.factors {
            let matrix = self.matrix(factor, indices, vectors)?;
            row = std::array::from_fn(|column| {
                let mut entry = Atom::Zero;
                for (k, value) in row.iter().enumerate() {
                    let element = if factor.transposed {
                        &matrix[column][k]
                    } else {
                        &matrix[k][column]
                    };
                    if !element.is_zero() && !value.is_zero() {
                        entry += value * element;
                    }
                }
                entry
            });
        }
        Ok(row
            .iter()
            .zip(&right)
            .fold(Atom::Zero, |sum, (a, b)| sum + a * b))
    }

    fn matrix(
        &self,
        factor: &Factor<'_>,
        indices: &BTreeMap<Atom, usize>,
        vectors: &BTreeMap<Atom, FourMomentum<Atom>>,
    ) -> Result<Matrix, PreparationError> {
        let symbol = factor.head;
        let no_argument = |matrix: Matrix| match factor.argument {
            None => Ok(matrix),
            Some(_) => Err(unsupported(
                self.source,
                "unsupported Dirac matrix argument",
            )),
        };
        if symbol == AGS.gamma5 {
            return no_argument(diagonal([-1, -1, 1, 1]));
        }
        if symbol == AGS.projm {
            return no_argument(diagonal([1, 1, 0, 0]));
        }
        if symbol == AGS.projp {
            return no_argument(diagonal([0, 0, 1, 1]));
        }
        if symbol == AGS.gamma0 {
            return no_argument(gamma(0));
        }
        let argument = factor
            .argument
            .ok_or_else(|| unsupported(self.source, "gamma matrix without a Lorentz argument"))?;
        if let Some(index) = lorentz_index(argument) {
            return Ok(gamma(indices[&index]));
        }
        let vector = vector(argument, vectors)?;
        let mut slashed: Matrix = std::array::from_fn(|_| std::array::from_fn(|_| Atom::Zero));
        for (mu, component) in vector.components().into_iter().enumerate() {
            let gamma = gamma(mu);
            for (row, entries) in slashed.iter_mut().enumerate() {
                for (column, entry) in entries.iter_mut().enumerate() {
                    if !gamma[row][column].is_zero() {
                        *entry += metric(mu) * &gamma[row][column] * component;
                    }
                }
            }
        }
        Ok(slashed)
    }
}

/// The components of a rank-one carrier `V(n, ...)`.
fn vector<'v>(
    argument: AtomView<'_>,
    vectors: &'v BTreeMap<Atom, FourMomentum<Atom>>,
) -> Result<&'v FourMomentum<Atom>, PreparationError> {
    let key =
        carrier(argument).ok_or_else(|| unsupported(argument, "unsupported slashed vector"))?;
    vectors.get(&key).ok_or_else(|| {
        PreparationError::SpinorChain(format!(
            "no components for vector {} in a spinor line",
            key.to_canonical_string()
        ))
    })
}

/// The open Lorentz index of a vector factor `V(n, mink(4, i))`.
fn vector_index(part: AtomView<'_>) -> Option<Atom> {
    let AtomView::Fun(function) = part else {
        return None;
    };
    let mut arguments = function.iter();
    let _label = arguments.next()?;
    let index = lorentz_index(arguments.next()?)?;
    arguments.next().is_none().then_some(index)
}

/// Replace every spinor line by its explicit components.
///
/// `vectors` and `spinors` are keyed by their carrier without its
/// representation argument, e.g. `K(0)`, `P(1)`, `ϵ(2)` and `u(0)`.
/// Lorentz indices shared between lines, or between a line and a vector, are
/// summed with the metric.
pub fn expand_spinor_chains(
    expression: &Atom,
    vectors: &BTreeMap<Atom, FourMomentum<Atom>>,
    spinors: &BTreeMap<Atom, [Atom; 4]>,
) -> Result<Atom, PreparationError> {
    if !has_spinor_chains(expression) {
        return Ok(expression.clone());
    }
    let mut failure = None;
    // Lines without open Lorentz indices are replaced in place.
    let local = expression.replace_map(|part, _, output| {
        if failure.is_some() || !is_spinor_line(part) {
            return;
        }
        let value = Line::parse(part).and_then(|line| {
            if line.indices().is_empty() {
                line.evaluate(&BTreeMap::new(), vectors, spinors).map(Some)
            } else {
                Ok(None)
            }
        });
        match value {
            Ok(Some(value)) => **output = value,
            Ok(None) => {}
            Err(error) => failure = Some(error),
        }
    });
    if let Some(error) = failure {
        return Err(error);
    }
    if !has_spinor_chains(&local) {
        return Ok(local);
    }
    // Contracted Lorentz indices: sum each term over components.
    let expanded = local.expand();
    let terms = match expanded.as_view() {
        AtomView::Add(sum) => sum.iter().map(|part| part.to_owned()).collect(),
        _ => vec![expanded.clone()],
    };
    let mut total = Atom::Zero;
    for term in terms {
        total += expand_term(&term, vectors, spinors)?;
    }
    Ok(total)
}

fn expand_term(
    term: &Atom,
    vectors: &BTreeMap<Atom, FourMomentum<Atom>>,
    spinors: &BTreeMap<Atom, [Atom; 4]>,
) -> Result<Atom, PreparationError> {
    let factors = match term.as_view() {
        AtomView::Mul(product) => product.iter().map(|part| part.to_owned()).collect(),
        _ => vec![term.clone()],
    };
    let mut lines = Vec::new();
    let mut open_vectors = Vec::new();
    let mut rest = Atom::num(1);
    for factor in factors {
        let (base, power) = match factor.as_view() {
            AtomView::Pow(pow) => {
                let (base, exponent) = pow.get_base_exp();
                match i64::try_from(exponent) {
                    Ok(power) if power > 0 => (base.to_owned(), power as usize),
                    _ => (factor.clone(), 1),
                }
            }
            _ => (factor.clone(), 1),
        };
        if is_spinor_line(base.as_view()) {
            lines.extend(std::iter::repeat_n(base, power));
        } else if let Some(index) = vector_index(base.as_view()) {
            open_vectors.extend(std::iter::repeat_n((base, index), power));
        } else {
            rest *= &factor;
        }
    }
    let lines = lines
        .iter()
        .map(|line| Line::parse(line.as_view()))
        .collect::<Result<Vec<_>, _>>()?;
    // Every index occurs on exactly two objects: lines or vectors.
    let mut occurrences = BTreeMap::<Atom, usize>::new();
    for index in lines
        .iter()
        .flat_map(Line::indices)
        .chain(open_vectors.iter().map(|(_, index)| index.clone()))
    {
        *occurrences.entry(index).or_default() += 1;
    }
    if let Some((index, _)) = occurrences.iter().find(|(_, count)| **count != 2) {
        return Err(PreparationError::SpinorChain(format!(
            "Lorentz index {} is not contracted between two factors of {}",
            index.to_canonical_string(),
            term.to_canonical_string()
        )));
    }
    let indices = occurrences.into_keys().collect::<Vec<_>>();
    let open_vectors = open_vectors
        .iter()
        .map(|(factor, index)| Ok((vector(factor.as_view(), vectors)?, index)))
        .collect::<Result<Vec<_>, PreparationError>>()?;
    let mut sum = Atom::Zero;
    for assignment in 0..4usize.pow(indices.len() as u32) {
        let values = indices
            .iter()
            .enumerate()
            .map(|(position, index)| (index.clone(), assignment / 4usize.pow(position as u32) % 4))
            .collect::<BTreeMap<_, _>>();
        let mut value = values
            .values()
            .fold(Atom::num(1), |product, mu| product * metric(*mu));
        for (vector, index) in &open_vectors {
            value *= vector.components()[values[*index]];
        }
        for line in &lines {
            value *= line.evaluate(&values, vectors, spinors)?;
        }
        sum += value;
    }
    Ok(rest * sum)
}

fn metric(mu: usize) -> Atom {
    Atom::num(if mu == 0 { 1 } else { -1 })
}

/// The index of a bare Lorentz slot `mink(d, i)`, as opposed to a vector `V(n, mink(d))`.
fn lorentz_index(argument: AtomView<'_>) -> Option<Atom> {
    let AtomView::Fun(slot) = argument else {
        return None;
    };
    (slot.get_nargs() == 2 && slot.get_symbol().get_name().ends_with("mink"))
        .then(|| slot.iter().nth(1).map(|part| part.to_owned()))
        .flatten()
}

/// The carrier key `head(first argument)` of a vector or spinor.
fn carrier(atom: AtomView<'_>) -> Option<Atom> {
    let AtomView::Fun(function) = atom else {
        return None;
    };
    let first = function.iter().next()?;
    Some(
        FunctionBuilder::new(function.get_symbol())
            .add_arg(first)
            .finish(),
    )
}

fn diagonal(values: [i64; 4]) -> Matrix {
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            if row == column {
                Atom::num(values[row])
            } else {
                Atom::Zero
            }
        })
    })
}

/// `γ^μ` in the chiral basis, with `i` as Symbolica's imaginary unit.
fn gamma(mu: usize) -> Matrix {
    let i = Atom::i();
    let one = Atom::num(1);
    let mut matrix: Matrix = std::array::from_fn(|_| std::array::from_fn(|_| Atom::Zero));
    let entries: [(usize, usize, Atom); 4] = match mu {
        0 => [
            (0, 2, one.clone()),
            (1, 3, one.clone()),
            (2, 0, one.clone()),
            (3, 1, one),
        ],
        1 => [
            (0, 3, one.clone()),
            (1, 2, one.clone()),
            (2, 1, -&one),
            (3, 0, -one),
        ],
        2 => [
            (0, 3, -&i),
            (1, 2, i.clone()),
            (2, 1, i.clone()),
            (3, 0, -i),
        ],
        _ => [
            (0, 2, one.clone()),
            (1, 3, -&one),
            (2, 0, -one.clone()),
            (3, 1, one),
        ],
    };
    for (row, column, value) in entries {
        matrix[row][column] = value;
    }
    matrix
}
