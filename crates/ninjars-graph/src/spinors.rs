//! Explicit components for open spinor chains.
//!
//! Contraction leaves each open fermion line as `chain(x, y, M1, ..., Mk)`,
//! the index contraction `x_a (M1 ⋯ Mk)_ab y_b` of the two external spinor
//! carriers `x`, `y` (`u`, `ubar`, `v` or `vbar`). Given explicit spinor
//! components, the chain becomes an ordinary polynomial in vector components.
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
use symbolica::atom::{Atom, AtomCore, AtomView, FunctionBuilder};

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

/// Whether a chain ends on an external spinor carrier `u`, `ubar`, `v` or
/// `vbar`; color chains end on color slots and are left to color algebra. A
/// bilinear without Dirac matrices is contracted as `dot(x, y)` instead.
fn is_spinor_chain(part: AtomView<'_>) -> bool {
    let AtomView::Fun(function) = part else {
        return false;
    };
    let symbol = function.get_symbol();
    (symbol == SPENSO_TAG.chain && function.iter().next().is_some_and(is_spinor))
        || (symbol == SPENSO_TAG.dot && function.get_nargs() == 2 && function.iter().all(is_spinor))
}

/// Whether the expression contains an open spinor chain.
pub fn has_spinor_chains(expression: &Atom) -> bool {
    let mut found = false;
    expression.visitor(&mut |part| {
        found |= is_spinor_chain(part);
        !found
    });
    found
}

/// Replace every spinor chain by its explicit components.
///
/// `vectors` and `spinors` are keyed by their carrier without its
/// representation argument, e.g. `K(0)`, `P(1)`, `ϵ(2)` and `u(0)`.
/// Lorentz indices shared between chains are summed with the metric.
pub fn expand_spinor_chains(
    expression: &Atom,
    vectors: &BTreeMap<Atom, FourMomentum<Atom>>,
    spinors: &BTreeMap<Atom, [Atom; 4]>,
) -> Result<Atom, PreparationError> {
    if !has_spinor_chains(expression) {
        return Ok(expression.clone());
    }
    let mut failure = None;
    // Chains without open Lorentz indices are replaced in place.
    let local = expression.replace_map(|part, _, output| {
        if failure.is_some() {
            return;
        }
        if !is_spinor_chain(part) {
            return;
        }
        match chain_indices(part) {
            Ok(indices) if indices.is_empty() => {
                match evaluate_chain(part, &BTreeMap::new(), vectors, spinors) {
                    Ok(value) => **output = value,
                    Err(error) => failure = Some(error),
                }
            }
            Ok(_) => {}
            Err(error) => failure = Some(error),
        }
    });
    if let Some(error) = failure {
        return Err(error);
    }
    if !has_spinor_chains(&local) {
        return Ok(local);
    }
    // Lorentz indices joining several chains: sum each term over components.
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
    let mut chains = Vec::new();
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
        if is_spinor_chain(base.as_view()) {
            chains.extend(std::iter::repeat_n(base, power));
        } else {
            rest *= &factor;
        }
    }
    let mut indices = Vec::<Atom>::new();
    for chain in &chains {
        for index in chain_indices(chain.as_view())? {
            if !indices.contains(&index) {
                indices.push(index);
            }
        }
    }
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
        for chain in &chains {
            value *= evaluate_chain(chain.as_view(), &values, vectors, spinors)?;
        }
        sum += value;
    }
    Ok(rest * sum)
}

fn metric(mu: usize) -> Atom {
    Atom::num(if mu == 0 { 1 } else { -1 })
}

/// Lorentz index arguments `mink(4, i)` of the chain's gamma matrices.
fn chain_indices(chain: AtomView<'_>) -> Result<Vec<Atom>, PreparationError> {
    let AtomView::Fun(function) = chain else {
        unreachable!("chain_indices takes a chain function");
    };
    let mut indices = Vec::new();
    for factor in function.iter().skip(2) {
        if let AtomView::Fun(matrix) = factor
            && matrix.get_symbol() == AGS.gamma
            && let Some(argument) = matrix.iter().nth(2)
            && let Some(index) = lorentz_index(argument)
        {
            indices.push(index);
        }
    }
    Ok(indices)
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

fn unsupported(chain: AtomView<'_>, message: &str) -> PreparationError {
    PreparationError::SpinorChain(format!("{message} in {}", chain.to_canonical_string()))
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

fn evaluate_chain(
    chain: AtomView<'_>,
    indices: &BTreeMap<Atom, usize>,
    vectors: &BTreeMap<Atom, FourMomentum<Atom>>,
    spinors: &BTreeMap<Atom, [Atom; 4]>,
) -> Result<Atom, PreparationError> {
    let AtomView::Fun(function) = chain else {
        unreachable!("evaluate_chain takes a chain function");
    };
    let mut arguments = function.iter();
    let spinor = |argument: Option<AtomView<'_>>| -> Result<[Atom; 4], PreparationError> {
        let key = argument
            .and_then(carrier)
            .ok_or_else(|| unsupported(chain, "a chain endpoint is not a spinor carrier"))?;
        spinors.get(&key).cloned().ok_or_else(|| {
            PreparationError::SpinorChain(format!(
                "no components for external spinor {}",
                key.to_canonical_string()
            ))
        })
    };
    let left = spinor(arguments.next())?;
    let right = spinor(arguments.next())?;
    let mut row = left;
    for factor in arguments {
        let matrix = dirac_matrix(chain, factor, indices, vectors)?;
        row = std::array::from_fn(|column| {
            let mut entry = Atom::Zero;
            for (k, value) in row.iter().enumerate() {
                if !matrix[k][column].is_zero() && !value.is_zero() {
                    entry += value * &matrix[k][column];
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

fn dirac_matrix(
    chain: AtomView<'_>,
    factor: AtomView<'_>,
    indices: &BTreeMap<Atom, usize>,
    vectors: &BTreeMap<Atom, FourMomentum<Atom>>,
) -> Result<Matrix, PreparationError> {
    let AtomView::Fun(function) = factor else {
        return Err(unsupported(chain, "unsupported Dirac factor"));
    };
    let symbol = function.get_symbol();
    if symbol == AGS.gamma5 {
        return Ok(diagonal([-1, -1, 1, 1]));
    }
    if symbol == AGS.projm {
        return Ok(diagonal([1, 1, 0, 0]));
    }
    if symbol == AGS.projp {
        return Ok(diagonal([0, 0, 1, 1]));
    }
    if symbol == AGS.gamma0 {
        return Ok(gamma(0));
    }
    if symbol != AGS.gamma {
        return Err(unsupported(chain, "unsupported Dirac factor"));
    }
    let argument = function
        .iter()
        .nth(2)
        .ok_or_else(|| unsupported(chain, "gamma matrix without a Lorentz argument"))?;
    if let Some(index) = lorentz_index(argument) {
        return Ok(gamma(indices[&index]));
    }
    let key = carrier(argument).ok_or_else(|| unsupported(chain, "unsupported slashed vector"))?;
    let vector = vectors.get(&key).ok_or_else(|| {
        PreparationError::SpinorChain(format!(
            "no components for vector {} in a spinor chain",
            key.to_canonical_string()
        ))
    })?;
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
