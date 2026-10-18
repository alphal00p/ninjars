//! Explicit components for four-dimensional Levi-Civita tensors.
//!
//! Idenso evaluates axial traces with `spenso::epsilon`, whose normalization
//! `ε^{0123} = −i` (for `γ⁵ = i γ⁰ γ¹ γ² γ³`, the `diag(−1, −1, 1, 1)` of the
//! chiral basis in [`crate::spinors`]) is documented and evaluated by
//! [`idenso::epsilon::contract_epsilon4`]. A fully contracted tensor is
//! `ε(a, b, c, d) = i det[a; b; c; d]` in contravariant components.
//!
//! The indices are four-dimensional, so the loop momentum enters through its
//! four components only (`ε(q + q̃, …) = ε(q, …)`, as in the 't Hooft–Veltman
//! scheme; see Shao, Zhang and Chao, arXiv:1106.5030, Sec. 2). Rational terms
//! of axial numerators depend on the `γ⁵` scheme and are not computed here.

use std::collections::BTreeMap;

use feynkit_kinematics::FourMomentum;
use idenso::epsilon::contract_epsilon4;
use symbolica::atom::{Atom, AtomCore, AtomView};

use crate::{
    PreparationError,
    spinors::{carrier, lorentz_index, metric},
};

/// Whether a factor is a four-dimensional Lorentz Levi-Civita tensor: four
/// arguments, each a Minkowski slot `mink(4, i)` or a vector `V(n, mink(4))`.
///
/// The symbol is compared by name, so inspecting a numerator registers no
/// symbol: evaluators built later order their operations by symbol identity.
/// Color tensors `epsilon(cof(3, i), …)` are left to color algebra.
pub(crate) fn is_levi_civita(part: AtomView<'_>) -> bool {
    let AtomView::Fun(function) = part else {
        return false;
    };
    function.get_symbol().get_name() == "spenso::epsilon"
        && function.get_nargs() == 4
        && function
            .iter()
            .all(|argument| four_dimensional_index(argument) || four_vector(argument))
}

/// Whether the expression contains a four-dimensional Levi-Civita tensor.
pub fn has_levi_civita(expression: &Atom) -> bool {
    let mut found = false;
    expression.visitor(&mut |part| {
        found |= is_levi_civita(part);
        !found
    });
    found
}

/// The open Lorentz indices `mink(4, i)` of a Levi-Civita tensor.
pub(crate) fn levi_civita_indices(tensor: AtomView<'_>) -> Vec<Atom> {
    let AtomView::Fun(function) = tensor else {
        unreachable!("levi_civita_indices takes a Levi-Civita tensor");
    };
    function.iter().filter_map(lorentz_index).collect()
}

/// The tensor with its open indices set to the contravariant components in
/// `indices` and its vectors contracted with their components.
pub(crate) fn evaluate_levi_civita(
    tensor: AtomView<'_>,
    indices: &BTreeMap<Atom, usize>,
    vectors: &BTreeMap<Atom, FourMomentum<Atom>>,
) -> Result<Atom, PreparationError> {
    let AtomView::Fun(function) = tensor else {
        unreachable!("evaluate_levi_civita takes a Levi-Civita tensor");
    };
    // Covariant vector components a_ν, or a unit row keeping an open index.
    let mut rows: [[Atom; 4]; 4] = std::array::from_fn(|_| std::array::from_fn(|_| Atom::Zero));
    for (row, argument) in rows.iter_mut().zip(function.iter()) {
        if let Some(index) = lorentz_index(argument) {
            let mu = indices.get(&index).ok_or_else(|| {
                PreparationError::LeviCivita(format!(
                    "index {} is not summed in {}",
                    index.to_canonical_string(),
                    tensor.to_canonical_string()
                ))
            })?;
            row[*mu] = Atom::num(1);
            continue;
        }
        let vector = carrier(argument)
            .and_then(|key| vectors.get(&key))
            .ok_or_else(|| {
                PreparationError::LeviCivita(format!(
                    "no components for {} in {}",
                    argument.to_canonical_string(),
                    tensor.to_canonical_string()
                ))
            })?;
        for (mu, component) in vector.components().into_iter().enumerate() {
            row[mu] = metric(mu) * component;
        }
    }
    Ok(contract_epsilon4(&rows))
}

/// A slot `mink(4, i)`.
fn four_dimensional_index(argument: AtomView<'_>) -> bool {
    lorentz_index(argument).is_some() && minkowski_dimension(argument) == Some(4)
}

/// A vector `V(n, mink(4))`.
fn four_vector(argument: AtomView<'_>) -> bool {
    let AtomView::Fun(vector) = argument else {
        return false;
    };
    vector.get_nargs() == 2
        && vector
            .iter()
            .nth(1)
            .is_some_and(|representation| minkowski_dimension(representation) == Some(4))
}

/// The dimension `d` of `mink(d)` or `mink(d, i)`.
fn minkowski_dimension(argument: AtomView<'_>) -> Option<i64> {
    let AtomView::Fun(slot) = argument else {
        return None;
    };
    if !slot.get_symbol().get_name().ends_with("mink") {
        return None;
    }
    slot.iter().next().and_then(|d| i64::try_from(d).ok())
}
