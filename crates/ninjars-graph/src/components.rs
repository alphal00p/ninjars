//! Exact component binding through the existing kinematics operations.
//!
//! # Dimensional schemes and R2
//!
//! FeynKit's Feynman rules and Idenso's tensor algebra contract every Lorentz,
//! Dirac and color index with identities valid for abstract vectors in a
//! `d_s = 4` index space (`g^mu_mu = 4`, `Tr 1 = 4`, Clifford relations) and
//! return a polynomial in scalar products of the loop momentum `K`, external
//! momenta and polarization vectors. Binding those products is where the
//! dimension of the loop momentum enters:
//!
//! - [`R2Scheme::Off`] binds every product with the four-dimensional metric,
//!   `K.K -> q0^2 - q1^2 - q2^2 - q3^2`. The numerator has no `mu^2` terms; the
//!   d-dimensional denominators of the reduction still give R1, but R2 is absent.
//! - [`R2Scheme::Fdh`] treats the loop momentum as d-dimensional,
//!   `Kbar^2 = q^2 - mu^2`, while external momenta and polarizations stay
//!   four-dimensional (`Kbar.p = q.p`). With the `d_s = 4` internal algebra this
//!   is the four-dimensional-helicity (FDH) numerator `N(q, mu^2)`, whose `mu^2`
//!   terms supply all of R2 in FDH. Closed fermion, scalar or ghost loops with
//!   only external bosons have scheme-independent R2 (e.g. MadLoop's `R2_GGHt`).
//!   HV/CDR would additionally need the `(d_s - 4)`-dimensional internal algebra
//!   (`N0 + eps N1 + eps^2 N2`), which is not provided.
//!
//! FDH binding is exact only when no four-dimensional-only identity acted on
//! the loop momentum before binding. Idenso's default `hep` algebra uses only
//! dimension-generic Clifford and trace recursions for closed traces without
//! `gamma5`; its four-dimensional rules are the axial (`gamma5`) trace
//! recursion with Levi-Civita tensors, the optional three-gamma Chisholm
//! expansion (disabled by default) and the Chisholm contraction of a repeated
//! Lorentz index around a Dirac chain. The latter is also applied to closed
//! traces, where it was checked to agree with a symbolic-dimension contraction
//! (five gammas on each arc, the loop momentum on both); at one loop such terms
//! are linear in the loop momentum. External polarization sums (Minkowski
//! metrics in the projector) are four-dimensional regular metrics that the
//! scalar-product binding cannot distinguish from internal ones, and open
//! fermion lines with symbolic spinors would need the loop momentum expanded in
//! components, which loses `mu^2`; hand-written spin-trace projectors are the
//! FDH route for open lines. [`PreparedIntegrand::check_fdh`] therefore rejects
//! `gamma5`/chiral projectors, Levi-Civita tensors, external spinors and
//! Minkowski metrics in the projector.

use std::collections::BTreeMap;

use feynkit_kinematics::{FourMomentum, Kinematics};
use idenso::{dirac::AGS, epsilon::EPSILON_SYMBOL};
use spenso::{
    network::library::symbolic::ETS,
    structure::representation::{LibraryRep, Minkowski},
};
use symbolica::atom::{Atom, AtomCore, AtomView};

use crate::{PreparationError, PreparedIntegrand, symbols};

/// Dimension of the loop momentum when binding scalar products to components.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum R2Scheme {
    /// Four-dimensional loop momentum: no `mu^2` numerator terms and no R2.
    #[default]
    Off,
    /// Four-dimensional-helicity scheme: `K.K -> q.q - mu^2`; the `mu^2` terms
    /// give R2 in FDH (see the module documentation).
    Fdh,
}

impl R2Scheme {
    /// Command-line and metadata name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Fdh => "fdh",
        }
    }

    /// The `mu^2` variable to bind in this scheme, if any.
    pub fn mu_squared(self, mu_squared: &Atom) -> Option<&Atom> {
        match self {
            Self::Off => None,
            Self::Fdh => Some(mu_squared),
        }
    }
}

impl PreparedIntegrand {
    /// Bind scalar products to exact component expressions using the shared metric.
    ///
    /// Keys include the literal `family.external_momenta()` atoms. Additional
    /// rank-one carriers, such as polarization vectors, may also be bound.
    /// The four loop components may themselves be symbols; external components
    /// may retain arbitrary-precision or Dual evaluator parameters.
    /// The loop momentum is four-dimensional ([`R2Scheme::Off`]).
    pub fn component_kinematics(
        &self,
        loop_components: &FourMomentum<Atom>,
        external_momenta: &BTreeMap<Atom, FourMomentum<Atom>>,
    ) -> Result<Kinematics, PreparationError> {
        self.dimensional_kinematics(loop_components, None, external_momenta)
    }

    /// [`Self::component_kinematics`] with an optional d-dimensional loop
    /// momentum. With `Some(mu_squared)` (FDH, [`R2Scheme::Fdh`]) the loop
    /// self-product is `K.K -> q.q - mu_squared`; products with external
    /// momenta and polarizations keep their four-dimensional values. The FDH
    /// binding first runs [`Self::check_fdh`].
    pub fn dimensional_kinematics(
        &self,
        loop_components: &FourMomentum<Atom>,
        mu_squared: Option<&Atom>,
        external_momenta: &BTreeMap<Atom, FourMomentum<Atom>>,
    ) -> Result<Kinematics, PreparationError> {
        if mu_squared.is_some() {
            self.check_fdh()?;
        }
        let loop_name = &self.family.loop_momenta()[0];
        let mut components = vec![(loop_name, loop_components)];
        for name in self.family.external_momenta() {
            if !external_momenta.contains_key(name) {
                return Err(PreparationError::MissingMomentum(name.clone()));
            }
        }
        components.extend(
            external_momenta
                .iter()
                .filter(|(name, _)| *name != loop_name),
        );
        let mut kinematics = self.family.kinematics().clone();
        for (i, (p_name, p)) in components.iter().enumerate() {
            for (q_name, q) in &components[i..] {
                let mut product = p.dot(q);
                // Only the loop momentum has extra-dimensional components.
                if let (0, Some(mu_squared)) = (i, mu_squared)
                    && *q_name == loop_name
                {
                    product = &product - mu_squared;
                }
                kinematics = kinematics.with_scalar_product(p_name, q_name, product)?;
            }
        }
        Ok(kinematics)
    }

    /// Express the contracted numerator in explicit loop and external components.
    pub fn component_numerator(
        &self,
        loop_components: &FourMomentum<Atom>,
        external_momenta: &BTreeMap<Atom, FourMomentum<Atom>>,
    ) -> Result<Atom, PreparationError> {
        self.dimensional_numerator(loop_components, None, external_momenta)
    }

    /// [`Self::component_numerator`] with an optional d-dimensional loop
    /// momentum, as in [`Self::dimensional_kinematics`].
    pub fn dimensional_numerator(
        &self,
        loop_components: &FourMomentum<Atom>,
        mu_squared: Option<&Atom>,
        external_momenta: &BTreeMap<Atom, FourMomentum<Atom>>,
    ) -> Result<Atom, PreparationError> {
        Ok(self
            .dimensional_kinematics(loop_components, mu_squared, external_momenta)?
            .apply(&self.numerator)
            .expand())
    }

    /// Reject numerators whose preparation may have used four-dimensional-only
    /// identities on the loop momentum, or whose external metrics are
    /// four-dimensional, so that FDH binding would not give the FDH numerator
    /// (see the module documentation). Inspects the graph numerator, its
    /// prefactor and projector before contraction.
    pub fn check_fdh(&self) -> Result<(), PreparationError> {
        let axial = [AGS.gamma5, AGS.projp, AGS.projm];
        let spinors = [symbols::u(), symbols::ubar(), symbols::v(), symbols::vbar()];
        let levi_civita = *EPSILON_SYMBOL;
        let minkowski = LibraryRep::from(Minkowski {}).symbol();
        let diagram = &self.diagram;
        let mut reason = None;
        for (index, expression) in [
            feynkit_graph::NumeratorInterface::export(diagram.numerator()).0,
            feynkit_graph::NumeratorInterface::export(diagram.numerator_prefactor()).0,
            feynkit_graph::NumeratorInterface::export(diagram.projector()).0,
        ]
        .into_iter()
        .enumerate()
        {
            expression.visitor(&mut |node| {
                if let AtomView::Fun(function) = node {
                    let symbol = function.get_symbol();
                    if axial.contains(&symbol) {
                        reason = Some("gamma5 or a chiral projector (FDH needs a gamma5 prescription; Idenso evaluates axial traces with four-dimensional Levi-Civita identities)");
                    } else if symbol == levi_civita {
                        reason = Some("a Levi-Civita tensor (four-dimensional; its contractions with the loop momentum are not d-dimensional)");
                    } else if spinors.contains(&symbol) {
                        reason = Some("an external spinor (open fermion lines would need the loop momentum in components, losing mu^2; use a spin-trace projector)");
                    } else if index == 2
                        && symbol == ETS.metric
                        && function.iter().any(|slot| {
                            matches!(slot, AtomView::Fun(rep) if rep.get_symbol() == minkowski)
                        })
                    {
                        reason = Some("a Minkowski metric in the external projector (four-dimensional polarization sums are not distinguished from internal metrics)");
                    }
                }
                reason.is_none()
            });
            if let Some(reason) = reason {
                return Err(PreparationError::UnsupportedFdh(reason));
            }
        }
        Ok(())
    }

    /// Recover each normalized `(q + V_i)` shift without repeating routing logic.
    pub fn shifts(
        &self,
        external_momenta: &BTreeMap<Atom, FourMomentum<Atom>>,
    ) -> Result<Vec<FourMomentum<Atom>>, PreparationError> {
        let loop_name = &self.family.loop_momenta()[0];
        self.quadratics
            .iter()
            .map(|quadratic| {
                let mut shift =
                    FourMomentum::from_args(Atom::Zero, Atom::Zero, Atom::Zero, Atom::Zero);
                for (momentum, coefficient) in self
                    .family
                    .kinematics()
                    .linear_terms(quadratic.momentum())?
                {
                    if momentum == *loop_name {
                        // QuadraticMomentum normalizes its pivot loop coefficient to one.
                        continue;
                    }
                    let components = external_momenta
                        .get(&momentum)
                        .ok_or_else(|| PreparationError::MissingMomentum(momentum.clone()))?;
                    shift += components.clone() * coefficient;
                }
                Ok(shift)
            })
            .collect()
    }

    /// Squared masses in normalized propagators, preserving exact expressions.
    pub fn squared_masses(&self) -> Vec<Atom> {
        self.quadratics
            .iter()
            .map(|quadratic| (-quadratic.remainder() / quadratic.scale()).together())
            .collect()
    }
}
