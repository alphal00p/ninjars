//! Exact component binding through the existing kinematics operations.

use std::collections::BTreeMap;

use feynkit_kinematics::{FourMomentum, Kinematics};
use symbolica::atom::{Atom, AtomCore};

use crate::{PreparationError, PreparedIntegrand};

impl PreparedIntegrand {
    /// Bind scalar products to exact component expressions using the shared metric.
    ///
    /// Keys include the literal `family.external_momenta()` atoms. Additional
    /// rank-one carriers, such as polarization vectors, may also be bound.
    /// The four loop components may themselves be symbols; external components
    /// may retain arbitrary-precision or Dual evaluator parameters.
    pub fn component_kinematics(
        &self,
        loop_components: &FourMomentum<Atom>,
        external_momenta: &BTreeMap<Atom, FourMomentum<Atom>>,
    ) -> Result<Kinematics, PreparationError> {
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
                kinematics = kinematics.with_scalar_product(p_name, q_name, p.dot(q))?;
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
        Ok(self
            .component_kinematics(loop_components, external_momenta)?
            .apply(&self.numerator)
            .expand())
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
                    if &momentum == loop_name {
                        // QuadraticMomentum normalizes its pivot loop coefficient to one.
                        continue;
                    }
                    let components = external_momenta
                        .get(&momentum)
                        .ok_or_else(|| PreparationError::MissingMomentum(momentum.clone()))?;
                    shift = shift + components.clone() * coefficient;
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
