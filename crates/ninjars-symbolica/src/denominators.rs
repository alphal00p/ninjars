//! Uncut-denominator Laurent expansions using shared momentum arithmetic.

use crate::{
    LaurentDivision,
    kernel::{CompiledDualKernel, CompiledKernel, ExactKernel},
};
use ninjars::{Complex, Error, ExpansionKind, FourMomentum, Real, Result, SingleFloat};
use symbolica::{
    atom::{Atom, AtomCore},
    domains::{atom::AtomField, dual::DualNumber},
    evaluate::EvaluationDomain,
    symbol,
};

/// Expand `D_uncut - D_anchor` on a cut, where `D_anchor=0`.
///
/// Inputs per row: callback vectors (three for T3, four for T2), callback
/// parameters (one/three), anchor shift (four components), anchor squared mass,
/// uncut shift (four components), uncut squared mass. FourMomentum component
/// order is `(E,x,y,z)`. Outputs follow `expansion()`, matching LaurentDivision.
/// Multiple uncut lines and phase-space lanes can share one instruction batch.
pub struct DenominatorExpansion {
    kind: ExpansionKind,
    pub(crate) exact: ExactKernel,
}

impl DenominatorExpansion {
    pub fn new(cut_size: usize) -> Result<Self> {
        let layout = LaurentDivision::new(cut_size, cut_size, cut_size)?;
        let kind = layout.denominator_expansion().clone();
        let bubble = cut_size == 2;
        let vectors = if bubble { 4 } else { 3 };
        let scalars = if bubble { 3 } else { 1 };
        let columns = vectors * 4 + scalars + 10;
        let p: Vec<_> = (0..columns)
            .map(|i| Atom::var(symbol!(format!("ninjars_denominator::p{i}"))))
            .collect();
        let vector = |i: usize| {
            FourMomentum::from_args(
                p[i].clone(),
                p[i + 1].clone(),
                p[i + 2].clone(),
                p[i + 3].clone(),
            )
        };
        let t = Atom::var(symbol!("ninjars_denominator::t"));
        let x = Atom::var(symbol!("ninjars_denominator::x"));
        let mu = Atom::var(symbol!("ninjars_denominator::mu_squared"));
        let q = if bubble {
            vector(0)
                + vector(4) * x.clone()
                + vector(8) * t.clone()
                + vector(12) * ((&p[16] + &p[17] * &x + &p[18] * x.pow(2) + &mu) / &t)
        } else {
            vector(0) + vector(4) * t.clone() + vector(8) * ((&p[12] + &mu) / &t)
        };
        let offset = vectors * 4 + scalars;
        let anchor = q.clone() + vector(offset);
        let uncut = q + vector(offset + 5);
        let expression =
            (uncut.dot(&uncut) - anchor.dot(&anchor) + &p[offset + 4] - &p[offset + 9]).expand();
        let variables = [t, x, mu];
        let field = AtomField {
            statistical_zero_test: false,
            ..AtomField::new()
        };
        let polynomial = expression.to_polynomial_in_vars_with_field::<i32>(&variables, &field);
        let powers: Vec<_> = match &kind {
            ExpansionKind::T3 { powers } => {
                powers.iter().map(|&(t, m)| [t, 0, i32::from(m)]).collect()
            }
            ExpansionKind::T2 { powers } => powers
                .iter()
                .map(|&(t, x, m)| [t, i32::from(x), i32::from(m)])
                .collect(),
            _ => unreachable!(),
        };
        if (&polynomial)
            .into_iter()
            .any(|term| !powers.iter().any(|p| p.as_slice() == term.exponents))
        {
            return Err(Error::Expansion(
                "unexpected uncut denominator Laurent power".into(),
            ));
        }
        let expressions = powers
            .iter()
            .map(|power| polynomial.coefficient(power).unwrap_or(Atom::Zero))
            .collect();
        Ok(Self {
            kind,
            exact: ExactKernel {
                expressions,
                parameters: p,
            },
        })
    }
    pub fn expansion(&self) -> &ExpansionKind {
        &self.kind
    }
    pub fn input_columns(&self) -> usize {
        self.exact.parameters.len()
    }
    pub fn compile<T: Real + SingleFloat>(&self, prototype: &T) -> Result<DenominatorEvaluator<T>>
    where
        Complex<T>: Real + EvaluationDomain,
    {
        Ok(DenominatorEvaluator {
            kernel: self.exact.compile(prototype)?,
        })
    }
    pub fn compile_dual<T: Real + SingleFloat, D: Real + DualNumber<Complex<T>>>(
        &self,
        prototype: &T,
        dual: &D,
    ) -> Result<DualDenominatorEvaluator<T, D>>
    where
        Complex<T>: Real + EvaluationDomain,
    {
        Ok(DualDenominatorEvaluator {
            kernel: self.exact.compile_dual(prototype, dual)?,
        })
    }
}
pub struct DenominatorEvaluator<T> {
    kernel: CompiledKernel<T>,
}
impl<T: Real + SingleFloat> DenominatorEvaluator<T>
where
    Complex<T>: Real + EvaluationDomain,
{
    pub fn evaluate_batch(
        &mut self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        rows: usize,
    ) -> Result<()> {
        self.kernel.evaluate_batch(input, output, rows)
    }
}
pub struct DualDenominatorEvaluator<T, D> {
    kernel: CompiledDualKernel<T, D>,
}
impl<T: Real + SingleFloat, D: Real + DualNumber<Complex<T>>> DualDenominatorEvaluator<T, D>
where
    Complex<T>: Real + EvaluationDomain,
{
    pub fn evaluate_batch(&mut self, input: &[D], output: &mut [D], rows: usize) -> Result<()> {
        self.kernel.evaluate_batch(input, output, rows)
    }
}

#[cfg(feature = "native-kernels")]
impl DenominatorExpansion {
    fn native_plan(&self) -> crate::native_kernels::DenominatorPlan {
        match self.kind {
            ExpansionKind::T2 { .. } => crate::native_kernels::DenominatorPlan::T2,
            ExpansionKind::T3 { .. } => crate::native_kernels::DenominatorPlan::T3,
            _ => unreachable!("DenominatorExpansion creates only T2/T3 plans"),
        }
    }

    /// Compile the existing plan and explicitly select generated Rust arithmetic.
    /// Constants use `prototype` precision; input values retain their own precision.
    pub fn compile_native<T: oneloop::rust_codegen::runtime::NativeFloat>(
        &self,
        prototype: &T,
    ) -> Result<DenominatorEvaluator<T>>
    where
        Complex<T>: Real + EvaluationDomain,
    {
        let mut result = self.compile(prototype)?;
        crate::native_kernels::install_denominator(
            &mut result.kernel,
            self.native_plan(),
            prototype,
        )?;
        Ok(result)
    }

    /// Select native Rust for the supported ordered five-component Dual shape.
    /// Other shapes keep the ordinary Symbolica evaluator and report that backend.
    pub fn compile_dual_native<
        T: oneloop::rust_codegen::runtime::NativeFloat,
        D: Real + DualNumber<Complex<T>>,
    >(
        &self,
        prototype: &T,
        dual: &D,
    ) -> Result<DualDenominatorEvaluator<T, D>>
    where
        Complex<T>: Real + EvaluationDomain,
    {
        let mut result = self.compile_dual(prototype, dual)?;
        crate::native_kernels::install_dual_denominator(
            &mut result.kernel,
            self.native_plan(),
            prototype,
        )?;
        Ok(result)
    }
}

#[cfg(feature = "native-kernels")]
impl<T: Real + SingleFloat> DenominatorEvaluator<T>
where
    Complex<T>: Real + EvaluationDomain,
{
    pub fn backend(&self) -> crate::KernelBackend {
        self.kernel.backend()
    }
}

#[cfg(feature = "native-kernels")]
impl<T: Real + SingleFloat, D: Real + DualNumber<Complex<T>>> DualDenominatorEvaluator<T, D>
where
    Complex<T>: Real + EvaluationDomain,
{
    pub fn backend(&self) -> crate::KernelBackend {
        self.kernel.backend()
    }
}
