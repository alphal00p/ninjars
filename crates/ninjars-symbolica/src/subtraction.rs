//! Coefficient-level subtraction generated from exact residue expressions.
//!
//! The triangle ansatz follows original Ninja `src/polys_implem.cxx`.
//! Symbolica substitutes the double-cut parameterization, expands at infinity
//! with its series API, and extracts the nonnegative Laurent coefficients.
//! Numerical execution and Dual transformations reuse the shared kernel body.

use std::ops::Range;

use ninjars::{Complex, Error, ExpansionKind, Real, Result, SingleFloat};
use symbolica::{
    atom::{Atom, AtomCore},
    domains::{atom::AtomField, dual::DualNumber},
    evaluate::EvaluationDomain,
    symbol,
};

use crate::{
    LaurentDivision,
    kernel::{CompiledDualKernel, CompiledKernel, ExactKernel},
};

#[cfg(all(test, feature = "rust-codegen"))]
#[path = "algorithm_rank_probe.rs"]
mod algorithm_rank_probe;

fn triangle_residue(c: &[Atom], u: &Atom, v: &Atom, mu: &Atom, higher: bool) -> Atom {
    let mut residue = &c[0]
        + &c[7] * mu
        + (&c[1] + &c[8] * mu) * u
        + &c[2] * u.pow(2)
        + &c[3] * u.pow(3)
        + (&c[4] + &c[9] * mu) * v
        + &c[5] * v.pow(2)
        + &c[6] * v.pow(3);
    if higher {
        residue += &c[14] * mu.pow(2)
            + &c[10] * mu * u.pow(2)
            + &c[12] * u.pow(4)
            + &c[11] * mu * v.pow(2)
            + &c[13] * v.pow(4);
    }
    residue
}

fn bubble_residue(c: &[Atom], w: &Atom, u: &Atom, v: &Atom, mu: &Atom, higher: bool) -> Atom {
    let mut residue = &c[0]
        + &c[9] * mu
        + &c[1] * w
        + &c[2] * w.pow(2)
        + &c[3] * u
        + &c[4] * u.pow(2)
        + &c[5] * v
        + &c[6] * v.pow(2)
        + &c[7] * w * u
        + &c[8] * w * v;
    if higher {
        residue += mu * (&c[10] * w + &c[11] * u + &c[12] * v)
            + &c[13] * w.pow(3)
            + &c[14] * u.pow(3)
            + &c[15] * v.pow(3)
            + &c[16] * w.pow(2) * u
            + &c[17] * w.pow(2) * v
            + &c[18] * w * u.pow(2)
            + &c[19] * w * v.pow(2);
    }
    residue
}

/// Column ranges in each triangle-to-bubble subtraction input row.
#[derive(Clone, Debug)]
pub struct TriangleBubbleInputLayout {
    /// Original Ninja triangle c[0..10], or c[0..15] for rank n+1.
    pub residue: Range<usize>,
    /// [(a0+V_triangle)·E3, a1·E3, e3·E3, e4_eff·E3].
    pub projection_e3: Range<usize>,
    /// [(a0+V_triangle)·E4, a1·E4, e3·E4, e4_eff·E4].
    pub projection_e4: Range<usize>,
    /// [beta0, beta1, beta2] in beta(x)=beta0+beta1*x+beta2*x².
    pub beta: Range<usize>,
    /// Uncut denominator coefficients in `denominator_expansion` order.
    pub denominator: Range<usize>,
}

/// Triangle residue divided by its one line outside the target bubble cut.
///
/// The double-cut momentum is
/// `q=a0+a1*x+e3*t+e4_eff*(beta0+beta1*x+beta2*x²+mu²)/t`.
/// Triangle projections use **its own** origin `V_triangle` and transverse
/// vectors `E3,E4`. In particular, `(a0+V_triangle)·E3/E4` must not be replaced
/// by the target cut's origin. Dot products use FeynKit's +--- convention.
///
/// Rows return the correction to subtract from the divided numerator; the
/// kernel does not mutate an existing coefficient array. All nonnegative
/// powers for degree two (ordinary rank) or three (rank n+1) are returned,
/// matching `LaurentDivision::new(3,3/4,2).quotient_expansion()`.
/// Lower-rank reducers can select the matching subset by explicit powers.
/// The caller must reject singular leading denominator coefficients first.
pub struct TriangleBubbleSubtraction {
    higher_rank: bool,
    layout: TriangleBubbleInputLayout,
    denominator: ExpansionKind,
    output: ExpansionKind,
    pub(crate) exact: ExactKernel,
}

impl TriangleBubbleSubtraction {
    /// Restrict the existing correction to a validated bubble quotient layout.
    ///
    /// Selection follows explicit `(t,x,mu²)` powers, preserving their order.
    /// It does not infer semantics from input/output buffer dimensions.
    pub fn for_quotient(division: &LaurentDivision) -> Result<Self> {
        if division.cut_size() != 2 {
            return Err(Error::Unsupported(
                "triangle-to-bubble correction requires a bubble quotient".into(),
            ));
        }
        let mut plan = Self::new(division.higher_rank())?;
        let ExpansionKind::T2 { powers: full } = &plan.output else {
            unreachable!()
        };
        let ExpansionKind::T2 { powers: selected } = division.quotient_expansion() else {
            unreachable!()
        };
        plan.exact.expressions = selected
            .iter()
            .map(|power| {
                full.iter()
                    .position(|p| p == power)
                    .map(|index| plan.exact.expressions[index].clone())
                    .ok_or_else(|| {
                        Error::Expansion(
                            "bubble quotient power is absent from triangle correction".into(),
                        )
                    })
            })
            .collect::<Result<_>>()?;
        plan.output = division.quotient_expansion().clone();
        Ok(plan)
    }

    pub fn new(higher_rank: bool) -> Result<Self> {
        let count = if higher_rank { 15 } else { 10 };
        let layout = TriangleBubbleInputLayout {
            residue: 0..count,
            projection_e3: count..count + 4,
            projection_e4: count + 4..count + 8,
            beta: count + 8..count + 11,
            denominator: count + 11..count + 18,
        };
        let parameters: Vec<_> = (0..layout.denominator.end)
            .map(|i| Atom::var(symbol!(format!("ninjars_triangle_bubble::p{i}"))))
            .collect();
        let z_symbol = symbol!("ninjars_triangle_bubble::z");
        let z = Atom::var(z_symbol);
        let x = Atom::var(symbol!("ninjars_triangle_bubble::x"));
        let mu = Atom::var(symbol!("ninjars_triangle_bubble::mu_squared"));
        let beta = &parameters[layout.beta.clone()];
        let inverse_part = &beta[0] + &beta[1] * &x + &beta[2] * x.pow(2) + &mu;
        let project = |range: Range<usize>| {
            let p = &parameters[range];
            &p[0] + &p[1] * &x + &p[2] / &z + &p[3] * &inverse_part * &z
        };
        let u = project(layout.projection_e3.clone());
        let v = project(layout.projection_e4.clone());
        let c = &parameters[layout.residue.clone()];
        let residue = triangle_residue(c, &u, &v, &mu, higher_rank);
        let denominator = ExpansionKind::T2 {
            powers: vec![
                (1, 0, 0),
                (0, 0, 0),
                (0, 1, 0),
                (-1, 0, 0),
                (-1, 0, 1),
                (-1, 1, 0),
                (-1, 2, 0),
            ],
        };
        let d = &parameters[layout.denominator.clone()];
        let divisor = &d[0] / &z
            + &d[1]
            + &d[2] * &x
            + (&d[3] + &d[4] * &mu + &d[5] * &x + &d[6] * x.pow(2)) * &z;
        let degree = if higher_rank { 3 } else { 2 };
        let normalized = residue / divisor * z.pow(degree);
        let series = normalized
            .series(z_symbol, 0, degree)
            .map_err(|error| Error::Symbolic(error.to_string()))?
            .to_atom()
            .expand();
        let field = AtomField {
            statistical_zero_test: false,
            ..AtomField::new()
        };
        let variables = [z, x, mu];
        let polynomial = series.to_polynomial_in_vars_with_field::<i32>(&variables, &field);
        if (&polynomial)
            .into_iter()
            .any(|term| variables.iter().any(|v| term.coefficient.contains(v)))
        {
            return Err(Error::Expansion(
                "triangle subtraction did not yield polynomial coefficients".into(),
            ));
        }
        let mut powers = Vec::new();
        let mut expressions = Vec::new();
        for t in (0..=degree).rev() {
            for xp in 0..=degree - t {
                for mp in 0..=(degree - t - xp) / 2 {
                    powers.push((t, xp as u16, mp as u16));
                    expressions.push(
                        polynomial
                            .coefficient(&[degree - t, xp, mp])
                            .unwrap_or(Atom::Zero),
                    );
                }
            }
        }
        Ok(Self {
            higher_rank,
            layout,
            denominator,
            output: ExpansionKind::T2 { powers },
            exact: ExactKernel {
                expressions,
                parameters,
            },
        })
    }

    pub fn higher_rank(&self) -> bool {
        self.higher_rank
    }

    pub fn input_layout(&self) -> &TriangleBubbleInputLayout {
        &self.layout
    }
    pub fn input_columns(&self) -> usize {
        self.exact.parameters.len()
    }
    pub fn denominator_expansion(&self) -> &ExpansionKind {
        &self.denominator
    }
    pub fn output_expansion(&self) -> &ExpansionKind {
        &self.output
    }

    pub fn compile<T: Real + SingleFloat>(
        &self,
        prototype: &T,
    ) -> Result<TriangleBubbleEvaluator<T>>
    where
        Complex<T>: Real + EvaluationDomain,
    {
        Ok(TriangleBubbleEvaluator {
            kernel: self.exact.compile(prototype)?,
        })
    }

    pub fn compile_dual<T: Real + SingleFloat, D: Real + DualNumber<Complex<T>>>(
        &self,
        prototype: &T,
        dual: &D,
    ) -> Result<DualTriangleBubbleEvaluator<T, D>>
    where
        Complex<T>: Real + EvaluationDomain,
    {
        Ok(DualTriangleBubbleEvaluator {
            kernel: self.exact.compile_dual(prototype, dual)?,
        })
    }
}

/// Batched arithmetic over the topology-only exact subtraction expressions.
pub struct TriangleBubbleEvaluator<T> {
    kernel: CompiledKernel<T>,
}

impl<T: Real + SingleFloat> TriangleBubbleEvaluator<T>
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

/// Symbolica's recursive Dual transformation of the same subtraction body.
pub struct DualTriangleBubbleEvaluator<T, D> {
    kernel: CompiledDualKernel<T, D>,
}

impl<T: Real + SingleFloat, D: Real + DualNumber<Complex<T>>> DualTriangleBubbleEvaluator<T, D>
where
    Complex<T>: Real + EvaluationDomain,
{
    pub fn evaluate_batch(&mut self, input: &[D], output: &mut [D], rows: usize) -> Result<()> {
        self.kernel.evaluate_batch(input, output, rows)
    }
}

/// Input-column ranges for triangle/bubble subtraction from a single cut.
#[derive(Clone, Debug)]
pub struct TadpoleSubtractionInputLayout {
    /// Original triangle c[0..10/15] or bubble c[0..10/20].
    pub residue: Range<usize>,
    /// Three scalars per axis: [(a+V_source)·E, e3·E, e4_eff·E].
    /// Axis order is [E3,E4] for a triangle and [E2,E3,E4] for a bubble.
    pub projections: Vec<Range<usize>>,
    /// beta in q=a+t*e3+(beta+mu²)*e4_eff/t.
    pub beta: usize,
    /// One T3 denominator block per source line outside the single cut.
    pub denominators: Vec<Range<usize>>,
}

/// Exact triangle/bubble residue contribution to a single-cut expansion.
///
/// `source_cut_size` is three for a triangle (two uncut lines) or two for a
/// bubble (one uncut line). Projection offsets retain the source residue's
/// origin, independently of the target tadpole. The returned coefficients are
/// subtracted from the divided numerator. Full degree-one/two T3 output covers
/// ordinary/higher rank, including the higher-rank constant-in-t mu² term.
/// Numerical singularity and lane stability checks belong to the reducer.
pub struct TadpoleSubtraction {
    source_cut_size: usize,
    layout: TadpoleSubtractionInputLayout,
    denominator: ExpansionKind,
    output: ExpansionKind,
    pub(crate) exact: ExactKernel,
}

impl TadpoleSubtraction {
    pub fn new(source_cut_size: usize, higher_rank: bool) -> Result<Self> {
        if !matches!(source_cut_size, 2 | 3) {
            return Err(Error::Unsupported(
                "tadpole subtraction requires a triangle or bubble source".into(),
            ));
        }
        let triangle = source_cut_size == 3;
        let count = match (triangle, higher_rank) {
            (true, true) => 15,
            (false, true) => 20,
            _ => 10,
        };
        let axes = if triangle { 2 } else { 3 };
        let beta = count + 3 * axes;
        let layout = TadpoleSubtractionInputLayout {
            residue: 0..count,
            projections: (0..axes)
                .map(|i| count + 3 * i..count + 3 * (i + 1))
                .collect(),
            beta,
            denominators: (0..source_cut_size - 1)
                .map(|i| beta + 1 + 4 * i..beta + 1 + 4 * (i + 1))
                .collect(),
        };
        let columns = layout.denominators.last().unwrap().end;
        let parameters: Vec<_> = (0..columns)
            .map(|i| Atom::var(symbol!(format!("ninjars_tadpole_subtraction::p{i}"))))
            .collect();
        let z_symbol = symbol!("ninjars_tadpole_subtraction::z");
        let z = Atom::var(z_symbol);
        let mu = Atom::var(symbol!("ninjars_tadpole_subtraction::mu_squared"));
        let projections: Vec<_> = layout
            .projections
            .iter()
            .map(|range| {
                let p = &parameters[range.clone()];
                &p[0] + &p[1] / &z + &p[2] * (&parameters[layout.beta] + &mu) * &z
            })
            .collect();
        let c = &parameters[layout.residue.clone()];
        let residue = if triangle {
            triangle_residue(c, &projections[0], &projections[1], &mu, higher_rank)
        } else {
            bubble_residue(
                c,
                &projections[0],
                &projections[1],
                &projections[2],
                &mu,
                higher_rank,
            )
        };
        let denominator = ExpansionKind::T3 {
            powers: vec![(1, 0), (0, 0), (-1, 0), (-1, 1)],
        };
        let mut quotient = residue;
        for range in &layout.denominators {
            let d = &parameters[range.clone()];
            quotient /= &d[0] / &z + &d[1] + (&d[2] + &d[3] * &mu) * &z;
        }
        let degree = if higher_rank { 2 } else { 1 };
        let series = (quotient * z.pow(degree))
            .series(z_symbol, 0, degree)
            .map_err(|error| Error::Symbolic(error.to_string()))?
            .to_atom()
            .expand();
        let field = AtomField {
            statistical_zero_test: false,
            ..AtomField::new()
        };
        let variables = [z, mu];
        let polynomial = series.to_polynomial_in_vars_with_field::<i32>(&variables, &field);
        if (&polynomial)
            .into_iter()
            .any(|term| variables.iter().any(|v| term.coefficient.contains(v)))
        {
            return Err(Error::Expansion(
                "tadpole subtraction did not yield polynomial coefficients".into(),
            ));
        }
        let mut powers = Vec::new();
        let mut expressions = Vec::new();
        for t in (0..=degree).rev() {
            for mp in 0..=(degree - t) / 2 {
                powers.push((t, mp as u16));
                expressions.push(
                    polynomial
                        .coefficient(&[degree - t, mp])
                        .unwrap_or(Atom::Zero),
                );
            }
        }
        Ok(Self {
            source_cut_size,
            layout,
            denominator,
            output: ExpansionKind::T3 { powers },
            exact: ExactKernel {
                expressions,
                parameters,
            },
        })
    }

    pub fn source_cut_size(&self) -> usize {
        self.source_cut_size
    }
    pub fn input_layout(&self) -> &TadpoleSubtractionInputLayout {
        &self.layout
    }
    pub fn input_columns(&self) -> usize {
        self.exact.parameters.len()
    }
    pub fn denominator_expansion(&self) -> &ExpansionKind {
        &self.denominator
    }
    pub fn output_expansion(&self) -> &ExpansionKind {
        &self.output
    }

    pub fn compile<T: Real + SingleFloat>(
        &self,
        prototype: &T,
    ) -> Result<TadpoleSubtractionEvaluator<T>>
    where
        Complex<T>: Real + EvaluationDomain,
    {
        Ok(TadpoleSubtractionEvaluator {
            kernel: self.exact.compile(prototype)?,
        })
    }

    pub fn compile_dual<T: Real + SingleFloat, D: Real + DualNumber<Complex<T>>>(
        &self,
        prototype: &T,
        dual: &D,
    ) -> Result<DualTadpoleSubtractionEvaluator<T, D>>
    where
        Complex<T>: Real + EvaluationDomain,
    {
        Ok(DualTadpoleSubtractionEvaluator {
            kernel: self.exact.compile_dual(prototype, dual)?,
        })
    }
}

pub struct TadpoleSubtractionEvaluator<T> {
    kernel: CompiledKernel<T>,
}

impl<T: Real + SingleFloat> TadpoleSubtractionEvaluator<T>
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

pub struct DualTadpoleSubtractionEvaluator<T, D> {
    kernel: CompiledDualKernel<T, D>,
}

impl<T: Real + SingleFloat, D: Real + DualNumber<Complex<T>>> DualTadpoleSubtractionEvaluator<T, D>
where
    Complex<T>: Real + EvaluationDomain,
{
    pub fn evaluate_batch(&mut self, input: &[D], output: &mut [D], rows: usize) -> Result<()> {
        self.kernel.evaluate_batch(input, output, rows)
    }
}

#[cfg(feature = "native-kernels")]
impl TriangleBubbleSubtraction {
    /// Select the catalogued ordinary correction body by rank class and explicit
    /// quotient degree. An empty quotient keeps its empty interpreter body.
    pub fn compile_native<T: oneloop::rust_codegen::runtime::NativeFloat>(
        &self,
        prototype: &T,
    ) -> Result<TriangleBubbleEvaluator<T>>
    where
        Complex<T>: Real + EvaluationDomain,
    {
        let mut result = self.compile(prototype)?;
        if let Some(plan) =
            crate::native_kernels::bubbles::BubblePlan::correction(self.higher_rank, &self.output)?
        {
            crate::native_kernels::bubbles::install(&mut result.kernel, plan, prototype)?;
        }
        Ok(result)
    }
}
#[cfg(feature = "native-kernels")]
impl<T: Real + SingleFloat> TriangleBubbleEvaluator<T>
where
    Complex<T>: Real + EvaluationDomain,
{
    pub fn backend(&self) -> crate::KernelBackend {
        self.kernel.backend()
    }
}
#[cfg(feature = "native-kernels")]
impl<T: Real + SingleFloat, D: Real + DualNumber<Complex<T>>> DualTriangleBubbleEvaluator<T, D>
where
    Complex<T>: Real + EvaluationDomain,
{
    /// No generated Dual correction is selected in the current catalog.
    pub fn backend(&self) -> crate::KernelBackend {
        self.kernel.backend()
    }
}
