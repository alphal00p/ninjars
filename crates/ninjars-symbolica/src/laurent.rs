//! Laurent quotient kernels generated once from exact Symbolica series.

use ninjars::{Complex, Error, ExpansionKind, Real, Result, SingleFloat};
use symbolica::{
    atom::{Atom, AtomCore},
    domains::{atom::AtomField, dual::DualNumber},
    evaluate::EvaluationDomain,
    symbol,
};

use crate::kernel::{CompiledDualKernel, CompiledKernel, ExactKernel};

/// Topology-only division of numerator Laurent coefficients by uncut lines.
///
/// Supports the T3 parameterization for triple/single cuts and T2 for double
/// cuts, including rank `n+1`. All powers are explicit; mu² has rank weight two.
/// Preparing this plan uses Symbolica series algebra once. Compiled evaluators
/// execute across the entire batch and support changing coefficients per row.
pub struct LaurentDivision {
    numerator: ExpansionKind,
    denominator: ExpansionKind,
    quotient: ExpansionKind,
    uncut: usize,
    cut_size: usize,
    higher_rank: bool,
    pub(crate) exact: ExactKernel,
}

fn triples(kind: &ExpansionKind) -> Vec<[i32; 3]> {
    match kind {
        ExpansionKind::T3 { powers } => powers
            .iter()
            .map(|&(t, mu)| [t, 0, i32::from(mu)])
            .collect(),
        ExpansionKind::T2 { powers } => powers
            .iter()
            .map(|&(t, x, mu)| [t, i32::from(x), i32::from(mu)])
            .collect(),
        ExpansionKind::Mu { .. } => unreachable!("Laurent division uses T2/T3 expansions"),
    }
}

fn powers(degree: i32, shift: i32, bubble: bool) -> ExpansionKind {
    if bubble {
        let mut result = Vec::new();
        for t in (0..=degree).rev() {
            for x in 0..=degree - t {
                for mu in 0..=(degree - t - x) / 2 {
                    result.push((t + shift, x as u16, mu as u16));
                }
            }
        }
        ExpansionKind::T2 { powers: result }
    } else {
        let mut result = Vec::new();
        for t in (0..=degree).rev() {
            for mu in 0..=(degree - t) / 2 {
                result.push((t + shift, mu as u16));
            }
        }
        ExpansionKind::T3 { powers: result }
    }
}

impl LaurentDivision {
    pub fn new(propagators: usize, rank: usize, cut_size: usize) -> Result<Self> {
        if propagators == 0 {
            return Err(Error::NoPropagators);
        }
        if rank > propagators.saturating_add(1) {
            return Err(Error::Rank { rank, propagators });
        }
        if !(1..=3).contains(&cut_size) || cut_size > propagators {
            return Err(Error::Unsupported(
                "Laurent division requires a single, double or triple cut".into(),
            ));
        }
        let higher_rank = rank > propagators;
        let uncut = propagators - cut_size;
        let shift =
            i32::try_from(uncut).map_err(|_| Error::Unsupported("too many propagators".into()))?;
        let rank =
            i32::try_from(rank).map_err(|_| Error::Unsupported("rank exceeds i32".into()))?;
        let degree = rank - shift;
        let bubble = cut_size == 2;
        let numerator = powers(degree, shift, bubble);
        let quotient = powers(degree, 0, bubble);
        let denominator = if bubble {
            ExpansionKind::T2 {
                powers: vec![
                    (1, 0, 0),
                    (0, 0, 0),
                    (0, 1, 0),
                    (-1, 0, 0),
                    (-1, 0, 1),
                    (-1, 1, 0),
                    (-1, 2, 0),
                ],
            }
        } else {
            ExpansionKind::T3 {
                powers: vec![(1, 0), (0, 0), (-1, 0), (-1, 1)],
            }
        };
        let columns = uncut
            .checked_mul(denominator.len())
            .and_then(|n| n.checked_add(numerator.len()))
            .ok_or_else(|| Error::Unsupported("Laurent dimensions overflow usize".into()))?;
        let parameters: Vec<_> = (0..columns)
            .map(|i| Atom::var(symbol!(format!("ninjars_division::a{i}"))))
            .collect();
        let z_symbol = symbol!("ninjars_division::z");
        let z = Atom::var(z_symbol);
        let x = Atom::var(symbol!("ninjars_division::x"));
        let mu = Atom::var(symbol!("ninjars_division::mu_squared"));
        let mut normalized = Atom::Zero;
        for (coefficient, [t, xp, mp]) in parameters.iter().zip(triples(&numerator)) {
            normalized += coefficient * z.pow(rank - t) * x.pow(xp) * mu.pow(mp);
        }
        for line in 0..uncut {
            let start = numerator.len() + line * denominator.len();
            let mut denom = Atom::Zero;
            for (coefficient, [t, xp, mp]) in parameters[start..start + denominator.len()]
                .iter()
                .zip(triples(&denominator))
            {
                denom += coefficient * z.pow(1 - t) * x.pow(xp) * mu.pow(mp);
            }
            normalized /= denom;
        }
        let expressions = if degree < 0 {
            Vec::new()
        } else {
            let series = normalized
                .series(z_symbol, 0, degree)
                .map_err(|e| Error::Symbolic(e.to_string()))?
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
                    "Laurent quotient did not yield polynomial coefficients".into(),
                ));
            }
            triples(&quotient)
                .iter()
                .map(|&[t, xp, mp]| {
                    polynomial
                        .coefficient(&[degree - t, xp, mp])
                        .unwrap_or(Atom::Zero)
                })
                .collect()
        };
        Ok(Self {
            numerator,
            denominator,
            quotient,
            uncut,
            cut_size,
            higher_rank,
            exact: ExactKernel {
                expressions,
                parameters,
            },
        })
    }

    /// Numerator callback coefficients needed to obtain nonnegative t powers.
    pub fn numerator_expansion(&self) -> &ExpansionKind {
        &self.numerator
    }
    /// Coefficients of each uncut denominator, in the documented explicit order.
    pub fn denominator_expansion(&self) -> &ExpansionKind {
        &self.denominator
    }
    pub fn quotient_expansion(&self) -> &ExpansionKind {
        &self.quotient
    }
    pub fn uncut_count(&self) -> usize {
        self.uncut
    }
    pub fn cut_size(&self) -> usize {
        self.cut_size
    }
    pub fn higher_rank(&self) -> bool {
        self.higher_rank
    }
    /// Each input row contains numerator coefficients followed by one block per
    /// uncut denominator. Output rows follow `quotient_expansion`.
    pub fn input_columns(&self) -> usize {
        self.exact.parameters.len()
    }

    pub fn compile<T: Real + SingleFloat>(
        &self,
        prototype: &T,
    ) -> Result<LaurentDivisionEvaluator<T>>
    where
        Complex<T>: Real + EvaluationDomain,
    {
        Ok(LaurentDivisionEvaluator {
            kernel: self.exact.compile(prototype)?,
        })
    }

    pub fn compile_dual<T: Real + SingleFloat, D: Real + DualNumber<Complex<T>>>(
        &self,
        prototype: &T,
        dual: &D,
    ) -> Result<DualLaurentDivisionEvaluator<T, D>>
    where
        Complex<T>: Real + EvaluationDomain,
    {
        Ok(DualLaurentDivisionEvaluator {
            kernel: self.exact.compile_dual(prototype, dual)?,
        })
    }
}

/// Arithmetic kernel for active, nonsingular rows selected by the reducer.
/// The reducer must check each denominator's leading coefficient and numerical
/// stability before committing outputs; this low-level kernel does no rescue.
pub struct LaurentDivisionEvaluator<T> {
    kernel: CompiledKernel<T>,
}

impl<T: Real + SingleFloat> LaurentDivisionEvaluator<T>
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

/// The same exact Laurent kernel transformed by Symbolica for a fixed Dual shape.
pub struct DualLaurentDivisionEvaluator<T, D> {
    kernel: CompiledDualKernel<T, D>,
}

impl<T: Real + SingleFloat, D: Real + DualNumber<Complex<T>>> DualLaurentDivisionEvaluator<T, D>
where
    Complex<T>: Real + EvaluationDomain,
{
    pub fn evaluate_batch(&mut self, input: &[D], output: &mut [D], rows: usize) -> Result<()> {
        self.kernel.evaluate_batch(input, output, rows)
    }
}
