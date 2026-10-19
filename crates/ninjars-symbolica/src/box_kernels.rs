//! Exact Symbolica graphs for quadruple-cut division and interpolation.

use crate::{
    kernel::{CompiledDualKernel, CompiledKernel, ExactKernel, KernelExecutor},
    value_checks::{
        DualChecks, Negligible, OrdinaryChecks, ValueChecks, denominator_scale,
        difference_dot_scale, square_scale,
    },
};
use ninjars::{
    BatchShape, BoxCoefficientMode, BoxDivisionRequest, BoxMuDivisionRequest, BoxMuMode,
    BoxProjectionRequest, BoxReductionKernels, BoxReductionOptions, Complex, Error, ExpansionKind,
    FloatLike, FourMomentum, RankClass, Real, Result, SingleFloat, Topology,
};
use std::marker::PhantomData;
use symbolica::{
    atom::{Atom, AtomCore},
    domains::{atom::AtomField, dual::DualNumber},
    evaluate::EvaluationDomain,
    symbol,
};

fn parameters(namespace: &str, count: usize) -> Vec<Atom> {
    (0..count)
        .map(|i| Atom::var(symbol!(format!("{namespace}::p{i}"))))
        .collect()
}
fn vector(p: &[Atom], offset: usize) -> FourMomentum<Atom> {
    FourMomentum::from_args(
        p[offset].clone(),
        p[offset + 1].clone(),
        p[offset + 2].clone(),
        p[offset + 3].clone(),
    )
}
fn append_vector<V: Clone>(out: &mut Vec<V>, v: &FourMomentum<V>) {
    out.extend(v.components().into_iter().cloned());
}

/// Projections `(u+, u-, u+ - u-)` of both roots of every sampled mu² level
/// on `v_orth`. Row: `[V_first, v_orth, (q+, q-) per level]`.
fn projections(levels: usize) -> ExactKernel {
    let p = parameters("ninjars_box_root_projections", 8 + 8 * levels);
    let origin = vector(&p, 0);
    let v = vector(&p, 4);
    let mut expressions = Vec::new();
    for level in 0..levels {
        let up = (vector(&p, 8 + level * 8) + origin.clone()).dot(&v);
        let um = (vector(&p, 12 + level * 8) + origin.clone()).dot(&v);
        expressions.extend([up.clone(), um.clone(), up - um]);
    }
    ExactKernel {
        expressions,
        parameters: p,
    }
}

/// Rank n+2 mu-expansion division on the exact d-dimensional quadruple cut.
///
/// Row: `[n0, n1, n2, v², (d0, d1) per uncut line, beta]`. Here n0, n1, n2 are
/// the coefficients of t^r, t^(r-1), t^(r-2) of N(q, mu²) along
/// `q = t v_orth + a`, `mu² = t² v² - beta` (`ExpansionKind::MuShifted`), on
/// which `D_first = 0` exactly and each uncut `D_l = d0 t + d1`. The residue
/// `c0 + c2 mu² + c4 mu⁴ + c6 mu⁶ + (c1 + c3 mu² + c5 mu⁴) q·v` then gives
/// `c6 = n0/(v²)³`, `c5 = n1/(v²)³` and `c4 = n2/(v²)² + 3 beta c6`
/// (`tools/rank_n2/README.md`). Pentagons only reach t¹ and need no
/// subtraction. Output `[c4, c5, c6]`.
fn mu_division_n2(uncut: usize) -> Result<ExactKernel> {
    let p = parameters("ninjars_box_mu_division_n2", 5 + 2 * uncut);
    let z_symbol = symbol!("ninjars_box_mu_division_n2::z");
    let z = Atom::var(z_symbol);
    let mut normalized = &p[0] + &p[1] * &z + &p[2] * z.pow(2);
    for line in 0..uncut {
        normalized /= &p[4 + 2 * line] + &p[5 + 2 * line] * &z;
    }
    let series = normalized
        .series(z_symbol, 0, 2)
        .map_err(|e| Error::Symbolic(e.to_string()))?
        .to_atom()
        .expand();
    let field = AtomField {
        statistical_zero_test: false,
        ..AtomField::new()
    };
    let polynomial = series.to_polynomial_in_vars_with_field::<i32>(&[z], &field);
    let coefficient = |power: i32| polynomial.coefficient(&[power]).unwrap_or(Atom::Zero);
    let v2 = &p[3];
    let beta = &p[4 + 2 * uncut];
    let c6 = coefficient(0) / v2.pow(3);
    let c5 = coefficient(1) / v2.pow(3);
    let c4 = coefficient(2) / v2.pow(2) + beta * &c6 * 3;
    Ok(ExactKernel {
        expressions: vec![c4, c5, c6],
        parameters: p,
    })
}

/// Rank n+2 interpolation. Row: `[s, c4, c5, c6, (F+, F-, u+, u-) per mu²
/// level]` with the levels `0, s, -s, 2s` truncated to the mode's count, F the
/// pentagon-subtracted quotients at both roots and `c4, c5, c6` from
/// [`mu_division_n2`] (zero in sampling mode). Output c0..c6. With E and O the
/// even and odd parts at a level:
/// - expansion: `c2 = (E(s) - c0)/s - c4 s - c6 s²`, `c3 = (O(s) - c1)/s - c5 s`;
/// - sampling: `c4 = (E(s) + E(-s) - 2c0)/(2s²)`,
///   `c6 = (E(2s) - c0 - (E(s) - E(-s)) - 4 c4 s²)/(6 s³)`,
///   `c2 = (E(s) - E(-s))/(2s) - c6 s²`, `c5 = (O(s) + O(-s) - 2c1)/(2s²)`,
///   `c3 = (O(s) - O(-s))/(2s)`.
fn interpolation_n2(options: BoxReductionOptions, levels: usize) -> ExactKernel {
    let p = parameters("ninjars_box_interpolation_n2", 4 + 4 * levels);
    let s = &p[0];
    let even = |level: usize| (&p[4 + 4 * level] + &p[5 + 4 * level]) / 2;
    let odd = |level: usize| {
        (&p[4 + 4 * level] - &p[5 + 4 * level]) / (&p[6 + 4 * level] - &p[7 + 4 * level])
    };
    let full = options.coefficients == BoxCoefficientMode::Full;
    let sampling = options.mu_extraction == BoxMuMode::Sampling;
    let c0 = even(0);
    let c1 = if full { odd(0) } else { Atom::Zero };
    let (c4, c6) = if sampling {
        let c4 = (even(1) + even(2) - &c0 * 2) / (s.pow(2) * 2);
        let c6 = (even(3) - &c0 - (even(1) - even(2)) - &c4 * s.pow(2) * 4) / (s.pow(3) * 6);
        (c4, c6)
    } else {
        (p[1].clone(), p[3].clone())
    };
    let (c2, c3, c5) = if !full {
        (Atom::Zero, Atom::Zero, Atom::Zero)
    } else if sampling {
        (
            (even(1) - even(2)) / (s * 2) - &c6 * s.pow(2),
            (odd(1) - odd(2)) / (s * 2),
            (odd(1) + odd(2) - &c1 * 2) / (s.pow(2) * 2),
        )
    } else {
        (
            (even(1) - &c0) / s - &c4 * s - &c6 * s.pow(2),
            (odd(1) - &c1) / s - &p[2] * s,
            p[2].clone(),
        )
    };
    ExactKernel {
        expressions: vec![c0, c1, c2, c3, c4, c5, c6],
        parameters: p,
    }
}

pub(crate) struct Plans {
    pub(crate) samples: ExactKernel,
    pub(crate) denominators: ExactKernel,
    pub(crate) division: ExactKernel,
    pub(crate) mu_denominators: ExactKernel,
    pub(crate) mu_division: ExactKernel,
    pub(crate) projections: ExactKernel,
    pub(crate) interpolation: ExactKernel,
}
impl Plans {
    pub(crate) fn new(n: usize, rank: usize, options: BoxReductionOptions) -> Result<Self> {
        let class = Topology::new(n, rank)?.rank_class();
        if n < 4 {
            return Err(Error::Unsupported(
                "box kernels require at least four propagators".into(),
            ));
        }
        let uncut = n - 4;
        let higher = rank > n;
        let levels = options.sample_count(n, rank);
        let p = parameters("ninjars_box_mu_samples", 1);
        let samples = ExactKernel {
            expressions: [Atom::Zero, p[0].clone(), -&p[0], &p[0] * 2][..levels].to_vec(),
            parameters: p,
        };
        let p = parameters("ninjars_box_denominators", 5 + 5 * uncut);
        let q = vector(&p, 0);
        let expressions = (0..uncut)
            .map(|i| {
                let shifted = q.clone() + vector(&p, 5 + 5 * i);
                shifted.dot(&shifted) - &p[9 + 5 * i] - &p[4]
            })
            .collect();
        let denominators = ExactKernel {
            expressions,
            parameters: p,
        };
        let p = parameters("ninjars_box_division", 2 + 2 * uncut);
        let mut numerator = p[0].clone();
        for parent in 0..uncut {
            let term = (0..uncut)
                .filter(|&line| line != parent)
                .fold(&p[1] * &p[2 + uncut + parent], |value, line| {
                    value * &p[2 + line]
                });
            numerator -= term;
        }
        let denominator = p[2..2 + uncut]
            .iter()
            .fold(Atom::num(1), |value, d| value * d);
        let division = ExactKernel {
            expressions: vec![numerator / denominator],
            parameters: p,
        };
        let p = parameters("ninjars_box_mu_denominators", 13 + 5 * uncut);
        let v = vector(&p, 0);
        let a = vector(&p, 4);
        let origin = vector(&p, 8);
        let mut expressions = vec![v.dot(&v)];
        for line in 0..uncut {
            let shift = vector(&p, 13 + line * 5);
            let delta = shift.clone() - origin.clone();
            expressions.push(v.dot(&delta) * 2);
            expressions.push(
                shift.dot(&shift) - origin.dot(&origin) - &p[17 + line * 5]
                    + &p[12]
                    + a.dot(&delta) * 2,
            );
        }
        let mu_denominators = ExactKernel {
            expressions,
            parameters: p,
        };
        if class == RankClass::NPlus2 {
            return Ok(Self {
                samples,
                denominators,
                division,
                mu_denominators,
                mu_division: mu_division_n2(uncut)?,
                projections: projections(levels),
                interpolation: interpolation_n2(options, levels),
            });
        }
        let width = if higher { 2 } else { 1 };
        let p = parameters("ninjars_box_mu_division", width + 1 + 2 * uncut);
        let z_symbol = symbol!("ninjars_box_mu_division::z");
        let z = Atom::var(z_symbol);
        let mut normalized = p[0].clone();
        if higher {
            normalized += &p[1] * &z;
        }
        for line in 0..uncut {
            normalized /= &p[width + 1 + 2 * line] + &p[width + 2 + 2 * line] * &z;
        }
        // Use Symbolica's series/coefficient facilities for the two leading
        // powers, rather than implementing a polynomial division recurrence.
        let series = normalized
            .series(z_symbol, 0, if higher { 1 } else { 0 })
            .map_err(|e| Error::Symbolic(e.to_string()))?
            .to_atom()
            .expand();
        let field = AtomField {
            statistical_zero_test: false,
            ..AtomField::new()
        };
        let polynomial = series.to_polynomial_in_vars_with_field::<i32>(&[z], &field);
        let leading = polynomial.coefficient(&[0]).unwrap_or(Atom::Zero);
        let next = if higher {
            polynomial.coefficient(&[1]).unwrap_or(Atom::Zero)
        } else {
            leading.clone()
        };
        let mu_division = ExactKernel {
            expressions: vec![
                next / p[width].pow(2),
                if higher {
                    leading / p[width].pow(3)
                } else {
                    Atom::Zero
                },
            ],
            parameters: p,
        };
        let projections = projections(levels);
        // [scale,c4_mu,c5_mu, then (Fplus,Fminus,uplus,uminus) per mu level].
        let p = parameters("ninjars_box_interpolation", 3 + 4 * levels);
        let even = |level: usize| (&p[3 + 4 * level] + &p[4 + 4 * level]) / 2;
        let odd = |level: usize| {
            (&p[3 + 4 * level] - &p[4 + 4 * level]) / (&p[5 + 4 * level] - &p[6 + 4 * level])
        };
        let c0 = even(0);
        let full = options.coefficients == BoxCoefficientMode::Full;
        let c1 = if full { odd(0) } else { Atom::Zero };
        let c4 = if rank < n {
            Atom::Zero
        } else if options.mu_extraction == BoxMuMode::Expansion {
            p[1].clone()
        } else {
            (even(1) + even(2) - &c0 * 2) / (p[0].pow(2) * 2)
        };
        let c5 = if higher && full {
            if options.mu_extraction == BoxMuMode::Expansion {
                p[2].clone()
            } else {
                (odd(1) + odd(2) - &c1 * 2) / (p[0].pow(2) * 2)
            }
        } else {
            Atom::Zero
        };
        let c2 = if full {
            if options.mu_extraction == BoxMuMode::Sampling {
                (even(1) - even(2)) / (&p[0] * 2)
            } else {
                (even(1) - &c0) / &p[0] - &c4 * &p[0]
            }
        } else {
            Atom::Zero
        };
        // Symmetric interpolation also avoids the original higher-rank
        // sampling path's suspected missing scale factor in its c3 update.
        let c3 = if full {
            if options.mu_extraction == BoxMuMode::Sampling {
                (odd(1) - odd(2)) / (&p[0] * 2)
            } else {
                (odd(1) - &c1) / &p[0] - &c5 * &p[0]
            }
        } else {
            Atom::Zero
        };
        let mut expressions = vec![c0, c1, c2, c3, c4];
        if higher {
            expressions.push(c5);
        }
        let interpolation = ExactKernel {
            expressions,
            parameters: p,
        };
        Ok(Self {
            samples,
            denominators,
            division,
            mu_denominators,
            mu_division,
            projections,
            interpolation,
        })
    }
}

struct Engine<V, K, Checks, R> {
    n: usize,
    rank: usize,
    /// Rank n+2: three mu-expansion terms on the exact d-dimensional cut and
    /// the seven-coefficient residue.
    rank_n2: bool,
    options: BoxReductionOptions,
    expansion: Option<ExpansionKind>,
    samples: K,
    denominators: K,
    division: K,
    mu_denominators: K,
    mu_division: K,
    projections: K,
    interpolation: K,
    zero: V,
    negligible: Negligible<R>,
    checks: PhantomData<fn() -> Checks>,
    input: Vec<V>,
    probe: Vec<V>,
    packed: Vec<V>,
    values: Vec<V>,
    eligible: Vec<usize>,
    active: Vec<usize>,
}
impl<V: FloatLike, K: KernelExecutor<V>, R: Real + SingleFloat, Checks: ValueChecks<V, Real = R>>
    Engine<V, K, Checks, R>
{
    fn new(
        n: usize,
        rank: usize,
        options: BoxReductionOptions,
        prototype: &V,
        mut compile: impl FnMut(&ExactKernel) -> Result<K>,
    ) -> Result<Self> {
        let plans = Plans::new(n, rank, options)?;
        let rank_n2 = RankClass::of(n, rank) == Some(RankClass::NPlus2);
        let expansion = if rank_n2 && options.mu_extraction == BoxMuMode::Expansion {
            let r = rank as i32;
            Some(ExpansionKind::MuShifted {
                powers: vec![r, r - 1, r - 2],
            })
        } else if rank >= n && options.mu_extraction == BoxMuMode::Expansion {
            Some(ExpansionKind::Mu {
                powers: if rank > n {
                    vec![rank as i32, rank as i32 - 1]
                } else {
                    vec![rank as i32]
                },
            })
        } else {
            None
        };
        Ok(Self {
            n,
            rank,
            rank_n2,
            options,
            expansion,
            samples: compile(&plans.samples)?,
            denominators: compile(&plans.denominators)?,
            division: compile(&plans.division)?,
            mu_denominators: compile(&plans.mu_denominators)?,
            mu_division: compile(&plans.mu_division)?,
            projections: compile(&plans.projections)?,
            interpolation: compile(&plans.interpolation)?,
            zero: prototype.zero(),
            negligible: Negligible::new(&Checks::magnitude(prototype)),
            checks: PhantomData,
            input: Vec::new(),
            probe: Vec::new(),
            packed: Vec::new(),
            values: Vec::new(),
            eligible: Vec::new(),
            active: Vec::new(),
        })
    }
    fn width(&self) -> usize {
        if self.rank_n2 {
            7
        } else if self.rank > self.n {
            6
        } else {
            5
        }
    }
    /// Mu-expansion numerator coefficients per row: t^r (rank n), plus t^(r-1)
    /// (n+1), plus t^(r-2) (n+2).
    fn mu_width(&self) -> usize {
        if self.rank_n2 {
            3
        } else if self.rank > self.n {
            2
        } else {
            1
        }
    }
    fn valid(&self, v: &V) -> bool {
        Checks::shape_matches(v, &self.zero) && Checks::finite(v)
    }
    fn valid_vector(&self, v: &FourMomentum<V>) -> bool {
        v.components().iter().all(|c| self.valid(c))
    }
    fn topology_valid(
        &self,
        p: &[usize],
        cp: &[usize],
        shifts: &[FourMomentum<V>],
        masses: &[V],
    ) -> bool {
        p.len() == 4
            && cp.len() == self.n - 4
            && shifts.len() == self.n
            && masses.len() == self.n
            && p.iter().chain(cp).all(|&i| i < self.n)
            && (0..self.n).all(|i| p.iter().chain(cp).filter(|&&j| i == j).count() == 1)
    }
    fn sample_mu(&mut self, scales: &[V], output: &mut [V]) -> Result<()> {
        BatchShape {
            rows: scales.len(),
            columns: self.options.sample_count(self.n, self.rank),
        }
        .validate("box mu samples", output.len())?;
        if !scales.iter().all(|s| self.valid(s)) {
            return Err(Error::Expansion(
                "box sampling scale has invalid carrier".into(),
            ));
        }
        self.samples.evaluate(scales, output, scales.len())
    }
    fn divide_samples(
        &mut self,
        requests: &[BoxDivisionRequest<'_, V>],
        numerator: &[V],
        output: &mut [V],
        stable: &mut [bool],
    ) -> Result<()> {
        let rows = requests.len();
        let uncut = self.n - 4;
        for (name, len) in [
            ("box numerator values", numerator.len()),
            ("box divided values", output.len()),
            ("box sample statuses", stable.len()),
        ] {
            BatchShape { rows, columns: 1 }.validate(name, len)?;
        }
        for r in requests {
            if !self.topology_valid(r.partition, r.complement, r.shifts, r.masses_squared)
                || r.pentagon_coefficients.is_some_and(|p| p.len() != uncut)
                || (uncut > 0 && r.pentagon_coefficients.is_none() && !r.mu_squared.is_fully_zero())
            {
                return Err(Error::Expansion(
                    "box sampled denominator topology or parent layout is invalid".into(),
                ));
            }
        }
        self.input.clear();
        self.eligible.clear();
        for (row, r) in requests.iter().enumerate() {
            stable[row] = false;
            if !self.valid(&numerator[row])
                || !self.valid(r.mu_squared)
                || !self.valid_vector(r.momentum)
                || !r.shifts.iter().all(|v| self.valid_vector(v))
                || !r.masses_squared.iter().all(|v| self.valid(v))
                || r.pentagon_coefficients
                    .is_some_and(|p| !p.iter().all(|v| self.valid(v)))
            {
                continue;
            }
            self.eligible.push(row);
            append_vector(&mut self.input, r.momentum);
            self.input.push(r.mu_squared.clone());
            for &line in r.complement {
                append_vector(&mut self.input, &r.shifts[line]);
                self.input.push(r.masses_squared[line].clone());
            }
        }
        self.probe
            .resize(self.eligible.len() * uncut, self.zero.clone());
        self.denominators
            .evaluate(&self.input, &mut self.probe, self.eligible.len())?;
        self.packed.clear();
        self.active.clear();
        for (index, &row) in self.eligible.iter().enumerate() {
            let r = &requests[row];
            let den = &self.probe[index * uncut..(index + 1) * uncut];
            if !den.iter().zip(r.complement).all(|(d, &line)| {
                self.valid(d) && {
                    let scale = denominator_scale::<V, Checks>(
                        r.momentum,
                        &r.shifts[line],
                        &r.masses_squared[line],
                        r.mu_squared,
                    );
                    !self.negligible.value::<V, Checks>(d, &scale)
                }
            }) {
                continue;
            }
            self.active.push(row);
            self.packed.push(numerator[row].clone());
            self.packed.push(r.mu_squared.clone());
            self.packed.extend_from_slice(den);
            if let Some(p) = r.pentagon_coefficients {
                self.packed.extend_from_slice(p);
            } else {
                self.packed.extend((0..uncut).map(|_| self.zero.clone()));
            }
        }
        self.values.resize(self.active.len(), self.zero.clone());
        self.division
            .evaluate(&self.packed, &mut self.values, self.active.len())?;
        for (index, &row) in self.active.iter().enumerate() {
            if self.valid(&self.values[index]) {
                output[row] = self.values[index].clone();
                stable[row] = true;
            }
        }
        Ok(())
    }
    fn divide_mu(
        &mut self,
        requests: &[BoxMuDivisionRequest<'_, V>],
        numerator: &[V],
        output: &mut [[V; 3]],
        stable: &mut [bool],
    ) -> Result<()> {
        let rows = requests.len();
        let uncut = self.n - 4;
        let width = self.mu_width();
        BatchShape {
            rows,
            columns: width,
        }
        .validate("box mu numerator", numerator.len())?;
        for (name, len) in [
            ("box mu coefficients", output.len()),
            ("box mu statuses", stable.len()),
        ] {
            BatchShape { rows, columns: 1 }.validate(name, len)?;
        }
        if self.expansion.is_none() {
            return Err(Error::Expansion(
                "box plan does not request mu expansion".into(),
            ));
        }
        for r in requests {
            if !self.topology_valid(r.partition, r.complement, r.shifts, r.masses_squared) {
                return Err(Error::Expansion(
                    "box mu denominator topology is invalid".into(),
                ));
            }
        }
        self.eligible.clear();
        self.input.clear();
        for (row, r) in requests.iter().enumerate() {
            stable[row] = false;
            if !numerator[row * width..(row + 1) * width]
                .iter()
                .all(|v| self.valid(v))
                || !r.shifts.iter().all(|v| self.valid_vector(v))
                || !r.masses_squared.iter().all(|v| self.valid(v))
                || !self.valid_vector(&r.geometry.a)
                || !self.valid_vector(&r.geometry.v_orth)
                || !self.valid(&r.geometry.v_orth_squared)
                || !self.valid(&r.geometry.beta)
            {
                continue;
            }
            self.eligible.push(row);
            append_vector(&mut self.input, &r.geometry.v_orth);
            append_vector(&mut self.input, &r.geometry.a);
            append_vector(&mut self.input, &r.shifts[r.partition[0]]);
            self.input.push(r.masses_squared[r.partition[0]].clone());
            for &line in r.complement {
                append_vector(&mut self.input, &r.shifts[line]);
                self.input.push(r.masses_squared[line].clone());
            }
        }
        let probe_width = 1 + uncut * 2;
        self.probe
            .resize(self.eligible.len() * probe_width, self.zero.clone());
        self.mu_denominators
            .evaluate(&self.input, &mut self.probe, self.eligible.len())?;
        self.active.clear();
        self.packed.clear();
        for (index, &row) in self.eligible.iter().enumerate() {
            let probe = &self.probe[index * probe_width..(index + 1) * probe_width];
            // Ninja's mp2(Vort) and den[0] = 2 Vort.(V_j - V_0) checks.
            let r = &requests[row];
            let v = &r.geometry.v_orth;
            let origin = &r.shifts[r.partition[0]];
            if !probe.iter().all(|v| self.valid(v))
                || self
                    .negligible
                    .value::<V, Checks>(&probe[0], &square_scale::<V, Checks>(v))
                || r.complement.iter().enumerate().any(|(i, &line)| {
                    let scale = difference_dot_scale::<V, Checks>(v, &r.shifts[line], origin);
                    self.negligible
                        .value::<V, Checks>(&probe[1 + i * 2], &scale)
                })
            {
                continue;
            }
            self.active.push(row);
            self.packed
                .extend_from_slice(&numerator[row * width..(row + 1) * width]);
            self.packed.extend_from_slice(probe);
            if self.rank_n2 {
                self.packed.push(requests[row].geometry.beta.clone());
            }
        }
        let outputs = if self.rank_n2 { 3 } else { 2 };
        self.values
            .resize(self.active.len() * outputs, self.zero.clone());
        self.mu_division
            .evaluate(&self.packed, &mut self.values, self.active.len())?;
        for (index, &row) in self.active.iter().enumerate() {
            let values = &self.values[index * outputs..(index + 1) * outputs];
            if values.iter().all(|v| self.valid(v)) {
                output[row] = [
                    values[0].clone(),
                    values[1].clone(),
                    values.get(2).unwrap_or(&self.zero).clone(),
                ];
                stable[row] = true;
            }
        }
        Ok(())
    }
    fn project(
        &mut self,
        requests: &[BoxProjectionRequest<'_, V>],
        output: &mut [V],
        stable: &mut [bool],
    ) -> Result<()> {
        let rows = requests.len();
        let levels = self.options.sample_count(self.n, self.rank);
        let width = self.width();
        let full = self.options.coefficients == BoxCoefficientMode::Full;
        BatchShape {
            rows,
            columns: width,
        }
        .validate("box projected coefficients", output.len())?;
        BatchShape { rows, columns: 1 }.validate("box projection statuses", stable.len())?;
        for r in requests {
            if r.momenta.len() != levels || r.quotients.len() != levels {
                return Err(Error::Expansion(
                    "box projection sample count differs from plan".into(),
                ));
            }
        }
        self.input.clear();
        self.eligible.clear();
        for (row, r) in requests.iter().enumerate() {
            stable[row] = false;
            if !self.valid(r.sampling_scale)
                || (levels > 1 && !Checks::nonzero(r.sampling_scale))
                || !r.mu_coefficients.iter().all(|v| self.valid(v))
                || !self.valid_vector(r.shift)
                || !self.valid_vector(&r.geometry.a)
                || !self.valid_vector(&r.geometry.v_orth)
                || !self.valid(&r.geometry.v_orth_squared)
                || !self.valid(&r.geometry.beta)
                || !r.momenta.iter().flatten().all(|v| self.valid_vector(v))
                || !r.quotients.iter().flatten().all(|v| self.valid(v))
            {
                continue;
            }
            self.eligible.push(row);
            append_vector(&mut self.input, r.shift);
            append_vector(&mut self.input, &r.geometry.v_orth);
            for q in r.momenta.iter().flatten() {
                append_vector(&mut self.input, q);
            }
        }
        self.probe
            .resize(self.eligible.len() * levels * 3, self.zero.clone());
        self.projections
            .evaluate(&self.input, &mut self.probe, self.eligible.len())?;
        self.active.clear();
        self.packed.clear();
        for (index, &row) in self.eligible.iter().enumerate() {
            let r = &requests[row];
            let probe = &self.probe[index * levels * 3..(index + 1) * levels * 3];
            if !probe.iter().all(|v| self.valid(v))
                || (full
                    && (0..levels).any(|level| {
                        // (q+ - q-).v_orth, the difference of two projections.
                        let scale = Checks::magnitude(&probe[level * 3])
                            + Checks::magnitude(&probe[level * 3 + 1]);
                        self.negligible
                            .value::<V, Checks>(&probe[level * 3 + 2], &scale)
                    }))
            {
                continue;
            }
            self.active.push(row);
            self.packed.push(r.sampling_scale.clone());
            self.packed
                .extend_from_slice(&r.mu_coefficients[..if self.rank_n2 { 3 } else { 2 }]);
            for level in 0..levels {
                self.packed.extend_from_slice(&r.quotients[level]);
                self.packed
                    .extend_from_slice(&probe[level * 3..level * 3 + 2]);
            }
        }
        self.values
            .resize(self.active.len() * width, self.zero.clone());
        self.interpolation
            .evaluate(&self.packed, &mut self.values, self.active.len())?;
        for (index, &row) in self.active.iter().enumerate() {
            let values = &self.values[index * width..(index + 1) * width];
            if values.iter().all(|v| self.valid(v)) {
                output[row * width..(row + 1) * width].clone_from_slice(values);
                stable[row] = true;
            }
        }
        Ok(())
    }
}

pub struct SymbolicaBoxKernels<T> {
    engine: Engine<Complex<T>, CompiledKernel<T>, OrdinaryChecks, T>,
}
impl<T: Real + SingleFloat> SymbolicaBoxKernels<T>
where
    Complex<T>: Real + EvaluationDomain,
{
    pub fn new(
        propagators: usize,
        rank: usize,
        options: BoxReductionOptions,
        prototype: T,
    ) -> Result<Self> {
        let zero = Complex::new(prototype.zero(), prototype.zero());
        Ok(Self {
            engine: Engine::new(propagators, rank, options, &zero, |exact| {
                exact.compile(&prototype)
            })?,
        })
    }
}
pub struct SymbolicaDualBoxKernels<T, D> {
    engine: Engine<D, CompiledDualKernel<T, D>, DualChecks<T>, T>,
}
impl<T: Real + SingleFloat, D: Real + DualNumber<Complex<T>>> SymbolicaDualBoxKernels<T, D>
where
    Complex<T>: Real + EvaluationDomain,
{
    pub fn new(
        propagators: usize,
        rank: usize,
        options: BoxReductionOptions,
        prototype: T,
        dual: &D,
    ) -> Result<Self> {
        Ok(Self {
            engine: Engine::new(propagators, rank, options, dual, |exact| {
                exact.compile_dual(&prototype, dual)
            })?,
        })
    }
}
macro_rules! implementation{
    ($name:ident<$($g:ident),+>,$value:ty,$($bounds:tt)*)=>{
        impl<$($g),+> BoxReductionKernels<$value> for $name<$($g),+> where $($bounds)*{
            fn propagators(&self)->usize{self.engine.n}
            fn rank(&self)->usize{self.engine.rank}
            fn options(&self)->BoxReductionOptions{self.engine.options}
            fn numerator_mu_expansion(&self)->Option<&ExpansionKind>{self.engine.expansion.as_ref()}
            fn coefficient_count(&self)->usize{self.engine.width()}
            fn sampling_mu_batch(&mut self,scales:&[$value],output:&mut[$value])->Result<()>{self.engine.sample_mu(scales,output)}
            fn divide_samples_batch(&mut self,requests:&[BoxDivisionRequest<'_,$value>],numerator:&[$value],output:&mut[$value],stable:&mut[bool])->Result<()>{self.engine.divide_samples(requests,numerator,output,stable)}
            fn divide_mu_batch(&mut self,requests:&[BoxMuDivisionRequest<'_,$value>],numerator:&[$value],output:&mut[[$value;3]],stable:&mut[bool])->Result<()>{self.engine.divide_mu(requests,numerator,output,stable)}
            fn project_batch(&mut self,requests:&[BoxProjectionRequest<'_,$value>],output:&mut[$value],stable:&mut[bool])->Result<()>{self.engine.project(requests,output,stable)}
            fn values_are_finite(&self,values:&[$value])->bool{values.iter().all(|v|self.engine.valid(v))}
        }
    }
}
implementation!(SymbolicaBoxKernels<T>,Complex<T>,T:Real+SingleFloat,Complex<T>:Real+EvaluationDomain);
implementation!(SymbolicaDualBoxKernels<T,D>,D,T:Real+SingleFloat,D:Real+DualNumber<Complex<T>>,Complex<T>:Real+EvaluationDomain);
