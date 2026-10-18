//! Adapters from the generic NinjaRS provider traits to the fast interfaces.
//!
//! [`GenericNumerator`] forwards Ninja's callbacks to any
//! [`crate::Numerator<Complex<f64>>`] by translating Ninja's output layouts
//! into explicit [`ExpansionKind`] power lists (explicit `mu²` dependence is
//! supported). [`GenericMasters`] forwards master families to a
//! [`crate::MasterProvider<Complex<f64>>`]. Both are correctness adapters and
//! allocate request views per call.

use symbolica::domains::float::Complex;

use super::complex::{C64, CMom};
use super::masters::{FastMaster, FastMasters};
use super::numerator::{FastNumerator, t2_len, t3_len};
use crate::{
    Error, ExpansionKind, ExpansionRequest, FourMomentum, Laurent, MasterIntegral, MasterProvider,
    MasterRequest, MasterStatus, Numerator, NumeratorRequest, Result, SampleId,
};

fn momentum(v: &CMom) -> FourMomentum<Complex<f64>> {
    FourMomentum::from_args(
        v[0].to_symbolica(),
        v[1].to_symbolica(),
        v[2].to_symbolica(),
        v[3].to_symbolica(),
    )
}

/// `T3` powers `(t, mu²)` of Ninja's `t3Expansion` layout with leading power
/// `t^rank`: order `i` lists `t^(rank-i) mu^(2j)` for `j = 0..=i/2`.
pub fn t3_powers(rank: usize, mindeg: usize) -> Vec<(i32, u16)> {
    let mut powers = Vec::with_capacity(t3_len(mindeg));
    for i in 0..=mindeg {
        for j in 0..=i / 2 {
            powers.push((rank as i32 - i as i32, j as u16));
        }
    }
    powers
}

/// `T2` powers `(t, x, mu²)` of Ninja's `t2Expansion` layout: order `i`
/// lists `t^(rank-i) x^a mu^(2j)` for `a + 2j <= i`, ordered by `a` then `j`:
/// `[1]`, `[1, x]`, `[1, mu², x, x²]`, `[1, mu², x, x mu², x², x³]` and, at
/// rank n+2, `[1, mu², mu⁴, x, x mu², x², x² mu², x³, x⁴]`.
pub fn t2_powers(rank: usize, mindeg: usize) -> Vec<(i32, u16, u16)> {
    let mut powers = Vec::with_capacity(t2_len(mindeg));
    for i in 0..=mindeg {
        for x in 0..=i {
            for mu in 0..=(i - x) / 2 {
                powers.push((rank as i32 - i as i32, x as u16, mu as u16));
            }
        }
    }
    powers
}

/// Deepest `t3_expansion` layout served by [`GenericNumerator`] (rank n+2
/// triangles).
const T3_MINDEG: usize = 5;
/// Deepest `t2_expansion` layout served by [`GenericNumerator`] (rank n+2
/// bubbles).
const T2_MINDEG: usize = 4;

/// A fast numerator backed by a generic NinjaRS [`Numerator`].
///
/// `rank` is the leading power `t^rank` of every expansion (Ninja's declared
/// numerator rank); [`GenericNumerator::new`] uses the wrapped numerator's
/// rank.
pub struct GenericNumerator<N> {
    inner: N,
    rank: usize,
    t3: Vec<ExpansionKind>,
    t2: Vec<ExpansionKind>,
    mu: [ExpansionKind; 2],
    mu_shifted: ExpansionKind,
    momenta: Vec<FourMomentum<Complex<f64>>>,
    scalars: Vec<Complex<f64>>,
    output: Vec<Complex<f64>>,
}

enum Kind {
    Mu(usize),
    MuShifted,
    T3(usize),
    T2(usize),
}

impl<N: Numerator<Complex<f64>>> GenericNumerator<N> {
    pub fn new(inner: N) -> Self {
        let rank = inner.rank();
        Self::with_rank(inner, rank)
    }

    /// Declare a leading power different from `inner.rank()` (it must not be
    /// smaller than the polynomial's actual degree).
    pub fn with_rank(inner: N, rank: usize) -> Self {
        Self {
            inner,
            rank,
            t3: (0..=T3_MINDEG)
                .map(|m| ExpansionKind::T3 {
                    powers: t3_powers(rank, m),
                })
                .collect(),
            t2: (0..=T2_MINDEG)
                .map(|m| ExpansionKind::T2 {
                    powers: t2_powers(rank, m),
                })
                .collect(),
            mu: [
                ExpansionKind::Mu {
                    powers: vec![rank as i32],
                },
                ExpansionKind::Mu {
                    powers: vec![rank as i32, rank as i32 - 1],
                },
            ],
            mu_shifted: ExpansionKind::MuShifted {
                powers: vec![rank as i32, rank as i32 - 1, rank as i32 - 2],
            },
            momenta: Vec::new(),
            scalars: Vec::new(),
            output: Vec::new(),
        }
    }

    pub fn inner(&self) -> &N {
        &self.inner
    }
    pub fn inner_mut(&mut self) -> &mut N {
        &mut self.inner
    }
    pub fn into_inner(self) -> N {
        self.inner
    }

    fn expand(
        &mut self,
        kind: Kind,
        cut: &[usize],
        lanes: &[usize],
        vectors: usize,
        scalars: usize,
        out: &mut [C64],
    ) -> Result<()> {
        let kind = match kind {
            Kind::Mu(m) => &self.mu[m],
            Kind::MuShifted => &self.mu_shifted,
            Kind::T3(m) => &self.t3[m],
            Kind::T2(m) => &self.t2[m],
        };
        if out.len() != lanes.len() * kind.len() {
            return Err(Error::Shape {
                name: "generic numerator output",
                expected: lanes.len() * kind.len(),
                actual: out.len(),
            });
        }
        let requests: Vec<_> = lanes
            .iter()
            .enumerate()
            .map(|(row, &lane)| ExpansionRequest {
                id: SampleId { lane, sample: row },
                partition: cut,
                vectors: &self.momenta[row * vectors..(row + 1) * vectors],
                parameters: &self.scalars[row * scalars..(row + 1) * scalars],
            })
            .collect();
        self.output.clear();
        self.output.resize(out.len(), Complex::new(0., 0.));
        self.inner
            .expansion_batch(kind, &requests, &mut self.output)?;
        for (o, v) in out.iter_mut().zip(&self.output) {
            *o = C64::from_symbolica(*v);
        }
        Ok(())
    }
}

impl<N: Numerator<Complex<f64>>> FastNumerator for GenericNumerator<N> {
    fn rank(&self) -> usize {
        self.rank
    }

    fn evaluate(
        &mut self,
        cut: &[usize],
        lanes: &[usize],
        q: &[CMom],
        mu_squared: &[C64],
        out: &mut [C64],
    ) -> Result<()> {
        if out.len() != lanes.len() {
            return Err(Error::Shape {
                name: "generic numerator output",
                expected: lanes.len(),
                actual: out.len(),
            });
        }
        self.momenta.clear();
        self.momenta.extend(q.iter().map(momentum));
        self.scalars.clear();
        self.scalars
            .extend(mu_squared.iter().map(|z| z.to_symbolica()));
        let requests: Vec<_> = lanes
            .iter()
            .enumerate()
            .map(|(row, &lane)| NumeratorRequest {
                id: SampleId { lane, sample: row },
                q: &self.momenta[row],
                mu_squared: &self.scalars[row],
                partition: cut,
            })
            .collect();
        self.output.clear();
        self.output.resize(out.len(), Complex::new(0., 0.));
        self.inner.evaluate_batch(&requests, &mut self.output)?;
        for (o, v) in out.iter_mut().zip(&self.output) {
            *o = C64::from_symbolica(*v);
        }
        Ok(())
    }

    fn mu_expansion(
        &mut self,
        cut: &[usize],
        lanes: &[usize],
        v0: &[CMom],
        v1: Option<&[CMom]>,
        out: &mut [C64],
    ) -> Result<()> {
        self.momenta.clear();
        self.scalars.clear();
        for row in 0..lanes.len() {
            self.momenta.push(momentum(&v0[row]));
            if let Some(v1) = v1 {
                self.momenta.push(momentum(&v1[row]));
            }
        }
        let vectors = if v1.is_some() { 2 } else { 1 };
        self.expand(Kind::Mu(vectors - 1), cut, lanes, vectors, 0, out)
    }

    fn mu_expansion_x2(
        &mut self,
        cut: &[usize],
        lanes: &[usize],
        v0: &[CMom],
        v1: &[CMom],
        beta: &[C64],
        out: &mut [C64],
    ) -> Result<()> {
        self.momenta.clear();
        self.scalars.clear();
        for row in 0..lanes.len() {
            self.momenta
                .extend([momentum(&v0[row]), momentum(&v1[row])]);
            self.scalars.push(beta[row].to_symbolica());
        }
        self.expand(Kind::MuShifted, cut, lanes, 2, 1, out)
    }

    fn t3_expansion(
        &mut self,
        cut: &[usize],
        lanes: &[usize],
        a: &[CMom],
        e3: &[CMom],
        e4: &[CMom],
        param: &[C64],
        mindeg: usize,
        out: &mut [C64],
    ) -> Result<()> {
        if mindeg > T3_MINDEG {
            return Err(Error::Expansion(format!(
                "t3 mindeg {mindeg} > {T3_MINDEG}"
            )));
        }
        self.momenta.clear();
        self.scalars.clear();
        for row in 0..lanes.len() {
            self.momenta
                .extend([momentum(&a[row]), momentum(&e3[row]), momentum(&e4[row])]);
            self.scalars.push(param[row].to_symbolica());
        }
        self.expand(Kind::T3(mindeg), cut, lanes, 3, 1, out)
    }

    fn t2_expansion(
        &mut self,
        cut: &[usize],
        lanes: &[usize],
        a0: &[CMom],
        a1: &[CMom],
        e3: &[CMom],
        e4: &[CMom],
        param: &[[C64; 3]],
        mindeg: usize,
        out: &mut [C64],
    ) -> Result<()> {
        if mindeg > T2_MINDEG {
            return Err(Error::Expansion(format!(
                "t2 mindeg {mindeg} > {T2_MINDEG}"
            )));
        }
        self.momenta.clear();
        self.scalars.clear();
        for row in 0..lanes.len() {
            self.momenta.extend([
                momentum(&a0[row]),
                momentum(&a1[row]),
                momentum(&e3[row]),
                momentum(&e4[row]),
            ]);
            self.scalars
                .extend(param[row].iter().map(|z| z.to_symbolica()));
        }
        self.expand(Kind::T2(mindeg), cut, lanes, 4, 3, out)
    }
}

/// Fast master families served by a generic NinjaRS [`MasterProvider`].
///
/// Bubble families issue separate `B0`, `B1`, `B11` (and `B111`, `B1111`)
/// requests.
/// NinjaRS master requests carry the squared renormalization scale, which is
/// exactly the last fast argument, so values are forwarded unchanged.
pub struct GenericMasters<M> {
    inner: M,
    invariants: Vec<Complex<f64>>,
    masses: Vec<Complex<f64>>,
    scales: Vec<Complex<f64>>,
    output: Vec<Laurent<Complex<f64>>>,
    statuses: Vec<MasterStatus>,
}

impl<M: MasterProvider<Complex<f64>>> GenericMasters<M> {
    pub fn new(inner: M) -> Self {
        Self {
            inner,
            invariants: Vec::new(),
            masses: Vec::new(),
            scales: Vec::new(),
            output: Vec::new(),
            statuses: Vec::new(),
        }
    }
    pub fn inner(&self) -> &M {
        &self.inner
    }
    pub fn inner_mut(&mut self) -> &mut M {
        &mut self.inner
    }
    pub fn into_inner(self) -> M {
        self.inner
    }
}

impl<M: MasterProvider<Complex<f64>>> FastMasters for GenericMasters<M> {
    fn evaluate(
        &mut self,
        family: FastMaster,
        args: &[C64],
        out: &mut [[C64; 3]],
        ok: &mut [bool],
    ) -> Result<()> {
        let arity = family.arity();
        let rows = ok.len();
        let width = family.outputs();
        if args.len() != rows * arity || out.len() != rows * width {
            return Err(Error::Shape {
                name: "generic master buffers",
                expected: rows * arity,
                actual: args.len(),
            });
        }
        let (invariants, masses) = match family {
            FastMaster::A0 => (0, 1),
            FastMaster::BubbleRank2 | FastMaster::BubbleRank3 | FastMaster::BubbleRank4 => (1, 2),
            FastMaster::C0 => (3, 3),
            FastMaster::D0 => (6, 4),
        };
        self.invariants.clear();
        self.masses.clear();
        self.scales.clear();
        for row in args.chunks_exact(arity) {
            self.invariants
                .extend(row[..invariants].iter().map(|z| z.to_symbolica()));
            self.masses.extend(
                row[invariants..invariants + masses]
                    .iter()
                    .map(|z| z.to_symbolica()),
            );
            self.scales.push(row[arity - 1].to_symbolica());
        }
        let integrals: &[MasterIntegral] = match family {
            FastMaster::A0 => &[MasterIntegral::A0],
            FastMaster::BubbleRank2 => {
                &[MasterIntegral::B0, MasterIntegral::B1, MasterIntegral::B11]
            }
            FastMaster::BubbleRank3 => &[
                MasterIntegral::B0,
                MasterIntegral::B1,
                MasterIntegral::B11,
                MasterIntegral::B111,
            ],
            FastMaster::BubbleRank4 => &[
                MasterIntegral::B0,
                MasterIntegral::B1,
                MasterIntegral::B11,
                MasterIntegral::B111,
                MasterIntegral::B1111,
            ],
            FastMaster::C0 => &[MasterIntegral::C0],
            FastMaster::D0 => &[MasterIntegral::D0],
        };
        ok.fill(true);
        let zero = Complex::new(0., 0.);
        for (slot, &integral) in integrals.iter().enumerate() {
            self.output.clear();
            self.output.resize(rows, [zero; 3]);
            self.statuses.clear();
            self.statuses.resize(rows, MasterStatus::Success);
            self.inner.evaluate_batch(
                MasterRequest {
                    integral,
                    invariants: &self.invariants,
                    masses_squared: &self.masses,
                    scales_squared: &self.scales,
                },
                &mut self.output,
                &mut self.statuses,
            )?;
            for row in 0..rows {
                ok[row] &= self.statuses[row] == MasterStatus::Success;
                out[row * width + slot] = self.output[row].map(C64::from_symbolica);
            }
        }
        Ok(())
    }
}
