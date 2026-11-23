//! Double-double re-reduction of the lanes whose binary64 reduction cancels
//! (`docs/triangle-gram-conditioning.md`): the workspace of the generated
//! double-double stages (`dd_stages.rs`), their input with consistent
//! light-like legs, the tensor numerator in double-double, and the
//! contraction with the binary64 masters of the binary64 pass, the bubble
//! tensor coefficients through their Passarino-Veltman reduction to `B0` and
//! `A0`.

use super::complex::C64;
use super::dd::{Cd, Dd, RMomD};
use super::dd_stages::{
    Basis, BubbleData, Calls, DdNumerator, TadpoleData, TadpoleRow, TriangleData, TriangleRow,
    Uncut, BubbleRow, horner_jet, t2_output, t3_output, tadpole_candidates,
};
use super::engine::{StageOptions, Topology};
use super::numerator::{t2_len_le3, t3_len};
use super::tensor_layout::ncoeff_homogeneous;
use super::{FastStatus, InstabilityTest};
use crate::{Error, Result};

/// The double-double stages' input (lanes `0..lanes`).
pub(super) struct FastInput<'a> {
    pub(super) lanes: usize,
    pub(super) shifts: &'a [RMomD],
    pub(super) masses_squared: &'a [Cd],
}

/// The binary64 pass ran the stability checks on these lanes: the
/// double-double stages only guard against non-finite values.
#[derive(Clone, Copy)]
pub(super) struct Stability;

impl Stability {
    pub(super) fn new(_: &StageOptions) -> Self {
        Self
    }
    pub(super) fn absolute(self, x: Dd) -> bool {
        x.is_finite() && x.hi != 0.
    }
    pub(super) fn relative(self, x: Dd, _: Dd) -> bool {
        self.absolute(x)
    }
    pub(super) fn relative_within(self, x: Dd, _: Dd, _: impl FnOnce() -> Dd) -> bool {
        self.absolute(x)
    }
}

pub(super) fn fail(alive: &mut [bool], status: &mut [FastStatus], lane: usize, _: InstabilityTest) {
    alive[lane] = false;
    status[lane] = FastStatus::Unstable;
}

pub(super) fn euclidean_or<A, B, C>(test: InstabilityTest, _: A, _: B, _: C) -> InstabilityTest {
    test
}
pub(super) fn difference_scale<A, B, C>(_: A, _: B, _: C) -> Dd {
    Dd::ZERO
}
pub(super) fn gram_scale<A, B>(_: A, _: B) -> Dd {
    Dd::ZERO
}
pub(super) fn taxicab_sum<A>(_: A) -> Dd {
    Dd::ZERO
}

/// Workspace of the double-double stages: the binary64 engine's fields of
/// the same names, in double-double.
pub(super) struct DdStages {
    pub(super) topo: Topology,
    pub(super) options: StageOptions,
    pub(super) tadpole_bases: [Basis; 3],
    pub(super) lanes: usize,
    pub(super) alive: Vec<bool>,
    pub(super) lane_status: Vec<FastStatus>,
    pub(super) active: Vec<usize>,
    pub(super) smat: Vec<Dd>,
    pub(super) vsq: Vec<Dd>,
    pub(super) bounds: Vec<(Dd, Dd)>,
    pub(super) tri: Vec<TriangleData>,
    pub(super) bub: Vec<BubbleData>,
    pub(super) tad: Vec<TadpoleData>,
    pub(super) calls: Calls,
    pub(super) tri_rows: Vec<TriangleRow>,
    pub(super) bub_rows: Vec<BubbleRow>,
    pub(super) tad_rows: Vec<TadpoleRow>,
    pub(super) second: Vec<usize>,
    pub(super) uncut: Vec<Uncut>,
    pub(super) pairs: Vec<[Cd; 4]>,
    /// Per (cut, lane): reduce this cut of this lane in double-double. The
    /// others keep the data the caller stored after [`Self::prepare`] (the
    /// binary64 reduction's, promoted).
    pub(super) sel_tri: Vec<bool>,
    pub(super) sel_bub: Vec<bool>,
    pub(super) sel_tad: Vec<bool>,
}

impl DdStages {
    pub(super) fn new(n: usize, rank: usize, options: StageOptions) -> Self {
        Self {
            topo: Topology::new(n, rank),
            tadpole_bases: tadpole_candidates(options.infrared_eps),
            options,
            lanes: 0,
            alive: Vec::new(),
            lane_status: Vec::new(),
            active: Vec::new(),
            smat: Vec::new(),
            vsq: Vec::new(),
            bounds: Vec::new(),
            tri: Vec::new(),
            bub: Vec::new(),
            tad: Vec::new(),
            calls: Calls::default(),
            tri_rows: Vec::new(),
            bub_rows: Vec::new(),
            tad_rows: Vec::new(),
            second: Vec::new(),
            uncut: vec![Uncut::default(); n],
            pairs: Vec::new(),
            sel_tri: Vec::new(),
            sel_bub: Vec::new(),
            sel_tad: Vec::new(),
        }
    }

    pub(super) fn any_triangle(&self) -> bool {
        self.topo.n >= 3 && self.options.cut_stop <= 3 && self.topo.rmn() + 3 >= 0
    }
    pub(super) fn any_bubble(&self) -> bool {
        self.topo.n >= 2 && self.options.cut_stop <= 2 && self.topo.rmn() + 2 >= 0
    }
    pub(super) fn any_tadpole(&self) -> bool {
        self.options.cut_stop <= 1 && self.topo.rmn() + 1 >= 0
    }
    pub(super) fn refresh_active(&mut self) {
        let alive = &self.alive;
        self.active.retain(|&lane| alive[lane]);
    }

    /// Reset for `input.lanes` lanes with the binary64 invariant matrix
    /// `smat` (lane-major, `n x n` per lane).
    pub(super) fn prepare(&mut self, input: &FastInput, smat: &[f64]) {
        let n = self.topo.n;
        let lanes = input.lanes;
        self.lanes = lanes;
        self.alive.clear();
        self.alive.resize(lanes, true);
        self.lane_status.clear();
        self.lane_status.resize(lanes, FastStatus::Success);
        self.active.clear();
        self.active.extend(0..lanes);
        self.smat.clear();
        self.smat.extend(smat.iter().map(|&x| Dd::from(x)));
        self.vsq.clear();
        self.bounds.clear();
        for lane in 0..lanes {
            let v = &input.shifts[lane * n..(lane + 1) * n];
            for i in 0..n {
                self.vsq.push(super::dd::mp_rr(&v[i], &v[i]));
            }
            let vmax = v.iter().flatten().fold(Dd::ZERO, |a, x| a.max(x.abs()));
            let mmax = input.masses_squared[lane * n..(lane + 1) * n]
                .iter()
                .fold(Dd::ZERO, |a, m| a.max(m.taxicab()));
            self.bounds.push((vmax, mmax));
        }
        self.tri.clear();
        self.tri
            .resize(self.topo.triangles.len() * lanes, TriangleData::default());
        self.bub.clear();
        self.bub
            .resize(self.topo.bubbles.len() * lanes, BubbleData::default());
        self.tad.clear();
        self.tad.resize(n * lanes, TadpoleData::default());
        for (sel, len) in [
            (&mut self.sel_tri, self.topo.triangles.len() * lanes),
            (&mut self.sel_bub, self.topo.bubbles.len() * lanes),
            (&mut self.sel_tad, n * lanes),
        ] {
            sel.clear();
            sel.resize(len, true);
        }
    }
}

/// The light-like projection of one lane's shifts: every pair `(i, j)` whose
/// invariant `(V_i - V_j)²` is given as exactly zero (an on-shell massless
/// leg, or a sum of collinear ones) and whose binary64 shifts differ is made
/// exactly light-like in double-double by the minimal-norm Gauss-Newton step
/// `x <- x - Jᵀ (J Jᵀ)⁻¹ g(x)` on the shift components, twice (the binary64
/// residuals are of order `ε |V|²`, so the second step reaches the carrier's
/// rounding). Without it the double-double stages would reduce a lane whose
/// light-like legs carry the binary64 rounding of their components as a
/// mass, which the degenerate triangles amplify as much as rounding
/// (`docs/triangle-gram-conditioning.md`, section 2). Returns `false` when the
/// constraints are degenerate (`J Jᵀ` singular), the shifts then unchanged.
pub(super) fn project_light_like(v: &mut [RMomD], smat: &[f64], n: usize) -> bool {
    let mut pairs = Vec::new();
    for i in 0..n {
        for j in i + 1..n {
            if smat[i * n + j] == 0. && (0..4).any(|mu| v[i][mu] != v[j][mu]) {
                pairs.push((i, j));
            }
        }
    }
    if pairs.is_empty() {
        return true;
    }
    let m = pairs.len();
    let metric = [1., -1., -1., -1.];
    for _ in 0..2 {
        // Rows of J: d g/d V_i = 2 g_{mu mu} (V_i - V_j)^mu, d g/d V_j = -that.
        let d: Vec<[Dd; 4]> = pairs
            .iter()
            .map(|&(i, j)| std::array::from_fn(|mu| v[i][mu] - v[j][mu]))
            .collect();
        let g: Vec<Dd> = d.iter().map(|x| super::dd::mp_rr(x, x)).collect();
        // J Jᵀ: rows a, b overlap where they share a shift.
        let mut a = vec![Dd::ZERO; m * m];
        for r in 0..m {
            for c in 0..m {
                let (ri, rj) = pairs[r];
                let (ci, cj) = pairs[c];
                let sign = |x: usize, y: usize| -> f64 {
                    // (+1 for V_i of a row, -1 for V_j).
                    if x == y { 1. } else { 0. }
                };
                let w = sign(ri, ci) - sign(ri, cj) - sign(rj, ci) + sign(rj, cj);
                if w != 0. {
                    let mut dot = Dd::ZERO;
                    for mu in 0..4 {
                        dot += d[r][mu] * d[c][mu];
                    }
                    a[r * m + c] = dot * (4. * w);
                }
            }
        }
        // Solve (J Jᵀ) y = g by Gaussian elimination with partial pivoting.
        let mut y = g.clone();
        for col in 0..m {
            let piv = (col..m)
                .max_by(|&p, &q| a[p * m + col].abs().partial_cmp(&a[q * m + col].abs()).unwrap())
                .unwrap();
            if a[piv * m + col].hi == 0. || !a[piv * m + col].is_finite() {
                return false;
            }
            if piv != col {
                for k in 0..m {
                    a.swap(col * m + k, piv * m + k);
                }
                y.swap(col, piv);
            }
            let inv = Dd::ONE / a[col * m + col];
            for r in col + 1..m {
                let f = a[r * m + col] * inv;
                if f.hi == 0. {
                    continue;
                }
                for k in col..m {
                    let t = a[col * m + k];
                    a[r * m + k] -= f * t;
                }
                let t = y[col];
                y[r] -= f * t;
            }
        }
        for col in (0..m).rev() {
            let mut s = y[col];
            for k in col + 1..m {
                s -= a[col * m + k] * y[k];
            }
            y[col] = s / a[col * m + col];
        }
        // x -= Jᵀ y.
        for (r, &(i, j)) in pairs.iter().enumerate() {
            for mu in 0..4 {
                let step = d[r][mu] * metric[mu] * 2. * y[r];
                v[i][mu] -= step;
                v[j][mu] += step;
            }
        }
    }
    true
}

/// The tensor numerator of one evaluation's selected lanes in double-double
/// (coefficients promoted exactly from binary64).
pub(super) struct DdTensor {
    pub(super) rank: usize,
    pub(super) ncoeff: usize,
    pub(super) coefficients: Vec<Cd>,
    pub(super) work: Vec<Cd>,
}

impl DdTensor {
    pub(super) fn new(rank: usize) -> Self {
        Self {
            rank,
            ncoeff: super::tensor_layout::ncoeff(rank),
            coefficients: Vec::new(),
            work: Vec::new(),
        }
    }

    pub(super) fn push_lane(&mut self, c: &[C64]) {
        self.coefficients.extend(c.iter().map(|&z| Cd::from_c64(z)));
        let level = if self.rank == 0 {
            1
        } else {
            ncoeff_homogeneous(self.rank - 1)
        };
        // The largest jet (T3 at mindeg 4: 9 components; T2 at 3: 13).
        if self.work.len() < 2 * 13 * level {
            self.work.resize(2 * 13 * level, Cd::ZERO);
        }
    }

    fn t3_rows<P: super::dd_stages::Jet<L>, const K: usize, const L: usize>(
        &mut self,
        lanes: &[usize],
        a: &[[Cd; 4]],
        e3: &[[Cd; 4]],
        e4: &[[Cd; 4]],
        param: &[Cd],
        out: &mut [Cd],
    ) {
        for (row, &lane) in lanes.iter().enumerate() {
            let c = &self.coefficients[lane * self.ncoeff..(lane + 1) * self.ncoeff];
            let xv = [e3[row], a[row], e4[row], [Cd::ZERO; 4]];
            let jet = horner_jet::<P, L>(c, self.rank, &xv, &mut self.work);
            t3_output(&jet, K, param[row], &mut out[row * L..(row + 1) * L]);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn t2_rows<P: super::dd_stages::Jet<L>, const K: usize, const L: usize>(
        &mut self,
        lanes: &[usize],
        a0: &[[Cd; 4]],
        a1: &[[Cd; 4]],
        e3: &[[Cd; 4]],
        e4: &[[Cd; 4]],
        param: &[[Cd; 3]],
        out: &mut [Cd],
    ) {
        for (row, &lane) in lanes.iter().enumerate() {
            let c = &self.coefficients[lane * self.ncoeff..(lane + 1) * self.ncoeff];
            let xv = [e3[row], a0[row], a1[row], e4[row]];
            let jet = horner_jet::<P, L>(c, self.rank, &xv, &mut self.work);
            t2_output(&jet, K, &param[row], &mut out[row * L..(row + 1) * L]);
        }
    }
}

impl DdNumerator for DdTensor {
    fn t3_expansion(
        &mut self,
        _cut: &[usize],
        lanes: &[usize],
        a: &[[Cd; 4]],
        e3: &[[Cd; 4]],
        e4: &[[Cd; 4]],
        param: &[Cd],
        mindeg: usize,
        out: &mut [Cd],
    ) -> Result<()> {
        use super::dd_stages::jets as j;
        if out.len() != lanes.len() * t3_len(mindeg) {
            return Err(Error::Expansion("double-double t3 rows".into()));
        }
        match mindeg {
            0 => self.t3_rows::<j::T3K0, 0, 1>(lanes, a, e3, e4, param, out),
            1 => self.t3_rows::<j::T3K1, 1, 2>(lanes, a, e3, e4, param, out),
            2 => self.t3_rows::<j::T3K2, 2, 4>(lanes, a, e3, e4, param, out),
            3 => self.t3_rows::<j::T3K3, 3, 6>(lanes, a, e3, e4, param, out),
            4 => self.t3_rows::<j::T3K4, 4, 9>(lanes, a, e3, e4, param, out),
            _ => return Err(Error::Expansion(format!("t3 mindeg {mindeg} > 4"))),
        }
        Ok(())
    }

    fn t2_expansion(
        &mut self,
        _cut: &[usize],
        lanes: &[usize],
        a0: &[[Cd; 4]],
        a1: &[[Cd; 4]],
        e3: &[[Cd; 4]],
        e4: &[[Cd; 4]],
        param: &[[Cd; 3]],
        mindeg: usize,
        out: &mut [Cd],
    ) -> Result<()> {
        use super::dd_stages::jets as j;
        if out.len() != lanes.len() * t2_len_le3(mindeg) {
            return Err(Error::Expansion("double-double t2 rows".into()));
        }
        match mindeg {
            0 => self.t2_rows::<j::T2K0, 0, 1>(lanes, a0, a1, e3, e4, param, out),
            1 => self.t2_rows::<j::T2K1, 1, 3>(lanes, a0, a1, e3, e4, param, out),
            2 => self.t2_rows::<j::T2K2, 2, 7>(lanes, a0, a1, e3, e4, param, out),
            3 => self.t2_rows::<j::T2K3, 3, 13>(lanes, a0, a1, e3, e4, param, out),
            _ => return Err(Error::Expansion(format!("t2 mindeg {mindeg} > 3"))),
        }
        Ok(())
    }
}

/// A Laurent triple `[finite, 1/eps, 1/eps²]` in double-double.
pub type L3 = [Cd; 3];

fn l3(x: &[C64; 3]) -> L3 {
    [Cd::from_c64(x[0]), Cd::from_c64(x[1]), Cd::from_c64(x[2])]
}

/// `B1`, `B11`, `B111` at `(p², m0², m1²)` from `B0`, `A0(m0)` and `A0(m1)`
/// (`p²` the invariant of the reduced geometry, the masters at the given
/// one)
/// by the Passarino-Veltman reduction of OneLOop's conventions
/// (oneloopmaster `two_point.rs`: `b1_at_nonzero_momentum`, `b00`, `b11`,
/// `b001`, `b111`), every Laurent order, in double-double. Used where the
/// reduction is well conditioned, `|p²| >= |m0²| + |m1²|` (`None` otherwise):
/// then the double-double bubble contraction keeps the exact relations
/// between the tensor coefficients that its large coefficients cancel
/// through, and only `B0` and `A0`'s own binary64 rounding remains.
pub fn bubble_tensors(
    p2: Dd,
    m0: Cd,
    m1: Cd,
    b0: &[C64; 3],
    a0m0: &[C64; 3],
    a0m1: &[C64; 3],
    b0dd: Option<L3>,
) -> Option<[L3; 4]> {
    if !(p2.abs().to_f64() >= (m0.taxicab() + m1.taxicab()).to_f64()) || p2.hi == 0. {
        return None;
    }
    let p2 = Cd::real(p2);
    let (b0, a0, a1) = (b0dd.unwrap_or_else(|| l3(b0)), l3(a0m0), l3(a0m1));
    let kin = m1 - m0 - p2;
    let inv2p = (p2 * 2.).inv();
    // Orders: 0 finite, 1 simple pole, 2 double pole (zero for B_{1..}).
    let b1: L3 = std::array::from_fn(|k| (a0[k] - a1[k] + kin * b0[k]) * inv2p);
    // B00 = N/6 + N_pole/9 (finite), N_pole/6 (pole), N = A0(m1) + 2 m0² B0 - K B1.
    let n00: L3 = std::array::from_fn(|k| a1[k] + m0 * b0[k] * 2. - kin * b1[k]);
    let b00: L3 = [n00[0] / 6. + n00[1] / 9., n00[1] / 6., Cd::ZERO];
    let b11: L3 = std::array::from_fn(|k| (a1[k] + kin * b1[k] - b00[k] * 2.) * inv2p);
    // B001 = N/8 + N_pole/16, N_pole/8, N = 2 m0² B1 - A0(m1) - K B11.
    let n001: L3 = std::array::from_fn(|k| m0 * b1[k] * 2. - a1[k] - kin * b11[k]);
    let b001: L3 = [n001[0] / 8. + n001[1] / 16., n001[1] / 8., Cd::ZERO];
    let b111: L3 = std::array::from_fn(|k| (-a1[k] + kin * b11[k] - b001[k] * 4.) * inv2p);
    Some([b0, b1, b11, b111])
}

/// `ln(-s/mu² - i0)` in double-double (`s` real, nonzero).
fn log_minus(s: Dd, mu2: f64) -> Cd {
    let l = (s.abs() / mu2).ln();
    if s > 0. {
        Cd::new(l, -std::f64::consts::PI)
    } else {
        Cd::real(l)
    }
}

/// The massless bubble `B0(p²; 0, 0)` (`p² != 0`) in OneLOop's normalisation,
/// `[2 - ln(-p²/mu² - i0), 1, 0]`, in double-double.
pub fn massless_b0(p2: Dd, mu2: f64) -> L3 {
    [Cd::real(Dd::from(2.)) - log_minus(p2, mu2), Cd::ONE, Cd::ZERO]
}

/// The massless-line triangle `C0(p1², p2², p3²; 0, 0, 0)` in OneLOop's
/// normalisation, in double-double, for one or two non-zero external
/// invariants (`zero[i]`: `p_i²` is exactly zero); `None` with three (the
/// binary64 master is used) or none (scaleless).
///
/// One mass `s`: `[L²/2, -L, 1] / s`; two masses `s, t`:
/// `[(L_s² - L_t²)/2, -(L_s - L_t), 0] / (s - t)`, `L = ln(-p²/mu² - i0)`
/// (Ellis-Zanderighi, QCDLoop, triangles 2 and 3). In double-double the
/// difference quotient keeps about `32 - log10|s/(s - t)|` digits.
pub fn massless_c0(p: [Dd; 3], zero: [bool; 3], mu2: f64) -> Option<L3> {
    let massive: Vec<Dd> = (0..3).filter(|&i| !zero[i]).map(|i| p[i]).collect();
    match massive.as_slice() {
        [s] => {
            let l = log_minus(*s, mu2);
            let inv = Dd::ONE / *s;
            Some([l * l * 0.5 * inv, -l * inv, Cd::real(inv)])
        }
        [s, t] => {
            let (ls, lt) = (log_minus(*s, mu2), log_minus(*t, mu2));
            let inv = Dd::ONE / (*s - *t);
            Some([(ls * ls - lt * lt) * 0.5 * inv, -(ls - lt) * inv, Cd::ZERO])
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `dd_stages.rs` is what `tools/fast/generate_dd_stages.py` makes of the
    /// current binary64 sources (regenerate with
    /// `python3 tools/fast/generate_dd_stages.py`).
    #[test]
    fn dd_stages_are_generated_from_the_binary64_sources() {
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
        let status = std::process::Command::new("python3")
            .arg("tools/fast/generate_dd_stages.py")
            .arg("--check")
            .current_dir(root)
            .status()
            .expect("python3 (the development shell provides it)");
        assert!(
            status.success(),
            "crates/ninjars/src/fast/dd_stages.rs is stale: run python3 tools/fast/generate_dd_stages.py"
        );
    }

    /// A light-like leg of binary64 shifts (a `g g > g g g` lane) becomes
    /// light-like to the double-double rounding.
    #[test]
    fn projection_makes_zero_invariants_light_like() {
        let f = [
            [157.94081412791513, -108.02862942094583, 288.50491082781895, -351.3419417147604],
            [442.0308107887862, 62.50590379162027, 419.0029296663757, 126.17763117438322],
            [-1.3e-13, 1.4e-14, -5e-14, 5e-15],
        ];
        let smat = [
            0., -1.9342957622827255e5, -1.9340112758684525e5,
            -1.9342957622827255e5, 0., 0.,
            -1.9340112758684525e5, 0., 0.,
        ];
        let mut v: Vec<RMomD> = f.iter().map(|x| x.map(Dd::from)).collect();
        let g0 = {
            let d: [Dd; 4] = std::array::from_fn(|mu| v[1][mu] - v[2][mu]);
            super::super::dd::mp_rr(&d, &d)
        };
        assert!(project_light_like(&mut v, &smat, 3));
        let d: [Dd; 4] = std::array::from_fn(|mu| v[1][mu] - v[2][mu]);
        let g = super::super::dd::mp_rr(&d, &d);
        eprintln!("g0 {g0:?} g {g:?} v1 {:?} v2 {:?}", v[1], v[2]);
        assert!(g.abs().hi < 1e-25, "{g:?}");
    }
}
