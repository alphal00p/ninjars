//! The rank <= n+1 stages with [`super::FastOptions::conditioned_cuts`]:
//! `evaluate_up_to_n1`, `boxes`, `bubbles` and `tadpoles` of `engine.rs` with
//! the rank n+2 engine's cut geometry (the box normal, center and solutions
//! of `box_x2.rs`; per lane the bubble reference and the tadpole basis of
//! `bases_x2.rs` that best condition the uncut lines). Reduced by
//! [`super::ConditionedEngine`].
//!
//! Generated from `engine.rs` by `fast/tests_conditioned.rs`; do not edit by
//! hand (`NINJARS_BLESS=1 cargo test -p ninjars --lib conditioned_stages`).

use super::*;
use super::super::bases_x2::bubble_basis_x2;
use super::super::box_x2::box_geometry_x2;

impl FastAmplitude {
    /// [`FastAmplitude::evaluate_up_to_n1`] with the conditioned stages.
    pub(super) fn evaluate_up_to_n1_conditioned<N: FastNumerator + ?Sized, M: FastMasters + ?Sized>(
        &mut self,
        input: &FastInput,
        numerator: &mut N,
        masters: &mut M,
        results: &mut [FastResult],
        statuses: &mut [FastStatus],
        tadpole_bases: &TadpoleBasesX2,
    ) -> Result<()> {
        let n = self.topo.n;
        let lanes = input.lanes;
        for (name, expected, actual) in [
            ("fast shifts", lanes * n, input.shifts.len()),
            ("fast masses", lanes * n, input.masses_squared.len()),
            ("fast scales", lanes, input.scales_squared.len()),
            ("fast results", lanes, results.len()),
            ("fast statuses", lanes, statuses.len()),
        ] {
            if expected != actual {
                return Err(Error::Shape {
                    name,
                    expected,
                    actual,
                });
            }
        }
        if let Some(matrix) = input.s_matrix
            && matrix.len() != lanes * n * n
        {
            return Err(Error::Shape {
                name: "fast S-matrix",
                expected: lanes * n * n,
                actual: matrix.len(),
            });
        }
        if numerator.rank() != self.topo.rank {
            return Err(Error::Shape {
                name: "fast numerator rank",
                expected: self.topo.rank,
                actual: numerator.rank(),
            });
        }
        self.prepare(input);
        if self.any_box() {
            self.boxes_conditioned(input, numerator)?;
        }
        if self.any_triangle() {
            self.triangles(input, numerator)?;
        }
        if self.any_bubble() {
            self.bubbles_conditioned(input, numerator)?;
        }
        if self.any_tadpole() {
            self.tadpoles_conditioned(input, numerator, tadpole_bases)?;
        }
        self.integrate(input, masters)?;
        // Publish only after every provider call succeeded.
        for lane in 0..lanes {
            if self.lane_status[lane] == FastStatus::Success {
                results[lane] = self.staged[lane];
            }
        }
        statuses.copy_from_slice(&self.lane_status[..lanes]);
        Ok(())
    }

    /// [`FastAmplitude::boxes`] with the rank n+2 engine's box geometry.
    fn boxes_conditioned<N: FastNumerator + ?Sized>(&mut self, input: &FastInput, num: &mut N) -> Result<()> {
        let n = self.topo.n;
        let lanes = self.lanes;
        let eps = self.options.infrared_eps;
        let check = Stability::new(&self.options);
        let x1 = self.topo.x1();
        let with_mu = self.topo.rank >= n;
        for b in 0..self.topo.boxes.len() {
            self.refresh_active();
            if self.active.is_empty() {
                return Ok(());
            }
            let p = self.topo.boxes[b];
            let [c1, c2, c3, c4] = p;
            self.box_rows.clear();
            self.calls.clear();
            for ai in 0..self.active.len() {
                let lane = self.active[ai];
                let v = &input.shifts[lane * n..(lane + 1) * n];
                let m = &input.masses_squared[lane * n..(lane + 1) * n];
                let k0 = sub(&v[c2], &v[c1]);
                let k1 = sub(&v[c3], &v[c2]);
                let k3 = sub(&v[c1], &v[c4]);
                if !check.relative(gram_det(&k3, &k0).abs(), gram_scale(&k3, &k0)) {
                    fail(&mut self.alive, &mut self.lane_status, lane);
                    continue;
                }
                let e = Basis::from_pair(&k3, &k0, eps);
                let p1e4 = mp_cr(&e.e4, &k1);
                let p1e3 = mp_cr(&e.e3, &k1);
                let mut vort = [C64::ZERO; 4];
                for mu in 0..4 {
                    vort[mu] = p1e4 * e.e3[mu] - p1e3 * e.e4[mu];
                }
                // mp2(Vort) = -2 (p1.e3) (p1.e4) e3.e4 vanishes exactly when the box
                // is degenerate (a light-like or vanishing normal): test the two
                // factors relative to their own scales.
                if !check.absolute(mp_cc(&vort, &vort).taxicab())
                    || !check.relative(p1e3.taxicab(), dot_scale(&e.e3, &k1))
                    || !check.relative(p1e4.taxicab(), dot_scale(&e.e4, &k1))
                {
                    fail(&mut self.alive, &mut self.lane_status, lane);
                    continue;
                }
                let l = box_solutions(&k3, &k0, &k1, &e, m[c1], m[c2], m[c3], m[c4]);
                // The rank n+2 engine's geometry (box_x2.rs).
                let Some(g) = box_geometry_x2(
                    &[v[c1], v[c2], v[c3], v[c4]],
                    &[m[c1], m[c2], m[c3], m[c4]],
                    &l,
                ) else {
                    fail(&mut self.alive, &mut self.lane_status, lane);
                    continue;
                };
                self.box_rows.push(BoxRow {
                    lane,
                    vort: g.normal,
                    q: g.q,
                    mid: g.mid,
                });
                for qq in g.q {
                    self.calls.lanes.push(lane);
                    self.calls.v0.push(qq);
                    self.calls.scalar.push(C64::ZERO);
                }
            }
            if self.box_rows.is_empty() {
                continue;
            }
            self.calls.prepare_out(1);
            num.evaluate(
                &p,
                &self.calls.lanes,
                &self.calls.v0,
                &self.calls.scalar,
                &mut self.calls.out,
            )?;
            let cp = &self.topo.box_complements[b * (n - 4)..(b + 1) * (n - 4)];
            let mut keep = 0;
            for r in 0..self.box_rows.len() {
                let row = self.box_rows[r];
                let lane = row.lane;
                let v = &input.shifts[lane * n..(lane + 1) * n];
                let m = &input.masses_squared[lane * n..(lane + 1) * n];
                let (vmax, mmax) = self.bounds[lane];
                let mut values = [C64::ZERO; 2];
                let mut stable = true;
                for s in 0..2 {
                    let mut d = C64::ONE;
                    let mut singular = false;
                    let bound = mmax
                        + row.q[s]
                            .iter()
                            .map(|z| (z.taxicab() + vmax).powi(2))
                            .sum::<f64>();
                    for &j in cp {
                        let factor = den(&row.q[s], &v[j], m[j]);
                        singular |= !check.relative_within(factor.taxicab(), bound, || {
                            den_scale(&row.q[s], &v[j], m[j])
                        });
                        d *= factor;
                    }
                    if singular || !check.absolute(d.taxicab()) {
                        stable = false;
                        break;
                    }
                    values[s] = self.calls.out[2 * r + s] / d;
                }
                if !stable {
                    fail(&mut self.alive, &mut self.lane_status, lane);
                    continue;
                }
                self.box_c[b * lanes + lane] = [(values[0] + values[1]) * 0.5, C64::ZERO];
                self.box_rows[keep] = row;
                keep += 1;
            }
            self.box_rows.truncate(keep);
            if !with_mu || self.box_rows.is_empty() {
                continue;
            }
            // Rational term from the mu-expansion.
            self.calls.clear();
            for row in &self.box_rows {
                self.calls.lanes.push(row.lane);
                self.calls.v0.push(row.vort);
                if x1 {
                    self.calls.v1.push(row.mid);
                }
            }
            self.calls.prepare_out(if x1 { 2 } else { 1 });
            num.mu_expansion(
                &p,
                &self.calls.lanes,
                &self.calls.v0,
                if x1 { Some(&self.calls.v1) } else { None },
                &mut self.calls.out,
            )?;
            for r in 0..self.box_rows.len() {
                let row = self.box_rows[r];
                let lane = row.lane;
                let v = &input.shifts[lane * n..(lane + 1) * n];
                let m = &input.masses_squared[lane * n..(lane + 1) * n];
                let vort2 = mp_cc(&row.vort, &row.vort);
                let bound = 4. * self.bounds[lane].0 * taxicab_sum(&row.vort);
                if !x1 {
                    let mut d = C64::ONE;
                    let mut singular = false;
                    for &j in cp {
                        let factor = mp_cr(&row.vort, &sub(&v[j], &v[c1])) * 2.;
                        singular |= !check.relative_within(factor.taxicab(), bound, || {
                            difference_scale(&row.vort, &v[j], &v[c1])
                        });
                        d *= factor;
                    }
                    if singular || !check.absolute(d.taxicab()) {
                        fail(&mut self.alive, &mut self.lane_status, lane);
                        continue;
                    }
                    self.box_c[b * lanes + lane][1] = self.calls.out[r] / d / (vort2 * vort2);
                } else {
                    let mut nmu = [self.calls.out[2 * r], self.calls.out[2 * r + 1]];
                    let vsq = &self.vsq[lane * n..(lane + 1) * n];
                    let mut stable = true;
                    for &j in cp {
                        let kk = sub(&v[j], &v[c1]);
                        let d0 = mp_cr(&row.vort, &kk) * 2.;
                        if !check.relative_within(d0.taxicab(), bound, || {
                            difference_scale(&row.vort, &v[j], &v[c1])
                        }) {
                            stable = false;
                            break;
                        }
                        let d1 =
                            C64::real(vsq[j] - vsq[c1]) - m[j] + m[c1] + mp_cr(&row.mid, &kk) * 2.;
                        nmu[0] /= d0;
                        nmu[1] -= d1 * nmu[0];
                        nmu[1] /= d0;
                    }
                    if !stable {
                        fail(&mut self.alive, &mut self.lane_status, lane);
                        continue;
                    }
                    self.box_c[b * lanes + lane][1] = nmu[1] / (vort2 * vort2);
                }
            }
        }
        Ok(())
    }

    /// [`FastAmplitude::bubbles`] with the rank n+2 engine's bubble references.
    fn bubbles_conditioned<N: FastNumerator + ?Sized>(&mut self, input: &FastInput, num: &mut N) -> Result<()> {
        let n = self.topo.n;
        let lanes = self.lanes;
        let eps = self.options.infrared_eps;
        let check = Stability::new(&self.options);
        let rmn = self.topo.rmn();
        let x1 = self.topo.x1();
        let mindeg = (rmn + 2) as usize;
        let width = t2_len_le3(mindeg);
        let nterms = mindeg + 1;
        let (lin, quad) = (rmn >= -1, rmn >= 0);
        // Triangle parents exist only when the triangle stage ran.
        let (correct_all, correct_spurious): (Option<BubbleCorrection>, Option<BubbleCorrection>) =
            if !self.any_triangle() {
                (None, None)
            } else {
                match rmn {
                    -2 => (Some(subs::bubble_from_triangle_rm2_all), None),
                    -1 => (
                        Some(subs::bubble_from_triangle_rm1_all),
                        Some(subs::bubble_from_triangle_rm1_spurious),
                    ),
                    0 => (
                        Some(subs::bubble_from_triangle_r0_all),
                        Some(subs::bubble_from_triangle_r0_spurious),
                    ),
                    _ => (
                        Some(subs::bubble_from_triangle_x1_all),
                        Some(subs::bubble_from_triangle_x1_spurious),
                    ),
                }
            };
        for b in 0..self.topo.bubbles.len() {
            self.refresh_active();
            if self.active.is_empty() {
                return Ok(());
            }
            let p = self.topo.bubbles[b];
            let [c1, c2] = p;
            self.bub_rows.clear();
            self.calls.clear();
            for ai in 0..self.active.len() {
                let lane = self.active[ai];
                let v = &input.shifts[lane * n..(lane + 1) * n];
                let m = &input.masses_squared[lane * n..(lane + 1) * n];
                let s = self.smat[lane * n * n + c2 * n + c1];
                // Zero-initialized like a fresh Ninja CutsVector entry.
                self.bub[b * lanes + lane] = BubbleData::default();
                if is_zero(m[c1]) && is_zero(m[c2]) && s.abs() < eps {
                    continue;
                }
                let k = sub(&v[c1], &v[c2]);
                let e = {
                    // The reference that best conditions the uncut lines.
                    let cp = &self.topo.bubble_complements[b * (n - 2)..(b + 1) * (n - 2)];
                    bubble_basis_x2(&k, v, c1, cp, eps)
                };
                let (b0, b1, param) = bubble_solution(&k, &e, m[c1], m[c2]);
                let e1effl = [b0 * e.e2[0], b0 * e.e2[1], b0 * e.e2[2], b0 * e.e2[3]];
                let e2eff = [
                    e.e1[0] + b1 * e.e2[0],
                    e.e1[1] + b1 * e.e2[1],
                    e.e1[2] + b1 * e.e2[2],
                    e.e1[3] + b1 * e.e2[3],
                ];
                let e1eff = sub_cr(&e1effl, &v[c1]);
                self.calls.lanes.push(lane);
                self.calls.v0.push(e1eff);
                self.calls.v1.push(cmom_from_real(&e2eff));
                self.calls.v2.push(e.e3);
                self.calls.v3.push(scale_c(&e.e4, 0.5 / e.mp34()));
                self.calls.params.push(param);
                self.bub_rows.push(BubbleRow {
                    lane,
                    e,
                    b0,
                    b1,
                    e1eff,
                    e2eff,
                    param,
                });
            }
            if self.bub_rows.is_empty() {
                continue;
            }
            self.calls.prepare_out(width);
            num.t2_expansion(
                &p,
                &self.calls.lanes,
                &self.calls.v0,
                &self.calls.v1,
                &self.calls.v2,
                &self.calls.v3,
                &self.calls.params,
                mindeg,
                &mut self.calls.out,
            )?;
            let cp = &self.topo.bubble_complements[b * (n - 2)..(b + 1) * (n - 2)];
            let parents = &self.topo.bubble_parents[b];
            self.second.clear();
            for r in 0..self.bub_rows.len() {
                let row = &self.bub_rows[r];
                let lane = row.lane;
                let v = &input.shifts[lane * n..(lane + 1) * n];
                let m = &input.masses_squared[lane * n..(lane + 1) * n];
                let vsq = &self.vsq[lane * n..(lane + 1) * n];
                let numexp = &mut self.calls.out[r * width..(r + 1) * width];
                let e = &row.e;
                let inv34 = 1. / e.mp34();
                let e4s = 0.5 * inv34;
                let uncut = &mut self.uncut;
                let mut stable = true;
                for &j in cp {
                    let pdiff = sub(&v[j], &v[c1]);
                    let e3k = mp_cr(&e.e3, &pdiff);
                    if !check.absolute(e3k.taxicab() * 2.) {
                        stable = false;
                        break;
                    }
                    // a0 = b0 e2 and a1 = e1 + b1 e2 with real e1, e2.
                    let e2p = mp_rr(&e.e2, &pdiff);
                    let e1k = row.b0 * e2p;
                    uncut[j] = Uncut {
                        e3k,
                        ie3k: e3k.inv(),
                        e4k: mp_cr(&e.e4, &pdiff),
                        e1k,
                        e2k: mp_rr(&e.e1, &pdiff) + row.b1 * e2p,
                        d1: (e1k - mp_rr(&v[c1], &pdiff)) * 2.
                            + (vsq[j] - vsq[c1])
                            + (-m[j] + m[c1]),
                        f: mp2(&pdiff) + (m[c1] - m[j]),
                    };
                }
                if !stable {
                    fail(&mut self.alive, &mut self.lane_status, lane);
                    continue;
                }
                for &j in cp {
                    let u = &uncut[j];
                    let d2 = u.e4k * inv34;
                    divide_t2(
                        numexp,
                        nterms,
                        u.ie3k * 0.5,
                        [u.d1, C64::real(2. * u.e2k)],
                        [row.param[0] * d2, d2, row.param[1] * d2, row.param[2] * d2],
                    );
                }
                if let Some(all) = correct_all {
                    let e4eff = scale_c(&e.e4, e4s);
                    let np = parents.len();
                    if self.pairs.len() < (r + 1) * np {
                        self.pairs.resize((r + 1) * np, [C64::ZERO; 4]);
                    }
                    for (pi, &(t, unc)) in parents.iter().enumerate() {
                        let tri = &self.tri[t * lanes + lane];
                        let u = &uncut[unc];
                        // Shared with the second orientation: a0 = b0 e2,
                        // a1 = e1 + b1 e2 against the triangle's e3, e4.
                        let (e2t3, e2t4) = (mp_cr(&tri.e3, &e.e2), mp_cr(&tri.e4, &e.e2));
                        let pair = [
                            row.b0 * e2t3,
                            row.b0 * e2t4,
                            mp_cr(&tri.e3, &e.e1) + e2t3 * row.b1,
                            mp_cr(&tri.e4, &e.e1) + e2t4 * row.b1,
                        ];
                        self.pairs[r * np + pi] = pair;
                        all(
                            numexp,
                            &tri.e3,
                            &tri.e4,
                            &tri.c,
                            &e.e3,
                            &e4eff,
                            &row.param,
                            u.e1k,
                            u.e2k,
                            u.e3k,
                            u.ie3k,
                            u.e4k * e4s,
                            &pair,
                            u.f,
                        );
                    }
                }
                let inv12 = 1. / e.mp12;
                let data = &mut self.bub[b * lanes + lane];
                data.e2 = e.e2;
                data.e3 = e.e3;
                data.e4 = e.e4;
                let c = &mut data.c;
                if !x1 {
                    let jext2 = rmn;
                    let jext1 = rmn + 1;
                    let jext1x1 = 2;
                    let jext0 = (rmn + 2) as usize + usize::from(quad);
                    let jext0mu2 = 4;
                    let jext0x1 = if quad { 5 } else { 2 };
                    let jext0x2 = 6;
                    c[0] = numexp[jext0];
                    if lin {
                        c[1] = numexp[jext0x1] * inv12;
                        c[5] = numexp[jext1 as usize] * inv34;
                    }
                    if quad {
                        c[9] = numexp[jext0mu2];
                        c[2] = numexp[jext0x2] * (inv12 * inv12);
                        c[6] = numexp[jext2 as usize] * (inv34 * inv34);
                        c[8] = numexp[jext1x1] * (inv12 * inv34);
                    }
                } else {
                    c[0] = numexp[7];
                    c[1] = numexp[9] * inv12;
                    c[5] = numexp[3] * inv34;
                    c[9] = numexp[8];
                    c[2] = numexp[11] * (inv12 * inv12);
                    c[6] = numexp[1] * (inv34 * inv34);
                    c[8] = numexp[5] * (inv12 * inv34);
                    c[10] = numexp[10] * inv12;
                    c[12] = numexp[4] * inv34;
                    c[13] = numexp[12] * (inv12 * inv12 * inv12);
                    c[15] = numexp[0] * (inv34 * inv34 * inv34);
                    c[17] = numexp[6] * (inv34 * inv12 * inv12);
                    c[19] = numexp[2] * (inv34 * inv34 * inv12);
                }
                // Spurious coefficients are needed only as subtraction terms
                // of massive tadpoles (Ninja's spurious_terms_needed).
                if lin && (!is_zero(m[c1]) || !is_zero(m[c2])) {
                    self.second.push(r);
                }
            }
            if self.second.is_empty() {
                continue;
            }
            // Second orientation (e3 <-> e4) with reduced depth.
            let mindeg2 = mindeg - 1;
            let width2 = t2_len_le3(mindeg2);
            self.calls.clear();
            // Rows of lanes that failed in the first orientation never enter
            // `second`, so call row `s` belongs to `bub_rows[second[s]]`.
            for &r in &self.second {
                let row = &self.bub_rows[r];
                self.calls.lanes.push(row.lane);
                self.calls.v0.push(row.e1eff);
                self.calls.v1.push(cmom_from_real(&row.e2eff));
                self.calls.v2.push(row.e.e4);
                self.calls.v3.push(scale_c(&row.e.e3, 0.5 / row.e.mp34()));
                self.calls.params.push(row.param);
            }
            self.calls.prepare_out(width2);
            num.t2_expansion(
                &p,
                &self.calls.lanes,
                &self.calls.v0,
                &self.calls.v1,
                &self.calls.v2,
                &self.calls.v3,
                &self.calls.params,
                mindeg2,
                &mut self.calls.out,
            )?;
            for s in 0..self.second.len() {
                let row = &self.bub_rows[self.second[s]];
                let lane = row.lane;
                let v = &input.shifts[lane * n..(lane + 1) * n];
                let m = &input.masses_squared[lane * n..(lane + 1) * n];
                let vsq = &self.vsq[lane * n..(lane + 1) * n];
                let numexp = &mut self.calls.out[s * width2..(s + 1) * width2];
                let e = &row.e;
                let inv34 = 1. / e.mp34();
                let e3s = 0.5 * inv34;
                let uncut = &mut self.uncut;
                let mut stable = true;
                for &j in cp {
                    let pdiff = sub(&v[j], &v[c1]);
                    // Orientation e3 <-> e4: `e3k` holds e4.(V_j - V_0) here.
                    let e4k = mp_cr(&e.e4, &pdiff);
                    if !check.absolute(e4k.taxicab() * 2.) {
                        stable = false;
                        break;
                    }
                    let e2p = mp_rr(&e.e2, &pdiff);
                    let e1k = row.b0 * e2p;
                    uncut[j] = Uncut {
                        e3k: e4k,
                        ie3k: e4k.inv(),
                        e4k: mp_cr(&e.e3, &pdiff),
                        e1k,
                        e2k: mp_rr(&e.e1, &pdiff) + row.b1 * e2p,
                        d1: (e1k - mp_rr(&v[c1], &pdiff)) * 2.
                            + (vsq[j] - vsq[c1])
                            + (-m[j] + m[c1]),
                        f: mp2(&pdiff) + (m[c1] - m[j]),
                    };
                }
                if !stable {
                    fail(&mut self.alive, &mut self.lane_status, lane);
                    continue;
                }
                for &j in cp {
                    let u = &uncut[j];
                    let d2 = u.e4k * inv34;
                    divide_t2(
                        numexp,
                        nterms - 1,
                        u.ie3k * 0.5,
                        [u.d1, C64::real(2. * u.e2k)],
                        [row.param[0] * d2, d2, row.param[1] * d2, row.param[2] * d2],
                    );
                }
                if let Some(spurious) = correct_spurious {
                    let e3eff = scale_c(&e.e3, e3s);
                    let np = parents.len();
                    let r = self.second[s];
                    for (pi, &(t, unc)) in parents.iter().enumerate() {
                        let tri = &self.tri[t * lanes + lane];
                        let u = &uncut[unc];
                        spurious(
                            numexp,
                            &tri.e3,
                            &tri.e4,
                            &tri.c,
                            &e.e4,
                            &e3eff,
                            &row.param,
                            u.e1k,
                            u.e2k,
                            u.e3k,
                            u.ie3k,
                            u.e4k * e3s,
                            &self.pairs[r * np + pi],
                            u.f,
                        );
                    }
                }
                let inv12 = 1. / e.mp12;
                let c = &mut self.bub[b * lanes + lane].c;
                if !x1 {
                    c[3] = numexp[(rmn + 1) as usize] * inv34;
                    if quad {
                        c[4] = numexp[rmn as usize] * (inv34 * inv34);
                        c[7] = numexp[2] * (inv12 * inv34);
                    }
                } else {
                    c[3] = numexp[3] * inv34;
                    c[4] = numexp[1] * (inv34 * inv34);
                    c[7] = numexp[5] * (inv12 * inv34);
                    c[11] = numexp[4] * inv34;
                    c[14] = numexp[0] * (inv34 * inv34 * inv34);
                    c[16] = numexp[6] * (inv34 * inv12 * inv12);
                    c[18] = numexp[2] * (inv34 * inv34 * inv12);
                }
            }
        }
        Ok(())
    }

    /// [`FastAmplitude::tadpoles`] with the rank n+2 engine's tadpole bases.
    fn tadpoles_conditioned<N: FastNumerator + ?Sized>(
        &mut self,
        input: &FastInput,
        num: &mut N,
        bases: &TadpoleBasesX2,
    ) -> Result<()> {
        let n = self.topo.n;
        let lanes = self.lanes;
        let eps = self.options.infrared_eps;
        let check = Stability::new(&self.options);
        let rmn = self.topo.rmn();
        let x1 = self.topo.x1();
        let mindeg = (rmn + 1) as usize;
        let width = t3_len(mindeg);
        let nterms = mindeg + 1;
        let any_triangle = self.any_triangle();
        let any_bubble = self.any_bubble();
        let orientations = if x1 { 2 } else { 1 };
        for a in 0..n {
            self.refresh_active();
            if self.active.is_empty() {
                return Ok(());
            }
            self.tad_rows.clear();
            for ai in 0..self.active.len() {
                let lane = self.active[ai];
                if is_zero(input.masses_squared[lane * n + a]) {
                    continue;
                }
                let v = &input.shifts[lane * n..(lane + 1) * n];
                let basis = (0..2)
                    .find(|&i| {
                        let e = &self.tadpole_bases[i];
                        reference_vectors_ok(v, a, &e.e1, &e.e2, eps)
                    })
                    .unwrap_or(2);
                // The basis that best conditions the uncut lines.
                let basis = {
                    let uncut = &self.topo.tadpole_complements[a * (n - 1)..(a + 1) * (n - 1)];
                    bases.choose(basis, v, a, uncut)
                };
                self.tad_rows.push(TadpoleRow { lane, basis });
            }
            if self.tad_rows.is_empty() {
                continue;
            }
            // Expansion vectors per orientation: rank <= n uses (e3, e4);
            // rank n+1 uses (e1, e2) rows followed by (e3, e4) rows.
            let vectors = |e: &Basis, orientation: usize| -> (CMom, CMom) {
                if x1 && orientation == 0 {
                    (
                        cmom_from_real(&e.e1),
                        cmom_from_real(&scale(&e.e2, 0.5 / e.mp12)),
                    )
                } else {
                    (e.e3, scale_c(&e.e4, 0.5 / e.mp34()))
                }
            };
            self.calls.clear();
            for orientation in 0..orientations {
                for row in &self.tad_rows {
                    let lane = row.lane;
                    let (ev3, ev4) = vectors(&bases.bases[row.basis], orientation);
                    let v0 = input.shifts[lane * n + a];
                    self.calls.lanes.push(lane);
                    self.calls
                        .v0
                        .push(cmom_from_real(&[-v0[0], -v0[1], -v0[2], -v0[3]]));
                    self.calls.v1.push(ev3);
                    self.calls.v2.push(ev4);
                    self.calls.scalar.push(input.masses_squared[lane * n + a]);
                }
            }
            self.calls.prepare_out(width);
            let cut = [a];
            num.t3_expansion(
                &cut,
                &self.calls.lanes,
                &self.calls.v0,
                &self.calls.v1,
                &self.calls.v2,
                &self.calls.scalar,
                mindeg,
                &mut self.calls.out,
            )?;
            let rows = self.tad_rows.len();
            let cp = &self.topo.tadpole_complements[a * (n - 1)..(a + 1) * (n - 1)];
            let tri_parents = &self.topo.tadpole_triangle_parents[a];
            let bub_parents = &self.topo.tadpole_bubble_parents[a];
            for r in 0..rows {
                let row = self.tad_rows[r];
                let lane = row.lane;
                let v = &input.shifts[lane * n..(lane + 1) * n];
                let m = &input.masses_squared[lane * n..(lane + 1) * n];
                let vsq = &self.vsq[lane * n..(lane + 1) * n];
                let e = bases.bases[row.basis];
                let param = m[a];
                let amu = [-v[a][0], -v[a][1], -v[a][2], -v[a][3]];
                let mut stable = true;
                let mut data = TadpoleData {
                    e34: -mp_rr(&e.e1, &e.e2),
                    ..TadpoleData::default()
                };
                for orientation in 0..orientations {
                    let offset = (orientation * rows + r) * width;
                    let numexp = &mut self.calls.out[offset..offset + width];
                    let (ev3, ev4) = vectors(&e, orientation);
                    let uncut = &mut self.uncut;
                    for &j in cp {
                        let pdiff = sub(&v[j], &v[a]);
                        let e3k = mp_cr(&ev3, &pdiff);
                        if !check.absolute(e3k.taxicab() * 2.) {
                            stable = false;
                            break;
                        }
                        uncut[j] = Uncut {
                            e3k,
                            ie3k: e3k.inv(),
                            e4k: mp_cr(&ev4, &pdiff),
                            e1k: C64::ZERO,
                            e2k: 0.,
                            d1: C64::real(2. * mp_rr(&amu, &pdiff) + (vsq[j] - vsq[a])) - m[j]
                                + m[a],
                            f: mp2(&pdiff) + (m[a] - m[j]),
                        };
                    }
                    if !stable {
                        break;
                    }
                    for &j in cp {
                        let u = &uncut[j];
                        let d2 = u.e4k * 2.;
                        divide_t3(numexp, nterms, u.ie3k * 0.5, u.d1, [param * d2, d2]);
                    }
                    if any_triangle {
                        for &(t, [u0, u1]) in tri_parents {
                            let tri = &self.tri[t * lanes + lane];
                            let (k0, k1) = (&uncut[u0], &uncut[u1]);
                            if x1 {
                                subs::tadpole_from_triangle_x1(
                                    numexp, &tri.e3, &tri.e4, &tri.c, &ev3, &ev4, param, k0.e3k,
                                    k0.ie3k, k0.e4k, k0.f, k1.e3k, k1.ie3k, k1.e4k, k1.f,
                                );
                            } else {
                                tadpole_from_triangle(
                                    numexp, &tri.e3, &tri.e4, &tri.c, &ev3, k0, k1, rmn,
                                );
                            }
                        }
                    }
                    if any_bubble {
                        for &(bb, unc) in bub_parents {
                            let bub = &self.bub[bb * lanes + lane];
                            let kbol = sub(&v[self.topo.bubbles[bb][0]], &v[a]);
                            let u = &uncut[unc];
                            if x1 {
                                subs::tadpole_from_bubble_x1(
                                    numexp, &bub.e2, &bub.e3, &bub.e4, &kbol, &bub.c, &ev3, &ev4,
                                    param, u.e3k, u.ie3k, u.e4k, u.f,
                                );
                            } else {
                                tadpole_from_bubble(
                                    numexp, &bub.e2, &bub.e3, &bub.e4, &kbol, &bub.c, &ev3, u, rmn,
                                );
                            }
                        }
                    }
                    if !x1 {
                        data.c0 = numexp[(1 + rmn) as usize];
                    } else if orientation == 0 {
                        data.c0 = numexp[2];
                        data.c14 = numexp[3];
                    } else {
                        data.c15 = (numexp[3] - data.c14) * (2. / e.mp34());
                    }
                }
                if !stable {
                    fail(&mut self.alive, &mut self.lane_status, lane);
                    continue;
                }
                self.tad[a * lanes + lane] = data;
            }
        }
        Ok(())
    }
}
