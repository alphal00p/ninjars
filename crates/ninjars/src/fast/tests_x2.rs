//! Unit tests of the rank n+2 fast engine: the rank n+2 jets against direct
//! polynomial expansions, the deeper Laurent divisions against multiplication
//! back, the generated subtractions against the series-definition fixtures
//! (validation V3), and engine-level identities (rank padding against the
//! rank <= n+1 engine, pinch identities with mu²-dependent numerators,
//! batching, replay and lane statuses).

use super::bases_x2::with_ninja_choices;
use super::complex::{C64, CMom, RMom, mp_cc};
use super::engine_x2::{divide_t2_5, divide_t3_6};
use super::numerator::{FastNumerator, t2_len, t3_len};
use super::replay::{RecordingMasters, RecordingNumerator};
use super::subtractions_series_x2 as series;
use super::subtractions_x2 as subs;
use super::tensor::TensorNumerator;
use super::tensor_layout::{monomial_index, monomials, ncoeff, ncoeff_mu};
use super::tensor_mu::MuTensorNumerator;
use super::tests::{
    Poly, Rng, UnitMasters, assert_close, get, linear, poly_mul, reference_mu_numerator,
    reference_numerator, unit_tensor,
};
use super::{
    FastAmplitude, FastInput, FastMaster, FastMasters, FastOptions, FastResult, FastStatus,
};
use crate::{Error, Result};

// ------------------------------------------------------------------ layouts

/// `(order, x power, mu² power)` of the t2 layout: order `i` lists
/// `x^a mu^(2j)` with `a + 2j <= i`, ordered by `a` then `j`.
fn t2_layout(mindeg: usize) -> Vec<(usize, u16, u16)> {
    let mut out = Vec::new();
    for i in 0..=mindeg {
        for a in 0..=i {
            for j in 0..=(i - a) / 2 {
                out.push((i, a as u16, j as u16));
            }
        }
    }
    out
}

/// `(order, 0, mu² power)` of the t3 layout.
fn t3_layout(mindeg: usize) -> Vec<(usize, u16, u16)> {
    let mut out = Vec::new();
    for i in 0..=mindeg {
        for j in 0..=i / 2 {
            out.push((i, 0, j as u16));
        }
    }
    out
}

#[test]
fn rank_n2_layouts_are_canonical() {
    assert_eq!(t3_layout(5).len(), t3_len(5));
    assert_eq!(t2_layout(4).len(), t2_len(4));
    // Design 1.3: T2 orientation A, slots 13..21 are the t^0 level.
    let level4: Vec<(u16, u16)> = t2_layout(4)[13..].iter().map(|&(_, a, j)| (a, j)).collect();
    assert_eq!(
        level4,
        [
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 1),
            (2, 0),
            (2, 1),
            (3, 0),
            (4, 0)
        ]
    );
}

fn t3_poly(a: &CMom, e3: &CMom, e4: &CMom, param: C64) -> [Poly; 4] {
    std::array::from_fn(|mu| {
        linear(&[
            ((0, 0, 0), a[mu]),
            ((1, 0, 0), e3[mu]),
            ((-1, 0, 0), param * e4[mu]),
            ((-1, 0, 1), e4[mu]),
        ])
    })
}

fn t2_poly(a0: &CMom, a1: &CMom, e3: &CMom, e4: &CMom, p: &[C64; 3]) -> [Poly; 4] {
    std::array::from_fn(|mu| {
        linear(&[
            ((0, 0, 0), a0[mu]),
            ((0, 1, 0), a1[mu]),
            ((1, 0, 0), e3[mu]),
            ((-1, 0, 0), p[0] * e4[mu]),
            ((-1, 1, 0), p[1] * e4[mu]),
            ((-1, 2, 0), p[2] * e4[mu]),
            ((-1, 0, 1), e4[mu]),
        ])
    })
}

fn expected(poly: &Poly, rank: usize, layout: &[(usize, u16, u16)]) -> Vec<C64> {
    let r = rank as i32;
    layout
        .iter()
        .map(|&(i, x, m)| get(poly, (r - i as i32, x, m)))
        .collect()
}

// -------------------------------------------------------------------- jets

#[test]
fn rank_n2_tensor_expansions_match_direct_polynomial_expansions() {
    let mut rng = Rng(71);
    for rank in 0..=9 {
        let lanes = 2;
        let mut numerator = TensorNumerator::new(rank, lanes);
        for c in numerator.coefficients_mut() {
            *c = rng.c();
        }
        let r = rank as i32;
        for lane in 0..lanes {
            let coefficients = numerator.lane_coefficients(lane).to_vec();
            let cut = [0usize];
            // Box: q = t v0 + v1; the q-only tensor ignores mu² and beta.
            let (v0, v1, beta) = (rng.cmom(), rng.cmom(), rng.c());
            let poly = reference_numerator(
                &coefficients,
                rank,
                &std::array::from_fn(|mu| linear(&[((1, 0, 0), v0[mu]), ((0, 0, 0), v1[mu])])),
            );
            let mut out = [C64::ZERO; 3];
            numerator
                .mu_expansion_x2(&cut, &[lane], &[v0], &[v1], &[beta], &mut out)
                .unwrap();
            assert_close(
                &out,
                &[
                    get(&poly, (r, 0, 0)),
                    get(&poly, (r - 1, 0, 0)),
                    get(&poly, (r - 2, 0, 0)),
                ],
                &format!("Mu3 rank {rank}"),
            );
            // Triangle, mindeg 5 (T3K5).
            let (a, e3, e4, param) = (rng.cmom(), rng.cmom(), rng.cmom(), rng.c());
            let poly = reference_numerator(&coefficients, rank, &t3_poly(&a, &e3, &e4, param));
            let mut out = vec![C64::ZERO; t3_len(5)];
            numerator
                .t3_expansion_x2(&cut, &[lane], &[a], &[e3], &[e4], &[param], &mut out)
                .unwrap();
            assert_close(
                &out,
                &expected(&poly, rank, &t3_layout(5)),
                &format!("T3K5 rank {rank}"),
            );
            // Bubble, mindeg 4 (T2K4).
            let (a0, a1, e3, e4) = (rng.cmom(), rng.cmom(), rng.cmom(), rng.cmom());
            let p = [rng.c(), rng.c(), rng.c()];
            let poly = reference_numerator(&coefficients, rank, &t2_poly(&a0, &a1, &e3, &e4, &p));
            let mut out = vec![C64::ZERO; t2_len(4)];
            numerator
                .t2_expansion_x2(&cut, &[lane], &[a0], &[a1], &[e3], &[e4], &[p], &mut out)
                .unwrap();
            assert_close(
                &out,
                &expected(&poly, rank, &t2_layout(4)),
                &format!("T2K4 rank {rank}"),
            );
        }
    }
    // `t3_expansion`/`t2_expansion` keep Ninja's depth limits (4 and 3):
    // the rank n+2 depths are reached only through the `_x2` methods.
    let mut numerator = TensorNumerator::new(4, 1);
    let v = [C64::ZERO; 4];
    assert!(
        numerator
            .t3_expansion(
                &[0],
                &[0],
                &[v],
                &[v],
                &[v],
                &[C64::ZERO],
                5,
                &mut [C64::ZERO; 12]
            )
            .is_err()
    );
    assert!(
        numerator
            .t2_expansion(
                &[0],
                &[0],
                &[v],
                &[v],
                &[v],
                &[v],
                &[[C64::ZERO; 3]],
                4,
                &mut [C64::ZERO; 22]
            )
            .is_err()
    );
}

#[test]
fn rank_n2_mu_tensor_expansions_match_direct_polynomial_expansions() {
    let mut rng = Rng(73);
    for rank in 0..=9 {
        for mu_powers in 0..=rank / 2 {
            let lanes = 2;
            let mut numerator = MuTensorNumerator::new(rank, mu_powers, lanes).unwrap();
            for c in numerator.coefficients_mut() {
                *c = rng.c();
            }
            let r = rank as i32;
            let context = |what: &str| format!("{what}: rank {rank} mu powers {mu_powers}");
            for lane in 0..lanes {
                let cut = [0usize];
                // Box on the d-dimensional cut: q = t v0 + v1, mu² = t² v0² - beta.
                let (v0, v1, beta) = (rng.cmom(), rng.cmom(), rng.c());
                let poly = reference_mu_numerator(
                    &numerator,
                    lane,
                    &std::array::from_fn(|mu| linear(&[((1, 0, 0), v0[mu]), ((0, 0, 0), v1[mu])])),
                    &linear(&[((2, 0, 0), mp_cc(&v0, &v0)), ((0, 0, 0), -beta)]),
                );
                let mut out = [C64::ZERO; 3];
                numerator
                    .mu_expansion_x2(&cut, &[lane], &[v0], &[v1], &[beta], &mut out)
                    .unwrap();
                assert_close(
                    &out,
                    &[
                        get(&poly, (r, 0, 0)),
                        get(&poly, (r - 1, 0, 0)),
                        get(&poly, (r - 2, 0, 0)),
                    ],
                    &context("MuShifted expansion"),
                );
                let mu2 = linear(&[((0, 0, 1), C64::ONE)]);
                let (a, e3, e4, param) = (rng.cmom(), rng.cmom(), rng.cmom(), rng.c());
                let poly =
                    reference_mu_numerator(&numerator, lane, &t3_poly(&a, &e3, &e4, param), &mu2);
                let mut out = vec![C64::ZERO; t3_len(5)];
                numerator
                    .t3_expansion_x2(&cut, &[lane], &[a], &[e3], &[e4], &[param], &mut out)
                    .unwrap();
                assert_close(
                    &out,
                    &expected(&poly, rank, &t3_layout(5)),
                    &context("t3 mindeg 5"),
                );
                let (a0, a1, e3, e4) = (rng.cmom(), rng.cmom(), rng.cmom(), rng.cmom());
                let p = [rng.c(), rng.c(), rng.c()];
                let poly = reference_mu_numerator(
                    &numerator,
                    lane,
                    &t2_poly(&a0, &a1, &e3, &e4, &p),
                    &mu2,
                );
                let mut out = vec![C64::ZERO; t2_len(4)];
                numerator
                    .t2_expansion_x2(&cut, &[lane], &[a0], &[a1], &[e3], &[e4], &[p], &mut out)
                    .unwrap();
                assert_close(
                    &out,
                    &expected(&poly, rank, &t2_layout(4)),
                    &context("t2 mindeg 4"),
                );
            }
        }
    }
}

#[test]
fn rank_n2_mu_free_tensor_numerators_agree_bitwise() {
    let mut rng = Rng(79);
    let rank = 7;
    let mut plain = TensorNumerator::new(rank, 2);
    for c in plain.coefficients_mut() {
        *c = rng.c();
    }
    let mut mu = MuTensorNumerator::new(rank, 0, 2).unwrap();
    mu.coefficients_mut().copy_from_slice(plain.coefficients());
    let lanes = [1, 0];
    let vecs = |rng: &mut Rng| -> Vec<CMom> { (0..2).map(|_| rng.cmom()).collect() };
    let (a, e3, e4, e5) = (
        vecs(&mut rng),
        vecs(&mut rng),
        vecs(&mut rng),
        vecs(&mut rng),
    );
    let param: Vec<C64> = (0..2).map(|_| rng.c()).collect();
    let p: Vec<[C64; 3]> = (0..2).map(|_| [rng.c(), rng.c(), rng.c()]).collect();
    let (mut x, mut y) = (vec![C64::ZERO; 24], vec![C64::ZERO; 24]);
    plain
        .t3_expansion_x2(&[0], &lanes, &a, &e3, &e4, &param, &mut x)
        .unwrap();
    mu.t3_expansion_x2(&[0], &lanes, &a, &e3, &e4, &param, &mut y)
        .unwrap();
    assert_eq!(x, y);
    let (mut x, mut y) = (vec![C64::ZERO; 44], vec![C64::ZERO; 44]);
    plain
        .t2_expansion_x2(&[0], &lanes, &a, &e3, &e4, &e5, &p, &mut x)
        .unwrap();
    mu.t2_expansion_x2(&[0], &lanes, &a, &e3, &e4, &e5, &p, &mut y)
        .unwrap();
    assert_eq!(x, y);
    let (mut x, mut y) = (vec![C64::ZERO; 6], vec![C64::ZERO; 6]);
    plain
        .mu_expansion_x2(&[0], &lanes, &a, &e3, &param, &mut x)
        .unwrap();
    mu.mu_expansion_x2(&[0], &lanes, &a, &e3, &[C64::ZERO; 2], &mut y)
        .unwrap();
    assert_eq!(x, y);
}

// --------------------------------------------------------------- divisions

#[test]
fn rank_n2_divisions_match_multiplication_back() {
    let mut rng = Rng(83);
    for t2 in [false, true] {
        for _ in 0..4 {
            let entries = if t2 { t2_layout(4) } else { t3_layout(5) };
            let numerator: Vec<C64> = entries.iter().map(|_| rng.c()).collect();
            let d0 = rng.c();
            let mut quotient = numerator.clone();
            let mut d = Poly::new();
            d.insert((1, 0, 0), d0);
            if t2 {
                let (d1, d2) = ([rng.c(), rng.c()], [rng.c(), rng.c(), rng.c(), rng.c()]);
                divide_t2_5(&mut quotient, d0.inv(), d1, d2);
                d.insert((0, 0, 0), d1[0]);
                d.insert((0, 1, 0), d1[1]);
                d.insert((-1, 0, 0), d2[0]);
                d.insert((-1, 0, 1), d2[1]);
                d.insert((-1, 1, 0), d2[2]);
                d.insert((-1, 2, 0), d2[3]);
            } else {
                let (d1, d2) = (rng.c(), [rng.c(), rng.c()]);
                divide_t3_6(&mut quotient, d0.inv(), d1, d2);
                d.insert((0, 0, 0), d1);
                d.insert((-1, 0, 0), d2[0]);
                d.insert((-1, 0, 1), d2[1]);
            }
            // Quotient order i multiplies t^(r-1-i): D Q reproduces N on
            // every order kept (NT = 6 for T3, NT = 5 for T2).
            let q: Poly = entries
                .iter()
                .zip(&quotient)
                .map(|(&(i, x, m), &v)| ((-1 - i as i32, x, m), v))
                .collect();
            let product = poly_mul(&d, &q);
            let actual: Vec<C64> = entries
                .iter()
                .map(|&(i, x, m)| get(&product, (-(i as i32), x, m)))
                .collect();
            assert_close(&actual, &numerator, &format!("division t2={t2}"));
        }
    }
}

// ------------------------------------------------------------ subtractions

/// Minimal JSON reader for the fixtures (the crate has no JSON dependency).
#[derive(Debug)]
enum Json {
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
    Other,
}

struct JsonReader<'a> {
    s: &'a [u8],
    i: usize,
}

impl JsonReader<'_> {
    fn ws(&mut self) {
        while self.i < self.s.len() && self.s[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }
    fn value(&mut self) -> Json {
        self.ws();
        match self.s[self.i] {
            b'{' => {
                self.i += 1;
                let mut fields = Vec::new();
                loop {
                    self.ws();
                    if self.s[self.i] == b'}' {
                        self.i += 1;
                        return Json::Obj(fields);
                    }
                    let Json::Str(key) = self.value() else {
                        panic!("object key")
                    };
                    self.ws();
                    assert_eq!(self.s[self.i], b':');
                    self.i += 1;
                    let value = self.value();
                    fields.push((key, value));
                    self.ws();
                    if self.s[self.i] == b',' {
                        self.i += 1;
                    }
                }
            }
            b'[' => {
                self.i += 1;
                let mut items = Vec::new();
                loop {
                    self.ws();
                    if self.s[self.i] == b']' {
                        self.i += 1;
                        return Json::Arr(items);
                    }
                    items.push(self.value());
                    self.ws();
                    if self.s[self.i] == b',' {
                        self.i += 1;
                    }
                }
            }
            b'"' => {
                self.i += 1;
                let mut out = String::new();
                while self.s[self.i] != b'"' {
                    if self.s[self.i] == b'\\' {
                        self.i += 1;
                    }
                    out.push(self.s[self.i] as char);
                    self.i += 1;
                }
                self.i += 1;
                Json::Str(out)
            }
            b't' | b'f' | b'n' => {
                while self.i < self.s.len() && self.s[self.i].is_ascii_alphabetic() {
                    self.i += 1;
                }
                Json::Other
            }
            _ => {
                let start = self.i;
                while self.i < self.s.len()
                    && matches!(
                        self.s[self.i],
                        b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9'
                    )
                {
                    self.i += 1;
                }
                let text = std::str::from_utf8(&self.s[start..self.i]).unwrap();
                Json::Num(text.parse().unwrap())
            }
        }
    }
}

impl Json {
    fn parse(text: &str) -> Json {
        JsonReader {
            s: text.as_bytes(),
            i: 0,
        }
        .value()
    }
    fn field(&self, name: &str) -> &Json {
        let Json::Obj(fields) = self else {
            panic!("not an object")
        };
        &fields.iter().find(|(k, _)| k == name).expect(name).1
    }
    fn list(&self) -> &[Json] {
        let Json::Arr(items) = self else {
            panic!("not an array")
        };
        items
    }
    fn num(&self) -> f64 {
        let Json::Num(x) = self else {
            panic!("not a number")
        };
        *x
    }
    fn text(&self) -> &str {
        let Json::Str(s) = self else {
            panic!("not a string")
        };
        s
    }
    fn c(&self) -> C64 {
        C64::new(self.list()[0].num(), self.list()[1].num())
    }
    fn cs(&self) -> Vec<C64> {
        self.list().iter().map(Json::c).collect()
    }
    fn cmom(&self) -> CMom {
        self.cs().try_into().unwrap()
    }
    fn rmom(&self) -> RMom {
        let v: Vec<f64> = self.list().iter().map(Json::num).collect();
        v.try_into().unwrap()
    }
}

const FIXTURES: &str = include_str!("../../../../validation/rank-n2/subtraction_fixtures.json");

/// The two implementations of the rank n+2 subtractions.
#[derive(Clone, Copy, Debug)]
enum Implementation {
    /// `subtractions_series_x2` (used by the engine).
    Series,
    /// `subtractions_x2`, generated from Ninja's FORM output and the CSE.
    Generated,
}

/// Run one fixture case with (possibly perturbed) parent coefficients; the
/// scaled difference to the series values and whether every other slot is
/// untouched.
fn run_fixture(case: &Json, c: &[C64], implementation: Implementation) -> (f64, bool) {
    let generated = matches!(implementation, Implementation::Generated);
    let kind = case.field("kind").text();
    let width = match kind {
        "tri_to_bub_all" => 22,
        "tri_to_bub_spurious" => 13,
        _ => 6,
    };
    let mut divnum = vec![C64::ZERO; width];
    match kind {
        "tri_to_bub_all" | "tri_to_bub_spurious" => {
            let eb3k = case.field("eb3k").c();
            let args = (
                case.field("et3").cmom(),
                case.field("et4").cmom(),
                case.field("eb3").cmom(),
                case.field("eb4").cmom(),
            );
            let param: [C64; 3] = case.field("param").cs().try_into().unwrap();
            let eb1t: [C64; 4] = case.field("eb1t").cs().try_into().unwrap();
            let c: &[C64; 21] = c.try_into().unwrap();
            let f = if kind == "tri_to_bub_all" {
                if generated {
                    subs::bubble_from_triangle_x2_all
                } else {
                    series::bubble_from_triangle_all
                }
            } else if generated {
                subs::bubble_from_triangle_x2_spurious
            } else {
                series::bubble_from_triangle_spurious
            };
            f(
                &mut divnum,
                &args.0,
                &args.1,
                c,
                &args.2,
                &args.3,
                &param,
                case.field("eb1k").c(),
                case.field("eb2k").num(),
                eb3k,
                eb3k.inv(),
                case.field("eb4k").c(),
                &eb1t,
                case.field("f").c(),
            );
        }
        "tri_to_tad" => {
            let (e3k0, e3k1) = (case.field("e3k0").c(), case.field("e3k1").c());
            let f = if generated {
                subs::tadpole_from_triangle_x2
            } else {
                series::tadpole_from_triangle
            };
            f(
                &mut divnum,
                &case.field("et3").cmom(),
                &case.field("et4").cmom(),
                c.try_into().unwrap(),
                &case.field("e3").cmom(),
                &case.field("e4").cmom(),
                case.field("param").c(),
                e3k0,
                e3k0.inv(),
                case.field("e4k0").c(),
                case.field("f0").c(),
                e3k1,
                e3k1.inv(),
                case.field("e4k1").c(),
                case.field("f1").c(),
            );
        }
        "bub_to_tad" => {
            let e3k = case.field("e3k").c();
            let f = if generated {
                subs::tadpole_from_bubble_x2
            } else {
                series::tadpole_from_bubble
            };
            f(
                &mut divnum,
                &case.field("eb2").rmom(),
                &case.field("eb3").cmom(),
                &case.field("eb4").cmom(),
                &case.field("kbol").rmom(),
                c.try_into().unwrap(),
                &case.field("e3").cmom(),
                &case.field("e4").cmom(),
                case.field("param").c(),
                e3k,
                e3k.inv(),
                case.field("e4k").c(),
                case.field("f").c(),
            );
        }
        other => panic!("unknown fixture kind {other}"),
    }
    let indices: Vec<usize> = case
        .field("indices")
        .list()
        .iter()
        .map(|x| x.num() as usize)
        .collect();
    let expected = case.field("expected").cs();
    let scale = expected.iter().map(|z| z.abs()).fold(0., f64::max);
    let mut error = 0f64;
    for (&i, &e) in indices.iter().zip(&expected) {
        error = error.max((-divnum[i] - e).abs() / scale);
    }
    let untouched = (0..width)
        .filter(|i| !indices.contains(i))
        .all(|i| divnum[i] == C64::ZERO);
    (error, untouched)
}

/// Parent coefficients that reach the slots a routine writes: a monomial of
/// q-degree `a` over `m` uncut denominators reaches t-powers up to `a - m`.
fn feeds(kind: &str, index: usize) -> bool {
    // Triangle (j, a3, a4) and bubble (j, e2, e3, e4) monomial tables.
    const TRIANGLE_DEGREE: [usize; 21] = [
        0, 1, 2, 3, 1, 2, 3, 0, 1, 1, 2, 2, 4, 4, 0, 5, 5, 3, 3, 1, 1,
    ];
    match kind {
        "tri_to_bub_all" => TRIANGLE_DEGREE[index] >= 1,
        // t >= 1 slots (one uncut denominator) and t^0 slots of tadpoles (two).
        "tri_to_bub_spurious" | "tri_to_tad" => TRIANGLE_DEGREE[index] >= 2,
        // Bubble -> tadpole: every monomial with a loop-momentum factor.
        _ => !matches!(index, 0 | 9 | 22),
    }
}

#[test]
fn rank_n2_subtractions_match_series_fixtures() {
    let fixtures = Json::parse(FIXTURES);
    assert_eq!(
        fixtures.field("format").text(),
        "ninjars-rank-n2-subtraction-fixtures-v1"
    );
    let cases = fixtures.field("cases").list();
    assert!(cases.len() >= 24);
    for implementation in [Implementation::Series, Implementation::Generated] {
        let mut worst = 0f64;
        let mut controls = 0;
        for case in cases {
            let kind = case.field("kind").text();
            let c = case.field("c").cs();
            let (error, untouched) = run_fixture(case, &c, implementation);
            assert!(
                untouched,
                "{implementation:?} {kind}: a slot outside the routine's layout was written"
            );
            assert!(
                error <= 1e-13,
                "{implementation:?} {kind}: scaled error {error:e}"
            );
            worst = worst.max(error);
            // Control: perturbing a coefficient that feeds the routine must
            // be detected; one that does not feed it must change nothing.
            for index in 0..c.len() {
                let mut perturbed = c.clone();
                perturbed[index] += C64::new(1e-4, -1e-4) * (1. + c[index].abs());
                let (error, untouched) = run_fixture(case, &perturbed, implementation);
                assert!(untouched);
                if feeds(kind, index) {
                    assert!(
                        error > 1e-9,
                        "{implementation:?} {kind}: perturbed c{index} not detected ({error:e})"
                    );
                    controls += 1;
                } else {
                    assert!(
                        error <= 1e-13,
                        "{implementation:?} {kind}: c{index} must not feed ({error:e})"
                    );
                }
            }
        }
        assert!(controls > 300);
        eprintln!(
            "SUBTRACTIONS_X2 {implementation:?} cases={} worst={worst:e} controls={controls}",
            cases.len()
        );
    }
}

// ------------------------------------------------------------------ engine

/// Deterministic pseudo-random master values: a function of the argument
/// bits and of the output position only, so a bubble's `B0..B11` agree
/// across the rank-2, rank-3 and rank-4 families and identical requests of
/// different reductions get identical values.
struct HashMasters;

fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn unit(z: u64) -> f64 {
    (z >> 11) as f64 / (1u64 << 53) as f64 * 2. - 1.
}

impl FastMasters for HashMasters {
    fn evaluate(
        &mut self,
        family: FastMaster,
        args: &[C64],
        out: &mut [[C64; 3]],
        ok: &mut [bool],
    ) -> Result<()> {
        let width = family.outputs();
        let tag = match family {
            FastMaster::A0 => 1,
            FastMaster::C0 => 2,
            FastMaster::D0 => 3,
            _ => 4,
        };
        for (row, point) in args.chunks_exact(family.arity()).enumerate() {
            let mut h = mix(tag);
            for z in point {
                h = mix(h ^ z.re.to_bits());
                h = mix(h ^ z.im.to_bits());
            }
            for k in 0..width {
                let g = mix(h ^ (k as u64 + 1));
                let v: [C64; 3] = std::array::from_fn(|e| {
                    let a = mix(g ^ (10 + e as u64));
                    C64::new(unit(a), unit(mix(a)))
                });
                out[row * width + k] = v;
            }
            ok[row] = true;
        }
        Ok(())
    }
}

/// Random lanes with `V_0 = 0` and complex masses; lane 1 (if any) has every
/// second line massless (skipped tadpoles, one-sided swapped bubbles).
/// Momenta are redrawn until every plane spanned by two differences is
/// Lorentzian (Ninja's real bases), with spacelike and timelike invariants.
fn kinematics(rng: &mut Rng, n: usize, lanes: usize) -> (Vec<RMom>, Vec<C64>) {
    let mut shifts = Vec::with_capacity(lanes * n);
    for _ in 0..lanes {
        loop {
            let lane: Vec<RMom> = (0..n)
                .map(|i| {
                    if i == 0 {
                        [0.; 4]
                    } else {
                        let mut v = rng.rmom(1.);
                        v[0] = 1.3 * i as f64 + 0.6 * v[0];
                        v
                    }
                })
                .collect();
            let lorentzian = (0..n).all(|i| {
                (i + 1..n).all(|j| {
                    (j + 1..n).all(|k| {
                        let (a, b) = (
                            super::geometry::sub(&lane[j], &lane[i]),
                            super::geometry::sub(&lane[k], &lane[i]),
                        );
                        let ab = super::complex::mp_rr(&a, &b);
                        ab * ab - super::geometry::mp2(&a) * super::geometry::mp2(&b) > 1e-2
                    })
                })
            });
            if lorentzian {
                shifts.extend(lane);
                break;
            }
        }
    }
    let masses = (0..lanes * n)
        .map(|i| {
            let m = C64::new(0.5 + rng.uniform().abs(), -0.1 * rng.uniform().abs());
            if i / n == 1 && (i % n) % 2 == 1 {
                C64::ZERO
            } else {
                m
            }
        })
        .collect();
    (shifts, masses)
}

fn evaluate<N: FastNumerator + ?Sized>(
    n: usize,
    rank: usize,
    input: &FastInput,
    numerator: &mut N,
) -> (Vec<FastResult>, Vec<FastStatus>) {
    let mut amplitude = FastAmplitude::new(n, rank, FastOptions::default()).unwrap();
    let mut results = vec![FastResult::default(); input.lanes];
    let mut statuses = vec![FastStatus::NotEvaluated; input.lanes];
    amplitude
        .evaluate(
            input,
            numerator,
            &mut HashMasters,
            &mut results,
            &mut statuses,
        )
        .unwrap();
    (results, statuses)
}

fn values(r: &FastResult) -> [C64; 5] {
    [
        r.laurent[0],
        r.laurent[1],
        r.laurent[2],
        r.cut_constructible,
        r.rational,
    ]
}

/// Scaled difference of two results (relative to the reference's largest
/// value, at least `floor`).
fn scaled(a: &FastResult, reference: &FastResult, floor: f64) -> f64 {
    let (a, b) = (values(a), values(reference));
    let scale = b.iter().map(|z| z.abs()).fold(floor, f64::max);
    a.iter()
        .zip(&b)
        .map(|(x, y)| (*x - *y).abs() / scale)
        .fold(0., f64::max)
}

/// A rank-`rank` tensor stored as a rank-`declared` one (zero top blocks).
fn padded(tensor: &TensorNumerator, declared: usize) -> TensorNumerator {
    let lanes = tensor.lanes();
    let mut out = TensorNumerator::new(declared, lanes);
    for lane in 0..lanes {
        out.lane_coefficients_mut(lane)[..tensor.ncoeff()]
            .copy_from_slice(tensor.lane_coefficients(lane));
    }
    out
}

#[test]
fn rank_n2_padding_reproduces_the_rank_n1_engine() {
    let mut worst = 0f64;
    for n in 1..=7 {
        for rank in [n, n + 1] {
            let mut rng = Rng(200 + 10 * n as u64 + rank as u64);
            let lanes = 3;
            let (shifts, masses) = kinematics(&mut rng, n, lanes);
            let scales = vec![1.7; lanes];
            let input = FastInput {
                lanes,
                shifts: &shifts,
                masses_squared: &masses,
                scales_squared: &scales,
                s_matrix: None,
            };
            let mut tensor = TensorNumerator::new(rank, lanes);
            for c in tensor.coefficients_mut() {
                *c = rng.c();
            }
            let mut wide = padded(&tensor, n + 2);
            let (reference, reference_statuses) = evaluate(n, rank, &input, &mut tensor);
            // With original Ninja's bubble references and tadpole bases, i.e.
            // the rank <= n+1 engine's conventions. The rank n+2 engine's own
            // choices (bases_x2.rs) move terms between the bubble and tadpole
            // coefficients, which only cancel with true master integrals (not
            // `HashMasters`): ninjars-symbolica's `fast_rank_n2.rs` and
            // `rank_n2_padding.rs` check them with oneloopmaster's masters.
            let (results, statuses) = with_ninja_choices(|| evaluate(n, n + 2, &input, &mut wide));
            assert_eq!(statuses, reference_statuses, "n={n} rank={rank}");
            for lane in 0..lanes {
                assert_eq!(statuses[lane], FastStatus::Success, "n={n} rank={rank}");
                let error = scaled(&results[lane], &reference[lane], 1e-300);
                assert!(
                    error <= 1e-11,
                    "n={n} rank {rank} declared n+2, lane {lane}: scaled error {error:e}"
                );
                worst = worst.max(error);
            }
        }
    }
    eprintln!("PADDING_X2 worst scaled error vs rank <= n+1 engine: {worst:e}");
}

/// `sum_k mu^(2k) T_k` with the blocks of `m` multiplied by
/// `D = (q + V)² - m² - mu²`: block `k` of the product is
/// `((q+V)² - m²) M_k - M_(k-1)`.
fn times_denominator(blocks: &[Vec<C64>], rank: usize, v: &RMom, m2: C64) -> Vec<Vec<C64>> {
    let k_max = blocks.len() - 1;
    let mut d: Vec<([u32; 4], C64)> = vec![
        ([2, 0, 0, 0], C64::ONE),
        ([0, 2, 0, 0], -C64::ONE),
        ([0, 0, 2, 0], -C64::ONE),
        ([0, 0, 0, 2], -C64::ONE),
        ([1, 0, 0, 0], C64::real(2. * v[0])),
        ([0, 1, 0, 0], C64::real(-2. * v[1])),
        ([0, 0, 1, 0], C64::real(-2. * v[2])),
        ([0, 0, 0, 1], C64::real(-2. * v[3])),
    ];
    d.push((
        [0; 4],
        C64::real(v[0] * v[0] - v[1] * v[1] - v[2] * v[2] - v[3] * v[3]) - m2,
    ));
    (0..=k_max + 1)
        .map(|k| {
            let out_rank = rank + 2 - 2 * k;
            let mut out = vec![C64::ZERO; ncoeff(out_rank)];
            if k <= k_max {
                let block_rank = rank - 2 * k;
                for (b, e) in monomials(block_rank).iter().enumerate() {
                    for &(de, dc) in &d {
                        let sum = std::array::from_fn(|i| e[i] + de[i]);
                        out[monomial_index(sum)] += dc * blocks[k][b];
                    }
                }
            }
            if k >= 1 {
                for (b, c) in blocks[k - 1].iter().enumerate() {
                    out[b] -= *c;
                }
            }
            out
        })
        .collect()
}

/// `I_n[D_j M]` against `I_(n-1)[M]` for a random `M` of rank `m_rank` with
/// every `mu²` power: the scaled difference per lane.
fn pinch_errors(n: usize, j: usize, m_rank: usize, seed: u64) -> Vec<f64> {
    let mut rng = Rng(seed);
    let lanes = 2;
    let (shifts, masses) = kinematics(&mut rng, n, lanes);
    let scales = vec![2.3; lanes];
    let mu_powers = m_rank / 2;
    let mut small = MuTensorNumerator::new(m_rank, mu_powers, lanes).unwrap();
    for c in small.coefficients_mut() {
        *c = rng.c();
    }
    let mut big = MuTensorNumerator::new(m_rank + 2, mu_powers + 1, lanes).unwrap();
    assert_eq!(big.ncoeff(), ncoeff_mu(m_rank + 2, mu_powers + 1));
    for lane in 0..lanes {
        let blocks: Vec<Vec<C64>> = (0..=mu_powers)
            .map(|k| small.lane_coefficients(lane)[small.block_range(k)].to_vec())
            .collect();
        let product =
            times_denominator(&blocks, m_rank, &shifts[lane * n + j], masses[lane * n + j]);
        for (k, block) in product.iter().enumerate() {
            let range = big.block_range(k);
            big.lane_coefficients_mut(lane)[range].copy_from_slice(block);
        }
    }
    let drop_j = |lane: &[RMom]| -> Vec<RMom> {
        lane.iter()
            .enumerate()
            .filter(|&(i, _)| i != j)
            .map(|(_, v)| *v)
            .collect()
    };
    let small_shifts: Vec<RMom> = shifts.chunks(n).flat_map(drop_j).collect();
    let small_masses: Vec<C64> = masses
        .chunks(n)
        .flat_map(|lane| {
            lane.iter()
                .enumerate()
                .filter(|&(i, _)| i != j)
                .map(|(_, v)| *v)
        })
        .collect();
    let input = FastInput {
        lanes,
        shifts: &shifts,
        masses_squared: &masses,
        scales_squared: &scales,
        s_matrix: None,
    };
    let small_input = FastInput {
        lanes,
        shifts: &small_shifts,
        masses_squared: &small_masses,
        scales_squared: &scales,
        s_matrix: None,
    };
    let (results, statuses) = evaluate(n, m_rank + 2, &input, &mut big);
    let (reference, reference_statuses) = evaluate(n - 1, m_rank, &small_input, &mut small);
    (0..lanes)
        .map(|lane| {
            assert_eq!(statuses[lane], FastStatus::Success, "n={n} j={j}");
            assert_eq!(reference_statuses[lane], FastStatus::Success, "n={n} j={j}");
            // A pinched massless tadpole integrates to exactly zero: the
            // floor makes that lane an absolute comparison.
            scaled(&results[lane], &reference[lane], 1.)
        })
        .collect()
}

#[test]
fn rank_n2_pinch_identity_with_mu_dependent_numerators() {
    // I_n[D_j M] = I_(n-1)[M] for rank(M) = n: the n-point rank n+2 reduction
    // against the (n-1)-point rank (n-1)+1 one, with explicit mu² powers.
    // The same identity one rank lower (rank(M) = n - 1, both sides in the
    // rank <= n+1 engine) on the same kinematics measures its conditioning.
    // Every residue of a cut containing j vanishes only up to rounding, and
    // at rank n+2 that noise reaches the deeper subtraction terms (up to
    // 1/(e.k)^5): rank n+2 may lose at most three more digits (observed:
    // ratios 1..30 typically, 789 at worst, on lanes whose rank n+1 control
    // is itself already ill-conditioned). Both sides use original Ninja's
    // bubble references and tadpole bases: the identity holds for any
    // choice only with true master integrals (see the padding test).
    let (mut worst, mut worst_ratio) = (0f64, 0f64);
    for n in 2..=6 {
        for j in 0..n {
            let seed = 300 + 10 * n as u64 + j as u64;
            let x2 = with_ninja_choices(|| pinch_errors(n, j, n, seed));
            let control = pinch_errors(n, j, n - 1, seed);
            for lane in 0..x2.len() {
                let bound = 1e-10f64.max(1000. * control[lane]);
                assert!(
                    x2[lane] <= bound,
                    "n={n} j={j} lane {lane}: scaled error {:e} (rank n+1 control {:e})",
                    x2[lane],
                    control[lane]
                );
                worst = worst.max(x2[lane]);
                worst_ratio = worst_ratio.max(x2[lane] / control[lane].max(1e-15));
            }
        }
    }
    eprintln!(
        "PINCH_X2 worst scaled error {worst:e}, worst ratio to the rank n+1 control {worst_ratio:.1}"
    );
}

#[test]
fn rank_n2_batches_and_replay_are_bitwise() {
    for n in 1..=7 {
        let rank = n + 2;
        let mut rng = Rng(400 + n as u64);
        let lanes = 4;
        let (shifts, masses) = kinematics(&mut rng, n, lanes);
        let scales = vec![1.5; lanes];
        let input = FastInput {
            lanes,
            shifts: &shifts,
            masses_squared: &masses,
            scales_squared: &scales,
            s_matrix: None,
        };
        let mut tensor = TensorNumerator::new(rank, lanes);
        for c in tensor.coefficients_mut() {
            *c = rng.c();
        }
        let mut amplitude = FastAmplitude::new(n, rank, FastOptions::default()).unwrap();
        // Width 1, lane by lane.
        let mut single = Vec::new();
        for lane in 0..lanes {
            let mut one = TensorNumerator::new(rank, 1);
            one.lane_coefficients_mut(0)
                .copy_from_slice(tensor.lane_coefficients(lane));
            let input = FastInput {
                lanes: 1,
                shifts: &shifts[lane * n..(lane + 1) * n],
                masses_squared: &masses[lane * n..(lane + 1) * n],
                scales_squared: &scales[lane..lane + 1],
                s_matrix: None,
            };
            let mut results = vec![FastResult::default(); 1];
            let mut statuses = vec![FastStatus::NotEvaluated; 1];
            amplitude
                .evaluate(
                    &input,
                    &mut one,
                    &mut HashMasters,
                    &mut results,
                    &mut statuses,
                )
                .unwrap();
            assert_eq!(statuses[0], FastStatus::Success, "n={n} lane {lane}");
            single.push(results[0]);
        }
        // The whole batch on the reused engine, recorded.
        let mut recording = RecordingNumerator::new(tensor);
        let mut masters = RecordingMasters::new(HashMasters);
        let mut results = vec![FastResult::default(); lanes];
        let mut statuses = vec![FastStatus::NotEvaluated; lanes];
        amplitude
            .evaluate(
                &input,
                &mut recording,
                &mut masters,
                &mut results,
                &mut statuses,
            )
            .unwrap();
        assert_eq!(results, single, "n={n}: batch differs from width 1");
        let mut replay_numerator = recording.replay();
        let mut replay_masters = masters.replay();
        for _ in 0..2 {
            let mut replayed = vec![FastResult::default(); lanes];
            amplitude
                .evaluate(
                    &input,
                    &mut replay_numerator,
                    &mut replay_masters,
                    &mut replayed,
                    &mut statuses,
                )
                .unwrap();
            assert_eq!(replayed, single, "n={n}: replay differs");
        }
    }
}

#[test]
fn rank_n2_lane_statuses_isolate_failures() {
    let n = 4;
    let rank = 6;
    let regular: Vec<RMom> = vec![
        [0., 0., 0., 0.],
        [1.902, -0.517, -0.537, -0.876],
        [-1.599, 0.29, 0.378, 0.142],
        [-0.892, -0.823, 0.499, 0.629],
    ];
    let collinear: Vec<RMom> = vec![
        [0., 0., 0., 0.],
        [1., 0., 0., 0.],
        [2., 0., 0., 0.],
        [3., 0.5, 0., 0.],
    ];
    let lanes = 5;
    let mut shifts = Vec::new();
    for lane in 0..lanes {
        shifts.extend_from_slice(if lane == 1 { &collinear } else { &regular });
    }
    let masses = vec![C64::new(0.7, -0.05); lanes * n];
    let scales = [1., 1., 7., 9., 2.];
    let input = FastInput {
        lanes,
        shifts: &shifts,
        masses_squared: &masses,
        scales_squared: &scales,
        s_matrix: None,
    };
    let mut amplitude = FastAmplitude::new(n, rank, FastOptions::default()).unwrap();
    assert_eq!(amplitude.rank(), rank);
    assert_eq!(amplitude.propagators(), n);
    let sentinel = FastResult {
        laurent: [C64::real(42.); 3],
        cut_constructible: C64::real(42.),
        rational: C64::real(42.),
    };
    let mut results = vec![sentinel; lanes];
    let mut statuses = vec![FastStatus::NotEvaluated; lanes];
    let mut tensor = unit_tensor(rank, lanes);
    amplitude
        .evaluate(
            &input,
            &mut tensor,
            &mut UnitMasters,
            &mut results,
            &mut statuses,
        )
        .unwrap();
    assert_eq!(
        statuses,
        [
            FastStatus::Success,
            FastStatus::Unstable,
            FastStatus::MasterFailed,
            FastStatus::NonFinite,
            FastStatus::Success
        ]
    );
    for (lane, result) in results.iter().enumerate().take(4).skip(1) {
        assert_eq!(*result, sentinel, "failed lane {lane} must keep its result");
    }
    let finite = results[0];
    assert!((finite.cut_constructible + finite.rational - finite.laurent[0]).abs() < 1e-12);
    // Structural errors: shapes and the numerator rank.
    let before = results.clone();
    assert!(
        amplitude
            .evaluate(
                &FastInput {
                    lanes,
                    shifts: &shifts[..lanes * n - 1],
                    ..input
                },
                &mut tensor,
                &mut UnitMasters,
                &mut results,
                &mut statuses,
            )
            .is_err()
    );
    assert_eq!(results, before);
    let mut wrong_rank = unit_tensor(rank - 1, lanes);
    assert!(
        amplitude
            .evaluate(
                &input,
                &mut wrong_rank,
                &mut UnitMasters,
                &mut results,
                &mut statuses
            )
            .is_err()
    );
}

/// A tensor numerator without the rank n+2 box expansion (the trait default).
struct NoMuShifted(TensorNumerator);

impl FastNumerator for NoMuShifted {
    fn rank(&self) -> usize {
        self.0.rank()
    }
    fn evaluate(
        &mut self,
        cut: &[usize],
        lanes: &[usize],
        q: &[CMom],
        mu_squared: &[C64],
        out: &mut [C64],
    ) -> Result<()> {
        self.0.evaluate(cut, lanes, q, mu_squared, out)
    }
    fn mu_expansion(
        &mut self,
        cut: &[usize],
        lanes: &[usize],
        v0: &[CMom],
        v1: Option<&[CMom]>,
        out: &mut [C64],
    ) -> Result<()> {
        self.0.mu_expansion(cut, lanes, v0, v1, out)
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
        self.0
            .t3_expansion(cut, lanes, a, e3, e4, param, mindeg, out)
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
        self.0
            .t2_expansion(cut, lanes, a0, a1, e3, e4, param, mindeg, out)
    }
    fn t3_expansion_x2(
        &mut self,
        cut: &[usize],
        lanes: &[usize],
        a: &[CMom],
        e3: &[CMom],
        e4: &[CMom],
        param: &[C64],
        out: &mut [C64],
    ) -> Result<()> {
        self.0.t3_expansion_x2(cut, lanes, a, e3, e4, param, out)
    }
    fn t2_expansion_x2(
        &mut self,
        cut: &[usize],
        lanes: &[usize],
        a0: &[CMom],
        a1: &[CMom],
        e3: &[CMom],
        e4: &[CMom],
        param: &[[C64; 3]],
        out: &mut [C64],
    ) -> Result<()> {
        self.0
            .t2_expansion_x2(cut, lanes, a0, a1, e3, e4, param, out)
    }
}

#[test]
fn rank_n2_construction_and_numerator_requirements() {
    for n in 1..=8 {
        assert!(FastAmplitude::new(n, n + 2, FastOptions::default()).is_ok());
        assert!(matches!(
            FastAmplitude::new(n, n + 3, FastOptions::default()),
            Err(Error::Rank { .. })
        ));
    }
    // Boxes need `mu_expansion_x2`; without boxes the default is never called.
    for (n, needs) in [(3, false), (4, true), (5, true)] {
        let mut rng = Rng(500 + n as u64);
        let (shifts, masses) = kinematics(&mut rng, n, 1);
        let scales = [1.];
        let input = FastInput {
            lanes: 1,
            shifts: &shifts,
            masses_squared: &masses,
            scales_squared: &scales,
            s_matrix: None,
        };
        let mut numerator = NoMuShifted(unit_tensor(n + 2, 1));
        let mut amplitude = FastAmplitude::new(n, n + 2, FastOptions::default()).unwrap();
        let mut results = [FastResult::default()];
        let mut statuses = [FastStatus::NotEvaluated];
        let outcome = amplitude.evaluate(
            &input,
            &mut numerator,
            &mut HashMasters,
            &mut results,
            &mut statuses,
        );
        assert_eq!(
            matches!(outcome, Err(Error::Unsupported(_))),
            needs,
            "n={n}: {outcome:?}"
        );
    }
}

// ----------------------------------------------- residues on the cuts

/// Canonical rank n+2 triangle monomials `(mu² power, (q̄.e3) power,
/// (q̄.e4) power)` (design 1.2).
const TRIANGLE_MONOMIALS: [(i32, i32, i32); 21] = [
    (0, 0, 0),
    (0, 1, 0),
    (0, 2, 0),
    (0, 3, 0),
    (0, 0, 1),
    (0, 0, 2),
    (0, 0, 3),
    (1, 0, 0),
    (1, 1, 0),
    (1, 0, 1),
    (1, 2, 0),
    (1, 0, 2),
    (0, 4, 0),
    (0, 0, 4),
    (2, 0, 0),
    (0, 5, 0),
    (0, 0, 5),
    (1, 3, 0),
    (1, 0, 3),
    (2, 1, 0),
    (2, 0, 1),
];

/// Canonical rank n+2 bubble monomials `(mu² power, (q̄.e2), (q̄.e3), (q̄.e4)
/// powers)` (design 1.2, `tools/rank_n2/theory/formulas_rank_n2.json`).
const BUBBLE_MONOMIALS: [(i32, i32, i32, i32); 35] = [
    (0, 0, 0, 0),
    (0, 1, 0, 0),
    (0, 2, 0, 0),
    (0, 0, 1, 0),
    (0, 0, 2, 0),
    (0, 0, 0, 1),
    (0, 0, 0, 2),
    (0, 1, 1, 0),
    (0, 1, 0, 1),
    (1, 0, 0, 0),
    (1, 1, 0, 0),
    (1, 0, 1, 0),
    (1, 0, 0, 1),
    (0, 3, 0, 0),
    (0, 0, 3, 0),
    (0, 0, 0, 3),
    (0, 2, 1, 0),
    (0, 2, 0, 1),
    (0, 1, 2, 0),
    (0, 1, 0, 2),
    (0, 4, 0, 0),
    (1, 2, 0, 0),
    (2, 0, 0, 0),
    (0, 0, 4, 0),
    (0, 0, 0, 4),
    (0, 1, 3, 0),
    (0, 1, 0, 3),
    (0, 2, 2, 0),
    (0, 2, 0, 2),
    (0, 3, 1, 0),
    (0, 3, 0, 1),
    (1, 0, 2, 0),
    (1, 0, 0, 2),
    (1, 1, 1, 0),
    (1, 1, 0, 1),
];

fn powi(z: C64, k: i32) -> C64 {
    (0..k).fold(C64::ONE, |acc, _| acc * z)
}

fn shifted(q: &CMom, v: &RMom) -> CMom {
    [q[0] + v[0], q[1] + v[1], q[2] + v[2], q[3] + v[3]]
}

/// `(value, sum of |terms|)` of a triangle residue at `q̄`.
fn triangle_value(c: &[C64; 21], e3: &CMom, e4: &CMom, qbar: &CMom, mu2: C64) -> (C64, f64) {
    let (u, v) = (mp_cc(qbar, e3), mp_cc(qbar, e4));
    let mut total = (C64::ZERO, 0.);
    for (k, &(j, a3, a4)) in TRIANGLE_MONOMIALS.iter().enumerate() {
        let term = c[k] * powi(mu2, j) * powi(u, a3) * powi(v, a4);
        total = (total.0 + term, total.1 + term.abs());
    }
    total
}

/// `(value, sum of |terms|)` of a bubble residue at `q̄`.
fn bubble_value(
    c: &[C64; 35],
    e2: &RMom,
    e3: &CMom,
    e4: &CMom,
    qbar: &CMom,
    mu2: C64,
) -> (C64, f64) {
    let w = super::complex::mp_cr(qbar, e2);
    let (u, v) = (mp_cc(qbar, e3), mp_cc(qbar, e4));
    let mut total = (C64::ZERO, 0.);
    for (k, &(j, a2, a3, a4)) in BUBBLE_MONOMIALS.iter().enumerate() {
        let term = c[k] * powi(mu2, j) * powi(w, a2) * powi(u, a3) * powi(v, a4);
        total = (total.0 + term, total.1 + term.abs());
    }
    total
}

fn propagator(q: &CMom, v: &RMom, m2: C64, mu2: C64) -> C64 {
    let s = shifted(q, v);
    mp_cc(&s, &s) - m2 - mu2
}

/// One lane (all lines massive) and a random rank n+2 numerator with every
/// mu² power, reduced by the rank n+2 engine.
fn reduced_lane(n: usize, seed: u64) -> (FastAmplitude, MuTensorNumerator, Vec<RMom>, Vec<C64>) {
    let mut rng = Rng(seed);
    let (shifts, masses) = kinematics(&mut rng, n, 1);
    let rank = n + 2;
    let mut numerator = MuTensorNumerator::new(rank, rank / 2, 1).unwrap();
    for c in numerator.coefficients_mut() {
        *c = rng.c();
    }
    let mut amplitude = FastAmplitude::new(n, rank, FastOptions::default()).unwrap();
    let input = FastInput {
        lanes: 1,
        shifts: &shifts,
        masses_squared: &masses,
        scales_squared: &[1.9],
        s_matrix: None,
    };
    let mut results = [FastResult::default()];
    let mut statuses = [FastStatus::NotEvaluated];
    amplitude
        .evaluate(
            &input,
            &mut numerator,
            &mut HashMasters,
            &mut results,
            &mut statuses,
        )
        .unwrap();
    assert_eq!(statuses[0], FastStatus::Success);
    (amplitude, numerator, shifts, masses)
}

fn numerator_at(numerator: &mut MuTensorNumerator, cut: &[usize], q: &CMom, mu2: C64) -> C64 {
    let mut out = [C64::ZERO];
    numerator
        .evaluate(cut, &[0], &[*q], &[mu2], &mut out)
        .unwrap();
    out[0]
}

#[test]
fn rank_n2_triangle_and_bubble_residues_reconstruct_the_integrand() {
    use super::bases_x2::bubble_basis_x2;
    use super::geometry::{Basis, bubble_solution, combine_real2, sub, triangle_solution};
    let eps = FastOptions::default().infrared_eps;
    let mut worst = 0f64;
    let mut replaced = 0;
    for (n, seed) in [(2, 601), (2, 602), (3, 603), (3, 604), (3, 605)] {
        let (amplitude, mut numerator, v, m) = reduced_lane(n, seed);
        let engine = amplitude.x2().unwrap();
        let mut rng = Rng(seed + 1000);
        // Triangle (n = 3): N = Delta_T on the cut, with mu² dependence.
        let triangle = if n == 3 {
            let k0 = sub(&v[1], &v[0]);
            let k2 = sub(&v[0], &v[2]);
            let e = Basis::from_pair(&k2, &k0, eps);
            let (y1, y2, param) = triangle_solution(&k2, &k0, &e, m[0], m[1], m[2]);
            let a = combine_real2(y1, &e.e1, y2, &e.e2);
            let (e3, e4, c) = engine.triangle_residue(0, 0);
            assert_eq!((e3, e4), (e.e3, e.e4));
            Some((a, param, e, c))
        } else {
            None
        };
        if let Some((a, param, e, c)) = &triangle {
            for _ in 0..6 {
                let (t, mu2) = (rng.c() * 2., rng.c());
                let s = (*param + mu2) / t / (2. * e.mp34());
                let q: CMom =
                    std::array::from_fn(|mu| a[mu] - v[0][mu] + t * e.e3[mu] + s * e.e4[mu]);
                for i in 0..3 {
                    assert!(propagator(&q, &v[i], m[i], mu2).abs() < 1e-10);
                }
                let value = numerator_at(&mut numerator, &[0, 1, 2], &q, mu2);
                let (residue, size) = triangle_value(c, &e.e3, &e.e4, &shifted(&q, &v[0]), mu2);
                let error = (value - residue).abs() / size;
                assert!(error < 1e-11, "n={n} triangle: {error:e}");
                worst = worst.max(error);
            }
        }
        // Bubbles: N / D_k - Delta_T / D_k = Delta_B on the cut (n = 3, one
        // uncut line k), N = Delta_B (n = 2). All 35 coefficients, with
        // both orientations (massive lines).
        let bubbles: &[([usize; 2], Option<usize>)] = if n == 2 {
            &[([0, 1], None)]
        } else {
            &[([0, 1], Some(2)), ([0, 2], Some(1)), ([1, 2], Some(0))]
        };
        for (b, &([c1, c2], uncut)) in bubbles.iter().enumerate() {
            let k = sub(&v[c1], &v[c2]);
            // The engine's conditioned reference (bases_x2.rs).
            let lines: Vec<usize> = uncut.into_iter().collect();
            let e = bubble_basis_x2(&k, &v, c1, &lines, eps);
            replaced += usize::from(e.e2 != Basis::from_single(&k, eps).e2);
            let (b0, b1, p) = bubble_solution(&k, &e, m[c1], m[c2]);
            let (e2, e3, e4, c) = engine.bubble_residue(b, 0);
            assert_eq!((e2, e3, e4), (e.e2, e.e3, e.e4));
            for _ in 0..6 {
                let (t, x, mu2) = (rng.c() * 2., rng.c(), rng.c());
                let s = (p[0] + p[1] * x + p[2] * x * x + mu2) / t / (2. * e.mp34());
                let q: CMom = std::array::from_fn(|mu| {
                    b0 * e.e2[mu] - v[c1][mu]
                        + x * (e.e1[mu] + b1 * e.e2[mu])
                        + t * e.e3[mu]
                        + s * e.e4[mu]
                });
                assert!(propagator(&q, &v[c1], m[c1], mu2).abs() < 1e-10);
                assert!(propagator(&q, &v[c2], m[c2], mu2).abs() < 1e-10);
                let mut value = numerator_at(&mut numerator, &[c1, c2], &q, mu2);
                let mut size = value.abs();
                if let Some(l) = uncut {
                    let d = propagator(&q, &v[l], m[l], mu2);
                    let (_, _, _, tc) = triangle.as_ref().unwrap();
                    let (te3, te4, _) = engine.triangle_residue(0, 0);
                    let (parent, parent_size) =
                        triangle_value(tc, &te3, &te4, &shifted(&q, &v[0]), mu2);
                    value = (value - parent) / d;
                    size = (size + parent_size) / d.abs();
                }
                let (residue, residue_size) =
                    bubble_value(&c, &e2, &e3, &e4, &shifted(&q, &v[c1]), mu2);
                let error = (value - residue).abs() / size.max(residue_size);
                assert!(error < 1e-11, "n={n} bubble {b}: {error:e}");
                worst = worst.max(error);
            }
        }
    }
    // The conditioned references (bases_x2.rs) are exercised: their residues
    // reconstruct the integrand like Ninja's.
    assert!(
        replaced > 0,
        "no bubble replaced original Ninja's reference"
    );
    eprintln!(
        "RESIDUES_X2 worst relative error {worst:e}, {replaced} bubbles with a conditioned reference"
    );
}

#[test]
fn rank_n2_box_mu_expansion_matches_finite_mu_samples() {
    // n = 4: on the box cut at mu² = s the even part of N is exactly
    // c0 + c2 s + c4 s² + c6 s³, so four samples fix c4 and c6
    // independently of the d-dimensional mu-expansion (with its beta offset
    // for mu²-dependent numerators).
    use super::geometry::{Basis, box_solutions, sub};
    let eps = FastOptions::default().infrared_eps;
    let mut worst = 0f64;
    for seed in [701, 702, 703, 704] {
        let (amplitude, mut numerator, v, m) = reduced_lane(4, seed);
        let engine = amplitude.x2().unwrap();
        let [c0, c4, c6] = engine.box_coefficients(0, 0);
        let k0 = sub(&v[1], &v[0]);
        let k1 = sub(&v[2], &v[1]);
        let k3 = sub(&v[0], &v[3]);
        let e = Basis::from_pair(&k3, &k0, eps);
        let samples = [0., 0.35, -0.45, 0.8];
        let even: Vec<C64> = samples
            .iter()
            .map(|&s| {
                let mu2 = C64::real(s);
                let l = box_solutions(
                    &k3,
                    &k0,
                    &k1,
                    &e,
                    m[0] + mu2,
                    m[1] + mu2,
                    m[2] + mu2,
                    m[3] + mu2,
                );
                let mut total = C64::ZERO;
                for sol in l {
                    let q: CMom = std::array::from_fn(|mu| sol[mu] - v[0][mu]);
                    for i in 0..4 {
                        assert!(propagator(&q, &v[i], m[i], mu2).abs() < 1e-9);
                    }
                    total += numerator_at(&mut numerator, &[0, 1, 2, 3], &q, mu2) * 0.5;
                }
                total
            })
            .collect();
        // Newton divided differences on the cubic through the samples.
        let mut coef = even.clone();
        for level in 1..4 {
            for i in (level..4).rev() {
                coef[i] = (coef[i] - coef[i - 1]) / (samples[i] - samples[i - level]);
            }
        }
        // c6 is the leading divided difference; c4 follows from expanding
        // the Newton form (s - 0)(s - s1)(s - s2) at s².
        let fit_c6 = coef[3];
        let fit_c4 = coef[2] - coef[3] * (samples[1] + samples[2]);
        let scale = even.iter().map(|z| z.abs()).fold(0., f64::max);
        let errors = [
            (c0 - even[0]).abs() / scale,
            (c4 - fit_c4).abs() / scale,
            (c6 - fit_c6).abs() / scale,
        ];
        for (name, error) in ["c0", "c4", "c6"].iter().zip(errors) {
            assert!(
                error < 1e-9,
                "seed {seed}: {name} {error:e} ({c4:?} vs {fit_c4:?}, {c6:?} vs {fit_c6:?})"
            );
            worst = worst.max(error);
        }
        assert!(
            c6.abs() > 1e-6 * scale,
            "the test needs a non-zero mu⁶ term"
        );
    }
    eprintln!("BOX_X2 worst scaled error {worst:e}");
}

#[test]
fn rank_n2_tadpole_integrates_rank_three_numerators() {
    // n = 1, rank 3: with q = q̄ - V and odd q̄ integrals vanishing,
    // int (q.a)³/D = -3 (V.a) a² A00 - (V.a)³ A0 with A00 = m² (A0 + m²/2)/4,
    // and int mu² (q.a)/D = -(V.a) m⁴/2 (I1[mu²] = m⁴/2, m² the squared mass).
    // Coefficients of the A0 value and the rational part must match for any
    // A0, so the pseudo-random masters suffice.
    let mut rng = Rng(801);
    for _ in 0..4 {
        let v = [rng.uniform(), rng.uniform(), rng.uniform(), rng.uniform()];
        let a = rng.cmom();
        let m2 = C64::new(0.5 + rng.uniform().abs(), -0.1 * rng.uniform().abs());
        let mu_r2 = 1.3;
        // N = (q.a)³ + mu² (q.a) as a rank-3 numerator with one mu² block.
        let mut numerator = MuTensorNumerator::new(3, 1, 1).unwrap();
        let mut cubic = vec![C64::ZERO; ncoeff(3)];
        let metric = [1., -1., -1., -1.];
        for e in monomials(3) {
            if e.iter().sum::<u32>() != 3 {
                continue;
            }
            // Multinomial coefficient of q^e in (sum_mu g_mu a_mu q_mu)³.
            let fact = |k: u32| (1..=k).product::<u32>() as f64;
            let multinomial = 6. / (fact(e[0]) * fact(e[1]) * fact(e[2]) * fact(e[3]));
            let mut c = C64::real(multinomial);
            for mu in 0..4 {
                c *= powi(a[mu] * metric[mu], e[mu] as i32);
            }
            cubic[monomial_index(e)] = c;
        }
        let mut linear_block = vec![C64::ZERO; ncoeff(1)];
        for mu in 0..4 {
            let mut e = [0u32; 4];
            e[mu] = 1;
            linear_block[monomial_index(e)] = a[mu] * metric[mu];
        }
        let lane = numerator.lane_coefficients_mut(0);
        lane[..ncoeff(3)].copy_from_slice(&cubic);
        lane[ncoeff(3)..].copy_from_slice(&linear_block);
        let input = FastInput {
            lanes: 1,
            shifts: &[v],
            masses_squared: &[m2],
            scales_squared: &[mu_r2],
            s_matrix: None,
        };
        let (results, statuses) = evaluate(1, 3, &input, &mut numerator);
        assert_eq!(statuses[0], FastStatus::Success);
        let mut a0 = [[C64::ZERO; 3]];
        HashMasters
            .evaluate(
                FastMaster::A0,
                &[m2, C64::real(mu_r2)],
                &mut a0,
                &mut [false],
            )
            .unwrap();
        let va = a[0] * v[0] - a[1] * v[1] - a[2] * v[2] - a[3] * v[3];
        let aa = mp_cc(&a, &a);
        let x = -(va * aa * 3. * m2 * 0.25) - va * va * va;
        let y = -(va * aa * 3. * m2 * m2 * 0.125) - va * m2 * m2 * 0.5;
        let expected = [x * a0[0][0] + y, x * a0[0][1], x * a0[0][2]];
        assert_close(&results[0].laurent, &expected, "rank-3 tadpole");
        assert_close(&[results[0].rational], &[y], "rank-3 tadpole rational");
    }
}
