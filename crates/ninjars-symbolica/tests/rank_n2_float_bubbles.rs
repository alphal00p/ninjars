//! The generic engine in `Float(256)` at rank n+2 on bubbles with
//! `|p²| << m²`, where B1111 comes from oneloopmaster's native family.
//!
//! Lanes: n = 2 (rank 4, with an exact S-matrix) and n = 3 (rank 5, a
//! triangle whose invariants the engine computes from the shifts, so `p²`
//! carries the tracked precision lost to its own cancellation) with
//! `p² = 2^(1-k) + 2^-2k`, k = 8 .. 47 (|p²|/m² down to 1e-14), for real
//! unequal, equal, complex and one-massless mass sets, and dense random
//! numerators. Reference: the same engine in `Float(512)`.
//!
//! - With B0..B111 (and the triangle's C0) taken from 512 bits, the
//!   `Float(256)` engine with the provider's native B1111 agrees with the
//!   reference to 1e-68 on every lane (measured <= 1.4e-73): B1111 no longer
//!   limits the accuracy at small `p²`. The exact B1111 that `Float` used
//!   before limited the same lanes to 1.3e-11 (with the S-matrix) and
//!   O(100) relative errors (without) at `|p²|/m² = 1.4e-14`.
//! - With the provider alone, the existing exact Passarino-Veltman B0, B1,
//!   B11 and B111 still lose digits at small `p²`: about 1, 2, 3 and 4
//!   digits per decade of `m²/|p²|` against the Feynman-parameter values of
//!   `rank_n2_data/b_family_feynman.txt`, in `Float(256)` and `Float(512)`
//!   alike (their tracked precision follows the loss to within 2^15). The
//!   lanes reach 1e-60 for k <= 12 (`|p²|/m² >= 2^-12`, asserted) and lose
//!   about 4-5 digits per decade below (reported; at k = 47, 7e-24 for
//!   n = 2, where B111 alone from 512 bits gives 3e-37, and 2e-10 for n = 3,
//!   whose `p²` enters the interpreter with fewer tracked bits).
//!
//! Run with `cargo test --release -p ninjars-symbolica --test rank_n2_float_bubbles`.

#[path = "../../ninjars-cli/src/case_check/harness.rs"]
mod harness;

use harness::{BenchCase, BenchLane, GenericEngine, GenericLane};
use ninjars::{
    Complex, Float, FloatLike, Laurent, MasterIntegral, MasterProvider, MasterRequest,
    MasterStatus, Result,
    fast::{C64, tensor_layout::ncoeff},
};
use ninjars_symbolica::SymbolicaMasters;
use symbolica::domains::float::RealLike;

/// Exponents `k` of `p² = 2^(1-k) + 2^-2k` (the shift `(1 + 2^-k, 0, 0, 1)`).
const LADDER: [i32; 7] = [8, 12, 16, 24, 32, 40, 47];

/// The shipped provider must reach 1e-60 for `k <= ASSERTED_K`.
const ASSERTED_K: i32 = 12;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// A dyadic number in [-1, 1) (exact in every carrier).
    fn uniform(&mut self) -> f64 {
        ((self.next() >> 40) as f64 / (1u64 << 24) as f64) * 2. - 1.
    }
}

/// The lanes of one `n` (2: with S-matrices; 3: invariants from the shifts),
/// at rank `n + 2`, and each lane's `k`.
fn case(n: usize) -> (BenchCase, Vec<i32>) {
    let rank = n + 2;
    let mut rng = Rng(20_261_018 + n as u64);
    let mass_sets = [
        [C64::new(1., 0.), C64::new(1.5, 0.), C64::new(2., 0.)],
        [C64::new(1., 0.), C64::new(1., 0.), C64::new(1., 0.)],
        [
            C64::new(0.7, -0.03),
            C64::new(1.4, -0.08),
            C64::new(1.1, -0.05),
        ],
        [C64::new(0., 0.), C64::new(2., 0.), C64::new(1.5, 0.)],
    ];
    let mut lanes = Vec::new();
    let mut ks = Vec::new();
    for k in LADDER {
        for (set, masses) in mass_sets.iter().enumerate() {
            let d = 0.5f64.powi(k);
            let mut shifts = vec![[0., 0., 0., 0.], [1. + d, 0., 0., 1.]];
            if n == 3 {
                shifts.push([0.5, 0.25, -0.375, 0.125]);
            }
            let p2 = 2. * d + d * d;
            lanes.push(BenchLane {
                id: format!("n{n}-k{k}-m{set}"),
                shifts,
                masses: masses[..n].to_vec(),
                mu2: 1.,
                s_matrix: (n == 2).then(|| vec![0., p2, p2, 0.]),
                tensor: (0..ncoeff(rank))
                    .map(|_| C64::new(rng.uniform(), rng.uniform()))
                    .collect(),
            });
            ks.push(k);
        }
    }
    let case = BenchCase {
        name: format!("float-bubbles-n{n}-r{rank}"),
        path: "rank_n2_float_bubbles".into(),
        n,
        rank,
        mu_powers: 0,
        lanes,
    };
    (case, ks)
}

fn rounded(value: &Float, bits: u32) -> Float {
    let mut value = value.clone();
    value.set_prec(bits);
    value
}

/// B1111 from the `Float(256)` provider under test; every other master from
/// a `Float(512)` provider (inputs taken exactly, results rounded to 256 bits).
struct AccurateLowerMasters {
    tested: SymbolicaMasters<Float>,
    high: SymbolicaMasters<Float>,
}

impl MasterProvider<Complex<Float>> for AccurateLowerMasters {
    fn evaluate(
        &mut self,
        request: MasterRequest<'_, Complex<Float>>,
        output: &mut [Laurent<Complex<Float>>],
    ) -> Result<()> {
        let mut statuses = vec![MasterStatus::Success; output.len()];
        self.evaluate_batch(request, output, &mut statuses)
    }

    fn evaluate_batch(
        &mut self,
        request: MasterRequest<'_, Complex<Float>>,
        output: &mut [Laurent<Complex<Float>>],
        statuses: &mut [MasterStatus],
    ) -> Result<()> {
        if request.integral == MasterIntegral::B1111 {
            return self.tested.evaluate_batch(request, output, statuses);
        }
        let up = |z: &Complex<Float>| Complex::new(rounded(&z.re, 512), rounded(&z.im, 512));
        let invariants: Vec<_> = request.invariants.iter().map(up).collect();
        let masses: Vec<_> = request.masses_squared.iter().map(up).collect();
        let scales: Vec<_> = request.scales_squared.iter().map(up).collect();
        let zero = Complex::new(Float::new(512), Float::new(512));
        let mut high = vec![[zero.clone(), zero.clone(), zero]; output.len()];
        self.high.evaluate_batch(
            MasterRequest {
                integral: request.integral,
                invariants: &invariants,
                masses_squared: &masses,
                scales_squared: &scales,
            },
            &mut high,
            statuses,
        )?;
        for (output, high) in output.iter_mut().zip(high) {
            for (output, high) in output.iter_mut().zip(high) {
                *output = Complex::new(rounded(&high.re, 256), rounded(&high.im, 256));
            }
        }
        Ok(())
    }
}

/// Largest difference of the Laurent coefficients, relative to the largest
/// reference coefficient.
fn scaled_difference(values: &GenericLane<Float>, reference: &GenericLane<Float>) -> f64 {
    let abs = |z: &Complex<Float>| z.re.to_f64().hypot(z.im.to_f64());
    let scale = reference.values[..3].iter().map(abs).fold(0., f64::max);
    values.values[..3]
        .iter()
        .zip(&reference.values[..3])
        .map(|(value, reference)| {
            abs(&Complex::new(
                value.re.clone() - &reference.re,
                value.im.clone() - &reference.im,
            ))
        })
        .fold(0., f64::max)
        / scale
}

#[test]
fn rank_n2_bubbles_with_small_momentum_in_float256() {
    let mut low = GenericEngine::new(Float::new(256)).unwrap();
    let mut high = GenericEngine::new(Float::new(512)).unwrap();
    for n in [2, 3] {
        let (case, ks) = case(n);
        let reference = high
            .evaluate(
                &case,
                case.rank,
                &mut SymbolicaMasters::new(Float::new(512)),
            )
            .unwrap();
        let provider = low
            .evaluate(
                &case,
                case.rank,
                &mut SymbolicaMasters::new(Float::new(256)),
            )
            .unwrap();
        let accurate_lower = low
            .evaluate(
                &case,
                case.rank,
                &mut AccurateLowerMasters {
                    tested: SymbolicaMasters::new(Float::new(256)),
                    high: SymbolicaMasters::new(Float::new(512)),
                },
            )
            .unwrap();
        let mut worst = (0f64, String::new());
        for (((lane, k), reference), (provider, accurate_lower)) in case
            .lanes
            .iter()
            .zip(&ks)
            .zip(&reference)
            .zip(provider.iter().zip(&accurate_lower))
        {
            for result in [reference, provider, accurate_lower] {
                assert_eq!(
                    result.status,
                    ninjars::EvaluationStatus::Success,
                    "{}",
                    lane.id
                );
            }
            let isolated = scaled_difference(accurate_lower, reference);
            let shipped = scaled_difference(provider, reference);
            assert!(isolated <= 1e-68, "{}: {isolated:e}", lane.id);
            if *k <= ASSERTED_K {
                assert!(shipped <= 1e-60, "{}: {shipped:e}", lane.id);
            }
            if isolated > worst.0 {
                worst = (isolated, lane.id.clone());
            }
            eprintln!(
                "{:14} |p2|/m2 ~ {:.1e}: native B1111, B0..B111 from 512 bits {isolated:.1e}; provider alone {shipped:.1e}{} (eps0 tracked at {} bits)",
                lane.id,
                0.5f64.powi(k - 1) / lane.masses.iter().map(|m| m.abs()).fold(0., f64::max),
                if *k <= ASSERTED_K {
                    ""
                } else {
                    " (report only)"
                },
                provider.values[0].re.get_precision(),
            );
        }
        eprintln!(
            "n = {n}: worst with accurate B0..B111 {:.1e} ({})",
            worst.0, worst.1
        );
    }
}
