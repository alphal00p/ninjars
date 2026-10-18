//! Rank n+2 boxes in the generic engine (V2c).
//!
//! - The three-term mu-expansion on the exact d-dimensional cut
//!   (`ExpansionKind::MuShifted`, `mu² = t² v² - beta`) and mu² sampling at
//!   `0, s, -s, 2s` give the same box coefficients in every mode: full
//!   expansion, full sampling, integrated expansion and integrated sampling
//!   (n = 4..6, numerators with and without explicit mu² terms; Float(256) to
//!   <= 1e-60, binary64 to <= 1e-8 relative to the largest coefficient), and
//!   the same integrated amplitudes.
//! - Integrated mode computes exactly c0, c4 and c6.
//! - Control: answering the box expansion with Ninja's unshifted
//!   `mu² = t² v²` (`ExpansionKind::Mu`, beta dropped) leaves q-only
//!   numerators unchanged but gives a wrong c4 for numerators with explicit
//!   mu² terms (review §3), so the offset is required.

mod rank_n2_support;

use ninjars::{
    BoxCoefficientMode, BoxMuMode, BoxReductionOptions, Complex, CutOutcome, ExpansionKind,
    ExpansionRequest, Float, Numerator, NumeratorRequest, Real,
};
use ninjars_symbolica::PolynomialNumerator;
use oneloop::NativeFloat;
use rank_n2_support::{Problem, ZeroMasters, distance, laurent_error, modulus, options, reduce};
use symbolica::evaluate::EvaluationDomain;

const MODES: [(BoxCoefficientMode, BoxMuMode); 4] = [
    (BoxCoefficientMode::Full, BoxMuMode::Expansion),
    (BoxCoefficientMode::Full, BoxMuMode::Sampling),
    (BoxCoefficientMode::Integrated, BoxMuMode::Expansion),
    (BoxCoefficientMode::Integrated, BoxMuMode::Sampling),
];

type Boxes<T> = Vec<(Vec<Complex<T>>, Vec<bool>)>;

fn boxes<T, N>(
    problem: &Problem,
    numerator: &mut N,
    prototype: &T,
    mode: BoxReductionOptions,
) -> (Boxes<T>, Vec<[Complex<T>; 5]>)
where
    T: NativeFloat,
    Complex<T>: Real + EvaluationDomain,
    N: Numerator<Complex<T>>,
{
    let amplitude = reduce(problem, numerator, prototype, mode, &mut ZeroMasters);
    let boxes = amplitude
        .boxes()
        .iter()
        .map(|outcome| {
            let CutOutcome::Stable(r) = outcome else {
                panic!("unstable box")
            };
            (r.coefficients.clone(), r.computed.clone())
        })
        .collect();
    (boxes, amplitude.results().to_vec())
}

/// Worst coefficient difference between modes, relative to each box's
/// largest coefficient, and worst integrated difference.
fn modes_agree<T>(n: usize, mu_terms: bool, prototype: T) -> (f64, f64)
where
    T: NativeFloat + Into<Float>,
    Complex<T>: Real + EvaluationDomain,
{
    let problem = Problem::new(n, n + 2, 0xb0c5 + n as u64, mu_terms, stride(n));
    let mut numerator = problem.numerator(&prototype);
    let (reference, integrated) = boxes(
        &problem,
        &mut numerator,
        &prototype,
        options(MODES[0].0, MODES[0].1),
    );
    let mut worst = (0_f64, 0_f64);
    for (coefficients, mu_extraction) in MODES {
        let (other, results) = boxes(
            &problem,
            &mut numerator,
            &prototype,
            options(coefficients, mu_extraction),
        );
        assert_eq!(other.len(), reference.len());
        for ((a, computed), (b, _)) in other.iter().zip(&reference) {
            assert_eq!(a.len(), 7);
            let expected: Vec<bool> = (0..7)
                .map(|i| coefficients == BoxCoefficientMode::Full || matches!(i, 0 | 4 | 6))
                .collect();
            assert_eq!(computed, &expected);
            let scale = b.iter().map(modulus).fold(f64::MIN_POSITIVE, f64::max);
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                if computed[i] {
                    worst.0 = worst.0.max(distance(x, y) / scale);
                }
            }
        }
        for (a, b) in results.iter().zip(&integrated) {
            worst.1 = worst.1.max(laurent_error(a, b));
        }
    }
    worst
}

#[test]
fn rank_n2_box_modes_agree() {
    for n in [4, 5, 6] {
        for mu_terms in [true, false] {
            if n == 6 && !mu_terms {
                continue;
            }
            let float = modes_agree(n, mu_terms, Float::new(256));
            let binary64 = modes_agree(n, mu_terms, 0f64);
            eprintln!(
                "RANK_N2_BOX_MODES n={n} mu={mu_terms} coefficients/integrated \
                 float256={:.2e}/{:.2e} binary64={:.2e}/{:.2e}",
                float.0, float.1, binary64.0, binary64.1
            );
            assert!(float.0 <= 1e-60 && float.1 <= 1e-60, "n={n}: {float:?}");
            assert!(
                binary64.0 <= 1e-8 && binary64.1 <= 1e-8,
                "n={n}: {binary64:?}"
            );
        }
    }
    let sampled = BoxReductionOptions {
        coefficients: BoxCoefficientMode::Integrated,
        mu_extraction: BoxMuMode::Sampling,
    };
    assert_eq!(sampled.sample_count(4, 6), 4);
    assert_eq!(sampled.sample_count(4, 5), 3);
    assert_eq!(
        options(BoxCoefficientMode::Full, BoxMuMode::Sampling).sample_count(5, 7),
        4
    );
    assert_eq!(
        options(BoxCoefficientMode::Full, BoxMuMode::Expansion).sample_count(5, 7),
        2
    );
}

/// Answers the box expansion with Ninja's rank <= n+1 convention
/// `mu² = t² v0²`, dropping the offset beta.
struct Unshifted<N>(N);

impl<T, N: Numerator<T>> Numerator<T> for Unshifted<N> {
    fn rank(&self) -> usize {
        self.0.rank()
    }
    fn evaluate_batch(
        &mut self,
        requests: &[NumeratorRequest<'_, T>],
        output: &mut [T],
    ) -> ninjars::Result<()> {
        self.0.evaluate_batch(requests, output)
    }
    fn expansion_batch(
        &mut self,
        kind: &ExpansionKind,
        requests: &[ExpansionRequest<'_, T>],
        output: &mut [T],
    ) -> ninjars::Result<()> {
        match kind {
            ExpansionKind::MuShifted { powers } => {
                let requests: Vec<_> = requests
                    .iter()
                    .map(|r| ExpansionRequest {
                        id: r.id,
                        partition: r.partition,
                        vectors: r.vectors,
                        parameters: &[],
                    })
                    .collect();
                self.0.expansion_batch(
                    &ExpansionKind::Mu {
                        powers: powers.clone(),
                    },
                    &requests,
                    output,
                )
            }
            _ => self.0.expansion_batch(kind, requests, output),
        }
    }
}

/// Worst relative c4 difference between the shifted and unshifted expansions.
fn offset_effect(n: usize, mu_terms: bool) -> f64 {
    let p = Float::new(256);
    let problem = Problem::new(n, n + 2, 0xbe7a + n as u64, mu_terms, stride(n));
    let mode = options(BoxCoefficientMode::Integrated, BoxMuMode::Expansion);
    let mut shifted: PolynomialNumerator<Float> = problem.numerator(&p);
    let (right, _) = boxes(&problem, &mut shifted, &p, mode);
    let mut unshifted = Unshifted(problem.numerator(&p));
    let (wrong, _) = boxes(&problem, &mut unshifted, &p, mode);
    right
        .iter()
        .zip(&wrong)
        .map(|((a, _), (b, _))| {
            // c6 is independent of the offset; c4 is not.
            assert!(distance(&a[6], &b[6]) <= 1e-60 * modulus(&a[6]).max(1e-300));
            distance(&a[4], &b[4]) / modulus(&a[4])
        })
        .fold(0., f64::max)
}

#[test]
fn rank_n2_box_expansion_needs_the_cut_offset() {
    for n in [4, 5] {
        let q_only = offset_effect(n, false);
        let with_mu = offset_effect(n, true);
        eprintln!(
            "RANK_N2_BOX_OFFSET n={n} c4 change: q-only {q_only:.2e}, with mu² {with_mu:.2e}"
        );
        assert!(q_only <= 1e-60, "n={n}: {q_only:e}");
        assert!(with_mu > 1e-3, "n={n}: {with_mu:e}");
    }
}

/// Top-weight term stride: dense through rank 6, sparser above (Symbolica's
/// preparation of the polynomial expansions grows steeply with the terms).
fn stride(n: usize) -> usize {
    match n + 2 {
        0..=6 => 1,
        7 => 8,
        _ => 16,
    }
}
