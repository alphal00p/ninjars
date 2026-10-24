//! `fast::tensor_uv_residue` against the generic reduction engine with
//! oneloopmaster's UV-selected masters.
//!
//! The fast engine drops scaleless-bubble coefficients and refuses UV/IR
//! providers (`pole_selection.rs`), so a consumer of the fast engine cannot
//! separate the UV and IR poles of its combined result through the masters.
//! The UV residue of a one-loop tensor integral is fixed by the numerator's
//! large-`q` behaviour, so it can be read off the tensor coefficients:
//! `tensor_uv_residue` must equal the generic engine's `1/ε` coefficient under
//! `PoleSelection::UV`, scaleless bubbles included, for tadpoles, bubbles,
//! triangles and boxes with massless and massive, real and light-like
//! kinematics.
use ninjars::fast::tensor_layout::{monomial_index, monomials, ncoeff};
use ninjars::fast::{TensorUvResidues, tensor_uv_residue};
use ninjars::{Complex, EvaluationStatus, FourMomentum};
use ninjars_symbolica::{
    OneLoopNativeMasters, PoleSelection, PolynomialNumerator, SymbolicaAmplitudeKernels,
    SymbolicaCuts, SymbolicaHighCuts,
};
use symbolica::parse;

fn c(x: f64) -> Complex<f64> {
    Complex::new(x, 0.)
}

fn abs(z: &Complex<f64>) -> f64 {
    z.re.hypot(z.im)
}

fn distance(a: &Complex<f64>, b: &Complex<f64>) -> f64 {
    (a.re - b.re).hypot(a.im - b.im)
}

/// A dense numerator with small integer coefficients on every monomial up to
/// `rank` (deterministic), as tensor coefficients and as an expression.
fn numerator(rank: usize, seed: u64) -> (Vec<Complex<f64>>, String) {
    let mut state = seed;
    let mut tensor = vec![c(0.); ncoeff(rank)];
    let mut terms = Vec::new();
    for m in monomials(rank) {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let value = ((state >> 33) % 7) as i64 - 3;
        if value == 0 {
            continue;
        }
        tensor[monomial_index(m)] = c(value as f64);
        let mut term = format!("({value})");
        for (mu, &e) in m.iter().enumerate() {
            if e > 0 {
                term.push_str(&format!("*q{mu}^{e}"));
            }
        }
        terms.push(term);
    }
    (tensor, terms.join("+"))
}

/// The generic engine's `[finite, 1/ε, 1/ε²]` with `selection` (the first
/// three of its result columns).
fn generic(
    rank: usize,
    shifts: &[[f64; 4]],
    masses: &[f64],
    expression: &str,
    selection: PoleSelection,
) -> [Complex<f64>; 3] {
    let n = shifts.len();
    let vars = [
        parse!("q0"),
        parse!("q1"),
        parse!("q2"),
        parse!("q3"),
        parse!("mu2"),
    ];
    let mut numerator = PolynomialNumerator::new(parse!(expression), vars, vec![], 0.).unwrap();
    numerator.set_parameter_batch(&[], 1).unwrap();
    let mut amplitude = SymbolicaAmplitudeKernels::new(n, rank, 0.)
        .unwrap()
        .into_amplitude(1)
        .unwrap();
    let mut cuts = SymbolicaCuts::new(0.).unwrap();
    let mut masters = OneLoopNativeMasters::new()
        .unwrap()
        .with_pole_selection(selection);
    let momenta: Vec<_> = shifts
        .iter()
        .map(|v| FourMomentum::from(v.map(c)))
        .collect();
    let squared: Vec<_> = masses.iter().map(|&m| c(m)).collect();
    // Ninja's dimensional sampling scale (`amplitude_high.rs`).
    let mut sampling = 1_f64;
    for (i, a) in shifts.iter().enumerate() {
        for b in &shifts[i + 1..] {
            let d: [f64; 4] = std::array::from_fn(|mu| b[mu] - a[mu]);
            sampling = sampling.max((d[0] * d[0] - d[1] * d[1] - d[2] * d[2] - d[3] * d[3]).abs());
        }
    }
    for m in masses {
        sampling = sampling.max(m.abs());
    }
    let mut high = SymbolicaHighCuts::new(0.).unwrap();
    amplitude.reset_batch(1);
    amplitude
        .evaluate_batch_with_high_cuts(
            &momenta,
            &squared,
            &[c(7.)],
            &[c(sampling)],
            &mut numerator,
            &mut cuts,
            &mut high,
            &mut masters,
        )
        .unwrap();
    assert_eq!(amplitude.statuses(), &[EvaluationStatus::Success]);
    let out = &amplitude.results()[0];
    [out[0], out[1], out[2]]
}

#[test]
fn tensor_uv_residue_equals_the_generic_engine_with_uv_masters() {
    // (shifts V_i, squared masses m_i², rank): massive and massless, a
    // scaleless light-like bubble, a bubble inside a box with light-like legs.
    let cases: Vec<(Vec<[f64; 4]>, Vec<f64>, usize)> = vec![
        (vec![[0.3, 0.1, -0.2, 0.4]], vec![2.5], 2),
        (vec![[0.; 4], [3., 1., -2., 0.5]], vec![5., 11.], 3),
        (vec![[0.; 4], [2., 0., 0., 2.]], vec![0., 0.], 3),
        (
            vec![[0.; 4], [2., 0., 0., 2.], [3., 1., 0., 2.5]],
            vec![0., 0., 1.7],
            3,
        ),
        (
            vec![
                [0.; 4],
                [2., 0., 0., 2.],
                [4., 0., 0., 0.],
                [2., 1.2, 0.9, 1.],
            ],
            vec![0., 0., 0., 0.],
            4,
        ),
        (
            vec![
                [0.; 4],
                [2., 0., 0., 2.],
                [4., 0., 0., 0.],
                [2., 1.2, 0.9, 1.],
            ],
            vec![3., 3., 3., 3.],
            4,
        ),
    ];
    for (index, (shifts, masses, rank)) in cases.iter().enumerate() {
        let n = shifts.len();
        let (tensor, expression) = numerator(*rank, 17 + index as u64);
        let flat: Vec<_> = shifts.iter().flat_map(|v| v.map(c)).collect();
        let squared: Vec<_> = masses.iter().map(|&m| c(m)).collect();
        let residue = tensor_uv_residue(n, *rank, &flat, &squared, &tensor);
        assert_eq!(
            TensorUvResidues::new(n, *rank).residue(&flat, &squared, &tensor),
            residue
        );
        let uv = generic(*rank, shifts, masses, &expression, PoleSelection::UV);
        let both = generic(*rank, shifts, masses, &expression, PoleSelection::BOTH);
        let scale = abs(&uv[1]).max(abs(&residue)).max(1.);
        assert!(
            distance(&residue, &uv[1]) < 1e-10 * scale,
            "case {index}: residue {residue:?}, generic UV {:?}",
            uv[1]
        );
        // The double pole is infrared; the finite part does not depend on
        // the selection.
        assert!(abs(&uv[2]) < 1e-12 * scale, "case {index}: {:?}", uv[2]);
        assert_eq!(uv[0], both[0], "case {index}");
        // Every case is UV divergent (rank + 4 >= 2n): a vanishing residue
        // would not discriminate.
        assert!(abs(&residue) > 1e-3, "case {index}: {residue:?}");
    }
}

#[test]
fn finite_shapes_have_no_residue() {
    // rank + 4 < 2n: no UV pole, as the generic engine's UV masters agree.
    let shifts = [[0.; 4], [2., 0., 0., 2.], [3., 1., 0., 2.5]];
    let (tensor, expression) = numerator(1, 5);
    let flat: Vec<_> = shifts.iter().flat_map(|v| v.map(c)).collect();
    let residue = tensor_uv_residue(3, 1, &flat, &[c(1.), c(0.), c(2.)], &tensor);
    assert_eq!(residue, c(0.));
    let uv = generic(1, &shifts, &[1., 0., 2.], &expression, PoleSelection::UV);
    assert!(abs(&uv[1]) < 1e-12, "{:?}", uv[1]);
}
