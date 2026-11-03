//! The fast engine's light-like spinors near the `z` axis.
//!
//! `e3 = <1|γ|2]/2` and `e4 = <2|γ|1]/2` of every cut basis are built from the
//! spinors of the light-like `e1` and `e2`. As original Ninja's
//! `Spinor(const RealMomentum&)`, the fast engine formed them with
//! `√(p⁰ + p³)` and took the spinor of the beam axis when `|p⁰ ∓ p³|` was
//! below the absolute `INFRARED_EPS = 1e-9`. Near the `-z` axis `p⁰ + p³` is
//! formed by cancellation, and within the threshold of either axis the
//! transverse momentum was dropped, so `e3` and `e4` were silently wrong (every
//! lane reported success) in two cases:
//!
//! - external legs close to the `z` axis;
//! - a beam along `-z` that enters a cut basis as a difference of loop shifts
//!   (sums of external momenta): its `p⁰ + p³` is a rounding residue instead of
//!   0, which the threshold catches in GeV-like units but not once the energies
//!   are large in the units of the inputs.
//!
//! The spinors are now formed from the larger light-cone component
//! (`fast/geometry.rs`). The fast engine (binary64, `TensorNumerator`,
//! `OneLoopNativeMasters`) is compared with the generic engine in `Float(256)`
//! (`SymbolicaMasters`) on the same binary64 inputs: a massless rank-3 box
//! with a dense numerator and the exact S-matrix of its on-shell legs. Run
//! with `--test-threads=1`.

use ninjars::fast::{
    C64, FastAmplitude, FastInput, FastOptions, FastResult, FastStatus, RMom, TensorNumerator,
    tensor_layout::{monomials, ncoeff},
};
use ninjars::{
    AmplitudeInput, Complex, EvaluationStatus, Float, FloatLike, FourMomentum, SingleFloat,
};
use ninjars_symbolica::{
    OneLoopNativeMasters, PolynomialNumerator, SymbolicaAmplitudeKernels, SymbolicaCuts,
    SymbolicaHighCuts, SymbolicaMasters,
};
use symbolica::{
    atom::Atom,
    domains::{float::RealLike, rational::Rational},
    parser::ParseSettings,
};

const NAMESPACE: &str = "ninjars_fast_lightlike_spinors";
const RANK: usize = 3;
const BITS: u32 = 256;

/// Largest scaled error of the three Laurent coefficients allowed of the fast
/// engine (measured: at most 6.6e-15 on these points; with Ninja's spinors
/// up to 2.3e-6 on the legs near the axis and 0.79 in large units).
const TOLERANCE: f64 = 1e-13;

fn parse(text: &str) -> Atom {
    Atom::parse(text, NAMESPACE, ParseSettings::default()).unwrap()
}

fn mp(a: &RMom, b: &RMom) -> f64 {
    a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
}

fn add(a: &RMom, b: &RMom) -> RMom {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
}

fn neg(a: &RMom) -> RMom {
    a.map(|x| -x)
}

/// A box lane: the shifts of the incoming momenta `k` (`V0 = 0`,
/// `V_i = k_1 + ... + k_i`, sums in binary64), massless lines and legs, and
/// the S-matrix with exactly vanishing leg invariants.
struct Lane {
    label: String,
    shifts: Vec<RMom>,
    matrix: Vec<f64>,
    mu2: f64,
    /// Units of the lane: the numerator is homogeneous in them.
    unit: f64,
}

fn lane(label: String, k: [RMom; 4], mu2: f64, unit: f64) -> Lane {
    let mut shifts = vec![[0.; 4]];
    for i in 0..3 {
        let next = add(&shifts[i], &k[i]);
        shifts.push(next);
    }
    let mut matrix = vec![0.; 16];
    for (i, j) in [(0, 2), (1, 3)] {
        let d: RMom = std::array::from_fn(|mu| shifts[i][mu] - shifts[j][mu]);
        matrix[i * 4 + j] = mp(&d, &d);
        matrix[j * 4 + i] = matrix[i * 4 + j];
    }
    Lane {
        label,
        shifts,
        matrix,
        mu2,
        unit,
    }
}

/// Beams along `x` and the outgoing legs at an angle `theta` from the `z`
/// axis, `E = 500`.
fn near_axis(theta: f64) -> Lane {
    let e = 500.;
    let (s, c) = (theta.sin(), theta.cos());
    let p3 = [e, e * s, 0., e * c];
    let p4 = [e, -e * s, 0., -e * c];
    let k = [[e, e, 0., 0.], [e, -e, 0., 0.], neg(&p3), neg(&p4)];
    lane(format!("legs at {theta:e} from the z axis"), k, 1e4, 1.)
}

/// Beams along `±z` at `E = 400 f` and the top-like scattering angle of a
/// collider point, in units `f` (`mu² = (100 f)²`). The propagators are ordered
/// `(-p3, p2, p1, -p4)`, so the beam along `-z` enters the cut bases as
/// `V2 - V1 = (p2 - p3) + p3`.
fn units(f: f64) -> Lane {
    let e = 400. * f;
    let (cos, phi) = (0.21_f64, 0.7_f64);
    let sin = (1. - cos * cos).sqrt();
    let p3 = [e, e * sin * phi.cos(), e * sin * phi.sin(), e * cos];
    let p4 = [e, -p3[1], -p3[2], -p3[3]];
    let (p1, p2) = ([e, 0., 0., e], [e, 0., 0., -e]);
    let k = [neg(&p3), p2, p1, neg(&p4)];
    lane(
        format!("collider point in units {f:.3e}"),
        k,
        (100. * f).powi(2),
        f,
    )
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64 * 2. - 1.
    }
}

/// `sum_k p_k q^m_k` over the rank-3 tensor monomials (`tensor_layout` order).
fn polynomial() -> (Atom, Vec<Atom>) {
    let names = ["q0", "q1", "q2", "q3"];
    let mut terms = Vec::new();
    for (k, m) in monomials(RANK).into_iter().enumerate() {
        let mut term = format!("p{k}");
        for (mu, &e) in m.iter().enumerate() {
            if e > 0 {
                term.push_str(&format!("*{}^{e}", names[mu]));
            }
        }
        terms.push(term);
    }
    let parameters = (0..ncoeff(RANK)).map(|k| parse(&format!("p{k}"))).collect();
    (parse(&terms.join("+")), parameters)
}

/// The same dense numerator for every lane, in the lane's units: the
/// coefficient of a monomial of degree `d` carries `unit^(3 - d)`.
fn coefficients(lane: &Lane) -> Vec<C64> {
    let mut rng = Rng(0x5eed);
    monomials(RANK)
        .into_iter()
        .map(|m| {
            let degree = m.iter().sum::<u32>() as i32;
            let z = C64::new(rng.uniform(), rng.uniform());
            z * lane.unit.powi(RANK as i32 - degree)
        })
        .collect()
}

fn fast(lanes: &[Lane]) -> Vec<(FastResult, FastStatus)> {
    let count = lanes.len();
    let shifts: Vec<RMom> = lanes.iter().flat_map(|l| l.shifts.clone()).collect();
    let masses = vec![C64::ZERO; 4 * count];
    let scales: Vec<f64> = lanes.iter().map(|l| l.mu2).collect();
    let matrix: Vec<f64> = lanes.iter().flat_map(|l| l.matrix.clone()).collect();
    let mut tensor = TensorNumerator::new(RANK, count);
    for (i, lane) in lanes.iter().enumerate() {
        tensor
            .lane_coefficients_mut(i)
            .copy_from_slice(&coefficients(lane));
    }
    let mut amplitude = FastAmplitude::new(4, RANK, FastOptions::default()).unwrap();
    let mut masters = OneLoopNativeMasters::new().unwrap();
    let mut results = vec![FastResult::default(); count];
    let mut statuses = vec![FastStatus::NotEvaluated; count];
    amplitude
        .evaluate(
            &FastInput {
                lanes: count,
                shifts: &shifts,
                masses_squared: &masses,
                scales_squared: &scales,
                s_matrix: Some(&matrix),
            },
            &mut tensor,
            &mut masters,
            &mut results,
            &mut statuses,
        )
        .unwrap();
    results.into_iter().zip(statuses).collect()
}

/// The generic engine in `Float(256)` on the same (binary64) inputs.
fn exact(lanes: &[Lane]) -> Vec<[C64; 3]> {
    let prototype = Float::new(BITS);
    let real = |x: f64| {
        Complex::new(
            prototype.from_rational(&Rational::try_from(x).unwrap()),
            prototype.zero(),
        )
    };
    let complex = |z: C64| {
        Complex::new(
            prototype.from_rational(&Rational::try_from(z.re).unwrap()),
            prototype.from_rational(&Rational::try_from(z.im).unwrap()),
        )
    };
    let count = lanes.len();
    let shifts: Vec<FourMomentum<Complex<Float>>> = lanes
        .iter()
        .flat_map(|l| l.shifts.iter().map(|v| FourMomentum::from(v.map(real))))
        .collect();
    let masses = vec![real(0.); 4 * count];
    let scales: Vec<_> = lanes.iter().map(|l| real(l.mu2)).collect();
    let matrix: Vec<_> = lanes
        .iter()
        .flat_map(|l| l.matrix.iter().map(|&x| real(x)))
        .collect();
    let sampling: Vec<_> = lanes
        .iter()
        .map(|l| real(l.matrix.iter().map(|x| x.abs()).fold(1., f64::max)))
        .collect();
    let values: Vec<_> = lanes
        .iter()
        .flat_map(|l| coefficients(l).into_iter().map(complex))
        .collect();
    let (expression, parameters) = polynomial();
    let variables = ["q0", "q1", "q2", "q3", "mu2"].map(parse);
    let mut numerator =
        PolynomialNumerator::new(expression, variables, parameters, prototype.clone()).unwrap();
    numerator.set_parameter_batch(&values, count).unwrap();
    let mut low = SymbolicaCuts::new(prototype.clone()).unwrap();
    let mut high = SymbolicaHighCuts::new(prototype.clone()).unwrap();
    let mut masters = SymbolicaMasters::new(prototype.clone());
    let mut amplitude = SymbolicaAmplitudeKernels::new(4, RANK, prototype.clone())
        .unwrap()
        .into_amplitude(count)
        .unwrap();
    amplitude
        .evaluate_batch_with_inputs(
            AmplitudeInput {
                shifts: &shifts,
                masses_squared: &masses,
                scales_squared: &scales,
                sampling_scales: &sampling,
                s_matrix: Some(&matrix),
            },
            &mut numerator,
            &mut low,
            &mut high,
            &mut masters,
        )
        .unwrap();
    for (lane, status) in lanes.iter().zip(amplitude.statuses()) {
        assert_eq!(*status, EvaluationStatus::Success, "{}", lane.label);
    }
    let to_f64 = |z: &Complex<Float>| C64::new(z.re.to_f64(), z.im.to_f64());
    amplitude
        .results()
        .iter()
        .map(|r| [to_f64(&r[0]), to_f64(&r[1]), to_f64(&r[2])])
        .collect()
}

/// Largest error of the three Laurent coefficients over the largest modulus.
fn scaled_error(fast: &FastResult, exact: &[C64; 3]) -> f64 {
    let scale = exact.iter().map(|z| z.abs()).fold(0., f64::max);
    (0..3)
        .map(|k| (fast.laurent[k] - exact[k]).abs())
        .fold(0., f64::max)
        / scale
}

#[test]
fn spinors_near_the_z_axis_keep_the_fast_engine_accurate() {
    let mut lanes: Vec<Lane> = [1e-3, 1e-5, 1e-6, 1e-8].map(near_axis).into();
    // Units from 1 to 1e8 in quarter decades.
    lanes.extend((0..=32).map(|k| units(10f64.powf(f64::from(k) / 4.))));
    let fast = fast(&lanes);
    let exact = exact(&lanes);
    let mut failures = Vec::new();
    for ((lane, (result, status)), exact) in lanes.iter().zip(&fast).zip(&exact) {
        let error = scaled_error(result, exact);
        eprintln!(
            "FAST_LIGHTLIKE_SPINORS {}: {status:?}, scaled error {error:.1e}",
            lane.label
        );
        if *status != FastStatus::Success || error.is_nan() || error > TOLERANCE {
            failures.push(format!("{}: {status:?}, {error:e}", lane.label));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
