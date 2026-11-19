//! Which test made a lane `Unstable`, and by how much
//! ([`super::take_instabilities`]): one lane per kind of test, in the rank
//! <= n+1 engine and the rank n+2 engine, between regular lanes (the records
//! name the lane of the call's input). The configurations put shifts on, or
//! within rounding of, the degenerate configuration each test guards.

use super::complex::{C64, RMom};
use super::tensor_layout::ncoeff;
use super::tests::{UnitMasters, unit_tensor};
use super::{
    FastAmplitude, FastInput, FastOptions, FastResult, FastStatus, Instability, InstabilityTest,
    take_instabilities,
};

const A: RMom = [1.902, -0.517, -0.537, -0.876];
const B: RMom = [-1.599, 0.29, 0.378, 0.142];
const C: RMom = [-0.892, -0.823, 0.499, 0.629];
const ZERO: RMom = [0.; 4];

/// A regular lane of `n <= 6` propagators.
fn regular(n: usize) -> Vec<RMom> {
    let extra: [RMom; 2] = [
        [0.731, 0.212, -0.403, 0.118],
        [-0.305, 0.611, 0.207, -0.452],
    ];
    [ZERO, A, B, C]
        .iter()
        .chain(extra.iter())
        .take(n)
        .copied()
        .collect()
}

/// `x A + y B + z C + e w`.
fn combination(x: f64, y: f64, z: f64, e: f64, w: RMom) -> RMom {
    std::array::from_fn(|i| x * A[i] + y * B[i] + z * C[i] + e * w[i])
}

/// Evaluate `lanes` (unit masters, the unit tensor; the lanes in `lowered`
/// with a zero top-rank block, so that at rank n+2 they are reduced at rank
/// n+1) and return the statuses and the records.
fn run_lanes(
    rank: usize,
    lanes: &[Vec<RMom>],
    lowered: &[usize],
    options: FastOptions,
) -> (Vec<FastStatus>, Vec<(usize, Instability)>) {
    let n = lanes[0].len();
    let shifts: Vec<RMom> = lanes.iter().flatten().copied().collect();
    let count = lanes.len();
    let masses = vec![C64::new(0.7, -0.05); count * n];
    let scales = vec![1.; count];
    let input = FastInput {
        lanes: count,
        shifts: &shifts,
        masses_squared: &masses,
        scales_squared: &scales,
        s_matrix: None,
    };
    let mut amplitude = FastAmplitude::new(n, rank, options).unwrap();
    let mut results = vec![FastResult::default(); count];
    let mut statuses = vec![FastStatus::NotEvaluated; count];
    let mut tensor = unit_tensor(rank, count);
    let width = tensor.ncoeff();
    for &lane in lowered {
        tensor.coefficients_mut()[lane * width + ncoeff(rank - 1)..(lane + 1) * width]
            .fill(C64::ZERO);
    }
    amplitude
        .evaluate(
            &input,
            &mut tensor,
            &mut UnitMasters,
            &mut results,
            &mut statuses,
        )
        .unwrap();
    (statuses, take_instabilities())
}

/// `[regular, odd, regular]`: lane 1 is `Unstable` by `test`, the others
/// succeed; returns lane 1's record.
#[track_caller]
fn refused(rank: usize, odd: &[RMom], options: FastOptions, test: InstabilityTest) -> Instability {
    let n = odd.len();
    let (statuses, records) =
        run_lanes(rank, &[regular(n), odd.to_vec(), regular(n)], &[], options);
    use FastStatus::{Success, Unstable};
    assert_eq!(statuses, [Success, Unstable, Success], "{records:?}");
    assert_eq!(records.len(), 1, "{records:?}");
    let (lane, record) = records[0];
    assert_eq!(lane, 1);
    assert_eq!(record.test, test, "{record}");
    record
}

/// A refused noise test: the measured quantity at most its bound.
#[track_caller]
fn noise(rank: usize, odd: &[RMom], options: FastOptions, test: InstabilityTest) -> Instability {
    let record = refused(rank, odd, options, test);
    assert!(
        record.value.is_finite() && record.bound > 0. && record.value <= record.bound,
        "{record}"
    );
    record
}

fn threshold(stability_threshold: f64) -> FastOptions {
    FastOptions {
        stability_threshold,
        ..FastOptions::default()
    }
}

#[test]
fn records_name_the_test_and_the_numbers_up_to_n1() {
    use InstabilityTest::*;
    let default = FastOptions::default;
    // Box (0, 1, 2, 3): k3 = -V3 parallel to k0 = V1.
    let parallel = [ZERO, [1., 0., 0., 0.], B, [-2., 0., 0., 0.]];
    let record = noise(4, &parallel, default(), BoxGram);
    assert_eq!(record.value, 0.);
    // Four coplanar shifts: the box normal's factors vanish.
    let coplanar = [ZERO, A, B, combination(0.5, 0.5, 0., 0., ZERO)];
    noise(4, &coplanar, default(), BoxNormal);
    // A repeated propagator vanishes at the cut solutions of every box
    // that does not cut it.
    noise(5, &[ZERO, A, B, C, ZERO], default(), BoxCutDenominator);
    // An uncut shift in the box's hyperplane: d0 = 2 v.(V_4 - V_0) = 0.
    let hyperplane = [ZERO, A, B, C, combination(0.5, 0.5, 0., 0., ZERO)];
    noise(5, &hyperplane, default(), BoxMuDenominator);
    noise(6, &hyperplane, default(), BoxMuExpansion);
    // Three collinear shifts: an exactly zero triangle Gram determinant.
    let collinear = [ZERO, [1., 0., 0., 0.], [2., 0., 0., 0.], [3., 0.5, 0., 0.]];
    let record = noise(3, &collinear, default(), TriangleGram);
    assert_eq!(record.value, 0.);
    // An uncut shift 3e-12 off the plane of a triangle: within the box
    // normal's tolerance, not the triangle expansion's.
    let off_plane = [
        ZERO,
        A,
        B,
        combination(2.5, -1.2, 0., 3e-12, [0., 0., 0., 1.]),
    ];
    noise(3, &off_plane, default(), TriangleExpansion);
    // The bubble (0, 1)'s uncut V2 in the span of its momentum and Ninja's
    // reference: e3.(V2 - V0) is rounding, below an absolute 1e-10.
    let k: RMom = std::array::from_fn(|i| -A[i]);
    let sign = |x: f64, y: f64| if y >= 0. { x.abs() } else { -x.abs() };
    let s3 = 1. / 3f64.sqrt();
    let reference: RMom = [
        sign(1., k[0]),
        -sign(s3, k[1]),
        -sign(s3, k[2]),
        -sign(s3, k[3]),
    ];
    let spanned: RMom = std::array::from_fn(|i| 0.7 * k[i] + 1.3 * reference[i]);
    let record = noise(2, &[ZERO, A, spanned], threshold(1e-10), BubbleExpansion);
    assert_eq!(record.bound, 1e-10);
    // Tadpoles (n = 2: the bubble has no uncut line) against a threshold
    // above every quantity.
    let (statuses, records) = run_lanes(2, &[regular(2)], &[], threshold(1e300));
    assert_eq!(statuses, [FastStatus::Unstable]);
    assert_eq!(records[0].1.test, TadpoleExpansion);
    assert_eq!(records[0].1.bound, 1e300);
    // The conditioning screen: an uncut shift 1e-8 off the box's
    // hyperplane passes the noise tests and loses too many digits; the
    // record is the predicted rounding error against its limit.
    let near = [
        ZERO,
        A,
        B,
        C,
        combination(0.3, 0.4, 0.5, 1e-8, [0.137, -0.291, 0.443, 0.613]),
    ];
    let record = refused(5, &near, default(), BoxMuConditioning);
    assert!(record.value > record.bound && record.bound > 0., "{record}");
    // The same with the thresholds zeroed (pyAmpliCol's unchecked mode):
    // the screen stays.
    let unchecked = FastOptions {
        stability_threshold: 0.,
        relative_tolerance: 0.,
        ..FastOptions::default()
    };
    refused(5, &near, unchecked, BoxMuConditioning);
    // Euclidean planes: spatial shifts (every box basis), and a triangle
    // basis of two orthogonal spatial momenta after a regular box.
    let spatial = [ZERO, [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]];
    let record = refused(4, &spatial, default(), EuclideanPlane);
    assert_eq!((record.value, record.bound), (1., 0.));
    let triangle = [
        ZERO,
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [1.5, 0.2, 0.3, 0.4],
    ];
    let record = refused(3, &triangle, default(), EuclideanPlane);
    assert_eq!((record.value, record.bound), (1., 0.));
    refused(3, &triangle, unchecked, EuclideanPlane);
}

#[test]
fn records_name_the_test_and_the_numbers_at_n2() {
    use InstabilityTest::*;
    let default = FastOptions::default;
    let parallel = [ZERO, [1., 0., 0., 0.], B, [-2., 0., 0., 0.]];
    noise(6, &parallel, default(), BoxGram);
    let hyperplane = [ZERO, A, B, C, combination(0.5, 0.5, 0., 0., ZERO)];
    noise(7, &hyperplane, default(), BoxMuExpansion);
    let near = [
        ZERO,
        A,
        B,
        C,
        combination(0.3, 0.4, 0.5, 1e-6, [0.137, -0.291, 0.443, 0.613]),
    ];
    let record = refused(7, &near, default(), BoxMuConditioning);
    assert!(record.value > record.bound, "{record}");
    let spatial = [ZERO, [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]];
    refused(6, &spatial, default(), EuclideanPlane);
    let triangle = [
        ZERO,
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [1.5, 0.2, 0.3, 0.4],
    ];
    refused(6, &triangle, default(), EuclideanPlane);
    let (statuses, records) = run_lanes(4, &[regular(2)], &[], threshold(1e300));
    assert_eq!(statuses, [FastStatus::Unstable]);
    assert_eq!(records[0].1.test, TadpoleExpansion);
}

/// At a declared rank n+2 the lanes reduced at a lower rank and the rank
/// n+2 lanes are evaluated as separate sub-batches; their records name the
/// lanes of the caller's batch.
#[test]
fn records_of_lowered_and_rank_n2_lanes_name_the_callers_lanes() {
    let parallel = vec![ZERO, [1., 0., 0., 0.], B, [-2., 0., 0., 0.]];
    let spatial = vec![ZERO, [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]];
    let lanes = [
        regular(4),
        parallel.clone(),
        regular(4),
        spatial.clone(),
        parallel,
    ];
    // Lanes 0, 3 and 4 lowered to rank n+1; 1 and 2 at rank n+2.
    let (statuses, mut records) = run_lanes(6, &lanes, &[0, 3, 4], FastOptions::default());
    use FastStatus::{Success, Unstable};
    assert_eq!(statuses, [Success, Unstable, Success, Unstable, Unstable]);
    records.sort_by_key(|(lane, _)| *lane);
    let tests: Vec<(usize, InstabilityTest)> = records.iter().map(|(l, r)| (*l, r.test)).collect();
    use InstabilityTest::{BoxGram, EuclideanPlane};
    assert_eq!(tests, [(1, BoxGram), (3, EuclideanPlane), (4, BoxGram)]);
}

/// Each evaluation starts by forgetting the previous one's records; taking
/// them empties the list; successful lanes leave none.
#[test]
fn records_belong_to_the_last_evaluation() {
    let spatial = vec![ZERO, [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]];
    let options = FastOptions::default();
    let mut amplitude = FastAmplitude::new(4, 4, options).unwrap();
    let masses = vec![C64::new(0.7, -0.05); 4];
    let mut results = vec![FastResult::default()];
    let mut statuses = vec![FastStatus::NotEvaluated];
    let mut evaluate = |shifts: &[RMom]| {
        amplitude
            .evaluate(
                &FastInput {
                    lanes: 1,
                    shifts,
                    masses_squared: &masses,
                    scales_squared: &[1.],
                    s_matrix: None,
                },
                &mut unit_tensor(4, 1),
                &mut UnitMasters,
                &mut results,
                &mut statuses,
            )
            .unwrap();
    };
    evaluate(&spatial);
    evaluate(&regular(4));
    assert!(take_instabilities().is_empty());
    evaluate(&spatial);
    assert_eq!(take_instabilities().len(), 1);
    assert!(take_instabilities().is_empty());
}

/// A triangle without uncut lines (n = 3) whose two momenta span a
/// Euclidean plane has no check that names it: its non-finite lane is
/// refused `Unstable` with an `EuclideanPlane` record (it was `NonFinite`,
/// served by neither), at every rank; the regular lanes keep their values.
#[test]
fn euclidean_triangles_without_uncut_lines_are_named() {
    let plane = vec![ZERO, [0., 1., 0., 0.], [0., 0., 1., 0.]];
    for rank in [1, 2, 3, 4, 5] {
        let record = refused(
            rank,
            &plane,
            FastOptions::default(),
            InstabilityTest::EuclideanPlane,
        );
        assert_eq!((record.value, record.bound), (1., 0.), "rank {rank}");
    }
}

#[test]
fn records_display_found_and_required() {
    let show = |test, value, bound| Instability { test, value, bound }.to_string();
    use InstabilityTest::*;
    assert_eq!(
        show(BoxNormal, 3.09e-11, 3.26e-8),
        "box normal: found 3.09e-11, required > 3.26e-8"
    );
    assert_eq!(
        show(TriangleGram, 0., 1.82e-12),
        "triangle Gram determinant: found 0, required > 1.82e-12"
    );
    assert_eq!(
        show(BoxMuConditioning, 3.19e-15, 2.36e-15),
        "box mu-expansion conditioning: predicted error 3.19e-15, allowed <= 2.36e-15"
    );
    assert_eq!(
        show(EuclideanPlane, 1., 0.),
        "Euclidean-plane cut: Gram determinant 1.00e0, required <= 0"
    );
    assert_eq!(
        show(BoxNormal, f64::NAN, 3.26e-8),
        "box normal: found NaN, required > 3.26e-8"
    );
    assert_eq!(
        show(BoxGeometry, f64::NAN, f64::NAN),
        "box geometry: degenerate, no quantity to compare"
    );
}
