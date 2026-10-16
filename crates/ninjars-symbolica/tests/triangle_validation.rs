//! Regression controls for lane-local domains and public request validation.

use ninjars::{
    Complex, CutFailure, CutOutcome, FourMomentum, SingleFloat, TriangleDivisionRequest,
    TriangleReducer, TriangleReductionKernels,
};
use ninjars_symbolica::{
    DualPolynomialNumerator, PolynomialNumerator, SymbolicaDualCuts, SymbolicaDualTriangleKernels,
    SymbolicaTriangleKernels,
};
use symbolica::{
    domains::{dual::HyperDual, float::FloatLike},
    parse,
};

fn c(value: f64) -> Complex<f64> {
    Complex::new(value, 0.)
}

#[test]
fn uncut_dual_shape_failure_preserves_the_other_phase_space_lane() {
    let prototype = HyperDual::from_values(vec![vec![0], vec![1]], vec![c(0.); 2]);
    let lift = |value| {
        let mut dual = prototype.zero();
        dual.values[0] = c(value);
        dual
    };
    let routing = [
        [0., 0., 0., 0.],
        [2., 1., 0., 0.],
        [3., 0., 1., 0.],
        [1., 0., 0., 1.],
    ];
    let source = PolynomialNumerator::new(
        parse!("q3^2+mu2"),
        [
            parse!("q0"),
            parse!("q1"),
            parse!("q2"),
            parse!("q3"),
            parse!("mu2"),
        ],
        vec![],
        0.,
    )
    .unwrap();
    let mut numerator = DualPolynomialNumerator::new(source, prototype.clone()).unwrap();
    numerator.set_parameter_batch(&[], 2).unwrap();
    let mut geometry = SymbolicaDualCuts::new(0., prototype.clone()).unwrap();
    let kernels = SymbolicaDualTriangleKernels::new(4, 2, 0., &prototype).unwrap();
    let mut reducer = TriangleReducer::new(kernels, prototype.clone()).unwrap();
    let target_cut = reducer
        .topology()
        .cuts()
        .iter()
        .position(|cut| cut.partition == [0, 1, 2])
        .unwrap();
    let target_row = reducer
        .topology()
        .cuts()
        .iter()
        .filter(|cut| cut.partition.len() == 3)
        .position(|cut| cut.partition == [0, 1, 2])
        .unwrap()
        * 2;

    for malformed_mass in [true, false] {
        let mut shifts: Vec<_> = routing
            .into_iter()
            .cycle()
            .take(8)
            .map(|v| FourMomentum::from_args(lift(v[0]), lift(v[1]), lift(v[2]), lift(v[3])))
            .collect();
        let mut masses = vec![lift(1.); 8];
        let wrong_shape = HyperDual::from_values(vec![vec![0]], vec![c(1.)]);
        // Only line 3 of lane 1 is malformed. Triple-cut geometry [0,1,2]
        // cannot detect it; the denominator division must isolate this row.
        if malformed_mass {
            masses[7] = wrong_shape;
        } else {
            shifts[7].spatial.pz = wrong_shape;
        }
        let mut output =
            vec![CutOutcome::Unstable(CutFailure::SingularParameter); reducer.cut_count() * 2];
        reducer
            .evaluate_batch(
                &shifts,
                &masses,
                2,
                &mut numerator,
                &mut geometry,
                &mut output,
            )
            .unwrap();
        let CutOutcome::Stable(good) = &output[target_row] else {
            panic!("the valid phase-space lane was contaminated by its neighbor")
        };
        assert_eq!(good.cut_index, target_cut);
        assert_eq!(good.id.lane, 0);
        assert!(
            good.coefficients
                .iter()
                .all(|value| { value.values.iter().all(|component| component.is_finite()) })
        );
        assert!(matches!(output[target_row + 1], CutOutcome::Unstable(_)));
    }
}

#[test]
fn duplicate_or_overlapping_propagator_indices_are_rejected_before_mutation() {
    let mut kernels = SymbolicaTriangleKernels::new(4, 2, 0.).unwrap();
    let shifts = [FourMomentum::from_args(c(0.), c(0.), c(0.), c(0.)); 4];
    let masses = [c(1.); 4];
    let vectors = [shifts[0]; 3];
    let beta = c(0.);
    let numerator = vec![c(0.); kernels.numerator_expansion().len()];
    let mut output = vec![c(17.); kernels.quotient_columns()];
    let mut stable = [true];
    for (partition, complement) in [([0, 0, 1], [3]), ([0, 1, 2], [2])] {
        let request = TriangleDivisionRequest {
            partition: &partition,
            complement: &complement,
            shifts: &shifts,
            masses_squared: &masses,
            vectors: &vectors,
            beta: &beta,
        };
        assert!(
            kernels
                .divide_batch(&[request], &numerator, &mut output, &mut stable)
                .is_err()
        );
        assert!(output.iter().all(|value| *value == c(17.)));
        assert_eq!(stable, [true]);
    }
}
