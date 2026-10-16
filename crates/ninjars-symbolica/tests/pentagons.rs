use ninjars::{
    Complex, CutFailure, CutOutcome, FourMomentum, PentagonCut, PentagonDivisionRequest,
    PentagonReducer, PentagonReductionKernels, Real,
};
use ninjars_symbolica::{
    DualPolynomialNumerator, PolynomialNumerator, SymbolicaDualHighCuts,
    SymbolicaDualPentagonKernels, SymbolicaHighCuts, SymbolicaPentagonKernels,
};
use symbolica::{
    atom::Atom,
    domains::{
        dual::{DualNumber, HyperDual},
        float::FloatLike,
    },
    parse, symbol,
};

fn c(x: f64) -> Complex<f64> {
    Complex::new(x, 0.)
}
fn vars() -> [Atom; 5] {
    [
        parse!("q0"),
        parse!("q1"),
        parse!("q2"),
        parse!("q3"),
        parse!("mu2"),
    ]
}
const SHIFTS: [[i64; 4]; 6] = [
    [0, 0, 0, 0],
    [2, 1, 0, 0],
    [3, 0, 1, 0],
    [4, 0, 0, 1],
    [1, 2, 3, 4],
    [3, -2, 1, 2],
];
fn shifts(n: usize) -> Vec<FourMomentum<Complex<f64>>> {
    SHIFTS[..n]
        .iter()
        .map(|v| FourMomentum::from(v.map(|x| c(x as f64))))
        .collect()
}
fn near(x: Complex<f64>, y: Complex<f64>) {
    assert!((x - y).norm().re < 1e-9 * (1. + y.norm().re), "{x} != {y}");
}

#[test]
fn ordered_five_cut_residues_restore_tail_batches_and_failed_lanes() {
    for n in [5, 6] {
        let q = vars();
        let momentum =
            FourMomentum::from_args(q[0].clone(), q[1].clone(), q[2].clone(), q[3].clone());
        let external: Vec<_> = (0..if n == 5 { 1 } else { 6 })
            .map(|i| Atom::var(symbol!(format!("pentagon_test::a{i}"))))
            .collect();
        let expression = if n == 5 {
            &external[0] * &q[4]
        } else {
            (0..6).fold(Atom::Zero, |sum, i| {
                let shifted = momentum.clone() + FourMomentum::from(SHIFTS[i].map(Atom::num));
                sum + &external[i]
                    * &q[4]
                    * (shifted.dot(&shifted) - Atom::num((i + 1) as i64) - &q[4])
            })
        };
        let mut numerator = PolynomialNumerator::new(expression, q, external.clone(), 0.).unwrap();
        let mut reducer =
            PentagonReducer::new(SymbolicaPentagonKernels::new(n, n + 1, 0.).unwrap(), c(0.))
                .unwrap();
        let mut geometry = SymbolicaHighCuts::new(0.).unwrap();
        for lanes in [0, 1, 8, 17] {
            let mut points: Vec<_> = (0..lanes).flat_map(|_| shifts(n)).collect();
            let masses: Vec<_> = (0..lanes)
                .flat_map(|_| (1..=n).map(|i| c(i as f64)))
                .collect();
            let parameters: Vec<_> = (0..lanes)
                .flat_map(|lane| {
                    (0..external.len()).map(move |i| {
                        Complex::new((3 * lane + i + 1) as f64, -0.25 * (lane as f64))
                    })
                })
                .collect();
            numerator.set_parameter_batch(&parameters, lanes).unwrap();
            if lanes > 8 {
                points[4 * n + 1] = points[4 * n];
            }
            let mut output = vec![
                CutOutcome::Unstable(CutFailure::NonFiniteResult);
                lanes * reducer.cut_count()
            ];
            reducer
                .evaluate_batch(
                    &points,
                    &masses,
                    lanes,
                    &mut numerator,
                    &mut geometry,
                    &mut output,
                )
                .unwrap();
            for (row, outcome) in output.iter().enumerate() {
                let lane = row % lanes;
                match outcome {
                    CutOutcome::Unstable(_) if lane == 4 && lanes > 8 => (),
                    CutOutcome::Unstable(f) => panic!("n={n},row={row},failure={f:?}"),
                    CutOutcome::Stable(residue) => {
                        assert_eq!(residue.id.lane, lane);
                        assert_eq!(residue.id.sample, residue.cut_index);
                        if lane == 4 && lanes > 8 {
                            continue;
                        }
                        let cut = &reducer.topology().cuts()[residue.cut_index];
                        let coefficient = if n == 5 { 0 } else { cut.complement[0] };
                        near(
                            residue.coefficient,
                            parameters[lane * external.len() + coefficient],
                        );
                    }
                }
            }
            let before = format!("{output:?}");
            assert!(
                reducer
                    .evaluate_batch(
                        &points,
                        &masses,
                        lanes + 1,
                        &mut numerator,
                        &mut geometry,
                        &mut output
                    )
                    .is_err()
            );
            assert_eq!(format!("{output:?}"), before);
        }
    }
}

#[test]
fn five_cut_division_preserves_all_mixed_derivatives() {
    let prototype = HyperDual::from_values(
        vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![2, 0], vec![1, 1]],
        vec![c(0.); 5],
    );
    let mut coefficient = prototype.zero();
    coefficient
        .values_mut()
        .clone_from_slice(&[c(3.), c(2.), c(-1.), c(5.), c(7.)]);
    let source = PolynomialNumerator::new(parse!("a*mu2"), vars(), vec![parse!("a")], 0.).unwrap();
    let mut numerator = DualPolynomialNumerator::new(source, prototype.clone()).unwrap();
    numerator
        .set_parameter_batch(std::slice::from_ref(&coefficient), 1)
        .unwrap();
    let lift = |x: Complex<f64>| {
        let mut d = prototype.zero();
        d.values_mut()[0] = x;
        d
    };
    let mut points: Vec<_> = shifts(5).iter().map(|p| (*p).map(lift)).collect();
    points[1].temporal.value.values_mut()[1] = c(1.);
    points[4].spatial.pz.values_mut()[2] = c(1.);
    let mut masses: Vec<_> = (1..=5).map(|i| prototype.from_i64(i)).collect();
    masses[2].values_mut()[2] = c(2.);
    let mut geometry = SymbolicaDualHighCuts::new(0., prototype.clone()).unwrap();
    let kernels = SymbolicaDualPentagonKernels::new(5, 6, 0., &prototype).unwrap();
    let mut reducer = PentagonReducer::new(kernels, prototype).unwrap();
    let mut output = [CutOutcome::Unstable(CutFailure::NonFiniteResult)];
    reducer
        .evaluate_batch(
            &points,
            &masses,
            1,
            &mut numerator,
            &mut geometry,
            &mut output,
        )
        .unwrap();
    let CutOutcome::Stable(residue) = &output[0] else {
        panic!("{output:?}")
    };
    for (&x, &y) in residue
        .coefficient
        .values()
        .iter()
        .zip(coefficient.values())
    {
        near(x, y);
    }
}

#[test]
fn division_rejects_zero_mu_uncut_zeros_and_invalid_topology_transactionally() {
    let points = shifts(6);
    let mut masses = vec![c(1.); 6];
    let q = FourMomentum::from([c(0.); 4]);
    // D5(q=0,mu²=1) = V5²-m5²-1. This value sets the uncut denominator to zero.
    masses[5] = points[5].dot(&points[5]) - c(1.);
    let cuts = [
        PentagonCut {
            q,
            mu_squared: c(0.),
        },
        PentagonCut {
            q,
            mu_squared: c(1.),
        },
        PentagonCut {
            q,
            mu_squared: c(2.),
        },
    ];
    let requests: Vec<_> = cuts
        .iter()
        .map(|geometry| PentagonDivisionRequest {
            partition: &[0, 1, 2, 3, 4],
            complement: &[5],
            shifts: &points,
            masses_squared: &masses,
            geometry,
        })
        .collect();
    let mut kernels = SymbolicaPentagonKernels::new(6, 7, 0.).unwrap();
    let mut output = [c(99.); 3];
    let mut failures = [None; 3];
    kernels
        .divide_batch(&requests, &[c(3.); 3], &mut output, &mut failures)
        .unwrap();
    assert_eq!(
        failures,
        [
            Some(CutFailure::SingularParameter),
            Some(CutFailure::SingularParameter),
            None
        ]
    );
    assert_eq!(&output[..2], &[c(99.); 2]);
    near(output[2], c(-1.5));
    let invalid = [PentagonDivisionRequest {
        partition: &[0, 1, 2, 3, 4],
        complement: &[4],
        shifts: &points,
        masses_squared: &masses,
        geometry: &cuts[2],
    }];
    assert!(
        kernels
            .divide_batch(&invalid, &[c(1.)], &mut output[..1], &mut failures[..1])
            .is_err()
    );
    assert_eq!(output[0], c(99.));
    assert_eq!(failures[0], Some(CutFailure::SingularParameter));
}
