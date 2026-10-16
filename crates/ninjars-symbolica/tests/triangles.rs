//! Actual triangle residue reduction. These are coefficient-stage checks;
//! integrated amplitudes additionally require lower cuts and master contraction.

use ninjars::{
    Complex, CutFailure, CutOutcome, CutSampler, FourMomentum, Numerator, NumeratorRequest,
    SampleId, TriangleReducer, TriangleSampleRequest,
};
use ninjars_symbolica::{
    DualPolynomialNumerator, PolynomialNumerator, SymbolicaCuts, SymbolicaDualCuts,
    SymbolicaDualTriangleKernels, SymbolicaTriangleKernels,
};
use symbolica::{
    atom::Atom,
    domains::{
        dual::{DualNumber, HyperDual},
        float::{FloatLike, Real},
    },
    parse, symbol,
};

fn c(v: f64) -> Complex<f64> {
    Complex::new(v, 0.)
}
fn qvars() -> [Atom; 5] {
    [
        parse!("q0"),
        parse!("q1"),
        parse!("q2"),
        parse!("q3"),
        parse!("mu2"),
    ]
}
fn shift(v: [i64; 4]) -> FourMomentum<Complex<f64>> {
    FourMomentum::from_args(
        c(v[0] as f64),
        c(v[1] as f64),
        c(v[2] as f64),
        c(v[3] as f64),
    )
}
fn near(actual: Complex<f64>, expected: Complex<f64>) {
    assert!(
        (actual - expected).norm().re <= 1e-9 * (1. + expected.norm().re),
        "actual={actual}, expected={expected}"
    );
}

#[test]
fn higher_rank_triangle_reduction_batches_and_errors() {
    let mut numerator = PolynomialNumerator::new(
        parse!("a*mu2+b*mu2^2"),
        qvars(),
        vec![parse!("a"), parse!("b")],
        0.,
    )
    .unwrap();
    assert_eq!(numerator.rank(), 4);
    let mut reducer =
        TriangleReducer::new(SymbolicaTriangleKernels::new(3, 4, 0.).unwrap(), c(0.)).unwrap();
    let mut geometry = SymbolicaCuts::new(0.).unwrap();
    assert_eq!(reducer.cut_count(), 1);
    let points = [
        [
            shift([0, 0, 0, 0]),
            shift([2, 1, 0, 0]),
            shift([3, 0, 1, 0]),
        ],
        [
            shift([0, 0, 0, 0]),
            shift([0, 1, 0, 0]),
            shift([0, 0, 1, 0]),
        ],
        [shift([0, 0, 0, 0]); 3],
        [
            shift([0, 0, 0, 0]),
            shift([1, 0, 0, 1]),
            shift([2, 0, 0, 0]),
        ],
    ];
    let parameters = [
        [c(2.), Complex::new(3., -0.5)],
        [Complex::new(-1., 0.25), c(5.)],
        [c(7.), c(11.)],
        [c(-3.), c(2.)],
    ];
    for order in [&[2, 0, 3, 1][..], &[1][..], &[][..], &[3, 0, 1][..]] {
        let shifts: Vec<_> = order.iter().flat_map(|&lane| points[lane]).collect();
        let masses = vec![c(1.); shifts.len()];
        let coefficients: Vec<_> = order.iter().flat_map(|&lane| parameters[lane]).collect();
        numerator
            .set_parameter_batch(&coefficients, order.len())
            .unwrap();
        let mut output = vec![CutOutcome::Unstable(CutFailure::NonFiniteResult); order.len()];
        reducer
            .evaluate_batch(
                &shifts,
                &masses,
                order.len(),
                &mut numerator,
                &mut geometry,
                &mut output,
            )
            .unwrap();
        for (lane, (&source, result)) in order.iter().zip(output).enumerate() {
            if source == 2 {
                assert!(matches!(
                    result,
                    CutOutcome::Unstable(CutFailure::DegenerateReference)
                ));
                continue;
            }
            let CutOutcome::Stable(residue) = result else {
                panic!("unexpected unstable source lane {source}")
            };
            assert_eq!(residue.id.lane, lane);
            assert_eq!(
                reducer.topology().cuts()[residue.cut_index].partition,
                [0, 1, 2]
            );
            assert_eq!(residue.coefficients.len(), 15);
            for (index, value) in residue.coefficients.into_iter().enumerate() {
                near(
                    value,
                    match index {
                        7 => parameters[source][0],
                        14 => parameters[source][1],
                        _ => c(0.),
                    },
                );
            }
        }
    }
    let mut output = vec![CutOutcome::Unstable(CutFailure::SingularParameter); 1];
    let masses = [c(1.); 3];
    assert!(
        reducer
            .evaluate_batch(
                &points[0][..2],
                &masses,
                1,
                &mut numerator,
                &mut geometry,
                &mut output
            )
            .is_err()
    );
    assert!(matches!(
        output[0],
        CutOutcome::Unstable(CutFailure::SingularParameter)
    ));
    // A late callback error must also leave the output unchanged.
    numerator.set_parameter_batch(&[], 0).unwrap();
    assert!(
        reducer
            .evaluate_batch(
                &points[0],
                &masses,
                1,
                &mut numerator,
                &mut geometry,
                &mut output
            )
            .is_err()
    );
    assert!(matches!(
        output[0],
        CutOutcome::Unstable(CutFailure::SingularParameter)
    ));
}

fn denominator(values: [i64; 4], mass: i64) -> Atom {
    let q = qvars();
    let momentum = FourMomentum::from_args(
        q[0].clone() + values[0],
        q[1].clone() + values[1],
        q[2].clone() + values[2],
        q[3].clone() + values[3],
    );
    momentum.mass_squared() - mass - &q[4]
}

#[test]
fn uncut_denominator_division_cancels_prepared_propagators() {
    let routing = [
        [0, 0, 0, 0],
        [2, 1, 0, 0],
        [3, 0, 1, 0],
        [1, 0, 0, 1],
        [4, -1, 2, 1],
        [5, 2, -1, 3],
    ];
    let mut geometry = SymbolicaCuts::new(0.).unwrap();
    for (n, expression, target_coefficient) in [
        (4, parse!("mu2") * denominator(routing[3], 4), 7),
        (
            6,
            (3..6).fold(Atom::num(1), |a, i| {
                a * denominator(routing[i], i as i64 + 1)
            }),
            0,
        ),
    ] {
        let shifts: Vec<_> = routing[..n].iter().copied().map(shift).collect();
        let masses: Vec<_> = (1..=n).map(|i| c(i as f64)).collect();
        let mut numerator = PolynomialNumerator::new(expression, qvars(), vec![], 0.).unwrap();
        assert_eq!(numerator.rank(), n);
        numerator.set_parameter_batch(&[], 1).unwrap();
        let mut reducer =
            TriangleReducer::new(SymbolicaTriangleKernels::new(n, n, 0.).unwrap(), c(0.)).unwrap();
        let mut output =
            vec![CutOutcome::Unstable(CutFailure::NonFiniteResult); reducer.cut_count()];
        reducer
            .evaluate_batch(
                &shifts,
                &masses,
                1,
                &mut numerator,
                &mut geometry,
                &mut output,
            )
            .unwrap();
        let mut target_checked = false;
        for result in output {
            let CutOutcome::Stable(residue) = result else {
                continue;
            };
            let target = reducer.topology().cuts()[residue.cut_index].partition == [0, 1, 2];
            target_checked |= target;
            for (index, value) in residue.coefficients.into_iter().enumerate() {
                near(
                    value,
                    if target && index == target_coefficient {
                        c(1.)
                    } else {
                        c(0.)
                    },
                );
            }
        }
        assert!(
            target_checked,
            "target triple cut must remain stable for n={n}"
        );
    }
}

#[test]
fn dynamic_dual_triangle_residue_is_independent_of_basis_derivatives() {
    let prototype = HyperDual::from_values(
        vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![2, 0], vec![1, 1]],
        vec![c(0.); 5],
    );
    let source = PolynomialNumerator::new(parse!("a*mu2"), qvars(), vec![parse!("a")], 0.).unwrap();
    let mut numerator = DualPolynomialNumerator::new(source, prototype.clone()).unwrap();
    let mut a = prototype.zero();
    a.values_mut()
        .clone_from_slice(&[c(3.), c(2.), c(-1.), c(5.), c(7.)]);
    numerator
        .set_parameter_batch(&[a.clone(), a.clone()], 2)
        .unwrap();
    let lift = |value: Complex<f64>| {
        let mut d = prototype.zero();
        d.values_mut()[0] = value;
        d
    };
    let mut shifts: Vec<_> = [
        [0, 0, 0, 0],
        [1, 0, 0, 1],
        [2, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ]
    .into_iter()
    .map(|v| shift(v).map(lift))
    .collect();
    shifts[1].temporal.value.values_mut()[1] = c(1.);
    shifts[5].spatial.py.values_mut()[2] = c(1.);
    let mut masses = vec![prototype.from_i64(1); 6];
    masses[2].values_mut()[2] = c(2.);
    let mut geometry = SymbolicaDualCuts::new(0., prototype.clone()).unwrap();
    let kernels = SymbolicaDualTriangleKernels::new(3, 2, 0., &prototype).unwrap();
    let mut reducer = TriangleReducer::new(kernels, prototype).unwrap();
    let mut output = vec![CutOutcome::Unstable(CutFailure::NonFiniteResult); 2];
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
    for result in output {
        let CutOutcome::Stable(residue) = result else {
            panic!("unexpected unstable Dual triangle stage")
        };
        for (index, value) in residue.coefficients.into_iter().enumerate() {
            for (&actual, &expected) in value.values().iter().zip(a.values()) {
                near(actual, if index == 7 { expected } else { c(0.) });
            }
        }
    }
}

#[test]
fn higher_rank_residue_reconstructs_numerator_on_cut() {
    let mut numerator = PolynomialNumerator::new(
        parse!("1+q0+2*q1^2+3*q2^3+q3^4+2*q0*q1*q2*q3+mu2*(q0^2+2*q1*q2+q3)+3*mu2^2"),
        qvars(),
        vec![],
        0.,
    )
    .unwrap();
    numerator.set_parameter_batch(&[], 1).unwrap();
    let shifts = [
        shift([1, -1, 2, 0]),
        shift([3, 0, 2, 0]),
        shift([4, -1, 3, 0]),
    ];
    let masses = [c(1.), c(2.), c(3.)];
    let mut geometry = SymbolicaCuts::new(0.).unwrap();
    let mut reducer =
        TriangleReducer::new(SymbolicaTriangleKernels::new(3, 4, 0.).unwrap(), c(0.)).unwrap();
    let mut output = [CutOutcome::Unstable(CutFailure::NonFiniteResult)];
    reducer
        .evaluate_batch(
            &shifts,
            &masses,
            1,
            &mut numerator,
            &mut geometry,
            &mut output,
        )
        .unwrap();
    let CutOutcome::Stable(residue) = &output[0] else {
        panic!("reconstruction triangle unstable")
    };
    // Independent original Ninja ansatz (polys_implem.cxx20-31), evaluated by
    // Symbolica. Here q0=(q+V0).e3 and q1=(q+V0).e4 are residue coordinates.
    let mut reconstructed=PolynomialNumerator::new(parse!("c0+c7*mu2+(c1+c8*mu2)*q0+c2*q0^2+c3*q0^3+(c4+c9*mu2)*q1+c5*q1^2+c6*q1^3+c14*mu2^2+c10*mu2*q0^2+c12*q0^4+c11*mu2*q1^2+c13*q1^4"),qvars(),(0..15).map(|i|Atom::var(symbol!(format!("c{i}")))).collect(),0.).unwrap();
    reconstructed
        .set_parameter_batch(&residue.coefficients, 1)
        .unwrap();
    let parameters = [
        (Complex::new(2., 0.5), c(0.)),
        (c(-3.), c(1.)),
        (Complex::new(1., -0.5), Complex::new(2., 0.25)),
        (c(5.), c(-2.)),
    ];
    let requests: Vec<_> = parameters
        .iter()
        .enumerate()
        .map(|(sample, (t, mu))| TriangleSampleRequest {
            id: SampleId { lane: 0, sample },
            cut: &residue.geometry,
            t,
            mu_squared: mu,
        })
        .collect();
    let mut samples = vec![CutOutcome::Unstable(CutFailure::NonFiniteResult); requests.len()];
    geometry
        .sample_triangle_batch(&requests, &mut samples)
        .unwrap();
    let loop_momenta: Vec<_> = samples
        .into_iter()
        .map(|q| match q {
            CutOutcome::Stable(q) => q,
            _ => panic!("sample unstable"),
        })
        .collect();
    let coordinates: Vec<_> = loop_momenta
        .iter()
        .map(|q| {
            FourMomentum::from_args(
                (*q + shifts[0]).dot(&residue.geometry.basis.e3),
                (*q + shifts[0]).dot(&residue.geometry.basis.e4),
                c(0.),
                c(0.),
            )
        })
        .collect();
    let requests: Vec<_> = loop_momenta
        .iter()
        .enumerate()
        .map(|(sample, q)| NumeratorRequest {
            id: SampleId { lane: 0, sample },
            q,
            mu_squared: &parameters[sample].1,
            partition: &[0, 1, 2],
        })
        .collect();
    let residue_requests: Vec<_> = coordinates
        .iter()
        .enumerate()
        .map(|(sample, q)| NumeratorRequest {
            id: SampleId { lane: 0, sample },
            q,
            mu_squared: &parameters[sample].1,
            partition: &[0, 1, 2],
        })
        .collect();
    let mut expected = vec![c(0.); requests.len()];
    let mut actual = expected.clone();
    numerator.evaluate_batch(&requests, &mut expected).unwrap();
    reconstructed
        .evaluate_batch(&residue_requests, &mut actual)
        .unwrap();
    for (actual, expected) in actual.into_iter().zip(expected) {
        near(actual, expected);
    }
}
