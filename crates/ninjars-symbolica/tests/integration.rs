use ninjars::{
    Complex, CutBasis, FourMomentum, IntegrationStatus, Laurent, MasterFailure, MasterIntegral,
    MasterProvider, MasterRequest, MasterStatus, ResidueIntegrationRequest, ResidueIntegrator,
    Result, SampleId, SingleFloat,
};
use ninjars_symbolica::{
    SymbolicaDualIntegrationKernels, SymbolicaDualMasters, SymbolicaIntegrationKernels,
    SymbolicaMasters,
};
use symbolica::domains::dual::HyperDual;

fn c(v: f64) -> Complex<f64> {
    Complex::new(v, 0.)
}
fn q(v: [f64; 4]) -> FourMomentum<Complex<f64>> {
    FourMomentum::from(v.map(c))
}
fn basis() -> CutBasis<Complex<f64>> {
    CutBasis {
        e1: q([1., -1., 0., 0.]),
        e2: q([1., 1., 0., 0.]),
        e3: FourMomentum::from([c(0.), c(0.), c(1.), Complex::new(0., 1.)]),
        e4: FourMomentum::from([c(0.), c(0.), c(1.), Complex::new(0., -1.)]),
        e12: c(2.),
    }
}
fn close(value: Complex<f64>, expected: f64) {
    assert!(
        (value.re - expected).abs() < 1e-9 * (1. + expected.abs()) && value.im.abs() < 1e-9,
        "{value} != {expected}"
    );
}

#[test]
fn bubble_null_moments_match_original_ninja_integrals() {
    // Independent original Ninja results for (q0-q1)^1,2,3, equal unit masses,
    // V0=0,V1=(0,1,0,0). No integration arithmetic is used to construct expected values.
    let kernels = SymbolicaIntegrationKernels::new(2, true, 0.).unwrap();
    let mut integration = ResidueIntegrator::new(kernels, c(0.));
    let shifts = [q([0.; 4]), q([0., 1., 0., 0.])];
    let basis = basis();
    let masses = [c(1.); 2];
    let scale = c(1.);
    let mut coefficients = [[c(0.); 20]; 3];
    for (row, index) in coefficients.iter_mut().zip([1, 2, 13]) {
        row[index] = c(1.);
    }
    let requests: Vec<_> = coefficients
        .iter()
        .enumerate()
        .map(|(lane, coefficients)| ResidueIntegrationRequest {
            id: SampleId { lane, sample: 0 },
            partition: &[0, 1],
            shifts: &shifts,
            masses_squared: &masses,
            coefficients,
            basis: &basis,
            scale_squared: &scale,
        })
        .collect();
    let mut output = [[c(99.); 5]; 3];
    let mut statuses = vec![IntegrationStatus::InvalidInput; 3];
    integration
        .evaluate_batch(
            &requests,
            &mut SymbolicaMasters::new(0.),
            &mut output,
            &mut statuses,
        )
        .unwrap();
    assert!(statuses.iter().all(|s| *s == IntegrationStatus::Success));
    for (i, (row, finite)) in output
        .iter()
        .zip([
            -0.07602235241001004,
            -0.04580758099112625,
            -0.03070019528000567,
        ])
        .enumerate()
    {
        for (v, e) in row
            .iter()
            .zip([finite, 1. / (i + 2) as f64, 0., finite, 0.])
        {
            close(*v, e);
        }
    }
}

struct ForbiddenMasters;
impl<T> MasterProvider<T> for ForbiddenMasters {
    fn evaluate(&mut self, _: MasterRequest<'_, T>, _: &mut [Laurent<T>]) -> Result<()> {
        panic!("zero master weight must not be evaluated")
    }
    fn evaluate_batch(
        &mut self,
        _: MasterRequest<'_, T>,
        _: &mut [Laurent<T>],
        _: &mut [MasterStatus],
    ) -> Result<()> {
        panic!("zero master weight must not be evaluated")
    }
}

#[test]
fn explicit_mu_terms_skip_zero_master_weights_and_keep_rational_separation() {
    let mut integration = ResidueIntegrator::new(
        SymbolicaIntegrationKernels::new(3, true, 0.).unwrap(),
        c(0.),
    );
    let shifts = [q([0.; 4]), q([2., 1., 0., 0.]), q([3., 0., 1., 0.])];
    let basis = basis();
    let masses = [c(1.); 3];
    let scale = c(1.);
    let mut coefficients = [[c(0.); 15]; 2];
    coefficients[0][7] = c(1.);
    coefficients[1][14] = c(1.);
    let requests: Vec<_> = coefficients
        .iter()
        .enumerate()
        .map(|(lane, coefficients)| ResidueIntegrationRequest {
            id: SampleId { lane, sample: 0 },
            partition: &[0, 1, 2],
            shifts: &shifts,
            masses_squared: &masses,
            coefficients,
            basis: &basis,
            scale_squared: &scale,
        })
        .collect();
    let mut output = [[c(99.); 5]; 2];
    let mut statuses = vec![IntegrationStatus::InvalidInput; 2];
    integration
        .evaluate_batch(&requests, &mut ForbiddenMasters, &mut output, &mut statuses)
        .unwrap();
    for (row, r) in output.iter().zip([0.5, -1. / 12.]) {
        for (v, e) in row.iter().zip([r, 0., 0., 0., r]) {
            close(*v, e);
        }
    }
    assert!(statuses.iter().all(|s| *s == IntegrationStatus::Success));
}

#[derive(Default)]
struct RecordingMasters {
    calls: Vec<(MasterIntegral, usize)>,
}
impl MasterProvider<Complex<f64>> for RecordingMasters {
    fn evaluate(
        &mut self,
        _: MasterRequest<'_, Complex<f64>>,
        _: &mut [Laurent<Complex<f64>>],
    ) -> Result<()> {
        panic!("integration must use the batched status interface")
    }
    fn evaluate_batch(
        &mut self,
        r: MasterRequest<'_, Complex<f64>>,
        out: &mut [Laurent<Complex<f64>>],
        statuses: &mut [MasterStatus],
    ) -> Result<()> {
        r.validate(out.len())?;
        self.calls.push((r.integral, r.rows()));
        for row in 0..r.rows() {
            if r.masses_squared[row].re < 0. {
                statuses[row] = MasterStatus::Failed(MasterFailure::InvalidMass);
            } else {
                out[row] = [c(2.), c(3.), c(4.)];
                statuses[row] = MasterStatus::Success;
            }
        }
        Ok(())
    }
}

#[test]
fn compact_master_batches_and_transactional_failed_rows() {
    let mut integration = ResidueIntegrator::new(
        SymbolicaIntegrationKernels::new(1, false, 0.).unwrap(),
        c(0.),
    );
    let mut provider = RecordingMasters::default();
    let basis = basis();
    let shifts = [q([0.; 4])];
    for lanes in [0, 1, 8, 33] {
        let coefficients: Vec<_> = (0..lanes)
            .map(|i| [c((i + 1) as f64), c(0.), c(0.), c(0.), c(0.)])
            .collect();
        let masses: Vec<_> = (0..lanes)
            .map(|i| if i % 5 == 3 { c(-1.) } else { c(1.) })
            .collect();
        let scales: Vec<_> = (0..lanes)
            .map(|i| if i % 5 == 2 { c(0.) } else { c(1.) })
            .collect();
        let requests: Vec<_> = (0..lanes)
            .rev()
            .map(|lane| ResidueIntegrationRequest {
                id: SampleId { lane, sample: 0 },
                partition: &[0],
                shifts: &shifts,
                masses_squared: &masses[lane..lane + 1],
                coefficients: &coefficients[lane],
                basis: &basis,
                scale_squared: &scales[lane],
            })
            .collect();
        let mut output = vec![[c(77.); 5]; lanes];
        let mut statuses = vec![IntegrationStatus::NonFiniteResult; lanes];
        provider.calls.clear();
        integration
            .evaluate_batch(&requests, &mut provider, &mut output, &mut statuses)
            .unwrap();
        assert_eq!(provider.calls.len(), usize::from(lanes > 0));
        if lanes > 0 {
            assert_eq!(
                provider.calls[0],
                (
                    MasterIntegral::A0,
                    (0..lanes).filter(|i| i % 5 != 2).count()
                )
            );
        }
        for (row, lane) in (0..lanes).rev().enumerate() {
            match lane % 5 {
                2 => {
                    assert_eq!(statuses[row], IntegrationStatus::InvalidInput);
                    assert_eq!(output[row], [c(77.); 5]);
                }
                3 => {
                    assert!(matches!(
                        statuses[row],
                        IntegrationStatus::MasterFailed { .. }
                    ));
                    assert_eq!(output[row], [c(77.); 5]);
                }
                _ => {
                    assert_eq!(statuses[row], IntegrationStatus::Success);
                    for (v, e) in output[row].iter().zip([2., 3., 4., 2., 0.]) {
                        close(*v, e * (lane + 1) as f64);
                    }
                }
            }
        }
        if lanes > 0 {
            let saved = output.clone();
            let saved_status = statuses.clone();
            assert!(
                integration
                    .evaluate_batch(
                        &requests,
                        &mut provider,
                        &mut output[..lanes - 1],
                        &mut statuses
                    )
                    .is_err()
            );
            assert_eq!(output, saved);
            assert_eq!(statuses, saved_status);
        }
    }
}

#[test]
fn derivative_only_weights_retain_higher_mixed_master_products() {
    let shape = vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![2, 0], vec![1, 1]];
    let d = |v: [f64; 5]| HyperDual::from_values(shape.clone(), v.map(c).to_vec());
    let zero = d([0.; 5]);
    let one = d([1., 0., 0., 0., 0.]);
    let map = |v: &Complex<f64>| {
        HyperDual::from_values(shape.clone(), vec![*v, c(0.), c(0.), c(0.), c(0.)])
    };
    let base = basis();
    let basis = CutBasis {
        e1: base.e1.map_ref(map),
        e2: base.e2.map_ref(map),
        e3: base.e3.map_ref(map),
        e4: base.e4.map_ref(map),
        e12: d([2., 0., 0., 0., 0.]),
    };
    let shifts = [q([0.; 4]).map_ref(map)];
    let mass = [d([2., 1., 0., 0., 0.])];
    let mut coefficients = vec![zero.clone(); 5];
    coefficients[0] = d([0., 1., 0., 0., 1.]);
    let requests = [ResidueIntegrationRequest {
        id: SampleId { lane: 0, sample: 0 },
        partition: &[0],
        shifts: &shifts,
        masses_squared: &mass,
        coefficients: &coefficients,
        basis: &basis,
        scale_squared: &one,
    }];
    let kernels = SymbolicaDualIntegrationKernels::new(1, false, 0., &zero).unwrap();
    let mut integration = ResidueIntegrator::new(kernels, zero.clone());
    let mut output = [std::array::from_fn(|_| zero.clone())];
    let mut statuses = [IntegrationStatus::InvalidInput];
    integration
        .evaluate_batch(
            &requests,
            &mut SymbolicaDualMasters::new(0., &zero),
            &mut output,
            &mut statuses,
        )
        .unwrap();
    assert_eq!(statuses, [IntegrationStatus::Success]);
    // x(1+y) A0(2+x): A0(m)=m(1-ln m), A0'(m)=-ln m.
    let a = 2. * (1. - 2f64.ln());
    for (value, expected) in output[0].iter().zip([
        [0., a, 0., -2f64.ln(), a],
        [0., 2., 0., 1., 2.],
        [0.; 5],
        [0., a, 0., -2f64.ln(), a],
        [0.; 5],
    ]) {
        for (v, e) in value.values.iter().zip(expected) {
            close(*v, e);
        }
    }
    assert!(zero.is_zero()); // The derivative-only weight has the same primal, but was not skipped.
}
