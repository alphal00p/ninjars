use ninjars::{
    Complex, CutBasis, FloatLike, FourMomentum, IntegrationKernels, IntegrationStatus, Laurent,
    MasterFailure, MasterIntegral, MasterProvider, MasterRequest, MasterStatus, RankClass,
    ResidueIntegrationRequest, ResidueIntegrator, Result, SampleId, SingleFloat,
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
fn close_complex(value: Complex<f64>, expected: Complex<f64>) {
    assert!(
        (value.re - expected.re).hypot(value.im - expected.im)
            < 1e-9 * (1. + expected.re.hypot(expected.im)),
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
            basis: Some(&basis),
            scale_squared: &scale,
            s_matrix: None,
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
            basis: Some(&basis),
            scale_squared: &scale,
            s_matrix: None,
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
                basis: Some(&basis),
                scale_squared: &scales[lane],
                s_matrix: None,
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
        basis: Some(&basis),
        scale_squared: &one,
        s_matrix: None,
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

/// B0 = 1 + 2i (single pole 1) and B1 = 3 (single pole -1/2), lifted into the
/// carrier; any other master has a zero weight here.
struct FixedBubbleMasters<V>(fn(Complex<f64>) -> V);
impl<V: Clone> MasterProvider<V> for FixedBubbleMasters<V> {
    fn evaluate(&mut self, request: MasterRequest<'_, V>, output: &mut [Laurent<V>]) -> Result<()> {
        let mut statuses = vec![MasterStatus::Success; request.rows()];
        self.evaluate_batch(request, output, &mut statuses)
    }
    fn evaluate_batch(
        &mut self,
        request: MasterRequest<'_, V>,
        output: &mut [Laurent<V>],
        statuses: &mut [MasterStatus],
    ) -> Result<()> {
        let values = match request.integral {
            MasterIntegral::B0 => [Complex::new(1., 2.), c(1.), c(0.)],
            MasterIntegral::B1 => [c(3.), c(-0.5), c(0.)],
            integral => panic!("{integral:?} has a zero weight"),
        };
        for (out, status) in output.iter_mut().zip(statuses) {
            *out = values.map(self.0);
            *status = MasterStatus::Success;
        }
        Ok(())
    }
}

/// A rank n+2 bubble residue with `c0 = c1 = c9 = c21 = c22 = 1`, unit masses
/// and `V1 - V0 = (0, 1, 0, 0)` (`s = -1`, `ke2 = -1`): the masters
/// contribute `B0 - B1`, and the rational part is `7/6 + 23/60 - 41/60 =
/// 13/15` (the terms of c9, c21 and c22; that of c10 vanishes). Returns the
/// contracted row and its
/// term moduli.
fn rank_n2_bubble<V: FloatLike, K: IntegrationKernels<V>>(
    kernels: K,
    lift: fn(Complex<f64>) -> V,
    modulus: impl Fn(&V) -> V,
) -> ([V; 5], Laurent<V>) {
    let zero = lift(c(0.));
    let base = basis();
    let map = |v: &Complex<f64>| lift(*v);
    let basis = CutBasis {
        e1: base.e1.map_ref(map),
        e2: base.e2.map_ref(map),
        e3: base.e3.map_ref(map),
        e4: base.e4.map_ref(map),
        e12: lift(base.e12),
    };
    let shifts = [q([0.; 4]).map_ref(map), q([0., 1., 0., 0.]).map_ref(map)];
    let masses = [lift(c(1.)), lift(c(1.))];
    let scale = lift(c(1.));
    let mut coefficients = vec![zero.clone(); 35];
    for index in [0, 1, 9, 21, 22] {
        coefficients[index] = lift(c(1.));
    }
    let requests = [ResidueIntegrationRequest {
        id: SampleId { lane: 0, sample: 0 },
        partition: &[0, 1],
        shifts: &shifts,
        masses_squared: &masses,
        coefficients: &coefficients,
        basis: Some(&basis),
        scale_squared: &scale,
        s_matrix: None,
    }];
    let mut integration = ResidueIntegrator::new(kernels, zero.clone());
    let mut output = [std::array::from_fn(|_| zero.clone())];
    let mut statuses = [IntegrationStatus::InvalidInput];
    integration
        .evaluate_batch(
            &requests,
            &mut FixedBubbleMasters(lift),
            &mut output,
            &mut statuses,
        )
        .unwrap();
    assert_eq!(statuses, [IntegrationStatus::Success]);
    let [output] = output;
    (output, integration.term_moduli(0, modulus))
}

/// The cancellation test's term moduli: each master times its weight per
/// Laurent coefficient, plus each rational term (or the whole rational part
/// when the kernels evaluate no terms). The contracted values do not depend on
/// the terms, and Dual carriers measure their primal values.
#[test]
fn term_moduli_count_every_master_and_rational_term() {
    let taxicab = |v: &Complex<f64>| c(v.re.abs() + v.im.abs());
    let finite = Complex::new(-2. + 13. / 15., 2.);
    let expected = [finite, c(1.5), c(0.), finite - c(13. / 15.), c(13. / 15.)];
    let (whole, whole_moduli) = rank_n2_bubble(
        SymbolicaIntegrationKernels::new_class(2, RankClass::NPlus2, 0.).unwrap(),
        |v| v,
        taxicab,
    );
    let kernels =
        SymbolicaIntegrationKernels::new_class_with_rational_terms(2, RankClass::NPlus2, 0.)
            .unwrap();
    assert_eq!(kernels.rational_term_count(), 4);
    let (split, split_moduli) = rank_n2_bubble(kernels, |v| v, taxicab);
    assert_eq!(whole, split, "the terms change no contracted value");
    for (value, expected) in whole.iter().zip(expected) {
        close_complex(*value, expected);
    }
    for (moduli, rational) in [(whole_moduli, 13. / 15.), (split_moduli, 67. / 30.)] {
        for (value, expected) in moduli.iter().zip([6. + rational, 1.5, 0.]) {
            close(*value, expected);
        }
    }

    // Dual carriers: derivative components (here 7 times every input and
    // master) do not enter the moduli, which carry the primal value only.
    let shape = vec![vec![0], vec![1]];
    let zero = HyperDual::from_values(shape.clone(), vec![c(0.); 2]);
    let kernels = SymbolicaDualIntegrationKernels::new_class_with_rational_terms(
        2,
        RankClass::NPlus2,
        0.,
        &zero,
    )
    .unwrap();
    assert_eq!(kernels.rational_term_count(), 4);
    let (dual, dual_moduli) = rank_n2_bubble(
        kernels,
        |v| {
            HyperDual::from_values(
                vec![vec![0], vec![1]],
                vec![v, Complex::new(7. * v.re, 7. * v.im)],
            )
        },
        |v| {
            let mut modulus = v.clone();
            modulus.values = vec![taxicab(&v.values[0]), c(0.)];
            modulus
        },
    );
    for (value, expected) in dual.iter().zip(&split) {
        close_complex(value.values[0], *expected);
    }
    for (value, expected) in dual_moduli.iter().zip(&split_moduli) {
        close(value.values[0], expected.re);
        assert_eq!(value.values[1], c(0.));
    }
}
