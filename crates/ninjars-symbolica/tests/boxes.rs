//! Independent box-residue controls across both rational extraction modes.

use ninjars::{
    BoxCoefficientMode, BoxDivisionRequest, BoxMuMode, BoxReducer, BoxReductionKernels,
    BoxReductionOptions, Complex, CutFailure, CutOutcome, DoubleFloat, Float, FourMomentum,
    Numerator, PentagonReducer, Real, SingleFloat, Topology,
};
use ninjars_symbolica::{
    DualPolynomialNumerator, PolynomialNumerator, SymbolicaBoxKernels, SymbolicaDualBoxKernels,
    SymbolicaDualHighCuts, SymbolicaHighCuts, SymbolicaPentagonKernels,
};
use symbolica::{
    atom::{Atom, AtomCore},
    domains::{dual::HyperDual, rational::Rational},
    evaluate::EvaluationDomain,
    parse,
};

fn precise_complex_mass_box<T: Real + SingleFloat + PartialOrd>(prototype: T, digits: u64)
where
    Complex<T>: Real + EvaluationDomain,
{
    let zero = Complex::new(prototype.zero(), prototype.zero());
    let value = Complex::new(
        prototype.one() + prototype.from_rational(&Rational::from((1, 10)).pow(30)),
        prototype.zero(),
    );
    let p: Vec<_> = ROUTING[..4]
        .iter()
        .map(|p| {
            FourMomentum::from(p.map(|v| Complex::new(prototype.from_i64(v), prototype.zero())))
        })
        .collect();
    let mass = Complex::new(
        prototype.one(),
        prototype.from_rational(&Rational::from((-1, 10))),
    );
    let scale = Complex::new(prototype.from_i64(2), prototype.zero());
    let tolerance = prototype.from_rational(&Rational::from((1, 10)).pow(digits));
    let mut geometry = SymbolicaHighCuts::new(prototype.clone()).unwrap();
    for mode in [BoxMuMode::Expansion, BoxMuMode::Sampling] {
        let mut numerator = PolynomialNumerator::new(
            parse!("a*(1+mu2^2)"),
            vars(),
            vec![parse!("a")],
            prototype.clone(),
        )
        .unwrap();
        numerator
            .set_parameter_batch(std::slice::from_ref(&value), 1)
            .unwrap();
        let mut reducer = BoxReducer::new(
            SymbolicaBoxKernels::new(4, 4, options(true, mode), prototype.clone()).unwrap(),
            zero.clone(),
        )
        .unwrap();
        let mut output = vec![CutOutcome::Unstable(CutFailure::NonFiniteResult)];
        reducer
            .evaluate_batch(
                &p,
                &vec![mass.clone(); 4],
                std::slice::from_ref(&scale),
                1,
                &Topology::new(4, 4).unwrap(),
                &[],
                &mut numerator,
                &mut geometry,
                &mut output,
            )
            .unwrap();
        let CutOutcome::Stable(residue) = &output[0] else {
            panic!("{mode:?}: {:?}", output[0])
        };
        for &index in &[0, 4] {
            let got = &residue.coefficients[index];
            assert_ne!(got.re, prototype.one());
            assert!(
                (got.re.clone() - &value.re).norm() < tolerance,
                "{mode:?}: {got} != {value}"
            );
            assert!(got.im.norm() < tolerance);
        }
        for &index in &[1, 2, 3] {
            assert!(residue.coefficients[index].re.norm() < tolerance);
            assert!(residue.coefficients[index].im.norm() < tolerance);
        }
    }
}

#[test]
fn complex_mass_box_coefficients_preserve_doublefloat_and_float192_inputs() {
    precise_complex_mass_box(DoubleFloat::default(), 31);
    precise_complex_mass_box(Float::new(192), 50);
}

fn c(v: f64) -> Complex<f64> {
    Complex::new(v, 0.)
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
const ROUTING: [[i64; 4]; 6] = [
    [0, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [2, 2, 3, 4],
    [-3, 2, -1, 5],
];
fn shifts(n: usize) -> Vec<FourMomentum<Complex<f64>>> {
    ROUTING[..n]
        .iter()
        .map(|p| FourMomentum::from(p.map(|v| c(v as f64))))
        .collect()
}
fn near(a: Complex<f64>, b: Complex<f64>) {
    assert!(
        (a.re - b.re).abs() < 1e-9 * (1. + b.re.abs())
            && (a.im - b.im).abs() < 1e-9 * (1. + b.im.abs()),
        "{a} != {b}"
    );
}
fn options(full: bool, mu: BoxMuMode) -> BoxReductionOptions {
    BoxReductionOptions {
        coefficients: if full {
            BoxCoefficientMode::Full
        } else {
            BoxCoefficientMode::Integrated
        },
        mu_extraction: mu,
    }
}
fn polynomial(rank: usize) -> Atom {
    let terms = ["1", "2*q0", "3*mu2", "4*q0*mu2", "5*mu2^2", "6*q0*mu2^2"];
    terms[..=rank.min(5)]
        .iter()
        .fold(Atom::Zero, |sum, t| sum + parse!(*t))
}
fn denominator(p: [i64; 4]) -> Atom {
    let q = vars();
    let v = FourMomentum::from_args(&q[0] + p[0], &q[1] + p[1], &q[2] + p[2], &q[3] + p[3]);
    v.dot(&v) - 1 - &q[4]
}

#[test]
fn all_box_ranks_modes_and_batch_tails_recover_canonical_coefficients() {
    let mut geometry = SymbolicaHighCuts::new(0.).unwrap();
    for rank in 0..=5 {
        for full in [false, true] {
            for mode in [BoxMuMode::Expansion, BoxMuMode::Sampling] {
                let mut numerator = PolynomialNumerator::new(
                    parse!("a") * polynomial(rank),
                    vars(),
                    vec![parse!("a")],
                    0.,
                )
                .unwrap();
                assert_eq!(numerator.rank(), rank);
                let mut reducer = BoxReducer::new(
                    SymbolicaBoxKernels::new(4, rank, options(full, mode), 0.).unwrap(),
                    c(0.),
                )
                .unwrap();
                let topology = Topology::new(4, rank).unwrap();
                for lanes in [
                    &[2., -1., 3.][..],
                    &[4.][..],
                    &[][..],
                    &[3., 2., -1., 4., 5.][..],
                ] {
                    numerator
                        .set_parameter_batch(
                            &lanes.iter().copied().map(c).collect::<Vec<_>>(),
                            lanes.len(),
                        )
                        .unwrap();
                    let p: Vec<_> = (0..lanes.len()).flat_map(|_| shifts(4)).collect();
                    let mut output =
                        vec![CutOutcome::Unstable(CutFailure::NonFiniteResult); lanes.len()];
                    reducer
                        .evaluate_batch(
                            &p,
                            &vec![c(1.); p.len()],
                            &vec![c(2.); lanes.len()],
                            lanes.len(),
                            &topology,
                            &[],
                            &mut numerator,
                            &mut geometry,
                            &mut output,
                        )
                        .unwrap();
                    for (lane, out) in output.iter().enumerate() {
                        let CutOutcome::Stable(residue) = out else {
                            panic!("rank={rank},full={full},mode={mode:?}: {out:?}")
                        };
                        assert_eq!(residue.id.lane, lane);
                        for (i, value) in residue.coefficients.iter().enumerate() {
                            let computed = full || i == 0 || i == 4;
                            assert_eq!(residue.computed[i], computed);
                            let expected = if computed && i <= rank {
                                (i + 1) as f64 * lanes[lane]
                            } else {
                                0.
                            };
                            near(*value, c(expected));
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn five_and_six_line_boxes_divide_all_uncut_lines_and_subtract_actual_pentagons() {
    let mut geometry = SymbolicaHighCuts::new(0.).unwrap();
    for n in [5, 6] {
        for higher in [false, true] {
            let rank = n + usize::from(higher);
            // On the target box, D_j-D_0 equals D_j and is only linear in q.
            // This realizes a complete degree4/5 box residue at rank n/n+1.
            let mut expression = ROUTING[4..n]
                .iter()
                .fold(polynomial(if higher { 5 } else { 4 }), |value, &p| {
                    value * (denominator(p) - denominator(ROUTING[0]))
                })
                .expand();
            // Add nonzero canonical pentagon residues. Each source contributes
            // c*mu² times precisely its complementary propagator product.
            for parent in 4..n {
                expression += (4..n).filter(|&line| line != parent).fold(
                    parse!("mu2") * ((7 + 4 * (parent - 4)) as i64),
                    |value, line| value * denominator(ROUTING[line]),
                );
            }
            let mut numerator = PolynomialNumerator::new(expression, vars(), vec![], 0.).unwrap();
            assert_eq!(numerator.rank(), rank);
            numerator.set_parameter_batch(&[], 1).unwrap();
            let p = shifts(n);
            let masses = vec![c(1.); n];
            let mut pentagon =
                PentagonReducer::new(SymbolicaPentagonKernels::new(n, rank, 0.).unwrap(), c(0.))
                    .unwrap();
            let mut parents =
                vec![CutOutcome::Unstable(CutFailure::NonFiniteResult); pentagon.cut_count()];
            pentagon
                .evaluate_batch(&p, &masses, 1, &mut numerator, &mut geometry, &mut parents)
                .unwrap();
            assert!(
                parents.iter().all(|p| matches!(p, CutOutcome::Stable(_))),
                "n={n}: {parents:?}"
            );
            for parent in &parents {
                let CutOutcome::Stable(parent) = parent else {
                    unreachable!()
                };
                let partition = &pentagon.topology().cuts()[parent.cut_index].partition;
                if (0..4).all(|line| partition.contains(&line)) {
                    let extra = *partition.iter().find(|&&line| line >= 4).unwrap();
                    near(parent.coefficient, c((7 + 4 * (extra - 4)) as f64));
                }
            }
            for mode in [BoxMuMode::Expansion, BoxMuMode::Sampling] {
                let mut reducer = BoxReducer::new(
                    SymbolicaBoxKernels::new(n, rank, options(true, mode), 0.).unwrap(),
                    c(0.),
                )
                .unwrap();
                let mut output =
                    vec![CutOutcome::Unstable(CutFailure::NonFiniteResult); reducer.cut_count()];
                reducer
                    .evaluate_batch(
                        &p,
                        &masses,
                        &[c(2.)],
                        1,
                        pentagon.topology(),
                        &parents,
                        &mut numerator,
                        &mut geometry,
                        &mut output,
                    )
                    .unwrap();
                // The first four-cut is [0,1,2,3], with u=q0 in its unit time chart.
                let CutOutcome::Stable(target) = &output[0] else {
                    panic!("n={n},mode={mode:?}: {:?}", output[0])
                };
                assert_eq!(
                    reducer.topology().cuts()[target.cut_index].partition,
                    [0, 1, 2, 3]
                );
                for (i, value) in target.coefficients.iter().enumerate() {
                    near(*value, c((i + 1) as f64));
                }
            }
        }
    }
}

#[test]
fn higher_rank_sampling_interpolation_resolves_reference_scale_hypothesis() {
    // Substitution into the original odd part c1+c3*mu+c5*mu².
    // Its +s provisional c3 is c3+c5*s; the source's final '-c5' leaves
    // a residual c5*(s-1). Symbolica verifies this independently of our plan.
    let residual = parse!("((c1+c3*s+c5*s^2)-c1)/s-c5-c3")
        .expand()
        .to_rational_polynomial::<_, _, u8>(
            &symbolica::domains::rational::Q,
            &symbolica::domains::integer::Z,
            None,
        )
        .to_expression()
        .expand();
    assert_eq!(residual, parse!("c5*s-c5").expand());
    // The canonical c5=6 test above uses s=2 and recovers c3=4 rather than10.
}

#[test]
fn mixed_higher_dual_components_survive_both_box_extraction_modes() {
    let shape = vec![
        vec![0, 0],
        vec![1, 0],
        vec![0, 1],
        vec![2, 0],
        vec![1, 1],
        vec![2, 1],
    ];
    let dual = |v: [f64; 6]| HyperDual::from_values(shape.clone(), v.map(c).to_vec());
    let zero = dual([0.; 6]);
    let one = dual([1., 0., 0., 0., 0., 0.]);
    let coefficient = dual([2., 3., 5., 7., 11., 13.]);
    let p: Vec<_> = ROUTING[..4]
        .iter()
        .map(|p| FourMomentum::from(p.map(|v| dual([v as f64, 0., 0., 0., 0., 0.]))))
        .collect();
    let masses = vec![dual([1., 1., 0., 0., 0., 0.]); 4];
    let mut geometry = SymbolicaDualHighCuts::new(0., zero.clone()).unwrap();
    for mode in [BoxMuMode::Expansion, BoxMuMode::Sampling] {
        let source =
            PolynomialNumerator::new(parse!("a") * polynomial(5), vars(), vec![parse!("a")], 0.)
                .unwrap();
        let mut numerator = DualPolynomialNumerator::new(source, zero.clone()).unwrap();
        numerator
            .set_parameter_batch(std::slice::from_ref(&coefficient), 1)
            .unwrap();
        let mut reducer = BoxReducer::new(
            SymbolicaDualBoxKernels::new(4, 5, options(true, mode), 0., &zero).unwrap(),
            zero.clone(),
        )
        .unwrap();
        let mut output = vec![CutOutcome::Unstable(CutFailure::NonFiniteResult)];
        reducer
            .evaluate_batch(
                &p,
                &masses,
                std::slice::from_ref(&one),
                1,
                &Topology::new(4, 5).unwrap(),
                &[],
                &mut numerator,
                &mut geometry,
                &mut output,
            )
            .unwrap();
        let CutOutcome::Stable(residue) = &output[0] else {
            panic!("{mode:?}: {:?}", output[0])
        };
        for (i, value) in residue.coefficients.iter().enumerate() {
            for (actual, expected) in value.values.iter().zip(&coefficient.values) {
                near(*actual, *expected * c((i + 1) as f64));
            }
        }
    }
}

#[test]
fn box_lane_failures_and_parent_api_errors_preserve_other_results() {
    let n = 5;
    let rank = 6;
    let topology = Topology::new(n, rank).unwrap();
    let mut p: Vec<_> = (0..2).flat_map(|_| shifts(n)).collect();
    let masses = vec![c(1.); 2 * n];
    let mut geometry = SymbolicaHighCuts::new(0.).unwrap();
    let mut numerator = PolynomialNumerator::new(parse!("1+q0^6"), vars(), vec![], 0.).unwrap();
    numerator.set_parameter_batch(&[], 2).unwrap();
    let mut pentagon =
        PentagonReducer::new(SymbolicaPentagonKernels::new(n, rank, 0.).unwrap(), c(0.)).unwrap();
    let mut parents =
        vec![CutOutcome::Unstable(CutFailure::NonFiniteResult); 2 * pentagon.cut_count()];
    pentagon
        .evaluate_batch(&p, &masses, 2, &mut numerator, &mut geometry, &mut parents)
        .unwrap();
    let mut reducer = BoxReducer::new(
        SymbolicaBoxKernels::new(n, rank, options(true, BoxMuMode::Sampling), 0.).unwrap(),
        c(0.),
    )
    .unwrap();
    let mut output =
        vec![CutOutcome::Unstable(CutFailure::NonFiniteResult); 2 * reducer.cut_count()];
    parents[1] = CutOutcome::Unstable(CutFailure::DegenerateGram);
    reducer
        .evaluate_batch(
            &p,
            &masses,
            &[c(2.); 2],
            2,
            &topology,
            &parents,
            &mut numerator,
            &mut geometry,
            &mut output,
        )
        .unwrap();
    assert!(matches!(output[0], CutOutcome::Stable(_)));
    assert!(matches!(
        output[1],
        CutOutcome::Unstable(CutFailure::DegenerateGram)
    ));
    let before = format!("{output:?}");
    if let CutOutcome::Stable(parent) = &mut parents[0] {
        parent.id.sample += 1;
    } else {
        panic!("first parent failed")
    }
    assert!(
        reducer
            .evaluate_batch(
                &p,
                &masses,
                &[c(2.); 2],
                2,
                &topology,
                &parents,
                &mut numerator,
                &mut geometry,
                &mut output
            )
            .is_err()
    );
    assert_eq!(format!("{output:?}"), before);
    // Minimal expansion mode deliberately does not depend on pentagons.
    let mut minimal = BoxReducer::new(
        SymbolicaBoxKernels::new(n, rank, options(false, BoxMuMode::Expansion), 0.).unwrap(),
        c(0.),
    )
    .unwrap();
    p[n + 4].spatial.pz = c(f64::NAN); // only an uncut line for target [0,1,2,3].
    minimal
        .evaluate_batch(
            &p,
            &masses,
            &[c(2.); 2],
            2,
            &topology,
            &[],
            &mut numerator,
            &mut geometry,
            &mut output,
        )
        .unwrap();
    assert!(matches!(output[0], CutOutcome::Stable(_)));
    assert!(matches!(
        output[1],
        CutOutcome::Unstable(CutFailure::NonFiniteInput)
    ));
}

#[test]
fn direct_sample_division_isolates_bad_dual_shapes_and_rejects_duplicate_partition() {
    let shape = vec![vec![0], vec![1], vec![2]];
    let dual = |v| HyperDual::from_values(shape.clone(), vec![c(v), c(0.), c(0.)]);
    let zero = dual(0.);
    let one = dual(1.);
    let p: Vec<_> = ROUTING[..5]
        .iter()
        .map(|p| FourMomentum::from(p.map(|v| dual(v as f64))))
        .collect();
    let good_masses = vec![one.clone(); 5];
    let mut bad_masses = good_masses.clone();
    bad_masses[4] = HyperDual::from_values(vec![vec![0], vec![1]], vec![c(1.), c(0.)]);
    let q = FourMomentum::from([dual(2.), dual(3.), dual(4.), dual(5.)]);
    let make = |masses| BoxDivisionRequest {
        partition: &[0, 1, 2, 3],
        complement: &[4],
        shifts: &p,
        masses_squared: masses,
        momentum: &q,
        mu_squared: &zero,
        pentagon_coefficients: None,
    };
    let requests = [make(good_masses.as_slice()), make(bad_masses.as_slice())];
    let mut kernel =
        SymbolicaDualBoxKernels::new(5, 6, options(false, BoxMuMode::Expansion), 0., &zero)
            .unwrap();
    let sentinel = HyperDual::from_values(shape, vec![c(13.), c(17.), c(19.)]);
    let mut output = vec![sentinel.clone(); 2];
    let mut stable = [true; 2];
    kernel
        .divide_samples_batch(
            &requests,
            &[one.clone(), one.clone()],
            &mut output,
            &mut stable,
        )
        .unwrap();
    assert_eq!(stable, [true, false]);
    assert_eq!(output[1].values, sentinel.values);
    let invalid = BoxDivisionRequest {
        partition: &[0, 1, 1, 3],
        ..make(good_masses.as_slice())
    };
    let before = output.iter().map(|x| x.values.clone()).collect::<Vec<_>>();
    let statuses = stable;
    assert!(
        kernel
            .divide_samples_batch(
                &[invalid, make(good_masses.as_slice())],
                &[one.clone(), one],
                &mut output,
                &mut stable
            )
            .is_err()
    );
    assert_eq!(stable, statuses);
    assert_eq!(
        output.iter().map(|x| x.values.clone()).collect::<Vec<_>>(),
        before
    );
}
