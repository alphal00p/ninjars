use ninjars::{Complex, ExpansionKind};
use ninjars_symbolica::{LaurentDivision, ResidueProjection};
use std::collections::HashMap;
use symbolica::{
    atom::{Atom, AtomCore},
    domains::{
        atom::AtomField,
        dual::{DualNumber, HyperDual},
    },
    parse,
};

fn c(value: f64) -> Complex<f64> {
    Complex::new(value, 0.)
}

// Independent residue ansatz from original Ninja's polys_implem.cxx.
fn residue(cut: usize, higher: bool) -> Atom {
    let ordinary = if cut == 3 {
        "1+8*mu+(2+9*mu)*u+3*u^2+4*u^3+(5+10*mu)*v+6*v^2+7*v^3"
    } else {
        "1+10*mu+2*w+3*w^2+4*u+5*u^2+6*v+7*v^2+8*w*u+9*w*v"
    };
    let high = if cut == 3 {
        "15*mu^2+11*mu*u^2+13*u^4+12*mu*v^2+14*v^4"
    } else {
        "mu*(11*w+12*u+13*v)+14*w^3+15*u^3+16*v^3+17*w^2*u+18*w^2*v+19*w*u^2+20*w*v^2"
    };
    if higher {
        parse!(ordinary) + parse!(high)
    } else {
        parse!(ordinary)
    }
}

#[test]
fn projected_coefficients_recover_original_residue_ansatz() {
    let field = AtomField {
        statistical_zero_test: false,
        ..AtomField::new()
    };
    let variables = [parse!("t"), parse!("x"), parse!("mu")];
    for cut in [2, 3] {
        for higher in [false, true] {
            let division = LaurentDivision::new(cut, cut + usize::from(higher), cut).unwrap();
            let plan = ResidueProjection::new(&division).unwrap();
            let mut evaluator = plan.compile(&0.).unwrap();
            let mut inputs = Vec::new();
            for g in [2, -3, 7] {
                for swapped in [false, true] {
                    let growing = parse!("t") * Atom::num(-g);
                    let reciprocal = parse!("(5+3*x+7*x^2+mu)/(2*t)");
                    let (u, v) = if swapped {
                        (growing, reciprocal)
                    } else {
                        (reciprocal, growing)
                    };
                    let substituted = residue(cut, higher)
                        .replace(parse!("u"))
                        .with(u.to_pattern())
                        .replace(parse!("v"))
                        .with(v.to_pattern())
                        .replace(parse!("w"))
                        .with((parse!("x") * Atom::num(g)).to_pattern())
                        .expand();
                    let polynomial =
                        substituted.to_polynomial_in_vars_with_field::<i32>(&variables, &field);
                    let powers: Vec<_> = match plan.quotient_expansion() {
                        ExpansionKind::T3 { powers } => {
                            powers.iter().map(|&(a, b)| [a, 0, i32::from(b)]).collect()
                        }
                        ExpansionKind::T2 { powers } => powers
                            .iter()
                            .map(|&(a, b, c)| [a, i32::from(b), i32::from(c)])
                            .collect(),
                        _ => unreachable!(),
                    };
                    for power in powers {
                        inputs.push(
                            polynomial
                                .coefficient(&power)
                                .unwrap_or(Atom::Zero)
                                .evaluate(&HashMap::<Atom, Complex<f64>>::new())
                                .unwrap(),
                        );
                    }
                }
                inputs.push(c(g as f64));
            }
            let mut outputs = vec![c(0.); 3 * plan.coefficient_count()];
            evaluator.evaluate_batch(&inputs, &mut outputs, 3).unwrap();
            for row in outputs.chunks_exact(plan.coefficient_count()) {
                for (index, value) in row.iter().enumerate() {
                    assert!(
                        (value.re - (index + 1) as f64).abs() < 1e-11,
                        "cut{cut} high{higher} c{index}: {value}"
                    );
                    assert!(value.im.abs() < 1e-12);
                }
            }
        }
    }
}

#[test]
fn projection_carries_mixed_basis_derivatives() {
    let division = LaurentDivision::new(3, 3, 3).unwrap();
    let plan = ResidueProjection::new(&division).unwrap();
    let prototype = HyperDual::from_values(
        vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![1, 1], vec![0, 2]],
        vec![c(0.); 5],
    );
    let mut kernel = plan.compile_dual(&0., &prototype).unwrap();
    let mut input = vec![prototype.clone(); plan.input_columns()];
    let ExpansionKind::T3 { powers } = plan.quotient_expansion() else {
        unreachable!()
    };
    let i = powers.iter().position(|&p| p == (1, 0)).unwrap();
    input[i].values_mut()[0] = c(6.);
    input[i].values_mut()[1] = c(1.);
    let last = input.last_mut().unwrap();
    last.values_mut()[0] = c(2.);
    last.values_mut()[2] = c(1.);
    let mut out = vec![prototype; plan.coefficient_count()];
    kernel.evaluate_batch(&input, &mut out, 1).unwrap();
    // c4 = -a/g; repeated y component is the Taylor coefficient, not d²/dy².
    assert_eq!(
        out[4].values(),
        &[c(-3.), c(-0.5), c(1.5), c(0.25), c(-0.75)]
    );
}
