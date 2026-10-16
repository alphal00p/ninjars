//! Generate original Ninja callbacks from a polynomial in q0,q1,q2,q3,mu2.
//!
//! cargo run -p ninjars-symbolica --features reference-codegen \
//!   --example generate_reference -- 3 'q0^2+mu2^2' target/triangle.hh

use std::{env, error::Error, fs, path::Path};

use ninjars_symbolica::PolynomialNumerator;
use symbolica::atom::Atom;

fn main() -> Result<(), Box<dyn Error>> {
    let arguments: Vec<_> = env::args().skip(1).collect();
    if arguments.len() < 3 {
        return Err(
            "usage: generate_reference PROPAGATORS EXPRESSION HEADER [PARAMETER ...]".into(),
        );
    }
    let propagators = arguments[0].parse()?;
    let parse = |input: &str| Atom::parse(input, "ninjars_reference", Default::default());
    let expression = parse(&arguments[1])?;
    let parameters = arguments[3..]
        .iter()
        .map(|name| parse(name))
        .collect::<Result<_, _>>()?;
    let variables = ["q0", "q1", "q2", "q3", "mu2"].map(|name| parse(name).unwrap());
    let numerator = PolynomialNumerator::new(expression, variables, parameters, 0.)?;
    let output = Path::new(&arguments[2]);
    if let Some(parent) = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    fs::write(output, numerator.export_ninja_cpp(propagators)?)?;
    println!("{}", output.display());
    Ok(())
}
