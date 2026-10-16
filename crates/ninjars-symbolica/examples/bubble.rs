//! Complete batched bubble reduction, including lower cuts and master integrals.

use ninjars::{Complex, FourMomentum, Numerator};
use ninjars_symbolica::{
    PolynomialNumerator, SymbolicaAmplitudeKernels, SymbolicaCuts, SymbolicaMasters,
};
use symbolica::parse;

fn evaluate() -> Result<(), Box<dyn std::error::Error>> {
    let c = |v| Complex::new(v, 0.);
    let q = |components: [f64; 4]| FourMomentum::from(components.map(c));
    let mut numerator = PolynomialNumerator::new(
        parse!("(q0-q1)^3"),
        [
            parse!("q0"),
            parse!("q1"),
            parse!("q2"),
            parse!("q3"),
            parse!("mu2"),
        ],
        vec![],
        0_f64,
    )?;
    numerator.set_parameter_batch(&[], 2)?;
    let mut amplitude =
        SymbolicaAmplitudeKernels::new(2, numerator.rank(), 0.)?.into_amplitude(2)?;
    let shifts = [
        q([0.; 4]),
        q([0., 1., 0., 0.]),
        q([0.; 4]),
        q([0., 2., 0., 0.]),
    ];
    amplitude.evaluate_batch(
        &shifts,
        &[c(1.); 4],
        &[c(1.); 2],
        &mut numerator,
        &mut SymbolicaCuts::new(0.)?,
        &mut SymbolicaMasters::new(0.),
    )?;
    for (lane, (value, status)) in amplitude
        .results()
        .iter()
        .zip(amplitude.statuses())
        .enumerate()
    {
        println!(
            "lane {lane}: {status:?}, finite={}, pole={}, double_pole={}, rational={}",
            value[0], value[1], value[2], value[4]
        );
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // oneloopmaster expression construction needs a larger thread stack.
    std::thread::Builder::new()
        .name("ninjars-example".into())
        .stack_size(128 * 1024 * 1024)
        .spawn(|| evaluate().map_err(|e| e.to_string()))?
        .join()
        .map_err(|_| "evaluation thread panicked")??;
    Ok(())
}
