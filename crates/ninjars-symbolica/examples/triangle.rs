//! Run the triangle-residue stage for a numerator with an explicit mu² term.
//! Complete amplitude reduction also needs the lower-cut stages.

use ninjars::{Complex, CutFailure, CutOutcome, FourMomentum, TriangleReducer};
use ninjars_symbolica::{PolynomialNumerator, SymbolicaCuts, SymbolicaTriangleKernels};
use symbolica::parse;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let complex = |v| Complex::new(v, 0.);
    let momentum = |[e, x, y, z]: [f64; 4]| {
        FourMomentum::from_args(complex(e), complex(x), complex(y), complex(z))
    };
    let mut numerator = PolynomialNumerator::new(
        parse!("1+mu2"),
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
    numerator.set_parameter_batch(&[], 1)?;
    let mut reducer = TriangleReducer::new(SymbolicaTriangleKernels::new(3, 2, 0.)?, complex(0.))?;
    let mut geometry = SymbolicaCuts::new(0.)?;
    let shifts = [
        momentum([0., 0., 0., 0.]),
        momentum([2., 1., 0., 0.]),
        momentum([3., 0., 1., 0.]),
    ];
    let mut residues = vec![CutOutcome::Unstable(CutFailure::NonFiniteResult); reducer.cut_count()];
    reducer.evaluate_batch(
        &shifts,
        &[complex(1.); 3],
        1,
        &mut numerator,
        &mut geometry,
        &mut residues,
    )?;
    for result in residues {
        match result {
            CutOutcome::Stable(residue) => println!(
                "lane {}: c0={}, c7={}",
                residue.id.lane, residue.coefficients[0], residue.coefficients[7]
            ),
            CutOutcome::Unstable(reason) => println!("unstable triangle: {reason:?}"),
        }
    }
    Ok(())
}
