//! Optional validation export of the exact prepared integrand to original Ninja.
//!
//! Algebra uses Symbolica multiplication/expansion and its supported evaluator;
//! callback generation delegates to the adapter's Symbolica instruction graphs.
//! No reference values pass back into production graph preparation.

use std::{collections::HashMap, error::Error, fmt::Write, fs, path::Path};

use ninjars::Numerator;
use ninjars_symbolica::PolynomialNumerator;
use serde_json::{Value, json};
use symbolica::{
    atom::{Atom, AtomCore},
    domains::float::{Complex, Float, RealLike, SingleFloat},
    symbol,
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

use crate::prepare::PreparedSample as ExportSample;

/// Evaluate only at the independent reference's required binary64 boundary.
/// The higher precision probe also prevents tiny complex components from being
/// silently discarded when checking original Ninja's real-momentum interface.
fn numerical(expression: &Atom, label: &str, real_only: bool) -> Result<Complex<f64>> {
    let values = HashMap::<Atom, Complex<Float>>::new();
    let value = expression
        .evaluate_with_prec(&values, 200)
        .map_err(|error| format!("{label} is not fully bound for reference export: {error}"))?;
    if !value.is_finite() {
        return Err(format!("{label} is nonfinite in reference export").into());
    }
    if real_only && !value.im.is_zero() {
        return Err(format!("{label}: original Ninja requires real external momenta and scales; complex values are unsupported").into());
    }
    let result = Complex::new(value.re.to_f64(), value.im.to_f64());
    if !result.re.is_finite() || !result.im.is_finite() {
        return Err(format!("{label} overflows binary64 in reference export").into());
    }
    if (!value.re.is_zero() && result.re == 0.) || (!value.im.is_zero() && result.im == 0.) {
        return Err(format!("{label} underflows binary64 in reference export").into());
    }
    Ok(result)
}

pub(crate) fn export(
    directory: &Path,
    samples: &[ExportSample],
    scale_squared: &Atom,
) -> Result<Value> {
    if samples.is_empty() {
        return Err("reference export requires at least one fully specified sample".into());
    }
    let scale = numerical(scale_squared, "scale_squared", true)?;
    if scale.re <= 0. {
        return Err("reference scale_squared must be positive".into());
    }
    let variables = [
        Atom::var(symbol!("ninjars::q0")),
        Atom::var(symbol!("ninjars::q1")),
        Atom::var(symbol!("ninjars::q2")),
        Atom::var(symbol!("ninjars::q3")),
        Atom::var(symbol!("ninjars::mu2")),
    ];
    let mut files = Vec::new();
    let mut entries = Vec::new();
    for sample in samples {
        let n = sample.shifts.len();
        if !(1..=6).contains(&n) || n != sample.mass_squared.len() {
            return Err(format!(
                "sample {}: reference runner requires 1..=6 matched propagators",
                sample.lane
            )
            .into());
        }
        // The graph factor and loop-independent propagators enter exactly once,
        // inside the callback; runner results need no subsequent rescaling.
        let full_numerator = (&sample.numerator * &sample.overall_factor).expand();
        let numerator = PolynomialNumerator::new(
            full_numerator.clone(),
            variables.clone(),
            Vec::new(),
            0_f64,
        ).map_err(|error| format!("sample {}: numerator is not fully bound and polynomial for reference export: {error}", sample.lane))?;
        for (index, term) in numerator.polynomial().into_iter().enumerate() {
            numerical(
                term.coefficient,
                &format!("sample {} numerator coefficient {index}", sample.lane),
                false,
            )?;
        }
        let rank = numerator.rank();
        let header = numerator.export_ninja_cpp(n)?;
        let mut input = format!("{n} {rank} 1 {:.17e}\n", scale.re);
        let mut numeric_shifts = Vec::new();
        let mut numeric_masses = Vec::new();
        for (index, (shift, mass)) in sample.shifts.iter().zip(&sample.mass_squared).enumerate() {
            let mut vector = Vec::new();
            for component in shift.components() {
                let value = numerical(
                    component,
                    &format!("sample {} shift {index}", sample.lane),
                    true,
                )?;
                vector.push(value.re);
                write!(input, "{:.17e} ", value.re)?;
            }
            let mass = numerical(
                mass,
                &format!("sample {} mass_squared {index}", sample.lane),
                false,
            )?;
            writeln!(input, "{:.17e} {:.17e}", mass.re, mass.im)?;
            numeric_shifts.push(vector);
            numeric_masses.push([mass.re, mass.im]);
        }
        let header_name = format!("sample-{:06}.hh", sample.lane);
        let input_name = format!("sample-{:06}.inputs", sample.lane);
        entries.push(json!({
            "lane": sample.lane,
            "header": header_name,
            "input": input_name,
            "propagator_count": n,
            "rank": rank,
            "parameter_count": 0,
            "full_numerator": full_numerator.to_canonical_string(),
            "shifts_binary64": numeric_shifts,
            "mass_squared_binary64": numeric_masses,
        }));
        files.push((header_name, header));
        files.push((input_name, input));
    }
    let manifest = json!({
        "schema_version": 1,
        "operation": "original_ninja_reference_export",
        "reference_precision": "binary64",
        "numerical_conversion": "Symbolica 200-bit evaluation followed by binary64 rounding; exact expressions retained in prepared.json",
        "callback_precision": "Symbolica exact instruction graph exported to C++ complex<double>",
        "overall_factor_included_in_callback": true,
        "epsilon_order": ["finite", "single_pole", "double_pole"],
        "scale_squared_exact": scale_squared.to_canonical_string(),
        "scale_squared_binary64": scale.re,
        "samples": entries,
    });
    files.push((
        "manifest.json".into(),
        serde_json::to_string_pretty(&manifest)? + "\n",
    ));
    // Validate and generate every sample before writing a reviewable bundle.
    fs::create_dir_all(directory)?;
    for (name, contents) in files {
        fs::write(directory.join(name), contents)?;
    }
    Ok(manifest)
}
