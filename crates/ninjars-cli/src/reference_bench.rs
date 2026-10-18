//! Common-template benchmark export. All numerical rows use the production
//! f64 binding path; no reference values enter production preparation.

use std::{error::Error, fmt::Write, fs, path::Path};

use ninjars::{Complex, Numerator};
use ninjars_symbolica::PolynomialNumerator;
use serde_json::{Value, json};
use symbolica::{
    atom::{Atom, AtomCore},
    symbol,
};

use crate::{prepare::PreparedDocument, runtime::NumericalPool};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn real(value: &Complex<f64>) -> Result<f64> {
    if value.im != 0. || !value.re.is_finite() {
        return Err(
            "original Ninja benchmark requires finite real shifts, scales and S-matrices".into(),
        );
    }
    Ok(value.re)
}

pub(crate) fn export(directory: &Path, prepared: &PreparedDocument) -> Result<Value> {
    let n = prepared.propagators;
    if n == 0 || prepared.samples.is_empty() {
        return Err("reference benchmark requires propagators and a nonempty sample pool".into());
    }
    // Check original Ninja's real-input domain before the f64 binding can
    // underflow an imaginary component. The values written below still come
    // exclusively from the same pool used by the Rust benchmark.
    crate::reference::numerical(&prepared.scale_squared, "scale_squared", true)?;
    for sample in &prepared.samples {
        for shift in &sample.shifts {
            for component in shift.components() {
                crate::reference::numerical(component, "shift component", true)?;
            }
        }
        if let Some(matrix) = &sample.s_matrix {
            for value in matrix {
                crate::reference::numerical(value, "S-matrix component", true)?;
            }
        }
    }
    let variables =
        ["q0", "q1", "q2", "q3", "mu2"].map(|name| Atom::var(symbol!(format!("ninjars::{name}"))));
    // This template already includes the complete graph factor exactly once.
    // Its rank stays fixed even when a particular parameter row lowers it.
    let numerator = PolynomialNumerator::new(
        prepared.numerator_template.clone(),
        variables,
        prepared.parameters.clone(),
        0_f64,
    )?;
    let header = numerator.export_ninja_cpp(n)?;
    let pool = NumericalPool::new(prepared, &0_f64)?;
    let mut input = format!(
        "NINJARS_BENCH_V1\n{n} {} {} {}\n",
        numerator.rank(),
        prepared.parameters.len(),
        prepared.samples.len()
    );
    let mut samples = Vec::new();
    for lane in 0..prepared.samples.len() {
        let batch = pool.batch(&[lane])?;
        let row = batch.input();
        let scale = real(&row.scales_squared[0])?;
        writeln!(
            input,
            "{scale:.17e} {}",
            usize::from(row.s_matrix.is_some())
        )?;
        let mut shifts = Vec::new();
        let mut masses = Vec::new();
        for (shift, mass) in row.shifts.iter().zip(row.masses_squared) {
            let values = shift
                .components()
                .map(real)
                .into_iter()
                .collect::<Result<Vec<_>>>()?;
            writeln!(
                input,
                "{:.17e} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e}",
                values[0], values[1], values[2], values[3], mass.re, mass.im
            )?;
            shifts.push(values);
            masses.push([mass.re, mass.im]);
        }
        for parameter in batch.parameters() {
            writeln!(input, "{:.17e} {:.17e}", parameter.re, parameter.im)?;
        }
        let matrix = row
            .s_matrix
            .map(|values| values.iter().map(real).collect::<Result<Vec<_>>>())
            .transpose()?;
        if let Some(matrix) = &matrix {
            for i in 0..n {
                if matrix[i * n + i] != 0. || (0..i).any(|j| matrix[i * n + j] != matrix[j * n + i])
                {
                    return Err("reference S-matrix must be symmetric with a zero diagonal".into());
                }
            }
            for value in matrix {
                writeln!(input, "{value:.17e}")?;
            }
        }
        samples.push(json!({"lane":prepared.samples[lane].lane,"shifts_binary64":shifts,
                           "mass_squared_binary64":masses,"scale_squared_binary64":scale,
                           "parameters_binary64":batch.parameters().iter().map(|v|[v.re,v.im]).collect::<Vec<_>>(),
                           "s_matrix_binary64":matrix}));
    }
    let manifest = json!({
        "schema_version":1,"operation":"original_ninja_benchmark_export",
        "header":"numerator.hh","pool":"pool.inputs","protocol":"NINJARS_BENCH_V1",
        "pool_protocol_scope":"header and sample pool only; append job_count and job rows before execution",
        "propagator_count":n,"rank":numerator.rank(),"parameter_count":prepared.parameters.len(),
        "numerator_template":prepared.numerator_template.to_canonical_string(),
        "parameter_order":prepared.parameters.iter().map(AtomCore::to_canonical_string).collect::<Vec<_>>(),
        "overall_factor_included_in_callback":true,"samples":samples,
        "numerical_binding":"same NumericalPool<f64> path as Rust bench; complex values encoded as [real,imaginary]",
        "callback_coefficients":"Symbolica 200-bit constant resolution, C++ literals rounded to binary64",
        "callback_depths":"separate Symbolica evaluator for each T3/T2 requested depth",
        "performance_acceptance":false,
    });
    fs::create_dir_all(directory)?;
    fs::write(directory.join("numerator.hh"), header)?;
    fs::write(directory.join("pool.inputs"), input)?;
    fs::write(
        directory.join("manifest.json"),
        serde_json::to_string_pretty(&manifest)? + "\n",
    )?;
    Ok(manifest)
}
