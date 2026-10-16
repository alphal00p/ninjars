//! Original-Ninja callback export for independent reference reduction.
//!
//! Symbolica performs substitution, Laurent coefficient extraction, optimization,
//! and arithmetic code generation. The C++ wrapper only adapts callback layouts.

use std::fmt::Write;

use ninjars::{Error, ExpansionKind, Numerator, Result};
use symbolica::{
    domains::float::{Complex, Float, Real, SingleFloat},
    evaluate::{CompiledNumber, EvaluationDomain, ExportSettings, InlineASM},
};

use crate::PolynomialNumerator;

impl<T: Real + SingleFloat> PolynomialNumerator<T>
where
    Complex<T>: Real + EvaluationDomain,
{
    /// Export the same exact numerator as original Ninja's C++ callbacks.
    ///
    /// The returned single-translation-unit header declares
    /// `ninjars_reference::GeneratedNumerator`, constructed from a pointer to
    /// complex double external parameters in [`Self::parameter_names`] order.
    /// `set_parameters` rebinds that borrowed row without allocation; it must
    /// remain valid while callbacks use it. Each Laurent depth has its own
    /// evaluator and computes only the requested coefficients.
    /// No parameter values from this instance are embedded. Exact coefficients
    /// and registered constants such as π are mapped by Symbolica at 200 bits
    /// before emission; the C++ compiler rounds literals to double precision.
    ///
    /// The header implements direct, mu, T3, and T2 callbacks, including explicit
    /// mu-squared terms and ranks through `propagator_count + 1`. It can be used
    /// with `validation/reference/build.py --numerator-header <header>`.
    pub fn export_ninja_cpp(&self, propagator_count: usize) -> Result<String> {
        if propagator_count == 0 {
            return Err(Error::NoPropagators);
        }
        if propagator_count > u8::MAX as usize {
            return Err(Error::Unsupported(
                "original Ninja uses byte-sized cut partitions".into(),
            ));
        }
        let rank = self.rank();
        if rank > propagator_count + 1 {
            return Err(Error::Rank {
                rank,
                propagators: propagator_count,
            });
        }
        let signed_rank = i32::try_from(rank)
            .map_err(|_| Error::Unsupported("rank exceeds signed Laurent exponent range".into()))?;
        let direct = self.exact_evaluator()?;
        let mu_count: usize = if rank > propagator_count { 2 } else { 1 };
        let mu = self.exact_expansion(&ExpansionKind::Mu {
            powers: (0..mu_count)
                .map(|depth| signed_rank - depth as i32)
                .collect(),
        })?;
        // Original Ninja's mindeg is the subleading depth. Export exactly the
        // requested coefficients for each depth: calculating the maximum and
        // then truncating needlessly inflates reference callback timings.
        let t3 = (0..=4)
            .map(|limit| {
                self.exact_expansion(&ExpansionKind::T3 {
                    powers: (0..=limit)
                        .flat_map(|depth| {
                            (0..=depth / 2).map(move |mu| (signed_rank - depth, mu as u16))
                        })
                        .collect(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let t2 = (0..=3)
            .map(|limit| {
                self.exact_expansion(&ExpansionKind::T2 {
                    powers: (0..=limit)
                        .flat_map(|depth| {
                            (0..=depth).flat_map(move |x| {
                                (0..=(depth - x) / 2)
                                    .map(move |mu| (signed_rank - depth, x as u16, mu as u16))
                            })
                        })
                        .collect(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let evaluators = [("direct".to_owned(), &direct), ("mu".to_owned(), &mu)]
            .into_iter()
            .chain(
                t3.iter()
                    .enumerate()
                    .map(|(depth, e)| (format!("t3_{depth}"), e)),
            )
            .chain(
                t2.iter()
                    .enumerate()
                    .map(|(depth, e)| (format!("t2_{depth}"), e)),
            )
            .collect::<Vec<_>>();

        let mut source = String::from(
            "// Generated from a NinjaRS Symbolica numerator; validation only.\n#pragma once\n#include <algorithm>\n#include <cmath>\n#include <complex>\n#include <cstddef>\n#include <iostream>\n#include <stdexcept>\n#include <vector>\n#include <ninja/num_defs.hh>\n\nnamespace ninjars_reference {\n",
        );
        writeln!(source, "constexpr unsigned parameter_count = {};\nconstexpr unsigned rank = {rank};\nconstexpr unsigned propagator_count = {propagator_count};", self.parameter_names().len()).unwrap();
        for (name, evaluator) in &evaluators {
            writeln!(source, "namespace generated_{name} {{").unwrap();
            // Exact rational evaluators reserve zero placeholder slots for
            // registered constants, including π. Mapping initializes those
            // slots through Symbolica's registered implementations. Exporting
            // the unmapped evaluator silently turns those constants into zero.
            let precision = 200;
            let prototype = Float::new(precision);
            let numerical = (*evaluator).clone().map_coeff_with_prec(
                &|value| {
                    Complex::new(
                        prototype.from_rational(&value.re),
                        prototype.from_rational(&value.im),
                    )
                },
                precision,
            );
            source.push_str(
                &numerical
                    .export_cpp_str::<Complex<f64>>(
                        &format!("ninjars_{name}"),
                        ExportSettings::new()
                            .include_header(false)
                            .inline_asm(InlineASM::None),
                    )
                    .map_err(Error::Symbolic)?,
            );
            source.push_str("}\n");
        }
        source.push_str("\nclass GeneratedNumerator final : public ninja::Numerator {\npublic:\n  explicit GeneratedNumerator(const ninja::Complex* parameters)\n    : ");
        for (index, (name, evaluator)) in [
            ("direct", &direct),
            ("mu", &mu),
            ("t3", &t3[4]),
            ("t2", &t2[3]),
        ]
        .into_iter()
        .enumerate()
        {
            if index != 0 {
                source.push_str(",\n      ");
            }
            let buffers = evaluators
                .iter()
                .filter(|(candidate, _)| {
                    candidate == name || candidate.starts_with(&format!("{name}_"))
                })
                .map(|(candidate, _)| {
                    let function =
                        Complex::<f64>::construct_function_name(&format!("ninjars_{candidate}"));
                    format!("generated_{candidate}::{function}_get_buffer_len()")
                })
                .collect::<Vec<_>>()
                .join(",");
            write!(
                source,
                "{name}_inputs_({}), {name}_scratch_(std::max({{{buffers}}})), {name}_outputs_({})",
                evaluator.get_input_len(),
                evaluator.get_output_len()
            )
            .unwrap();
        }
        source.push_str(" { set_parameters(parameters); }\n\n  // Caller retains the parameter row until the next rebind or destruction.\n  void set_parameters(const ninja::Complex* parameters) {\n    if (parameter_count != 0 && parameters == nullptr) throw std::invalid_argument(\"missing numerator parameters\");\n    parameters_ = parameters;\n  }\n\n  ninja::Complex evaluate(const ninja::ComplexMomentum& q, const ninja::Complex& muq, int, const ninja::PartitionInt[]) override {\n    put(direct_inputs_, 0, q);\n    direct_inputs_[4] = muq;\n    bind(direct_inputs_, 5);\n");
        emit_call(&mut source, "direct", "direct");
        source.push_str("    return direct_outputs_[0];\n  }\n\n  void muExpansion(const ninja::ComplexMomentum v[], const ninja::PartitionInt[], ninja::Complex c[]) override {\n    put(mu_inputs_, 0, v[0]);\n");
        if mu_count == 2 {
            source.push_str("    put(mu_inputs_, 4, v[1]);\n");
        }
        source.push_str("    bind(mu_inputs_, 8);\n");
        emit_call(&mut source, "mu", "mu");
        writeln!(
            source,
            "    std::copy_n(mu_outputs_.data(), {mu_count}, c);\n  }}"
        )
        .unwrap();
        source.push_str("\n  void t3Expansion(const ninja::ComplexMomentum& a, const ninja::ComplexMomentum& e3, const ninja::ComplexMomentum& e4, const ninja::Complex& beta, int mindeg, int, const ninja::PartitionInt[], ninja::Complex c[]) override {\n    if (mindeg < 0 || mindeg > 4) throw std::out_of_range(\"T3 Laurent depth\");\n    put(t3_inputs_, 0, a);\n    put(t3_inputs_, 4, e3);\n    put(t3_inputs_, 8, e4);\n    t3_inputs_[12] = beta;\n");
        source.push_str("    bind(t3_inputs_, 13);\n");
        emit_depth_dispatch(&mut source, "t3", 4);
        source.push_str("    const unsigned counts[] = {1, 2, 4, 6, 9};\n    std::copy_n(t3_outputs_.data(), counts[mindeg], c);\n  }\n\n  void t2Expansion(const ninja::ComplexMomentum& a0, const ninja::ComplexMomentum& a1, const ninja::ComplexMomentum& e3, const ninja::ComplexMomentum& e4, const ninja::Complex beta[], int mindeg, int, const ninja::PartitionInt[], ninja::Complex c[]) override {\n    if (mindeg < 0 || mindeg > 3) throw std::out_of_range(\"T2 Laurent depth\");\n    put(t2_inputs_, 0, a0);\n    put(t2_inputs_, 4, a1);\n    put(t2_inputs_, 8, e3);\n    put(t2_inputs_, 12, e4);\n    std::copy_n(beta, 3, t2_inputs_.begin()+16);\n");
        source.push_str("    bind(t2_inputs_, 19);\n");
        emit_depth_dispatch(&mut source, "t2", 3);
        source.push_str("    const unsigned counts[] = {1, 3, 7, 13};\n    std::copy_n(t2_outputs_.data(), counts[mindeg], c);\n  }\n\nprivate:\n  static void put(std::vector<ninja::Complex>& input, unsigned offset, const ninja::ComplexMomentum& vector) {\n    for (unsigned component = 0; component < 4; ++component) input[offset+component] = vector[component];\n  }\n");
        source.push_str("  const ninja::Complex* parameters_;\n  void bind(std::vector<ninja::Complex>& input, unsigned offset) const {\n    for (unsigned i = 0; i < parameter_count; ++i) input[offset+i] = parameters_[i];\n  }\n");
        for name in ["direct", "mu", "t3", "t2"] {
            writeln!(
                source,
                "  std::vector<ninja::Complex> {name}_inputs_, {name}_scratch_, {name}_outputs_;"
            )
            .unwrap();
        }
        source.push_str("};\n} // namespace ninjars_reference\n");
        Ok(source)
    }
}

fn emit_call(source: &mut String, name: &str, storage: &str) {
    let function = Complex::<f64>::construct_function_name(&format!("ninjars_{name}"));
    writeln!(source, "    generated_{name}::{function}({storage}_inputs_.data(), {storage}_scratch_.data(), {storage}_outputs_.data());").unwrap();
}

fn emit_depth_dispatch(source: &mut String, storage: &str, maximum: usize) {
    source.push_str("    switch (mindeg) {\n");
    for depth in 0..=maximum {
        writeln!(source, "    case {depth}:").unwrap();
        emit_call(source, &format!("{storage}_{depth}"), storage);
        source.push_str("      break;\n");
    }
    source.push_str("    }\n");
}
