//! Original-Ninja callback export for independent reference reduction.
//!
//! Symbolica performs substitution, Laurent coefficient extraction, optimization,
//! and arithmetic code generation. The C++ wrapper only adapts callback layouts.

use std::fmt::Write;

use ninjars::{Error, ExpansionKind, Numerator, Result};
use symbolica::{
    domains::float::{Complex, Real, SingleFloat},
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
    /// No parameter values or numerical coefficient approximations from this
    /// instance are embedded. Reference execution uses double precision.
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
        // Original Ninja's mindeg is the subleading depth. These fixed maxima
        // cover every R <= n+1 callback, including tadpole subtractions.
        let t3 = self.exact_expansion(&ExpansionKind::T3 {
            powers: (0..=4)
                .flat_map(|depth| (0..=depth / 2).map(move |mu| (signed_rank - depth, mu as u16)))
                .collect(),
        })?;
        let t2 = self.exact_expansion(&ExpansionKind::T2 {
            powers: (0..=3)
                .flat_map(|depth| {
                    (0..=depth).flat_map(move |x| {
                        (0..=(depth - x) / 2)
                            .map(move |mu| (signed_rank - depth, x as u16, mu as u16))
                    })
                })
                .collect(),
        })?;

        let mut source = String::from(
            "// Generated from a NinjaRS Symbolica numerator; validation only.\n#pragma once\n#include <algorithm>\n#include <cmath>\n#include <complex>\n#include <cstddef>\n#include <iostream>\n#include <stdexcept>\n#include <vector>\n#include <ninja/num_defs.hh>\n\nnamespace ninjars_reference {\n",
        );
        writeln!(source, "constexpr unsigned parameter_count = {};\nconstexpr unsigned rank = {rank};\nconstexpr unsigned propagator_count = {propagator_count};", self.parameter_names().len()).unwrap();
        for (name, evaluator) in [("direct", &direct), ("mu", &mu), ("t3", &t3), ("t2", &t2)] {
            writeln!(source, "namespace generated_{name} {{").unwrap();
            source.push_str(
                &evaluator
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
        for (index, (name, evaluator)) in
            [("direct", &direct), ("mu", &mu), ("t3", &t3), ("t2", &t2)]
                .into_iter()
                .enumerate()
        {
            if index != 0 {
                source.push_str(",\n      ");
            }
            let function = Complex::<f64>::construct_function_name(&format!("ninjars_{name}"));
            write!(source, "{name}_inputs_({}), {name}_scratch_(generated_{name}::{function}_get_buffer_len()), {name}_outputs_({})", evaluator.get_input_len(), evaluator.get_output_len()).unwrap();
        }
        source.push_str(" {\n    if (parameter_count != 0 && parameters == nullptr) throw std::invalid_argument(\"missing numerator parameters\");\n");
        for (name, offset) in [("direct", 5), ("mu", 8), ("t3", 13), ("t2", 19)] {
            writeln!(source, "    for (unsigned i = 0; i < parameter_count; ++i) {name}_inputs_[{offset}+i] = parameters[i];").unwrap();
        }
        source.push_str("  }\n\n  ninja::Complex evaluate(const ninja::ComplexMomentum& q, const ninja::Complex& muq, int, const ninja::PartitionInt[]) override {\n    put(direct_inputs_, 0, q);\n    direct_inputs_[4] = muq;\n");
        emit_call(&mut source, "direct");
        source.push_str("    return direct_outputs_[0];\n  }\n\n  void muExpansion(const ninja::ComplexMomentum v[], const ninja::PartitionInt[], ninja::Complex c[]) override {\n    put(mu_inputs_, 0, v[0]);\n");
        if mu_count == 2 {
            source.push_str("    put(mu_inputs_, 4, v[1]);\n");
        }
        emit_call(&mut source, "mu");
        writeln!(
            source,
            "    std::copy_n(mu_outputs_.data(), {mu_count}, c);\n  }}"
        )
        .unwrap();
        source.push_str("\n  void t3Expansion(const ninja::ComplexMomentum& a, const ninja::ComplexMomentum& e3, const ninja::ComplexMomentum& e4, const ninja::Complex& beta, int mindeg, int, const ninja::PartitionInt[], ninja::Complex c[]) override {\n    if (mindeg < 0 || mindeg > 4) throw std::out_of_range(\"T3 Laurent depth\");\n    put(t3_inputs_, 0, a);\n    put(t3_inputs_, 4, e3);\n    put(t3_inputs_, 8, e4);\n    t3_inputs_[12] = beta;\n");
        emit_call(&mut source, "t3");
        source.push_str("    const unsigned counts[] = {1, 2, 4, 6, 9};\n    std::copy_n(t3_outputs_.data(), counts[mindeg], c);\n  }\n\n  void t2Expansion(const ninja::ComplexMomentum& a0, const ninja::ComplexMomentum& a1, const ninja::ComplexMomentum& e3, const ninja::ComplexMomentum& e4, const ninja::Complex beta[], int mindeg, int, const ninja::PartitionInt[], ninja::Complex c[]) override {\n    if (mindeg < 0 || mindeg > 3) throw std::out_of_range(\"T2 Laurent depth\");\n    put(t2_inputs_, 0, a0);\n    put(t2_inputs_, 4, a1);\n    put(t2_inputs_, 8, e3);\n    put(t2_inputs_, 12, e4);\n    std::copy_n(beta, 3, t2_inputs_.begin()+16);\n");
        emit_call(&mut source, "t2");
        source.push_str("    const unsigned counts[] = {1, 3, 7, 13};\n    std::copy_n(t2_outputs_.data(), counts[mindeg], c);\n  }\n\nprivate:\n  static void put(std::vector<ninja::Complex>& input, unsigned offset, const ninja::ComplexMomentum& vector) {\n    for (unsigned component = 0; component < 4; ++component) input[offset+component] = vector[component];\n  }\n");
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

fn emit_call(source: &mut String, name: &str) {
    let function = Complex::<f64>::construct_function_name(&format!("ninjars_{name}"));
    writeln!(source, "    generated_{name}::{function}({name}_inputs_.data(), {name}_scratch_.data(), {name}_outputs_.data());").unwrap();
}
