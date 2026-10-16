//! Ignored, source-only inventory for the reduction algorithm audit.
//! No production kernel selection or numerical evaluator is installed here.

use crate::{
    LaurentDivision, ResidueProjection, TadpoleSubtraction, TriangleBubbleSubtraction,
    kernel::ExactKernel,
    rust_codegen::{EmitOptions, RustEmitter},
};
use ninjars::ExpansionKind;
use serde_json::{Value, json};
use std::{error::Error, ops::Range};
use symbolica::{
    atom::{Atom, AtomCore},
    id::{Pattern, Replacement},
};

fn rank_zero_slots(n: usize, rank: usize, source_size: usize) -> Vec<usize> {
    let division = LaurentDivision::new(n, rank, source_size).unwrap();
    let projection = ResidueProjection::new(&division).unwrap();
    projection
        .exact
        .expressions
        .iter()
        .enumerate()
        .filter_map(|(index, value)| (value == &Atom::Zero).then_some(index))
        .collect()
}

fn select_and_specialize(
    plan: &ExactKernel,
    heads: &[usize],
    residue: Range<usize>,
    zero_slots: &[usize],
) -> ExactKernel {
    let replacements: Vec<_> = zero_slots
        .iter()
        .map(|&slot| {
            assert!(slot < residue.len());
            Replacement::new(
                Pattern::Literal(plan.parameters[residue.start + slot].clone()),
                Pattern::Literal(Atom::Zero),
            )
        })
        .collect();
    ExactKernel {
        parameters: plan.parameters.clone(),
        expressions: heads
            .iter()
            .map(|&index| plan.expressions[index].replace_multiple(&replacements))
            .collect(),
    }
}

fn live_columns(plan: &ExactKernel) -> Vec<usize> {
    plan.parameters
        .iter()
        .enumerate()
        .filter_map(|(index, parameter)| {
            plan.expressions
                .iter()
                .any(|expression| expression.contains(parameter))
                .then_some(index)
        })
        .collect()
}

fn record(name: &str, plan: &ExactKernel) -> Result<Value, Box<dyn Error>> {
    let evaluator = plan.evaluator()?;
    let operations = evaluator.count_operations();
    let instructions = evaluator.export_instructions();
    // The operation count below is root-only; fail rather than undercount
    // hidden bodies if these straight-line plans ever acquire helper calls.
    assert!(instructions.sub_evaluators.is_empty());
    assert!(instructions.constant_functions.is_empty());
    assert_eq!(operations.function_calls, 0);
    let mut emitter = RustEmitter::default();
    let generated = emitter.emit_batch(evaluator, EmitOptions::default())?;
    let constants = emitter.constants_source();
    Ok(json!({
        "name": name,
        "inputs": plan.parameters.len(),
        "outputs": plan.expressions.len(),
        "live_input_columns": live_columns(plan),
        "operations": {
            "additions": operations.additions,
            "multiplications": operations.multiplications,
            "inversions": operations.inversions,
            "function_calls": operations.function_calls
        },
        "instructions": generated.instruction_count,
        "temporary_slots": instructions.temporary_count,
        "source_bytes": generated.source.len(),
        "constants_bytes": constants.len(),
        "constant_count": emitter.constants().len(),
        "helper_count": generated.helper_count,
        "outlined_branches": generated.outlined_branch_count
    }))
}

#[test]
#[ignore = "explicit symbolic/code-generation inventory; no performance or runtime acceptance"]
fn inventory_rank_specialization_n5_r4() -> Result<(), Box<dyn Error>> {
    let triangle_zeros = rank_zero_slots(5, 4, 3);
    let bubble_zeros = rank_zero_slots(5, 4, 2);
    assert_eq!(triangle_zeros, [3, 6, 8, 9]);
    assert_eq!(bubble_zeros, [2, 4, 6, 7, 8, 9]);

    let division = LaurentDivision::new(5, 4, 2)?;
    let correction = TriangleBubbleSubtraction::for_quotient(&division)?;
    let ExpansionKind::T2 { powers } = correction.output_expansion() else {
        unreachable!()
    };
    let full: Vec<_> = (0..powers.len()).collect();
    let swapped: Vec<_> = powers
        .iter()
        .enumerate()
        .filter_map(|(index, &(t, _, _))| (t > 0).then_some(index))
        .collect();
    assert_eq!(full.len(), 3);
    assert_eq!(swapped.len(), 1);
    let residue = correction.input_layout().residue.clone();
    let specialized =
        select_and_specialize(&correction.exact, &full, residue.clone(), &triangle_zeros);
    let swapped_generic = select_and_specialize(&correction.exact, &swapped, residue.clone(), &[]);
    let swapped_specialized =
        select_and_specialize(&correction.exact, &swapped, residue, &triangle_zeros);
    let den = correction.input_layout().denominator.clone();
    let live = live_columns(&specialized);
    assert!(live.contains(&den.start));
    assert!(den.clone().skip(3).all(|column| !live.contains(&column)));
    let swapped_live = live_columns(&swapped_specialized);
    assert!(swapped_live.contains(&den.start));
    assert!(den.skip(1).all(|column| !swapped_live.contains(&column)));

    let mut records = vec![
        record("triangle_bubble_current_degree1", &correction.exact)?,
        record("triangle_bubble_rank_specialized_degree1", &specialized)?,
        record(
            "triangle_bubble_generic_swapped_positive_t",
            &swapped_generic,
        )?,
        record(
            "triangle_bubble_rank_specialized_swapped_positive_t",
            &swapped_specialized,
        )?,
    ];
    for (source_size, zeros, label) in [
        (3, &triangle_zeros, "triangle_tadpole"),
        (2, &bubble_zeros, "bubble_tadpole"),
    ] {
        let correction = TadpoleSubtraction::new(source_size, false)?;
        let ExpansionKind::T3 { powers } = correction.output_expansion() else {
            unreachable!()
        };
        let constant = [powers.iter().position(|&power| power == (0, 0)).unwrap()];
        let residue = correction.input_layout().residue.clone();
        let selected = select_and_specialize(&correction.exact, &constant, residue.clone(), &[]);
        let specialized = select_and_specialize(&correction.exact, &constant, residue, zeros);
        let live = live_columns(&specialized);
        for den in &correction.input_layout().denominators {
            assert!(live.contains(&den.start));
            assert!(den.clone().skip(1).all(|column| !live.contains(&column)));
        }
        records.push(record(
            &format!("{label}_current_degree1"),
            &correction.exact,
        )?);
        records.push(record(&format!("{label}_selected_constant"), &selected)?);
        records.push(record(
            &format!("{label}_rank_specialized_constant"),
            &specialized,
        )?);
    }
    let report = json!({
        "schema": "ninjars-rank-specialization-inventory-v1",
        "propagators": 5,
        "rank": 4,
        "production_dispatch": false,
        "numerical_execution": false,
        "performance_acceptance": false,
        "triangle_structural_zero_slots": triangle_zeros,
        "bubble_structural_zero_slots": bubble_zeros,
        "plans": records,
        "scope": "Existing Symbolica subtraction expressions with output selection and structural-zero substitution; emitted Rust source is counted but not compiled or executed.",
        "provenance_contract": "Specialization requires residues produced by same-topology n5/r4 projection, or explicit all-component validation/fallback for externally supplied residues. Standalone generic plans are unchanged.",
        "limitations": [
            "No elapsed-time claim; instruction counts do not predict end-to-end speed.",
            "Input ABI remains full width; only expression liveness changes.",
            "No assertion that algebraic simplification preserves every floating-point overflow/failure boundary.",
            "Higher/mixed Dual execution and numerical controls are future acceptance work."
        ]
    });
    let output = std::env::var_os("NINJARS_RANK_INVENTORY_OUTPUT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../target/rank-specialization-inventory.json")
        });
    if let Some(parent) = output.parent().filter(|path| !path.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&output, serde_json::to_string_pretty(&report)? + "\n")?;
    println!("{}", output.display());
    Ok(())
}
