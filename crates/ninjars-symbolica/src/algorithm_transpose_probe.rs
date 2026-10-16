//! Bounded, ignored symbolic experiment. No production dispatch is changed.
//!
//! This is orchestration of existing exact kernels, not an algebra engine:
//! Symbolica performs substitution, linear coefficient extraction, identities,
//! expression optimization and instruction generation. Geometry is abstract;
//! independently named projections are never assumed to obey cut identities.

use std::{error::Error, path::PathBuf};

use ninjars::ExpansionKind;
use serde_json::{Value, json};
use symbolica::{
    atom::{Atom, AtomCore},
    domains::atom::AtomField,
    id::{Pattern, Replacement},
    symbol,
};

use crate::{
    LaurentDivision, MasterContraction, ResidueProjection, TadpoleSubtraction,
    TriangleBubbleSubtraction,
    kernel::ExactKernel,
    rust_codegen::{EmitOptions, RustEmitter},
};

type ProbeResult<T> = Result<T, Box<dyn Error>>;
const OUTPUTS: usize = 5;
// Stop this experiment rather than silently turn it into a large symbolic job.
const MAX_COMPOSED_BYTES: usize = 8 * 1024 * 1024;

fn substitute(expressions: &[Atom], from: &[Atom], to: &[Atom]) -> Vec<Atom> {
    assert_eq!(from.len(), to.len());
    let replacements: Vec<_> = from
        .iter()
        .zip(to)
        .map(|(a, b)| Replacement::new(Pattern::Literal(a.clone()), Pattern::Literal(b.clone())))
        .collect();
    expressions
        .iter()
        .map(|e| e.replace_multiple(&replacements))
        .collect()
}

fn bind(plan: &ExactKernel, arguments: &[Atom]) -> Vec<Atom> {
    substitute(&plan.expressions, &plan.parameters, arguments)
}

struct Block {
    name: String,
    category: &'static str,
    exact: ExactKernel,
    data: Vec<Atom>,
    outputs: Vec<Atom>,
}

#[derive(Default)]
struct Network {
    external: Vec<Atom>,
    callbacks: Vec<Atom>,
    variables: Vec<Atom>,
    data: Vec<Atom>,
    blocks: Vec<Block>,
    master_values: Vec<Atom>,
}

impl Network {
    fn external(&mut self, label: &str, count: usize, callback: bool) -> Vec<Atom> {
        let values: Vec<_> = (0..count)
            .map(|i| Atom::var(symbol!(format!("ninjars_transpose::{label}_{i}"))))
            .collect();
        for value in &values {
            assert!(
                !self.variables.contains(value),
                "duplicate graph input {value}"
            );
        }
        self.external.extend(values.clone());
        self.variables.extend(values.clone());
        if callback {
            self.callbacks.extend(values.clone());
            self.data.extend(values.clone());
        }
        values
    }

    fn add(&mut self, label: &str, category: &'static str, expressions: Vec<Atom>) -> Vec<Atom> {
        let parameters: Vec<_> = self
            .variables
            .iter()
            .filter(|p| expressions.iter().any(|e| e.contains(*p)))
            .cloned()
            .collect();
        let data = parameters
            .iter()
            .filter(|p| self.data.contains(p))
            .cloned()
            .collect();
        let outputs: Vec<_> = (0..expressions.len())
            .map(|i| Atom::var(symbol!(format!("ninjars_transpose::{label}_out_{i}"))))
            .collect();
        assert!(outputs.iter().all(|p| !self.variables.contains(p)));
        self.variables.extend(outputs.clone());
        self.data.extend(outputs.clone());
        self.blocks.push(Block {
            name: label.into(),
            category,
            exact: ExactKernel {
                expressions,
                parameters,
            },
            data,
            outputs: outputs.clone(),
        });
        outputs
    }

    fn plan(
        &mut self,
        label: &str,
        category: &'static str,
        exact: &ExactKernel,
        args: &[Atom],
    ) -> Vec<Atom> {
        self.add(label, category, bind(exact, args))
    }

    fn compose(&self, outputs: &[Atom]) -> ProbeResult<ExactKernel> {
        let mut from = Vec::new();
        let mut to = Vec::new();
        let mut bytes = 0;
        for block in &self.blocks {
            let expressions = substitute(&block.exact.expressions, &from, &to);
            bytes += expressions
                .iter()
                .map(|e| e.to_string().len())
                .sum::<usize>();
            if bytes > MAX_COMPOSED_BYTES {
                return Err(format!(
                    "composition budget exceeded after {}: {bytes} bytes",
                    block.name
                )
                .into());
            }
            from.extend(block.outputs.clone());
            to.extend(expressions);
        }
        Ok(ExactKernel {
            parameters: self.external.clone(),
            expressions: substitute(outputs, &from, &to),
        })
    }
}

fn division(
    network: &mut Network,
    label: &str,
    plan: &LaurentDivision,
    denominators: &[Vec<Atom>],
) -> Vec<Atom> {
    let mut arguments = network.external(
        &format!("{label}_callback"),
        plan.numerator_expansion().len(),
        true,
    );
    arguments.extend(denominators.iter().flatten().cloned());
    network.plan(label, "division", &plan.exact, &arguments)
}

fn subtract(network: &mut Network, label: &str, raw: &[Atom], parents: &[Vec<Atom>]) -> Vec<Atom> {
    assert!(parents.iter().all(|p| p.len() == raw.len()));
    let expressions = raw
        .iter()
        .enumerate()
        .map(|(i, x)| parents.iter().fold(x.clone(), |sum, p| sum - &p[i]))
        .collect();
    network.add(label, "subtraction_sum", expressions)
}

fn integrate(
    network: &mut Network,
    label: &str,
    size: usize,
    coefficients: &[Atom],
    masses: &[Atom],
    invariants: &[Atom],
) -> ProbeResult<Vec<Atom>> {
    let plan = MasterContraction::new(size, false)?;
    assert_eq!(coefficients.len(), plan.input_layout().coefficients.len());
    let mut arguments = coefficients.to_vec();
    arguments.extend_from_slice(invariants);
    arguments.extend_from_slice(masses);
    if size < 3 {
        arguments.extend(network.external(&format!("{label}_integration_basis"), 1, false));
    }
    let weights = network.plan(
        &format!("{label}_weights"),
        "master_weights",
        &plan.weights,
        &arguments,
    );
    let masters = network.external(&format!("{label}_masters"), plan.masters().len() * 3, false);
    network.master_values.extend(masters.clone());
    let mut arguments = weights;
    arguments.extend(masters);
    Ok(network.plan(
        &format!("{label}_contract"),
        "master_contraction",
        &plan.contraction,
        &arguments,
    ))
}

/// All n3/r3 cuts. Denominator rows are shared only for the identical target,
/// orientation and extra line. Projection inputs retain the source identity.
fn forward_network() -> ProbeResult<(Network, Vec<Atom>)> {
    let mut network = Network::default();
    let masses = network.external("mass_squared", 3, false);
    let invariants = network.external("s01_s02_s12", 3, false);
    let triangle_division = LaurentDivision::new(3, 3, 3)?;
    let bubble_division = LaurentDivision::new(3, 3, 2)?;
    let tadpole_division = LaurentDivision::new(3, 3, 1)?;
    let triangle_projection = ResidueProjection::new(&triangle_division)?;
    let bubble_projection = ResidueProjection::new(&bubble_division)?;
    let triangle_bubble = TriangleBubbleSubtraction::for_quotient(&bubble_division)?;
    let triangle_tadpole = TadpoleSubtraction::new(3, false)?;
    let bubble_tadpole = TadpoleSubtraction::new(2, false)?;
    assert_eq!(
        triangle_bubble.output_expansion(),
        bubble_division.quotient_expansion()
    );
    assert_eq!(
        triangle_tadpole.output_expansion(),
        tadpole_division.quotient_expansion()
    );
    assert_eq!(
        bubble_tadpole.output_expansion(),
        tadpole_division.quotient_expansion()
    );

    let mut q3 = Vec::new();
    for orientation in 0..2 {
        q3.extend(division(
            &mut network,
            &format!("tri012_o{orientation}_division"),
            &triangle_division,
            &[],
        ));
    }
    q3.extend(network.external("tri012_g", 1, false));
    let c3 = network.plan(
        "tri012_projection",
        "projection",
        &triangle_projection.exact,
        &q3,
    );
    let mut contributions = vec![integrate(
        &mut network,
        "tri012",
        3,
        &c3,
        &masses,
        &[
            invariants[0].clone(),
            invariants[2].clone(),
            invariants[1].clone(),
        ],
    )?];
    let partitions = [[0, 1], [0, 2], [1, 2]];
    let mut bubbles = Vec::new();
    for (index, partition) in partitions.iter().enumerate() {
        let label = format!("bub{}{}", partition[0], partition[1]);
        let extra = (0..3).find(|i| !partition.contains(i)).unwrap();
        let beta = network.external(&format!("{label}_beta"), 3, false);
        let mut q2 = Vec::new();
        for orientation in 0..2 {
            let target = format!("{label}_o{orientation}");
            let denominator = network.external(
                &format!("{target}_line{extra}_denominator"),
                bubble_division.denominator_expansion().len(),
                false,
            );
            let raw = division(
                &mut network,
                &format!("{target}_division"),
                &bubble_division,
                std::slice::from_ref(&denominator),
            );
            let mut arguments = c3.clone();
            arguments.extend(network.external(
                &format!("tri012_to_{target}_source_projections"),
                8,
                false,
            ));
            arguments.extend(beta.clone());
            arguments.extend(denominator);
            let correction = network.plan(
                &format!("tri012_to_{target}"),
                "parent_correction",
                &triangle_bubble.exact,
                &arguments,
            );
            q2.extend(subtract(
                &mut network,
                &format!("{target}_sum"),
                &raw,
                &[correction],
            ));
        }
        q2.extend(network.external(&format!("{label}_g"), 1, false));
        let c2 = network.plan(
            &format!("{label}_projection"),
            "projection",
            &bubble_projection.exact,
            &q2,
        );
        contributions.push(integrate(
            &mut network,
            &label,
            2,
            &c2,
            &[masses[partition[0]].clone(), masses[partition[1]].clone()],
            &[invariants[index].clone()],
        )?);
        bubbles.push(c2);
    }
    let ExpansionKind::T3 { powers } = tadpole_division.quotient_expansion() else {
        unreachable!()
    };
    let constant = powers.iter().position(|&p| p == (0, 0)).unwrap();
    for (target, mass) in masses.iter().enumerate() {
        let label = format!("tad{target}_o0");
        let extras: Vec<_> = (0..3).filter(|&i| i != target).collect();
        let denominators: Vec<_> = extras
            .iter()
            .map(|line| {
                network.external(
                    &format!("{label}_line{line}_denominator"),
                    tadpole_division.denominator_expansion().len(),
                    false,
                )
            })
            .collect();
        let raw = division(
            &mut network,
            &format!("{label}_division"),
            &tadpole_division,
            &denominators,
        );
        let beta = network.external(&format!("{label}_beta"), 1, false);
        let mut arguments = c3.clone();
        arguments.extend(network.external(
            &format!("tri012_to_{label}_source_projections"),
            6,
            false,
        ));
        arguments.extend(beta.clone());
        arguments.extend(denominators.iter().flatten().cloned());
        let mut corrections = vec![network.plan(
            &format!("tri012_to_{label}"),
            "parent_correction",
            &triangle_tadpole.exact,
            &arguments,
        )];
        for (partition, c2) in partitions.iter().zip(&bubbles) {
            if !partition.contains(&target) {
                continue;
            }
            let source = format!("bub{}{}_to_{label}", partition[0], partition[1]);
            let extra = *partition.iter().find(|&&i| i != target).unwrap();
            let mut arguments = c2.clone();
            arguments.extend(network.external(&format!("{source}_source_projections"), 9, false));
            arguments.extend(beta.clone());
            arguments
                .extend(denominators[extras.iter().position(|&i| i == extra).unwrap()].clone());
            corrections.push(network.plan(
                &source,
                "parent_correction",
                &bubble_tadpole.exact,
                &arguments,
            ));
        }
        let q1 = subtract(&mut network, &format!("{label}_sum"), &raw, &corrections);
        // Existing ordinary Engine::new projection selects c[0] and zeros the
        // four unused tadpole slots. No higher-rank projection is duplicated.
        let mut c1 = vec![Atom::Zero; 5];
        c1[0] = q1[constant].clone();
        let c1 = network.add(&format!("{label}_projection"), "projection", c1);
        contributions.push(integrate(
            &mut network,
            &label,
            1,
            &c1,
            std::slice::from_ref(mass),
            &[],
        )?);
    }
    let outputs = network.add(
        "amplitude_sum",
        "contribution_sum",
        (0..OUTPUTS)
            .map(|i| contributions.iter().fold(Atom::Zero, |sum, c| sum + &c[i]))
            .collect(),
    );
    assert_eq!(
        network.callbacks.len(),
        2 * triangle_division.numerator_expansion().len()
            + 6 * bubble_division.numerator_expansion().len()
            + 3 * tadpole_division.numerator_expansion().len()
    );
    Ok((network, outputs))
}

/// Symbolica extracts the exact linear map. Reject hidden dependence, affine
/// terms or nonlinear data terms rather than assuming the graph is linear.
fn linear_coefficients(block: &Block) -> ProbeResult<Vec<Vec<Atom>>> {
    let field = AtomField {
        statistical_zero_test: false,
        ..AtomField::new()
    };
    block
        .exact
        .expressions
        .iter()
        .map(|expression| {
            let polynomial =
                expression.to_polynomial_in_vars_with_field::<i32>(&block.data, &field);
            for term in &polynomial {
                if term.exponents.iter().any(|&p| p < 0)
                    || term.exponents.iter().sum::<i32>() != 1
                    || block.data.iter().any(|x| term.coefficient.contains(x))
                {
                    return Err(
                        format!("{} is not homogeneous linear in its data", block.name).into(),
                    );
                }
            }
            let coefficients: Vec<_> = (0..block.data.len())
                .map(|i| {
                    let mut powers = vec![0; block.data.len()];
                    powers[i] = 1;
                    polynomial.coefficient(&powers).unwrap_or(Atom::Zero)
                })
                .collect();
            let reconstruction = coefficients
                .iter()
                .zip(&block.data)
                .fold(Atom::Zero, |sum, (c, x)| sum + c * x);
            assert_eq!(
                (expression - reconstruction).expand(),
                Atom::Zero,
                "{} linear reconstruction",
                block.name
            );
            Ok(coefficients)
        })
        .collect()
}

fn adjoint(adjoint_rows: &[(Atom, Vec<Atom>)], variable: &Atom) -> Vec<Atom> {
    adjoint_rows.iter().filter(|(v, _)| v == variable).fold(
        vec![Atom::Zero; OUTPUTS],
        |mut sum, (_, row)| {
            for (a, b) in sum.iter_mut().zip(row) {
                *a += b;
            }
            sum
        },
    )
}

/// Each local pullback remains a separate exact block. This deliberately avoids
/// forming a dense A12*A23 matrix. All five output components travel together.
fn transposed_network(
    forward: &Network,
    outputs: &[Atom],
) -> ProbeResult<(Network, Vec<Atom>, usize)> {
    let mut reverse = Network::default();
    reverse.external = forward
        .external
        .iter()
        .filter(|x| !forward.callbacks.contains(x))
        .cloned()
        .collect();
    reverse.variables = reverse.external.clone();
    let mut adjoints: Vec<_> = outputs
        .iter()
        .enumerate()
        .map(|(i, x)| {
            (
                x.clone(),
                (0..OUTPUTS)
                    .map(|j| Atom::num(i32::from(i == j)))
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    for block in forward.blocks.iter().rev() {
        eprintln!("TRANSPOSE_STAGE pullback {}", block.name);
        let coefficients = linear_coefficients(block)?;
        let seeds: Vec<_> = block
            .outputs
            .iter()
            .map(|x| adjoint(&adjoints, x))
            .collect();
        let mut slots = Vec::new();
        let mut expressions = Vec::new();
        for (input, variable) in block.data.iter().enumerate() {
            for component in 0..OUTPUTS {
                let value = coefficients
                    .iter()
                    .zip(&seeds)
                    .fold(Atom::Zero, |sum, (row, seed)| {
                        sum + &row[input] * &seed[component]
                    });
                if value != Atom::Zero {
                    slots.push((variable.clone(), component));
                    expressions.push(value);
                }
            }
        }
        if expressions.is_empty() {
            continue;
        }
        let values = reverse.add(
            &format!("reverse_{}", block.name),
            "reverse_prepare",
            expressions,
        );
        for ((variable, component), value) in slots.into_iter().zip(values) {
            let mut row = vec![Atom::Zero; OUTPUTS];
            row[component] = value;
            adjoints.push((variable, row));
        }
    }
    let weights: Vec<_> = forward
        .callbacks
        .iter()
        .flat_map(|x| adjoint(&adjoints, x))
        .collect();
    assert!(
        weights
            .iter()
            .all(|w| forward.callbacks.iter().all(|x| !w.contains(x)))
    );
    let weights = reverse.add("reverse_callback_weights", "reverse_prepare", weights);
    let preparation_blocks = reverse.blocks.len();
    reverse.external.extend(forward.callbacks.clone());
    reverse.variables.extend(forward.callbacks.clone());
    let result = reverse.add(
        "reverse_application",
        "reverse_apply",
        (0..OUTPUTS)
            .map(|component| {
                forward
                    .callbacks
                    .iter()
                    .enumerate()
                    .fold(Atom::Zero, |sum, (i, x)| {
                        sum + &weights[OUTPUTS * i + component] * x
                    })
            })
            .collect(),
    );
    Ok((reverse, result, preparation_blocks))
}

fn record(name: &str, category: &str, plan: &ExactKernel) -> ProbeResult<Value> {
    eprintln!("TRANSPOSE_STAGE optimize_and_emit {name}");
    let evaluator = plan.evaluator()?;
    let operations = evaluator.count_operations();
    let instructions = evaluator.export_instructions();
    assert!(instructions.sub_evaluators.is_empty());
    assert!(instructions.constant_functions.is_empty());
    assert_eq!(operations.function_calls, 0);
    let mut emitter = RustEmitter::default();
    let generated = emitter.emit_batch(evaluator, EmitOptions::default())?;
    Ok(
        json!({"name": name, "category": category, "inputs": plan.parameters.len(), "outputs": plan.expressions.len(),
        "instructions": generated.instruction_count, "temporary_slots": instructions.temporary_count,
        "source_bytes": generated.source.len(), "constants_bytes": emitter.constants_source().len(),
        "expression_bytes": plan.expressions.iter().map(|x| x.to_string().len()).sum::<usize>(),
        "operations": {"additions": operations.additions, "multiplications": operations.multiplications,
            "inversions": operations.inversions, "function_calls": operations.function_calls}}),
    )
}

fn records(network: &Network) -> ProbeResult<Vec<Value>> {
    network
        .blocks
        .iter()
        .map(|b| record(&b.name, b.category, &b.exact))
        .collect()
}

fn totals(records: &[Value]) -> Value {
    let sum = |key: &str| {
        records
            .iter()
            .map(|r| r[key].as_u64().unwrap())
            .sum::<u64>()
    };
    let operations = |key: &str| {
        records
            .iter()
            .map(|r| r["operations"][key].as_u64().unwrap())
            .sum::<u64>()
    };
    json!({"instructions": sum("instructions"),
        "separate_temporary_slots_sum": sum("temporary_slots"),
        "largest_local_temporary_slots": records.iter().map(|r| r["temporary_slots"].as_u64().unwrap()).max().unwrap_or(0),
        "unrolled_callsite_source_bytes": sum("source_bytes"),
        "operations": {"additions": operations("additions"), "multiplications": operations("multiplications"),
            "inversions": operations("inversions"), "function_calls": operations("function_calls")}})
}

fn write_report(output: &PathBuf, report: &Value) -> ProbeResult<()> {
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(output, serde_json::to_string_pretty(report)? + "\n")?;
    Ok(())
}

#[test]
#[ignore = "explicit bounded symbolic transpose inventory; no numerical or performance acceptance"]
fn inventory_transpose_n3_r3() -> ProbeResult<()> {
    let output = std::env::var_os("NINJARS_TRANSPOSE_INVENTORY_OUTPUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../target/transpose-inventory-n3-r3.json")
        });
    // Run the produced executable under an external timeout INSIDE flock. A
    // byte cap cannot interrupt one expensive Symbolica expansion/optimizer.
    write_report(
        &output,
        &json!({"schema": "ninjars-transpose-inventory-v1", "status": "incomplete_preparing_network"}),
    )?;
    eprintln!("TRANSPOSE_STAGE prepare_forward_network");
    let (forward, outputs) = forward_network()?;
    write_report(
        &output,
        &json!({"schema": "ninjars-transpose-inventory-v1", "status": "incomplete_recording_forward",
        "forward_blocks": forward.blocks.len(), "callback_heads": forward.callbacks.len()}),
    )?;
    let forward_records = records(&forward)?;
    // Retain a clearly incomplete checkpoint if bounded composition fails.
    write_report(
        &output,
        &json!({"schema": "ninjars-transpose-inventory-v1", "status": "incomplete_before_composition",
        "forward_separated": forward_records, "forward_totals": totals(&forward_records)}),
    )?;
    eprintln!("TRANSPOSE_STAGE compose_forward");
    let forward_fused = forward.compose(&outputs)?;
    eprintln!("TRANSPOSE_STAGE construct_transpose");
    let (reverse, reverse_outputs, preparation_blocks) = transposed_network(&forward, &outputs)?;
    eprintln!("TRANSPOSE_STAGE compose_transpose");
    let reverse_fused = reverse.compose(&reverse_outputs)?;
    for (index, (a, b)) in forward_fused
        .expressions
        .iter()
        .zip(&reverse_fused.expressions)
        .enumerate()
    {
        eprintln!("TRANSPOSE_STAGE exact_identity {index}");
        assert_eq!(
            (a - b).expand(),
            Atom::Zero,
            "exact integrated output {index}"
        );
    }
    let reverse_records = records(&reverse)?;
    let report = json!({
        "schema": "ninjars-transpose-inventory-v1", "propagators": 3, "rank": 3,
        "status": "complete_exact_identities",
        "production_dispatch": false, "numerical_execution": false, "performance_acceptance": false,
        "exact_output_identities": OUTPUTS, "callback_heads": forward.callbacks.len(),
        "symbolic_master_components": forward.master_values.len(),
        "composition_byte_budget": MAX_COMPOSED_BYTES,
        "forward_separated": forward_records,
        "forward_totals": totals(&forward_records),
        "forward_fused": record("forward_fused", "complete", &forward_fused)?,
        "transposed_factored": reverse_records,
        "transposed_totals": totals(&reverse_records),
        "transposed_prepare_totals": totals(&reverse_records[..preparation_blocks]),
        "transposed_apply_totals": totals(&reverse_records[preparation_blocks..]),
        "transposed_preparation_blocks": preparation_blocks,
        "transposed_fused": record("transposed_fused", "complete", &reverse_fused)?,
        "limitations": [
            "Abstract source projection inputs obey no assumed cross-cut identities; no physical cancellation claim.",
            "A negative abstract-geometry result does not exclude savings after actual cut geometry is composed.",
            "Counts include reduction and symbolic master contraction, but no numerator head preparation, geometry construction or master evaluation.",
            "Local forward kernel counts include currently computed unused heads; fused/reverse graphs target integrated outputs only.",
            "Per-block temporary counts describe separate evaluators, not aggregate peak memory; retained cross-block arrays are unmeasured.",
            "Summed source sizes describe unrolled call sites, not the current library's binary size or code shared between cuts.",
            "Reverse weights contain symbolic master values. Production must preserve zero-weight lazy master calls and required-cut failure provenance.",
            "Residue diagnostics, reconstruction and cut-stop require separate output graphs or the full forward fallback.",
            "No conditioning, overflow-boundary, Dual-component, batching or latency equivalence is established by exact identities.",
            "No map-construction amortization is permitted for one numerator at changing kinematics. Multi-numerator reuse needs separate build/apply measurements.",
            "n3/r4 and physical geometry composition are deferred until this bounded n3/r3 graph is evaluated."
        ]
    });
    write_report(&output, &report)?;
    println!("{}", output.display());
    Ok(())
}
