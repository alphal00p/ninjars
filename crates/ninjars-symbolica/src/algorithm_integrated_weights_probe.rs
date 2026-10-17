//! Bounded pre-master fusion inventory over the unchanged n3/r3 network.
//! Symbolica owns algebra; this module only chooses graph boundaries.

use ninjars::MasterIntegral;

use super::*;

const WEIGHT_HEADS: usize = 20;
const TRIANGLE_HEADS: usize = 10;
const BUBBLE_HEADS: usize = 30;

fn pre_master(block: &Block) -> bool {
    !matches!(block.category, "master_contraction" | "contribution_sum")
}

fn block<'a>(network: &'a Network, name: &str) -> &'a Block {
    network.blocks.iter().find(|b| b.name == name).unwrap()
}

fn symbols(atoms: &[Atom]) -> Vec<String> {
    atoms.iter().map(ToString::to_string).collect()
}

#[derive(Default)]
struct ByteBudget(usize);

impl ByteBudget {
    fn include(&mut self, context: &str, expressions: &[Atom]) -> ProbeResult<()> {
        self.0 += expressions
            .iter()
            .map(|e| e.to_string().len())
            .sum::<usize>();
        if self.0 > MAX_COMPOSED_BYTES {
            return Err(format!("{context}: expression budget exceeded ({} bytes)", self.0).into());
        }
        Ok(())
    }
}

struct Stage {
    plan: ExactKernel,
    blocks: Vec<usize>,
    prefix: Vec<Atom>,
    composition_bytes: usize,
}

/// Backward graph selection stops at the explicitly retained boundary atoms.
/// Forward substitution expands only selected local blocks. This is graph
/// wiring, not an algebra optimizer; the evaluator performs dead-head removal.
fn stage(network: &Network, heads: &[Atom], prefix: &[Atom]) -> ProbeResult<Stage> {
    let mut required = heads.to_vec();
    let mut selected = Vec::new();
    for (index, candidate) in network.blocks.iter().enumerate().rev() {
        if candidate
            .outputs
            .iter()
            .any(|x| required.contains(x) && !prefix.contains(x))
        {
            assert!(
                pre_master(candidate),
                "master block entered a pre-master stage"
            );
            assert!(candidate.outputs.iter().all(|x| !prefix.contains(x)));
            selected.push(index);
            required.extend(candidate.exact.parameters.clone());
        }
    }
    selected.reverse();
    let mut from = Vec::new();
    let mut to = Vec::new();
    let mut budget = ByteBudget::default();
    for &index in &selected {
        let current = &network.blocks[index];
        let expressions = substitute(&current.exact.expressions, &from, &to);
        budget.include(&current.name, &expressions)?;
        from.extend(current.outputs.clone());
        to.extend(expressions);
    }
    let expressions = substitute(heads, &from, &to);
    budget.include("stage outputs", &expressions)?;
    let mut parameters = prefix.to_vec();
    parameters.extend(
        network
            .external
            .iter()
            .filter(|p| required.contains(p))
            .cloned(),
    );
    let plan = ExactKernel {
        expressions,
        parameters,
    };
    check_pre_master(network, &plan, prefix);
    Ok(Stage {
        plan,
        blocks: selected,
        prefix: prefix.to_vec(),
        composition_bytes: budget.0,
    })
}

fn check_pre_master(network: &Network, plan: &ExactKernel, prefix: &[Atom]) {
    assert_eq!(&plan.parameters[..prefix.len()], prefix);
    for (index, parameter) in plan.parameters.iter().enumerate() {
        assert!(
            !plan.parameters[..index].contains(parameter),
            "duplicate parameter"
        );
        assert!(
            !network.master_values.contains(parameter),
            "master parameter before weights"
        );
    }
    let suffix: Vec<_> = network
        .external
        .iter()
        .filter(|p| plan.parameters.contains(p))
        .cloned()
        .collect();
    assert_eq!(&plan.parameters[prefix.len()..], suffix.as_slice());
    for expression in &plan.expressions {
        assert!(
            network
                .master_values
                .iter()
                .all(|m| !expression.contains(m))
        );
        for intermediate in network.blocks.iter().flat_map(|b| &b.outputs) {
            assert!(
                prefix.contains(intermediate) || !expression.contains(intermediate),
                "unresolved nonboundary intermediate {intermediate}"
            );
        }
        for variable in &network.variables {
            assert!(
                !expression.contains(variable) || plan.parameters.contains(variable),
                "unbound stage input {variable}"
            );
        }
    }
}

fn stage_layout(network: &Network, stage: &Stage, heads: &[Atom]) -> Value {
    json!({
        "boundary_prefix":symbols(&stage.prefix),
        "boundary_prefix_range":[0,stage.prefix.len()],
        "external_suffix":symbols(&stage.plan.parameters[stage.prefix.len()..]),
        "external_suffix_policy":"inputs used by selected original blocks, including slots eliminated after fusion",
        "external_suffix_range":[stage.prefix.len(),stage.plan.parameters.len()],
        "expression_live_inputs":symbols(&stage.plan.parameters.iter()
            .filter(|p| stage.plan.expressions.iter().any(|e| e.contains(*p)))
            .cloned().collect::<Vec<_>>()),
        "retained_unused_inputs":symbols(&stage.plan.parameters.iter()
            .filter(|p| stage.plan.expressions.iter().all(|e| !e.contains(*p)))
            .cloned().collect::<Vec<_>>()),
        "callback_inputs":symbols(&stage.plan.parameters.iter()
            .filter(|p| network.callbacks.contains(p)).cloned().collect::<Vec<_>>()),
        "outputs":symbols(heads),
        "local_blocks":stage.blocks.iter().map(|&i| network.blocks[i].name.as_str()).collect::<Vec<_>>(),
        "composition_bytes":stage.composition_bytes
    })
}

/// Assert cut/master order using the owning MasterContraction API, and bind its
/// unchanged contraction against the network's own ordered weight/master atoms.
fn ordered_weights(network: &Network) -> ProbeResult<(Vec<Atom>, Vec<Value>)> {
    let cuts = [
        ("tri012", 3, vec![MasterIntegral::C0]),
        (
            "bub01",
            2,
            vec![MasterIntegral::B0, MasterIntegral::B1, MasterIntegral::B11],
        ),
        (
            "bub02",
            2,
            vec![MasterIntegral::B0, MasterIntegral::B1, MasterIntegral::B11],
        ),
        (
            "bub12",
            2,
            vec![MasterIntegral::B0, MasterIntegral::B1, MasterIntegral::B11],
        ),
        ("tad0_o0", 1, vec![MasterIntegral::A0]),
        ("tad1_o0", 1, vec![MasterIntegral::A0]),
        ("tad2_o0", 1, vec![MasterIntegral::A0]),
    ];
    let mut heads = Vec::new();
    let mut inventory = Vec::new();
    let mut master_offset = 0;
    for (label, size, expected_masters) in cuts {
        let plan = MasterContraction::new(size, false)?;
        assert_eq!(plan.masters(), expected_masters.as_slice());
        let weights = block(network, &format!("{label}_weights"));
        assert_eq!(weights.category, "master_weights");
        assert_eq!(weights.outputs.len(), plan.weight_columns());
        let end = master_offset + plan.masters().len() * 3;
        let masters = &network.master_values[master_offset..end];
        let expected_symbols: Vec<_> = (0..masters.len())
            .map(|i| Atom::var(symbol!(format!("ninjars_transpose::{label}_masters_{i}"))))
            .collect();
        assert_eq!(masters, expected_symbols.as_slice());
        let mut arguments = weights.outputs.clone();
        arguments.extend_from_slice(masters);
        let contract = block(network, &format!("{label}_contract"));
        assert_eq!(contract.category, "master_contraction");
        assert_eq!(
            bind(&plan.contraction, &arguments),
            contract.exact.expressions
        );
        for (local, head) in weights.outputs.iter().enumerate() {
            let kind = plan
                .masters()
                .get(local)
                .map(|m| format!("{m:?}"))
                .unwrap_or_else(|| "rational".into());
            inventory.push(json!({"column":heads.len(),"cut":label,"cut_size":size,
                "local_column":local,"kind":kind,"symbol":head.to_string()}));
            heads.push(head.clone());
        }
        master_offset = end;
    }
    assert_eq!(master_offset, network.master_values.len());
    assert_eq!(heads.len(), WEIGHT_HEADS);
    let network_order: Vec<_> = network
        .blocks
        .iter()
        .filter(|b| b.category == "master_weights")
        .flat_map(|b| b.outputs.clone())
        .collect();
    assert_eq!(heads, network_order);
    Ok((heads, inventory))
}

fn exact_rows(
    label: &str,
    actual: &[Atom],
    expected: &[Atom],
    budget: &mut ByteBudget,
) -> ProbeResult<Value> {
    assert_eq!(actual.len(), expected.len());
    budget.include(label, actual)?;
    budget.include(label, expected)?;
    for (column, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert_eq!((a - b).expand(), Atom::Zero, "{label}[{column}]");
    }
    Ok(json!({"label":label,"exact_identities":actual.len()}))
}

/// Only unchanged contraction/sum blocks are added after the 20 weight values.
fn recover_integrated(
    network: &Network,
    integrated: &[Atom],
    weights: &[Atom],
    values: &[Atom],
) -> ProbeResult<Vec<Atom>> {
    assert_eq!(weights.len(), WEIGHT_HEADS);
    assert_eq!(values.len(), WEIGHT_HEADS);
    let mut from = weights.to_vec();
    let mut to = values.to_vec();
    let mut budget = ByteBudget::default();
    for current in network.blocks.iter().filter(|b| !pre_master(b)) {
        let expressions = substitute(&current.exact.expressions, &from, &to);
        budget.include(&current.name, &expressions)?;
        from.extend(current.outputs.clone());
        to.extend(expressions);
    }
    let result = substitute(integrated, &from, &to);
    budget.include("integrated recovery", &result)?;
    for expression in &result {
        assert!(
            network
                .blocks
                .iter()
                .flat_map(|b| &b.outputs)
                .all(|p| !expression.contains(p))
        );
    }
    Ok(result)
}

#[test]
#[ignore = "bounded symbolic pre-master fusion inventory; no numerical or performance acceptance"]
fn inventory_integrated_weights_n3_r3() -> ProbeResult<()> {
    let output = std::env::var_os("NINJARS_INTEGRATED_WEIGHTS_OUTPUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../target/integrated-weights-n3-r3.json")
        });
    let mut report = json!({
        "schema":"ninjars-integrated-weights-v1","propagators":3,"rank":3,
        "status":"incomplete_preparing_network","production_dispatch":false,
        "numerical_execution":false,"performance_acceptance":false,
        "composition_byte_budget":MAX_COMPOSED_BYTES,"identity_byte_budget":MAX_COMPOSED_BYTES,
        "composition_budget_scope":"Each candidate stage, reference composition, and integrated recovery has its own cumulative 8 MiB cap; not a global memory bound",
        "identity_budget_scope":"One cumulative 8 MiB cap across all exact-row identity groups",
        "required_external_timeout_seconds":180,
        "candidates":["weights20","staged_triangle10_bubbles30_weights20"],
        "boundaries":{
            "geometry":"Existing abstract inputs; geometry/denominator/source-projection construction excluded",
            "callbacks":"Unchanged independent Laurent heads; no callback execution",
            "masters":"No master values in weight dependencies; recovery alone uses symbolic Laurent components",
            "staging":"Three actual evaluators; boundary values are parameters, not recomputed expressions"
        },
        "limitations":[
            "Exact symbolic equality does not validate numerical overflow, conditioning, Dual zero masks, or failure semantics.",
            "The one-shot path requires eligible pure callbacks or an explicit changed callback contract.",
            "Stage boundaries permit parent checks but do not implement callback eligibility, cut-stop, diagnostics, or transactionality.",
            "Temporary sums are not peak memory; buffer packing and output writes are not measured.",
            "No arithmetic count is a runtime or performance claim."
        ]
    });
    write_report(&output, &report)?;
    let (network, integrated) = forward_network()?;
    assert_eq!(network.callbacks.len(), 60);
    assert_eq!(network.master_values.len(), 39);
    let (weights, order) = ordered_weights(&network)?;
    let triangle = block(&network, "tri012_projection").outputs.clone();
    let bubbles: Vec<_> = ["bub01_projection", "bub02_projection", "bub12_projection"]
        .iter()
        .flat_map(|name| block(&network, name).outputs.clone())
        .collect();
    assert_eq!(triangle.len(), TRIANGLE_HEADS);
    assert_eq!(bubbles.len(), BUBBLE_HEADS);
    let parents: Vec<_> = triangle.iter().chain(&bubbles).cloned().collect();
    report["ordered_weights"] = json!(order);
    report["external_inputs"] = json!(symbols(&network.external));
    report["callback_inputs"] = json!(symbols(&network.callbacks));
    report["symbolic_master_inputs"] = json!(symbols(&network.master_values));
    report["pre_master_external_inputs"] = json!(symbols(
        &network
            .external
            .iter()
            .filter(|p| !network.master_values.contains(p))
            .cloned()
            .collect::<Vec<_>>()
    ));
    report["status"] = json!("incomplete_separate_inventory");
    write_report(&output, &report)?;
    let mut separate = Vec::new();
    let mut before_masters = Vec::new();
    for current in &network.blocks {
        let row = record(&current.name, current.category, &current.exact)?;
        if pre_master(current) {
            before_masters.push(row.clone());
        }
        separate.push(row);
        report["separate_blocks"] = json!(separate);
        report["last_recorded_block"] = json!(current.name);
        write_report(&output, &report)?;
    }
    assert_eq!(
        before_masters
            .iter()
            .map(|v| v["outputs"].as_u64().unwrap())
            .sum::<u64>(),
        243
    );
    report["separate_full_totals"] = totals(&separate);
    report["separate_pre_master_totals"] = totals(&before_masters);
    report["separate_pre_master_blocks"] = json!(before_masters);
    report["status"] = json!("incomplete_composing_weights20");
    write_report(&output, &report)?;

    let one_shot = stage(&network, &weights, &[])?;
    report["weights20_layout"] = stage_layout(&network, &one_shot, &weights);
    report["status"] = json!("incomplete_composing_triangle10");
    write_report(&output, &report)?;
    let triangles = stage(&network, &triangle, &[])?;
    let mut stage_layouts = vec![stage_layout(&network, &triangles, &triangle)];
    report["staged_layouts"] = json!(stage_layouts);
    report["status"] = json!("incomplete_composing_bubbles30");
    write_report(&output, &report)?;
    let bubbles_stage = stage(&network, &bubbles, &triangle)?;
    stage_layouts.push(stage_layout(&network, &bubbles_stage, &bubbles));
    report["staged_layouts"] = json!(stage_layouts);
    report["status"] = json!("incomplete_composing_final_weights20");
    write_report(&output, &report)?;
    let final_stage = stage(&network, &weights, &parents)?;
    stage_layouts.push(stage_layout(&network, &final_stage, &weights));
    report["staged_layouts"] = json!(stage_layouts);
    report["status"] = json!("incomplete_checking_stage_boundaries");
    write_report(&output, &report)?;
    let stages = [&triangles, &bubbles_stage, &final_stage];
    let selected: Vec<_> = stages
        .iter()
        .flat_map(|s| s.blocks.iter().copied())
        .collect();
    for (i, index) in selected.iter().enumerate() {
        assert!(!selected[..i].contains(index), "earlier stage recomputed");
    }
    let mut selected_order = selected;
    selected_order.sort_unstable();
    assert_eq!(
        selected_order,
        network
            .blocks
            .iter()
            .enumerate()
            .filter_map(|(i, b)| pre_master(b).then_some(i))
            .collect::<Vec<_>>()
    );
    let callback_groups: Vec<_> = stages
        .iter()
        .map(|s| {
            s.plan
                .parameters
                .iter()
                .filter(|p| network.callbacks.contains(p))
                .cloned()
                .collect::<Vec<_>>()
        })
        .collect();
    let staged_callbacks: Vec<_> = callback_groups.iter().flatten().cloned().collect();
    assert_eq!(staged_callbacks, network.callbacks);
    report["staged_parent_buffer_values"] = json!(parents.len());
    report["candidate_composition_bytes"] = json!({
        "weights20":one_shot.composition_bytes,
        "staged_sum":stages.iter().map(|s| s.composition_bytes).sum::<usize>(),
        "all_candidate_compositions_sum":one_shot.composition_bytes
            + stages.iter().map(|s| s.composition_bytes).sum::<usize>(),
        "scope":"Cumulative substituted expression text plus output text per candidate; reference/recovery composition excluded; not resident memory"
    });
    report["status"] = json!("incomplete_exact_identities");
    write_report(&output, &report)?;

    // Reference composition does not use stage selection, and is not inventoried
    // as a fictitious single 60-output staged evaluator.
    let all_heads: Vec<_> = parents.iter().chain(&weights).cloned().collect();
    let reference = network.compose(&all_heads)?;
    for expression in &reference.expressions {
        assert!(
            network
                .master_values
                .iter()
                .all(|m| !expression.contains(m))
        );
        assert!(
            network
                .blocks
                .iter()
                .flat_map(|b| &b.outputs)
                .all(|p| !expression.contains(p))
        );
    }
    let full_bubbles = substitute(
        &bubbles_stage.plan.expressions,
        &triangle,
        &triangles.plan.expressions,
    );
    let full_parents: Vec<_> = triangles
        .plan
        .expressions
        .iter()
        .chain(&full_bubbles)
        .cloned()
        .collect();
    let full_weights = substitute(&final_stage.plan.expressions, &parents, &full_parents);
    let mut identity_budget = ByteBudget::default();
    let mut identities = Vec::new();
    for (label, actual, expected) in [
        (
            "triangle_boundary10",
            triangles.plan.expressions.as_slice(),
            &reference.expressions[..10],
        ),
        (
            "bubble_boundary30",
            full_bubbles.as_slice(),
            &reference.expressions[10..40],
        ),
        (
            "staged_weights20",
            full_weights.as_slice(),
            &reference.expressions[40..],
        ),
        (
            "weights_only20",
            one_shot.plan.expressions.as_slice(),
            &reference.expressions[40..],
        ),
    ] {
        identities.push(exact_rows(label, actual, expected, &mut identity_budget)?);
        report["exact_identities"] = json!(identities);
        write_report(&output, &report)?;
    }
    let integrated_reference = network.compose(&integrated)?;
    for (label, values) in [
        (
            "weights_only_recovered_integrated5",
            &one_shot.plan.expressions,
        ),
        ("staged_recovered_integrated5", &full_weights),
    ] {
        let recovered = recover_integrated(&network, &integrated, &weights, values)?;
        identities.push(exact_rows(
            label,
            &recovered,
            &integrated_reference.expressions,
            &mut identity_budget,
        )?);
        report["exact_identities"] = json!(identities);
        write_report(&output, &report)?;
    }
    report["identity_expression_bytes"] = json!(identity_budget.0);
    report["status"] = json!("incomplete_candidate_inventory");
    write_report(&output, &report)?;
    report["weights20"] = record("weights20", "pre_master_fused", &one_shot.plan)?;
    write_report(&output, &report)?;
    let mut staged_records = Vec::new();
    for (name, stage) in [
        ("triangle10", &triangles),
        ("bubbles30", &bubbles_stage),
        ("weights20", &final_stage),
    ] {
        staged_records.push(record(name, "pre_master_stage", &stage.plan)?);
        report["staged_evaluators"] = json!(staged_records);
        write_report(&output, &report)?;
    }
    report["staged_totals"] = totals(&staged_records);
    let baseline = report["separate_pre_master_totals"]["instructions"]
        .as_i64()
        .unwrap();
    report["instruction_differences"] = json!({
        "separate_pre_master_minus_weights20":baseline-report["weights20"]["instructions"].as_i64().unwrap(),
        "separate_pre_master_minus_staged":baseline-report["staged_totals"]["instructions"].as_i64().unwrap()
    });
    report["status"] = json!("complete_exact_identities");
    write_report(&output, &report)?;
    println!("{}", output.display());
    Ok(())
}
