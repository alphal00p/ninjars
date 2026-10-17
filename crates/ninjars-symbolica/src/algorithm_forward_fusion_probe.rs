//! Output-retention control over the existing abstract n3/r3 forward network.
//! No new algebra, production dispatch, transpose or numerical execution.

use std::ops::Range;

use super::*;

/// Five integrated outputs first, then every other existing block output.
/// This is a conservative superset of the current arithmetic diagnostics, not
/// a claim to encode validation order, lane masks or control-flow semantics.
fn retained_heads(network: &Network, integrated: &[Atom]) -> (Vec<Atom>, Vec<Range<usize>>) {
    assert_eq!(integrated.len(), OUTPUTS);
    let mut heads = integrated.to_vec();
    let mut ranges = Vec::with_capacity(network.blocks.len());
    let mut final_blocks = 0;
    for block in &network.blocks {
        if block.outputs.as_slice() == integrated {
            assert_eq!(block.category, "contribution_sum");
            final_blocks += 1;
            ranges.push(0..OUTPUTS);
        } else {
            let start = heads.len();
            heads.extend(block.outputs.clone());
            ranges.push(start..heads.len());
        }
    }
    assert_eq!(final_blocks, 1);
    assert_eq!(
        heads.len(),
        network
            .blocks
            .iter()
            .map(|b| b.outputs.len())
            .sum::<usize>()
    );
    for (index, head) in heads.iter().enumerate() {
        assert!(
            !heads[..index].contains(head),
            "duplicate retained head {head}"
        );
    }
    for (block, range) in network.blocks.iter().zip(&ranges) {
        assert_eq!(&heads[range.clone()], block.outputs.as_slice());
    }
    (heads, ranges)
}

fn head_inventory(network: &Network, ranges: &[Range<usize>]) -> Value {
    let blocks: Vec<_> = network
        .blocks
        .iter()
        .zip(ranges)
        .map(|(block, range)| {
            json!({"name":block.name,"category":block.category,
                "output_range":[range.start,range.end],
                "heads":block.outputs.iter().map(ToString::to_string).collect::<Vec<_>>()})
        })
        .collect();
    let categories: Vec<_> = [
        "division",
        "parent_correction",
        "subtraction_sum",
        "projection",
        "master_weights",
        "master_contraction",
        "contribution_sum",
    ]
    .into_iter()
    .map(|category| {
        let selected: Vec<_> = network
            .blocks
            .iter()
            .filter(|b| b.category == category)
            .collect();
        json!({"category":category,"blocks":selected.len(),
            "heads":selected.iter().map(|b| b.outputs.len()).sum::<usize>()})
    })
    .collect();
    assert_eq!(
        categories
            .iter()
            .map(|r| r["blocks"].as_u64().unwrap() as usize)
            .sum::<usize>(),
        network.blocks.len(),
    );
    json!({"blocks":blocks,"categories":categories})
}

/// Reconstruct every local block from the retained upstream values, instead of
/// comparing a second invocation of whole-network composition with itself.
/// Symbolica substitution and expansion provide all algebraic manipulation.
fn check_local_identities(
    network: &Network,
    heads: &[Atom],
    ranges: &[Range<usize>],
    fused: &ExactKernel,
) -> ProbeResult<Vec<Value>> {
    assert_eq!(fused.parameters, network.external);
    assert_eq!(fused.expressions.len(), heads.len());
    let mut results = Vec::with_capacity(network.blocks.len());
    let mut identity_bytes = 0;
    for (block, range) in network.blocks.iter().zip(ranges) {
        eprintln!("FORWARD_DIAGNOSTICS_STAGE exact_block {}", block.name);
        let arguments: Vec<_> = block
            .exact
            .parameters
            .iter()
            .map(|parameter| {
                if let Some(index) = heads.iter().position(|head| head == parameter) {
                    fused.expressions[index].clone()
                } else {
                    assert!(
                        network.external.contains(parameter),
                        "unbound input {parameter}"
                    );
                    parameter.clone()
                }
            })
            .collect();
        let reconstructed = bind(&block.exact, &arguments);
        identity_bytes += reconstructed
            .iter()
            .map(|e| e.to_string().len())
            .sum::<usize>();
        if identity_bytes > MAX_COMPOSED_BYTES {
            return Err(format!(
                "identity budget exceeded at {}: {identity_bytes} bytes",
                block.name
            )
            .into());
        }
        for (column, (expected, actual)) in reconstructed
            .iter()
            .zip(&fused.expressions[range.clone()])
            .enumerate()
        {
            assert_eq!(
                (actual - expected).expand(),
                Atom::Zero,
                "retained exact output {}[{column}]",
                block.name
            );
            assert!(
                heads.iter().all(|head| !actual.contains(head)),
                "unresolved output reference"
            );
            if !matches!(block.category, "master_contraction" | "contribution_sum") {
                assert!(
                    network
                        .master_values
                        .iter()
                        .all(|master| !actual.contains(master)),
                    "pre-contraction diagnostic depends on an abstract master value"
                );
            }
        }
        results.push(json!({"name":block.name,"category":block.category,
            "output_range":[range.start,range.end],"exact_identities":reconstructed.len()}));
    }
    Ok(results)
}

#[test]
#[ignore = "bounded symbolic forward fusion with retained diagnostics; no performance acceptance"]
fn inventory_forward_diagnostics_n3_r3() -> ProbeResult<()> {
    let output = std::env::var_os("NINJARS_FORWARD_DIAGNOSTICS_OUTPUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../target/forward-diagnostics-n3-r3.json")
        });
    let mut report = json!({
        "schema":"ninjars-forward-diagnostics-v1","propagators":3,"rank":3,
        "status":"incomplete_preparing_network","production_dispatch":false,
        "numerical_execution":false,"performance_acceptance":false,
        "composition_byte_budget":MAX_COMPOSED_BYTES,"required_external_timeout_seconds":180,
        "retention_policy":"all existing forward block outputs; conservative diagnostic superset",
        "final_output_range":[0,OUTPUTS],
        "boundaries":{
            "included":"Existing exact division, parent correction, subtraction, residue projection, symbolic master weights/contractions and amplitude sum",
            "callback_heads":"Unchanged independent callback inputs; no callback execution or depth change",
            "geometry":"Unchanged abstract geometry, denominator and source-projection inputs; their construction is excluded",
            "masters":"Unchanged independent symbolic Laurent components; no master provider or eager master evaluation"
        },
        "unrepresented_checks":[
            "Input shape, finite/all-component carrier checks and callback API errors must remain before evaluation.",
            "Denominator/source-projection construction is outside this network; their finite and leading-denominator checks are not arithmetic outputs here.",
            "Gram/pivot/branch choice, cut topology/provenance/containment and source computed masks require existing control logic.",
            "Output retention does not preserve numerical failure order, active-lane compaction, failed-parent propagation or lane transactionality.",
            "Symbolic identities do not preserve overflow/underflow, rounding, signed zero, conditioning or full Dual numerical behavior after reassociation.",
            "Master weights are retained, but zero-weight lazy master calls, derivative-only weights and nonanalytic-domain errors need independent runtime control.",
            "Complete current residue coefficient arrays are retained; ordinary tadpole unused slots remain zero with computed-mask semantics external.",
            "No cut-stop, reconstruction execution, geometry composition, n3/r4 or transpose extension is implemented."
        ],
        "count_limitations":[
            "Instruction and arithmetic counts are not latency or throughput predictions.",
            "Separate temporary sums are not peak memory; retained-output writes and cross-block packing are unmeasured.",
            "Generated source sizes are unrolled callsite inventories, not binary sizes."
        ]
    });
    write_report(&output, &report)?;
    eprintln!("FORWARD_DIAGNOSTICS_STAGE prepare_forward_network");
    let (forward, integrated) = forward_network()?;
    assert_eq!(forward.callbacks.len(), 60);
    assert_eq!(forward.master_values.len(), 39);
    let (heads, ranges) = retained_heads(&forward, &integrated);
    report["callback_heads"] = json!(forward.callbacks.len());
    report["symbolic_master_components"] = json!(forward.master_values.len());
    report["external_inputs"] = json!(
        forward
            .external
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
    );
    report["retained_output_count"] = json!(heads.len());
    report["retained_head_inventory"] = head_inventory(&forward, &ranges);
    report["status"] = json!("incomplete_recording_separated");
    write_report(&output, &report)?;
    let separated = records(&forward)?;
    report["forward_totals"] = totals(&separated);
    report["forward_separated"] = json!(separated);
    report["status"] = json!("incomplete_five_output_baseline");
    write_report(&output, &report)?;
    let baseline = forward.compose(&integrated)?;
    assert_eq!(baseline.parameters, forward.external);
    assert_eq!(baseline.expressions.len(), OUTPUTS);
    report["forward_fused_five"] = record("forward_fused_five", "integrated_only", &baseline)?;
    report["status"] = json!("incomplete_retained_composition");
    write_report(&output, &report)?;
    eprintln!("FORWARD_DIAGNOSTICS_STAGE compose_retained");
    let retained = forward.compose(&heads)?;
    assert_eq!(retained.parameters, baseline.parameters);
    report["status"] = json!("incomplete_exact_identities");
    write_report(&output, &report)?;
    for (index, (a, b)) in baseline
        .expressions
        .iter()
        .zip(&retained.expressions[..OUTPUTS])
        .enumerate()
    {
        assert_eq!(
            (a - b).expand(),
            Atom::Zero,
            "retained integrated output {index}"
        );
    }
    report["exact_integrated_identities"] = json!(OUTPUTS);
    let identities = check_local_identities(&forward, &heads, &ranges, &retained)?;
    assert_eq!(
        identities
            .iter()
            .map(|r| r["exact_identities"].as_u64().unwrap() as usize)
            .sum::<usize>(),
        heads.len()
    );
    report["exact_block_identities"] = json!(identities);
    report["status"] = json!("incomplete_optimizing_retained");
    write_report(&output, &report)?;
    report["forward_fused_retained"] =
        record("forward_fused_retained", "diagnostic_superset", &retained)?;
    let count = |name: &str| report[name]["instructions"].as_i64().unwrap();
    let differences = json!({
        "separated_minus_five_output":count("forward_totals")-count("forward_fused_five"),
        "retained_minus_five_output":count("forward_fused_retained")-count("forward_fused_five"),
        "separated_minus_retained":count("forward_totals")-count("forward_fused_retained")
    });
    report["instruction_differences"] = differences;
    report["status"] = json!("complete_exact_identities");
    write_report(&output, &report)?;
    println!("{}", output.display());
    Ok(())
}
