//! Bounded symbolic inventory only; no production dispatch or numeric execution.
//!
//! Rebind existing exact plans with Symbolica simultaneous replacement, then
//! concatenate their evaluators with Symbolica's existing merge optimization.
//! Transverse vectors remain independent and all thirty heads remain outputs.

use std::{error::Error, path::Path};

use ninjars::{Complex, ExpansionKind};
use serde_json::{Value, json};
use symbolica::{
    atom::{Atom, AtomCore},
    domains::rational::Rational,
    evaluate::ExpressionEvaluator,
    id::{Pattern, Replacement},
    symbol,
};

use crate::{BubbleSourceProjection, DenominatorExpansion, kernel::ExactKernel};

type ProbeResult<T> = Result<T, Box<dyn Error>>;
type Evaluator = ExpressionEvaluator<Complex<Rational>>;
const MAX_EXPRESSION_BYTES: usize = 8 * 1024 * 1024;

// These are parameter indices, not new physical or polynomial formulae.
const DENOMINATOR_FIRST: [usize; 29] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
    34, 35, 36,
];
const DENOMINATOR_SWAPPED: [usize; 29] = [
    0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
    34, 35, 36,
];
const PROJECTION_FIRST: [usize; 28] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
    34, 35,
];
const PROJECTION_SWAPPED: [usize; 28] = [
    0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
    34, 35,
];

#[derive(Default)]
struct ExpressionBudget {
    bytes: usize,
}

impl ExpressionBudget {
    fn text(&mut self, expression: &Atom) -> ProbeResult<String> {
        let text = expression.to_string();
        self.bytes = self
            .bytes
            .checked_add(text.len())
            .ok_or("expression text size overflow")?;
        if self.bytes > MAX_EXPRESSION_BYTES {
            return Err(format!("expression text budget exceeded: {} bytes", self.bytes).into());
        }
        Ok(text)
    }
}

fn substitute(expressions: &[Atom], from: &[Atom], to: &[Atom]) -> Vec<Atom> {
    assert_eq!(from.len(), to.len());
    let replacements: Vec<_> = from
        .iter()
        .zip(to)
        .map(|(a, b)| Replacement::new(Pattern::Literal(a.clone()), Pattern::Literal(b.clone())))
        .collect();
    expressions
        .iter()
        .map(|expression| expression.replace_multiple(&replacements))
        .collect()
}

fn vector_labels(names: &[&str]) -> Vec<String> {
    names
        .iter()
        .flat_map(|name| ["E", "x", "y", "z"].map(|component| format!("{name}_{component}")))
        .collect()
}

struct Family {
    name: &'static str,
    original: ExactKernel,
    labels: Vec<String>,
    maps: [Vec<usize>; 2],
    heads: Vec<Value>,
    shared_heads: Vec<usize>,
}

fn denominator_family() -> ProbeResult<Family> {
    let plan = DenominatorExpansion::new(2)?;
    let ExpansionKind::T2 { powers } = plan.expansion() else {
        return Err("expected existing T2 denominator layout".into());
    };
    assert_eq!(
        powers.as_slice(),
        &[
            (1, 0, 0),
            (0, 0, 0),
            (0, 1, 0),
            (-1, 0, 0),
            (-1, 0, 1),
            (-1, 1, 0),
            (-1, 2, 0)
        ]
    );
    let heads = powers
        .iter()
        .map(|power| json!({"laurent_power_t_x_mu2":power}))
        .collect();
    let mut labels = vector_labels(&[
        "a0",
        "a1",
        "growing0",
        "shrinking0",
        "growing1",
        "shrinking1",
    ]);
    labels.extend(["beta0", "beta1", "beta2"].map(str::to_owned));
    labels.extend(vector_labels(&["Vanchor"]));
    labels.push("manchor2".into());
    labels.extend(vector_labels(&["Vline"]));
    labels.push("mline2".into());
    assert_eq!(labels.len(), 37);
    Ok(Family {
        name: "denominator",
        original: plan.exact,
        labels,
        maps: [DENOMINATOR_FIRST.to_vec(), DENOMINATOR_SWAPPED.to_vec()],
        heads,
        shared_heads: vec![1, 2],
    })
}

fn projection_family() -> Family {
    let labels = vector_labels(&[
        "a0",
        "a1",
        "growing0",
        "shrinking0",
        "growing1",
        "shrinking1",
        "Vtriangle",
        "E3",
        "E4",
    ]);
    assert_eq!(labels.len(), 36);
    let heads = ["E3", "E4"]
        .into_iter()
        .flat_map(|axis| {
            ["a0_plus_Vtriangle", "a1", "growing", "shrinking"]
                .map(|projected| json!({"source_axis":axis,"projected":projected}))
        })
        .collect();
    Family {
        name: "source_projection",
        original: BubbleSourceProjection::new().exact,
        labels,
        maps: [PROJECTION_FIRST.to_vec(), PROJECTION_SWAPPED.to_vec()],
        heads,
        shared_heads: vec![0, 1, 4, 5],
    }
}

fn inventory(evaluator: &Evaluator, expression_bytes: usize) -> Value {
    let exported = evaluator.export_instructions();
    let operations = evaluator.count_operations();
    assert_eq!(
        operations.inversions, 0,
        "no normalization/division composition"
    );
    assert_eq!(
        operations.function_calls, 0,
        "no helpers or master execution"
    );
    assert!(exported.sub_evaluators.is_empty());
    assert!(exported.constant_functions.is_empty());
    json!({
        "inputs":exported.input_count,"outputs":exported.output_count,
        "instructions":exported.instructions.len(),
        "additions":operations.additions,"multiplications":operations.multiplications,
        "inversions":operations.inversions,"function_calls":operations.function_calls,
        "arithmetic_operations":operations.additions+operations.multiplications+operations.inversions,
        "operation_scope":"Symbolica scalar-carrier operations; not real flop counts or SIMD instructions",
        "temporary_slots":exported.temporary_count,"constants":exported.constants.len(),
        "expression_text_bytes":expression_bytes,
        "expression_size_scope":"retained exact heads before instruction optimization"
    })
}

fn separate_total(first: &Value, swapped: &Value) -> Value {
    let mut total = json!({
        "evaluations":2,
        "input_columns_per_orientation":[first["inputs"],swapped["inputs"]],
        "temporary_slots_largest":first["temporary_slots"].as_u64().unwrap()
            .max(swapped["temporary_slots"].as_u64().unwrap()),
        "temporary_scope":"sum is not peak memory; the runtime may reuse a single evaluator",
    });
    for key in [
        "outputs",
        "instructions",
        "additions",
        "multiplications",
        "inversions",
        "function_calls",
        "arithmetic_operations",
        "temporary_slots",
        "constants",
        "expression_text_bytes",
    ] {
        total[key] = json!(first[key].as_u64().unwrap() + swapped[key].as_u64().unwrap());
    }
    total
}

fn write_report(path: &Path, report: &Value) -> ProbeResult<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_vec_pretty(report)?)?;
    Ok(())
}

fn record_family(
    family: Family,
    output: &Path,
    report: &mut Value,
    budget: &mut ExpressionBudget,
) -> ProbeResult<()> {
    let canonical: Vec<_> = family
        .labels
        .iter()
        .map(|label| Atom::var(symbol!(format!("ninjars_paired_{}::{label}", family.name))))
        .collect();
    assert_eq!(family.original.expressions.len(), family.heads.len());
    let original_text: Vec<_> = family
        .original
        .expressions
        .iter()
        .map(|expression| budget.text(expression))
        .collect::<ProbeResult<_>>()?;
    let original_bytes = original_text.iter().map(String::len).sum();
    let mut mapped = Vec::new();
    let mut heads = Vec::new();
    let mut mapped_bytes = Vec::new();
    for (orientation, map) in family.maps.iter().enumerate() {
        assert_eq!(map.len(), family.original.parameters.len());
        assert!(map.iter().all(|&index| index < canonical.len()));
        let arguments: Vec<_> = map.iter().map(|&index| canonical[index].clone()).collect();
        for (index, parameter) in arguments.iter().enumerate() {
            assert!(
                !arguments[..index].contains(parameter),
                "noninjective parameter map"
            );
        }
        let expressions = substitute(
            &family.original.expressions,
            &family.original.parameters,
            &arguments,
        );
        let restored = substitute(&expressions, &arguments, &family.original.parameters);
        let mut bytes = 0;
        for (column, ((expression, restored), original)) in expressions
            .iter()
            .zip(&restored)
            .zip(&family.original.expressions)
            .enumerate()
        {
            assert_eq!((restored - original).expand(), Atom::Zero);
            let text = budget.text(expression)?;
            bytes += text.len();
            heads.push(json!({
                "output_index":orientation*family.heads.len()+column,
                "orientation":if orientation==0 {"first"} else {"swapped"},
                "original_head_index":column,"head":family.heads[column],
                "exact_expression":text,"roundtrip_identity":true,
            }));
        }
        mapped_bytes.push(bytes);
        mapped.push(ExactKernel {
            parameters: canonical.clone(),
            expressions,
        });
    }
    // These are exact identities of existing expressions, not replacement rules
    // used to derive or prune any of the thirty retained outputs.
    for &column in &family.shared_heads {
        assert_eq!(
            (&mapped[0].expressions[column] - &mapped[1].expressions[column]).expand(),
            Atom::Zero
        );
    }
    assert_eq!(mapped[0].parameters, mapped[1].parameters);
    let name = family.name;
    report["families"][name] = json!({
        "canonical_parameters":canonical.iter().map(ToString::to_string).collect::<Vec<_>>(),
        "canonical_input_labels":family.labels,
        "original_parameters":family.original.parameters.iter().map(ToString::to_string).collect::<Vec<_>>(),
        "original_head_expressions":original_text,
        "parameter_maps":{"first":family.maps[0],"swapped":family.maps[1]},
        "retained_outputs":heads,"exact_roundtrip_identities":heads.len(),
        "output_ranges":{"first":[0,family.heads.len()],"swapped":[family.heads.len(),2*family.heads.len()]},
        "exact_shared_head_indices":family.shared_heads,
    });
    report["retained_expression_text_bytes"] = json!(budget.bytes);
    report["status"] = json!(format!("incomplete_{name}_separate_evaluators"));
    write_report(output, report)?;

    let original = family.original.evaluator()?;
    let original_inventory = inventory(&original, original_bytes);
    report["families"][name]["original_two_evaluations"] =
        separate_total(&original_inventory, &original_inventory);
    report["families"][name]["original_single"] = original_inventory;
    let first = mapped[0].evaluator()?;
    let swapped = mapped[1].evaluator()?;
    assert_eq!(first.get_input_len(), canonical.len());
    assert_eq!(swapped.get_input_len(), canonical.len());
    let first_inventory = inventory(&first, mapped_bytes[0]);
    let swapped_inventory = inventory(&swapped, mapped_bytes[1]);
    report["families"][name]["separated"] = separate_total(&first_inventory, &swapped_inventory);
    report["families"][name]["separate_first"] = first_inventory;
    report["families"][name]["separate_swapped"] = swapped_inventory;
    for rounds in [0, 2] {
        let label = format!("merged_cpe_{rounds}");
        report["status"] = json!(format!("incomplete_{name}_{label}"));
        write_report(output, report)?;
        let mut merged = first.clone();
        merged.merge(swapped.clone(), Some(rounds))?;
        assert_eq!(merged.get_input_len(), canonical.len());
        assert_eq!(merged.get_output_len(), 2 * family.heads.len());
        report["families"][name][label.as_str()] = inventory(&merged, mapped_bytes.iter().sum());
    }
    report["status"] = json!(format!("complete_{name}_inventory"));
    write_report(output, report)
}

#[test]
#[ignore = "bounded symbolic paired orientations; no numerical or performance acceptance"]
fn inventory_paired_orientations() -> ProbeResult<()> {
    let output = std::env::var_os("NINJARS_PAIRED_ORIENTATION_OUTPUT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../target/paired-orientation-inventory.json")
        });
    let mut report = json!({
        "schema":"ninjars-paired-orientation-inventory-v1",
        "source_base":"c13cc2fdce0508bbf0dd5d8ce88aa340f581c94c",
        "status":"incomplete_preparing_existing_plans",
        "production_dispatch":false,"numerical_execution":false,
        "performance_acceptance":false,"expression_byte_budget":MAX_EXPRESSION_BYTES,
        "required_external_timeout_seconds":180,"retained_expression_text_bytes":0,
        "families":{},
        "boundaries":[
            "Two independent paired exact plans: T2 denominators and bubble source projections.",
            "All seven/eight heads per orientation are retained: thirty total.",
            "Four transverse directions are independent supplied inputs; no swap/helper composition or cut identities.",
            "No quotient, correction, master, callback or physical-geometry graph is composed.",
            "The original plan inventory is an additional control for canonical parameter remapping.",
            "Merged expression sizes describe unchanged retained heads, not reconstructed optimized expressions.",
            "Exact checks establish parameter/head wiring, not independent physics or merged numerical execution.",
            "Public paired provenance, eligibility, compaction, per-orientation status, full Dual precision/shape and transactional publication remain unresolved runtime work.",
            "Operation counts exclude packing, output transport and validation and do not predict speedup."
        ]
    });
    write_report(&output, &report)?;
    let mut budget = ExpressionBudget::default();
    for family in [denominator_family()?, projection_family()] {
        eprintln!("PAIRED_ORIENTATION_INVENTORY {}", family.name);
        record_family(family, &output, &mut report, &mut budget)?;
    }
    let identities: u64 = report["families"]
        .as_object()
        .unwrap()
        .values()
        .map(|family| family["exact_roundtrip_identities"].as_u64().unwrap())
        .sum();
    assert_eq!(identities, 30);
    report["exact_roundtrip_identities"] = json!(identities);
    report["status"] = json!("complete_symbolic_inventory");
    write_report(&output, &report)?;
    println!("{}", output.display());
    Ok(())
}
