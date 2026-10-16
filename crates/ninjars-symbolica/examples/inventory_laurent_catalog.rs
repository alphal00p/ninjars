//! Inventory ordinary finite-topology Laurent exports without installing them.
use std::{collections::BTreeMap, error::Error, fs, path::PathBuf};

use ninjars_symbolica::{
    LaurentDivision,
    rust_codegen::{EmitOptions, RustEmitter, RustKernelExport},
};
use serde_json::json;

fn main() -> Result<(), Box<dyn Error>> {
    let output = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| "target/laurent-catalog-inventory.json".into());
    let mut records = Vec::new();
    let mut identical_sources: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut empty = 0usize;
    for n in 1..=6 {
        for rank in 0..=n + 1 {
            for cut in 1..=n.min(3) {
                let plan = LaurentDivision::new(n, rank, cut)?;
                if plan.quotient_expansion().is_empty() {
                    empty += 1;
                    continue;
                }
                // A fresh constant table makes byte comparison meaningful.
                // Equal generated source is an inventory observation, not
                // proof that runtime layouts may be selected by dimensions.
                let mut emitter = RustEmitter::default();
                let generated = plan.export_rust_batch(&mut emitter, EmitOptions::default())?;
                let constants = emitter.constants_source();
                let key = format!("{constants}\n{}", generated.source);
                let name = format!("n{n}_rank{rank}_cut{cut}");
                identical_sources.entry(key).or_default().push(name.clone());
                records.push(json!({
                    "name":name,"propagators":n,"rank":rank,"cut_size":cut,
                    "uncut":plan.uncut_count(),"higher_rank":plan.higher_rank(),
                    "numerator":format!("{:?}",plan.numerator_expansion()),
                    "denominator":format!("{:?}",plan.denominator_expansion()),
                    "quotient":format!("{:?}",plan.quotient_expansion()),
                    "inputs":generated.input_count,"outputs":generated.output_count,
                    "instructions":generated.instruction_count,"helpers":generated.helper_count,
                    "outlined_branches":generated.outlined_branch_count,
                    "source_bytes":generated.source.len(),"constants_bytes":constants.len(),
                    "constant_count":emitter.constants().len()
                }));
            }
        }
    }
    let groups: Vec<_> = identical_sources.values().cloned().collect();
    let bytes: usize = records
        .iter()
        .map(|r| r["source_bytes"].as_u64().unwrap() as usize)
        .sum();
    let report = json!({
        "scope":"Existing exact LaurentDivision plans for n<=6 and rank<=n+1; ordinary RustEmitter batch export only",
        "production_dispatch":false,"dual_generated":false,"performance_acceptance":false,
        "nonempty_plans":records.len(),"empty_plans":empty,
        "total_source_bytes":bytes,"unique_source_and_constant_groups":groups.len(),
        "unique_source_and_constant_bytes":identical_sources.keys().map(String::len).sum::<usize>(),
        "identical_generated_source_groups":groups,"plans":records,
        "limitation":"No native execution, compile-size or timing claim; source equality does not replace semantic runtime selection."
    });
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    fs::write(&output, serde_json::to_string_pretty(&report)? + "\n")?;
    println!("{}", output.display());
    Ok(())
}
