//! Export existing exact kernel graphs for the independent native probe.
use std::{error::Error, fmt::Write, fs, path::PathBuf};

use ninjars::Complex;
use ninjars_symbolica::{
    DenominatorExpansion, LaurentDivision, TriangleBubbleSubtraction,
    rust_codegen::{EmitOptions, RustEmitter, RustKernelExport},
};
use serde_json::{Value, json};
use symbolica::domains::{dual::HyperDual, rational::Rational};

fn export<K: RustKernelExport>(
    plan: &K,
    name: &str,
    directory: &std::path::Path,
    emitter: &mut RustEmitter,
    records: &mut Vec<Value>,
) -> Result<(), Box<dyn Error>> {
    let structure = HyperDual::<Complex<Rational>>::new(vec![
        vec![0, 0],
        vec![1, 0],
        vec![0, 1],
        vec![2, 0],
        vec![1, 1],
    ]);
    for dual in [false, true] {
        let generated = if dual {
            plan.export_rust_dual(&structure, emitter, EmitOptions::default())?
        } else {
            plan.export_rust(emitter, EmitOptions::default())?
        };
        let module = format!("{name}{}", if dual { "_dual" } else { "" });
        let provenance = format!("// NinjaRS existing exact {name} plan; flattened Dual={dual}.\n");
        fs::write(
            directory.join(format!("{module}.rs")),
            provenance + generated.source.as_str(),
        )?;
        records.push(
            json!({"module":module,"dual":dual,"inputs":generated.input_count,
            "outputs":generated.output_count,"instructions":generated.instruction_count,
            "helpers":generated.helper_count,"outlined_branches":generated.outlined_branch_count}),
        );
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let directory = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| "target/rust-kernel-probe/src".into());
    fs::create_dir_all(&directory)?;
    let directory = fs::canonicalize(directory)?;
    let mut emitter = RustEmitter::default();
    let mut records = Vec::new();
    export(
        &LaurentDivision::new(4, 5, 3)?,
        "laurent",
        &directory,
        &mut emitter,
        &mut records,
    )?;
    export(
        &TriangleBubbleSubtraction::new(true)?,
        "subtraction",
        &directory,
        &mut emitter,
        &mut records,
    )?;
    export(
        &DenominatorExpansion::new(2)?,
        "denominator",
        &directory,
        &mut emitter,
        &mut records,
    )?;
    fs::write(directory.join("constants.rs"), emitter.constants_source())?;
    let mut dispatch = String::from("// Mechanical dispatch for validation only.\n");
    for module in records
        .iter()
        .map(|record| record["module"].as_str().unwrap())
        .chain(["constants"])
    {
        writeln!(
            dispatch,
            "#[path = {:?}] mod {module};",
            directory.join(format!("{module}.rs"))
        )?;
    }
    let layouts: Vec<_> = records
        .iter()
        .step_by(2)
        .map(|record| {
            (
                record["inputs"].as_u64().unwrap(),
                record["outputs"].as_u64().unwrap(),
            )
        })
        .collect();
    writeln!(dispatch, "const LAYOUTS: [(usize,usize);3] = {layouts:?};")?;
    dispatch.push_str("fn generated<T: NativeFloat>(kernel:usize,dual:bool,input:&[C<T>],ctx:&Context<T>,output:&mut[C<T>]) {\nmatch (kernel,dual) {\n");
    for (index, record) in records.iter().enumerate() {
        writeln!(
            dispatch,
            "({}, {}) => output.clone_from_slice(&{}::evaluate(input,ctx)),",
            index / 2,
            record["dual"],
            record["module"].as_str().unwrap()
        )?;
    }
    dispatch.push_str("_ => unreachable!(),\n}\n}\n");
    fs::write(directory.join("dispatch.rs"), dispatch)?;
    fs::write(
        directory.join("metadata.json"),
        serde_json::to_string_pretty(&json!({
            "kernels":records,"constant_count":emitter.constants().len(),
            "plans": {
                "laurent": "LaurentDivision::new(4, 5, 3)",
                "subtraction": "TriangleBubbleSubtraction::new(true)",
                "denominator": "DenominatorExpansion::new(2)"
            },
            "dual_shape":[[0,0],[1,0],[0,1],[2,0],[1,1]],
            "execution":"compiled scalar rows; validation control only; not pipeline batching"
        }))?,
    )?;
    println!("{}", directory.display());
    Ok(())
}
