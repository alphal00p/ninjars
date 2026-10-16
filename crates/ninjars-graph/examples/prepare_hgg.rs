//! Generate physical graph sources and their color-projected scalar numerators.
use std::{error::Error, fs, path::PathBuf};

use ninjars_graph::{Kinematics, Model, prepare_top_higgs_gluons};
use symbolica::atom::AtomCore;

fn main() -> Result<(), Box<dyn Error>> {
    let directory = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/physical/hgg"));
    fs::create_dir_all(&directory)?;
    let model = Model::standard_model();
    fs::write(directory.join("model.json"), model.to_json_pretty()?)?;
    for (orientation, prepared) in prepare_top_higgs_gluons(model, &Kinematics::new())?
        .into_iter()
        .enumerate()
    {
        let name = format!("top-triangle-{orientation}");
        fs::write(
            directory.join(format!("{name}.dot")),
            prepared.diagram.to_dot()?,
        )?;
        fs::write(
            directory.join(format!("{name}.numerator.txt")),
            prepared.numerator.to_canonical_string(),
        )?;
        fs::write(
            directory.join(format!("{name}.prefactor.txt")),
            prepared.overall_factor.to_canonical_string(),
        )?;
    }
    println!("{}", directory.display());
    Ok(())
}
