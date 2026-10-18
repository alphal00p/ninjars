//! Generate the one-loop QCD gluon self-energy graphs (ghost, gluon, down- and
//! top-quark loops) used by the R2/Ward-identity fixtures.
use std::{error::Error, fs, path::PathBuf};

use ninjars_graph::{Kinematics, Model, prepare_gluon_self_energy};
use symbolica::atom::AtomCore;

fn main() -> Result<(), Box<dyn Error>> {
    let directory = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/physical/gluon-self-energy"));
    fs::create_dir_all(&directory)?;
    let model = Model::standard_model();
    for prepared in prepare_gluon_self_energy(model, &Kinematics::new(), &[1, 6])? {
        let loop_particle = prepared
            .diagram
            .edges()
            .filter(|(_, _, data)| data.external.is_none())
            .map(|(_, _, data)| prepared.diagram.model().particle_by_id(data.particle))
            .next()
            .ok_or("graph without loop edges")??
            .pdg_code;
        let name = match loop_particle.abs() {
            9000005 => "ghost-loop",
            21 => "gluon-loop",
            1 => "down-loop",
            6 => "top-loop",
            _ => return Err(format!("unexpected loop particle {loop_particle}").into()),
        };
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
