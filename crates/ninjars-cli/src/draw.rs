//! Draw FeynKit DOT graphs as Typst documents with Linnest.
//!
//! Each input graph is imported with FeynKit and written as a standalone Typst
//! document (`FeynmanDiagram::to_linnest`) that lays the graph out and draws it
//! with GammaLoop's Linnest package and shared physics edge styles (particle
//! line patterns, fermion-flow arrows, external-leg placement). The document
//! imports `crates/linnest/typst`, `crates/kurvst/typst` and
//! `assets/embedded/drawing/templates` from the Typst project root (`--root`);
//! `docs/benchmark/linnet` vendors exactly these trees.

use std::{collections::BTreeSet, error::Error, fs, path::PathBuf, sync::Arc};

use clap::Args as ClapArgs;
use feynkit_graph::FeynmanDiagram;

use crate::numerator_bench::model;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(ClapArgs)]
pub struct Args {
    /// FeynKit DOT graphs to draw.
    pub graphs: Vec<PathBuf>,
    /// Model: `sm`, `phi3`, `phi4` or a model JSON file.
    #[arg(long, default_value = "sm")]
    pub model: String,
    /// Directory receiving one `<graph stem>.typ` document per input.
    #[arg(short, long)]
    pub output_dir: PathBuf,
    /// Linnest render configuration (a Typst dictionary expression).
    #[arg(long, default_value = "(:)")]
    pub config: String,
}

pub fn run(args: Args) -> Result<()> {
    let model = Arc::new(model(&args.model, std::path::Path::new("."))?);
    fs::create_dir_all(&args.output_dir)?;
    for path in &args.graphs {
        let dot = fs::read_to_string(path)?;
        let diagram = FeynmanDiagram::from_dot(model.clone(), &dot)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        // FeynKit emits imports relative to the Typst project root; make them
        // root-absolute so documents can live in any subdirectory.
        let typst = diagram
            .to_linnest(None, &BTreeSet::new(), None, &args.config)
            .map_err(|error| format!("{}: {error}", path.display()))?
            .replace("#import \"crates/", "#import \"/crates/")
            .replace("#import \"assets/", "#import \"/assets/");
        let stem = path
            .file_stem()
            .ok_or_else(|| format!("{}: no file name", path.display()))?;
        let output = args.output_dir.join(stem).with_extension("typ");
        fs::write(&output, typst)?;
        println!("{}", output.display());
    }
    Ok(())
}
