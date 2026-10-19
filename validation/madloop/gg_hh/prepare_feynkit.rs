//! Validation fixture generator; all graph/routing/algebra uses shared APIs.
use feynkit_amplitude::{Amplitude, ColorSum};
use feynkit_generator::{GenerationFilter, GenerationOptions, NumeratorGrouping, Process};
use ninjars_graph::{Kinematics, Model, NumeratorMode, prepare_diagram};
use serde_json::json;
use std::{error::Error, fs, path::PathBuf, sync::Arc};
use symbolica::atom::AtomCore;

fn main() -> Result<(), Box<dyn Error>> {
    let output = PathBuf::from(std::env::args_os().nth(1).expect("output directory"));
    fs::create_dir_all(&output)?;
    let model = Arc::new(Model::standard_model());
    let kinematics = Kinematics::new();
    let top = model.particle_id_by_pdg(6)?;
    let antitop = model.particle_id_by_pdg(-6)?;
    let higgs = model.particle_id_by_pdg(25)?;
    let gluon = model.particle_id_by_pdg(21)?;
    let mut vertices = Vec::new();
    for mut particles in [
        vec![top, antitop, gluon],
        vec![top, antitop, higgs],
        vec![higgs; 3],
    ] {
        particles.sort();
        let selected = model
            .vertex_rules()
            .iter()
            .filter(|v| {
                let mut actual = v.particles.clone();
                actual.sort();
                actual == particles
            })
            .map(|v| v.name.clone().into())
            .collect::<Vec<_>>();
        assert_eq!(selected.len(), 1, "unique ttg, ttH and HHH vertices");
        vertices.extend(selected);
    }
    let options = GenerationOptions::default()
        .with_loop_count(1, 1)?
        .threads(1)
        .max_vertices(4)
        .numerator_grouping(NumeratorGrouping::None)
        .with_graph_filter(GenerationFilter::MaxNumberOfBridges(1))
        .with_graph_filter(GenerationFilter::VertexAllow(vertices));
    let g = model.particle_by_id(gluon)?.name.clone();
    let h = model.particle_by_id(higgs)?.name.clone();
    let generated =
        Process::new([g.clone(), g], [h.clone(), h]).generate_diagrams(model.clone(), &options)?;
    assert_eq!(
        generated.diagrams.len(),
        8,
        "six boxes and two Higgs-channel triangles"
    );
    let color = ColorSum::new(model.particle_by_id(gluon)?)?.averaged(true);
    let mut graphs = Vec::new();
    let mut triangles = 0;
    let mut boxes = 0;
    for (index, diagram) in generated.diagrams.into_iter().enumerate() {
        let amplitude = Amplitude::from_diagram(diagram.clone())?;
        let gluons = amplitude
            .legs()
            .iter()
            .filter(|l| l.particle == gluon)
            .collect::<Vec<_>>();
        assert_eq!(gluons.len(), 2);
        let projector = feynkit_graph::NumeratorInterface::admit(
            feynkit_graph::NumeratorInterface::export(diagram.projector()).0
                * color.expression([gluons[0].index_atom(), gluons[1].index_atom()]),
            None,
        )?;
        let prepared = prepare_diagram(
            diagram.with_projector(projector),
            &kinematics,
            NumeratorMode::Explicit,
        )?
        .with_color_dimension_invariants()?;
        let topology = match prepared.quadratics.len() {
            3 => {
                triangles += 1;
                "triangle"
            }
            4 => {
                boxes += 1;
                "box"
            }
            _ => panic!("unexpected loop topology"),
        };
        let name = format!("top-{topology}-{index}");
        let dot = output.join(format!("{name}.dot"));
        if dot.exists() {
            return Err(format!("refusing to replace {}", dot.display()).into());
        }
        fs::write(dot, prepared.diagram.to_dot()?)?;
        fs::write(
            output.join(format!("{name}.numerator.txt")),
            prepared.numerator.to_canonical_string(),
        )?;
        fs::write(
            output.join(format!("{name}.prefactor.txt")),
            prepared.overall_factor.to_canonical_string(),
        )?;
        let edges = prepared.diagram.edges().map(|(id, ends, data)| {
            let particle = model.particle_by_id(data.particle).unwrap();
            json!({"id":id.0, "source":ends.source.map(|v|v.0), "target":ends.target.map(|v|v.0),
                   "pdg":particle.pdg_code, "external":data.external.as_ref().map(|e|json!({"index":e.index,"state":format!("{:?}",e.state)})),
                   "routing":prepared.diagram.loop_momentum_basis().route_expression(&ninjars_graph::symbols::momentum().call(id.0)).to_canonical_string()})
        }).collect::<Vec<_>>();
        graphs.push(json!({"index":index,"name":name,"topology":topology,
            "loop_propagator_edges":prepared.propagator_edges.iter().map(|e|e.0).collect::<Vec<_>>(),
            "denominators":prepared.family.denominators().iter().map(AtomCore::to_canonical_string).collect::<Vec<_>>(),
            "overall_factor":prepared.overall_factor.to_canonical_string(),"edges":edges}));
        eprintln!("prepared {name}");
    }
    assert_eq!((triangles, boxes), (2, 6));
    let result = json!({"schema_version":1,"process":"g g -> H H","graphs":graphs,
        "generation":"FeynKit one-loop ttg/ttH/HHH-only vertex allowlist; no numerator grouping; at most one bridge",
        "color":"delta_ab/8 projector: coefficient of delta_ab, not an initial-state color average of a squared amplitude",
        "convention":"four-dimensional model rules; fermion-loop sign and color projector once; no R2/UV; no spin/helicity average or final-state symmetry factor",
        "triangle_bridge":"existing prepare_diagram moves the loop-independent Higgs denominator into overall_factor"});
    fs::write(
        output.join("feynkit-graphs.json"),
        serde_json::to_string_pretty(&result)? + "\n",
    )?;
    Ok(())
}
