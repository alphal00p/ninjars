//! Shared, explicitly rounded phase-space inputs; exact checks use FeynKit/Symbolica.
use ninjars_graph::FourMomentum;
use serde_json::json;
use std::collections::HashMap;
use symbolica::{
    atom::{Atom, AtomCore},
    domains::float::{Complex, Float, RealLike},
    parse,
};

fn rounded(expression: &Atom) -> f64 {
    let value: Complex<Float> = expression
        .evaluate_with_prec(&HashMap::<Atom, Complex<Float>>::new(), 256)
        .unwrap();
    assert!(value.im.to_f64() == 0.);
    let value = value.re.to_f64();
    assert!(value.is_finite());
    value
}
fn main() {
    let model_exact = [
        parse!("173"),
        parse!("173"),
        parse!("125"),
        parse!("116639/10^10"),
        parse!("59/500"),
        parse!("132507/1000"),
        parse!("22797/250"),
    ];
    let model = model_exact.iter().map(rounded).collect::<Vec<_>>();
    let mut points = Vec::new();
    for energy in [150, 175, 200] {
        let e = Atom::num(energy);
        let p = parse!("sqrt(ninjars::E^2-125^2)")
            .replace(parse!("ninjars::E"))
            .with(e.clone());
        let z = Atom::Zero;
        let beam = &e * 3 / 2;
        let momenta = [
            FourMomentum::from_args(beam.clone(), z.clone(), z.clone(), beam.clone()),
            FourMomentum::from_args(beam.clone(), z.clone(), z.clone(), -&beam),
            FourMomentum::from_args(e.clone(), p.clone(), z.clone(), z.clone()),
            FourMomentum::from_args(e.clone(), -&p / 2, &p * parse!("sqrt(3)") / 2, z.clone()),
            FourMomentum::from_args(e.clone(), -&p / 2, -&p * parse!("sqrt(3)") / 2, z.clone()),
        ];
        let conservation = momenta[0].clone() + momenta[1].clone()
            - momenta[2].clone()
            - momenta[3].clone()
            - momenta[4].clone();
        assert!((&conservation).into_iter().all(|x| x.expand().is_zero()));
        for (index, momentum) in momenta.iter().enumerate() {
            let mass = if index < 2 {
                Atom::Zero
            } else {
                Atom::num(15625)
            };
            assert!((momentum.mass_squared() - mass).expand().is_zero());
        }
        points.push(json!({"id":format!("sqrt_s_{}",3*energy),"helicities":[1,1,0,0,0],"model":model,"scale_squared":10000,
            "momenta":momenta.iter().map(|p|p.into_iter().map(rounded).collect::<Vec<_>>()).collect::<Vec<_>>(),
            "exact_momenta":momenta.iter().map(|p|p.into_iter().map(AtomCore::to_canonical_string).collect::<Vec<_>>()).collect::<Vec<_>>() }));
    }
    println!("{}",serde_json::to_string_pretty(&json!({"schema_version":1,"points":points,
        "model_parameters":["MT","ymt","MH","Gf","aS","aEWM1","MZ"],"exact_model_inputs":model_exact.iter().map(AtomCore::to_canonical_string).collect::<Vec<_>>(),
        "construction":"Exact rational/radical momenta; shared FeynKit mass-square and conservation checks via Symbolica. Components evaluated at256bits then rounded once tobinary64, shared identically by both numerical preparations and MadLoop.",
        "widths":"WT=WH=0","performance_acceptance":false})).unwrap());
}
