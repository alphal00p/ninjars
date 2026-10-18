//! V0p: fingerprints of every exact Symbolica plan for numerator rank <= n+1.
//!
//! Rank n+2 support must not change any existing plan. For n <= 6 and every
//! rank <= n+1 this records the SHA-256 of `(expressions, parameters)`, as
//! Symbolica canonical strings (independent of symbol registration order), of
//! - the standalone plans: `LaurentDivision`, `ResidueProjection`,
//!   `TriangleBubbleSubtraction` (`for_quotient` and `new`),
//!   `TadpoleSubtraction`, the box `Plans` in every box mode,
//!   `MasterContraction`, `DenominatorExpansion`, `BubbleSourceProjection`,
//!   `TadpoleSourceProjection`, `TadpoleGeometryProbe` and the amplitude
//!   accumulation;
//! - every plan compiled while constructing each stage of
//!   `SymbolicaAmplitudeKernels` (all box modes, pruned swapped arithmetic,
//!   the n=5/r=4 rank specialization), the integration kernels and the cut,
//!   high-cut and sampling-scale providers, in compilation order.
//!
//! Expected values: `validation/regression/plan-fingerprints-e4c16a7.json`,
//! generated at e4c16a7 before any rank n+2 change. `NINJARS_BLESS=1`
//! rewrites the file.

use std::collections::BTreeMap;

use ninjars::{AmplitudeKernels, BoxCoefficientMode, BoxMuMode, BoxReductionOptions};

use crate::kernel::{COMPILED_PLANS, ExactKernel};
use crate::{
    BubbleSourceProjection, DenominatorExpansion, LaurentDivision, MasterContraction,
    ResidueProjection, SymbolicaAmplitudeKernels, SymbolicaCuts, SymbolicaHighCuts,
    SymbolicaIntegrationKernels, SymbolicaSamplingScales, TadpoleGeometryProbe,
    TadpoleSourceProjection, TadpoleSubtraction, TriangleBubbleSubtraction,
};

const FIXTURE: &str = "validation/regression/plan-fingerprints-e4c16a7.json";
const MAX_N: usize = 6;

/// FIPS 180-4 SHA-256 (no dependency for a test-only digest).
fn sha256(message: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut data = message.to_vec();
    data.push(0x80);
    while data.len() % 64 != 56 {
        data.push(0);
    }
    data.extend_from_slice(&((message.len() as u64) * 8).to_be_bytes());
    for block in data.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, word) in block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let mut v = h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7]
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v = [
                t1.wrapping_add(t2),
                v[0],
                v[1],
                v[2],
                v[3].wrapping_add(t1),
                v[4],
                v[5],
                v[6],
            ];
        }
        for (state, value) in h.iter_mut().zip(v) {
            *state = state.wrapping_add(value);
        }
    }
    h.iter().map(|word| format!("{word:08x}")).collect()
}

fn digest(kernel: &ExactKernel) -> String {
    sha256(kernel.canonical_text().as_bytes())
}

const BOX_MODES: [(BoxCoefficientMode, BoxMuMode); 4] = [
    (BoxCoefficientMode::Integrated, BoxMuMode::Expansion),
    (BoxCoefficientMode::Integrated, BoxMuMode::Sampling),
    (BoxCoefficientMode::Full, BoxMuMode::Expansion),
    (BoxCoefficientMode::Full, BoxMuMode::Sampling),
];

fn options((coefficients, mu_extraction): (BoxCoefficientMode, BoxMuMode)) -> BoxReductionOptions {
    BoxReductionOptions {
        coefficients,
        mu_extraction,
    }
}

#[derive(Default)]
struct Fingerprints(BTreeMap<String, String>);

impl Fingerprints {
    fn plan(&mut self, label: String, kernel: &ExactKernel) {
        self.0.insert(label, digest(kernel));
    }

    fn outcome<X>(&mut self, label: String, outcome: ninjars::Result<X>) -> Option<X> {
        match outcome {
            Ok(value) => Some(value),
            Err(error) => {
                self.0.insert(label, format!("error: {error}"));
                None
            }
        }
    }

    /// Every plan compiled by `build`, in order, as one digest of the list.
    fn compiled<X>(&mut self, label: String, build: impl FnOnce() -> ninjars::Result<X>) {
        COMPILED_PLANS.with_borrow_mut(|plans| *plans = Some(Vec::new()));
        let outcome = build();
        let texts = COMPILED_PLANS.with_borrow_mut(|plans| plans.take().unwrap());
        let list: String = texts
            .iter()
            .map(|text| sha256(text.as_bytes()) + "\n")
            .collect();
        let value = match outcome {
            Ok(_) => format!("{} plans {}", texts.len(), sha256(list.as_bytes())),
            Err(error) => format!("error after {} plans: {error}", texts.len()),
        };
        self.0.insert(label, value);
    }
}

fn standalone(prints: &mut Fingerprints) {
    for cut in 1..=3usize {
        for n in cut..=MAX_N {
            for rank in 0..=n + 1 {
                let label = format!("laurent n{n} r{rank} cut{cut}");
                let Some(division) =
                    prints.outcome(label.clone(), LaurentDivision::new(n, rank, cut))
                else {
                    continue;
                };
                prints.0.insert(
                    format!("{label} layout"),
                    format!(
                        "{:?} | {:?} | {:?} | higher {}",
                        division.numerator_expansion(),
                        division.denominator_expansion(),
                        division.quotient_expansion(),
                        division.higher_rank()
                    ),
                );
                prints.plan(label, &division.exact);
                if cut >= 2 {
                    let label = format!("projection n{n} r{rank} cut{cut}");
                    if let Some(projection) =
                        prints.outcome(label.clone(), ResidueProjection::new(&division))
                    {
                        prints.plan(label, &projection.exact);
                    }
                }
                if cut == 2 {
                    let label = format!("triangle-bubble for_quotient n{n} r{rank}");
                    if let Some(plan) = prints.outcome(
                        label.clone(),
                        TriangleBubbleSubtraction::for_quotient(&division),
                    ) {
                        prints.plan(label, &plan.exact);
                    }
                }
            }
        }
    }
    for higher in [false, true] {
        let label = format!("triangle-bubble new higher={higher}");
        if let Some(plan) = prints.outcome(label.clone(), TriangleBubbleSubtraction::new(higher)) {
            prints.plan(label, &plan.exact);
        }
        for source in [2, 3] {
            let label = format!("tadpole subtraction source{source} higher={higher}");
            if let Some(plan) =
                prints.outcome(label.clone(), TadpoleSubtraction::new(source, higher))
            {
                prints.plan(label, &plan.exact);
            }
        }
        for cut in 1..=4 {
            let label = format!("contraction cut{cut} higher={higher}");
            if let Some(plan) = prints.outcome(label.clone(), MasterContraction::new(cut, higher)) {
                prints
                    .0
                    .insert(format!("{label} masters"), format!("{:?}", plan.masters()));
                prints.plan(format!("{label} weights"), &plan.weights);
                prints.plan(format!("{label} contraction"), &plan.contraction);
            }
        }
    }
    for cut in 1..=3 {
        let label = format!("denominators cut{cut}");
        if let Some(plan) = prints.outcome(label.clone(), DenominatorExpansion::new(cut)) {
            prints.plan(label, &plan.exact);
        }
    }
    prints.plan(
        "bubble source projection".into(),
        &BubbleSourceProjection::new().exact,
    );
    for source in [2, 3] {
        let label = format!("tadpole source projection source{source}");
        if let Some(plan) = prints.outcome(label.clone(), TadpoleSourceProjection::new(source)) {
            prints.plan(label, &plan.exact);
        }
    }
    prints.plan(
        "tadpole geometry probe".into(),
        &TadpoleGeometryProbe::new().exact,
    );
    for n in 1..=MAX_N {
        for rank in 0..=n + 1 {
            let label = format!("accumulation n{n} r{rank}");
            if let Some(plan) =
                prints.outcome(label.clone(), crate::amplitude::accumulation(n, rank))
            {
                prints.plan(label, &plan);
            }
            if n < 4 {
                continue;
            }
            for mode in BOX_MODES {
                let label = format!("box n{n} r{rank} {mode:?}");
                let Some(plans) = prints.outcome(
                    label.clone(),
                    crate::box_kernels::Plans::new(n, rank, options(mode)),
                ) else {
                    continue;
                };
                for (name, kernel) in [
                    ("samples", &plans.samples),
                    ("denominators", &plans.denominators),
                    ("division", &plans.division),
                    ("mu_denominators", &plans.mu_denominators),
                    ("mu_division", &plans.mu_division),
                    ("projections", &plans.projections),
                    ("interpolation", &plans.interpolation),
                ] {
                    prints.plan(format!("{label} {name}"), kernel);
                }
            }
        }
    }
}

fn stages(prints: &mut Fingerprints) {
    prints.compiled("cuts".into(), || SymbolicaCuts::new(0f64));
    prints.compiled("high cuts".into(), || SymbolicaHighCuts::new(0f64));
    for n in 1..=MAX_N {
        prints.compiled(format!("sampling scales n{n}"), || {
            SymbolicaSamplingScales::new(n, 0f64)
        });
        for rank in 0..=n + 1 {
            let label = format!("stages n{n} r{rank}");
            let Some(kernels) =
                prints.outcome(label.clone(), SymbolicaAmplitudeKernels::new(n, rank, 0f64))
            else {
                continue;
            };
            for mode in BOX_MODES {
                let kernels = SymbolicaAmplitudeKernels::new(n, rank, 0f64)
                    .unwrap()
                    .with_box_options(options(mode));
                prints.compiled(format!("{label} boxes {mode:?}"), || kernels.box_kernels());
            }
            prints.compiled(format!("{label} pentagons"), || kernels.pentagon_kernels());
            prints.compiled(format!("{label} triangles"), || kernels.triangle_kernels());
            prints.compiled(format!("{label} bubbles"), || kernels.bubble_kernels());
            prints.compiled(format!("{label} tadpoles"), || kernels.tadpole_kernels());
            let pruned = SymbolicaAmplitudeKernels::new(n, rank, 0f64)
                .unwrap()
                .with_pruned_swapped_arithmetic();
            prints.compiled(format!("{label} bubbles pruned"), || {
                pruned.bubble_kernels()
            });
            if let Ok(specialized) = SymbolicaAmplitudeKernels::new(n, rank, 0f64)
                .unwrap()
                .with_rank_specialized_corrections()
            {
                prints.compiled(format!("{label} bubbles specialized"), || {
                    specialized.bubble_kernels()
                });
                prints.compiled(format!("{label} tadpoles specialized"), || {
                    specialized.tadpole_kernels()
                });
            }
        }
    }
    for cut in 1..=5 {
        for higher in [false, true] {
            prints.compiled(format!("integration cut{cut} higher={higher}"), || {
                SymbolicaIntegrationKernels::new(cut, higher, 0f64)
            });
        }
    }
}

#[test]
fn exact_plans_for_rank_up_to_n_plus_one_are_unchanged() {
    let started = std::time::Instant::now();
    let mut prints = Fingerprints::default();
    standalone(&mut prints);
    let standalone_seconds = started.elapsed().as_secs_f64();
    stages(&mut prints);
    eprintln!(
        "V0p {} fingerprints (standalone {standalone_seconds:.1}s, total {:.1}s)",
        prints.0.len(),
        started.elapsed().as_secs_f64()
    );
    let actual = serde_json::json!({
        "format": "ninjars-plan-fingerprints-v1",
        "baseline": "e4c16a7 (NinjaRS main before rank n+2)",
        "description": "SHA-256 of the canonical strings of (expressions, parameters) of every exact plan for n <= 6 and rank <= n+1; stage entries digest the ordered list of plans compiled while constructing the stage. Test: crates/ninjars-symbolica/src/plan_fingerprints_tests.rs.",
        "fingerprints": prints.0,
    });
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(FIXTURE);
    if std::env::var("NINJARS_BLESS").is_ok_and(|v| v == "1") {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut text = serde_json::to_string_pretty(&actual).unwrap();
        text.push('\n');
        std::fs::write(&path, text).unwrap();
        eprintln!("V0p blessed {}", path.display());
        return;
    }
    let expected: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("{}: {e} (bless with NINJARS_BLESS=1)", path.display())),
    )
    .unwrap();
    let expected = expected["fingerprints"].as_object().unwrap();
    let mut differences = Vec::new();
    for (key, value) in expected {
        match prints.0.get(key) {
            Some(actual) if actual == value.as_str().unwrap_or_default() => {}
            actual => differences.push(format!("{key}: expected {value}, got {actual:?}")),
        }
    }
    for key in prints.0.keys().filter(|k| !expected.contains_key(*k)) {
        differences.push(format!("{key}: not in the fixture"));
    }
    assert!(
        differences.is_empty(),
        "{} exact plans changed:\n{}",
        differences.len(),
        differences.join("\n")
    );
}

#[test]
fn sha256_matches_known_vectors() {
    assert_eq!(
        sha256(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        sha256(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(
        sha256(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );
}
