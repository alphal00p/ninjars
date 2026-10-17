//! Time oneloopmaster's direct native Rust masters (`NativeEvaluator<f64>`,
//! feature `generated-evaluators`; no expression interpreter, no SymJIT) on
//! exactly the master-integral calls original Ninja makes through AvHOneLoop,
//! as recorded by `ninja_bench` in `validation/benchmark/ninja/<case>.masters.json`.
//!
//! Family mapping (OneLOop argument order, mu^2 last):
//!   box_cm          -> D0 [s21,s32,s43,s14,s31,s42, m1..m4, mu2]
//!   triangle_cm     -> C0 [s21,s32,s13, m1..m3, mu2]
//!   bubble_cm       -> B0 [s21, m1, m2, mu2]
//!   rank2_bubble_cm -> without feature `tensor`: B0 only (OneLOop's olo_b11
//!                      returns B11, B00, B1 and B0; NOT the same amount of work);
//!                      with `tensor`: NativeTensorEvaluator B11 = [B0, B1, B11]
//!                      (all three compared with the recorded b0, b1, b11)
//!   rank3_bubble_cm -> with `tensor` only: NativeTensorEvaluator B111
//!   tadpole_cm      -> A0 [m, mu2]
//!
//! usage: masters-native-bench OUT.json CASE.masters.json... [--block-ms 25] [--rounds 9]
//!        [--label NAME]
//! Pin externally with taskset. `build_native.py` builds it against a chosen
//! oneloopmaster / Symbolica revision and records the provenance.

use oneloop::{NativeEvaluator, ScalarIntegral};
#[cfg(feature = "tensor")]
use oneloop::{NativeTensorEvaluator, TensorIntegral};
use serde_json::{Value, json};
use std::time::Instant;
use symbolica::domains::float::Complex;

type C = Complex<f64>;

#[derive(Clone)]
struct Call {
    // 0 D0, 1 C0, 2 B0 (scalar bubble), 3 rank-2 bubble (B0 only, or B11 tensor with
    // feature `tensor`), 4 A0, 5 rank-3 bubble (B111 tensor, feature `tensor` only)
    family: usize,
    input: Vec<C>,
    // recorded OneLOop outputs, in the native evaluator's output order
    reference: Vec<C>,
}

#[cfg(not(feature = "tensor"))]
const FAMILY_NAMES: [&str; 6] = ["D0", "C0", "B0", "B0_of_rank2_bubble", "A0", "B111_unsupported"];
#[cfg(feature = "tensor")]
const FAMILY_NAMES: [&str; 6] = ["D0", "C0", "B0", "B11_tensor_of_rank2_bubble", "A0", "B111_tensor_of_rank3_bubble"];
const MAX_OUT: usize = 12;

fn c(v: &Value) -> C {
    C::new(v[0].as_f64().unwrap(), v[1].as_f64().unwrap())
}

fn real(x: f64) -> C {
    C::new(x, 0.0)
}

fn parse_call(v: &Value) -> Option<Call> {
    let kind = v["kind"].as_str().unwrap();
    let s: Vec<f64> = v["s"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
    let m: Vec<C> = v["m2"].as_array().unwrap().iter().map(c).collect();
    let mu2 = real(v["mu_squared"].as_f64().unwrap());
    let result: Vec<C> = v["result"].as_array().unwrap().iter().map(c).collect();
    let r3 = |o: usize| vec![result[o], result[o + 1], result[o + 2]];
    // ninja_bench records olo_b11 as [b11, b1, b0] and the rank-3 bubble as
    // [b111, b11, b1, b0]; the native tensors return [B0, B1, B11(, B111)].
    let rev = |blocks: &[usize]| blocks.iter().flat_map(|&o| r3(o)).collect::<Vec<C>>();
    match kind {
        "box_cm" => Some(Call {
            family: 0,
            input: vec![real(s[0]), real(s[1]), real(s[2]), real(s[3]), real(s[4]), real(s[5]), m[0], m[1], m[2], m[3], mu2],
            reference: r3(0),
        }),
        "triangle_cm" => Some(Call {
            family: 1,
            input: vec![real(s[0]), real(s[1]), real(s[2]), m[0], m[1], m[2], mu2],
            reference: r3(0),
        }),
        "bubble_cm" => Some(Call { family: 2, input: vec![real(s[0]), m[0], m[1], mu2], reference: r3(0) }),
        #[cfg(not(feature = "tensor"))]
        "rank2_bubble_cm" => Some(Call { family: 3, input: vec![real(s[0]), m[0], m[1], mu2], reference: r3(6) }),
        #[cfg(feature = "tensor")]
        "rank2_bubble_cm" => Some(Call { family: 3, input: vec![real(s[0]), m[0], m[1], mu2], reference: rev(&[6, 3, 0]) }),
        #[cfg(feature = "tensor")]
        "rank3_bubble_cm" => Some(Call { family: 5, input: vec![real(s[0]), m[0], m[1], mu2], reference: rev(&[9, 6, 3, 0]) }),
        "tadpole_cm" => Some(Call { family: 4, input: vec![m[0], mu2], reference: r3(0) }),
        _ => None,
    }
}

struct Evaluators {
    ev: Vec<NativeEvaluator<f64>>,
    #[cfg(feature = "tensor")]
    b11: NativeTensorEvaluator<f64>,
    #[cfg(feature = "tensor")]
    b111: NativeTensorEvaluator<f64>,
}

impl Evaluators {
    fn new() -> Self {
        let fam = [ScalarIntegral::D0, ScalarIntegral::C0, ScalarIntegral::B0, ScalarIntegral::B0, ScalarIntegral::A0];
        Self {
            ev: fam.iter().map(|&f| NativeEvaluator::<f64>::new(f).unwrap()).collect(),
            #[cfg(feature = "tensor")]
            b11: NativeTensorEvaluator::<f64>::new(TensorIntegral::B11).unwrap(),
            #[cfg(feature = "tensor")]
            b111: NativeTensorEvaluator::<f64>::new(TensorIntegral::B111).unwrap(),
        }
    }
    /// Evaluate one call into out[..call.reference.len()].
    #[inline]
    fn eval(&mut self, call: &Call, out: &mut [C; MAX_OUT]) {
        let k = call.reference.len();
        match call.family {
            #[cfg(feature = "tensor")]
            3 => self.b11.evaluate(&call.input, &mut out[..k]).unwrap(),
            #[cfg(feature = "tensor")]
            5 => self.b111.evaluate(&call.input, &mut out[..k]).unwrap(),
            f => self.ev[f].evaluate(&call.input, &mut out[..k]).unwrap(),
        }
    }
}

/// Equivalent relabellings of a C0 [p1,p2,p3, m1,m2,m3, mu2] or D0
/// [p1,p2,p3,p4,s12,s23, m1..m4, mu2] call: the cyclic shifts and the reflection.
fn symmetric_inputs(call: &Call) -> Vec<Vec<C>> {
    let x = &call.input;
    match call.family {
        1 => vec![
            vec![x[1], x[2], x[0], x[4], x[5], x[3], x[6]],
            vec![x[2], x[0], x[1], x[5], x[3], x[4], x[6]],
            vec![x[2], x[1], x[0], x[3], x[5], x[4], x[6]],
        ],
        0 => vec![
            vec![x[1], x[2], x[3], x[0], x[5], x[4], x[7], x[8], x[9], x[6], x[10]],
            vec![x[2], x[3], x[0], x[1], x[4], x[5], x[8], x[9], x[6], x[7], x[10]],
            vec![x[3], x[2], x[1], x[0], x[4], x[5], x[6], x[9], x[8], x[7], x[10]],
        ],
        _ => vec![],
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) }
}

/// ns per "point" where one point = one pass over `calls` of one lane, lanes cycled.
fn time(lanes: &[Vec<Call>], filter: &dyn Fn(&Call) -> bool, ev: &mut Evaluators, block_ms: f64, rounds: usize) -> Option<(f64, Vec<f64>, u64)> {
    let sel: Vec<Vec<Call>> = lanes.iter().map(|l| l.iter().filter(|c| filter(c)).cloned().collect()).collect();
    if sel.iter().all(|l| l.is_empty()) {
        return None;
    }
    let mut out = [C::new(0.0, 0.0); MAX_OUT];
    let mut sink = 0.0f64;
    let mut run = |iters: u64, ev: &mut Evaluators| -> f64 {
        let t0 = Instant::now();
        for _ in 0..iters {
            for lane in &sel {
                for call in lane {
                    ev.eval(call, &mut out);
                    sink += out[0].re;
                }
            }
        }
        t0.elapsed().as_nanos() as f64
    };
    let mut iters = 1u64;
    loop {
        let t = run(iters, ev);
        if t >= 0.2 * block_ms * 1e6 {
            iters = ((block_ms * 1e6) / (t / iters as f64)).ceil().max(1.0) as u64;
            break;
        }
        iters *= 4;
    }
    run(iters, ev); // warm-up block
    let points = iters as f64 * sel.len() as f64;
    let blocks: Vec<f64> = (0..rounds).map(|_| run(iters, ev) / points).collect();
    std::hint::black_box(sink);
    Some((median(blocks.clone()), blocks, iters))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut block_ms = 25.0;
    let mut rounds = 9usize;
    let mut label = String::from("unlabelled");
    let mut files = vec![];
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--block-ms" => { block_ms = args[i + 1].parse().unwrap(); i += 1; }
            "--rounds" => { rounds = args[i + 1].parse().unwrap(); i += 1; }
            "--label" => { label = args[i + 1].clone(); i += 1; }
            f => files.push(f.to_string()),
        }
        i += 1;
    }
    // Same setup as oneloopmaster's performance_survey: register the numerical
    // callbacks once before constructing evaluators.
    oneloop::initialize().unwrap();
    if std::env::var("DILOG_TEST").is_ok() {
        // Cost of the f64 dilogarithm used by the native generated code
        // (Symbolica's registered polylog(2, z) callback).
        use oneloop::NativeFloat;
        for (re, im) in [(0.1, 0.0), (0.5, 0.0), (0.9, 0.0), (0.99, 0.0), (-0.5, 0.0), (-3.0, 0.0), (2.0, 0.0),
                         (0.5, 0.5), (0.9, 0.3), (1.0, 1e-10), (-10.0, 1.0), (0.3, -0.8)] {
            let z = C::new(re, im);
            let reps = 200;
            let t0 = Instant::now();
            let mut acc = C::new(0.0, 0.0);
            for _ in 0..reps { acc = acc + <f64 as NativeFloat>::dilog(&std::hint::black_box(z)); }
            let ns = t0.elapsed().as_nanos() as f64 / reps as f64;
            eprintln!("dilog({re}, {im}) = {:?}: {ns:.0} ns", <f64 as NativeFloat>::dilog(&z));
            std::hint::black_box(acc);
        }
        return;
    }
    let mut ev = Evaluators::new();
    let mut reports = vec![];
    for file in &files {
        let data: Value = serde_json::from_str(&std::fs::read_to_string(file).unwrap()).unwrap();
        let mut skipped = 0usize;
        let lanes: Vec<Vec<Call>> = data["lanes"].as_array().unwrap().iter().map(|lane| {
            lane["calls"].as_array().unwrap().iter().filter_map(|v| {
                let call = parse_call(v);
                if call.is_none() { skipped += 1; }
                call
            }).collect()
        }).collect();
        // correctness vs the recorded OneLOop results
        let mut max_diff = [0.0f64; 6];
        let mut worst: Vec<Value> = vec![Value::Null; 6];
        let mut counts = [0usize; 6];
        let mut max_sym = [0.0f64; 6];
        let mut worst_sym: Vec<Value> = vec![Value::Null; 6];
        let mut out = [C::new(0.0, 0.0); MAX_OUT];
        for lane in &lanes {
            for call in lane {
                ev.eval(call, &mut out);
                let scale = call.reference.iter().map(|z| z.re.hypot(z.im)).fold(0.0, f64::max).max(1e-300);
                for k in 0..call.reference.len() {
                    let d = (out[k].re - call.reference[k].re).hypot(out[k].im - call.reference[k].im) / scale;
                    if d > max_diff[call.family] {
                        max_diff[call.family] = d;
                        worst[call.family] = json!({
                            "input": call.input.iter().map(|z| [z.re, z.im]).collect::<Vec<_>>(),
                            "native": out[..call.reference.len()].iter().map(|z| [z.re, z.im]).collect::<Vec<_>>(),
                            "oneloop": call.reference.iter().map(|z| [z.re, z.im]).collect::<Vec<_>>(),
                        });
                    }
                }
                counts[call.family] += 1;
                // Native self-consistency: C0/D0 are invariant under the cyclic
                // and reflected relabellings of the propagators.
                for perm in symmetric_inputs(call) {
                    let mut alt = [C::new(0.0, 0.0); MAX_OUT];
                    ev.eval(&Call { family: call.family, input: perm.clone(), reference: call.reference.clone() }, &mut alt);
                    let scale = out[..3].iter().map(|z| z.re.hypot(z.im)).fold(0.0, f64::max).max(1e-300);
                    for k in 0..3 {
                        let d = (alt[k].re - out[k].re).hypot(alt[k].im - out[k].im) / scale;
                        if d > max_sym[call.family] {
                            max_sym[call.family] = d;
                            worst_sym[call.family] = json!({
                                "input": call.input.iter().map(|z| [z.re, z.im]).collect::<Vec<_>>(),
                                "native": out[..3].iter().map(|z| [z.re, z.im]).collect::<Vec<_>>(),
                                "relabelled_input": perm.iter().map(|z| [z.re, z.im]).collect::<Vec<_>>(),
                                "native_relabelled": alt[..3].iter().map(|z| [z.re, z.im]).collect::<Vec<_>>(),
                            });
                        }
                    }
                }
            }
        }
        let mut per_call = vec![];
        if std::env::var("PER_CALL").is_ok() {
            for call in &lanes[0] {
                let reps = 20;
                let t0 = Instant::now();
                for _ in 0..reps { ev.eval(call, &mut out); }
                let ns = t0.elapsed().as_nanos() as f64 / reps as f64;
                eprintln!("{} {:>12.0} ns  {:?}", FAMILY_NAMES[call.family], ns, call.input.iter().map(|z| (z.re, z.im)).collect::<Vec<_>>());
                per_call.push(json!({"family": FAMILY_NAMES[call.family], "ns": ns}));
            }
        }
        let mut timing = serde_json::Map::new();
        let all = time(&lanes, &|_| true, &mut ev, block_ms, rounds);
        if let Some((m, b, it)) = all {
            timing.insert("all_in_call_order".into(), json!({"median_ns": m, "blocks_ns": b, "iterations_per_block": it}));
        }
        #[cfg(not(feature = "tensor"))]
        const BUBBLES: &str = "bubbles_b0_only";
        #[cfg(feature = "tensor")]
        const BUBBLES: &str = "bubbles";
        let groups: [(&str, &dyn Fn(&Call) -> bool); 4] = [
            ("boxes", &|c: &Call| c.family == 0),
            ("triangles", &|c: &Call| c.family == 1),
            (BUBBLES, &|c: &Call| c.family == 2 || c.family == 3 || c.family == 5),
            ("tadpoles", &|c: &Call| c.family == 4),
        ];
        for (name, f) in groups.iter() {
            if let Some((m, b, it)) = time(&lanes, *f, &mut ev, block_ms, rounds) {
                timing.insert((*name).into(), json!({"median_ns": m, "blocks_ns": b, "iterations_per_block": it}));
            }
        }
        let report = json!({
            "case": data["case"],
            "masters_file": file,
            "calls_per_family_all_lanes": FAMILY_NAMES.iter().zip(counts.iter()).map(|(n, c)| (n.to_string(), json!(c))).collect::<serde_json::Map<_, _>>(),
            "max_scaled_difference_vs_oneloop": FAMILY_NAMES.iter().zip(max_diff.iter()).map(|(n, d)| (n.to_string(), json!(d))).collect::<serde_json::Map<_, _>>(),
            "worst_call_per_family": FAMILY_NAMES.iter().zip(worst.iter()).filter(|(_, w)| !w.is_null()).map(|(n, w)| (n.to_string(), w.clone())).collect::<serde_json::Map<_, _>>(),
            "max_scaled_relabelling_deviation_native": FAMILY_NAMES.iter().zip(max_sym.iter()).take(2).map(|(n, d)| (n.to_string(), json!(d))).collect::<serde_json::Map<_, _>>(),
            "worst_relabelling_per_family": FAMILY_NAMES.iter().zip(worst_sym.iter()).filter(|(_, w)| !w.is_null()).map(|(n, w)| (n.to_string(), w.clone())).collect::<serde_json::Map<_, _>>(),
            "skipped_calls": skipped,
            "timing_unit": "ns per phase-space point (one lane's calls), lanes cycled, median of blocks",
            "timing": timing,
        });
        eprintln!("{}: {}", data["case"], report["timing"]["all_in_call_order"]["median_ns"]);
        reports.push(report);
    }
    let out = json!({
        "format": "ninjars-reference-native-masters-v1",
        "label": label,
        "tensor_bubbles": cfg!(feature = "tensor"),
        "backend": if cfg!(feature = "tensor") {
            "oneloopmaster NativeEvaluator<f64> (generated-evaluators: direct ahead-of-time Rust, no interpreter/JIT) for D0/C0/B0/A0 and NativeTensorEvaluator<f64> B11/B111 for Ninja's rank-2/3 bubble calls, scalar evaluate per call"
        } else {
            "oneloopmaster NativeEvaluator<f64> (generated-evaluators: direct ahead-of-time Rust, no interpreter/JIT), scalar evaluate per call"
        },
        "block_ms": block_ms, "rounds": rounds,
        "cases": reports,
    });
    std::fs::write(&args[1], serde_json::to_string_pretty(&out).unwrap()).unwrap();
}
