//! Rank n+2 performance guard (validation/performance/rank-n2-ab): layout
//! probe for the replayed pure reduction of one case at width 1.
//!
//! The fast engine is evaluated with case-bench's replayed numerator and
//! masters (the `reduction` metric), called through a frame padded by
//! 0, 16, ..., 112 bytes of stack, so that the engine's stack frame (and its
//! stack-resident C64 temporaries) sits at every offset modulo 64. Per
//! padding: median ns per call over 15 blocks of 20000 calls, plus user-space
//! cycles and instructions per call (perf_event_open).
//!
//! Copied into `crates/ninjars-symbolica/tests/` of both sides of the A/B and
//! run with `--ignored --exact stackprobe`; case file from $PROBE_CASE
//! (relative to the repository root, default
//! `validation/benchmark/synthetic/syn-n2-r2.json`), output TSV on $PROBE_OUT
//! (pad, ns per call, cycles per call, instructions per call).

use ninjars::fast::{
    C64, FastAmplitude, FastInput, FastOptions, FastResult, FastStatus, RMom, RecordingMasters,
    RecordingNumerator, TensorNumerator, tensor_layout::ncoeff_mu,
};
use ninjars_symbolica::OneLoopNativeMasters;
use serde_json::Value;
use std::hint::black_box;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

unsafe extern "C" {
    fn syscall(num: i64, ...) -> i64;
    fn ioctl(fd: i32, req: u64, ...) -> i32;
    fn read(fd: i32, buf: *mut u8, n: usize) -> isize;
}

struct Counter(i32);
impl Counter {
    fn new(config: u64) -> Counter {
        let mut attr = [0u64; 8];
        attr[0] = 64u64 << 32;
        attr[1] = config;
        attr[5] = 1 | (1 << 5) | (1 << 6);
        let fd = unsafe { syscall(298, attr.as_ptr(), 0i64, -1i64, -1i64, 0u64) };
        assert!(fd >= 0, "perf_event_open failed");
        Counter(fd as i32)
    }
    fn start(&self) {
        unsafe {
            ioctl(self.0, 0x2403, 0);
            ioctl(self.0, 0x2400, 0);
        }
    }
    fn stop(&self) -> u64 {
        let mut v = 0u64;
        unsafe {
            ioctl(self.0, 0x2401, 0);
            read(self.0, &mut v as *mut u64 as *mut u8, 8);
        }
        v
    }
}

#[inline(never)]
fn padded<const P: usize>(f: &mut dyn FnMut()) {
    let pad = [0u8; P];
    black_box(&pad);
    f();
    black_box(&pad);
}

fn with_pad(p: usize, f: &mut dyn FnMut()) {
    match p {
        0 => padded::<0>(f),
        16 => padded::<16>(f),
        32 => padded::<32>(f),
        48 => padded::<48>(f),
        64 => padded::<64>(f),
        80 => padded::<80>(f),
        96 => padded::<96>(f),
        112 => padded::<112>(f),
        _ => unreachable!(),
    }
}

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap()
}

#[test]
#[ignore]
fn stackprobe() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let case = std::env::var("PROBE_CASE").unwrap_or_else(|_| "validation/benchmark/synthetic/syn-n2-r2.json".into());
    let v: Value = serde_json::from_str(&std::fs::read_to_string(root.join(case)).unwrap()).unwrap();
    let n = v["n"].as_u64().unwrap() as usize;
    let rank = v["rank"].as_u64().unwrap() as usize;
    let lanes = v["lanes"].as_array().unwrap();
    let width = 1;
    let l = &lanes[0];
    let shifts: Vec<RMom> = l["shifts"].as_array().unwrap().iter().map(|p| [num(&p[0]), num(&p[1]), num(&p[2]), num(&p[3])]).collect();
    let masses: Vec<C64> = l["masses_squared"].as_array().unwrap().iter().map(|z| C64::new(num(&z[0]), num(&z[1]))).collect();
    let scales = vec![num(&l["mu_squared"])];
    let s: Option<Vec<f64>> = match &l["s_matrix"] {
        Value::Null => None,
        rows => Some(rows.as_array().unwrap().iter().flat_map(|r| r.as_array().unwrap().iter().map(num)).collect()),
    };
    let tensor: Vec<C64> = l["tensor"].as_array().unwrap().iter().map(|z| C64::new(num(&z[0]), num(&z[1]))).collect();
    let mut t = TensorNumerator::new(rank, width);
    t.coefficients_mut()[..ncoeff_mu(rank, 0)].copy_from_slice(&tensor);
    let input = FastInput { lanes: width, shifts: &shifts, masses_squared: &masses, scales_squared: &scales, s_matrix: s.as_deref() };
    let mut amp = FastAmplitude::new(n, rank, FastOptions::default()).unwrap();
    let mut results = vec![FastResult::default(); width];
    let mut statuses = vec![FastStatus::NotEvaluated; width];
    let mut rec = RecordingNumerator::new(t);
    let mut recm = RecordingMasters::new(OneLoopNativeMasters::new().unwrap());
    amp.evaluate(&input, &mut rec, &mut recm, &mut results, &mut statuses).unwrap();
    let mut rn = rec.replay();
    let mut rm = recm.replay();
    let ins = Counter::new(1);
    let cyc = Counter::new(0);
    let mut out = std::io::BufWriter::new(std::fs::File::create(std::env::var("PROBE_OUT").unwrap()).unwrap());
    const CALLS: usize = 20000;
    for round in 0..2 {
        for pad in (0..8).map(|k| 16 * k) {
            let mut body = || {
                rn.rewind();
                rm.rewind();
                amp.evaluate(&input, &mut rn, &mut rm, &mut results, &mut statuses).unwrap();
                black_box(&results);
            };
            for _ in 0..2000 {
                with_pad(pad, &mut body);
            }
            let mut ns = Vec::new();
            let (mut c, mut i) = (0u64, 0u64);
            for _ in 0..15 {
                cyc.start();
                ins.start();
                let start = Instant::now();
                for _ in 0..CALLS {
                    with_pad(pad, &mut body);
                }
                ns.push(start.elapsed().as_nanos() as f64 / CALLS as f64);
                i += ins.stop();
                c += cyc.stop();
            }
            ns.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if round == 1 {
                writeln!(out, "{pad}\t{:.1}\t{:.1}\t{:.1}", ns[7], c as f64 / (15 * CALLS) as f64, i as f64 / (15 * CALLS) as f64).unwrap();
            }
        }
    }
}
