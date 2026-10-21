//! Rank n+2 performance guard (validation/performance/rank-n2-ab): user-space
//! instructions and cycles per point of the fast engine, measured in-process
//! with perf_event_open, on case-bench's two paths:
//! - the replayed pure reduction (recorded numerator and master values), and
//! - the live path, TensorNumerator (or MuTensorNumerator) + OneLoopNativeMasters.
//!
//! The same file is copied into `crates/ninjars-symbolica/tests/` of both
//! sides of the A/B and run with `--ignored --exact icount_split`. Cases: every
//! `ninjars-bench-case-v1/v2` file of the directories in $ICOUNT_DIRS
//! (colon-separated, relative to the repository root; default
//! `validation/benchmark/cases:validation/benchmark/synthetic`), or the files
//! listed in $ICOUNT_FILES. Widths: $ICOUNT_WIDTHS (default `1,32`).
//! Output: TSV on $ICOUNT_OUT with columns case, width, replay instructions,
//! replay cycles, live instructions, live cycles, then (split probe) the
//! instructions and cycles of the replayed numerator with live native masters
//! (reduction + masters), all per point.

use ninjars::fast::{
    C64, FastAmplitude, FastInput, FastNumerator, FastOptions, FastResult, FastStatus,
    MuTensorNumerator, RMom, RecordingMasters, RecordingNumerator, TensorNumerator,
    tensor_layout::ncoeff_mu,
};
use ninjars_symbolica::OneLoopNativeMasters;
use serde_json::Value;
use std::io::Write;
use std::path::PathBuf;

unsafe extern "C" {
    fn syscall(num: i64, ...) -> i64;
    fn ioctl(fd: i32, req: u64, ...) -> i32;
    fn read(fd: i32, buf: *mut u8, n: usize) -> isize;
}

struct Counter(i32);
impl Counter {
    fn new(config: u64) -> Counter {
        let mut attr = [0u64; 8];
        attr[0] = 64u64 << 32; // type 0 (hardware), size 64
        attr[1] = config;
        attr[5] = 1 | (1 << 5) | (1 << 6); // disabled, exclude_kernel, exclude_hv
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

struct Lane {
    shifts: Vec<RMom>,
    masses: Vec<C64>,
    mu2: f64,
    s: Option<Vec<f64>>,
    tensor: Vec<C64>,
}

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap()
}

fn measure<N: FastNumerator>(
    name: &str,
    n: usize,
    rank: usize,
    width: usize,
    lanes: &[Lane],
    mut numerator: N,
    reps: usize,
    out: &mut impl Write,
    ins: &Counter,
    cyc: &Counter,
) {
    let idx: Vec<usize> = (0..width).map(|i| i % lanes.len()).collect();
    let shifts: Vec<RMom> = idx.iter().flat_map(|&i| lanes[i].shifts.iter().copied()).collect();
    let masses: Vec<C64> = idx.iter().flat_map(|&i| lanes[i].masses.iter().copied()).collect();
    let scales: Vec<f64> = idx.iter().map(|&i| lanes[i].mu2).collect();
    let s: Option<Vec<f64>> = idx.iter().map(|&i| lanes[i].s.clone()).collect::<Option<Vec<_>>>().map(|r| r.concat());
    let input = FastInput { lanes: width, shifts: &shifts, masses_squared: &masses, scales_squared: &scales, s_matrix: s.as_deref() };
    let mut amp = FastAmplitude::new(n, rank, FastOptions::default()).unwrap();
    let mut results = vec![FastResult::default(); width];
    let mut statuses = vec![FastStatus::NotEvaluated; width];
    let _ = &mut numerator;
    let mut rec = RecordingNumerator::new(numerator);
    let mut recm = RecordingMasters::new(OneLoopNativeMasters::new().unwrap());
    amp.evaluate(&input, &mut rec, &mut recm, &mut results, &mut statuses).unwrap();
    let mut rn = rec.replay();
    let mut rm = recm.replay();
    // replayed pure reduction
    for _ in 0..3 {
        rn.rewind();
        rm.rewind();
        amp.evaluate(&input, &mut rn, &mut rm, &mut results, &mut statuses).unwrap();
    }
    ins.start();
    cyc.start();
    for _ in 0..reps {
        rn.rewind();
        rm.rewind();
        amp.evaluate(&input, &mut rn, &mut rm, &mut results, &mut statuses).unwrap();
    }
    let c = cyc.stop();
    let i = ins.stop();
    let pts = (reps * width) as f64;
    // live path: TensorNumerator (inner of the recorder) + native masters
    let mut inner = rec.into_inner();
    let mut masters = recm.into_inner();
    // split probe: replayed numerator + live native masters (reduction + masters)
    let reps3 = (reps / 4).max(2);
    for _ in 0..3 {
        rn.rewind();
        amp.evaluate(&input, &mut rn, &mut masters, &mut results, &mut statuses).unwrap();
    }
    ins.start();
    cyc.start();
    for _ in 0..reps3 {
        rn.rewind();
        amp.evaluate(&input, &mut rn, &mut masters, &mut results, &mut statuses).unwrap();
    }
    let c3 = cyc.stop();
    let i3 = ins.stop();
    let pts3 = (reps3 * width) as f64;
    for _ in 0..3 {
        amp.evaluate(&input, &mut inner, &mut masters, &mut results, &mut statuses).unwrap();
    }
    let reps2 = (reps / 4).max(2);
    ins.start();
    cyc.start();
    for _ in 0..reps2 {
        amp.evaluate(&input, &mut inner, &mut masters, &mut results, &mut statuses).unwrap();
    }
    let c2 = cyc.stop();
    let i2 = ins.stop();
    let pts2 = (reps2 * width) as f64;
    writeln!(out, "{name}\t{width}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}", i as f64 / pts, c as f64 / pts, i2 as f64 / pts2, c2 as f64 / pts2, i3 as f64 / pts3, c3 as f64 / pts3).unwrap();
}

#[test]
#[ignore]
fn icount_split() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut out = std::io::BufWriter::new(std::fs::File::create(std::env::var("ICOUNT_OUT").unwrap()).unwrap());
    let ins = Counter::new(1);
    let cyc = Counter::new(0);
    let mut files = Vec::new();
    if let Ok(list) = std::env::var("ICOUNT_FILES") {
        files.extend(list.split(':').filter(|s| !s.is_empty()).map(|s| root.join(s)));
    } else {
        let dirs = std::env::var("ICOUNT_DIRS")
            .unwrap_or_else(|_| "validation/benchmark/cases:validation/benchmark/synthetic".into());
        for d in dirs.split(':').filter(|s| !s.is_empty()) {
            let mut f: Vec<PathBuf> = std::fs::read_dir(root.join(d)).unwrap().map(|e| e.unwrap().path())
                .filter(|p| p.extension().is_some_and(|e| e == "json")).collect();
            f.sort();
            files.extend(f);
        }
    }
    let widths: Vec<usize> = std::env::var("ICOUNT_WIDTHS").unwrap_or_else(|_| "1,32".into())
        .split(',').map(|w| w.trim().parse().unwrap()).collect();
    for f in files {
        let v: Value = serde_json::from_str(&std::fs::read_to_string(&f).unwrap()).unwrap();
        let mu = match v["format"].as_str().unwrap() {
            "ninjars-bench-case-v2" => v["mu_powers"].as_u64().unwrap() as usize,
            _ => 0,
        };
        let n = v["n"].as_u64().unwrap() as usize;
        let rank = v["rank"].as_u64().unwrap() as usize;
        let lanes: Vec<Lane> = v["lanes"].as_array().unwrap().iter().map(|l| Lane {
            shifts: l["shifts"].as_array().unwrap().iter().map(|p| [num(&p[0]), num(&p[1]), num(&p[2]), num(&p[3])]).collect(),
            masses: l["masses_squared"].as_array().unwrap().iter().map(|z| C64::new(num(&z[0]), num(&z[1]))).collect(),
            mu2: num(&l["mu_squared"]),
            s: match &l["s_matrix"] { Value::Null => None, rows => Some(rows.as_array().unwrap().iter().flat_map(|r| r.as_array().unwrap().iter().map(num)).collect()) },
            tensor: l["tensor"].as_array().unwrap().iter().map(|z| C64::new(num(&z[0]), num(&z[1]))).collect(),
        }).collect();
        let name = v["name"].as_str().unwrap().to_string();
        let size = ncoeff_mu(rank, mu);
        for &width in &widths {
            let reps = (400 / width).max(24);
            let idx: Vec<usize> = (0..width).map(|i| i % lanes.len()).collect();
            if mu == 0 {
                let mut t = TensorNumerator::new(rank, width);
                for (l, &i) in idx.iter().enumerate() {
                    t.coefficients_mut()[l * size..(l + 1) * size].copy_from_slice(&lanes[i].tensor);
                }
                measure(&name, n, rank, width, &lanes, t, reps, &mut out, &ins, &cyc);
            } else {
                let mut t = MuTensorNumerator::new(rank, mu, width).unwrap();
                for (l, &i) in idx.iter().enumerate() {
                    t.coefficients_mut()[l * size..(l + 1) * size].copy_from_slice(&lanes[i].tensor);
                }
                measure(&name, n, rank, width, &lanes, t, reps, &mut out, &ins, &cyc);
            }
        }
    }
}
