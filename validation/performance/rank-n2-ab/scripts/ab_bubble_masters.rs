//! Rank n+2 performance guard (validation/performance/rank-n2-ab): native
//! bubble masters per row, OneLoopNativeMasters through the FastMasters
//! interface, families BubbleRank2 (B0, B1, B11), BubbleRank3 (+ B111) and
//! BubbleRank4 (+ B1111, rank n+2 only), on identical rows: every propagator
//! pair of every lane of the benchmark cases (`p² = (V_i - V_j)²`, the two
//! masses, the lane's mu²). Batches of 1 and of all rows; median ns per row
//! over 15 blocks. Integration tree only. Output TSV on $BUBBLE_OUT.

use ninjars::fast::{C64, FastMaster, FastMasters, RMom};
use ninjars_symbolica::OneLoopNativeMasters;
use serde_json::Value;
use std::hint::black_box;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap()
}

#[test]
#[ignore]
fn bubble_masters() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut rows: Vec<C64> = Vec::new();
    for d in ["validation/benchmark/cases", "validation/benchmark/synthetic"] {
        let mut files: Vec<PathBuf> = std::fs::read_dir(root.join(d)).unwrap().map(|e| e.unwrap().path())
            .filter(|p| p.extension().is_some_and(|e| e == "json")).collect();
        files.sort();
        for f in files {
            let v: Value = serde_json::from_str(&std::fs::read_to_string(&f).unwrap()).unwrap();
            for l in v["lanes"].as_array().unwrap() {
                let shifts: Vec<RMom> = l["shifts"].as_array().unwrap().iter().map(|p| [num(&p[0]), num(&p[1]), num(&p[2]), num(&p[3])]).collect();
                let masses: Vec<C64> = l["masses_squared"].as_array().unwrap().iter().map(|z| C64::new(num(&z[0]), num(&z[1]))).collect();
                let mu2 = num(&l["mu_squared"]);
                for i in 0..shifts.len() {
                    for j in i + 1..shifts.len() {
                        let k: Vec<f64> = (0..4).map(|m| shifts[i][m] - shifts[j][m]).collect();
                        let p2 = k[0] * k[0] - k[1] * k[1] - k[2] * k[2] - k[3] * k[3];
                        rows.extend([C64::new(p2, 0.), masses[i], masses[j], C64::new(mu2, 0.)]);
                    }
                }
            }
        }
    }
    let nrows = rows.len() / 4;
    let mut masters = OneLoopNativeMasters::new().unwrap();
    let mut out = std::io::BufWriter::new(std::fs::File::create(std::env::var("BUBBLE_OUT").unwrap()).unwrap());
    writeln!(out, "family\tbatch\trows\tns_per_row\tfailed_rows").unwrap();
    for family in [FastMaster::BubbleRank2, FastMaster::BubbleRank3, FastMaster::BubbleRank4] {
        for batch in [1usize, nrows] {
            let mut res = vec![[C64::new(0., 0.); 3]; batch * family.outputs()];
            let mut ok = vec![false; batch];
            let mut run = || {
                for chunk in rows.chunks(4 * batch) {
                    let b = chunk.len() / 4;
                    masters.evaluate(family, chunk, &mut res[..b * family.outputs()], &mut ok[..b]).unwrap();
                    black_box(&res);
                }
            };
            for _ in 0..3 {
                run();
            }
            let mut ns = Vec::new();
            for _ in 0..15 {
                let start = Instant::now();
                for _ in 0..20 {
                    run();
                }
                ns.push(start.elapsed().as_nanos() as f64 / (20 * nrows) as f64);
            }
            ns.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let failed = ok.iter().filter(|x| !**x).count();
            writeln!(out, "{family:?}\t{batch}\t{nrows}\t{:.1}\t{failed}", ns[7]).unwrap();
        }
    }
}
