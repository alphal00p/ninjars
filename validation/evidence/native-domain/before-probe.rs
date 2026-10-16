use ninjars::{Complex, Float, Real, SingleFloat};
use symbolica::domains::float::RealLike;
use ninjars_symbolica::{LaurentDivision, TriangleBubbleSubtraction};
fn main() {
    let plan = TriangleBubbleSubtraction::for_quotient(&LaurentDivision::new(5, 3, 2).unwrap()).unwrap();
    let layout = plan.input_layout();
    let mut native = plan.compile_native(&0.).unwrap();
    let mut interpreted = plan.compile(&0.).unwrap();
    let mut oracle = plan.compile(&Float::new(256)).unwrap();
    println!("input_columns={}, outputs={:?}, d0_column={}", plan.input_columns(), plan.output_expansion(), layout.denominator.start);
    for (label, coefficient_kind) in [("simple", 0), ("recoverable_cubic", 1)] {
        for scale in [1e90, 1e100, 1e102, 4e102, 8e102, 1e103, 1e105, 1e110, 1e120, 1e155, -1e110, 1e-110] {
            let mut row = vec![Complex::new(0., 0.); plan.input_columns()];
            row[layout.projection_e3.start + 2] = Complex::new(1., 0.);
            row[layout.denominator.start] = Complex::new(scale, 0.);
            if coefficient_kind == 0 {
                row[layout.residue.start + 1] = Complex::new(1., 0.);
            } else {
                row[layout.residue.start + 3] = Complex::new(scale, 0.);
                row[layout.denominator.start + 1] = Complex::new(scale, 0.);
            }
            for rows in [1, 3] {
                let inputs = row.repeat(rows);
                let high: Vec<_> = inputs.iter().map(|z| Complex::new(Float::with_val(256, z.re), Float::with_val(256, z.im))).collect();
                let count = rows * plan.output_expansion().len();
                let mut a = vec![Complex::new(0., 0.); count];
                let mut b = a.clone();
                let mut c = vec![Complex::new(Float::new(256), Float::new(256)); count];
                let native_status = native.evaluate_batch(&inputs, &mut a, rows);
                let interpreted_status = interpreted.evaluate_batch(&inputs, &mut b, rows);
                let oracle_status = oracle.evaluate_batch(&high, &mut c, rows);
                let scaled_error: Vec<_> = a.iter().zip(&c).map(|(x,y)| {
                    let expected = Complex::new(y.re.to_f64(), y.im.to_f64());
                    (x.clone() - &expected).norm().re / expected.norm().re.max(1.)
                }).collect();
                println!("case={label},scale={scale:e},rows={rows},native_status={native_status:?},interpreted_status={interpreted_status:?},oracle_status={oracle_status:?},native={a:?},interpreted={b:?},oracle={c:?},native_finite={},interpreted_finite={},oracle_finite={},scaled_error={scaled_error:?}", a.iter().all(SingleFloat::is_finite), b.iter().all(SingleFloat::is_finite), c.iter().all(SingleFloat::is_finite));
            }
        }
    }
}
