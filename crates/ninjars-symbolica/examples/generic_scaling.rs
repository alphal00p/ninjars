//! Setup and per-point cost of the generic engine as the propagator count grows.
//!
//! For each n, a random rank-n polynomial numerator with explicit mu² terms
//! (every term of weight <= 3, the maximal-weight q0/mu² terms, 1/40 of the
//! other monomials through weight six and 1/97 above, as in `tests/any_n.rs`) is reduced on random real
//! kinematics with complex masses. Reported per n and precision:
//! - `kernels`: construction and compilation of every generated reduction,
//!   integration and accumulation kernel
//!   (`SymbolicaAmplitudeKernels::new` + `into_amplitude`);
//! - `numerator`: polynomial-numerator preparation, including the Laurent
//!   expansion evaluators built on first use (a property of the numerator
//!   adapter and its rank, not of the reduction);
//! - `point`: steady-state time per phase-space point of the full reduction
//!   with oneloopmaster masters (native for f64, interpreted for Float);
//! - `reduction`: the same with cheap polynomial stand-in masters, i.e. the
//!   reduction alone (cuts, divisions, subtractions, contraction, accumulation).
//!
//! ```text
//! cargo run --profile perf -p ninjars-symbolica --example generic_scaling -- \
//!     [--bits 256] [--lanes 8] [--float-lanes 2] [--repeat 2] [--no-float] 6 7 8 10
//! ```

use ninjars::fast::{C64, RMom, tensor_layout::monomials};
use ninjars::{
    Amplitude, AmplitudeInput, Complex, EvaluationStatus, Float, FloatLike, FourMomentum, Laurent,
    MasterProvider, MasterRequest, MasterStatus,
};
use ninjars_symbolica::{
    PolynomialNumerator, SymbolicaAmplitudeKernels, SymbolicaCuts, SymbolicaHighCuts,
    SymbolicaMasters,
};
use oneloop::NativeFloat;
use std::time::Instant;
use symbolica::{
    atom::Atom, domains::rational::Rational, evaluate::EvaluationDomain, parser::ParseSettings,
};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        let x = (self.next() >> 11) as f64 / (1u64 << 53) as f64 * 2. - 1.;
        (x * 1048576.).round() / 1048576.
    }
}

fn parse(text: &str) -> Atom {
    Atom::parse(text, "ninjars_generic_scaling", ParseSettings::default()).unwrap()
}

fn polynomial(rank: usize) -> (Atom, Vec<Atom>) {
    let names = ["q0", "q1", "q2", "q3"];
    let mut terms: Vec<_> = monomials(rank).into_iter().map(|m| (m, 0)).collect();
    for j in 1..=rank as u32 / 2 {
        terms.extend(monomials(rank - 2 * j as usize).into_iter().map(|m| (m, j)));
    }
    let mut text = String::from("0");
    let mut parameters = Vec::new();
    for (k, (m, j)) in terms.iter().enumerate() {
        let weight = (m.iter().sum::<u32>() + 2 * j) as usize;
        let hash = k.wrapping_mul(2_654_435_761) >> 7;
        let keep = weight <= 3
            || (weight == rank && m[1] + m[2] + m[3] == 0)
            || (weight <= 6 && hash.is_multiple_of(40))
            || hash.is_multiple_of(97);
        if !keep {
            continue;
        }
        text.push_str(&format!("+p{k}"));
        for (mu, &e) in m.iter().enumerate() {
            if e > 0 {
                text.push_str(&format!("*{}^{e}", names[mu]));
            }
        }
        if *j > 0 {
            text.push_str(&format!("*mu2^{j}"));
        }
        parameters.push(parse(&format!("p{k}")));
    }
    (parse(&text), parameters)
}

fn mp(a: &RMom, b: &RMom) -> f64 {
    a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
}

fn exact<T: NativeFloat>(p: &T, x: f64) -> T {
    p.from_rational(&Rational::try_from(x).unwrap())
}

/// Cheap polynomial stand-ins for every master family (reduction-only timing).
struct PolynomialMasters;

impl<V: FloatLike> MasterProvider<V> for PolynomialMasters {
    fn evaluate(
        &mut self,
        request: MasterRequest<'_, V>,
        output: &mut [Laurent<V>],
    ) -> ninjars::Result<()> {
        let mut statuses = vec![MasterStatus::Success; request.rows()];
        self.evaluate_batch(request, output, &mut statuses)
    }

    fn evaluate_batch(
        &mut self,
        request: MasterRequest<'_, V>,
        output: &mut [Laurent<V>],
        statuses: &mut [MasterStatus],
    ) -> ninjars::Result<()> {
        let ni = request.integral.invariant_count();
        let nm = request.integral.mass_count();
        for row in 0..request.rows() {
            let scale = &request.scales_squared[row];
            let mut sum = scale.clone();
            for a in request.invariants[row * ni..(row + 1) * ni]
                .iter()
                .chain(&request.masses_squared[row * nm..(row + 1) * nm])
            {
                sum += a;
            }
            output[row] = [sum.clone() * &sum, sum, scale.one()];
            statuses[row] = MasterStatus::Success;
        }
        Ok(())
    }
}

struct Timing {
    kernels: f64,
    numerator: f64,
    point: f64,
    reduction: f64,
}

fn measure<T: NativeFloat>(n: usize, prototype: T, lanes: usize, repeat: usize) -> Timing
where
    Complex<T>: ninjars::Real + EvaluationDomain,
{
    let p = &prototype;
    let rank = n;
    let mut rng = Rng(0x5ca1e + n as u64);
    let shifts: Vec<RMom> = (0..lanes * n)
        .map(|i| {
            if i % n == 0 {
                [0.; 4]
            } else {
                [
                    2. * rng.uniform(),
                    rng.uniform(),
                    rng.uniform(),
                    rng.uniform(),
                ]
            }
        })
        .collect();
    let masses: Vec<C64> = (0..lanes * n)
        .map(|_| C64::new(0.3 + rng.uniform().abs(), -0.1 * rng.uniform().abs()))
        .collect();
    let (expression, parameters) = polynomial(rank);
    let c = |z: C64| Complex::new(exact(p, z.re), exact(p, z.im));
    let values: Vec<_> = (0..lanes * parameters.len())
        .map(|_| c(C64::new(rng.uniform(), rng.uniform())))
        .collect();
    let variables = ["q0", "q1", "q2", "q3", "mu2"].map(parse);

    let start = Instant::now();
    let mut amplitude = SymbolicaAmplitudeKernels::new(n, rank, p.clone())
        .unwrap()
        .into_amplitude(lanes)
        .unwrap();
    let kernels = start.elapsed().as_secs_f64();

    let numerator_start = Instant::now();
    let mut numerator =
        PolynomialNumerator::new(expression, variables, parameters, p.clone()).unwrap();
    numerator.set_parameter_batch(&values, lanes).unwrap();
    let numerator_new = numerator_start.elapsed().as_secs_f64();

    let generic_shifts: Vec<FourMomentum<Complex<T>>> = shifts
        .iter()
        .map(|v| FourMomentum::from(v.map(|x| c(C64::real(x)))))
        .collect();
    let generic_masses: Vec<_> = masses.iter().map(|&m| c(m)).collect();
    let scales = vec![c(C64::real(1.)); lanes];
    let sampling: Vec<_> = (0..lanes)
        .map(|lane| {
            let v = &shifts[lane * n..(lane + 1) * n];
            let mut s = 1_f64;
            for i in 0..n {
                for j in i + 1..n {
                    let d = [
                        v[i][0] - v[j][0],
                        v[i][1] - v[j][1],
                        v[i][2] - v[j][2],
                        v[i][3] - v[j][3],
                    ];
                    s = s.max(mp(&d, &d).abs());
                }
            }
            c(C64::real(s))
        })
        .collect();
    let mut low = SymbolicaCuts::new(p.clone()).unwrap();
    let mut high = SymbolicaHighCuts::new(p.clone()).unwrap();
    let mut masters = SymbolicaMasters::new(p.clone());
    let mut run = |amplitude: &mut Amplitude<_, _>, standin: bool| {
        amplitude.reset();
        let input = AmplitudeInput {
            shifts: &generic_shifts,
            masses_squared: &generic_masses,
            scales_squared: &scales,
            sampling_scales: &sampling,
            s_matrix: None,
        };
        let start = Instant::now();
        if standin {
            amplitude
                .evaluate_batch_with_inputs(
                    input,
                    &mut numerator,
                    &mut low,
                    &mut high,
                    &mut PolynomialMasters,
                )
                .unwrap();
        } else {
            amplitude
                .evaluate_batch_with_inputs(
                    input,
                    &mut numerator,
                    &mut low,
                    &mut high,
                    &mut masters,
                )
                .unwrap();
        }
        let elapsed = start.elapsed().as_secs_f64();
        let ok = amplitude
            .statuses()
            .iter()
            .filter(|s| **s == EvaluationStatus::Success)
            .count();
        (elapsed, ok)
    };
    // The first evaluation also builds the numerator's expansion evaluators.
    let (first, _) = run(&mut amplitude, true);
    let mut reduction = f64::INFINITY;
    let mut point = f64::INFINITY;
    let mut successes = 0;
    for _ in 0..repeat {
        reduction = reduction.min(run(&mut amplitude, true).0);
        let (elapsed, ok) = run(&mut amplitude, false);
        point = point.min(elapsed);
        successes = ok;
    }
    assert!(successes > 0, "no successful lane at n={n}");
    Timing {
        kernels,
        numerator: numerator_new + first - reduction,
        point: point / lanes as f64,
        reduction: reduction / lanes as f64,
    }
}

fn main() {
    let mut bits = 256u32;
    let mut lanes = 8usize;
    let mut float_lanes = 2usize;
    let mut repeat = 2usize;
    let mut float = true;
    let mut ns = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--bits" => bits = args.next().unwrap().parse().unwrap(),
            "--lanes" => lanes = args.next().unwrap().parse().unwrap(),
            "--float-lanes" => float_lanes = args.next().unwrap().parse().unwrap(),
            "--repeat" => repeat = args.next().unwrap().parse().unwrap(),
            "--no-float" => float = false,
            _ => ns.push(arg.parse::<usize>().unwrap()),
        }
    }
    if ns.is_empty() {
        ns = vec![6, 7, 8, 10];
    }
    println!(
        "{:>3} {:>5} {:>11} {:>12} {:>14} {:>16} {:>16}",
        "n", "rank", "precision", "kernels [s]", "numerator [s]", "point [ms]", "reduction [ms]"
    );
    let row = |n: usize, label: &str, t: Timing| {
        println!(
            "{n:>3} {n:>5} {label:>11} {:>12.3} {:>14.2} {:>16.3} {:>16.3}",
            t.kernels,
            t.numerator,
            t.point * 1e3,
            t.reduction * 1e3
        );
    };
    for &n in &ns {
        row(n, "f64", measure(n, 0_f64, lanes, repeat));
        if !float {
            continue;
        }
        row(
            n,
            &format!("Float({bits})"),
            measure(n, Float::new(bits), float_lanes, repeat),
        );
    }
}
