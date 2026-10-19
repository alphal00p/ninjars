//! Generic-engine results must not depend on what the process prepared
//! before: Symbolica sorts terms and breaks ties in its Horner schemes by
//! their encoded bytes, which contain symbol IDs (registration indices), so
//! evaluators built after unrelated preparations could otherwise round
//! differently.
//!
//! The reference is a fresh child process that evaluates only the probe case.
//! The parent first registers a few hundred unrelated symbols (every later
//! symbol ID needs two bytes) and prepares and evaluates unrelated cases
//! (other topologies, numerators registering the probe's parameter names in
//! another order, other master integrals first), then the probe in this
//! thread and concurrently in several threads, and every result must be
//! bitwise equal to the child's.

use ninjars::{
    AmplitudeInput, Complex, DoubleFloat, FourMomentum, Laurent, MasterProvider, MasterRequest,
    MasterStatus, Real, SingleFloat,
};
use ninjars_symbolica::{
    OneLoopNativeMasters, PolynomialNumerator, SymbolicaAmplitudeKernels, SymbolicaCuts,
    SymbolicaHighCuts, SymbolicaMasters,
};
use symbolica::{
    atom::Atom,
    domains::{float::RealLike, rational::Rational},
    evaluate::EvaluationDomain,
    parser::ParseSettings,
};

const NAMESPACE: &str = "ninjars_determinism";
const CHILD: &str = "NINJARS_DETERMINISM_CHILD";

fn parse(text: &str) -> Atom {
    Atom::parse(text, NAMESPACE, ParseSettings::default()).unwrap()
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform on [-1, 1] rounded to 2^-12: exact in every carrier.
    fn uniform(&mut self) -> f64 {
        let x = (self.next() >> 11) as f64 / (1u64 << 53) as f64 * 2. - 1.;
        (x * 4096.).round() / 4096.
    }
}

struct Case {
    n: usize,
    rank: usize,
    expression: Atom,
    parameters: Vec<Atom>,
    shifts: Vec<[f64; 4]>,
    masses: Vec<f64>,
    coefficients: Vec<(f64, f64)>,
}

/// A random numerator of the given rank with explicit `mu2` terms and named
/// coefficients `c{k}`, registered in reverse order if `reverse_names`.
fn case(n: usize, rank: usize, seed: u64, reverse_names: bool) -> Case {
    let mut rng = Rng(seed);
    let mut terms = vec![(vec![0; 4], 0)];
    for w in 1..=rank {
        let mut exponents = vec![0u32; 4];
        for _ in 0..w {
            exponents[(rng.next() % 4) as usize] += 1;
        }
        terms.push((exponents, 0));
        if w >= 2 {
            let mut exponents = vec![0u32; 4];
            for _ in 0..w - 2 {
                exponents[(rng.next() % 4) as usize] += 1;
            }
            terms.push((exponents, 1));
        }
    }
    let names: Vec<String> = (0..terms.len()).map(|k| format!("c{k}")).collect();
    let mut parameters: Vec<Atom> = if reverse_names {
        names.iter().rev().map(|name| parse(name)).collect()
    } else {
        names.iter().map(|name| parse(name)).collect()
    };
    if reverse_names {
        parameters.reverse();
    }
    let mut text = String::from("0");
    for (k, (exponents, mu)) in terms.iter().enumerate() {
        text.push_str(&format!("+c{k}"));
        for (i, &e) in exponents.iter().enumerate() {
            if e > 0 {
                text.push_str(&format!("*q{i}^{e}"));
            }
        }
        if *mu > 0 {
            text.push_str(&format!("*mu2^{mu}"));
        }
    }
    let shifts = (0..n)
        .map(|i| {
            if i == 0 {
                [0.; 4]
            } else {
                [
                    rng.uniform() * 2.,
                    rng.uniform(),
                    rng.uniform(),
                    rng.uniform(),
                ]
            }
        })
        .collect();
    let masses = (0..n).map(|_| 0.25 + rng.uniform().abs()).collect();
    let coefficients = (0..terms.len())
        .map(|_| (rng.uniform(), rng.uniform()))
        .collect();
    Case {
        n,
        rank,
        expression: parse(&text),
        parameters,
        shifts,
        masses,
        coefficients,
    }
}

fn variables() -> [Atom; 5] {
    ["q0", "q1", "q2", "q3", "mu2"].map(parse)
}

/// Status and bits of every Laurent component of the generic engine's result.
fn evaluate<T>(
    case: &Case,
    prototype: T,
    masters: &mut impl MasterProvider<Complex<T>>,
) -> Vec<String>
where
    T: Real + SingleFloat + RealLike,
    Complex<T>: Real + EvaluationDomain,
{
    let p = &prototype;
    let exact = |x: f64| p.from_rational(&Rational::try_from(x).unwrap());
    let value = |x: f64| Complex::new(exact(x), p.zero());
    let mut numerator = PolynomialNumerator::new(
        case.expression.clone(),
        variables(),
        case.parameters.clone(),
        p.clone(),
    )
    .unwrap();
    let coefficients: Vec<_> = case
        .coefficients
        .iter()
        .map(|&(re, im)| Complex::new(exact(re), exact(im)))
        .collect();
    numerator.set_parameter_batch(&coefficients, 1).unwrap();
    let shifts: Vec<_> = case
        .shifts
        .iter()
        .map(|v| FourMomentum::from_args(value(v[0]), value(v[1]), value(v[2]), value(v[3])))
        .collect();
    let masses: Vec<_> = case.masses.iter().map(|&m| value(m)).collect();
    let mut amplitude = SymbolicaAmplitudeKernels::new(case.n, case.rank, p.clone())
        .unwrap()
        .into_amplitude(1)
        .unwrap();
    amplitude
        .evaluate_batch_with_inputs(
            AmplitudeInput {
                shifts: &shifts,
                masses_squared: &masses,
                scales_squared: &[value(1.5)],
                sampling_scales: &[value(2.)],
                s_matrix: None,
            },
            &mut numerator,
            &mut SymbolicaCuts::new(p.clone()).unwrap(),
            &mut SymbolicaHighCuts::new(p.clone()).unwrap(),
            masters,
        )
        .unwrap();
    let status = format!("{:?}", amplitude.statuses()[0]);
    std::iter::once(status)
        .chain(
            amplitude.results()[0]
                .iter()
                .map(|z| format!("{:?}|{:?}", z.re, z.im)),
        )
        .collect()
}

/// Binary64 native masters widened exactly into any carrier.
struct Widened(OneLoopNativeMasters);
impl<T: SingleFloat + RealLike> MasterProvider<Complex<T>> for Widened {
    fn evaluate(
        &mut self,
        request: MasterRequest<'_, Complex<T>>,
        output: &mut [Laurent<Complex<T>>],
    ) -> ninjars::Result<()> {
        let mut statuses = vec![MasterStatus::Success; request.rows()];
        self.evaluate_batch(request, output, &mut statuses)
    }
    fn evaluate_batch(
        &mut self,
        request: MasterRequest<'_, Complex<T>>,
        output: &mut [Laurent<Complex<T>>],
        statuses: &mut [MasterStatus],
    ) -> ninjars::Result<()> {
        let round = |values: &[Complex<T>]| -> Vec<Complex<f64>> {
            values
                .iter()
                .map(|z| Complex::new(z.re.to_f64(), z.im.to_f64()))
                .collect()
        };
        let prototype = request.scales_squared[0].re.clone();
        let (invariants, masses, scales) = (
            round(request.invariants),
            round(request.masses_squared),
            round(request.scales_squared),
        );
        let mut values = vec![[Complex::new(0., 0.); 3]; request.rows()];
        self.0.evaluate_batch(
            MasterRequest {
                integral: request.integral,
                invariants: &invariants,
                masses_squared: &masses,
                scales_squared: &scales,
            },
            &mut values,
            statuses,
        )?;
        let exact = |x: f64| prototype.from_rational(&Rational::try_from(x).unwrap());
        for (out, value) in output.iter_mut().zip(&values) {
            *out = value.map(|z| Complex::new(exact(z.re), exact(z.im)));
        }
        Ok(())
    }
}

/// Evaluate `case` in binary64 and DoubleFloat with native masters, and in
/// DoubleFloat with oneloopmaster's expressions (tensor bubbles and tadpoles
/// through Symbolica evaluators).
fn bits(case: &Case) -> Vec<String> {
    let double = || DoubleFloat::from(0.);
    let mut bits = evaluate(
        case,
        0_f64,
        &mut Widened(OneLoopNativeMasters::new().unwrap()),
    );
    bits.extend(evaluate(
        case,
        double(),
        &mut Widened(OneLoopNativeMasters::new().unwrap()),
    ));
    bits.extend(evaluate(
        case,
        double(),
        &mut SymbolicaMasters::new(double()),
    ));
    bits
}

/// The probe: six propagators at rank seven.
fn probe() -> Vec<String> {
    bits(&case(6, 7, 17, false))
}

/// Unrelated preparations: other topologies and ranks (other master integrals
/// first), whose numerators register the probe's coefficient names in
/// reverse order.
fn unrelated() {
    for (n, rank, seed) in [(1, 2, 3), (4, 5, 5), (2, 3, 7), (5, 6, 11)] {
        bits(&case(n, rank, seed, true));
    }
}

#[test]
#[ignore = "child process of generic_results_do_not_depend_on_process_history"]
fn determinism_child() {
    if std::env::var_os(CHILD).is_some() {
        println!("BITS {}", probe().join(";"));
    }
}

#[test]
fn generic_results_do_not_depend_on_process_history() {
    for i in 0..300 {
        drop(parse(&format!("filler{i}")));
    }
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "determinism_child", "--ignored", "--nocapture"])
        .env(CHILD, "1")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let reference: Vec<String> = stdout
        .lines()
        .find_map(|line| line.strip_prefix("BITS "))
        .unwrap_or_else(|| panic!("child failed: {stdout}"))
        .split(';')
        .map(str::to_owned)
        .collect();
    assert!(
        reference
            .iter()
            .filter(|bits| bits.contains("Success"))
            .count()
            == 3,
        "{reference:?}"
    );
    unrelated();
    assert_eq!(probe(), reference, "after unrelated preparations");
    let threads: Vec<_> = (0..4)
        .map(|thread| {
            std::thread::spawn(move || {
                if thread % 2 == 1 {
                    unrelated();
                }
                probe()
            })
        })
        .collect();
    for thread in threads {
        assert_eq!(thread.join().unwrap(), reference, "concurrent threads");
    }
}
