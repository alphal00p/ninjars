#![cfg(feature = "reference-codegen")]

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

use ninjars::{
    Complex, ExpansionKind, ExpansionRequest, FourMomentum, Numerator, NumeratorRequest, SampleId,
};
use ninjars_symbolica::PolynomialNumerator;
use symbolica::{atom::Atom, parse};

fn variables() -> [Atom; 5] {
    [
        parse!("q0"),
        parse!("q1"),
        parse!("q2"),
        parse!("q3"),
        parse!("mu2"),
    ]
}

#[test]
fn reference_export_checks_topology_and_maps_coefficients_at_the_reference_boundary() {
    let numerator =
        PolynomialNumerator::new(parse!("q0^2/3+mu2^2"), variables(), vec![], 0.).unwrap();
    assert!(numerator.export_ninja_cpp(0).is_err());
    assert!(numerator.export_ninja_cpp(2).is_err());
    assert!(numerator.export_ninja_cpp(256).is_err());
    let source = numerator.export_ninja_cpp(3).unwrap();
    assert!(source.contains("constexpr unsigned rank = 4;"));
    assert!(source.contains("constexpr unsigned parameter_count = 0;"));
    assert!(source.contains("T(3.333333333333333"));
}

fn momentum(values: [(f64, f64); 4]) -> FourMomentum<Complex<f64>> {
    let [t, x, y, z] = values.map(|(re, im)| Complex::new(re, im));
    FourMomentum::from_args(t, x, y, z)
}

fn command_ok(command: &mut Command) {
    let output = command.output().expect("start C++ compiler");
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Uses the genuine original Ninja headers. Run after building the reference:
/// cargo test -p ninjars-symbolica --features reference-codegen --test reference \
///   -- --ignored --test-threads=1
#[test]
#[ignore = "requires C++ compiler and validation/reference/build.py artifacts"]
fn generated_callbacks_match_rust_and_preserve_caller_buffer_bounds() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let reference = std::env::var_os("NINJARS_REFERENCE_BUILD")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("target/reference"));
    assert!(
        reference
            .join("ninja-build/src/ninja/ninja_config.h")
            .exists(),
        "build the original Ninja reference first"
    );
    let temporary = std::env::temp_dir().join(format!(
        "ninjars-callback-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir(&temporary).unwrap();
    let result = std::panic::catch_unwind(|| {
        // This is the exact electroweak coupling factor occurring in the Hgg
        // fixture. Leaving π's evaluator slot uninitialized makes its inverse
        // square root singular. Compiled direct and every Laurent callback
        // must agree with the independently mapped Rust evaluator.
        let coupling = parse!(
            "pi^(3/2)*(((1/2)^(1/2)*(-9239168160000000/1717275997)*pi+270091425444897681/15625000000)^(1/2)*(-62500/519703209)+1/2)^(-1/2)*(((1/2)^(1/2)*(-9239168160000000/1717275997)*pi+270091425444897681/15625000000)^(1/2)+519703209/125000)^(-1/2)"
        );
        let mut numerator = PolynomialNumerator::new(
            parse!("a*q0^2+mu2*q1+mu2^2") + coupling * parse!("q0^2+mu2"),
            variables(),
            vec![parse!("a")],
            0.,
        )
        .unwrap();
        numerator
            .set_parameter_batch(&[Complex::new(0.3, 0.7)], 1)
            .unwrap();
        let vectors = [
            momentum([(1., 1.), (2., -1.), (3., 2.), (4., 0.)]),
            momentum([(2., 0.), (1., 1.), (0., 2.), (3., 0.)]),
            momentum([(3., 1.), (2., 0.), (1., 0.), (2., -1.)]),
            momentum([(4., 0.), (0., 1.), (2., 1.), (1., 0.)]),
        ];
        let beta = [
            Complex::new(2., -1.),
            Complex::new(3., 1.),
            Complex::new(5., 0.),
        ];
        for propagators in [3, 4] {
            // two-vector and one-vector mu extraction
            fs::write(
                temporary.join("numerator.hh"),
                numerator.export_ninja_cpp(propagators).unwrap(),
            )
            .unwrap();
            fs::write(temporary.join("probe.cc"), r#"
#include "numerator.hh"
#include <iomanip>
int main() {
  using ninja::Complex;
  using ninja::ComplexMomentum;
  const Complex parameters[] = {Complex(.3,.7)};
  ninjars_reference::GeneratedNumerator numerator(parameters);
  const ComplexMomentum v[] = {
    ComplexMomentum(Complex(1,1),Complex(2,-1),Complex(3,2),Complex(4,0)),
    ComplexMomentum(Complex(2,0),Complex(1,1),Complex(0,2),Complex(3,0)),
    ComplexMomentum(Complex(3,1),Complex(2,0),Complex(1,0),Complex(2,-1)),
    ComplexMomentum(Complex(4,0),Complex(0,1),Complex(2,1),Complex(1,0))};
  const Complex beta[] = {Complex(2,-1),Complex(3,1),Complex(5,0)};
  const ninja::PartitionInt partition[] = {0,1,2};
  const Complex sentinel(1234567,-9876543);
  auto print = [](const Complex& value) { std::cout << std::setprecision(17) << value.real() << ' ' << value.imag() << '\n'; };
  print(numerator.evaluate(v[0],beta[0],3,partition));
  Complex output[16];
  std::fill_n(output,16,sentinel);
  // Allocate exactly one vector in the ordinary-rank case: sanitizers can
  // detect any accidental access to the absent mu-expansion offset.
  if (ninjars_reference::rank > ninjars_reference::propagator_count)
    numerator.muExpansion(v,partition,output);
  else {
    const ComplexMomentum only_vector[] = {v[0]};
    numerator.muExpansion(only_vector,partition,output);
  }
  const unsigned mu_count = ninjars_reference::rank > ninjars_reference::propagator_count ? 2 : 1;
  for (unsigned i = 0; i < mu_count; ++i) print(output[i]);
  if (output[mu_count] != sentinel) return 2;
  const unsigned t3_count[] = {1,2,4,6,9};
  for (int depth = 0; depth <= 4; ++depth) {
    std::fill_n(output,16,sentinel);
    numerator.t3Expansion(v[0],v[1],v[2],beta[0],depth,3,partition,output);
    if (output[t3_count[depth]] != sentinel) return 3;
    for (unsigned i = 0; i < t3_count[depth]; ++i) print(output[i]);
  }
  const unsigned t2_count[] = {1,3,7,13};
  for (int depth = 0; depth <= 3; ++depth) {
    std::fill_n(output,16,sentinel);
    numerator.t2Expansion(v[0],v[1],v[2],v[3],beta,depth,2,partition,output);
    if (output[t2_count[depth]] != sentinel) return 4;
    for (unsigned i = 0; i < t2_count[depth]; ++i) print(output[i]);
  }
}
"#).unwrap();
            command_ok(
                Command::new(std::env::var_os("CXX").unwrap_or_else(|| "c++".into()))
                    .args(["-O2", "-std=c++11"])
                    .arg("-I")
                    .arg(reference.join("ninja-source/src"))
                    .arg("-I")
                    .arg(reference.join("ninja-build/src"))
                    .arg(temporary.join("probe.cc"))
                    .arg("-o")
                    .arg(temporary.join("probe")),
            );
            let output = Command::new(temporary.join("probe")).output().unwrap();
            assert!(
                output.status.success(),
                "callback driver returned {:?}",
                output.status
            );
            let actual: Vec<f64> = String::from_utf8(output.stdout)
                .unwrap()
                .split_whitespace()
                .map(|value| value.parse().unwrap())
                .collect();
            let mut expected = vec![Complex::new(0., 0.)];
            numerator
                .evaluate_batch(
                    &[NumeratorRequest {
                        id: SampleId { lane: 0, sample: 0 },
                        q: &vectors[0],
                        mu_squared: &beta[0],
                        partition: &[],
                    }],
                    &mut expected,
                )
                .unwrap();
            let mu_vectors = if propagators == 3 {
                &vectors[..2]
            } else {
                &vectors[..1]
            };
            let kind = ExpansionKind::Mu {
                powers: if propagators == 3 {
                    vec![4, 3]
                } else {
                    vec![4]
                },
            };
            let mut values = vec![Complex::new(0., 0.); kind.len()];
            numerator
                .expansion_batch(
                    &kind,
                    &[ExpansionRequest {
                        id: SampleId { lane: 0, sample: 0 },
                        partition: &[],
                        vectors: mu_vectors,
                        parameters: &[],
                    }],
                    &mut values,
                )
                .unwrap();
            expected.extend(values);
            for depth in 0..=4 {
                let kind = ExpansionKind::T3 {
                    powers: (0..=depth)
                        .flat_map(|d| (0..=d / 2).map(move |mu| (4 - d, mu as u16)))
                        .collect(),
                };
                let mut values = vec![Complex::new(0., 0.); kind.len()];
                numerator
                    .expansion_batch(
                        &kind,
                        &[ExpansionRequest {
                            id: SampleId { lane: 0, sample: 0 },
                            partition: &[],
                            vectors: &vectors[..3],
                            parameters: &beta[..1],
                        }],
                        &mut values,
                    )
                    .unwrap();
                expected.extend(values);
            }
            for depth in 0..=3 {
                let kind = ExpansionKind::T2 {
                    powers: (0..=depth)
                        .flat_map(|d| {
                            (0..=d).flat_map(move |x| {
                                (0..=(d - x) / 2).map(move |mu| (4 - d, x as u16, mu as u16))
                            })
                        })
                        .collect(),
                };
                let mut values = vec![Complex::new(0., 0.); kind.len()];
                numerator
                    .expansion_batch(
                        &kind,
                        &[ExpansionRequest {
                            id: SampleId { lane: 0, sample: 0 },
                            partition: &[],
                            vectors: &vectors,
                            parameters: &beta,
                        }],
                        &mut values,
                    )
                    .unwrap();
                expected.extend(values);
            }
            assert_eq!(actual.len(), 2 * expected.len());
            for (actual, expected) in actual.as_chunks::<2>().0.iter().zip(expected) {
                for (actual, expected) in actual.iter().zip([expected.re, expected.im]) {
                    assert!(
                        (actual - expected).abs() < 1e-11 * (1. + expected.abs()),
                        "{actual} != {expected}"
                    );
                }
            }
        }
    });
    fs::remove_dir_all(&temporary).unwrap();
    if let Err(panic) = result {
        std::panic::resume_unwind(panic);
    }
}
