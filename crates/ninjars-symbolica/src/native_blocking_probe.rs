//! Test-only blocking of unchanged native instruction graphs. No production policy.
use super::{CompiledKernel, ExactKernel, KernelExecutor};
use crate::{
    BubbleSourceProjection, DenominatorExpansion, LaurentDivision, TadpoleSourceProjection,
    TriangleBubbleSubtraction,
    native_kernels::{self, DenominatorPlan, bubbles::BubblePlan, tadpoles::TadpoleSourcePlan},
};
use ninjars::{BatchShape, Complex, DoubleFloat, Float, FloatLike, Real, Result, SingleFloat};
use oneloop::rust_codegen::runtime::NativeFloat;
use std::{cell::Cell, hint::black_box, time::Instant};
use symbolica::{
    domains::{
        dual::{DualNumber, HyperDual},
        rational::Rational,
    },
    evaluate::EvaluationDomain,
};

thread_local! { static BLOCK: Cell<Option<usize>> = const { Cell::new(None) }; }
const BLOCKS: [Option<usize>; 5] = [None, Some(8), Some(16), Some(32), Some(64)];

fn with_block<R>(block: Option<usize>, f: impl FnOnce() -> R) -> R {
    assert!(block != Some(0));
    struct Restore(Option<usize>);
    impl Drop for Restore {
        fn drop(&mut self) {
            BLOCK.set(self.0);
        }
    }
    let _restore = Restore(BLOCK.replace(block));
    f()
}

struct Blocked<V> {
    inner: Box<dyn KernelExecutor<V> + Send + Sync>,
    inputs: usize,
    outputs: usize,
    block: usize,
}
impl<V> KernelExecutor<V> for Blocked<V> {
    fn set_native_batch_execution(&mut self, policy: crate::BatchExecutionPolicy) {
        self.inner.set_native_batch_execution(policy);
    }

    fn evaluate(&mut self, input: &[V], output: &mut [V], rows: usize) -> Result<()> {
        // Validate the entire public call before the first chunk writes anything.
        BatchShape {
            rows,
            columns: self.inputs,
        }
        .validate("blocked probe input", input.len())?;
        BatchShape {
            rows,
            columns: self.outputs,
        }
        .validate("blocked probe output", output.len())?;
        if rows == 0 {
            return self.inner.evaluate(input, output, 0);
        }
        for first in (0..rows).step_by(self.block) {
            let end = first.saturating_add(self.block).min(rows);
            self.inner.evaluate(
                &input[first * self.inputs..end * self.inputs],
                &mut output[first * self.outputs..end * self.outputs],
                end - first,
            )?;
        }
        Ok(())
    }
    fn probe_workspace(&self) -> Option<(usize, usize)> {
        self.inner.probe_workspace()
    }
}
pub(super) fn wrap<T: 'static>(
    inner: Box<dyn KernelExecutor<Complex<T>> + Send + Sync>,
    inputs: usize,
    outputs: usize,
) -> Box<dyn KernelExecutor<Complex<T>> + Send + Sync> {
    if let Some(block) = BLOCK.get() {
        Box::new(Blocked {
            inner,
            inputs,
            outputs,
            block,
        })
    } else {
        inner
    }
}

#[derive(Clone, Copy, Debug)]
enum Case {
    T2,
    T3,
    BubbleProjection,
    Correction(usize),
    Tadpole(usize),
}
impl Case {
    fn name(self) -> String {
        format!("{self:?}")
    }
    fn exact(self) -> ExactKernel {
        match self {
            Self::T2 => DenominatorExpansion::new(2).unwrap().exact,
            Self::T3 => DenominatorExpansion::new(3).unwrap().exact,
            Self::BubbleProjection => BubbleSourceProjection::new().exact,
            Self::Correction(degree) => {
                TriangleBubbleSubtraction::for_quotient(
                    &LaurentDivision::new(3, degree + 1, 2).unwrap(),
                )
                .unwrap()
                .exact
            }
            Self::Tadpole(source) => TadpoleSourceProjection::new(source).unwrap().exact,
        }
    }
    fn dimensions(self) -> (usize, usize) {
        match self {
            Self::T2 => (29, 7),
            Self::T3 => (23, 4),
            Self::BubbleProjection => (28, 8),
            Self::Correction(d) => (if d == 3 { 33 } else { 28 }, [1, 3, 7, 13][d]),
            Self::Tadpole(3) => (24, 6),
            Self::Tadpole(2) => (28, 9),
            _ => unreachable!(),
        }
    }
    fn has_dual(self) -> bool {
        matches!(self, Self::T2 | Self::T3 | Self::Tadpole(_))
    }
    fn install<T: NativeFloat>(self, kernel: &mut CompiledKernel<T>, p: &T)
    where
        Complex<T>: Real + EvaluationDomain,
    {
        match self {
            Self::T2 | Self::T3 => native_kernels::install_denominator(
                kernel,
                if matches!(self, Self::T2) {
                    DenominatorPlan::T2
                } else {
                    DenominatorPlan::T3
                },
                p,
            ),
            Self::BubbleProjection => {
                native_kernels::bubbles::install(kernel, BubblePlan::SourceProjection, p)
            }
            Self::Correction(d) => native_kernels::bubbles::install(
                kernel,
                [
                    BubblePlan::OrdinaryDegree0,
                    BubblePlan::OrdinaryDegree1,
                    BubblePlan::OrdinaryDegree2,
                    BubblePlan::HigherDegree3,
                ][d],
                p,
            ),
            Self::Tadpole(n) => native_kernels::tadpoles::install_projection(
                kernel,
                TadpoleSourcePlan::for_source(n).unwrap(),
                p,
            ),
        }
        .unwrap();
    }
    fn kernel<T: NativeFloat>(self, p: &T, dual: bool, block: Option<usize>) -> CompiledKernel<T>
    where
        Complex<T>: Real + EvaluationDomain,
    {
        with_block(block, || {
            let (inputs, outputs) = self.dimensions();
            let mut kernel = CompiledKernel {
                native: None,
                inputs,
                outputs,
                evaluator: None,
            };
            if !dual {
                self.install(&mut kernel, p);
                return kernel;
            }
            assert!(self.has_dual());
            let zero = Complex::new(p.zero(), p.zero());
            kernel.inputs *= 5;
            kernel.outputs *= 5;
            let mut d = super::CompiledDualKernel {
                kernel,
                inputs,
                outputs,
                shape: shape(),
                prototype: HyperDual::from_values(shape(), vec![zero.clone(); 5]),
                base_zero: zero,
                arguments: Vec::new(),
                results: Vec::new(),
            };
            match self {
                Self::T2 | Self::T3 => native_kernels::install_dual_denominator(
                    &mut d,
                    if matches!(self, Self::T2) {
                        DenominatorPlan::T2
                    } else {
                        DenominatorPlan::T3
                    },
                    p,
                ),
                Self::Tadpole(n) => native_kernels::tadpoles::install_dual_projection(
                    &mut d,
                    TadpoleSourcePlan::for_source(n).unwrap(),
                    p,
                ),
                _ => unreachable!(),
            }
            .unwrap();
            assert!(d.kernel.native.is_some());
            d.kernel
        })
    }
}
fn cases() -> Vec<Case> {
    vec![
        Case::T2,
        Case::T3,
        Case::BubbleProjection,
        Case::Correction(0),
        Case::Correction(1),
        Case::Correction(2),
        Case::Correction(3),
        Case::Tadpole(3),
        Case::Tadpole(2),
    ]
}
fn shape() -> Vec<Vec<usize>> {
    vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![2, 0], vec![1, 1]]
}
fn inputs<T: NativeFloat>(p: &T, columns: usize, rows: usize) -> Vec<Complex<T>> {
    (0..rows * columns)
        .map(|i| {
            Complex::new(
                p.from_rational(&Rational::from((64 + (i % 37) as i64, 64))),
                p.from_rational(&Rational::from(((i % 13) as i64 - 6, 128))),
            )
        })
        .collect()
}
trait ExactProbe: NativeFloat {
    fn same(a: &Self, b: &Self);
}
impl ExactProbe for f64 {
    fn same(a: &Self, b: &Self) {
        if a.is_nan() {
            assert!(b.is_nan());
        } else {
            assert_eq!(a.to_bits(), b.to_bits(), "{a:?} != {b:?}");
        }
    }
}
impl ExactProbe for DoubleFloat {
    fn same(a: &Self, b: &Self) {
        let (a, b) = (a.into_inner(), b.into_inner());
        <f64 as ExactProbe>::same(&a.hi(), &b.hi());
        <f64 as ExactProbe>::same(&a.lo(), &b.lo());
    }
}
impl ExactProbe for Float {
    fn same(a: &Self, b: &Self) {
        assert_eq!(
            a.get_precision(),
            b.get_precision(),
            "tracked precision differs: {a} / {b}"
        );
        if a.is_finite() {
            assert_eq!(a, b);
            assert_eq!(a.negative_sign(), b.negative_sign());
        } else {
            assert_eq!(a.real_cmp(a).is_none(), b.real_cmp(b).is_none());
            assert_eq!(a.is_finite(), b.is_finite());
            if a.real_cmp(a).is_some() {
                assert_eq!(a.negative_sign(), b.negative_sign());
            }
        }
    }
}
fn same<T: ExactProbe>(a: &[Complex<T>], b: &[Complex<T>]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        T::same(&a.re, &b.re);
        T::same(&a.im, &b.im);
    }
}
fn selected_kernel<T: NativeFloat>(
    case: Case,
    p: &T,
    dual: bool,
    block: Option<usize>,
    supported: bool,
) -> CompiledKernel<T>
where
    Complex<T>: Real + EvaluationDomain,
{
    let mut kernel = case.kernel(p, dual, if supported { None } else { block });
    if supported {
        kernel.set_native_batch_execution(block.map_or(
            crate::BatchExecutionPolicy::Whole,
            |rows| {
                crate::BatchExecutionPolicy::FixedRows(std::num::NonZeroUsize::new(rows).unwrap())
            },
        ));
    }
    kernel
}

fn exercise<T: ExactProbe>(p: T, supported: bool)
where
    Complex<T>: Real + EvaluationDomain,
{
    let mut compared = 0;
    for case in cases() {
        for dual in [false, true] {
            if dual && !case.has_dual() {
                continue;
            }
            let multiplier = if dual { 5 } else { 1 };
            let (columns, outputs) = case.dimensions();
            let (columns, outputs) = (columns * multiplier, outputs * multiplier);
            // Dedicated, fresh workspaces; reuse across growing/shrinking/tail calls.
            let mut variants: Vec<_> = BLOCKS
                .iter()
                .map(|&b| selected_kernel(case, &p, dual, b, supported))
                .collect();
            for rows in [
                0, 1, 3, 8, 9, 16, 17, 20, 30, 31, 32, 33, 40, 60, 64, 65, 128, 3, 0, 1,
            ] {
                let mut input = inputs(&p, columns, rows);
                if rows % 2 == 1 {
                    input = input.chunks(columns).rev().flatten().cloned().collect();
                }
                let zero = Complex::new(p.zero(), p.zero());
                let mut expected = vec![zero.clone(); rows * outputs];
                variants[0]
                    .evaluate_batch(&input, &mut expected, rows)
                    .unwrap();
                for variant in &mut variants[1..] {
                    let mut actual = vec![zero.clone(); rows * outputs];
                    variant.evaluate_batch(&input, &mut actual, rows).unwrap();
                    same(&actual, &expected);
                    compared += actual.len();
                    let before = actual.clone();
                    assert!(
                        variant
                            .evaluate_batch(&input, &mut actual, usize::MAX)
                            .is_err()
                    );
                    same(&actual, &before);
                }
            }
            for (index, variant) in variants.iter().enumerate().skip(1) {
                let (capacity, _) = variant.native.as_ref().unwrap().probe_workspace().unwrap();
                assert!(capacity <= BLOCKS[index].unwrap());
            }
        }
    }
    println!(
        "NATIVE_BLOCKING_EXACT {}",
        serde_json::json!({"precision_bits":p.get_precision(),"complex_components_compared":compared,"signed_zero_bits_checked":true,"derivative_shape":shape(),"timing":false})
    );
}
#[test]
fn native_blocking_preserves_all_components_and_precision() {
    exercise(0f64, false);
    exercise(DoubleFloat::default(), false);
    exercise(Float::new(192), false);
}

#[test]
fn native_blocking_natural_scalar_tails_preserve_exceptional_values() {
    exceptional_control(false);
}
fn exceptional_control(supported: bool) {
    for case in cases() {
        for dual in [false, true] {
            if dual && !case.has_dual() {
                continue;
            }
            let multiplier = if dual { 5 } else { 1 };
            let (columns, outputs) = case.dimensions();
            let (columns, outputs) = (columns * multiplier, outputs * multiplier);
            for rows in [9, 17, 33, 65] {
                let mut input = inputs(&0., columns, rows);
                for row in 0..rows {
                    let offset = row * columns;
                    match row % 8 {
                        0 => input[offset] = Complex::new(-0., 0.),
                        1 => input[offset] = Complex::new(f64::INFINITY, 0.),
                        2 => input[offset] = Complex::new(f64::NAN, 0.),
                        3 => input[offset] = Complex::new(f64::MIN_POSITIVE, -0.),
                        4 => input[offset] = Complex::new(1e155, -1e155),
                        5 => input[offset] = Complex::new(1e-155, -1e-155),
                        _ => (),
                    }
                    if matches!(case, Case::Correction(_)) {
                        let denominator = if matches!(case, Case::Correction(3)) {
                            26
                        } else {
                            21
                        };
                        input[offset + denominator] = if row % 2 == 0 {
                            Complex::new(0., -0.)
                        } else {
                            Complex::new(1e110, 0.)
                        };
                    }
                    if dual && row % 3 == 0 {
                        input[offset + 4] = Complex::new(f64::NAN, 0.);
                    }
                }
                let mut expected = vec![Complex::new(0., 0.); rows * outputs];
                case.kernel(&0., dual, None)
                    .evaluate_batch(&input, &mut expected, rows)
                    .unwrap();
                for block in BLOCKS.into_iter().skip(1) {
                    let mut actual = expected.clone();
                    selected_kernel(case, &0., dual, block, supported)
                        .evaluate_batch(&input, &mut actual, rows)
                        .unwrap();
                    same(&actual, &expected);
                }
            }
        }
    }
}

#[test]
fn native_blocking_retains_mixed_input_significant_bits() {
    mixed_precision_control(false);
}
fn mixed_precision_control(supported: bool) {
    let p = Float::new(256);
    for case in cases() {
        let (columns, outputs) = case.dimensions();
        let rows = 65;
        let input: Vec<_> = (0..rows * columns)
            .map(|i| {
                let bits = [53, 96, 160, 256][i % 4];
                let p = Float::new(bits);
                Complex::new(
                    p.from_rational(&Rational::from((64 + (i % 37) as i64, 64))),
                    p.from_rational(&Rational::from(((i % 13) as i64 - 6, 128))),
                )
            })
            .collect();
        let mut expected = vec![Complex::new(p.zero(), p.zero()); rows * outputs];
        case.kernel(&p, false, None)
            .evaluate_batch(&input, &mut expected, rows)
            .unwrap();
        for block in BLOCKS.into_iter().skip(1) {
            let mut actual = expected.clone();
            selected_kernel(case, &p, false, block, supported)
                .evaluate_batch(&input, &mut actual, rows)
                .unwrap();
            same(&actual, &expected);
        }
    }
}

#[test]
#[ignore = "requires a coordinated quiet window and NINJARS_BLOCKING_QUIET_WINDOW=approved"]
#[allow(clippy::assertions_on_constants)]
fn native_blocking_microbenchmark() {
    assert!(!cfg!(debug_assertions), "use --release");
    assert_eq!(
        std::env::var("NINJARS_BLOCKING_QUIET_WINDOW").as_deref(),
        Ok("approved")
    );
    let root = std::env::var("NINJARS_BLOCKING_OUTPUT").expect("set a new output JSON path");
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&root)
        .unwrap();
    let metadata = benchmark_metadata();
    assert_eq!(metadata["status"], "", "commit the probe before timing");
    for dependency in metadata["dependencies"].as_array().unwrap() {
        assert_eq!(
            dependency["status"], "",
            "dependency must be clean before timing"
        );
    }
    write_record(
        &mut file,
        &serde_json::json!({"type":"metadata","value":metadata}),
    );
    let mut records = Vec::new();
    // Seven actual n5/r4 executor instances, after same-call denominator reuse.
    for (case, rows_per_lane) in [
        (Case::T2, 60),
        (Case::T3, 40),
        (Case::T3, 20),
        (Case::BubbleProjection, 60),
        (Case::Correction(1), 60),
        (Case::Tadpole(3), 30),
        (Case::Tadpole(2), 20),
    ] {
        let (columns, outputs) = case.dimensions();
        for width in [1, 8, 32, 128, 1024, 31] {
            let rows = rows_per_lane * width;
            let input = inputs(&0., columns, rows);
            let mut variants: Vec<_> = BLOCKS.iter().map(|&b| case.kernel(&0., false, b)).collect();
            let mut values = vec![vec![Complex::new(0., 0.); rows * outputs]; BLOCKS.len()];
            for (variant, output) in variants.iter_mut().zip(&mut values) {
                variant.evaluate_batch(&input, output, rows).unwrap();
            }
            for output in &values[1..] {
                same(output, &values[0]);
            }
            let rounds = 262144usize.div_ceil(rows).max(4);
            let mut blocks = Vec::new();
            for repeat in 0..10 {
                let mut order: Vec<_> = (0..5).map(|i| (i + repeat) % 5).collect();
                if repeat >= 5 {
                    order.reverse();
                }
                for (position, &mode) in order.iter().enumerate() {
                    let start = Instant::now();
                    for _ in 0..rounds {
                        variants[mode]
                            .evaluate_batch(black_box(&input), black_box(&mut values[mode]), rows)
                            .unwrap();
                    }
                    let elapsed = start.elapsed().as_nanos();
                    same(&values[mode], &values[0]);
                    let raw = serde_json::json!({"type":"timing","case":case.name(),"phase_width":width,"rows_per_lane":rows_per_lane,"rows":rows,"rounds":rounds,"repeat":repeat,"position":position,"mode":mode,"block_rows":BLOCKS[mode],"elapsed_ns":elapsed,"below_20ms":elapsed<20_000_000});
                    write_record(&mut file, &raw);
                    blocks.push(raw);
                }
            }
            let workspaces: Vec<_> = variants.iter().map(|v| { let (capacity,cells)=v.native.as_ref().unwrap().probe_workspace().unwrap(); serde_json::json!({"capacity":capacity,"complex_cells":cells,"bytes":cells*std::mem::size_of::<Complex<f64>>()}) }).collect();
            records.push(serde_json::json!({"case":case.name(),"phase_width":width,"rows_per_lane":rows_per_lane,"rows":rows,"inputs":columns,"outputs":outputs,"rounds":rounds,"blocks":blocks,"workspaces":workspaces}));
        }
    }
    let report = serde_json::json!({"scope":"unchanged native kernels, packing/arithmetic/unpacking; no callbacks, geometry coordinators or master execution","performance_acceptance":false,"workload":"synthetic repeated row-major matrices; distinct deterministic rows","block_modes":BLOCKS,"records":records});
    let after = benchmark_metadata();
    assert_eq!(
        metadata["source_hashes"], after["source_hashes"],
        "source changed during probe"
    );
    assert_eq!(
        metadata["dependencies"], after["dependencies"],
        "dependency revisions changed during probe"
    );
    assert_eq!(metadata["head"], after["head"], "HEAD changed during probe");
    assert_eq!(
        metadata["executable_sha256"], after["executable_sha256"],
        "executable changed during probe"
    );
    assert_eq!(
        metadata["status"], after["status"],
        "source status changed during probe"
    );
    write_record(
        &mut file,
        &serde_json::json!({"type":"complete","value":report,"metadata_after":after}),
    );
    println!("NATIVE_BLOCKING_BENCHMARK {root}");
}

#[test]
fn native_blocking_finite_heads_match_symbolica_oracle() {
    oracle_control(false);
}
fn oracle_control(supported: bool) {
    let p = Float::new(192);
    for case in cases() {
        let exact = case.exact();
        for dual in [false, true] {
            if dual && !case.has_dual() {
                continue;
            }
            let components = if dual { 5 } else { 1 };
            let (columns, outputs) = case.dimensions();
            let rows = 17;
            let input = inputs(&p, columns * components, rows);
            let zero = Complex::new(p.zero(), p.zero());
            let mut expected = vec![zero.clone(); rows * outputs * components];
            if dual {
                let prototype = HyperDual::from_values(shape(), vec![zero.clone(); 5]);
                let mut oracle = exact.compile_dual(&p, &prototype).unwrap();
                let arguments: Vec<_> = input
                    .chunks(5)
                    .map(|c| HyperDual::from_values(shape(), c.to_vec()))
                    .collect();
                let mut results = vec![prototype; rows * outputs];
                oracle
                    .evaluate_batch(&arguments, &mut results, rows)
                    .unwrap();
                expected = results.into_iter().flat_map(|v| v.values).collect();
            } else {
                exact
                    .compile(&p)
                    .unwrap()
                    .evaluate_batch(&input, &mut expected, rows)
                    .unwrap();
            }
            for block in BLOCKS {
                let mut actual = vec![zero.clone(); expected.len()];
                selected_kernel(case, &p, dual, block, supported)
                    .evaluate_batch(&input, &mut actual, rows)
                    .unwrap();
                for (actual, expected) in actual.iter().zip(&expected) {
                    assert!(actual.is_finite() && expected.is_finite());
                    let tolerance = Float::parse("1e-45", Some(192)).unwrap();
                    assert!(
                        (actual.clone() - expected).norm().re
                            < tolerance * (p.one() + expected.norm().re)
                    );
                }
            }
        }
    }
}

// A deterministic contraction control only. No actual scalar-master arithmetic.
struct FrozenProbeMasters<V>(V);
impl<V: FloatLike> ninjars::MasterProvider<V> for FrozenProbeMasters<V> {
    fn evaluate(
        &mut self,
        request: ninjars::MasterRequest<'_, V>,
        output: &mut [ninjars::Laurent<V>],
    ) -> Result<()> {
        request.validate(output.len())?;
        for result in output {
            *result = [self.0.one(), self.0.zero(), self.0.zero()];
        }
        Ok(())
    }
    fn evaluate_batch(
        &mut self,
        request: ninjars::MasterRequest<'_, V>,
        output: &mut [ninjars::Laurent<V>],
        statuses: &mut [ninjars::MasterStatus],
    ) -> Result<()> {
        BatchShape {
            rows: request.rows(),
            columns: 1,
        }
        .validate("probe master statuses", statuses.len())?;
        self.evaluate(request, output)?;
        statuses.fill(ninjars::MasterStatus::Success);
        Ok(())
    }
}
fn variables() -> [symbolica::atom::Atom; 5] {
    use symbolica::parse;
    [
        parse!("q0"),
        parse!("q1"),
        parse!("q2"),
        parse!("q3"),
        parse!("mu2"),
    ]
}
fn routing<V: FloatLike>(p: &V, n: usize, lanes: usize) -> (Vec<ninjars::FourMomentum<V>>, Vec<V>) {
    let routing = if n >= 4 {
        [
            [0, 0, 0, 0],
            [2, 1, 0, 0],
            [4, 0, 1, 0],
            [7, 0, 0, 1],
            [11, 1, 2, 1],
        ]
    } else {
        [
            [0, 0, 0, 0],
            [1, 1, 0, 0],
            [2, 0, 1, 0],
            [3, 0, 0, 1],
            [4, 2, 1, 0],
        ]
    };
    let shifts = (0..lanes)
        .flat_map(|_| {
            routing[..n]
                .iter()
                .map(|v| ninjars::FourMomentum::from(v.map(|x| p.from_i64(x))))
        })
        .collect();
    let masses = (0..lanes)
        .flat_map(|lane| (0..n).map(move |line| p.from_i64(100 + lane as i64 + line as i64)))
        .collect();
    (shifts, masses)
}
fn snapshot<V, K: ninjars::AmplitudeKernels<V>>(
    amplitude: &ninjars::Amplitude<V, K>,
) -> (Vec<String>, Vec<V>)
where
    V: FloatLike,
{
    let mut states = vec![format!("{:?}", amplitude.statuses())];
    let mut values = amplitude
        .results()
        .iter()
        .flatten()
        .cloned()
        .collect::<Vec<_>>();
    macro_rules! collect {
        ($stage:expr) => {
            for result in $stage {
                match result {
                    ninjars::CutOutcome::Stable(residue) => {
                        states.push(format!("{:?}:{}", residue.id, residue.coefficients.len()));
                        values.extend_from_slice(&residue.coefficients);
                    }
                    ninjars::CutOutcome::Unstable(failure) => states.push(format!("{failure:?}")),
                }
            }
        };
    }
    collect!(amplitude.triangles());
    collect!(amplitude.bubbles());
    collect!(amplitude.tadpoles());
    collect!(amplitude.boxes());
    (states, values)
}
#[test]
fn native_blocking_preserves_reducer_statuses_residues_and_retry() {
    ordinary_reducer_control(false);
}
fn ordinary_reducer_control(supported: bool) {
    use crate::{PolynomialNumerator, SymbolicaAmplitudeKernels, SymbolicaCuts, SymbolicaHighCuts};
    use symbolica::parse;
    for (n, rank, expression) in [
        (5, 4, parse!("q0^4+q1*mu2+3*mu2^2+q2*q3+2")),
        (2, 3, parse!("q0^3+q1*mu2+q2*q3+2")),
    ] {
        let mut numerator = PolynomialNumerator::new(expression, variables(), vec![], 0.).unwrap();
        let mut geometry = SymbolicaCuts::new(0.).unwrap();
        let mut high = SymbolicaHighCuts::new(0.).unwrap();
        let mut masters = FrozenProbeMasters(Complex::new(0., 0.));
        let mut variants: Vec<_> = BLOCKS
            .iter()
            .map(|&block| {
                with_block(if supported { None } else { block }, || {
                    let kernels = SymbolicaAmplitudeKernels::new(n, rank, 0.)
                        .unwrap()
                        .with_native_kernels();
                    let kernels = if supported {
                        let policy = block.map_or(crate::BatchExecutionPolicy::Whole, |rows| {
                            crate::BatchExecutionPolicy::FixedRows(
                                std::num::NonZeroUsize::new(rows).unwrap(),
                            )
                        });
                        if n == 5 {
                            if matches!(block, Some(8 | 32)) {
                                kernels
                                    .with_native_batch_execution(policy)
                                    .with_native_rank_specialized_corrections()
                                    .unwrap()
                            } else {
                                kernels
                                    .with_native_rank_specialized_corrections()
                                    .unwrap()
                                    .with_native_batch_execution(policy)
                            }
                        } else {
                            kernels.with_native_batch_execution(policy)
                        }
                    } else {
                        kernels
                    };
                    kernels.into_amplitude(0).unwrap()
                })
            })
            .collect();
        for (lanes, bad) in [
            (1, false),
            (3, true),
            (3, false),
            (0, false),
            (9, false),
            (1, false),
        ] {
            numerator.set_parameter_batch(&[], lanes).unwrap();
            let (shifts, mut masses) = routing(&Complex::new(0., 0.), n, lanes);
            if bad {
                masses[n].re = f64::NAN;
            }
            let mut expected: Option<(Vec<String>, Vec<Complex<f64>>)> = None;
            for amplitude in &mut variants {
                amplitude.reset_batch(lanes);
                amplitude
                    .evaluate_batch_with_high_cuts(
                        &shifts,
                        &masses,
                        &vec![Complex::new(1., 0.); lanes],
                        &vec![Complex::new(1., 0.); lanes],
                        &mut numerator,
                        &mut geometry,
                        &mut high,
                        &mut masters,
                    )
                    .unwrap();
                let actual = snapshot(amplitude);
                if let Some((states, values)) = &expected {
                    assert_eq!(&actual.0, states);
                    same(&actual.1, values);
                } else {
                    expected = Some(actual);
                }
                for (lane, status) in amplitude.statuses().iter().enumerate() {
                    assert_eq!(
                        *status == ninjars::EvaluationStatus::Success,
                        !(bad && lane == 1),
                        "n={n} rank={rank} lanes={lanes} lane={lane} status={status:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn native_blocking_full_dual_reducer_and_failed_mixed_component() {
    dual_reducer_control(false);
}
fn dual_reducer_control(supported: bool) {
    dual_reducer_shape_control(supported, shape());
}
fn dual_reducer_shape_control(supported: bool, dual_shape: Vec<Vec<usize>>) {
    use crate::{
        DualPolynomialNumerator, PolynomialNumerator, SymbolicaDualAmplitudeKernels,
        SymbolicaDualCuts,
    };
    use symbolica::parse;
    let zero = HyperDual::from_values(dual_shape, vec![Complex::new(0., 0.); 5]);
    let source = PolynomialNumerator::new(
        parse!("a*(q0^4+q1*mu2+3*mu2^2+q2*q3+2)"),
        variables(),
        vec![parse!("a")],
        0.,
    )
    .unwrap();
    let mut numerator = DualPolynomialNumerator::new(source, zero.clone()).unwrap();
    let mut geometry = SymbolicaDualCuts::new(0., zero.clone()).unwrap();
    let mut masters = FrozenProbeMasters(zero.clone());
    let mut variants: Vec<_> = BLOCKS
        .iter()
        .map(|&block| {
            with_block(if supported { None } else { block }, || {
                let kernels = SymbolicaDualAmplitudeKernels::new(3, 4, 0., &zero)
                    .unwrap()
                    .with_native_kernels();
                let kernels = if supported {
                    kernels.with_native_batch_execution(block.map_or(
                        crate::BatchExecutionPolicy::Whole,
                        |rows| {
                            crate::BatchExecutionPolicy::FixedRows(
                                std::num::NonZeroUsize::new(rows).unwrap(),
                            )
                        },
                    ))
                } else {
                    kernels
                };
                kernels.into_amplitude(0).unwrap()
            })
        })
        .collect();
    for (lanes, bad) in [(3, false), (3, true), (9, false), (0, false), (1, false)] {
        let mut parameter = zero.one();
        parameter.values[1] = Complex::new(1., 0.);
        parameter.values[2] = Complex::new(0., 1.);
        numerator
            .set_parameter_batch(&vec![parameter; lanes], lanes)
            .unwrap();
        let (shifts, mut masses) = routing(&zero, 3, lanes);
        for mass in &mut masses {
            mass.values[1] = Complex::new(1., 0.);
            mass.values[4] = Complex::new(0.25, 0.);
        }
        if bad {
            masses[3].values[4].im = f64::NAN;
        }
        let mut expected: Option<(Vec<String>, Vec<Complex<f64>>)> = None;
        for amplitude in &mut variants {
            amplitude.reset_batch(lanes);
            amplitude
                .evaluate_batch(
                    &shifts,
                    &masses,
                    &vec![zero.one(); lanes],
                    &mut numerator,
                    &mut geometry,
                    &mut masters,
                )
                .unwrap();
            let (states, values) = snapshot(amplitude);
            let values: Vec<_> = values
                .iter()
                .flat_map(|v| v.values().iter().copied())
                .collect();
            if let Some((s, v)) = &expected {
                assert_eq!(&states, s);
                same(&values, v);
            } else {
                expected = Some((states, values));
            }
            for (lane, status) in amplitude.statuses().iter().enumerate() {
                assert_eq!(
                    *status == ninjars::EvaluationStatus::Success,
                    !(bad && lane == 1)
                );
            }
        }
    }
}

#[path = "native_blocking_fixture/mod.rs"]
mod lazy_fixture;
#[path = "native_batch_policy_tests.rs"]
mod supported_policy_tests;
#[test]
fn native_blocking_preserves_lazy_branches_and_shared_helper_calls() {
    lazy_fixture::check(0f64);
    lazy_fixture::check(DoubleFloat::default());
    lazy_fixture::check(Float::new(192));
}

fn write_record(file: &mut std::fs::File, record: &serde_json::Value) {
    use std::io::Write;
    serde_json::to_writer(&mut *file, record).unwrap();
    file.write_all(b"\n").unwrap();
    file.flush().unwrap();
}
fn benchmark_metadata() -> serde_json::Value {
    use std::{path::Path, process::Command};
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let command = |program: &str, args: &[&str]| {
        let output = Command::new(program)
            .args(args)
            .current_dir(&root)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "metadata command failed: {program} {args:?}"
        );
        String::from_utf8(output.stdout).unwrap().trim().to_owned()
    };
    let mut files = vec![
        "Cargo.lock".to_string(),
        "crates/ninjars-symbolica/src/kernel.rs".to_string(),
        "crates/ninjars-symbolica/src/native_blocking_probe.rs".to_string(),
    ];
    for directory in [
        "crates/ninjars-symbolica/src/native_kernels",
        "crates/ninjars-symbolica/src/native_kernels/generated",
        "crates/ninjars-symbolica/src/native_blocking_fixture",
    ] {
        for entry in std::fs::read_dir(root.join(directory)).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_some_and(|x| x == "rs") {
                files.push(
                    path.strip_prefix(&root)
                        .unwrap()
                        .to_str()
                        .unwrap()
                        .to_owned(),
                );
            }
        }
    }
    files.sort();
    let source_hashes = command(
        "sha256sum",
        &files.iter().map(String::as_str).collect::<Vec<_>>(),
    );
    let dependencies:[serde_json::Value;3]=["references/symbolica","references/oneloopmaster","references/gammaloop_ninja_shared"].map(|path| {
        serde_json::json!({"path":path,"revision":command("git",&["-C",path,"rev-parse","HEAD"]),"status":command("git",&["-C",path,"status","--porcelain"])})
    });
    let executable = std::env::current_exe().unwrap();
    serde_json::json!({"head":command("git",&["rev-parse","HEAD"]),"status":command("git",&["status","--porcelain"]),"source_hashes":source_hashes,"dependencies":dependencies,
        "executable":executable,"executable_sha256":command("sha256sum",&[executable.to_str().unwrap()]),"compiler":command("rustc",&["-Vv"]),
        "host":command("uname",&["-a"]),"affinity":std::fs::read_to_string("/proc/self/status").unwrap().lines().filter(|line|line.starts_with("Cpus_allowed")||line.starts_with("Mems_allowed")).collect::<Vec<_>>(),
        "cpuinfo":std::fs::read_to_string("/proc/cpuinfo").unwrap(),"master_computations":0,"quiet_window_acknowledged":std::env::var("NINJARS_BLOCKING_QUIET_WINDOW").as_deref()==Ok("approved"),"performance_acceptance":false})
}

#[test]
fn native_blocking_archive_metadata_and_jsonl_are_readable() {
    let metadata = benchmark_metadata();
    assert_eq!(metadata["head"].as_str().unwrap().len(), 40);
    assert!(!metadata["source_hashes"].as_str().unwrap().is_empty());
    assert_eq!(metadata["dependencies"].as_array().unwrap().len(), 3);
    let path = std::env::temp_dir().join(format!(
        "ninjars-blocking-metadata-{}.jsonl",
        std::process::id()
    ));
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .unwrap();
    write_record(
        &mut file,
        &serde_json::json!({"type":"metadata","value":metadata}),
    );
    write_record(
        &mut file,
        &serde_json::json!({"type":"control_complete","timing":false}),
    );
    let source = std::fs::read_to_string(&path).unwrap();
    let records: Vec<serde_json::Value> = source
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(records.len(), 2);
    assert_eq!(records[0]["value"], metadata);
    assert_eq!(records[1]["timing"], false);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn supported_batch_policy_matches_native_values_and_oracle() {
    exercise(0f64, true);
    exercise(DoubleFloat::default(), true);
    exercise(Float::new(192), true);
    oracle_control(true);
}

#[test]
fn supported_batch_policy_preserves_exceptional_and_mixed_precision_values() {
    exceptional_control(true);
    mixed_precision_control(true);
}

#[test]
fn supported_batch_policy_preserves_ordinary_and_dual_reducer_controls() {
    ordinary_reducer_control(true);
    dual_reducer_control(true);
    dual_reducer_shape_control(
        true,
        vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![2, 0], vec![1, 1]],
    );
}

#[test]
fn supported_batch_policy_preserves_lazy_shared_helpers() {
    lazy_fixture::check_policy(0f64, true);
    lazy_fixture::check_policy(DoubleFloat::default(), true);
    lazy_fixture::check_policy(Float::new(192), true);
}
