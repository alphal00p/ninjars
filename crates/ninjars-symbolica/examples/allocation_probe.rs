//! Diagnostic allocation counts for warmed, complete f64 reductions.
//!
//! Run with `cargo run --release -p ninjars-symbolica --example allocation_probe`.
//! Counts include reset and evaluation, but exclude preparation, warm-up, checks,
//! and reporting. They are allocation evidence, not a throughput acceptance test.

use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};

use ninjars::{
    BoxCoefficientMode, BoxMuMode, BoxReductionOptions, Complex, EvaluationStatus, FourMomentum,
    Numerator,
};
use ninjars_symbolica::{
    PolynomialNumerator, SymbolicaAmplitudeKernels, SymbolicaCuts, SymbolicaHighCuts,
    SymbolicaMasters,
};
use symbolica::{atom::Atom, parse};

struct CountingAllocator;
static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static REALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

// Only this diagnostic executable wraps allocation. Every request is forwarded
// unchanged to System; production library crates continue to forbid unsafe code.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            REALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.realloc(pointer, layout, size) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn complex(value: f64) -> Complex<f64> {
    Complex::new(value, 0.)
}

fn probe(n: usize, lanes: usize) {
    let variables: [Atom; 5] = [
        parse!("q0"),
        parse!("q1"),
        parse!("q2"),
        parse!("q3"),
        parse!("mu2"),
    ];
    let expression = match n {
        2 => parse!("(q0-q1)^3+mu2"),
        4 => parse!("(q0-q1)^5+mu2^2*q0"),
        5 => parse!("(q0-q1)^6+mu2^2*q0^2"),
        _ => unreachable!(),
    };
    let mut numerator = PolynomialNumerator::new(expression, variables, vec![], 0.).unwrap();
    numerator.set_parameter_batch(&[], lanes).unwrap();
    let mut amplitude = SymbolicaAmplitudeKernels::new(n, numerator.rank(), 0.)
        .unwrap()
        .with_box_options(BoxReductionOptions {
            coefficients: BoxCoefficientMode::Full,
            mu_extraction: BoxMuMode::Sampling,
        })
        .into_amplitude(lanes)
        .unwrap();
    let points = [
        [0., 0., 0., 0.],
        [2., 1., 0., 0.],
        [3., 0., 1., 0.],
        [4., 0., 0., 1.],
        [5., 2., 1., -1.],
    ];
    let shifts: Vec<_> = (0..lanes)
        .flat_map(|_| {
            points[..n]
                .iter()
                .map(|p| FourMomentum::from(p.map(complex)))
        })
        .collect();
    let masses = vec![complex(100.); n * lanes];
    let scales = vec![complex(1.); lanes];
    let sampling_scales = vec![complex(100.); lanes];
    let mut low = SymbolicaCuts::new(0.).unwrap();
    let mut high = SymbolicaHighCuts::new(0.).unwrap();
    let mut masters = SymbolicaMasters::new(0.);
    for iteration in 0..7 {
        ALLOCATIONS.store(0, Ordering::Relaxed);
        REALLOCATIONS.store(0, Ordering::Relaxed);
        COUNTING.store(iteration >= 2, Ordering::Relaxed);
        amplitude.reset_batch(lanes);
        let result = amplitude.evaluate_batch_with_high_cuts(
            &shifts,
            &masses,
            &scales,
            &sampling_scales,
            &mut numerator,
            &mut low,
            &mut high,
            &mut masters,
        );
        COUNTING.store(false, Ordering::Relaxed);
        let allocations = ALLOCATIONS.load(Ordering::Relaxed);
        let reallocations = REALLOCATIONS.load(Ordering::Relaxed);
        result.unwrap();
        assert!(
            amplitude
                .statuses()
                .iter()
                .all(|s| *s == EvaluationStatus::Success)
        );
        if iteration >= 2 {
            println!(
                "{{\"propagators\":{n},\"rank\":{},\"lanes\":{lanes},\"iteration\":{},\"allocations\":{allocations},\"reallocations\":{reallocations}}}",
                numerator.rank(),
                iteration - 2
            );
        }
    }
}

fn main() {
    std::thread::Builder::new()
        .stack_size(128 * 1024 * 1024)
        .spawn(|| {
            for n in [2, 4, 5] {
                for lanes in [1, 8] {
                    probe(n, lanes);
                }
            }
        })
        .unwrap()
        .join()
        .unwrap();
}
