//! Exercise public adapter boundaries in a process with no personal license.
//! In particular, two caller-owned threads must be able to construct symbolic
//! kernels and lazily prepare oneloopmaster evaluators independently.

use ninjars::{Complex, Float, MasterIntegral, MasterProvider, MasterRequest, Numerator};
use ninjars_symbolica::{
    OneLoopNativeMasters, PolynomialNumerator, SymbolicaAmplitudeKernels, SymbolicaCuts,
    SymbolicaHighCuts, SymbolicaMasters,
};
use std::{process::Command, sync::Barrier};
use symbolica::{domains::float::RealLike, license::LicenseManager, parse};

const CHILD: &str = "NINJARS_TEST_ADAPTER_LICENSE_CHILD";

fn caller_is_unlicensed() {
    assert!(
        !LicenseManager::is_licensed(),
        "a NinjaRS operation left the caller's Symbolica scope unlocked"
    );
}

fn adapter_operations() {
    caller_is_unlicensed();
    let _cuts = SymbolicaCuts::new(Float::new(128)).unwrap();
    caller_is_unlicensed();
    let _high_cuts = SymbolicaHighCuts::new(Float::new(128)).unwrap();
    caller_is_unlicensed();
    let _kernels = SymbolicaAmplitudeKernels::new(1, 0, Float::new(128)).unwrap();
    caller_is_unlicensed();
    let _native = OneLoopNativeMasters::new().unwrap();
    caller_is_unlicensed();

    // Preparing caller-authored symbolic inputs is separate from the public
    // adapter operation under test. Release the fixture's guard before giving
    // these inputs to the adapter, which must acquire its own scope.
    let (expression, variables) = {
        let _fixture = ninjars::licensing::unlock_for_ninjars_operation();
        (
            parse!("q0^2+mu2"),
            [
                parse!("q0"),
                parse!("q1"),
                parse!("q2"),
                parse!("q3"),
                parse!("mu2"),
            ],
        )
    };
    caller_is_unlicensed();
    let numerator = PolynomialNumerator::new(expression, variables, vec![], 0.).unwrap();
    assert_eq!(numerator.rank(), 2);
    caller_is_unlicensed();

    let mut masters = SymbolicaMasters::new(Float::new(128));
    caller_is_unlicensed();
    masters.prepare(MasterIntegral::B0, 1).unwrap();
    caller_is_unlicensed();
    let real = |x| Complex::new(Float::with_val(128, x), Float::new(128));
    let mut output = [[real(0.), real(0.), real(0.)]];
    masters
        .evaluate(
            MasterRequest {
                integral: MasterIntegral::B0,
                invariants: &[real(-1.)],
                masses_squared: &[real(1.), real(1.)],
                scales_squared: &[real(1.)],
            },
            &mut output,
        )
        .unwrap();
    caller_is_unlicensed();
    // Massive spacelike B0, independently retained original Ninja/OneLOop
    // reference also used by tests/masters.rs.
    for (value, expected) in output[0].iter().zip([-0.152_044_704_820_020_3, 1., 0.]) {
        assert!((value.re.to_f64() - expected).abs() < 1.0e-14);
        assert!(value.im.to_f64().abs() < 1.0e-14);
    }
}

#[test]
fn adapters_without_personal_license() {
    if std::env::var_os(CHILD).is_none() {
        let child = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "adapters_without_personal_license",
                "--nocapture",
            ])
            .env_remove("SYMBOLICA_LICENSE")
            .env(CHILD, "1")
            .output()
            .unwrap();
        assert!(
            child.status.success(),
            "adapter license subprocess failed: {}\n{}",
            String::from_utf8_lossy(&child.stdout),
            String::from_utf8_lossy(&child.stderr)
        );
        return;
    }
    assert!(std::env::var_os("SYMBOLICA_LICENSE").is_none());
    caller_is_unlicensed();
    {
        let _fixture = ninjars::licensing::unlock_for_ninjars_operation();
        assert!(LicenseManager::is_licensed());
    }
    caller_is_unlicensed();
    let barrier = Barrier::new(2);
    std::thread::scope(|scope| {
        let mut workers = Vec::new();
        for _ in 0..2 {
            let barrier = &barrier;
            workers.push(
                std::thread::Builder::new()
                    .stack_size(128 * 1024 * 1024)
                    .spawn_scoped(scope, move || {
                        barrier.wait();
                        adapter_operations();
                    })
                    .unwrap(),
            );
        }
        for worker in workers {
            worker.join().unwrap();
        }
    });
    caller_is_unlicensed();
}
