//! Direct graph preparation must scope its own Symbolica use: callers of this
//! crate do not necessarily enter through the NinjaRS command-line wrapper.

use std::{process::Command, sync::Arc};

use ninjars_graph::{
    Kinematics, Model, NumeratorMode, PolynomialCoefficientPlan, has_levi_civita,
    has_spinor_chains, prepare_dot,
};
use symbolica::{domains::float::Complex, license::LicenseManager, parse};

#[test]
fn direct_graph_operations_are_scoped_without_personal_key() {
    const CHILD: &str = "NINJARS_GRAPH_TEST_LIBRARY_LICENSE_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "direct_graph_operations_are_scoped_without_personal_key",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("SYMBOLICA_HIDE_BANNER", "1")
            .env_remove("SYMBOLICA_LICENSE")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "fresh-process graph checks failed:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }

    assert!(std::env::var_os("SYMBOLICA_LICENSE").is_none());
    assert!(!LicenseManager::is_licensed());
    // Construct test inputs under a separate fixture scope. Symbolica calls
    // made directly by a library's caller are not automatically authorized.
    let (model, kinematics, expression, variables, parameters) = {
        let _fixture = ninjars::licensing::unlock_for_ninjars_operation();
        assert!(LicenseManager::is_licensed());
        (
            Arc::new(Model::phi3()),
            Kinematics::new(),
            parse!("(a+b)*q0"),
            [
                parse!("q0"),
                parse!("q1"),
                parse!("q2"),
                parse!("q3"),
                parse!("mu2"),
            ],
            vec![parse!("a"), parse!("b")],
        )
    };
    assert!(!LicenseManager::is_licensed());

    // Keep both workers alive until both are done. A missed internal scope
    // would leave one worker holding the restricted permit and abort the other.
    let barrier = std::sync::Barrier::new(3);
    std::thread::scope(|threads| {
        for _ in 0..2 {
            let model = model.clone();
            let (kinematics, expression, variables, parameters, barrier) =
                (&kinematics, &expression, &variables, &parameters, &barrier);
            threads.spawn(move || {
                barrier.wait();
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    assert!(!LicenseManager::is_licensed());
                    let prepared = prepare_dot(
                        model,
                        include_str!("../../../examples/graphs/bubble.dot"),
                        kinematics,
                        NumeratorMode::Explicit,
                    )
                    .unwrap();
                    assert!(!LicenseManager::is_licensed());
                    assert_eq!(prepared.quadratics.len(), 2);
                    assert_eq!(prepared.squared_masses().len(), 2);
                    assert!(!has_spinor_chains(&prepared.numerator));
                    assert!(!has_levi_civita(&prepared.numerator));
                    let _factor = prepared.graph_factors.factor();
                    assert!(!LicenseManager::is_licensed());

                    let plan = PolynomialCoefficientPlan::new(
                        expression.clone(),
                        (*variables).clone(),
                        parameters.to_vec(),
                    )
                    .unwrap();
                    assert!(!LicenseManager::is_licensed());
                    let plan = plan.factor_coefficients().unwrap();
                    assert!(!LicenseManager::is_licensed());
                    assert_eq!(plan.rank(), 1);
                    assert_eq!(plan.coefficient_expressions().len(), 1);
                    let mut evaluator = plan.compile(0_f64).unwrap();
                    assert!(!LicenseManager::is_licensed());
                    let mut output = [Complex::new(0., 0.)];
                    evaluator
                        .evaluate_batch(
                            &[Complex::new(3., 0.), Complex::new(4., 0.)],
                            &mut output,
                            1,
                        )
                        .unwrap();
                    assert_eq!(output, [Complex::new(7., 0.)]);
                    assert!(!LicenseManager::is_licensed());
                }));
                barrier.wait();
                if let Err(panic) = result {
                    std::panic::resume_unwind(panic);
                }
            });
        }
        barrier.wait();
        assert!(!LicenseManager::is_licensed());
        barrier.wait();
    });
    assert!(!LicenseManager::is_licensed());
}
