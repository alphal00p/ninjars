//! Exercise the package key in a fresh process, independently of a developer's
//! personal license. The assertions outside the scope also catch accidentally
//! switching to an application-wide key or leaking a worker guard.

use std::process::Command;

use ninjars::licensing::unlock_for_ninjars_operation;
use symbolica::license::LicenseManager;

#[test]
fn scoped_license_without_personal_key() {
    const CHILD: &str = "NINJARS_TEST_LIBRARY_LICENSE_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "scoped_license_without_personal_key",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("SYMBOLICA_HIDE_BANNER", "1")
            .env_remove("SYMBOLICA_LICENSE")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "fresh-process license checks failed:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }

    assert!(std::env::var_os("SYMBOLICA_LICENSE").is_none());
    assert!(!LicenseManager::is_licensed());
    {
        let _outer = unlock_for_ninjars_operation();
        assert!(
            LicenseManager::is_licensed(),
            "NinjaRS package key rejected"
        );
        assert_eq!(LicenseManager::max_threads(2), 2);
        {
            let _inner = unlock_for_ninjars_operation();
            assert!(LicenseManager::is_licensed());
            let _symbol = symbolica::symbol!("ninjars_license_nested");
        }
        assert!(LicenseManager::is_licensed());
    }
    assert!(!LicenseManager::is_licensed(), "scope leaked into caller");
    assert_eq!(LicenseManager::max_threads(2), 1);

    let panic = std::panic::catch_unwind(|| {
        let _scope = unlock_for_ninjars_operation();
        assert!(LicenseManager::is_licensed());
        panic!("test unwinding a NinjaRS operation");
    });
    assert!(panic.is_err());
    assert!(!LicenseManager::is_licensed(), "panic leaked license scope");

    // Both workers enter Symbolica simultaneously. Without their own scoped
    // package guards the restricted single-thread permit would abort a worker.
    let barrier = std::sync::Barrier::new(3);
    std::thread::scope(|threads| {
        for _ in 0..2 {
            let barrier = &barrier;
            threads.spawn(move || {
                {
                    let _scope = unlock_for_ninjars_operation();
                    barrier.wait();
                    assert!(LicenseManager::is_licensed());
                    let _symbol = symbolica::symbol!("ninjars_license_worker");
                }
                assert!(!LicenseManager::is_licensed(), "worker scope leaked");
            });
        }
        barrier.wait();
        assert!(!LicenseManager::is_licensed(), "worker licensed its caller");
    });
    assert!(!LicenseManager::is_licensed());
}
