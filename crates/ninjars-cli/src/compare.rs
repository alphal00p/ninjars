//! Validation orchestration; reference code is never linked into NinjaRS.

use std::{
    error::Error,
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

use clap::Args as ClapArgs;
use serde_json::{Value, json};

use crate::{prepare, reduce, reference};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const TOLERANCE: f64 = 1e-9;
static DIRECTORY_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Exclusive ownership of a directory's mutable artifacts. Failed acquisition
/// never creates a guard and therefore cannot remove another invocation's lock.
struct DirectoryLock {
    path: PathBuf,
}

impl DirectoryLock {
    fn acquire(directory: &Path, filename: &str, operation: &str) -> Result<Self> {
        fs::create_dir_all(directory)?;
        let path = directory.join(filename);
        let mut owner = OpenOptions::new().write(true).create_new(true).open(&path)
            .map_err(|error| if error.kind() == std::io::ErrorKind::AlreadyExists {
                format!("{operation} directory is locked: {}; inspect the lock owner before retrying", path.display())
            } else { format!("cannot lock {operation} directory {}: {error}", directory.display()) })?;
        let guard = Self { path };
        writeln!(owner, "pid={} operation={operation}", std::process::id())?;
        Ok(guard)
    }
}

impl Drop for DirectoryLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn unique_artifact_directory(base: &Path) -> Result<PathBuf> {
    fs::create_dir_all(base)?;
    let epoch = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    for _ in 0..64 {
        let sequence = DIRECTORY_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let directory = base.join(format!("run-{}-{epoch:x}-{sequence:x}", std::process::id()));
        match fs::create_dir(&directory) {
            Ok(()) => return Ok(directory),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }
    Err("could not allocate a unique comparison artifact directory".into())
}

#[derive(ClapArgs)]
#[group(id = "CompareArgs")]
pub struct Args {
    #[command(flatten)]
    pub reduction: reduce::Args,
    /// NinjaRS source checkout containing validation/reference/build.py.
    #[arg(long)]
    pub reference_root: Option<PathBuf>,
    /// Existing or newly built original Ninja/OneLOop libraries.
    #[arg(long)]
    pub reference_build_dir: Option<PathBuf>,
    /// Keep generated callbacks, runner inputs, logs and the comparison report here.
    #[arg(long)]
    pub reference_dir: Option<PathBuf>,
}

fn absolute(path: PathBuf) -> Result<PathBuf> {
    Ok(if path.is_absolute() {
        path
    } else {
        std::env::current_dir()?.join(path)
    })
}

fn build(
    root: &Path,
    directory: &Path,
    log: &Path,
    extra: &[&Path],
    runner_only: bool,
) -> Result<()> {
    // build.py also writes shared generated headers when linking a runner.
    // Protect both library builds and runner-only builds with the same lock.
    let _build_lock = DirectoryLock::acquire(directory, ".ninjars-build.lock", "reference build")?;
    let mut command = Command::new("python3");
    command
        .arg(root.join("validation/reference/build.py"))
        .arg("--build-dir")
        .arg(directory);
    if runner_only {
        command.arg("--runner-only");
    }
    if let [header, output] = extra {
        command
            .arg("--numerator-header")
            .arg(header)
            .arg("--output")
            .arg(output);
    }
    let result = command.output()?;
    let mut file = fs::File::create(log)?;
    file.write_all(&result.stdout)?;
    file.write_all(&result.stderr)?;
    if !result.status.success() {
        return Err(format!("reference build failed; see {}", log.display()).into());
    }
    Ok(())
}

fn pair(value: &Value, strings: bool) -> Option<[f64; 2]> {
    let parse = |part: usize| {
        let value = if strings {
            value[part].as_str()?.parse::<f64>().ok()?
        } else {
            value[part].as_f64()?
        };
        value.is_finite().then_some(value)
    };
    Some([parse(0)?, parse(1)?])
}

fn error(actual: [f64; 2], expected: [f64; 2]) -> f64 {
    // Finite components can have an overflowing complex norm. Scale before
    // subtraction and norms so that this cannot turn a mismatch into zero.
    let scale = actual
        .into_iter()
        .chain(expected)
        .map(f64::abs)
        .fold(1., f64::max);
    (actual[0] / scale - expected[0] / scale).hypot(actual[1] / scale - expected[1] / scale)
        / (1. / scale + (expected[0] / scale).hypot(expected[1] / scale))
}

fn compare_lane(actual: &Value, original: &Value) -> Value {
    let mut eligible = actual["status"]["kind"] == "success" && original["status"] == 0;
    let errors: Vec<_> = (0..3)
        .map(|coefficient| {
            let value = pair(&actual["values"][coefficient], true)
                .zip(pair(&original["epsilon"][coefficient], false))
                .map(|(actual, expected)| error(actual, expected));
            eligible &= value.is_some_and(f64::is_finite);
            value
        })
        .collect();
    let passed = eligible
        && errors
            .iter()
            .all(|value| value.is_some_and(|value| value <= TOLERANCE));
    let split_errors: Vec<_> = ["cut_constructible", "rational"]
        .into_iter()
        .enumerate()
        .map(|(index, key)| {
            pair(&actual["values"][3 + index], true)
                .zip(pair(&original[key], false))
                .map(|(actual, expected)| error(actual, expected))
        })
        .collect();
    json!({"lane":actual["lane"], "passed":passed, "eligible":eligible,
        "laurent_scaled_errors":errors, "split_scaled_errors":split_errors,
        "reference":original})
}

pub fn run(args: Args) -> Result<()> {
    let root = absolute(
        args.reference_root
            .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")),
    )?;
    let libraries = absolute(
        args.reference_build_dir
            .unwrap_or_else(|| root.join("target/reference")),
    )?;
    let directory = match args.reference_dir {
        Some(directory) => absolute(directory)?,
        None => unique_artifact_directory(&root.join("target/compare"))?,
    };
    let _artifact_lock =
        DirectoryLock::acquire(&directory, ".ninjars-compare.lock", "comparison artifact")?;
    let mut prepared = prepare::prepare(&prepare::Args {
        input: args.reduction.input.clone(),
        samples: args.reduction.samples.clone(),
        model: args.reduction.model.clone(),
        feynman_rules: args.reduction.feynman_rules,
        output: None,
        reference_dir: None,
        reference_bench_dir: None,
    })?;
    let manifest = reference::export(&directory, &prepared.samples, &prepared.scale_squared)?;
    let reduction_ok = reduce::evaluate_prepared(&mut prepared, &args.reduction)?;
    fs::write(
        directory.join("prepared.json"),
        serde_json::to_vec_pretty(&prepared.document)?,
    )?;
    if !libraries.join("ninja-build/src/.libs/libninja.a").exists()
        || !libraries.join("oneloop/libavh_olo.a").exists()
        || !libraries.join("revisions.json").exists()
    {
        build(
            &root,
            &libraries,
            &directory.join("libraries.log"),
            &[],
            false,
        )?;
    }
    let mut checks = Vec::new();
    if manifest["samples"].as_array().map(Vec::len)
        != prepared.document["results"].as_array().map(Vec::len)
    {
        return Err("reference and Rust sample counts differ".into());
    }
    for (sample, actual) in manifest["samples"]
        .as_array()
        .ok_or("invalid exported samples")?
        .iter()
        .zip(
            prepared.document["results"]
                .as_array()
                .ok_or("missing Rust results")?,
        )
    {
        let lane = sample["lane"].as_u64().ok_or("missing reference lane")?;
        if actual["lane"] != lane {
            return Err("reference lane ordering differs from Rust results".into());
        }
        let executable = directory.join(format!("reference-{lane:06}"));
        let header = directory.join(
            sample["header"]
                .as_str()
                .ok_or("missing reference header")?,
        );
        let input = directory.join(sample["input"].as_str().ok_or("missing reference input")?);
        build(
            &root,
            &libraries,
            &directory.join(format!("build-{lane:06}.log")),
            &[&header, &executable],
            true,
        )?;
        let mut command = Command::new(&executable);
        command
            .arg("--cut-stop")
            .arg(args.reduction.cut_stop.clamp(1, 6).to_string());
        if matches!(args.reduction.mu_extraction, reduce::MuExtraction::Sampling) {
            command.arg("--mu-sampling");
        }
        if let Some(matrix) = sample["s_matrix_input"].as_str() {
            command.arg("--s-matrix").arg(directory.join(matrix));
        }
        // One process per point also avoids upstream cache contamination. All
        // setup, compilation and IPC remain outside any performance claims.
        let output = command
            .stdin(Stdio::from(fs::File::open(input)?))
            .output()?;
        fs::write(
            directory.join(format!("reference-{lane:06}.stdout")),
            &output.stdout,
        )?;
        fs::write(
            directory.join(format!("reference-{lane:06}.stderr")),
            &output.stderr,
        )?;
        let records: Vec<_> = String::from_utf8_lossy(&output.stdout)
            .lines()
            .filter(|line| line.starts_with('{'))
            .map(serde_json::from_str::<Value>)
            .collect();
        let original = if output.status.success() && records.len() == 1 {
            records.into_iter().next().unwrap()?
        } else {
            json!({"status":"runner_failed", "exit_code":output.status.code()})
        };
        let mut check = compare_lane(actual, &original);
        check["artifact"] = serde_json::from_slice(&fs::read(executable.with_extension("json"))?)?;
        checks.push(check);
    }
    let success = reduction_ok && checks.iter().all(|check| check["passed"] == true);
    prepared.document["operation"] = "compare".into();
    prepared.document["success"] = success.into();
    prepared.document["reference_export"] = manifest;
    prepared.document["comparison"] = json!({"backend":"original Ninja with OneLOop",
        "reference_precision":"binary64", "scaled_tolerance":TOLERANCE,
        "error_definition":"complex absolute difference / (1 + reference complex magnitude)",
        "split_order":["cut_constructible_finite","rational"],
        "split_policy":"Reported separately; different cut bases can transfer terms between the rational and cut-constructible pieces. Acceptance compares every total Laurent coefficient.",
        "timings_are_acceptance_evidence":false, "reference_cache_policy":"fresh process per phase-space sample",
        "reference_sampling_scale_policy":"Original Ninja's native per-cut scale; --mu-sampling-scale controls only NinjaRS.",
        "reference_coefficient_mode":"integrated-only; --full-box-coefficients controls NinjaRS and does not enable original Ninja reconstruction tests",
        "artifact_directory":directory, "checks":checks});
    let report = serde_json::to_string_pretty(&prepared.document)? + "\n";
    fs::write(directory.join("comparison.json"), &report)?;
    if let Some(output) = &args.reduction.output {
        fs::write(output, report)?;
    } else {
        print!("{report}");
    }
    if !success {
        return Err("reference comparison failed; inspect per-lane comparison checks".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_directory() -> PathBuf {
        unique_artifact_directory(&std::env::temp_dir().join("ninjars-compare-tests")).unwrap()
    }

    #[test]
    fn artifact_locks_preserve_existing_owners_and_release_on_errors() {
        let directory = test_directory();
        let path = directory.join(".ninjars-compare.lock");
        let first = DirectoryLock::acquire(&directory, ".ninjars-compare.lock", "first").unwrap();
        let owner = fs::read(&path).unwrap();
        let second = DirectoryLock::acquire(&directory, ".ninjars-compare.lock", "second");
        assert!(
            second
                .err()
                .unwrap()
                .to_string()
                .contains("directory is locked")
        );
        assert_eq!(fs::read(&path).unwrap(), owner);
        drop(first);
        assert!(!path.exists());

        let failing_operation = || -> Result<()> {
            let _lock = DirectoryLock::acquire(&directory, ".ninjars-compare.lock", "retry")?;
            Err("deliberate operation failure".into())
        };
        assert!(failing_operation().is_err());
        assert!(!path.exists());
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn concurrent_default_artifact_directories_are_distinct() {
        let base = test_directory();
        let workers: Vec<_> = (0..8)
            .map(|_| {
                let base = base.clone();
                std::thread::spawn(move || unique_artifact_directory(&base).unwrap())
            })
            .collect();
        let mut directories: Vec<_> = workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect();
        assert!(
            directories
                .iter()
                .all(|directory| directory.is_dir() && directory.parent() == Some(base.as_path()))
        );
        directories.sort();
        directories.dedup();
        assert_eq!(directories.len(), 8);
        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn shared_reference_build_lock_rejects_concurrent_library_and_runner_builds() {
        let directory = test_directory();
        let lock =
            DirectoryLock::acquire(&directory, ".ninjars-build.lock", "existing build").unwrap();
        let path = directory.join(".ninjars-build.lock");
        let owner = fs::read(&path).unwrap();
        let log = directory.join("must-not-be-written.log");
        for runner_only in [false, true] {
            let result = build(&directory, &directory, &log, &[], runner_only);
            assert!(
                result
                    .unwrap_err()
                    .to_string()
                    .contains("reference build directory is locked")
            );
            assert!(!log.exists());
            assert_eq!(fs::read(&path).unwrap(), owner);
        }
        drop(lock);
        assert!(!path.exists());
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn finite_large_components_cannot_hide_reference_mismatches() {
        let expected = [1.3e308, 1.3e308];
        assert_eq!(error(expected, expected), 0.);
        assert!(error([expected[0] + 1e301, expected[1]], expected) > TOLERANCE);
        assert!(error([-expected[0], -expected[1]], expected) > 1.9);
        assert!(error([f64::MAX, 0.], [0., 0.]) > TOLERANCE);
    }
    #[test]
    fn comparison_rejects_failed_nonfinite_and_mismatched_coefficients() {
        let actual = json!({"lane":7,"status":{"kind":"success"},"values":[["2","3"],["1","0"],["0","0"],["2","3"],["0","0"]]});
        let mut original = json!({"status":0,"epsilon":[[2,3],[1,0],[0,0]],"cut_constructible":[2,3],"rational":[0,0]});
        assert_eq!(compare_lane(&actual, &original)["passed"], true);
        original["epsilon"][2] = json!([1e-7, 0]);
        assert_eq!(compare_lane(&actual, &original)["passed"], false);
        original["epsilon"][2] = Value::Null;
        assert_eq!(compare_lane(&actual, &original)["eligible"], false);
        original["epsilon"][2] = json!([0, 0]);
        original["status"] = 1.into();
        assert_eq!(compare_lane(&actual, &original)["passed"], false);
    }
}
