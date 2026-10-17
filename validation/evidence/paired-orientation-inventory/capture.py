import hashlib
import json
import pathlib
import subprocess
import sys

directory = pathlib.Path(__file__).resolve().parent
baseline = json.loads((directory / "before.json").read_text())
root = pathlib.Path(baseline["worktree"])

def git(path, *arguments):
    return subprocess.check_output(["git", "-C", str(path), *arguments], text=True).strip()

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

files = git(root, "ls-files", "--", "Cargo.toml", "Cargo.lock",
            "rust-toolchain.toml", ".cargo", "crates").splitlines()
source = {name: digest(root / name) for name in files if (root / name).is_file()}
assert source == baseline["source_sha256"], "Runtime sources changed"
subprocess.run(["git", "-C", str(root), "diff", "--exit-code",
                baseline["source_revision"], "--", "Cargo.toml", "Cargo.lock", "crates"],
               check=True)
dependencies = {}
for name, expected in baseline["dependencies"].items():
    path = (root / "references" / name).resolve()
    current = {"path": str(path), "revision": git(path, "rev-parse", "HEAD"),
               "status": git(path, "status", "--porcelain"),
               "submodules": git(path, "submodule", "status", "--recursive")}
    assert current == expected, (name, current, expected)
    dependencies[name] = current
record = {"runtime_sources_unchanged": True, "dependency_identities_unchanged": True,
          "head": git(root, "rev-parse", "HEAD"),
          "status": git(root, "status", "--porcelain"),
          "source_sha256": source, "dependencies": dependencies}
if (directory / "ready.json").exists():
    ready = json.loads((directory / "ready.json").read_text())
    for name in ("original_executable", "retained_executable"):
        path = pathlib.Path(ready[name])
        actual = digest(path)
        assert actual == ready["executable_sha256"], (name, actual)
    record["test_executable_unchanged"] = True
(directory / f"{sys.argv[1]}.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps({key: value for key, value in record.items()
                  if key not in ("source_sha256", "dependencies")}))
