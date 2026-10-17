import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

directory = Path(__file__).resolve().parent
name, *command = sys.argv[1:]
metadata = directory / f"{name}.command.json"
stdout_path = directory / f"{name}.stdout"
stderr_path = directory / f"{name}.stderr"
output_path = os.environ.get("NINJARS_PAIRED_ORIENTATION_OUTPUT")
reserved = [metadata.resolve(), stdout_path.resolve(), stderr_path.resolve()]
if output_path is not None:
    reserved.append(Path(output_path).resolve())
assert len(set(reserved)) == len(reserved), \
    "Command evidence paths must not overwrite the symbolic report"
assert not any(path.exists() for path in reserved), \
    "Command evidence already exists; choose a new attempt label"
record = {
    "command": command,
    "cwd": os.getcwd(),
    "start_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    "environment": {
        key: os.environ.get(key)
        for key in (
            "CC", "CXX", "PKG_CONFIG", "CARGO_TARGET_DIR",
            "CARGO_BUILD_BUILD_DIR", "CARGO_BUILD_JOBS", "RUST_MIN_STACK",
            "RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "NINJARS_PAIRED_ORIENTATION_OUTPUT",
        )
    },
}
start = time.monotonic()
with stdout_path.open("w") as stdout:
    with stderr_path.open("w") as stderr:
        result = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
record["exit_code"] = result.returncode
record["elapsed_seconds"] = time.monotonic() - start
record["end_utc"] = datetime.datetime.now(datetime.UTC).isoformat()
metadata.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps({"check": name, "exit_code": result.returncode,
                  "elapsed_seconds": record["elapsed_seconds"]}), flush=True)
sys.exit(result.returncode)
