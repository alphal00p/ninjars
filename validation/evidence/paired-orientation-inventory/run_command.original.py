import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

directory = Path(__file__).resolve().parent
name, *command = sys.argv[1:]
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
with (directory / f"{name}.stdout").open("w") as stdout:
    with (directory / f"{name}.stderr").open("w") as stderr:
        result = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
record["exit_code"] = result.returncode
record["elapsed_seconds"] = time.monotonic() - start
record["end_utc"] = datetime.datetime.now(datetime.UTC).isoformat()
(directory / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps({"check": name, "exit_code": result.returncode,
                  "elapsed_seconds": record["elapsed_seconds"]}), flush=True)
sys.exit(result.returncode)
