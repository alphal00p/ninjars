"""Shared helpers for the NinjaRS benchmark-recording scripts in this directory."""
import hashlib
import os
import pathlib
import platform
import statistics
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
REPRESENTATIVES = [
    ("hgg", "top-triangle-0"),
    ("four-photon", "top-box-0"),
    ("gg-hh", "top-triangle-0"),
    ("gg-hh", "top-box-2"),
    ("gg-hhh", "top-triangle-0"),
    ("gg-hhh", "top-box-8"),
    ("gg-hhh", "top-pentagon-26"),
]
BUILD_COMMAND = "cargo build --locked --profile perf -p ninjars-cli --features numerator-jit"


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def git():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                    cwd=ROOT, text=True).strip()
    return {"commit": commit, "tracked_changes": bool(dirty)}


def host(cpu):
    model = next((line.split(":", 1)[1].strip() for line in
                  pathlib.Path("/proc/cpuinfo").read_text().splitlines()
                  if line.startswith("model name")), platform.processor())
    return {"cpu_model": model, "logical_cpus": os.cpu_count(), "pinned_cpu": cpu,
            "load_average_at_start": os.getloadavg(), "kernel": platform.release()}


def idle_fractions(interval=0.5):
    def sample():
        rows = {}
        for line in pathlib.Path("/proc/stat").read_text().splitlines():
            name, *fields = line.split()
            if name.startswith("cpu") and name != "cpu":
                values = list(map(int, fields))
                rows[int(name[3:])] = (values[3] + values[4], sum(values))
        return rows
    first = sample()
    time.sleep(interval)
    second = sample()
    return {cpu: (second[cpu][0] - first[cpu][0]) / max(1, second[cpu][1] - first[cpu][1])
            for cpu in second if cpu in first}


def pick_cpu():
    """An allowed CPU that was idle over a short sample (highest index first)."""
    allowed = sorted(os.sched_getaffinity(0), reverse=True)
    idle = idle_fractions()
    return max(allowed, key=lambda cpu: (idle.get(cpu, 0.0), cpu))


def median(values):
    return statistics.median(values)


def run_pinned(cpu, arguments):
    """Run the CLI pinned to one CPU; returns wall-clock seconds."""
    start = time.perf_counter()
    subprocess.run(["taskset", "-c", str(cpu), *map(str, arguments)], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL)
    return time.perf_counter() - start
