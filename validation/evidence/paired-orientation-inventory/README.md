# Paired-orientation symbolic inventory evidence

Frozen runtime source: `fda904779c47998b7bbe1b9d04ddc7bc48ad210d`; documentation
HEAD during validation: `3fc262d558e0c6651537ae6dc3d032bc9f3f6265`.
The [audit](../../../docs/audits/paired-orientation-inventory.md) defines the
independent denominator and source-projection boundaries and all input maps.

The retained `attempt2-inventory.json` reports **30 exact roundtrip identities**,
all **30 retained output heads**, **294→216 instructions**, and **358→258
scalar-carrier operations** across the two independent plans. Both merge-round
choices produce the same counts. It retains every original/rebound expression,
parameter map, head identity and individual inventory. `summary.json` only
extracts this report. No numerical execution, production behavior, or performance
acceptance follows from these symbolic counts.

Exactly **two symbolic executions** occurred. Attempt 1 passed, but the initial
wrapper's `inventory` label caused its command metadata to overwrite the symbolic
JSON after completion. Its raw output, argv/status and original wrapper are
preserved. `evidence-incident.json` records the defect and explicit authorization
for one replacement. The fixed wrapper uses distinct `.command.json` metadata,
checks all output paths for collisions/existing files before invocation, and
passed three rejection controls. Only the second report contributes counts;
the first attempt is not classified as a numerical failure. No runtime source,
executable, bound or test selection changed between attempts.

`before.json` records 180 source hashes, compiler version, clean dependency heads
and resolved paths. The three `after-*.json` records verify unchanged sources,
dependencies and original/retained test executables. `ready.json` contains the
exact Cargo artifact and read-only executable location/hash. The 483 MB binary
is retained at `/tmp/ninjars-paired-orientation/test-executable`, outside Git;
it can be rebuilt from the recorded frozen source and dependencies.

`build.*`, `format.*` and `clippy.*` preserve commands, environment, exit status
and raw logs. Focused formatting and strict Clippy passed. Dependency warnings
remain visible; dependencies were not edited. Compilation and checking elapsed
times are build evidence. Test-wrapper elapsed time includes numerical-lock
waiting and is not performance data. Both exact ignored invocations used
`flock` followed by the same 180-second timeout, with one test thread and the
unchanged 8 MiB expression budget.

`manifest.json` binds all raw evidence files and the retained executable. The
first-run logging defect, fixed wrapper and complete replacement evidence are
kept together to make the extra execution explicit.
