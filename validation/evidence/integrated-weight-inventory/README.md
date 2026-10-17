# Integrated master-weight inventory evidence

The single bounded ignored symbolic test passed at source
`ee0c8b6b77a2aa41a8c6feec0d53bd7a94cc3286`, based on `ea9a9da`.
All 90 exact identities passed; no source fix or repeated symbolic run was
needed. No numerical evaluator or performance benchmark ran.

- [inventory.json](inventory.json) records the original full/pre-master block
  inventories, 20-weight candidate, three actual 10/30/20 stages, explicit input
  and boundary layouts, all exact identities, operation counts, and byte caps.
- [manifest.json](manifest.json) records the 181 source hashes, clean dependency
  revisions, exact commands and successful statuses, retained executable
  identity, artifact hashes, and archive entries.
- [logs.tar.gz](logs.tar.gz) contains Cargo JSON/stderr, the complete symbolic
  log, strict Clippy/format logs, statuses, before-state capture, command and
  executable metadata, and the three frozen inventory source/document files.
  The large test executable remains at the immutable path in the manifest.

The separate pre-master baseline has 1,884 instructions. Weights-only uses
1,097; the actual staged sum is 19 + 753 + 299 = 1,071. All preserve the same
ordered master weights and rational heads. The staged input prefixes are
0/10/40; all 60 original callback inputs remain. These are symbolic instruction
counts, not runtime or precision acceptance.

Builds and strict checks used the existing `cd/5d62942031e3ee` cache with both
`CARGO_TARGET_DIR` and `CARGO_BUILD_BUILD_DIR` set explicitly, at most four build
jobs, and thread limits of one. The test executable was copied read-only and
hashed before execution. The ignored symbolic test ran once under
`/tmp/ninjars-symbolica-tests.lock`, with the 180-second timeout starting after
lock acquisition. The test's reported 1.14 seconds is symbolic-check elapsed
time only. All four command statuses were zero. Existing dependency warnings
remain in the logs; strict package Clippy used `--no-deps`.

The run has separate per-composition 8 MiB caps and one shared 8 MiB identity
budget. Candidate composition totals and the identity budget were below their
limits. Source/dependency/executable identities were checked again before the
result documentation was updated. The frozen source document in the archive
therefore retains its original pre-execution status; the current
[audit note](../../../docs/audits/integrated-weight-fusion.md) records the result
and the still-unvalidated runtime contracts.
