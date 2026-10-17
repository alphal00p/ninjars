# Factored Laurent division inventory evidence

The single bounded ignored symbolic test passed at source
`0988bbd38cb3f4789f0f00238c59e125a96e5cd5`, based on `c16e09b`.
All 16 final quotient identities and five layout-only guards passed. No
production kernel, numerical evaluator, or performance benchmark ran.

- [inventory.json](inventory.json) records exactly n6/r7/cut2 and n5/r4/cut2,
  their ordered power maps, local evaluator and repeated-call inventories,
  stage physical powers, exact checks, buffers and the shared byte budget.
- [manifest.json](manifest.json) records 182 source hashes, clean dependency
  revisions, successful command statuses, retained executable identity, the
  unchanged prior integrated-weight artifact, and archive/file hashes.
- [logs.tar.gz](logs.tar.gz) contains Cargo JSON/stderr, the complete symbolic
  log, strict Clippy/format logs, statuses, before-state capture, exact command
  and executable metadata, and the three frozen inventory source/doc files.
  The large executable remains read-only at its manifest path.

n6/r7 reproduces 712 instructions for the expanded product versus four local
118-instruction calls (472 total). n5/r4 is a negative control: 32 instructions
versus three local 12-instruction calls (36 total). Reusable body source and
unrolled repeated-call source are reported separately. No composed proof graph
is counted as a candidate evaluator, and no runtime benefit is claimed.

Builds and strict checks reused `cd/5d62942031e3ee` with both Cargo target/build
directories set explicitly, at most four build jobs and thread limits of one.
The copied executable was hashed and made read-only before its single ignored
run. The shared numerical lock was acquired before the external 180-second
timeout started. The reported 0.11 seconds is symbolic-check elapsed only.
The global 8 MiB expression budget consumed 69,512 bytes. No cap increase,
failure, source fix, or repeat run was needed. All four command statuses were
zero; existing dependency warnings are retained in the strict `--no-deps`
Clippy log.

Source/dependency/executable identities were checked again before updating the
result documentation. The archived frozen document retains its original
pre-run status. The current [audit note](../../../docs/audits/factored-laurent-inventory.md)
records the result and remaining numerical/runtime-contract limitations.
