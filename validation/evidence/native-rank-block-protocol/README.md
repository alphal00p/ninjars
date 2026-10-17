# Preselected native rank/block timing protocol

This checkpoint prepares the strict, build-free driver for frozen source
`8b868220f3a438a7f78e0e5329cd011646fef71c`. It is not timing evidence.
The twelve pure Python regressions use explicitly fabricated wire fixtures;
their durations and union size are not physical results.

The preflight requires a reviewed release-manifest hash, the exact runtime
sources, dependencies, copied shared library and input, the correct loader
target, and matching release controls for every selected mode and bound. The
four settings are native/Whole, native/fixed32, native-rank/Whole, and
native-rank/fixed32. Four cyclic process-order sweeps cover widths
1/8/31/32/128/1024 with eight balanced original/Rust pairs per record. Round
counts are fixed before timing. Every raw block is retained; any block shorter
than 20ms makes the complete run a pilot, without selective tuning or reruns.

Only shared-provider reduction is timed by this driver. Physical preparation
and actual master execution are excluded, with complete callback, preparation
and backend counters checked. Numerator callback rows are shared; output-head
work differs from original Ninja. Rank guards and native block invocations are
not counted by the existing instrumentation. Cross-setting comparisons retain
four process ratios and median/MAD, separate from per-record paired intervals.
The host is shared and these measurements cannot establish performance acceptance.

The frozen worktree remains outside this evidence directory. To reproduce the
protocol tests on this host, copy `ninjars-rank-block-sampling.py` to `/tmp` and
run `test-ninjars-rank-block-sampling.py` with Python. Running the driver without
`--sample` checks provenance only. Timing additionally requires the reviewed
release manifest and a coordinated window with this team's builds/tests idle.
Separate master/preparation execution accounting and complete evaluations
remain required.
