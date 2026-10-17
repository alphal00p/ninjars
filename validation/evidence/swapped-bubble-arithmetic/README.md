# Swapped bubble arithmetic: bounded correctness checkpoint

The optional projection-only path passes 87 tests with zero failures or ignored
tests. Every numerical command ran under `/tmp/ninjars-symbolica-tests.lock`
with `--test-threads=1`; builds and lint checks ran outside that lock. This is
correctness evidence, not a performance measurement.

| Numerical suite | rust-codegen | rust-codegen,native-kernels |
| --- | ---: | ---: |
| New swapped unit controls | 8 | 10 |
| New public residue/callback/amplitude controls | 3 | 3 |
| Bubble controls, including independent sampled reconstruction | 7 | 7 |
| Existing denominator reuse regressions | 7 | 9 |
| Existing guarded rank regressions | 6 | 18 |
| Existing native bubble regressions | — | 5 |
| Existing workspace reuse regressions | — | 4 |
| Total | 31 | 56 |

The new controls cover degrees 0 through 3, complete higher-rank residues,
unchanged full callback kinds and identities, old standalone/combined raw APIs,
denominator reuse and distinct-borrow recomputation, poisoned scratch/retry,
transactional residue publication, dead-input and source-projection validation,
and retained versus discarded output overflow. Ordinary comparisons include
f64, DoubleFloat and 192-bit Float. Static and dynamic Dual comparisons examine
all five asymmetric complex components, including higher and mixed derivatives.
The n5/r4 guard test invalidates only one swapped request and requires generic
dispatch for the entire active correction kind. The actual-master amplitude
test verifies finite/pole agreement, accumulation, integration-failed lane
isolation, recovery, reset and empty input.

All selected positive-head graphs remain **interpreted**, including the guarded
variant. Full parameter ABIs, source counts, denominator/source projection
validation and numerator callbacks remain unchanged. The opt-in affects only
the new projection-specific combined method; old raw methods remain complete.

## Sources and records

- `97e095eb2bedf4f770cb30249b7dd8e82e6ab932`: independently reviewed source draft.
- `bd01a8e`: root's actual-master amplitude test addition.
- `5978461`: cherry-picked equivalent of main `1d0106b`, restricting existing
  test workspace probes to native-feature tests. No production behavior change.
- Numerical executables were built at `5978461`. `initial-source.json` records
  that exact source and the clean dependency revisions. Both builds succeeded
  on their first attempt.
- `6da695e`: removal of one unused `DualNumberStructure` test import, identified
  by strict Clippy. This is the only source correction during validation. Both
  final strict checks compile this cleanup; numerical tests were not repeated
  for an import-only edit.

`test-records.json` keys each execution by executable SHA-256 and exact argv,
with passed/failed/ignored counts, complete invocation and log hashes. Artifact
records retain actual executable paths and hashes. `manifest.json` records
source snapshots, commands, check results and every retained evidence hash.

The initial feature-off rank filter used the source filename rather than its
module name and selected zero tests. Its log is retained but excluded from the
87-test total. The corrected `rank_specialization::tests::` invocation passed
all six feature-off tests. `clippy-feature-off-initial.log` retains the failed
unused-import check; `clippy-feature-off.log` and
`clippy-native-initial.log` are the passing final strict checks. Known warnings
from vendored dependencies remain visible in the raw build/check logs.

Raw test logs retain their final empty line. The initial staged diff check
reported those lines; its output is retained. A narrowly scoped `.gitattributes`
entry disables only end-of-file blank-line checking for these logs, preserving
their original bytes and hashes. The final staged diff check passes.

## Limits

The checkpoint does not establish a speedup, native execution of selected
graphs, trimmed callbacks, or additional direct original-Ninja comparisons.
Original Ninja matching and full project acceptance remain separate gates.
Future measurements must separate actual master computation from reduction and
include the shared-master reduction-only control. No timing jobs were run here.
