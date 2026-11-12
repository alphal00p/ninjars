# MadLoop NinjaRS bridge 0.1.0 (ABI 4)

This is a standalone source package of the MadLoop bridge and its numerical
regressions. It builds without a MadGraph source tree or the development
reference clones. Cargo downloads the exact dependency revisions in Cargo.lock
unless they are already cached; this is not an offline dependency bundle.

The bridge is GPL-3.0-or-later, including code adapted from the earlier
GPL-licensed NinjaRS revision identified in UPSTREAM.md. COPYING retains that
license and LICENSE.MadGraph preserves the MadGraph redistribution notice.
The current NinjaRS dependency is 0BSD; Symbolica has its own licensing terms.
Covered NinjaRS operations use its scoped OEM library key, including
oneloopmaster and MadLoop high-precision rescue, while that key remains valid.

Linux with GNU Fortran and Rust/Cargo >= 1.96 is the validated build target.
Run `make install PREFIX=/absolute/path FC=gfortran` to build, or use the
HEPToolsInstaller recipe. The recipe records dependency/archive provenance and
executes the installed Fortran DP/MP smoke test before publishing the result.

`make check` runs Rust numerical regressions and the Fortran smoke test.
`make check-oem` tests two simultaneous Fortran processes with no personal key.
`python3 -m unittest discover -s tests -p 'test_*.py'` runs platform-preflight and
dependency-pin tests. `python3 tools/ninjars_upstream.py --help` describes the
read-only upstream-version checker. Historical MG7-relative documentation links
in the preserved validation records refer to MadGraphTeam/MadGraph7 PR #315.

MadLoop integration harnesses additionally need a MadGraph7 checkout; set
`MADGRAPH_ROOT=/absolute/path/to/MadGraph7` when running
`tests/poles/madloop_validation.py`. The installed library has no such dependency.
