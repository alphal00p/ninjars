# Pinned source dependencies

`manifest.json` pins every non-registry source used by the workspace, including
the original Ninja validation source. The root `Cargo.lock` and `flake.lock` pin
the remaining Rust packages and development environment.

`patches/` records the minimal changes that NinjaRS needs while upstream review
is pending, consolidated into one base-to-head diff per dependency. Bootstrap
verifies each patch's SHA-256 and applies it to the pinned
base. Locally committed versions of exactly those patches are also recognized.
Patch source revisions, test evidence, and upstream PR URLs belong in
`references/PLAN.md`; a local commit does not imply an upstream PR exists.

The supplied MadGraph release is an additional validation dependency. Its bundled
OneLOop source checksum is recorded by `validation/reference/build.py` in the
generated `target/reference/revisions.json`.
