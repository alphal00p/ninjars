# Shared physical numerical preparation

`run_physical.py` is a correctness probe for the independently matched top pentagon graph26 in gg→HHH. It consumes `validation/evidence/gg-hhh-changing-pool.json` unchanged: three binary64 momentum rows at center-of-mass energies450,525,600, fixed `pp` helicities, and zero widths. The600 point lies across a massive-pair subchannel threshold from the first two; none lies on the threshold.

The per-point input boundary is five positive-energy four-momenta, five helicities, the free model inputs `[MT,ymt,MH,Gf,aS,aEWM1,MZ]`, and the squared renormalization scale. Each complete path calls the same `sp_physical_prepare`: shared FeynKit wavefunctions (two vector and three scalar), dependent model/coupling evaluation, one multi-output coefficient evaluation per row, routed propagators, squared masses, and the S matrix. It uses exact model expressions and shared Symbolica polynomial factoring before compilation. Graph loading, Feynman rules, symbolic contraction, coefficient splitting/factoring and compilation are setup, outside the per-point boundary.

Both reducers use the exact same Rust `PolynomialNumerator` through `sp_call`. Both master adapters call `sp_master_call`. Two policies remain explicit:

- Frozen union replay validates reduction/contraction/master lookup using the union of actual requests from both reducers. It never computes a missing integral.
- Live shared masters bypass all result-table lookups and insertions and execute the production Symbolica master provider on every request. Backend call/row counters demonstrate that both complete paths execute the masters even at repeated points.

The live control is not the separate production-backend comparison between NinjaRS/oneloopmaster and original Ninja/OneLOop. That comparison remains required. This probe takes no timings and makes no acceptance claim.

`SpPhysical` owns its numerator provider. The pointer returned by `sp_physical_provider` is borrowed, must never be destroyed through `sp_destroy`, and is invalidated when the physical handle is destroyed. Destroy the C++ borrowed adapter first. Preparation and reduction calls are serialized; buffers and handles must be valid and mutually disjoint for each call. Preparation failures preserve the caller's output and previously committed provider bindings. Input validation and nonfinite-coefficient failures reject the whole preparation batch explicitly.

The independent raw control uses all70 degree-four unisolvent complex loop samples at each of the three phase-space points. The verified mapping is FeynKit graph26→MadLoop filter55, `K→−K`, denominator permutation `[4,0,2,1,3]`. Raw MadLoop values receive the independently derived factor `−1/2`; integrated values receive `−8π²`. The DOT's color projector, fermion sign and graph prefactor are applied exactly once. Fixed MadLoop mode4 is the integrated oracle, with mode1 retained separately.

The coefficient oracle exactly promotes the represented binary64 momenta/wavefunction/model component inputs to Symbolica Float256. It validates coefficient arithmetic, not arbitrary-precision physical wavefunctions. Both coefficient providers use the same factored exact expressions. Per-component scaled comparison is `abs(actual-reference)/max(1,abs(reference)) ≤ 1e-9`; all comparisons reject nonfinite values. Total Laurent coefficients and each reducer's own finite=CC+rational identity are checked. Separate rational/CC parity is not claimed for different residue-basis conventions.

Preparation covers widths0,1,8,32,128,1024 and31. Integrated coverage is reported separately and controlled by `--maximum-integrated-batch`; do not infer integration coverage from preparation widths. The changing-point pool is distinct from helicity variation. Sources, dependency pins, input bytes, linked libraries and executable hashes are recorded before/after execution; the older shared-provider evidence remains immutable.

Run in the project's pinned tool environment with the local Symbolica license:

```sh
python3 validation/performance/shared_provider/run_physical.py \
  --output-dir target/performance/shared-physical-control \
  --maximum-integrated-batch 32
```

## Explicit reduction-kernel controls

Reduction selection is independent of physical coefficient preparation. The
latter remains factored SymJIT O2 heads with the shared interpreted
`PolynomialNumerator<f64>` direct/Mu/T3/T2 callbacks. The default reduction
selection remains `symbolica`; no option changes the shared numerator or master
policy.

`run_physical.py --reduction-kernels native-denominators|native` enables the
optional standalone `native-kernels` feature and selects the corresponding
public amplitude factory at session creation. `--native-catalog` compiles that
feature while retaining interpreted execution. For a single-artifact numerical
comparison, use:

```sh
python validation/performance/shared_provider/run_physical.py \
  --output-dir target/shared-kernel-parity --maximum-integrated-batch 8 \
  --all-reduction-kernels
```

This compares all three selectors using the same physical provider, numerical
inputs and frozen union of master requests. Each retained candidate session
is constructed before its reduction calls. The selected baseline is interpreted;
all five result components are compared to it, while the existing independent
MadLoop controls check the total Laurent coefficients. Empty/single/tail and
permuted batches remain covered. Frozen master backend counts and misses are
checked again after catalog comparisons.

The compatibility C ABI `sp_session_create` delegates to interpreted selection.
`sp_session_create_with_kernels` accepts only setup-time IDs 0/1/2, and errors
leave the caller's output slot unchanged. `sp_session_kernel_info` reports the
compiled feature and actual public-library selection for every family,
including interpreted integration and accumulation. The current report has ten
ABI cells: selection, compiled-feature flag, and eight family flags. The added
`tadpole_source_projection` family is native only for the full `native` selector;
`native-denominators` retains interpreted tadpole projections. Geometry and
tadpole subtraction remain independent interpreted stages. C++ and Python both reject
coverage inconsistent with the requested mode. There is no timed selector
mutation or rebuilding of the session during a reduction call. A normal batch
reset preserves the selected factory.

These are correctness controls. The partial native catalog does not change
geometry, master execution, other reduction kernels or numerator callbacks;
no timing or performance acceptance follows from this comparison.

The seven-family selector checkpoint is recorded in
`validation/evidence/shared-provider-kernel-selection.json`. One native-enabled
artifact checked all three selections at widths 0, 1, 3, and 8 against the same
frozen master union and numerator. The largest scaled difference across all
five complex output components was `3.50e-18`; this is a correctness control,
not a timing result. The compressed companion report retains source hashes,
commands, actual coverage, independent MadLoop checks and artifact identities.

The subsequent eight-family checkpoint is in
`validation/evidence/shared-provider-tadpole-selection.json`. It retains the
same widths, fixed provider and frozen master union, now reporting tadpole
source projections explicitly. All three selectors pass again, with a maximum
scaled difference of `3.50e-18`; the prior seven-family evidence is preserved.
