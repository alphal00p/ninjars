This standalone validation workspace checks original Ninja's callback adapter
against the **same Rust `PolynomialNumerator<f64>` implementation**. It does not
link C++ into a production package or measure timing. A second probe supplies
both reducers with the same captured union of scalar/tensor master requests.

With the pinned compiler environment and Symbolica license available, run:

```sh
python3 validation/performance/shared_provider/run.py \
  --output-dir target/performance/shared-provider-probe
```

The script uses already built original Ninja/OneLOop libraries from
`target/reference-uncached-v2` (override with `--reference-build-dir`). It builds
a Rust shared library, copies that immutable artifact into the output directory,
and compiles the C++ `ninja::Numerator` and `ninja::IntegralLibrary` subclasses
against it. The copied reference libraries, headers and configuration must
match their recorded hashes and have the OneLOop result cache disabled.
Build logs, commands,
source/library/executable hashes and correctness results are retained. The output
directory must be empty. Compilation completes before runtime checks begin.

`bridge.h` defines the array layouts, lengths, ownership and status contract.
Complex values are copied component by component; there is no assumption that
C++ and Rust complex structs share a layout. The provider owns bound parameters.
Every call carries explicit phase-space lane and sample identity. A common cut
partition is supplied per batch; a future mixed-cut benchmark adapter must group
requests or extend that metadata. Handles must be used serially and destroyed
exactly once. Nonempty pointers must be aligned and valid for their declared
lengths. Structural/provider errors leave outputs unchanged. Rust panics are
caught and return status two; the intentional panic control prints the normal
panic-hook message but does not unwind across C++.

Fixed fixtures cover ordinary box rank four and higher box rank five, complex
parameters and loop vectors, both mu layouts, every supported T3/T2 expansion
depth, and repeated/permuted lane identities. Safe Rust expected arrays use
explicit independently ordered exponent tuples. A Rust test also exercises the
flat batch interface, empty rows, malformed dimensions, failed lane identity and
recovery/error behavior. Finally original Ninja runs the shared callback for two
bindings of the analytic triangle numerator `a*mu_squared`, whose finite rational
term is `a/2` and poles vanish.

The master control captures the actual union of original Ninja and NinjaRS
requests for shifted ordinary/higher-rank bubbles with asymmetric masses and
two scales. It evaluates missing requests with the production Symbolica master
provider, freezes the table, and repeats both complete reductions using lookup
only. Keys retain every f64 real/imaginary bit in ordered invariants, masses and
scale, plus the master kind; there is no approximate matching, mass reordering,
or signed-zero normalization. Both reducers receive the same explicit S-matrix.
The report distinguishes shared, Rust-only and original-only entries. A replay
miss fails without changing output or running a master expression. Rust tests
cover permuted/tail/empty batches and reject changed scale, mass order and kind.
The C++ adapter explicitly overrides B111 as well as B1/B11; no original
rank-three derivation is used. A separate C0/D0 ordering control captures explicit
ordered ABI requests and freezes the table before calling the original real- and
complex-mass virtual methods. Asymmetric invariants, masses and a nonunit scale
make a reordered key fail even if the integral is permutation-symmetric.
Integrated shared-provider C0/D0 reductions are a later extension. This probe
validates real-invariant massive requests with real and negative-imaginary
squared masses; it does not establish the inherited massless virtual paths or
complex-invariant support in original Ninja.

The `q0²` and explicit `mu²` controls are also run with zero and unit time origin.
All total Laurent coefficients must agree, and each finite result must equal
its own cut-constructible plus rational parts. The explicit `mu²` result and
rational part equal `8/3`. The `q0²` split differs between the two residue bases;
the report retains both five-component results and marks rational-separation
parity incomplete. This discrepancy is not hidden by comparing only a checksum.

An active regression preserves the former oneloopmaster exceptional point:
C0 invariants `[-1,-2,-3]`, squared masses `[2,3,5]`, and scale squared `7`.
The fixed helper now keeps the inactive singular logarithm behind its locally
constant sheet-correction guard. The test requires success, finite coefficient
`-0.1352123962254360745` within `1e-12`, and exact zero poles. The value is
independently supported by 60-digit Feynman-parameter quadrature. The older
failing evidence is retained unchanged. `run.py` includes this regression in
its required Rust suite; it is no longer ignored. The asymmetric adapter-order
controls remain separate. Run it alone with `cargo test --manifest-path
validation/performance/shared_provider/Cargo.toml --locked
euclidean_c0_exceptional_point_must_be_finite -- --test-threads=1`.

The retained-session control constructs topology and geometry once, retains its
wire buffers, and routes the Rust numerator and master traits through the same
non-inlined `sp_call`/`sp_master_call` entry points that C++ uses. It checks all
five components against the direct Rust control, including after an empty call
and malformed-buffer rejection. A Rust wire regression groups mixed cut
partitions, preserves repeated/permuted lane identities, and checks retained
capacities across empty/reactivated batches. The numerator bridge retains numeric packing,
outputs and expansion layouts; its singleton request is a stack array. C++
master argument packing uses fixed stack arrays.

This is still a correctness prototype. Borrowed request descriptors for larger
batches, master-key capture/replay staging, and underlying providers retain
allocation work that has not yet been measured or removed. The earlier direct
Rust control deliberately still constructs its topology and geometry per call;
only the separate session is a candidate for a warmed timing path. No speed or
allocation-free claim follows from passing these controls.
See `docs/matched-provider-benchmark.md` for the complete
measurement boundaries and remaining physical preparation work.
