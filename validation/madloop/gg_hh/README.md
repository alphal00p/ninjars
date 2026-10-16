The `gg -> HH` validation uses eight unsquared top-quark graphs: two Higgs-channel
triangles and six boxes. FeynKit independently generates them with the existing
`ttg`, `ttH` and `HHH` model vertices. No production preparation or reduction
kernel is copied into this harness. Graph generation, ordered physical
propagators, routing, color projectors, contraction and algebra all use the
existing FeynKit, Idenso and Symbolica APIs.

Generate fixtures with a locked graph-library build, then the standalone Rust
validation program:

```sh
cargo build -p ninjars-graph --locked --message-format=json > target/graph-artifacts.jsonl
python3 validation/madloop/gg_hh/prepare_feynkit.py \
  --cargo-artifacts target/graph-artifacts.jsonl \
  --output-dir target/physical/gg-hh/regenerated
```

The output records the actual compiler, commands, library/source hashes and
graph metadata. Exact numerical fixtures are written separately from graph
preparation. The color projector is `delta_ab/8`, so each output is the
coefficient of `delta_ab`, without squaring, interference, helicity/color
averages or an identical-final-state phase-space factor. Triangle Higgs bridges
are retained once by `PreparedIntegrand::overall_factor`; their three physical
loop propagators remain distinct from the boxes' four propagators.

Discover and export the reference graphs with the **unmodified** local
`loop_sm` model:

```sh
python3 validation/madloop/gg_hh/prepare_madloop.py \
  --output-dir target/reference/madloop-gg-hh
python3 validation/madloop/gg_hh/probe.py --graph 9 \
  --exports-dir target/reference/madloop-gg-hh \
  --output-dir target/reference/madloop-gg-hh-probes
```

Repeat the probe for IDs 10 through 16. Discovery observes MadLoop's existing
filter before numerical grouping; each export contains exactly one selected
loop graph with multiplier one. Subprocess grouping and optimized loop output
are disabled. The full command history, original-model hashes, source hashes,
vertices, loop calls and linear color-amplitude coefficients are recorded.

Each exported process also contains a separate R2 amplitude. The probe links
only `loop_num` and `CT_interface` and calls loop amplitude **2** directly.
It supplies an aborting replacement for the inactive automatic
`MP_BORN_AMPS_AND_WFS` entry point, sets `MP_DONE=true`, and rejects CT mode zero.
The generated squared-amplitude driver and counterterm routines are never
called. Fixed CutTools modes **1** and **4** use AVH library **2**; no adaptive
backend/precision selection is allowed. Mode four promotes the same binary64
external wavefunctions and couplings, so it is not an independent
high-precision physical-input oracle.

For every graph and state the probe evaluates 70 affine complex loop momenta,
an unisolvent grid for total degree four. The triangle numerator has degree
two; the same grid covers it. The eight bound states are four physical
helicities, generic complex polarizations, and three Ward substitutions.
Every printed input/output component is checked finite, and the raw numerator
is compared against MadLoop's wrapper. Physical integrated reference rows must
also be finite and report stable status. Generic/Ward integrated rows remain
diagnostic, and no individual-graph Ward-zero assumption is made.

```sh
python3 validation/madloop/gg_hh/collect.py \
  --exports-dir target/reference/madloop-gg-hh \
  --probes-dir target/reference/madloop-gg-hh-probes \
  --output target/reference/madloop-gg-hh-probes/reference.json
python3 validation/madloop/gg_hh/match.py \
  --cargo-artifacts target/graph-artifacts.jsonl \
  --reference target/reference/madloop-gg-hh-probes/reference.json \
  --fixtures examples/graphs/physical/gg-hh \
  --output-dir target/physical/gg-hh/raw-match
```

The matcher identifies directed fermion cycles, builds the reference
propagator family and uses `IntegralFamily::find_mapping`. Every mapped
propagator is checked exactly. Shared `Kinematics::linear_terms` and
`FourMomentum<Atom>` bind the returned momentum image before numerical
comparison. Raw reference parameters are converted from their printed
binary64 values into exact Symbolica rationals, preserving a common input.
The independently verified mappings are:

| FeynKit graph | Directed cycle | MadLoop ID | Loop image |
|---|---|---|---|
| triangle 0 | 1,5,2 | 11 | −q |
| triangle 1 | 1,2,5 | 12 | q |
| box 2 | 1,4,3,2 | 10 | −q |
| box 3 | 1,3,4,2 | 9 | −q |
| box 4 | 1,2,4,3 | 13 | q |
| box 5 | 1,3,2,4 | 16 | q |
| box 6 | 1,2,3,4 | 15 | q |
| box 7 | 1,4,2,3 | 14 | −q |

Cycle entry 5 is the combined outgoing Higgs pair attached through the bridge.
The source-to-reference denominator permutations are `[2,0,1]` for triangles
and `[3,0,2,1]` for boxes.

The raw factor **−1/2** follows from one fermion-loop minus sign and
`Tr(Ta Tb)=delta_ab/2`. It is independently checked, without numerical fitting,
by exact zero-momentum temporal-polarization controls:

- Box: `−2 GC_11² GC_94² MT⁴`.
- Triangle including its Higgs bridge: `2 GC_11² GC_94 GC_69 MT³/MH²`.

The model coupling identities are `GC_11=GC_5=iG`,
`GC_94=GC_37=−i yt/sqrt(2)` and `GC_69=GC_30=−6i lambda vev`, where the right-hand
coupling names are MadLoop's. Its `CT_interface` contributes `1/(16*pi²)`, so
the corresponding linear Laurent factor is **−8π²**.

The archived raw stage passes **4,480 independent FeynKit/MadLoop comparisons**
with maximum scaled error **5.65e−14**, using the fixed coefficient-wise
`abs(actual-reference)/max(1,abs(reference)) <= 1e−9` criterion. MadLoop's
separate raw/wrapper comparisons number **8,960**. Unisolvence is in loop
components after each polarization binding, not in arbitrary polarization
variables. Full integrated NinjaRS comparison is the next stage; the reference
Laurent rows alone do not establish that gate. No performance claim is made.
