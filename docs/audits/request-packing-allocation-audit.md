# Borrowed request packing: bounded source audit

Source reviewed: NinjaRS `38f8569`, without builds, tests, allocation sampling or
timings. These are allocation candidates, not observed counts or speedup claims.
No implementation is proposed ahead of the paired-graph/pruning experiments.

## Remaining transient containers

The reducers retain numerical buffers and row-index vectors, but construct local
vectors of borrowed descriptors on each evaluation:

| Stage | Local descriptor containers | Source |
| --- | --- | --- |
| Triangle | cut requests, active cut references, numerator expansions, divisions | `crates/ninjars/src/triangles.rs:171,201,231,232` |
| Bubble | cut requests, active cut references, expansions, divisions, parent corrections | `crates/ninjars/src/bubbles.rs:387,432,462,463,497` |
| Box | cut requests, sample requests, numerator requests, sample divisions, projections; two more for optional mu expansion | `crates/ninjars/src/boxes.rs:326,404,453,454,528,529,570` |
| Tadpole | cut requests, expansions, divisions, parent corrections | `crates/ninjars/src/tadpoles.rs:488,549,550,584` |

An all-stable n5/r4 default path therefore constructs 18 nonempty transient
vectors across these four stages. This is a source inventory, not an allocator
measurement. Bubble/tadpole default combined trait methods create four additional
temporary vectors each, but the Symbolica kernels override those methods
(`bubble_kernels.rs:991`, `tadpole_kernels.rs:877`). They must not be charged to
the normal adapter merely because they appear in the core source.

The complete no-packing-allocation gate also includes the integration request
vector at `amplitude.rs:831` and pentagon vectors at `pentagons.rs:122,152,153`.
Fixing only these four stages cannot establish that gate.

## Smallest bounded candidate

Local `SmallVec` containers preserve the existing slice APIs, row order and
borrow lifetimes. The already locked version is 1.16.3; Symbolica uses SmallVec
and gammaloop enables its `const_generics` feature. This requires no NinjaRS
unsafe code, custom allocator, numerical trait bounds or new physics types.
Descriptors contain references and indices, so their inline footprint does not
grow with scalar precision or Dual derivative order.

Batch size one still represents many cuts: n5/r4 has 20 triangle orientation
requests, 20 bubble orientation requests, 60 bubble parent corrections and 50
tadpole parent corrections. For the existing suite through six propagators,
including higher rank/full sampling, maxima include 40 triangle orientations,
120 bubble corrections, 180 tadpole corrections and 90 box sample requests.
Inline capacities need an explicit stack-footprint review; a capacity of one is
not a scalar phase-space fast path. Larger topologies and batches would spill
anew on every call. This bounded change cannot promise the all-topology warmed
allocation requirement or improve runtime without measurements.

## Reusable scoped storage

`VecPool<T>` in the core recycles owned coefficients/masks; it does not erase
borrow lifetimes. Keeping a `Vec<Request<'call,T>>` in a persistent reducer cannot
be made safe merely by clearing it. Symbolica's `Workspace` recycles atoms
(`state.rs:1586`), and oneloopmaster's `BatchWorkspace` owns numerical slots
(`native/batch.rs:91`); neither is a general borrowed-descriptor arena. The
inspected shared FeynKit/gammaloop APIs expose no directly applicable scoped
request buffer.

Already locked bumpalo 3.20.3 offers safe `alloc_slice_fill_iter`, `with_capacity`
and `reset` APIs for borrowed descriptors. A library-owned per-call arena can
provide reusable storage without a custom allocator or lifetime casts in core.
However, `reset` drops all backing chunks except the current one
(`bumpalo/src/lib.rs:1049-1083`). Warm-up alone does not prove allocation-free
repeated calls; the complete checked frame footprint must fit retained capacity.
Its allocations do not run destructors, so use must remain restricted to
borrowed descriptors, not owned Float/Dual carriers. Bump is Send but not Sync;
adding it directly changes the reducer's public auto-trait behavior. Preserving
that behavior and robust capacity/reset handling is a larger ownership change
than local SmallVec substitution.

Either option leaves numerical arithmetic and all status/shape checks unchanged.
It also does not eliminate possible allocations in existing owned carrier
clones: for example, the reducers clear their direction vectors and clone
momenta back into them. Float and dynamic Dual clones may allocate even when the
outer vector retains capacity. Allocation evidence must distinguish descriptor
packing, owned numerical copying, evaluator work and masters. A later bounded
comparison should first verify request identities/order, failed-lane isolation,
empty/tail and grow/shrink behavior, then count allocations on warmed fixed
workloads separately for f64 and higher-precision/Dual carriers. Timing claims
require a separately coordinated measurement.
