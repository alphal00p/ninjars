# COLLIER 1.2.10: the global cache returns wrong rank-9 heptagon tensors

## Summary

COLLIER's global cache (`InitCacheSystem_cll` / `InitEvent_cll`) silently returns
wrong tensor integrals for a 7-point function at rank 9. The values are wrong from
the third event of a cache onward, by O(1) or more of the largest component, and
COLLIER flags them with accuracy flag −2. That is the only sign of the error.

Found while building the COLLIER rank n+2 timing benchmark for NinjaRS
(`validation/benchmark/collier/`, branch `claude/collier-rank-n2-bench`, merged
into `claude/todo-next-integration`). NinjaRS's published COLLIER numbers do not
use the cache, so they are not affected.

## Affected version and builds

- COLLIER 1.2.10, the tarball from hepforge (sha256 `460624ea…62713`), built unmodified
  by `validation/benchmark/collier/build.sh` with gfortran in binary64:
  - `precise`: the shipped Release flags without `-ffast-math`;
  - `stock`: the shipped flags.

  Both builds behave the same.
- Not checked: COLLIER 1.2.9 with the cache. The NinjaRS reference harness runs 1.2.9
  without the cache.

## Reproducer

`validation/benchmark/collier/collier_cache_repro.f90`. `build.sh` builds it as
`bin/collier_cache_repro-<variant>`. It follows the manual's usage exactly:
- one cache (`InitCacheSystem_cll(1, 7)`);
- `InitEvent_cll` before every event, then one `TNten_cll(N = 7, R = 9)` call per event;
- the same kinematic point every time: the first lane of
  `validation/rank-n2/cases/rn2-int-mixed-n7-r9.json`;
- no parameter changes between events.

It compares the cached tensor with an uncached evaluation of the same call.

## Observations

- **Which events are wrong:**
  - events 1 and 2 return exactly the uncached tensor;
  - from event 3 onward, checked up to event 20, `max |T_cache − T| / max |T| = 3.9` with accuracy flag −2;
  - the error is the same in COLLIER's optimisation runs (events 3–13) and in its fully optimised mode (from event `nopt + 4` = 14).
- **Which (N, R) are affected:**
  - rank 8 at N = 7 is bitwise identical with and without the cache, including the three rank-8 twin lanes of the suite;
  - all 24 suite lanes at N = 7, R = 9 are wrong, by 0.1 to 46 of the largest component (one cache per lane, the same lane every event);
  - every N ≤ 6 lane agrees within 1.4e-7, which is COLLIER's own accuracy level.
- **Which sub-integrals matter:**
  - caching only N ≤ 4 sub-integrals (`InitCacheSystem_cll(1, 4)`) leaves an 8.0e-6 difference with flag 0;
  - caching only two-point functions gives bitwise identical tensors;
  - `SetTenRed_cll(7)`, which uses direct tensor reduction only from 7 points on, shrinks the error to about 1e-5;
  - so cached 5- and 6-point sub-tensors carry most of the error, and cached 3- and 4-point functions a small part.
- **Ruled out:** patching the UV-coefficient slice of the 6-point tensor cache entries (`src/tensors/TensorReduction.F90`, around lines 834 and 936) changed nothing, so the layout of those entries is not the cause.

## How the cache works (`src/Aux/cache.F90`)

- **Event 1** (`cache_mode = 1`): COLLIER counts the calls.
- **Event 2** (`cache_mode = 2`): COLLIER identifies identical calls. It compares N, the library, the hash `sum(args)` and every argument.
  - Per entry it records how often the call occurs (`casa`), the largest rank requested (`cara`) and the largest number of values (`cava`).
  - Each call position is mapped to its entry: `CachePoint(id, nmascall)`. Here `id` is the sub-integral and `nmascall` counts the "master calls" of the event.
  - The 6-point tensors of the tensor-reduction path register with `id = 0`, so each one counts as a new master call and increments `nmascall`.
- **Events 3 .. nopt+3** (`cache_mode = 3`, optimisation runs) **and later events** (`cache_mode = 0`, fully optimised): `ReadCache` finds a call **only by its position** `CachePoint(id, nmascall)`. The source says so: "Identical function calls are identified assuming a fixed order of the function calls". The arguments are not checked.
- **Rank promotion:** the first occurrence of an entry is computed at `r = max(cara, r)`, the largest rank seen in event 2.
  - It goes through the "rank > rmax" branch: for example `CalcFred` and `CalcTensorFrRed` with auxiliary arrays at the higher rank, in `src/COLI/reductionEFG.F90` and `src/tensors/TensorReduction.F90`.
  - Later occurrences read the first `nval(rmax)` values whenever `r ≤ new_cara` and `nval ≤ new_cava`. The per-rank interleaved layout makes this prefix read consistent.

## Likely origin

The call order that the cache relies on is not fixed once rank promotion starts.

- **Event 2** runs without promotion and freezes the map `CachePoint(id, nmascall)` for that call sequence.
- **From event 3 on**, the first occurrence of a promoted entry is computed at the higher rank. That computation makes its own, different sub-integral calls: higher-rank lower-point functions, possibly more sub-integrals, and in the tensor-reduction path more `id = 0` "master calls" that move `nmascall`.
- **So later calls are matched to the wrong entries.** Lookups by position then return coefficients of other sub-integrals or ranks, and nothing checks the arguments.
  - The error-estimate slots are read from the wrong entries too, which explains accuracy flag −2.
  - Since every later event replays the same shifted sequence, the error persists.

This explains all observations:
- **Events 1–2 are right:** nothing is read from the cache then.
- **The error starts at event 3 and persists:** that is the first event that reads, and every later one replays the same shift.
- **Only N = 7, R = 9 is affected:** this is the first case in which some sub-integral is requested at two different ranks within one event, so promotion happens. At R = 8 there is no promotion, and the results are bitwise identical.
- **The small errors with partial caching:** promotions among lower-point functions explain the 8e-6 when caching only N ≤ 4; two-point functions are never promoted in a way that shifts the sequence, so caching only them is exact.
- **The `SetTenRed_cll(7)` reduction:** it removes the `id = 0` 6-point tensor path, which explains why the error shrinks.

**Status: hypothesis.** It is consistent with every observation but not yet confirmed by instrumentation.

### A second, latent defect (benign)
In `cache_mode = 2` (event 2), `cache.F90` updates the entry with
`cava(ncapo) = max(cava(ncapo), nval)` and only then tests
`if (nval .gt. cava(ncapo)) cachesize = cachesize + nval`. The test can never be
true, so the size estimate misses the growth of repeated calls with more values.
It seems harmless, because `WriteCache` enlarges `CacheVals` on demand, but the
increment should be computed before `cava` is updated.

## How to confirm

Instrument `ReadCache` in `cache_mode` 3 and 0 to compare the entry it reads with
the call. Check `catype(ncapo)` against `N`, `CacheLib(ncapo)` against `lib`, and
`argshash(ncapo)` against `sum(args)`. Print `(id, nmascall, r, cara, nval, cava)`
for the reproducer in events 2 and 3. A mismatch at event 3 confirms the
hypothesis.

## Likely fix (minimal)

1. **Verify the entry before reading it.** In `ReadCache`, modes 3 and 0, check that the cached entry belongs to this call: `N == catype(ncapo)`, `lib == CacheLib(ncapo)`, `sum(args) == argshash(ncapo)`, and optionally the full arguments, as mode 2 already does.
   - On a mismatch, compute without the cache (`nocalc = .false.`, `wrica = .false.`).
   - This costs one complex sum per lookup and turns silent wrong results into correct, uncached ones.
2. **Make rank promotion keep the call sequence.** Either bypass the cache for the sub-integral calls made inside a promoted computation, or determine the promoted ranks in an extra initialisation event before `CachePoint` is frozen.
3. **Fix the `cachesize` bookkeeping in mode 2.**

## Workarounds for users

- Do not use the global cache for heptagons at rank 9 (more generally, rank above N + 1 at N = 7). COLLIER's default is no global cache.
- Caching only N ≤ 4 or setting `SetTenRed_cll(7)` reduces the error but does not remove it.
- NinjaRS's benchmark validates the cached variant but does not time it, and reports COLLIER uncached.

## Reporting

Not reported upstream. Decision of the 2026-10-20 handoff: we have no channel or
authorisation to contact the COLLIER authors (A. Denner, S. Dittmaier, L. Hofer)
on the user's behalf. This file is the ready-to-send report; send it with
`collier_cache_repro.f90` through the contact on https://collier.hepforge.org if
the user wants. The hypothesis above should be confirmed first (see "How to
confirm").
