#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
struct SpProvider;
typedef struct { double re, im; } SpPair;
typedef struct { uint64_t lane, sample; } SpIdentity;
/* Status: 0 success, 1 dimension/provider error, 2 caught Rust panic.
 * Only handles returned by sp_create are valid. Calls on one handle must be
 * serialized. Destroy once. Buffers are borrowed for the duration of the call;
 * all nonempty pointers must be aligned and valid for their declared lengths.
 * Output storage may not overlap the provider. Errors preserve caller output.
 * Complex storage is explicitly copied; no std::complex layout is assumed. */
int32_t sp_create(uint32_t fixture, struct SpProvider **out);
int32_t sp_info(struct SpProvider *, uint64_t *out, size_t len);
int32_t sp_bind(struct SpProvider *, const SpPair *, size_t len, size_t rows);
/* kind=0 direct: five complex cells [q0,q1,q2,q3,mu2], no powers;
 * kind=1 mu: eight cells [v0,v1], signed t powers;
 * kind=2 T3: thirteen cells [a,e3,e4,beta], powers flattened (t,mu);
 * kind=3 T2: nineteen cells [a0,a1,e3,e4,beta0,beta1,beta2], (t,x,mu).
 * Output is row-major in explicit power order. Each row carries lane/sample.
 * A partition is common to all rows in this minimal probe interface. */
int32_t sp_call(struct SpProvider *, uint32_t kind, const int32_t *powers,
                size_t powers_len, const SpIdentity *ids, size_t rows,
                const SpPair *data, size_t data_len, const uint8_t *partition,
                size_t partition_len, SpPair *out, size_t out_len);
int32_t sp_destroy(struct SpProvider *);
size_t sp_error(uint8_t *out, size_t capacity);
int32_t sp_probe_panic(void);
int32_t sp_control(uint32_t fixture, uint32_t kind, size_t lane, SpPair *out, size_t len);
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif
struct SpMasters;
/* Integral IDs: A0=0 B0=1 B1=2 B11=3 B111=4 C0=5 D0=6.
 * Source IDs: safe Rust provider=1, original Ninja adapter=2.
 * Capture computes absent rows with SymbolicaMasters; freeze rejects every miss.
 * Keys retain every f64 bit, ordered invariants/masses, kind and scale. */
int32_t sp_master_create(struct SpMasters **);
int32_t sp_master_call(struct SpMasters *,uint32_t kind,uint32_t source,size_t rows,
                      const SpPair *invariants,size_t ni,const SpPair *masses,size_t nm,
                      const SpPair *scales,size_t ns,SpPair *out,size_t no);
int32_t sp_master_freeze(struct SpMasters *);
/* Stats: entries,both_sources,Rust_only,Ninja_only,backend_rows,replay_rows,misses. */
int32_t sp_master_stats(struct SpMasters *,uint64_t *out,size_t len);
int32_t sp_master_destroy(struct SpMasters *);
int32_t sp_rust_bubble(struct SpProvider *,struct SpMasters *,const SpPair *shifts,size_t nv,
                       const SpPair *masses,size_t nm,SpPair scale,const SpPair *matrix,size_t nmat,
                       SpPair *out,size_t no);
struct SpSession;
int32_t sp_session_create(size_t propagators,size_t rank,size_t rows,struct SpSession **);
/* Setup-only selector; default/old constructor remains interpreted. Failure
 * leaves *out unchanged. No selector changes numerator or master providers. */
enum SpReductionKernels { SP_KERNEL_SYMBOLICA=0, SP_KERNEL_DENOMINATORS=1, SP_KERNEL_NATIVE=2, SP_KERNEL_INTERPRETED_RANK=3, SP_KERNEL_NATIVE_RANK=4 };
int32_t sp_session_create_with_kernels(size_t propagators,size_t rank,size_t rows,uint32_t selection,struct SpSession **out);
/* Independent setup policy: Whole=(0,0), FixedRows=(1,N>0). A bound counts
 * complete native kernel rows, not phase-space lanes. Old constructors use
 * Whole. FixedRows requires a native selection/build. Invalid pairs preserve *out. */
enum SpNativeExecution { SP_EXECUTION_WHOLE=0, SP_EXECUTION_FIXED_ROWS=1 };
int32_t sp_session_create_with_execution(size_t propagators,size_t rank,size_t rows,uint32_t selection,uint32_t policy,size_t block_rows,struct SpSession **out);
/* Independent projection-only swapped arithmetic: 0 disabled, 1 enabled.
 * Selected graphs remain interpreted even when native kernels/blocks are set.
 * All old constructors default disabled; invalid tags preserve *out. */
enum SpPrunedSwapped { SP_PRUNED_SWAPPED_DISABLED=0, SP_PRUNED_SWAPPED_ENABLED=1 };
int32_t sp_session_create_with_pruning(size_t propagators,size_t rank,size_t rows,uint32_t selection,uint32_t policy,size_t block_rows,uint32_t pruning,struct SpSession **out);
/* selection,native_feature_enabled,then eight backend flags in exact order:
 * T2,T3,bubble_source,triangle_bubble,tadpole_source,other_reduction,integration,accumulation.
 * Backend0=interpreter,1=native Rust. Every family is reported, including unused. */
int32_t sp_session_kernel_info(struct SpSession *,uint64_t *out,size_t len);
/* Separate additive accessor: propagators,rank,enabled,alternate_backend.
 * The last two cells are boolean; backend0=interpreter,1=native Rust.
 * Generic/fallback family coverage remains in the unchanged ten-cell accessor. */
int32_t sp_session_rank_correction_info(struct SpSession *,uint64_t *out,size_t len);
/* Exactly two cells: configured policy tag and bound. Errors preserve output. */
int32_t sp_session_native_execution_info(struct SpSession *,uint64_t *out,size_t len);
/* Exactly five cells: enabled,backend0=interpreter,full_columns,swapped_columns,
 * guarded_rank_correction. Disabled returns allzero. Configured metadata only;
 * inactive bubble stages have zero columns. Errors preserve caller output. */
int32_t sp_session_pruned_swapped_info(struct SpSession *,uint64_t *out,size_t len);
/* Retained topology/geometry/wire buffers. Both Rust providers call the same
 * non-inlined sp_call/sp_master_call entry points used by the C++ adapters.
 * Handles must be mutually disjoint and exclusive. All arrays are row-major. */
int32_t sp_session_run(struct SpSession *,struct SpProvider *,struct SpMasters *,size_t rows,
                      const SpPair *shifts,size_t nv,const SpPair *masses,size_t nm,
                      const SpPair *scales,size_t ns,const SpPair *matrix,size_t nmat,
                      SpPair *out,size_t no);
/* numerator_calls,numerator_rows,master_calls,master_rows */
int32_t sp_session_stats(struct SpSession *,uint64_t *out,size_t len);
int32_t sp_session_destroy(struct SpSession *);
#ifdef __cplusplus
}
#endif
