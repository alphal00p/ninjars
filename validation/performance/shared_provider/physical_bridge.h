// Validation-only physical numerical boundary; no production core FFI.
#pragma once
#include "bridge.h"
#ifdef __cplusplus
extern "C" {
#endif
struct SpPhysical;
/* mode0=Symbolica interpreter, mode1=SymJIT O2. Both use exact factored heads.
 * Construction loads a DOT graph, prepares rules and compiles evaluators once.
 * The physical handle owns its provider. Destroy borrowed adapters first;
 * never pass the borrowed provider to sp_destroy. Every use is serialized. */
int32_t sp_physical_create(const uint8_t *path,size_t length,uint32_t mode,struct SpPhysical **out);
int32_t sp_physical_provider(struct SpPhysical *,struct SpProvider **out);
/* n,rank,legs,model_columns,coefficient_heads,output_columns,input_heads,wavefunctions_per_row */
int32_t sp_physical_info(struct SpPhysical *,uint64_t *out,size_t length);
/* Per row: momenta20 (five positive-energy four-vectors), helicities5,
 * model7=[MT,ymt,MH,Gf,aS,aEWM1,MZ], positive scale squared. WT=WH=0.
 * Output51=[20 shifted momenta,5 squared masses,25 S entries,scale squared].
 * Whole-batch transactional: errors preserve output and provider bindings. */
int32_t sp_physical_prepare(struct SpPhysical *,size_t rows,const double *momenta,size_t np,
    const int8_t *helicities,size_t nh,const double *model,size_t nm,const double *scales,size_t ns,
    SpPair *out,size_t no);
int32_t sp_physical_coefficients(struct SpPhysical *,SpPair *out,size_t length);
/* Audit outside any measurement: maximum scaled Float256 discrepancy, index,
 * actual re/im, expected re/im. Inputs exactly promote represented binary64. */
int32_t sp_physical_float256_audit(struct SpPhysical *,double *out,size_t length);
/* successful prepare calls, rows, wavefunction calls */
int32_t sp_physical_stats(struct SpPhysical *,uint64_t *out,size_t length);
int32_t sp_physical_destroy(struct SpPhysical *);
/* Live masters always execute expressions; frozen replay remains separate. */
int32_t sp_master_set_live(struct SpMasters *);
/* Backend calls (including failed), successful backend rows, live flag. */
int32_t sp_master_execution_stats(struct SpMasters *,uint64_t *out,size_t length);
int32_t sp_provider_stats(struct SpProvider *,uint64_t *out,size_t length);
/* Production master backend, bypasses validation master wire/replay. */
int32_t sp_session_run_production(struct SpSession *,struct SpProvider *,size_t rows,
    const SpPair *shifts,size_t nv,const SpPair *masses,size_t nm,const SpPair *scales,size_t ns,
    const SpPair *matrix,size_t nmat,SpPair *out,size_t no);
int32_t sp_session_production_stats(struct SpSession *,uint64_t *out,size_t length);
#ifdef __cplusplus
}
#endif
