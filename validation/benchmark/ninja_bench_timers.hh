// Timing instrumentation for a benchmark-only copy of original Ninja.
//
// This header is copied into src/ of the 'o3-fortranrules-instr' build
// (see validation/benchmark/build_ninja.py and ninja_instrumentation.patch).
// It is never part of the uninstrumented reference libraries.
//
// Cheap time-stamp-counter accumulators split the time spent inside
// ninja_tensor_evaluate (the MadLoop entry point) into TensorNumerator
// callbacks, AvHOneLoop/OneLOop master-integral calls and the remainder
// (Ninja's own reduction).  Counters are read from Fortran or C through
// the extern "C" functions defined in ninja_wraps.cc.

#ifndef NINJA_BENCH_TIMERS_HH
#define NINJA_BENCH_TIMERS_HH

#include <x86intrin.h>

namespace ninja_bench {

  enum Bucket {
    EVALUATE = 0,      // whole ninja_tensor_evaluate (incl. TensorNumerator ctor/dtor)
    NUMERATOR = 1,     // TensorNumerator::evaluate/muExpansion/t3Expansion/t2Expansion
    MASTERS = 2,       // AvHOneLoop master-integral methods (OneLOop calls)
    NUM_SETUP = 3,     // TensorNumerator constructor + destructor (allocations)
    N_BUCKETS = 4
  };

  struct Counters {
    unsigned long long cycles[N_BUCKETS];
    unsigned long long calls[N_BUCKETS];
    // (n, rank) histogram of the arguments received by ninja_tensor_evaluate
    unsigned long long n_rank_calls[9][11];
  };

  extern Counters counters;

  struct Scope {
    Bucket b;
    unsigned long long t0;
    explicit Scope(Bucket bucket) : b(bucket), t0(__rdtsc()) {}
    ~Scope()
    {
      counters.cycles[b] += __rdtsc() - t0;
      counters.calls[b] += 1;
    }
  };

} // namespace ninja_bench

#define NINJA_BENCH_SCOPE(bucket) \
  ::ninja_bench::Scope ninja_bench_scope_(::ninja_bench::bucket)

#endif // NINJA_BENCH_TIMERS_HH
