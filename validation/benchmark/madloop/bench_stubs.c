/* Weak no-op versions of the instrumentation accessors defined by the
 * instrumented Ninja build (validation/benchmark/instrument_ninja.py).  When
 * the bench is linked against an uninstrumented libninja.a these stubs are
 * used and report -1; the instrumented library's strong definitions take
 * precedence otherwise. */
__attribute__((weak)) void ninja_bench_reset_(void) {}

__attribute__((weak)) void ninja_bench_get_(double out[8])
{
  for (int i = 0; i < 8; ++i)
    out[i] = -1.0;
}

__attribute__((weak)) void ninja_bench_get_nrank_(double out[99])
{
  for (int i = 0; i < 99; ++i)
    out[i] = 0.0;
}

__attribute__((weak)) double ninja_bench_tsc_hz_(void) { return -1.0; }
