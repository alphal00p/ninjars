#!/usr/bin/env python3
"""Insert benchmark timers into a *copied* original Ninja source tree.

Used only for the 'o3-fortranrules-instr' build variant. The edits are purely
additive (an #include, one RAII scope per instrumented function and a few
extern "C" accessors) and leave every arithmetic statement unchanged.

  python3 instrument_ninja.py <copied-ninja-root>

Buckets (see ninja_bench_timers.hh):
  EVALUATE  : body of ninja_tensor_evaluate (MadLoop's entry point), including
              TensorNumerator/Amplitude construction and destruction
  NUMERATOR : TensorNumerator::{evaluate,muExpansion,t3Expansion,t2Expansion}
  MASTERS   : uncached AvHOneLoop master-integral methods (OneLOop calls)
Optional: NINJA_BENCH_DUMP=<file> appends one JSON line per
ninja_tensor_evaluate call with its exact inputs (outside the timed scope).
"""
import pathlib
import re
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent


def replace_once(text, old, new, path):
    if text.count(old) != 1:
        raise RuntimeError(f"{path}: expected exactly one occurrence of {old!r}, found {text.count(old)}")
    return text.replace(old, new)


def instrument(root):
    src = root / "src"
    shutil.copy2(HERE / "ninja_bench_timers.hh", src / "ninja_bench_timers.hh")

    # TensorNumerator callbacks.
    path = src / "tensor_ninja.cc"
    text = path.read_text()
    text = replace_once(text, "#include <ninja/tensor_ninja.hh>\n",
                        "#include <ninja/tensor_ninja.hh>\n#include \"ninja_bench_timers.hh\"\n", path)
    count = 0
    for signature in [r"Complex TensorNumerator::evaluate\(",
                      r"void TensorNumerator::muExpansion\(",
                      r"void TensorNumerator::t3Expansion\(",
                      r"void TensorNumerator::t2Expansion\("]:
        match = re.search(signature + r"[^{]*\{\n", text)
        if not match:
            raise RuntimeError(f"{path}: {signature} not found")
        text = text[:match.end()] + "    NINJA_BENCH_SCOPE(NUMERATOR);\n" + text[match.end():]
        count += 1
    assert count == 4
    path.write_text(text)

    # Uncached OneLOop master integrals.
    path = src / "avholo_interface.cc"
    text = path.read_text()
    text = replace_once(text, "#include <ninja/avholo.hh>\n",
                        "#include <ninja/avholo.hh>\n#include \"ninja_bench_timers.hh\"\n", path)
    start = text.index("// OneLoop without cache")
    end = text.index("#endif // ! NINJA_USE_ONELOOP_WITH_CACHE", start)
    section = text[start:end]
    section, n_uncached = re.subn(r"(\n  AvHOneLoop::get\w+Integral(?:CM|RM)\([^{]*\{\n)",
                                  r"\1    NINJA_BENCH_SCOPE(MASTERS);\n", section)
    text = text[:start] + section + text[end:]
    # Scalar bubbles are shared by the cached and uncached builds.
    common_start = text.index("// Common methods of cached and uncached OneLoop")
    common_end = text.index("}  // namespace ninja", common_start)
    common = text[common_start:common_end]
    common, n_common = re.subn(r"(\n  AvHOneLoop::getBubbleIntegral(?:CM|RM)\([^{]*\{\n)",
                               r"\1    NINJA_BENCH_SCOPE(MASTERS);\n", common)
    text = text[:common_start] + common + text[common_end:]
    if n_uncached != 8 or n_common != 2:
        raise RuntimeError(f"{path}: instrumented {n_uncached} uncached and {n_common} common methods")
    path.write_text(text)

    # MadLoop entry point, counters, accessors and optional input dump.
    path = src / "ninja_wraps.cc"
    text = path.read_text()
    text = replace_once(text, "#include <ninja/tensor_ninja.hh>\n",
                        "#include <ninja/tensor_ninja.hh>\n#include \"ninja_bench_timers.hh\"\n"
                        "#include <cstdio>\n#include <cstdlib>\n#include <cstring>\n#include <ctime>\n", path)
    helpers = r'''
#ifndef QUADNINJA_WRAPS
namespace ninja_bench {
  Counters counters;

  inline void dump_masses(std::FILE * f, int n, const ninja::Complex * m)
  {
    for (int i=0; i<n; ++i)
      std::fprintf(f, "%s[%.17g,%.17g]", i ? "," : "", m[i].real(), m[i].imag());
  }
  inline void dump_masses(std::FILE * f, int n, const ninja::Real * m)
  {
    for (int i=0; i<n; ++i)
      std::fprintf(f, "%s[%.17g,0]", i ? "," : "", m[i]);
  }
  template <typename Other>
  inline void dump_masses(std::FILE * f, int n, const Other &)
  {
    for (int i=0; i<n; ++i)
      std::fprintf(f, "%s[0,0]", i ? "," : "");
  }

  template <typename MassPtr>
  void dump_call(const ninja::Complex * tensor, int n, int rank,
                 const ninja::RealMomentum p[], MassPtr m2,
                 const ninja::Real * s_mat, ninja::Real mur2)
  {
    static std::FILE * f = 0;
    static bool checked = false;
    if (!checked) {
      checked = true;
      const char * path = std::getenv("NINJA_BENCH_DUMP");
      if (path && *path)
        f = std::fopen(path, "a");
    }
    if (!f)
      return;
    int ncoeff = (rank+1)*(rank+2)*(rank+3)*(rank+4)/24;
    std::fprintf(f, "{\"n\":%d,\"rank\":%d,\"mu_squared\":%.17g,\"shifts\":[", n, rank, mur2);
    for (int i=0; i<n; ++i)
      std::fprintf(f, "%s[%.17g,%.17g,%.17g,%.17g]", i ? "," : "",
                   p[i][0], p[i][1], p[i][2], p[i][3]);
    std::fprintf(f, "],\"masses_squared\":[");
    dump_masses(f, n, m2);
    std::fprintf(f, "],\"s_matrix\":");
    if (s_mat) {
      std::fputc('[', f);
      for (int i=0; i<n; ++i) {
        std::fprintf(f, "%s[", i ? "," : "");
        for (int j=0; j<n; ++j)
          std::fprintf(f, "%s%.17g", j ? "," : "", s_mat[i*n+j]);
        std::fputc(']', f);
      }
      std::fputc(']', f);
    } else {
      std::fprintf(f, "null");
    }
    std::fprintf(f, ",\"tensor\":[");
    for (int i=0; i<ncoeff; ++i)
      std::fprintf(f, "%s[%.17g,%.17g]", i ? "," : "", tensor[i].real(), tensor[i].imag());
    std::fprintf(f, "]}\n");
    std::fflush(f);
  }
}
#endif
'''
    text = replace_once(text, "using namespace ninja;\n", "using namespace ninja;\n" + helpers, path)
    old = """                             int & return_status)
  {
    TensorNumerator num(n,rank,tensor);"""
    new = """                             int & return_status)
  {
#ifndef QUADNINJA_WRAPS
    ninja_bench::dump_call(tensor, n, rank, p, m2, s_mat, mur2);
    if (n >= 0 && n <= 8 && rank >= 0 && rank <= 10)
      ninja_bench::counters.n_rank_calls[n][rank] += 1;
    NINJA_BENCH_SCOPE(EVALUATE);
#endif
    TensorNumerator num(n,rank,tensor);"""
    text = replace_once(text, old, new, path)
    accessors = r'''
#ifndef QUADNINJA_WRAPS
extern "C" {

  // Reset all counters.
  void ninja_bench_reset_()
  {
    std::memset(&ninja_bench::counters, 0, sizeof(ninja_bench::counters));
  }

  // out[0..3] = cycles of EVALUATE, NUMERATOR, MASTERS, (unused);
  // out[4..7] = number of scope entries of the same buckets.
  void ninja_bench_get_(double out[8])
  {
    for (int i=0; i<ninja_bench::N_BUCKETS; ++i) {
      out[i] = double(ninja_bench::counters.cycles[i]);
      out[4+i] = double(ninja_bench::counters.calls[i]);
    }
  }

  // out[n*11+rank] = number of ninja_tensor_evaluate calls with (n, rank).
  void ninja_bench_get_nrank_(double out[99])
  {
    for (int n=0; n<9; ++n)
      for (int r=0; r<11; ++r)
        out[n*11+r] = double(ninja_bench::counters.n_rank_calls[n][r]);
  }

  // Time-stamp-counter frequency (Hz), calibrated once against
  // CLOCK_MONOTONIC over ~200 ms.
  double ninja_bench_tsc_hz_()
  {
    static double hz = 0.;
    if (hz > 0.)
      return hz;
    timespec a, b;
    clock_gettime(CLOCK_MONOTONIC, &a);
    unsigned long long t0 = __rdtsc();
    do {
      clock_gettime(CLOCK_MONOTONIC, &b);
    } while ((b.tv_sec - a.tv_sec)*1e9 + (b.tv_nsec - a.tv_nsec) < 2e8);
    unsigned long long t1 = __rdtsc();
    hz = double(t1 - t0) / ((b.tv_sec - a.tv_sec) + 1e-9*(b.tv_nsec - a.tv_nsec));
    return hz;
  }

} // extern "C"
#endif
'''
    text = text.rstrip("\n") + "\n" + accessors
    path.write_text(text)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    instrument(pathlib.Path(sys.argv[1]).resolve())
