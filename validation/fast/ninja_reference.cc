// Original Ninja driver for the fast-engine cross-check and timing control.
// Independent validation executable; never linked into the Rust library.
//
// Whitespace-delimited requests on standard input, one per phase-space point:
//   n rank repetitions mu_squared has_s_matrix masters_mode
//   n rows: V0 V1 V2 V3 Re(m^2) Im(m^2)
//   [n*n S-matrix entries if has_s_matrix]
//   binomial(rank+4,4) rows: Re(c) Im(c)   (ninja::TensorNumerator layout)
//   [master table if masters_mode == 1]
// masters_mode: 0 = OneLOop (Ninja's default library); 1 = a table of master
// values supplied with the request (returned by argument lookup, so original
// Ninja reduces with exactly the masters the Rust engine used); 2 = OneLOop,
// printing every master call as a JSON line before the result.
// Table: "nD0 nC0 nBubble nA0", then rows of (re im) pairs: D0 = 11 arguments
// + 1 Laurent triple; C0 = 7 + 1; bubble = 4 + 4 triples (B0, B1, B11, B111);
// A0 = 2 + 1. Arguments include mu^2 last, as for the fast engine.
// Each request prints one JSON line with the Laurent coefficients, the
// rational and cut-constructible parts, the number of master calls without a
// matching table row, and the mean nanoseconds of `repetitions` evaluations
// after one warm-up (Ninja defaults: mu-expansion, no tests).
#include <ninja/ninja.hh>
#include <ninja/tensor_ninja.hh>
#include <ninja/avholo.hh>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

namespace {

typedef ninja::Complex C;

void complex_json(const C &value) {
  if (!std::isfinite(value.real()) || !std::isfinite(value.imag())) {
    std::cout << "null";
  } else {
    std::cout << '[' << value.real() << ',' << value.imag() << ']';
  }
}

struct Row {
  std::vector<C> args;
  std::vector<C> values;  // flattened Laurent triples
  bool used;
};

bool read_rows(std::vector<Row> &rows, unsigned count, unsigned arity, unsigned triples) {
  rows.assign(count, Row());
  for (unsigned r = 0; r < count; ++r) {
    rows[r].used = false;
    for (unsigned i = 0; i < arity + 3 * triples; ++i) {
      double re, im;
      if (!(std::cin >> re >> im)) return false;
      (i < arity ? rows[r].args : rows[r].values).push_back(C(re, im));
    }
  }
  return true;
}

// Master values supplied by the caller, matched by their arguments.
class TableLibrary : public ninja::IntegralLibrary {
 public:
  std::vector<Row> d0, c0, bubbles, a0;
  double mu2;
  unsigned misses;

  void init(ninja::Real muRsq) { mu2 = muRsq; }

  const Row *find(std::vector<Row> &rows, const C *args, unsigned count) {
    for (unsigned pass = 0; pass < 2; ++pass) {
      for (unsigned r = 0; r < rows.size(); ++r) {
        if (pass == 0 && rows[r].used) continue;
        bool same = true;
        for (unsigned i = 0; i < count && same; ++i) {
          const double scale = std::max(1.0, std::abs(args[i]));
          same = std::abs(rows[r].args[i] - args[i]) <= 1e-12 * scale;
        }
        same = same && std::abs(rows[r].args[count] - C(mu2)) <= 1e-12 * mu2;
        if (same) {
          rows[r].used = true;
          return &rows[r];
        }
      }
    }
    ++misses;
    return 0;
  }

  void fill(C *out, const Row *row, unsigned triple) {
    for (unsigned k = 0; k < 3; ++k)
      out[k] = row ? row->values[3 * triple + k] : C(std::nan(""), 0.);
  }

  void getBoxIntegralCM(C rslt[3], ninja::Real s21, ninja::Real s32, ninja::Real s43,
                        ninja::Real s14, ninja::Real s31, ninja::Real s42, const C &m1,
                        const C &m2, const C &m3, const C &m4) {
    const C args[10] = {s21, s32, s43, s14, s31, s42, m1, m2, m3, m4};
    fill(rslt, find(d0, args, 10), 0);
  }
  void getBoxIntegralRM(C rslt[3], ninja::Real s21, ninja::Real s32, ninja::Real s43,
                        ninja::Real s14, ninja::Real s31, ninja::Real s42, ninja::Real m1,
                        ninja::Real m2, ninja::Real m3, ninja::Real m4) {
    getBoxIntegralCM(rslt, s21, s32, s43, s14, s31, s42, m1, m2, m3, m4);
  }
  void getTriangleIntegralCM(C rslt[3], ninja::Real s21, ninja::Real s32, ninja::Real s13,
                             const C &m1, const C &m2, const C &m3) {
    const C args[6] = {s21, s32, s13, m1, m2, m3};
    fill(rslt, find(c0, args, 6), 0);
  }
  void getTriangleIntegralRM(C rslt[3], ninja::Real s21, ninja::Real s32, ninja::Real s13,
                             ninja::Real m1, ninja::Real m2, ninja::Real m3) {
    getTriangleIntegralCM(rslt, s21, s32, s13, m1, m2, m3);
  }
  void getBubbleIntegralCM(C rslt[3], ninja::Real s21, const C &m1, const C &m2) {
    const C args[3] = {s21, m1, m2};
    fill(rslt, find(bubbles, args, 3), 0);
  }
  void getBubbleIntegralRM(C rslt[3], ninja::Real s21, ninja::Real m1, ninja::Real m2) {
    getBubbleIntegralCM(rslt, s21, m1, m2);
  }
  void getRank2BubbleIntegralCM(C b11[3], C b1[3], C b0[3], ninja::Real s21, const C &m1,
                                const C &m2) {
    const C args[3] = {s21, m1, m2};
    const Row *row = find(bubbles, args, 3);
    fill(b0, row, 0);
    fill(b1, row, 1);
    fill(b11, row, 2);
  }
  void getRank2BubbleIntegralRM(C b11[3], C b1[3], C b0[3], ninja::Real s21, ninja::Real m1,
                                ninja::Real m2) {
    getRank2BubbleIntegralCM(b11, b1, b0, s21, m1, m2);
  }
  void getRank3BubbleIntegralCM(C b111[3], C b11[3], C b1[3], C b0[3], ninja::Real s21,
                                const C &m1, const C &m2) {
    const C args[3] = {s21, m1, m2};
    const Row *row = find(bubbles, args, 3);
    fill(b0, row, 0);
    fill(b1, row, 1);
    fill(b11, row, 2);
    fill(b111, row, 3);
  }
  void getRank3BubbleIntegralRM(C b111[3], C b11[3], C b1[3], C b0[3], ninja::Real s21,
                                ninja::Real m1, ninja::Real m2) {
    getRank3BubbleIntegralCM(b111, b11, b1, b0, s21, m1, m2);
  }
  void getTadpoleIntegralCM(C rslt[3], const C &m) {
    const C args[1] = {m};
    fill(rslt, find(a0, args, 1), 0);
  }
  void getTadpoleIntegralRM(C rslt[3], ninja::Real m) { getTadpoleIntegralCM(rslt, m); }
};

// OneLOop (Ninja's default library), printing every master call.
class PrintingLibrary : public ninja::IntegralLibrary {
 public:
  ninja::IntegralLibrary *inner;
  double mu2;
  bool nested;

  void print(const char *family, const C *args, unsigned count, const C *values,
             unsigned triples) {
    if (nested) return;
    std::cout << "{\"master\":\"" << family << "\",\"args\":[";
    for (unsigned i = 0; i < count; ++i) {
      if (i) std::cout << ',';
      complex_json(args[i]);
    }
    std::cout << ",[" << mu2 << ",0]],\"values\":[";
    for (unsigned i = 0; i < 3 * triples; ++i) {
      if (i) std::cout << ',';
      complex_json(values[i]);
    }
    std::cout << "]}\n";
  }

  void init(ninja::Real muRsq) {
    mu2 = muRsq;
    nested = false;
    inner->init(muRsq);
  }
  void exit() { inner->exit(); }
  void getBoxIntegralCM(C rslt[3], ninja::Real s21, ninja::Real s32, ninja::Real s43,
                        ninja::Real s14, ninja::Real s31, ninja::Real s42, const C &m1,
                        const C &m2, const C &m3, const C &m4) {
    inner->getBoxIntegralCM(rslt, s21, s32, s43, s14, s31, s42, m1, m2, m3, m4);
    const C args[10] = {s21, s32, s43, s14, s31, s42, m1, m2, m3, m4};
    print("D0", args, 10, rslt, 1);
  }
  void getBoxIntegralRM(C rslt[3], ninja::Real s21, ninja::Real s32, ninja::Real s43,
                        ninja::Real s14, ninja::Real s31, ninja::Real s42, ninja::Real m1,
                        ninja::Real m2, ninja::Real m3, ninja::Real m4) {
    getBoxIntegralCM(rslt, s21, s32, s43, s14, s31, s42, m1, m2, m3, m4);
  }
  void getTriangleIntegralCM(C rslt[3], ninja::Real s21, ninja::Real s32, ninja::Real s13,
                             const C &m1, const C &m2, const C &m3) {
    inner->getTriangleIntegralCM(rslt, s21, s32, s13, m1, m2, m3);
    const C args[6] = {s21, s32, s13, m1, m2, m3};
    print("C0", args, 6, rslt, 1);
  }
  void getTriangleIntegralRM(C rslt[3], ninja::Real s21, ninja::Real s32, ninja::Real s13,
                             ninja::Real m1, ninja::Real m2, ninja::Real m3) {
    getTriangleIntegralCM(rslt, s21, s32, s13, m1, m2, m3);
  }
  void getBubbleIntegralCM(C rslt[3], ninja::Real s21, const C &m1, const C &m2) {
    inner->getBubbleIntegralCM(rslt, s21, m1, m2);
  }
  void getBubbleIntegralRM(C rslt[3], ninja::Real s21, ninja::Real m1, ninja::Real m2) {
    getBubbleIntegralCM(rslt, s21, m1, m2);
  }
  void getRank2BubbleIntegralCM(C b11[3], C b1[3], C b0[3], ninja::Real s21, const C &m1,
                                const C &m2) {
    inner->getRank2BubbleIntegralCM(b11, b1, b0, s21, m1, m2);
    const C args[3] = {s21, m1, m2};
    const C values[12] = {b0[0], b0[1], b0[2], b1[0], b1[1], b1[2],
                          b11[0], b11[1], b11[2], 0., 0., 0.};
    print("B", args, 3, values, 4);
  }
  void getRank2BubbleIntegralRM(C b11[3], C b1[3], C b0[3], ninja::Real s21, ninja::Real m1,
                                ninja::Real m2) {
    getRank2BubbleIntegralCM(b11, b1, b0, s21, m1, m2);
  }
  void getRank3BubbleIntegralCM(C b111[3], C b11[3], C b1[3], C b0[3], ninja::Real s21,
                                const C &m1, const C &m2) {
    nested = true;
    ninja::IntegralLibrary::getRank3BubbleIntegralCM(b111, b11, b1, b0, s21, m1, m2);
    nested = false;
    const C args[3] = {s21, m1, m2};
    const C values[12] = {b0[0], b0[1], b0[2], b1[0], b1[1], b1[2],
                          b11[0], b11[1], b11[2], b111[0], b111[1], b111[2]};
    print("B", args, 3, values, 4);
  }
  void getRank3BubbleIntegralRM(C b111[3], C b11[3], C b1[3], C b0[3], ninja::Real s21,
                                ninja::Real m1, ninja::Real m2) {
    getRank3BubbleIntegralCM(b111, b11, b1, b0, s21, m1, m2);
  }
  void getTadpoleIntegralCM(C rslt[3], const C &m) {
    inner->getTadpoleIntegralCM(rslt, m);
    const C args[1] = {m};
    print("A0", args, 1, rslt, 1);
  }
  void getTadpoleIntegralRM(C rslt[3], ninja::Real m) { getTadpoleIntegralCM(rslt, m); }
};

}  // namespace

int main() {
  ninja::setVerbosity(ninja::Verbose::NONE);
  ninja::setTest(ninja::Test::NONE);
  std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);
  TableLibrary table;
  PrintingLibrary printing;
  printing.inner = &ninja::avh_olo;
  unsigned n, rank, repetitions, has_matrix, mode;
  double scale;
  while (std::cin >> n) {
    if (!(std::cin >> rank >> repetitions >> scale >> has_matrix >> mode) || n == 0 ||
        rank > n + 1 || mode > 2) {
      std::cerr << "invalid request header\n";
      return 2;
    }
    std::vector<ninja::RealMomentum> shifts(n);
    std::vector<C> masses(n);
    for (unsigned i = 0; i < n; ++i) {
      double p[4], re, im;
      for (unsigned j = 0; j < 4; ++j) std::cin >> p[j];
      std::cin >> re >> im;
      shifts[i] = ninja::RealMomentum(p[0], p[1], p[2], p[3]);
      masses[i] = C(re, im);
    }
    std::vector<ninja::Real> matrix(has_matrix ? n * n : 0);
    for (unsigned i = 0; i < matrix.size(); ++i) std::cin >> matrix[i];
    const unsigned count = (rank + 1) * (rank + 2) * (rank + 3) * (rank + 4) / 24;
    std::vector<C> coefficients(count);
    for (unsigned i = 0; i < count; ++i) {
      double re, im;
      std::cin >> re >> im;
      coefficients[i] = C(re, im);
    }
    if (mode == 1) {
      unsigned nd0, nc0, nb, na0;
      std::cin >> nd0 >> nc0 >> nb >> na0;
      if (!read_rows(table.d0, nd0, 11, 1) || !read_rows(table.c0, nc0, 7, 1) ||
          !read_rows(table.bubbles, nb, 4, 4) || !read_rows(table.a0, na0, 2, 1)) {
        std::cerr << "truncated master table\n";
        return 2;
      }
      table.misses = 0;
    }
    if (!std::cin) {
      std::cerr << "truncated request\n";
      return 2;
    }
    ninja::TensorNumerator numerator(n, rank, coefficients.data());
    ninja::Amplitude<ninja::ComplexMasses> amplitude(n, rank, shifts.data(), masses.data());
    amplitude.setRenormalizationScale(scale);
    if (has_matrix) amplitude.setSMatrix(matrix.data());
    if (mode == 1) amplitude.setIntegralLibrary(table);
    if (mode == 2) amplitude.setIntegralLibrary(printing);
    int status = amplitude.evaluate(numerator);
    const C eps[3] = {amplitude[0], amplitude[1], amplitude[2]};
    const C rational = amplitude.getRationalPart();
    const C cut = amplitude.getCutConstructiblePart();
    const unsigned misses = mode == 1 ? table.misses : 0;
    if (mode == 2) amplitude.setIntegralLibrary(ninja::avh_olo);
    const auto start = std::chrono::steady_clock::now();
    for (unsigned i = 0; i < repetitions; ++i) {
      amplitude.reset();
      status |= amplitude.evaluate(numerator);
    }
    const auto stop = std::chrono::steady_clock::now();
    const double ns = repetitions
        ? std::chrono::duration<double, std::nano>(stop - start).count() / repetitions : 0.;
    std::cout << "{\"status\":" << status << ",\"epsilon\":[";
    for (unsigned i = 0; i < 3; ++i) {
      if (i) std::cout << ',';
      complex_json(eps[i]);
    }
    std::cout << "],\"rational\":";
    complex_json(rational);
    std::cout << ",\"cut_constructible\":";
    complex_json(cut);
    std::cout << ",\"master_misses\":" << misses
              << ",\"nanoseconds_per_evaluation\":" << ns << "}\n" << std::flush;
  }
  return std::cin.eof() ? 0 : 2;
}
