// Independent validation executable. Never linked into the Rust implementation.
#include <ninja/ninja.hh>
#include <ninja/tensor_ninja.hh>
#include <ninja/avholo.hh>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#ifdef NINJARS_GENERATED_NUMERATOR_HEADER
#include NINJARS_GENERATED_NUMERATOR_HEADER
#endif

namespace {
bool read_real(double &value) {
  return bool(std::cin >> value) && std::isfinite(value);
}
void complex_json(const ninja::Complex &value) {
  if (!std::isfinite(value.real()) || !std::isfinite(value.imag())) {
    std::cout << "null";
  } else {
    std::cout << '[' << value.real() << ',' << value.imag() << ']';
  }
}
}

int main(int argc, char **argv) {
  int cut_stop = 1;
  bool mu_expansion = true;
  unsigned matrix_size = 0;
  std::vector<ninja::Real> matrix;
  for (int arg = 1; arg < argc; ++arg) {
    const std::string option(argv[arg]);
    if (option == "--cut-stop" && arg + 1 < argc) {
      char *end = 0;
      const long value = std::strtol(argv[++arg], &end, 10);
      if (!*argv[arg] || *end || value < 0 || value > 6) return 2;
      cut_stop = static_cast<int>(value);
    } else if (option == "--mu-sampling") {
      mu_expansion = false;
    } else if (option == "--s-matrix" && arg + 1 < argc) {
      std::ifstream input(argv[++arg]);
      if (!(input >> matrix_size) || matrix_size == 0 || matrix_size > 6) return 2;
      matrix.resize(matrix_size * matrix_size);
      for (unsigned i = 0; i < matrix.size(); ++i) {
        if (!(input >> matrix[i]) || !std::isfinite(matrix[i])) return 2;
      }
      std::string extra;
      if (input >> extra) return 2;
      for (unsigned i = 0; i < matrix_size; ++i) {
        if (matrix[i * matrix_size + i] != 0) return 2;
        for (unsigned j = 0; j < i; ++j) {
          if (matrix[i * matrix_size + j] != matrix[j * matrix_size + i]) return 2;
        }
      }
    } else {
      std::cerr << "invalid runner option\n";
      return 2;
    }
  }
  ninja::setVerbosity(ninja::Verbose::NONE);
  ninja::setTest(ninja::Test::NONE);
  std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);
  // Requests are whitespace-delimited and can be streamed through one process.
  // n rank repetitions mu_R^2, then n rows V0 V1 V2 V3 Re(m^2) Im(m^2),
  // followed by binomial(rank+4,4) complex tensor coefficients.
  unsigned n, rank, repetitions;
  double scale;
  while (std::cin >> n) {
    // Original Ninja's D0 key omits s14. Never reuse it across changed
    // phase-space requests. Within a request the warm/timed inputs are identical.
    ninja::avh_olo.clearIntegralCache();
    if (!(std::cin >> rank >> repetitions) || !read_real(scale) || n == 0 || n > 6 || rank > n + 1 ||
        repetitions == 0 || scale <= 0) {
      std::cerr << "invalid request header\n";
      return 2;
    }
#ifdef NINJARS_GENERATED_NUMERATOR_HEADER
    if (n != ninjars_reference::propagator_count || rank != ninjars_reference::rank) {
      std::cerr << "request topology/rank does not match the generated numerator\n";
      return 2;
    }
#endif
    std::vector<ninja::RealMomentum> shifts(n);
    std::vector<ninja::Complex> masses(n);
    for (unsigned i = 0; i < n; ++i) {
      double p[4], re, im;
      for (unsigned j = 0; j < 4; ++j) {
        if (!read_real(p[j])) return 2;
      }
      if (!read_real(re) || !read_real(im)) return 2;
      shifts[i] = ninja::RealMomentum(p[0], p[1], p[2], p[3]);
      masses[i] = ninja::Complex(re, im);
    }
#ifdef NINJARS_GENERATED_NUMERATOR_HEADER
    const unsigned count = ninjars_reference::parameter_count;
#else
    const unsigned count = (rank+1)*(rank+2)*(rank+3)*(rank+4)/24;
#endif
    std::vector<ninja::Complex> coefficients(count);
    for (unsigned i = 0; i < count; ++i) {
      double re, im;
      if (!read_real(re) || !read_real(im)) return 2;
      coefficients[i] = ninja::Complex(re, im);
    }
#ifdef NINJARS_GENERATED_NUMERATOR_HEADER
    ninjars_reference::GeneratedNumerator numerator(coefficients.data());
#else
    ninja::TensorNumerator numerator(n, rank, coefficients.data());
#endif
    ninja::Amplitude<ninja::ComplexMasses> amplitude(n, rank, shifts.data(), masses.data());
    amplitude.setRenormalizationScale(scale);
    amplitude.setCutStop(cut_stop).useMuExpansion(mu_expansion);
    if (matrix_size) {
      if (matrix_size != n) return 2;
      amplitude.setSMatrix(matrix.data());
    }
    const int warm_status = amplitude.evaluate(numerator);
    int status = warm_status;
    const auto start = std::chrono::steady_clock::now();
    for (unsigned i = 0; i < repetitions; ++i) {
      amplitude.reset();
      status |= amplitude.evaluate(numerator);
    }
    const auto stop = std::chrono::steady_clock::now();
    const double nanoseconds = std::chrono::duration<double, std::nano>(stop-start).count()/repetitions;
    std::cout << "{\"status\":" << status << ",\"epsilon\":[";
    for (unsigned i = 0; i < 3; ++i) {
      if (i) std::cout << ',';
      complex_json(amplitude[i]);
    }
    std::cout << "],\"rational\":";
    complex_json(amplitude.getRationalPart());
    std::cout << ",\"cut_constructible\":";
    complex_json(amplitude.getCutConstructiblePart());
    std::cout << ",\"nanoseconds_per_evaluation\":" << nanoseconds
              << ",\"repetitions\":" << repetitions << "}\n" << std::flush;
  }
  return std::cin.eof() ? 0 : 2;
}
