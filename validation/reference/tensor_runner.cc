// Independent validation executable. Never linked into the Rust implementation.
#include <ninja/ninja.hh>
#include <ninja/tensor_ninja.hh>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

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

int main() {
  ninja::setVerbosity(ninja::Verbose::NONE);
  ninja::setTest(ninja::Test::NONE);
  std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);
  // Requests are whitespace-delimited and can be streamed through one process.
  // n rank repetitions mu_R^2, then n rows V0 V1 V2 V3 Re(m^2) Im(m^2),
  // followed by binomial(rank+4,4) complex tensor coefficients.
  unsigned n, rank, repetitions;
  double scale;
  while (std::cin >> n >> rank >> repetitions) {
    if (!read_real(scale) || n == 0 || n > 6 || rank > n + 1 ||
        repetitions == 0 || scale <= 0) {
      std::cerr << "invalid request header\n";
      return 2;
    }
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
    const unsigned count = (rank+1)*(rank+2)*(rank+3)*(rank+4)/24;
    std::vector<ninja::Complex> coefficients(count);
    for (unsigned i = 0; i < count; ++i) {
      double re, im;
      if (!read_real(re) || !read_real(im)) return 2;
      coefficients[i] = ninja::Complex(re, im);
    }
    ninja::TensorNumerator numerator(n, rank, coefficients.data());
    ninja::Amplitude<ninja::ComplexMasses> amplitude(n, rank, shifts.data(), masses.data());
    amplitude.setRenormalizationScale(scale);
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
