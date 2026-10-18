// Validation-only scalar baseline. Never linked into the Rust implementation.
#include <config.h>
#if defined(NINJA_USE_ONELOOP_WITH_CACHE)
#error "The benchmark requires --disable-avholo_cache"
#endif
#if !defined(NINJA_USE_ONELOOP) || !defined(NINJA_X1RANK)
#error "The benchmark requires OneLOop and higher-rank support"
#endif
#ifndef NINJARS_GENERATED_NUMERATOR_HEADER
#error "The benchmark requires a common generated numerator"
#endif

#include <ninja/ninja.hh>
#include NINJARS_GENERATED_NUMERATOR_HEADER
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
typedef std::uint64_t Count;
typedef ninja::Amplitude<ninja::ComplexMasses> Amplitude;

Count count(const std::string &token) {
  if (token.empty() || token.find_first_not_of("0123456789") != std::string::npos)
    throw std::invalid_argument("expected an unsigned decimal integer");
  errno = 0;
  char *end = 0;
  const unsigned long long value = std::strtoull(token.c_str(), &end, 10);
  if (errno == ERANGE || *end || value > std::numeric_limits<Count>::max())
    throw std::invalid_argument("integer overflow");
  return static_cast<Count>(value);
}
Count read_count() {
  std::string token;
  if (!(std::cin >> token)) throw std::invalid_argument("truncated integer input");
  return count(token);
}
double read_real() {
  double value;
  if (!(std::cin >> value) || !std::isfinite(value))
    throw std::invalid_argument("expected a finite binary64 number");
  return value;
}
ninja::Complex read_complex() {
  const double real = read_real();
  const double imaginary = read_real();
  return ninja::Complex(real, imaginary);
}
std::size_t size(Count value) {
  if (value > std::numeric_limits<std::size_t>::max())
    throw std::invalid_argument("size overflow");
  return static_cast<std::size_t>(value);
}
Count product(Count first, Count second) {
  if (second && first > std::numeric_limits<Count>::max() / second)
    throw std::invalid_argument("evaluation count overflow");
  return first * second;
}

struct Sample {
  double scale;
  std::vector<ninja::RealMomentum> shifts;
  std::vector<ninja::Complex> masses, parameters;
  std::vector<ninja::Real> matrix;
};
struct Job {
  Count id, workload, points, warmup, rounds, iterations;
};
struct Result {
  int status;
  // Original Ninja order followed by its two finite-part components.
  std::array<ninja::Complex, 5> values;
  bool finite() const {
    for (std::size_t i = 0; i < values.size(); ++i)
      if (!std::isfinite(values[i].real()) || !std::isfinite(values[i].imag())) return false;
    return true;
  }
  bool valid() const { return status == Amplitude::SUCCESS && finite(); }
};
struct Timing {
  Job job;
  Count warmup_failed;
  std::vector<double> round_ns;
};

// The callback owns scratch buffers and borrows the current immutable parameter
// row. No object construction, parsing or sample packing enters this function.
int evaluate(Amplitude &amplitude, ninjars_reference::GeneratedNumerator &numerator,
             Sample &sample) {
  numerator.set_parameters(sample.parameters.data());
  amplitude.setKinematics(sample.shifts.data()).setInternalMasses(sample.masses.data());
  amplitude.setRenormalizationScale(sample.scale);
  amplitude.setSMatrix(sample.matrix.empty() ? 0 : sample.matrix.data());
  amplitude.reset();
  return amplitude.evaluate(numerator);
}
Result capture(const Amplitude &amplitude, int status) {
  Result result;
  result.status = status;
  for (unsigned i = 0; i < 3; ++i) result.values[i] = amplitude[i];
  result.values[3] = amplitude.getCutConstructiblePart();
  result.values[4] = amplitude.getRationalPart();
  return result;
}
// Equivalent scope to Rust black_box: consume the stored outputs without a
// numerical checksum or timer instrumentation inside the primary loop.
void consume(const Amplitude &amplitude, int status) {
  asm volatile("" : : "g"(&amplitude), "g"(status) : "memory");
}

void complex_json(const ninja::Complex &value) {
  if (!std::isfinite(value.real()) || !std::isfinite(value.imag())) std::cout << "null";
  else std::cout << '[' << value.real() << ',' << value.imag() << ']';
}
void results_json(const std::vector<Result> &results) {
  std::cout << '[';
  for (std::size_t lane = 0; lane < results.size(); ++lane) {
    if (lane) std::cout << ',';
    const Result &result = results[lane];
    std::cout << "{\"lane\":" << lane << ",\"status\":" << result.status
              << ",\"finite\":" << (result.finite() ? "true" : "false") << ",\"epsilon\":[";
    for (unsigned i = 0; i < 3; ++i) {
      if (i) std::cout << ',';
      complex_json(result.values[i]);
    }
    std::cout << "],\"cut_constructible\":";
    complex_json(result.values[3]);
    std::cout << ",\"rational\":";
    complex_json(result.values[4]);
    std::cout << '}';
  }
  std::cout << ']';
}
// FNV-1a on explicit little-endian words, independent of host byte order.
// This is a bitwise result fingerprint, not a cryptographic file digest.
std::string checksum(const std::vector<Result> &results) {
  Count hash = UINT64_C(14695981039346656037);
  const auto word = [&hash](Count value) {
    for (unsigned i = 0; i < 8; ++i) {
      hash ^= (value >> (8 * i)) & 255;
      hash *= UINT64_C(1099511628211);
    }
  };
  word(results.size());
  for (std::size_t row = 0; row < results.size(); ++row) {
    word(row);
    word(static_cast<Count>(results[row].status));
    for (unsigned value = 0; value < 5; ++value) {
      const double components[] = {results[row].values[value].real(), results[row].values[value].imag()};
      for (unsigned component = 0; component < 2; ++component) {
        Count bits;
        static_assert(sizeof(bits) == sizeof(components[component]), "binary64 required");
        std::memcpy(&bits, &components[component], sizeof(bits));
        word(bits);
      }
    }
  }
  std::ostringstream text;
  text << std::hex << std::setfill('0') << std::setw(16) << hash;
  return text.str();
}

int run(int argc, char **argv) {
  Count cut_stop = 1;
  bool mu_expansion = true;
  for (int arg = 1; arg < argc; ++arg) {
    const std::string option(argv[arg]);
    if (option == "--cut-stop" && arg + 1 < argc) {
      cut_stop = count(argv[++arg]);
      if (cut_stop > 6) throw std::invalid_argument("cut stop must lie in 0..6");
    } else if (option == "--mu-sampling") mu_expansion = false;
    else throw std::invalid_argument("unsupported benchmark option");
  }
  std::string magic;
  if (!(std::cin >> magic) || magic != "NINJARS_BENCH_V1")
    throw std::invalid_argument("invalid benchmark protocol version");
  const Count n = read_count(), rank = read_count(), parameters = read_count(), rows = read_count();
  if (n == 0 || rank > n + 1 || rows == 0
      || n != ninjars_reference::propagator_count || rank != ninjars_reference::rank
      || parameters != ninjars_reference::parameter_count)
    throw std::invalid_argument("pool shape does not match the compiled numerator");
  std::vector<Sample> pool(size(rows));
  for (std::size_t row = 0; row < pool.size(); ++row) {
    Sample &sample = pool[row];
    sample.scale = read_real();
    const Count has_matrix = read_count();
    if (sample.scale <= 0 || has_matrix > 1)
      throw std::invalid_argument("scale must be positive and matrix flag must be 0 or 1");
    sample.shifts.reserve(size(n));
    sample.masses.reserve(size(n));
    for (Count propagator = 0; propagator < n; ++propagator) {
      double components[4];
      for (unsigned c = 0; c < 4; ++c) components[c] = read_real();
      sample.shifts.push_back(ninja::RealMomentum(components[0], components[1], components[2], components[3]));
      sample.masses.push_back(read_complex());
    }
    sample.parameters.reserve(size(parameters));
    for (Count parameter = 0; parameter < parameters; ++parameter)
      sample.parameters.push_back(read_complex());
    if (has_matrix) {
      sample.matrix.reserve(size(n * n));
      for (Count i = 0; i < n * n; ++i) sample.matrix.push_back(read_real());
      for (Count i = 0; i < n; ++i) {
        if (sample.matrix[size(i * n + i)] != 0)
          throw std::invalid_argument("S-matrix diagonal must be zero");
        for (Count j = 0; j < i; ++j)
          if (sample.matrix[size(i * n + j)] != sample.matrix[size(j * n + i)])
            throw std::invalid_argument("S-matrix must be symmetric");
      }
    }
  }
  const Count job_count = read_count();
  if (!job_count) throw std::invalid_argument("at least one benchmark job is required");
  std::vector<Job> jobs(size(job_count));
  std::set<Count> identifiers;
  for (std::size_t i = 0; i < jobs.size(); ++i) {
    Job &job = jobs[i];
    job.id = read_count(); job.workload = read_count(); job.points = read_count();
    job.warmup = read_count(); job.rounds = read_count(); job.iterations = read_count();
    if (!identifiers.insert(job.id).second || job.workload > 1 || !job.points
        || !job.warmup || !job.rounds || !job.iterations
        || (job.workload && (rows < 2 || job.points < 2)))
      throw std::invalid_argument("invalid or duplicate benchmark job");
    product(job.points, job.warmup);
    product(product(job.points, job.iterations), job.rounds);
  }
  std::string trailing;
  if (std::cin >> trailing) throw std::invalid_argument("unexpected trailing protocol data");
  if (!std::cin.eof()) throw std::invalid_argument("input stream failure");

  ninja::setVerbosity(ninja::Verbose::NONE);
  ninja::setTest(ninja::Test::NONE);
  ninjars_reference::GeneratedNumerator numerator(pool[0].parameters.data());
  Amplitude amplitude(static_cast<int>(n), static_cast<int>(rank), pool[0].shifts.data(), pool[0].masses.data());
  amplitude.setCutStop(static_cast<int>(cut_stop)).useMuExpansion(mu_expansion);
  std::vector<Result> before, after;
  bool valid = true;
  for (std::size_t row = 0; row < pool.size(); ++row) {
    before.push_back(capture(amplitude, evaluate(amplitude, numerator, pool[row])));
    valid = valid && before.back().valid();
  }
  std::vector<Timing> timings;
  if (valid) {
    for (std::size_t index = 0; index < jobs.size(); ++index) {
      const Job &job = jobs[index];
      // Sequence construction and modulo arithmetic are outside timed loops.
      std::vector<Sample *> sequence;
      sequence.reserve(size(job.points));
      for (Count point = 0; point < job.points; ++point)
        sequence.push_back(&pool[size(job.workload ? point % rows : 0)]);
      Timing timing;
      timing.job = job;
      timing.warmup_failed = 0;
      timing.round_ns.reserve(size(job.rounds));
      for (Count pass = 0; pass < job.warmup; ++pass) {
        for (std::size_t point = 0; point < sequence.size(); ++point) {
          const int status = evaluate(amplitude, numerator, *sequence[point]);
          if (pass + 1 == job.warmup && !capture(amplitude, status).valid()) ++timing.warmup_failed;
        }
      }
      if (timing.warmup_failed) valid = false;
      if (valid) {
        for (Count round = 0; round < job.rounds; ++round) {
          const auto start = std::chrono::steady_clock::now();
          for (Count iteration = 0; iteration < job.iterations; ++iteration)
            for (std::size_t point = 0; point < sequence.size(); ++point) {
              const int status = evaluate(amplitude, numerator, *sequence[point]);
              consume(amplitude, status);
            }
          const auto stop = std::chrono::steady_clock::now();
          timing.round_ns.push_back(std::chrono::duration<double, std::nano>(stop - start).count());
        }
      }
      timings.push_back(timing);
      if (!valid) break;
    }
    for (std::size_t row = 0; row < pool.size(); ++row) {
      after.push_back(capture(amplitude, evaluate(amplitude, numerator, pool[row])));
      valid = valid && after.back().valid();
    }
  }
  const std::string before_checksum = checksum(before), after_checksum = checksum(after);
  const bool unchanged = before_checksum == after_checksum;
  valid = valid && unchanged;
  std::cout << std::scientific << std::setprecision(std::numeric_limits<double>::max_digits10)
            << "{\"schema_version\":1,\"operation\":\"original_ninja_benchmark\",\"success\":"
            << (valid ? "true" : "false")
            << ",\"performance_acceptance\":false,\"execution\":\"persistent serial scalar evaluations\","
            << "\"cache_policy\":{\"numerical_master_results\":\"disabled at configure time\"},"
            << "\"measurement_scope\":\"parameter and sample-pointer rebinding, scale and S-matrix controls, reset, original Ninja reduction and OneLOop masters; no parsing, setup, sequence packing, JSON, IPC or checksum\","
            << "\"status_scope\":\"all source rows before and after timing; final warmup passes; timed outputs consumed without per-iteration validation\","
            << "\"epsilon_order\":[\"finite\",\"single_pole\",\"double_pole\"],"
            << "\"coefficient_mode\":\"integrated\",\"mu_extraction\":\"" << (mu_expansion ? "expansion" : "sampling")
            << "\",\"sampling_scale_policy\":\"original Ninja per-box max of real masses-squared and absolute S-matrix invariants; fallback below INFRARED_EPS\","
            << "\"cut_stop\":" << cut_stop << ",\"propagators\":" << n << ",\"rank\":" << rank
            << ",\"parameter_count\":" << parameters << ",\"pool_rows\":" << rows
            << ",\"thread_count\":1,\"precision_rescue\":{\"enabled\":false,\"attempts\":0},"
            << "\"checksum_algorithm\":\"FNV1a64 over little-endian row count, row IDs, statuses and binary64 components\","
            << "\"pre_checksum\":\"" << before_checksum << "\",\"post_checksum\":\"" << after_checksum
            << "\",\"pre_post_equal\":" << (unchanged ? "true" : "false") << ",\"pre_results\":";
  results_json(before);
  std::cout << ",\"post_results\":";
  results_json(after);
  std::cout << ",\"jobs\":[";
  for (std::size_t i = 0; i < timings.size(); ++i) {
    if (i) std::cout << ',';
    const Timing &timing = timings[i];
    const Job &job = timing.job;
    std::cout << "{\"job_id\":" << job.id << ",\"workload\":\"" << (job.workload ? "changing" : "repeated")
              << "\",\"pool_policy\":\"" << (job.workload ? "cyclic fixture rows in fixture order" : "fixture row zero repeated")
              << "\",\"points_per_iteration\":" << job.points << ",\"warmup_passes\":" << job.warmup
              << ",\"rounds\":" << job.rounds << ",\"iterations_per_round\":" << job.iterations
              << ",\"timed_points_per_round\":" << product(job.points, job.iterations)
              << ",\"warmup_final_pass_failed_rows\":" << timing.warmup_failed << ",\"round_ns\":[";
    for (std::size_t round = 0; round < timing.round_ns.size(); ++round) {
      if (round) std::cout << ',';
      std::cout << timing.round_ns[round];
    }
    std::cout << "]}";
  }
  std::cout << "]}\n";
  return valid ? 0 : 1;
}
} // namespace

int main(int argc, char **argv) {
  try { return run(argc, argv); }
  catch (const std::exception &error) {
    std::cerr << "reference benchmark: " << error.what() << '\n';
    return 2;
  }
}
