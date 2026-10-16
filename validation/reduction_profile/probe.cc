// Validation-only external CPU profile boundary. No algebra or production hooks.
#include "physical_workload.hh"
#include <chrono>
#include <cstdio>
#include <numeric>

static std::vector<Point> read_pool() {
  size_t count = 0;
  std::cin >> count;
  if (count != 3) throw std::runtime_error("three physical points required");
  std::vector<Point> pool(count);
  for (auto &point : pool) {
    for (auto &value : point.momenta) std::cin >> value;
    for (auto &value : point.helicities) {
      int h; std::cin >> h; value = static_cast<int8_t>(h);
    }
    for (auto &value : point.model) std::cin >> value;
    std::cin >> point.scale;
    size_t samples = 0; std::cin >> samples;
    if (samples != 70) throw std::runtime_error("missing independent raw grid");
    point.raw_inputs.resize(samples); point.raw_expected.resize(samples);
    for (size_t i = 0; i < samples; ++i) {
      for (unsigned j = 0; j < 4; ++j)
        std::cin >> point.raw_inputs[i][j].re >> point.raw_inputs[i][j].im;
      std::cin >> point.raw_expected[i].re >> point.raw_expected[i].im;
    }
    for (auto &value : point.integrated) std::cin >> value.re >> value.im;
  }
  if (!std::cin) throw std::runtime_error("truncated physical pool");
  return pool;
}

#include "control.hh"
#include "diagnostic.hh"

int main(int argc, char **argv) {
  try {
    if (argc != 6) throw std::runtime_error("usage: probe DOT width rounds control-fifo|- ack-fifo|-");
    const size_t width = std::stoul(argv[2]), rounds = std::stoul(argv[3]);
    if (!width || width > 1024 || !rounds) throw std::runtime_error("invalid profile dimensions");
    ninja::setVerbosity(ninja::Verbose::NONE); ninja::setTest(ninja::Test::NONE);
    auto pool = read_pool(); Physical physical(argv[1], 1); SharedMasters replay; Runs runs(physical);
    std::vector<size_t> all{0,1,2};
    const auto original = runs.run_original(pool, all, replay);
    const auto rust = runs.run_rust(pool, all, replay);
    for (size_t row = 0; row < 3; ++row) for (size_t j = 0; j < 3; ++j) {
      near(complex(rust[row*5+j]), complex(original[row*5+j]), "capture paired Laurent");
      near(complex(rust[row*5+j]), complex(pool[row].integrated[j]), "capture independent Laurent");
    }
    checked(sp_master_freeze(replay.handle()));
    std::vector<size_t> indices(width, 0);
    physical.prepare(pool, indices);
    auto verify = [&]() {
      const auto &values = runs.reduce_rust(&replay);
      for (size_t row = 0; row < width; ++row) {
        for (size_t j = 0; j < 3; ++j)
          near(complex(values[row*5+j]), complex(pool[0].integrated[j]), "profile Laurent");
        near(complex(values[row*5]), complex(values[row*5+3])+complex(values[row*5+4]), "profile finite split");
      }
    };
    for (unsigned i = 0; i < 8; ++i) verify();
    std::array<uint64_t,8> numerator_before{}, numerator_after{};
    std::array<uint64_t,4> session_before{}, session_after{};
    checked(sp_provider_stats(physical.provider,numerator_before.data(),8));
    checked(sp_session_stats(runs.session.handle,session_before.data(),4));
    Control control(argv[4],argv[5]); Diagnostic diagnostic;
    double checksum = 0.;
    control.send("enable"); diagnostic.start();
    const auto start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < rounds; ++i) checksum += runs.reduce_rust(&replay)[0].re;
    const auto end = std::chrono::steady_clock::now();
    diagnostic.stop(); control.send("disable");
    checked(sp_provider_stats(physical.provider,numerator_after.data(),8));
    checked(sp_session_stats(runs.session.handle,session_after.data(),4));
    verify();
    std::array<uint64_t,7> masters{}; checked(sp_master_stats(replay.handle(),masters.data(),7));
    if (masters[6] || !std::isfinite(checksum)) throw std::runtime_error("profile failure or frozen union miss");
    std::cout << std::setprecision(17) << "{\"success\":true,\"performance_acceptance\":false,\"scope\":\"shared-reduction-repeated-fixed-point\",\"width\":" << width << ",\"rounds\":" << rounds << ",\"wall_ns_diagnostic\":" << std::chrono::duration<double,std::nano>(end-start).count() << ",\"checksum\":" << checksum << ",\"numerator_counter_deltas\":[";
    for (size_t i=0;i<8;++i) { if(i) std::cout<<','; std::cout<<numerator_after[i]-numerator_before[i]; }
    std::cout << "],\"session_counter_deltas\":[";
    for (size_t i=0;i<4;++i) { if(i) std::cout<<','; std::cout<<session_after[i]-session_before[i]; }
    std::cout << "],\"union_misses\":" << masters[6] << "}\n";
  } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
