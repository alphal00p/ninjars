#pragma once
#include <cstdlib>
#include <dlfcn.h>
#include <stdexcept>
// Resolve before sampling; one call at either boundary, none in the hot loop.
struct Diagnostic {
  using Function = int (*)();
  Function start_ = nullptr, stop_ = nullptr;
  Diagnostic() {
    if (!std::getenv("NINJARS_PROFILE_OUTPUT")) return;
    start_ = reinterpret_cast<Function>(dlsym(RTLD_DEFAULT, "sp_kernel_profile_start"));
    stop_ = reinterpret_cast<Function>(dlsym(RTLD_DEFAULT, "sp_kernel_profile_stop"));
    if (!start_ || !stop_) throw std::runtime_error("diagnostic library hooks missing");
  }
  void start() { if (start_ && start_()) throw std::runtime_error("diagnostic start failed"); }
  void stop() { if (stop_ && stop_()) throw std::runtime_error("diagnostic stop failed"); }
};
