#pragma once
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

struct Control {
  std::FILE *write = nullptr, *read = nullptr;
  Control(const char *command, const char *ack) {
    if (std::string(command) == "-") return;
    write = std::fopen(command, "w"); read = std::fopen(ack, "r");
    if (!write || !read) throw std::runtime_error("cannot open perf control FIFOs");
  }
  ~Control() { if (write) std::fclose(write); if (read) std::fclose(read); }
  void send(const char *command) {
    if (!write) return;
    if (std::fprintf(write, "%s\n", command) < 0 || std::fflush(write))
      throw std::runtime_error("perf control write failed");
    // perf writes sizeof(EVLIST_CTL_CMD_ACK_TAG), including the trailing NUL.
    // Reading a line leaves that NUL to corrupt the next acknowledgement.
    constexpr char expected[] = "ack\n";
    char response[sizeof(expected)];
    if (std::fread(response, 1, sizeof(response), read) != sizeof(response)
        || std::memcmp(response, expected, sizeof(expected)) != 0)
      throw std::runtime_error("perf control acknowledgement failed");
  }
};

