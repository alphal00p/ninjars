// Validation only: the unchanged Ninja OneLOop wrapper, with explicit D0 order.
#include <ninja/avholo.hh>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

int main(int argc, char **argv) {
  const bool keep_cache = argc == 2 && std::string(argv[1]) == "--keep-cache";
  ninja::AvHOneLoop library;
  std::cout << std::setprecision(17);
  // s21,s32,s43,s14,s31,s42; four (Re m²,Im m²) pairs; mu_R².
  double s[6], scale;
  while (std::cin >> s[0]) {
    for (int i=1;i<6;++i) if (!(std::cin >> s[i])) return 2;
    ninja::Complex masses[4], output[3];
    for (int i=0;i<4;++i) {double re,im; if (!(std::cin >> re >> im)) return 2; masses[i]=ninja::Complex(re,im);}
    if (!(std::cin >> scale) || !(scale>0)) return 2;
    library.init(scale);
    library.getBoxIntegralCM(output,s[0],s[1],s[2],s[3],s[4],s[5],masses[0],masses[1],masses[2],masses[3]);
    std::cout << "{\"epsilon\":[";
    for (int i=0;i<3;++i) {
      if (i) std::cout << ',';
      if (std::isfinite(output[i].real()) && std::isfinite(output[i].imag())) std::cout << '[' << output[i].real() << ',' << output[i].imag() << ']';
      else std::cout << "null";
    }
    std::cout << "]}\n" << std::flush;
    if (!keep_cache) library.clearIntegralCache();
  }
  library.freeIntegralCache();
}
