// Calls original Ninja's (FORM-generated) higher-rank coefficient-level
// subtractions on random inputs read from stdin, prints the produced -divnum.
#include <ninja/types.hh>
#include <ninja/momentum.hh>
#include <coefficient_level_subs.hh>
#include <iostream>
#include <iomanip>
#include <string>
using namespace ninja;

static Complex rc() { double a, b; std::cin >> a >> b; return Complex(a, b); }
static Real rr() { double a; std::cin >> a; return a; }
static ComplexMomentum rcm() { Complex a=rc(), b=rc(), c=rc(), d=rc(); return ComplexMomentum(a,b,c,d); }
static RealMomentum rrm() { Real a=rr(), b=rr(), c=rr(), d=rr(); return RealMomentum(a,b,c,d); }

int main() {
  std::cout << std::setprecision(17);
  std::string kind;
  while (std::cin >> kind) {
    if (kind == "tribub") {           // x1cuts::correctbubcoeffs
      int all; std::cin >> all;
      ComplexMomentum et3 = rcm(), et4 = rcm();
      Complex c[15]; for (int i=0;i<15;++i) c[i]=rc();
      ComplexMomentum eb1 = rcm(), eb2 = rcm(), eb3 = rcm(), eb4 = rcm();
      Complex param[3]; for (int i=0;i<3;++i) param[i]=rc();
      RealMomentum k = rrm(); Complex f = rc();
      Complex divnum[13]; for (int i=0;i<13;++i) divnum[i]=0.;
      x1cuts::correctbubcoeffs(divnum, et3, et4, c, eb1, eb2, eb3, eb4, param, k, f, 1, all!=0);
      for (int i=0;i<13;++i) std::cout << -real(divnum[i]) << " " << -imag(divnum[i]) << " ";
      std::cout << "\n";
    } else if (kind == "tritad" || kind == "tritadfull") {
      ComplexMomentum et3 = rcm(), et4 = rcm();
      Complex c[15]; for (int i=0;i<15;++i) c[i]=rc();
      ComplexMomentum e3 = rcm(), e4 = rcm(); Complex param = rc();
      RealMomentum k0 = rrm(); Complex f0 = rc(); RealMomentum k1 = rrm(); Complex f1 = rc();
      Complex divnum[4]; for (int i=0;i<4;++i) divnum[i]=0.;
      if (kind == "tritad") x1cuts::correcttadcoeffs(divnum, et3, et4, c, e3, e4, param, k0, f0, k1, f1, true);
      else x1cuts::correcttadcoeffsfull(divnum, et3, et4, c, e3, e4, param, k0, f0, k1, f1, 1);
      for (int i=0;i<4;++i) std::cout << -real(divnum[i]) << " " << -imag(divnum[i]) << " ";
      std::cout << "\n";
    } else if (kind == "bubtad" || kind == "bubtadfull") {
      ComplexMomentum eb2 = rcm(), eb3 = rcm(), eb4 = rcm(); RealMomentum kbol = rrm();
      Complex c[20]; for (int i=0;i<20;++i) c[i]=rc();
      ComplexMomentum e3 = rcm(), e4 = rcm(); Complex param = rc();
      RealMomentum k = rrm(); Complex f = rc();
      Complex divnum[4]; for (int i=0;i<4;++i) divnum[i]=0.;
      if (kind == "bubtad") x1cuts::correcttadcoeffs(divnum, eb2, eb3, eb4, kbol, c, e3, e4, param, k, f, true);
      else x1cuts::correcttadcoeffsfull(divnum, eb2, eb3, eb4, kbol, c, e3, e4, param, k, f, 1);
      for (int i=0;i<4;++i) std::cout << -real(divnum[i]) << " " << -imag(divnum[i]) << " ";
      std::cout << "\n";
    }
  }
  return 0;
}
