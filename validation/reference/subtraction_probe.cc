// Independent original-Ninja coefficient subtraction controls.
#include <coefficient_level_subs.hh>
#include <iomanip>
#include <iostream>

int main() {
  using namespace ninja;
  std::cout << std::setprecision(17);
  auto print = [](const Complex &v) { std::cout << v.real() << ' ' << v.imag() << '\n'; };
  const ComplexMomentum et3(Complex(1,.2),Complex(.5,-.1),Complex(-.4,.2),Complex(.3,.1));
  const ComplexMomentum et4(Complex(.7,-.1),Complex(-.3,.2),Complex(.6,.1),Complex(.2,-.3));
  const ComplexMomentum a1(Complex(.3,.1),Complex(.2,-.1),Complex(-.1,.2),Complex(.4,.1));
  const ComplexMomentum e3(Complex(.8,.2),Complex(.2,.1),Complex(.1,-.3),Complex(-.1,.1));
  const ComplexMomentum e4(Complex(.2,-.1),Complex(-.1,.2),Complex(.3,.1),Complex(.1,-.2));
  const RealMomentum k(2,.3,-.2,.1);
  const Complex beta[] = {Complex(.7,.1),Complex(.2,-.2),Complex(-.3,.05)};
  for (int higher=0; higher<2; ++higher) {
    for (int lane=0; lane<3; ++lane) {
      const unsigned nc = higher ? 15 : 10, no = higher ? 13 : 7;
      Complex coefficients[15];
      for (unsigned j=0; j<nc; ++j) coefficients[j]=Complex((j+1)*.03*(lane+1),.02*(int(j)%3-1));
      // The numerator origin differs from the target cut's origin. Shifting
      // eb1 and compensating f preserves the same actual denominator.
      const ComplexMomentum delta(.13*lane,-.07*lane,.05*lane,.09*lane);
      const ComplexMomentum a0(Complex(.4,.2),Complex(.1,-.1),Complex(.2,.1),Complex(-.3,.2));
      const ComplexMomentum eb1 = a0+delta;
      const Complex f = Complex(.9,-.15)-Real(2)*mp(delta,k);
      Complex correction[13] = {};
      if (higher) x1cuts::correctbubcoeffs(correction,et3,et4,coefficients,eb1,a1,e3,e4,beta,k,f,1,true);
      else cuts::correctbubcoeffs(correction,et3,et4,coefficients,eb1,a1,e3,e4,beta,k,f,0,true);
      std::cout << higher << ' ' << nc+18 << ' ' << no << '\n';
      for (unsigned j=0;j<nc;++j) print(coefficients[j]);
      for (const ComplexMomentum &e : {et3,et4}) {
        print(mp(eb1,e)); print(mp(a1,e)); print(mp(e3,e)); print(mp(e4,e));
      }
      for (const Complex &b : beta) print(b);
      const Complex e4k=Real(2)*mp(e4,k);
      print(Real(2)*mp(e3,k)); print(Real(2)*mp(eb1,k)+f); print(Real(2)*mp(a1,k));
      print(e4k*beta[0]); print(e4k); print(e4k*beta[1]); print(e4k*beta[2]);
      // correctbubcoeffs subtracts from the input; NinjaRS returns the
      // correction to subtract, so invert the original zero-array result.
      for (unsigned j=0;j<no;++j) print(-correction[j]);
    }
  }
}
