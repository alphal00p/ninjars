// Independent ordinary/higher-rank tadpole subtraction controls.
#include <coefficient_level_subs.hh>
#include <iomanip>
#include <iostream>

int main() {
  using namespace ninja;
  std::cout << std::setprecision(17);
  auto print=[](const Complex&v){std::cout<<v.real()<<' '<<v.imag()<<'\n';};
  const ComplexMomentum E2(Complex(1,.1),Complex(.2,-.1),Complex(.3,.2),Complex(-.1,.1));
  const ComplexMomentum E3(0,Complex(.5,-.1),Complex(-.4,.2),Complex(.3,.1));
  const ComplexMomentum E4(0,Complex(-.3,.2),Complex(.6,.1),Complex(.2,-.3));
  const ComplexMomentum e3(Complex(.8,.2),Complex(.2,.1),Complex(.1,-.3),Complex(-.1,.1));
  const ComplexMomentum e4(Complex(.2,-.1),Complex(-.1,.2),Complex(.3,.1),Complex(.1,-.2));
  const RealMomentum k0(2,.3,-.2,.1),k1(3,-.4,.1,-.2);
  const Complex beta(.7,.1), f0(.9,-.15), f1(1.2,.3);
  for(int source=2;source<=3;++source)for(int higher=0;higher<2;++higher)for(int lane=0;lane<2;++lane){
    const unsigned nc=higher?(source==3?15:20):10, axes=source==3?2:3, no=higher?4:2;
    Complex coefficients[20];
    for(unsigned j=0;j<nc;++j)coefficients[j]=Complex((j+1)*.03*(lane+1),.02*(int(j)%3-1));
    // Source origin is shifted along time. E3/E4 are transverse to it;
    // E2 retains the nonzero origin projection required by bubble residues.
    const RealMomentum delta(.2*lane,0,0,0);
    Complex correction[4]={};
    if(source==3){
      if(higher)x1cuts::correcttadcoeffsfull(correction,E3,E4,coefficients,e3,e4,beta,k0,f0,k1,f1,1);
      else cuts::correcttadcoeffsfull(correction,E3,E4,coefficients,e3,k0,f0,k1,f1,0);
    }else{
      if(higher)x1cuts::correcttadcoeffsfull(correction,E2,E3,E4,delta,coefficients,e3,e4,beta,k0,f0,1);
      else cuts::correcttadcoeffsfull(correction,E2,E3,E4,delta,coefficients,e3,k0,f0,0);
    }
    std::cout<<source<<' '<<higher<<' '<<nc+3*axes+1+4*(source-1)<<' '<<no<<'\n';
    for(unsigned j=0;j<nc;++j)print(coefficients[j]);
    if(source==2){print(mp(delta,E2));print(mp(e3,E2));print(mp(e4,E2));}
    for(const ComplexMomentum&E:{E3,E4}){print(mp(delta,E));print(mp(e3,E));print(mp(e4,E));}
    print(beta);
    print(Real(2)*mp(e3,k0));print(f0);print(Real(2)*mp(e4,k0)*beta);print(Real(2)*mp(e4,k0));
    if(source==3){print(Real(2)*mp(e3,k1));print(f1);print(Real(2)*mp(e4,k1)*beta);print(Real(2)*mp(e4,k1));}
    for(unsigned j=0;j<no;++j)print(-correction[j]);
  }
}
