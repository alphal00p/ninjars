// Validation only: expose existing methods without changing their implementation.
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <ninja/types.hh>
#include <ninja/momentum.hh>
#include <ninja/s_mat.hh>
#include <ninja/num_defs.hh>
#define private public
#include <ninja/ninja.hh>
#undef private
#include <Xcuts.hh>
#include "box-c5-only.hh"

int main() {
  using namespace ninja;
  std::cout << std::setprecision(17);
  const RealMomentum shifts[4]={RealMomentum(0,0,0,0),RealMomentum(2,1,0,0),RealMomentum(3,0,1,0),RealMomentum(4,0,0,1)};
  const Complex masses[4]={1.,1.,1.,1.};
  SMatrix matrix(4); matrix.fillFromKinematics(shifts);
  Amplitude<ComplexMasses> amplitude(4,5,shifts,masses);
  amplitude.setSMatrix(matrix);
  CutsVector<x1cuts::Pentagon> pentagons(0,4);
  Complex parameters[4]={};
  ninjars_reference::GeneratedNumerator numerator(parameters);
  x1cuts::Box first;
  for(int i=0;i<4;++i) first.p[i]=i;
  amplitude.evaluateFullBox(numerator,pentagons,first);
  // Use the original cut's own Vort, so the exact generated polynomial is
  // precisely c5*u*mu^4 with c5=1 in its convention and Vfirst=0.
  for(int i=0;i<4;++i) parameters[i]=first.Vort[i];
  ninjars_reference::GeneratedNumerator c5_only(parameters);
  for(int use_mu=0;use_mu<2;++use_mu) {
    amplitude.useMuExpansion(use_mu);
    x1cuts::Box box; for(int i=0;i<4;++i) box.p[i]=i;
    amplitude.evaluateFullBox(c5_only,pentagons,box);
    std::cout << "{\"use_mu_expansion\":" << use_mu << ",\"scale_choice\":15,\"status\":" << amplitude.return_val << ",\"coefficients\":[";
    for(int i=0;i<6;++i) {if(i)std::cout<<',';std::cout<<'['<<box.c[i].real()<<','<<box.c[i].imag()<<']';}
    // Direct residue evaluation at an arbitrary point independently checks
    // whether the reconstructed coefficient array reproduces the same input.
    ComplexMomentum q(Complex(.3,.1),Complex(.2,-.1),Complex(-.4,.2),Complex(.1,.3));
    Complex mu(0.7,0.2);
    Complex expected=c5_only.evaluate(q,mu,4,box.p), actual=box.poly(q,mu);
    std::cout << "],\"numerator\":[" << expected.real()<<','<<expected.imag()<<"],\"reconstructed\":["<<actual.real()<<','<<actual.imag()<<"]}\n";
  }
}
