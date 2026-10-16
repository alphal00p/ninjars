// Correctness only: no shared-master implementation and no timings.
#include "numerator.hh"

int main(){try{
  using ninja::Complex;using ninja::ComplexMomentum;using ninja::RealMomentum;
  const std::array<ComplexMomentum,4> vectors{{
    ComplexMomentum(Complex(1,1),Complex(2,0),Complex(3,-1),Complex(4,0)),
    ComplexMomentum(Complex(2,0),Complex(-1,1),Complex(3,0),Complex(-2,0)),
    ComplexMomentum(Complex(3,0),Complex(1,0),Complex(-2,1),Complex(4,0)),
    ComplexMomentum(Complex(1,0),Complex(2,-1),Complex(-3,0),Complex(-1,0))}};
  const std::array<Complex,3> beta{{Complex(2,-1),Complex(3,1),Complex(5,0)}};
  const ninja::PartitionInt partition[]={0,1,2,3};
  const unsigned t3counts[]={1,2,4,6,9},t2counts[]={1,3,7,13};
  unsigned responses=0;
  for(unsigned fixture=0;fixture<2;++fixture){
    SharedNumerator provider(fixture);provider.bind({{2,1},{3,-1},{-1,2},{1,1}},2);
    for(unsigned lane:{1u,0u,1u}){
      provider.lane(lane);std::array<SpPair,13> expected{};std::array<Complex,13> actual{};
      checked(sp_control(fixture,0,lane,expected.data(),1));
      close(provider.evaluate(vectors[0],Complex(7,2),3,partition),complex(expected[0]));++responses;
      const unsigned mucount=fixture==0?1:2;
      checked(sp_control(fixture,1,lane,expected.data(),mucount));provider.muExpansion(vectors.data(),partition,actual.data());
      for(unsigned j=0;j<mucount;++j){close(actual[j],complex(expected[j]));++responses;}
      checked(sp_control(fixture,2,lane,expected.data(),9));
      for(unsigned d=0;d<=4;++d){provider.t3Expansion(vectors[0],vectors[1],vectors[2],beta[0],d,3,partition,actual.data());for(unsigned j=0;j<t3counts[d];++j){close(actual[j],complex(expected[j]));++responses;}}
      checked(sp_control(fixture,3,lane,expected.data(),13));
      for(unsigned d=0;d<=3;++d){provider.t2Expansion(vectors[0],vectors[1],vectors[2],vectors[3],beta.data(),d,2,partition,actual.data());for(unsigned j=0;j<t2counts[d];++j){close(actual[j],complex(expected[j]));++responses;}}
    }
  }
  ninja::setVerbosity(ninja::Verbose::NONE);ninja::setTest(ninja::Test::NONE);
  const std::array<RealMomentum,3> shifts{{RealMomentum(0,0,0,0),RealMomentum(2,1,0,0),RealMomentum(3,0,1,0)}};
  const std::array<Complex,3> masses{{Complex(1,0),Complex(1,0),Complex(1,0)}};
  SharedNumerator provider(2);
  ninja::Amplitude<ninja::ComplexMasses> amplitude(3,2,shifts.data(),masses.data());
  amplitude.setRenormalizationScale(1.);
  for(SpPair a:std::vector<SpPair>{{2,1},{3,-2}}){provider.bind({a},1);amplitude.reset();const int status=amplitude.evaluate(provider);if(status!=0)throw std::runtime_error("original Ninja reduction failed");close(amplitude[0],complex(a)*.5);close(amplitude[1],Complex(0,0));close(amplitude[2],Complex(0,0));close(amplitude.getRationalPart(),complex(a)*.5);}
  if(sp_probe_panic()!=2)throw std::runtime_error("panic crossed ABI status boundary");
  std::cout<<"{\"success\":true,\"callback_components_checked\":"<<responses<<",\"analytic_triangle_rebindings\":2,\"shared_masters\":false,\"performance_acceptance\":false}\n";
  return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
