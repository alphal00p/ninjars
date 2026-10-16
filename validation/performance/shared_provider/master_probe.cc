#include "masters.hh"
#include <iomanip>

struct Session {
  SpSession *handle{};
  explicit Session(unsigned rank){checked(sp_session_create(2,rank,1,&handle));}
  ~Session(){sp_session_destroy(handle);}
  Session(const Session&)=delete;Session&operator=(const Session&)=delete;
};

// Capture ordered requests through the Rust ABI, freeze first, and then enter
// Ninja's virtual CM/RM methods. A permuted invariant or mass is an exact-key
// miss even when a numerical integral happens to be permutation-symmetric.
static unsigned higher_point_ordering(){
  SharedMasters masters;masters.init(7.);
  const std::array<SpPair,6> invariants{{{-1,0},{-2,0},{-3,0},{-4,0},{-5,0},{-6,0}}};
  const SpPair scale{7,0};
  std::array<std::array<SpPair,4>,2> masses{{
    {{{2.25,0},{3.5,0},{5.75,0},{7.125,0}}},
    {{{2.25,-.125},{3.5,-.25},{5.75,-.5},{7.125,-1.}}}
  }};
  std::array<std::array<SpPair,3>,4> expected{};
  for(unsigned domain=0;domain<2;++domain)for(unsigned kind=5;kind<=6;++kind){
    auto &out=expected[2*domain+kind-5];
    try{checked(sp_master_call(masters.handle(),kind,1,1,invariants.data(),kind==5?3:6,
      masses[domain].data(),kind==5?3:4,&scale,1,out.data(),out.size()));}
    catch(...){std::cerr<<"ordered adapter kind="<<kind<<" complex_masses="<<domain<<'\n';throw;}
  }
  checked(sp_master_freeze(masters.handle()));
  unsigned components=0;
  for(unsigned domain=0;domain<2;++domain)for(unsigned kind=5;kind<=6;++kind){
    std::array<ninja::Complex,3> out{};const auto &m=masses[domain];
    if(domain==0){
      if(kind==5)masters.getTriangleIntegralRM(out.data(),-1,-2,-3,m[0].re,m[1].re,m[2].re);
      else masters.getBoxIntegralRM(out.data(),-1,-2,-3,-4,-5,-6,m[0].re,m[1].re,m[2].re,m[3].re);
    }else{
      if(kind==5)masters.getTriangleIntegralCM(out.data(),-1,-2,-3,complex(m[0]),complex(m[1]),complex(m[2]));
      else masters.getBoxIntegralCM(out.data(),-1,-2,-3,-4,-5,-6,complex(m[0]),complex(m[1]),complex(m[2]),complex(m[3]));
    }
    for(unsigned j=0;j<3;++j){close(out[j],complex(expected[2*domain+kind-5][j]));++components;}
  }
  std::array<uint64_t,7> stats{};checked(sp_master_stats(masters.handle(),stats.data(),stats.size()));
  if(stats!=std::array<uint64_t,7>{{4,4,0,0,4,4,0}})throw std::runtime_error("C0/D0 ordered replay did not use captured keys");
  return components;
}

int main(){try{
  using ninja::Complex;using ninja::RealMomentum;
  SharedMasters masters;
  Session ordinary(2),higher(3);
  ninja::setVerbosity(ninja::Verbose::NONE);ninja::setTest(ninja::Test::NONE);
  const std::array<Complex,2> masses{{Complex(2,0),Complex(3,0)}};
  std::array<double,4> matrix{{0,-1,-1,0}};
  const std::array<SpPair,2> mass_pairs{{{2,0},{3,0}}};
  const std::array<SpPair,4> matrix_pairs{{{0,0},{-1,0},{-1,0},{0,0}}};
  std::vector<std::array<Complex,5>> captured;
  std::array<uint64_t,7> before{},after{};
  struct Point{unsigned fixture;double scale,origin;};
  const std::array<Point,8> points{{{3,1,1},{3,7,1},{4,1,1},{4,7,1},{5,1,0},{5,1,1},{6,1,0},{6,1,1}}};
  std::vector<std::array<Complex,5>> captured_rust;
  unsigned comparisons=0,split_comparisons=0,retained_comparisons=0;
  // Capture both actual reducers' request union, freeze it, then replay the
  // same complete amplitudes with no master expression execution.
  for(unsigned pass=0;pass<2;++pass){
    if(pass){checked(sp_master_stats(masters.handle(),before.data(),before.size()));checked(sp_master_freeze(masters.handle()));}
    unsigned point=0;
    for(auto input:points){
      const unsigned fixture=input.fixture;const double scale=input.scale,origin=input.origin;
      const std::array<RealMomentum,2> shifts{{RealMomentum(origin,0,0,0),RealMomentum(origin,1,0,0)}};
      const std::array<SpPair,8> shift_pairs{{{origin,0},{0,0},{0,0},{0,0},{origin,0},{1,0},{0,0},{0,0}}};
      SharedNumerator numerator(fixture);numerator.bind(fixture>=5?std::vector<SpPair>{}:std::vector<SpPair>{{2,1},{3,-1}},1);
      ninja::Amplitude<ninja::ComplexMasses> amplitude(2,fixture==4?3:2,shifts.data(),masses.data());
      amplitude.setIntegralLibrary(masters).setRenormalizationScale(scale).setSMatrix(matrix.data());
      const int status=amplitude.evaluate(numerator);if(status)throw std::runtime_error("original shared-master bubble failed");
      std::array<Complex,5> original{{amplitude[0],amplitude[1],amplitude[2],amplitude.getCutConstructiblePart(),amplitude.getRationalPart()}};
      std::array<SpPair,5> rust{};
      checked(sp_rust_bubble(numerator.handle(),masters.handle(),shift_pairs.data(),shift_pairs.size(),mass_pairs.data(),mass_pairs.size(),{scale,0},matrix_pairs.data(),matrix_pairs.size(),rust.data(),rust.size()));
      const SpPair scale_pair{scale,0};
      std::array<SpPair,5> retained{};
      auto session=fixture==4?higher.handle:ordinary.handle;
      checked(sp_session_run(session,numerator.handle(),masters.handle(),1,shift_pairs.data(),shift_pairs.size(),mass_pairs.data(),mass_pairs.size(),&scale_pair,1,matrix_pairs.data(),matrix_pairs.size(),retained.data(),retained.size()));
      for(unsigned j=0;j<5;++j){close(complex(retained[j]),complex(rust[j]));++retained_comparisons;}
      if(point==0){
        const auto previous=retained;
        if(sp_session_run(session,numerator.handle(),masters.handle(),1,shift_pairs.data(),shift_pairs.size(),mass_pairs.data(),mass_pairs.size(),&scale_pair,1,matrix_pairs.data(),matrix_pairs.size()-1,retained.data(),retained.size())==0)throw std::runtime_error("malformed retained request accepted");
        for(unsigned j=0;j<5;++j)close(complex(retained[j]),complex(previous[j]));
        checked(sp_session_run(session,numerator.handle(),masters.handle(),0,nullptr,0,nullptr,0,nullptr,0,nullptr,0,nullptr,0));
      }
      for(unsigned j=0;j<3;++j){close(original[j],complex(rust[j]));++comparisons;}
      // The explicit mu² polynomial has a basis-independent rational split.
      // For q0² record both splits; different residue bases move finite terms.
      if(fixture==6){
        for(unsigned j=3;j<5;++j){close(original[j],complex(rust[j]));++split_comparisons;}
        close(original[0],Complex(8./3.,0));close(original[3],Complex(0,0));close(original[4],Complex(8./3.,0));
      }
      close(original[0],original[3]+original[4]);close(complex(rust[0]),complex(rust[3])+complex(rust[4]));
      if(pass){for(unsigned j=0;j<5;++j){close(original[j],captured[point][j]);close(complex(rust[j]),captured_rust[point][j]);}}
      else{captured.push_back(original);std::array<Complex,5> values{};for(unsigned j=0;j<5;++j)values[j]=complex(rust[j]);captured_rust.push_back(values);}
      ++point;
    }
  }
  checked(sp_master_stats(masters.handle(),after.data(),after.size()));
  if(after[0]!=before[0]||after[4]!=before[4]||after[5]==0||after[1]==0||after[6]!=0)throw std::runtime_error("replay changed the union or executed a backend");
  const SpPair unknown_mass{99,0},scale{1,0};std::array<SpPair,3> unchanged{{{77,0},{77,0},{77,0}}};
  if(sp_master_call(masters.handle(),0,2,1,nullptr,0,&unknown_mass,1,&scale,1,unchanged.data(),3)==0)throw std::runtime_error("unknown replay key was accepted");
  for(auto value:unchanged)if(value.re!=77||value.im!=0)throw std::runtime_error("replay miss modified output");
  checked(sp_master_stats(masters.handle(),after.data(),after.size()));
  const unsigned high_components=higher_point_ordering();
  std::array<uint64_t,4> ordinary_stats{},higher_stats{};checked(sp_session_stats(ordinary.handle,ordinary_stats.data(),4));checked(sp_session_stats(higher.handle,higher_stats.data(),4));
  for(unsigned j=0;j<4;++j)if(ordinary_stats[j]==0||higher_stats[j]==0)throw std::runtime_error("retained session did not exercise both wire providers");
  std::cout<<std::setprecision(17)<<"{\"success\":true,\"full_bubble_laurent_components_checked\":"<<comparisons<<",\"retained_session_components_checked\":"<<retained_comparisons<<",\"shared_out_of_line_entry_points\":true,\"explicit_mu2_split_components_checked\":"<<split_comparisons<<",\"c0_d0_ordered_adapter_components_checked\":"<<high_components<<",\"union_entries\":"<<after[0]<<",\"both_sources\":"<<after[1]<<",\"rust_only\":"<<after[2]<<",\"ninja_only\":"<<after[3]<<",\"captured_backend_rows\":"<<after[4]<<",\"replayed_rows\":"<<after[5]<<",\"intentional_misses\":"<<after[6]<<",\"higher_rank_b111_override\":true,\"rational_separation_parity_complete\":false,\"split_controls\":[";
  for(unsigned i=0;i<points.size();++i){
    if(i){std::cout<<',';}
    std::cout<<"{\"fixture\":"<<points[i].fixture<<",\"scale\":"<<points[i].scale<<",\"time_origin\":"<<points[i].origin;
    for(unsigned source=0;source<2;++source){std::cout<<(source?",\"rust\":[":",\"original\":[");const auto &values=source?captured_rust[i]:captured[i];for(unsigned j=0;j<5;++j){if(j)std::cout<<',';std::cout<<'['<<values[j].real()<<','<<values[j].imag()<<']';}std::cout<<']';}
    std::cout<<'}';
  }
  std::cout<<"],\"performance_acceptance\":false}\n";
  return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
