// Synthetic R=n and R=n+1 controls through the same frozen master ABI.
#include "masters.hh"
#include "control.hh"
#include "diagnostic.hh"
#include <chrono>
#include <iomanip>

struct Session {
  SpSession *handle{};
  Session(size_t rank, size_t rows) { checked(sp_session_create(2,rank,rows,&handle)); }
  ~Session() { sp_session_destroy(handle); }
  Session(const Session&)=delete;Session&operator=(const Session&)=delete;
};
int main(int argc,char **argv) {
 try {
  if(argc!=6)throw std::runtime_error("usage: bubble rank width rounds control ack");
  const size_t rank=std::stoul(argv[1]),width=std::stoul(argv[2]),rounds=std::stoul(argv[3]);
  if((rank!=2&&rank!=3)||!width||width>1024||!rounds)throw std::runtime_error("invalid dimensions");
  ninja::setVerbosity(ninja::Verbose::NONE);ninja::setTest(ninja::Test::NONE);
  SharedNumerator numerator(rank==2?3:4);SharedMasters masters;Session session(rank,width);
  const std::array<ninja::RealMomentum,2> shifts{{ninja::RealMomentum(1,0,0,0),ninja::RealMomentum(1,1,0,0)}};
  const std::array<ninja::Complex,2> masses{{{2,0},{3,0}}};
  std::array<double,4> matrix{{0,-1,-1,0}};
  std::vector<SpPair> v,m,s,sm,parameters,output(width*5);
  for(size_t row=0;row<width;++row){
   for(auto value:std::array<SpPair,8>{{{1,0},{0,0},{0,0},{0,0},{1,0},{1,0},{0,0},{0,0}}})v.push_back(value);
   m.push_back({2,0});m.push_back({3,0});s.push_back({7,0});
   for(double value:matrix)sm.push_back({value,0});
   parameters.push_back({2,1});parameters.push_back({3,-1});
  }
  numerator.bind(parameters,width);
  ninja::Amplitude<ninja::ComplexMasses> original(2,rank,shifts.data(),masses.data());
  original.setIntegralLibrary(masters).setRenormalizationScale(7.).setSMatrix(matrix.data());
  if(original.evaluate(numerator))throw std::runtime_error("original bubble failed");
  std::array<ninja::Complex,3> expected{{original[0],original[1],original[2]}};
  auto run=[&](){checked(sp_session_run(session.handle,numerator.handle(),masters.handle(),width,v.data(),v.size(),m.data(),m.size(),s.data(),s.size(),sm.data(),sm.size(),output.data(),output.size()));};
  auto verify=[&](){run();for(size_t row=0;row<width;++row){for(size_t j=0;j<3;++j)close(complex(output[row*5+j]),expected[j]);close(complex(output[row*5]),complex(output[row*5+3])+complex(output[row*5+4]));}};
  verify();checked(sp_master_freeze(masters.handle()));for(unsigned i=0;i<8;++i)verify();
  std::array<uint64_t,4> before{},after{};checked(sp_session_stats(session.handle,before.data(),4));
  Control control(argv[4],argv[5]);Diagnostic diagnostic;double checksum=0.;
  control.send("enable");diagnostic.start();const auto start=std::chrono::steady_clock::now();
  for(size_t i=0;i<rounds;++i){run();checksum+=output[0].re;}
  const auto end=std::chrono::steady_clock::now();diagnostic.stop();control.send("disable");
  checked(sp_session_stats(session.handle,after.data(),4));verify();
  std::array<uint64_t,7> stats{};checked(sp_master_stats(masters.handle(),stats.data(),7));
  if(stats[6]||!std::isfinite(checksum))throw std::runtime_error("miss/nonfinite checksum");
  std::cout<<std::setprecision(17)<<"{\"success\":true,\"performance_acceptance\":false,\"scope\":\"shared-bubble-repeated-fixed-point\",\"rank\":"<<rank<<",\"width\":"<<width<<",\"rounds\":"<<rounds<<",\"wall_ns_diagnostic\":"<<std::chrono::duration<double,std::nano>(end-start).count()<<",\"checksum\":"<<checksum<<",\"union_misses\":"<<stats[6]<<",\"session_counter_deltas\":[";
  for(size_t i=0;i<4;++i){if(i)std::cout<<',';std::cout<<after[i]-before[i];}
  std::cout<<"],\"laurent\":[";for(size_t i=0;i<3;++i){if(i)std::cout<<',';std::cout<<'['<<expected[i].real()<<','<<expected[i].imag()<<']';}std::cout<<"]}\n";
 }catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
}
