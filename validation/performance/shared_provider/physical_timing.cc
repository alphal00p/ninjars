// Validation-only timing orchestration; correctness is the default mode.
#include "physical_workload.hh"
#include <ninja/avholo.hh>
#include <chrono>
#include <numeric>
#include <sstream>
#ifdef SP_ALLOCATION_DIAGNOSTICS
extern "C" int32_t sp_allocation_begin();
extern "C" int32_t sp_allocation_end(uint64_t *,size_t);
#endif
using Clock=std::chrono::steady_clock;
using Counts=std::array<uint64_t,20>;
static Counts counts(Physical&p,Runs&r,SharedMasters&live){
  Counts out{};checked(sp_provider_stats(p.provider,out.data(),8));checked(sp_physical_stats(p.handle,out.data()+8,3));
  checked(sp_session_stats(r.session.handle,out.data()+11,4));checked(sp_master_execution_stats(live.handle(),out.data()+15,3));
  checked(sp_session_production_stats(r.session.handle,out.data()+18,2));return out;
}
static Counts difference(const Counts&a,const Counts&b){Counts out{};for(size_t i=0;i<out.size();++i){if(b[i]<a[i])throw std::runtime_error("diagnostic counter decreased");out[i]=b[i]-a[i];}return out;}
static void accumulate(Counts&a,const Counts&b){for(size_t i=0;i<a.size();++i)a[i]+=b[i];}
static void emit_counts(std::ostream&stream,const Counts&values){stream<<'[';for(size_t i=0;i<values.size();++i){if(i)stream<<',';stream<<values[i];}stream<<']';}
static std::vector<Point> read_pool(){
  size_t count;std::cin>>count;if(count!=3)throw std::runtime_error("three-point pool required");std::vector<Point>pool(count);
  for(auto&p:pool){for(auto&x:p.momenta)std::cin>>x;for(auto&x:p.helicities){int value;std::cin>>value;x=static_cast<int8_t>(value);}for(auto&x:p.model)std::cin>>x;std::cin>>p.scale;size_t samples;std::cin>>samples;if(samples!=70)throw std::runtime_error("independent numerator grid absent");
    p.raw_inputs.resize(samples);p.raw_expected.resize(samples);for(size_t i=0;i<samples;++i){for(unsigned j=0;j<4;++j)std::cin>>p.raw_inputs[i][j].re>>p.raw_inputs[i][j].im;std::cin>>p.raw_expected[i].re>>p.raw_expected[i].im;}for(auto&x:p.integrated)std::cin>>x.re>>x.im;
  }if(!std::cin)throw std::runtime_error("truncated physical pool");return pool;
}
struct Workload{
  std::vector<Point>pool;Physical physical;SharedMasters replay,live;ninja::AvHOneLoop oneloop;Runs runs;
  explicit Workload(const char*path):pool(read_pool()),physical(path,1),runs(physical){
    const std::vector<size_t>all{0,1,2};
    const auto original=runs.run_original(pool,all,replay);const auto rust=runs.run_rust(pool,all,replay);
    for(size_t row=0;row<3;++row)for(size_t j=0;j<3;++j){near(complex(original[row*5+j]),complex(rust[row*5+j]),"capture shared Laurent");near(complex(rust[row*5+j]),complex(pool[row].integrated[j]),"capture independent MadLoop");}
    checked(sp_master_freeze(replay.handle()));checked(sp_master_set_live(live.handle()));
  }
  const std::vector<SpPair>&execute(unsigned scope,bool rust,const std::vector<size_t>&indices){
    if(scope!=1)physical.prepare(pool,indices);
    if(scope==0)return physical.packed;
    if(rust)return runs.reduce_rust(scope==3?nullptr:(scope==1?&replay:&live));
    return runs.reduce_original(scope==3?static_cast<ninja::IntegralLibrary&>(oneloop):static_cast<ninja::IntegralLibrary&>(scope==1?replay:live));
  }
  void prepare_outside(unsigned scope,const std::vector<size_t>&indices){if(scope==1)physical.prepare(pool,indices);}
};
int main(int argc,char**argv){try{
  if(argc!=7)throw std::runtime_error("usage: physical-timing DOT check|timing|allocations rounds trials widths scopes");
  const std::string mode=argv[2];const bool timing=mode=="timing",allocations=mode=="allocations";
  if(!timing&&!allocations&&mode!="check")throw std::runtime_error("unknown execution mode");
#ifdef SP_ALLOCATION_DIAGNOSTICS
  if(timing)throw std::runtime_error("allocation-instrumented builds cannot produce primary timings");
#else
  if(allocations)throw std::runtime_error("rebuild with allocation-diagnostics feature");
#endif
  const size_t rounds=std::stoul(argv[3]),trials=std::stoul(argv[4]);if(!rounds||trials<4||trials%2)throw std::runtime_error("positive rounds and >=4 balanced trials required");
  std::vector<size_t>widths;std::stringstream sizes(argv[5]);std::string entry;while(std::getline(sizes,entry,',')){auto n=std::stoul(entry);if(!n||n>1024)throw std::runtime_error("width must be1..1024");widths.push_back(n);}if(widths.empty())throw std::runtime_error("no batch widths");
  std::vector<unsigned>scopes;std::stringstream scope_input(argv[6]);while(std::getline(scope_input,entry,',')){auto n=std::stoul(entry);if(n>3)throw std::runtime_error("scope must be0..3");scopes.push_back(n);}if(scopes.empty())throw std::runtime_error("no scopes");
  const std::array<const char*,4> names{{"preparation","shared-reduction","shared-live-complete","production-complete"}};
  ninja::setVerbosity(ninja::Verbose::NONE);ninja::setTest(ninja::Test::NONE);Workload workload(argv[1]);
  double clock_pair_ns=0.;if(timing){std::array<double,101>overhead{};for(auto&sample:overhead){auto start=Clock::now();auto end=Clock::now();sample=std::chrono::duration<double,std::nano>(end-start).count();}std::sort(overhead.begin(),overhead.end());clock_pair_ns=overhead[overhead.size()/2];}
  std::ostringstream report;
  report<<std::setprecision(17)<<"{\"success\":true,\"mode\":\""<<mode<<"\",\"clock_pair_median_ns\":"<<clock_pair_ns<<",\"performance_acceptance\":false,\"original_production_backend_counters_available\":false,\"counter_columns\":[\"numerator_direct_calls\",\"numerator_mu_calls\",\"numerator_t3_calls\",\"numerator_t2_calls\",\"numerator_direct_rows\",\"numerator_mu_rows\",\"numerator_t3_rows\",\"numerator_t2_rows\",\"preparation_calls\",\"preparation_rows\",\"wavefunction_calls\",\"rust_numerator_wire_calls\",\"rust_numerator_wire_rows\",\"rust_master_wire_calls\",\"rust_master_wire_rows\",\"shared_live_backend_calls\",\"shared_live_backend_rows\",\"live_flag_delta\",\"direct_rust_master_calls\",\"direct_rust_master_rows\"],\"records\":[";
  bool first=true;double checksum=0.;
  for(unsigned scope:scopes)for(bool changing:{false,true})for(size_t width:widths){
    std::array<std::vector<size_t>,3>indices;for(size_t phase=0;phase<3;++phase){indices[phase].resize(width);for(size_t row=0;row<width;++row)indices[phase][row]=changing?(row+phase)%3:0;}
    double error=0.;size_t checked_components=0;
    // Warm every point and operation, checking both independent and paired
    // values before either optional timing or allocation diagnostics.
    for(size_t phase=0;phase<3;++phase){workload.prepare_outside(scope,indices[phase]);const auto original=workload.execute(scope,false,indices[phase]);workload.prepare_outside(scope,indices[phase]);const auto rust=workload.execute(scope,true,indices[phase]);
      if(scope==0){if(original.size()!=rust.size())throw std::runtime_error("preparation output mismatch");for(size_t i=0;i<rust.size();++i){error=std::max(error,near(complex(rust[i]),complex(original[i]),"baseline preparation"));++checked_components;}}
      else for(size_t row=0;row<width;++row){for(unsigned j=0;j<3;++j){error=std::max(error,near(complex(rust[row*5+j]),complex(original[row*5+j]),"scope paired Laurent"));error=std::max(error,near(complex(rust[row*5+j]),complex(workload.pool[indices[phase][row]].integrated[j]),"scope independent MadLoop"));++checked_components;}near(complex(rust[row*5]),complex(rust[row*5+3])+complex(rust[row*5+4]),"scope Rust split");near(complex(original[row*5]),complex(original[row*5+3])+complex(original[row*5+4]),"scope original split");}
    }
    if(!first){report<<',';}first=false;
    report<<"{\"scope\":\""<<names[scope]<<"\",\"workload\":\""<<(changing?"changing":"repeated")<<"\",\"width\":"<<width<<",\"rounds\":"<<rounds<<",\"correctness_components\":"<<checked_components<<",\"max_scaled_error\":"<<error<<",\"samples\":[";
    Counts original_counts{},rust_counts{};
    if(timing){for(size_t trial=0;trial<trials;++trial){std::array<double,2>times{};for(unsigned turn=0;turn<2;++turn){const bool rust=(turn+(trial%2))%2;for(size_t round=0;round<rounds;++round){const auto &input=indices[(round+trial)%3];workload.prepare_outside(scope,input);const auto before=counts(workload.physical,workload.runs,workload.live);auto start=Clock::now();const auto &output=workload.execute(scope,rust,input);auto end=Clock::now();times[rust]+=std::chrono::duration<double,std::nano>(end-start).count();const auto after=counts(workload.physical,workload.runs,workload.live);accumulate(rust?rust_counts:original_counts,difference(before,after));checksum+=output.back().re;}}
      if(trial){report<<',';}report<<"{\"original_ns\":"<<times[0]/rounds<<",\"rust_ns\":"<<times[1]/rounds<<",\"order\":\""<<(trial%2?"rust-first":"original-first")<<"\"}";
    }}
    report<<"],\"original_counter_deltas\":";emit_counts(report,original_counts);report<<",\"rust_counter_deltas\":";emit_counts(report,rust_counts);
#ifdef SP_ALLOCATION_DIAGNOSTICS
    if(allocations){report<<",\"rust_allocator_only\":true,\"allocation_columns\":[\"allocation_calls\",\"requested_bytes\",\"reallocation_calls\",\"reallocated_bytes\",\"deallocation_calls\"],\"allocation_scopes\":[\"total\",\"numerator\",\"masters\",\"physical_preparation\"],\"allocation_counts\":[";
      for(unsigned rust=0;rust<2;++rust){workload.prepare_outside(scope,indices[1]);checked(sp_allocation_begin());const auto&output=workload.execute(scope,rust,indices[1]);std::array<uint64_t,20>values{};checked(sp_allocation_end(values.data(),values.size()));
        for(size_t column=0;column<5;++column){uint64_t scoped=values[5+column]+values[10+column]+values[15+column];if(scoped>values[column])throw std::runtime_error("allocation scopes overlap totals");}
        if(rust && values[0]+values[2]==0)throw std::runtime_error("Rust allocation diagnostic failed to observe expected work");
        checksum+=output.back().re;if(rust){report<<',';}emit_counts(report,values);}report<<']';}
#endif
    report<<'}';
  }
  std::array<uint64_t,7>replay{};checked(sp_master_stats(workload.replay.handle(),replay.data(),7));if(replay[6]!=0)throw std::runtime_error("frozen union miss");
  report<<"],\"checksum\":"<<checksum<<",\"union_entries\":"<<replay[0]<<",\"union_replay_rows\":"<<replay[5]<<",\"union_misses\":"<<replay[6]<<"}\n";
  std::cout<<report.str()<<std::flush;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
