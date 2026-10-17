// Validation-only timing orchestration; correctness is the default mode.
#include "physical_workload.hh"
#include <ninja/avholo.hh>
#include <chrono>
#include <numeric>
#include <memory>
#include <sstream>
#ifdef SP_EXECUTION_DIAGNOSTICS
#include "execution.hh"
using ProductionOriginal=TimedOneLoop;
constexpr bool original_backend_counters=true;
#else
using ProductionOriginal=ninja::AvHOneLoop;
constexpr bool original_backend_counters=false;
#endif
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
template<size_t N>static void emit_counts(std::ostream&stream,const std::array<uint64_t,N>&values){stream<<'[';for(size_t i=0;i<values.size();++i){if(i)stream<<',';stream<<values[i];}stream<<']';}
static std::vector<Point> read_pool(){
  size_t count;std::cin>>count;if(count!=3)throw std::runtime_error("three-point pool required");std::vector<Point>pool(count);
  for(auto&p:pool){for(auto&x:p.momenta)std::cin>>x;for(auto&x:p.helicities){int value;std::cin>>value;x=static_cast<int8_t>(value);}for(auto&x:p.model)std::cin>>x;std::cin>>p.scale;size_t samples;std::cin>>samples;if(samples!=70)throw std::runtime_error("independent numerator grid absent");
    p.raw_inputs.resize(samples);p.raw_expected.resize(samples);for(size_t i=0;i<samples;++i){for(unsigned j=0;j<4;++j)std::cin>>p.raw_inputs[i][j].re>>p.raw_inputs[i][j].im;std::cin>>p.raw_expected[i].re>>p.raw_expected[i].im;}for(auto&x:p.integrated)std::cin>>x.re>>x.im;
  }if(!std::cin)throw std::runtime_error("truncated physical pool");return pool;
}
struct Workload{
  std::vector<Point>pool;Physical physical;SharedMasters replay,live;ProductionOriginal oneloop;Runs runs;std::array<uint64_t,3> captured_backend{};
  explicit Workload(const char*path,unsigned selection):pool(read_pool()),physical(path,1),runs(physical,selection){
    const std::vector<size_t>all{0,1,2};
    const auto original=runs.run_original(pool,all,replay);const auto rust=runs.run_rust(pool,all,replay);
    for(size_t row=0;row<3;++row)for(size_t j=0;j<3;++j){near(complex(original[row*5+j]),complex(rust[row*5+j]),"capture shared Laurent");near(complex(rust[row*5+j]),complex(pool[row].integrated[j]),"capture independent MadLoop");}
    checked(sp_master_freeze(replay.handle()));checked(sp_master_set_live(live.handle()));
    checked(sp_master_execution_stats(replay.handle(),captured_backend.data(),captured_backend.size()));
  }
  const std::vector<SpPair>&execute(unsigned scope,bool rust,const std::vector<size_t>&indices){
    if(scope!=1)physical.prepare(pool,indices);
    if(scope==0)return physical.packed;
    if(rust)return runs.reduce_rust(scope==3?nullptr:(scope==1?&replay:&live));
    return runs.reduce_original(scope==3?static_cast<ninja::IntegralLibrary&>(oneloop):static_cast<ninja::IntegralLibrary&>(scope==1?replay:live));
  }
};
int main(int argc,char**argv){try{
  if(argc!=7&&argc!=8)throw std::runtime_error("usage: physical-timing DOT check|timing|allocations|execution rounds trials widths scopes [symbolica|native-denominators|native|interpreted-rank|native-rank]");
  const unsigned selection=argc==8?kernel_selection(argv[7]):static_cast<unsigned>(SP_KERNEL_SYMBOLICA);
  const std::string mode=argv[2];const bool timing=mode=="timing",allocations=mode=="allocations",execution=mode=="execution";
  if(!timing&&!allocations&&!execution&&mode!="check")throw std::runtime_error("unknown execution mode");
#ifdef SP_ALLOCATION_DIAGNOSTICS
  if(timing)throw std::runtime_error("allocation-instrumented builds cannot produce primary timings");
#else
  if(allocations)throw std::runtime_error("rebuild with allocation-diagnostics feature");
#endif
#ifdef SP_EXECUTION_DIAGNOSTICS
  if(timing)throw std::runtime_error("execution-instrumented builds cannot produce primary timings");
#else
  if(execution)throw std::runtime_error("rebuild with execution-diagnostics feature");
#endif
  const size_t rounds=std::stoul(argv[3]),trials=std::stoul(argv[4]);if(!rounds||trials<4||trials%2)throw std::runtime_error("positive rounds and >=4 balanced trials required");
  if(timing&&rounds%3)throw std::runtime_error("timing rounds must include complete cycles of all three phase-space points");
  std::vector<size_t>widths;std::stringstream sizes(argv[5]);std::string entry;while(std::getline(sizes,entry,',')){auto n=std::stoul(entry);if(!n||n>1024)throw std::runtime_error("width must be1..1024");widths.push_back(n);}if(widths.empty())throw std::runtime_error("no batch widths");
  std::vector<unsigned>scopes;std::stringstream scope_input(argv[6]);while(std::getline(scope_input,entry,',')){auto n=std::stoul(entry);if(n>3)throw std::runtime_error("scope must be0..3");scopes.push_back(n);}if(scopes.empty())throw std::runtime_error("no scopes");
  const std::array<const char*,4> names{{"preparation","shared-reduction","shared-live-complete","production-complete"}};
  ninja::setVerbosity(ninja::Verbose::NONE);ninja::setTest(ninja::Test::NONE);Workload workload(argv[1],selection);
  double clock_pair_ns=0.;if(timing){std::array<double,101>overhead{};for(auto&sample:overhead){auto start=Clock::now();auto end=Clock::now();sample=std::chrono::duration<double,std::nano>(end-start).count();}std::sort(overhead.begin(),overhead.end());clock_pair_ns=overhead[overhead.size()/2];}
  std::ostringstream report;
  report<<std::setprecision(17)<<"{\"success\":true,\"reduction_kernels\":";emit_kernel_info(report,workload.runs.session.info,workload.runs.session.conditional);report<<",\"mode\":\""<<mode<<"\",\"clock_pair_median_ns\":"<<clock_pair_ns<<",\"performance_acceptance\":false,\"original_production_backend_counters_available\":"<<(original_backend_counters?"true":"false")<<",\"counter_columns\":[\"numerator_direct_calls\",\"numerator_mu_calls\",\"numerator_t3_calls\",\"numerator_t2_calls\",\"numerator_direct_rows\",\"numerator_mu_rows\",\"numerator_t3_rows\",\"numerator_t2_rows\",\"preparation_calls\",\"preparation_rows\",\"wavefunction_calls\",\"rust_numerator_wire_calls\",\"rust_numerator_wire_rows\",\"rust_master_wire_calls\",\"rust_master_wire_rows\",\"shared_live_backend_calls\",\"shared_live_backend_rows\",\"live_flag_delta\",\"direct_rust_master_calls\",\"direct_rust_master_rows\"],\"records\":[";
  bool first=true;double checksum=0.;
  for(unsigned scope:scopes)for(bool changing:{false,true})for(size_t width:widths){
    std::array<std::vector<size_t>,3>indices;for(size_t phase=0;phase<3;++phase){indices[phase].resize(width);for(size_t row=0;row<width;++row)indices[phase][row]=changing?(row+phase)%3:0;}
    // Immutable coefficient/input states share ONE retained reducer/session.
    // Pointer selection is included identically in each implementation's call.
    std::array<std::unique_ptr<Physical>,3>prepared;
    if(scope==1)for(size_t phase=0;phase<3;++phase){prepared[phase]=std::make_unique<Physical>(argv[1],1);prepared[phase]->prepare(workload.pool,indices[phase]);}
    const auto execute=[&](bool rust,size_t phase)->const std::vector<SpPair>&{if(scope==1){workload.runs.use_prepared(*prepared[phase]);return rust?workload.runs.reduce_rust(&workload.replay):workload.runs.reduce_original(workload.replay);}return workload.execute(scope,rust,indices[phase]);};
    const auto counters=[&](){Counts total=counts(workload.physical,workload.runs,workload.live);if(scope==1){std::fill(total.begin(),total.begin()+8,0);for(const auto&phase:prepared){std::array<uint64_t,8>values{};checked(sp_provider_stats(phase->provider,values.data(),values.size()));for(size_t i=0;i<values.size();++i)total[i]+=values[i];}}return total;};
    double error=0.;size_t checked_components=0;
    // Warm every point and operation, checking both independent and paired
    // values before either optional timing or allocation diagnostics.
    for(size_t phase=0;phase<3;++phase){const auto original=execute(false,phase);const auto rust=execute(true,phase);
      if(scope==0){if(original.size()!=rust.size())throw std::runtime_error("preparation output mismatch");for(size_t i=0;i<rust.size();++i){error=std::max(error,near(complex(rust[i]),complex(original[i]),"baseline preparation"));++checked_components;}}
      else for(size_t row=0;row<width;++row){for(unsigned j=0;j<3;++j){error=std::max(error,near(complex(rust[row*5+j]),complex(original[row*5+j]),"scope paired Laurent"));error=std::max(error,near(complex(rust[row*5+j]),complex(workload.pool[indices[phase][row]].integrated[j]),"scope independent MadLoop"));++checked_components;}near(complex(rust[row*5]),complex(rust[row*5+3])+complex(rust[row*5+4]),"scope Rust split");near(complex(original[row*5]),complex(original[row*5+3])+complex(original[row*5+4]),"scope original split");}
    }
    if(!first){report<<',';}first=false;
    report<<"{\"scope\":\""<<names[scope]<<"\",\"workload\":\""<<(changing?"changing":"repeated")<<"\",\"width\":"<<width<<",\"rounds\":"<<rounds<<",\"contiguous_blocks\":true,\"counters_outside_blocks\":true,\"retained_reducer_sessions\":1,\"prebound_numerator_states\":"<<(scope==1?3:0)<<",\"correctness_components\":"<<checked_components<<",\"max_scaled_error\":"<<error<<",\"samples\":[";
    Counts original_counts{},rust_counts{};std::vector<std::array<double,2>>raw_blocks;raw_blocks.reserve(trials);
    if(timing){for(size_t trial=0;trial<trials;++trial){std::array<double,2>times{};for(unsigned turn=0;turn<2;++turn){const bool rust=(turn+(trial%2))%2;const auto before=counters();const std::vector<SpPair>*output=nullptr;auto start=Clock::now();for(size_t round=0;round<rounds;++round)output=&execute(rust,(round+trial)%3);auto end=Clock::now();const auto after=counters();times[rust]=std::chrono::duration<double,std::nano>(end-start).count();accumulate(rust?rust_counts:original_counts,difference(before,after));checksum+=output->back().re;}
      raw_blocks.push_back(times);if(trial){report<<',';}report<<"{\"original_ns\":"<<times[0]/rounds<<",\"rust_ns\":"<<times[1]/rounds<<",\"order\":\""<<(trial%2?"rust-first":"original-first")<<"\"}";
    }}
    report<<"],\"raw_block_columns\":[\"original_ns\",\"rust_ns\"],\"raw_block_nanoseconds\":[";for(size_t i=0;i<raw_blocks.size();++i){if(i){report<<',';}report<<'['<<raw_blocks[i][0]<<','<<raw_blocks[i][1]<<']';}report<<"],\"original_counter_deltas\":";emit_counts(report,original_counts);report<<",\"rust_counter_deltas\":";emit_counts(report,rust_counts);
#ifdef SP_ALLOCATION_DIAGNOSTICS
    if(allocations){report<<",\"rust_allocator_only\":true,\"allocation_columns\":[\"allocation_calls\",\"requested_bytes\",\"reallocation_calls\",\"reallocated_bytes\",\"deallocation_calls\"],\"allocation_scopes\":[\"total\",\"numerator\",\"masters\",\"physical_preparation\"],\"allocation_counts\":[";
      for(unsigned rust=0;rust<2;++rust){checked(sp_allocation_begin());const auto&output=execute(rust,1);std::array<uint64_t,20>values{};checked(sp_allocation_end(values.data(),values.size()));
        for(size_t column=0;column<5;++column){uint64_t scoped=values[5+column]+values[10+column]+values[15+column];if(scoped>values[column])throw std::runtime_error("allocation scopes overlap totals");}
        if(rust && values[0]+values[2]==0)throw std::runtime_error("Rust allocation diagnostic failed to observe expected work");
        checksum+=output.back().re;if(rust){report<<',';}emit_counts(report,values);}report<<']';}
#endif
#ifdef SP_EXECUTION_DIAGNOSTICS
    if(execution){report<<",\"execution_records\":[";
      for(unsigned rust=0;rust<2;++rust){
        checked(sp_execution_begin());workload.oneloop.begin();execution_diagnostic::begin_preparation();
        const auto start=Clock::now();const auto&output=execute(rust,1);const auto stop=Clock::now();
        const auto outer_preparation=execution_diagnostic::end_preparation();const auto&original_backend=workload.oneloop.end();std::array<uint64_t,147>values{};checked(sp_execution_end(values.data(),values.size()));
        const uint64_t total=std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count();
        const auto elapsed=[&](size_t first,size_t last){uint64_t value=0;for(size_t i=first;i<last;++i)value+=values[i*3+2];return value;};
        const uint64_t numerator=elapsed(0,4),preparation_provider=elapsed(4,5),preparation=outer_preparation[2],shared_backend=elapsed(5,33),direct_backend=elapsed(33,47),wire=elapsed(47,49);
        uint64_t cpp_backend=0;for(size_t i=0;i<16;++i)cpp_backend+=original_backend[i][2];
        if(preparation<preparation_provider)throw std::runtime_error("preparation provider exceeds outer preparation span");
        if(wire<shared_backend)throw std::runtime_error("backend span exceeds containing shared transport span");
        const uint64_t master=shared_backend+direct_backend+cpp_backend,lookup=wire-shared_backend,partitioned=numerator+preparation+wire+direct_backend+cpp_backend;
        if(total<partitioned)throw std::runtime_error("diagnostic scopes overlap total elapsed time");
        uint64_t backend_calls=0,backend_rows=0;
        for(size_t i=5;i<47;++i){backend_calls+=values[i*3];backend_rows+=values[i*3+1];}
        for(size_t i=0;i<16;++i){backend_calls+=original_backend[i][0];backend_rows+=original_backend[i][1];}
        if(scope==1&&(master!=0||backend_calls!=0||backend_rows!=0))throw std::runtime_error("frozen reduction executed a real master backend");
        if(scope!=0)for(size_t row=0;row<width;++row)for(unsigned j=0;j<3;++j)near(complex(output[row*5+j]),complex(workload.pool[indices[1][row]].integrated[j]),"instrumented independent Laurent");
        checksum+=output.back().re;
        if(rust){report<<',';}
        report<<"{\"implementation\":\""<<(rust?"rust":"original")<<"\",\"overall_ns\":"<<total<<",\"numerator_provider_abi_ns\":"<<numerator<<",\"preparation_provider_ns\":"<<preparation_provider<<",\"preparation_total_ns\":"<<preparation<<",\"preparation_transport_ns\":"<<(preparation-preparation_provider)<<",\"master_backend_ns\":"<<master<<",\"master_lookup_transport_ns\":"<<lookup<<",\"remainder_ns\":"<<(total-partitioned)<<",\"non_master_elapsed_ns\":"<<(total-master)<<",\"reduction_excluding_master_and_preparation_ns\":"<<(total-master-preparation)<<",\"rust_span_columns\":[\"calls\",\"rows\",\"inclusive_ns\"],\"rust_spans\":";emit_counts(report,values);
        report<<",\"preparation_total_span\":";emit_counts(report,outer_preparation);report<<",\"original_backend_spans\":[";for(size_t i=0;i<original_backend.size();++i){if(i){report<<',';}emit_counts(report,original_backend[i]);}report<<"]}";
      }report<<']';
    }
#endif
    report<<'}';
    workload.runs.use_prepared(workload.physical);
  }
  std::array<uint64_t,7>replay{};checked(sp_master_stats(workload.replay.handle(),replay.data(),7));if(replay[6]!=0)throw std::runtime_error("frozen union miss");
  std::array<uint64_t,3> final_backend{};checked(sp_master_execution_stats(workload.replay.handle(),final_backend.data(),final_backend.size()));
  if(final_backend!=workload.captured_backend)throw std::runtime_error("frozen master backend counters changed");
  report<<"],\"frozen_backend_call_delta\":0,\"frozen_backend_row_delta\":0,\"checksum\":"<<checksum<<",\"union_entries\":"<<replay[0]<<",\"union_replay_rows\":"<<replay[5]<<",\"union_misses\":"<<replay[6]<<"}\n";
  std::cout<<report.str()<<std::flush;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
