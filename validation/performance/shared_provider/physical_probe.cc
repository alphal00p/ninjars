// Complete shared-provider correctness control. No timings are taken.
#include "physical_bridge.h"
#include "masters.hh"
#include <algorithm>
#include <iomanip>
#include <limits>
#include <string>

using C=ninja::Complex;
static double discrepancy(C actual,C expected){
  if(!std::isfinite(actual.real())||!std::isfinite(actual.imag())||!std::isfinite(expected.real())||!std::isfinite(expected.imag()))throw std::runtime_error("nonfinite comparison");
  return std::abs(actual-expected)/std::max(1.,std::abs(expected));
}
static double near(C actual,C expected,const char *stage){
  double error=discrepancy(actual,expected);
  if(error>1e-9){std::cerr<<stage<<" actual="<<actual<<" expected="<<expected<<" error="<<error<<'\n';throw std::runtime_error(stage);}
  return error;
}
struct Point{
  std::array<double,20> momenta{};std::array<int8_t,5> helicities{};std::array<double,7> model{};double scale{};
  std::vector<std::array<SpPair,5>> raw_inputs;std::vector<SpPair> raw_expected;std::array<SpPair,3> integrated{};
};
struct Physical{
  SpPhysical *handle{};SpProvider *provider{};std::array<uint64_t,8> info{};
  std::vector<double> momenta,model,scales;std::vector<int8_t> helicities;std::vector<SpPair> packed;
  explicit Physical(const std::string &path,unsigned mode){checked(sp_physical_create(reinterpret_cast<const uint8_t*>(path.data()),path.size(),mode,&handle));checked(sp_physical_provider(handle,&provider));checked(sp_physical_info(handle,info.data(),info.size()));}
  ~Physical(){sp_physical_destroy(handle);}
  Physical(const Physical&)=delete;Physical&operator=(const Physical&)=delete;
  void load(const std::vector<Point>&pool,const std::vector<size_t>&indices){
    momenta.clear();helicities.clear();model.clear();scales.clear();
    for(auto index:indices){const auto&p=pool.at(index);momenta.insert(momenta.end(),p.momenta.begin(),p.momenta.end());helicities.insert(helicities.end(),p.helicities.begin(),p.helicities.end());model.insert(model.end(),p.model.begin(),p.model.end());scales.push_back(p.scale);}
    packed.resize(indices.size()*51);
  }
  int prepare_status(){return sp_physical_prepare(handle,scales.size(),momenta.data(),momenta.size(),helicities.data(),helicities.size(),model.data(),model.size(),scales.data(),scales.size(),packed.data(),packed.size());}
  void prepare(const std::vector<Point>&pool,const std::vector<size_t>&indices){load(pool,indices);checked(prepare_status());}
  std::vector<SpPair> coefficients(){std::vector<SpPair> result(scales.size()*info[4]);checked(sp_physical_coefficients(handle,result.data(),result.size()));return result;}
};
struct Session{SpSession *handle{};Session(){checked(sp_session_create(5,4,3,&handle));}~Session(){sp_session_destroy(handle);}Session(const Session&)=delete;Session&operator=(const Session&)=delete;};
struct Runs{
  Physical &physical;SharedNumerator numerator;Session session;
  std::array<ninja::RealMomentum,5> shifts{};std::array<C,5> masses{};std::array<double,25> matrix{};
  ninja::Amplitude<ninja::ComplexMasses> original;
  std::vector<SpPair> v,m,s,sm,output;
  explicit Runs(Physical&p):physical(p),numerator(p.provider),original(5,4,shifts.data(),masses.data()){}
  const std::vector<SpPair>&run_original(const std::vector<Point>&pool,const std::vector<size_t>&indices,SharedMasters&masters){
    physical.prepare(pool,indices);output.resize(indices.size()*5);
    for(size_t row=0;row<indices.size();++row){const auto *data=physical.packed.data()+row*51;
      for(unsigned i=0;i<5;++i){shifts[i]=ninja::RealMomentum(data[4*i].re,data[4*i+1].re,data[4*i+2].re,data[4*i+3].re);masses[i]=complex(data[20+i]);}
      for(unsigned i=0;i<25;++i)matrix[i]=data[25+i].re;
      original.reset();original.setIntegralLibrary(masters).setRenormalizationScale(data[50].re).setSMatrix(matrix.data());numerator.lane(row);
      if(original.evaluate(numerator))throw std::runtime_error("original physical reduction failed");
      for(unsigned i=0;i<3;++i)output[row*5+i]=pair(original[i]);
      output[row*5+3]=pair(original.getCutConstructiblePart());output[row*5+4]=pair(original.getRationalPart());
    }
    return output;
  }
  const std::vector<SpPair>&run_rust(const std::vector<Point>&pool,const std::vector<size_t>&indices,SharedMasters&masters){
    physical.prepare(pool,indices);v.clear();m.clear();s.clear();sm.clear();
    for(size_t row=0;row<indices.size();++row){const auto *data=physical.packed.data()+row*51;v.insert(v.end(),data,data+20);m.insert(m.end(),data+20,data+25);sm.insert(sm.end(),data+25,data+50);s.push_back(data[50]);}
    output.resize(indices.size()*5);
    checked(sp_session_run(session.handle,physical.provider,masters.handle(),indices.size(),v.data(),v.size(),m.data(),m.size(),s.data(),s.size(),sm.data(),sm.size(),output.data(),output.size()));return output;
  }
};
static void emit(const std::vector<SpPair>&values){std::cout<<'[';for(size_t i=0;i<values.size();++i){if(i)std::cout<<',';std::cout<<'['<<values[i].re<<','<<values[i].im<<']';}std::cout<<']';}
int main(int argc,char**argv){try{
  if(argc!=3)throw std::runtime_error("usage: physical-probe DOT maximum-integrated-batch");
  const size_t maximum=std::stoul(argv[2]);size_t count{};std::cin>>count;if(count!=3)throw std::runtime_error("exact three-point pool required");std::vector<Point>pool(count);
  for(auto&p:pool){for(auto&x:p.momenta)std::cin>>x;for(auto&x:p.helicities){int value;std::cin>>value;x=static_cast<int8_t>(value);}for(auto&x:p.model)std::cin>>x;std::cin>>p.scale;size_t samples;std::cin>>samples;if(samples!=70)throw std::runtime_error("degree4 unisolvent grid required");p.raw_inputs.resize(samples);p.raw_expected.resize(samples);
    for(size_t i=0;i<samples;++i){for(unsigned j=0;j<4;++j)std::cin>>p.raw_inputs[i][j].re>>p.raw_inputs[i][j].im;std::cin>>p.raw_expected[i].re>>p.raw_expected[i].im;}
    for(auto&x:p.integrated)std::cin>>x.re>>x.im;
  }if(!std::cin)throw std::runtime_error("truncated physical protocol");
  ninja::setVerbosity(ninja::Verbose::NONE);ninja::setTest(ninja::Test::NONE);
  Physical physical(argv[1],1);if(physical.info[0]!=5||physical.info[1]!=4||physical.info[2]!=5||physical.info[3]!=7||physical.info[5]!=51||physical.info[7]!=5)throw std::runtime_error("physical dimensions mismatch");
  const std::vector<size_t> all{0,1,2};physical.prepare(pool,all);auto heads=physical.coefficients();
  std::array<double,6> oracle{};checked(sp_physical_float256_audit(physical.handle,oracle.data(),oracle.size()));if(oracle[0]>1e-9)throw std::runtime_error("factored coefficient Float256 oracle failed");
  double head_error=0.;{Physical interpreter(argv[1],0);interpreter.prepare(pool,all);auto other=interpreter.coefficients();if(other.size()!=heads.size())throw std::runtime_error("head count mismatch");for(size_t i=0;i<heads.size();++i)head_error=std::max(head_error,near(complex(heads[i]),complex(other[i]),"interpreter/JIT heads"));}
  double raw_error=0.;size_t raw_checks=0;
  for(size_t point=0;point<count;++point){const auto&p=pool[point];std::vector<SpIdentity>ids(p.raw_inputs.size(),{point,0});std::vector<SpPair>output(p.raw_inputs.size());checked(sp_call(physical.provider,0,nullptr,0,ids.data(),ids.size(),p.raw_inputs[0].data(),p.raw_inputs.size()*5,nullptr,0,output.data(),output.size()));for(size_t i=0;i<output.size();++i){raw_error=std::max(raw_error,near(complex(output[i]),complex(p.raw_expected[i]),"MadLoop raw numerator"));++raw_checks;}}
  // Fail a middle row at different boundary stages, retaining the complete
  // previous coefficient binding and caller output, then reactivate it.
  const auto packed=physical.packed;
  for(unsigned failure=0;failure<4;++failure){
    physical.load(pool,all);
    if(failure==0)physical.helicities[5]=0;
    if(failure==1)physical.momenta[20]+=1.;
    if(failure==2)physical.model[7]=1e200;
    if(failure==3)physical.scales[1]=std::numeric_limits<double>::quiet_NaN();
    if(physical.prepare_status()==0)throw std::runtime_error("invalid physical preparation accepted");
    for(size_t i=0;i<packed.size();++i)if(discrepancy(complex(physical.packed[i]),complex(packed[i]))!=0)throw std::runtime_error("preparation error mutated output");
    const auto retained=physical.coefficients();for(size_t i=0;i<heads.size();++i)if(discrepancy(complex(retained[i]),complex(heads[i]))!=0)throw std::runtime_error("preparation error mutated coefficient binding");
  }
  physical.prepare(pool,all);
  // Model parameters are free numerical inputs, not silently frozen at setup.
  auto changed_pool=pool;changed_pool[1].model[1]*=1.01;changed_pool[1].model[4]*=1.02;
  physical.prepare(changed_pool,all);auto changed_heads=physical.coefficients();bool model_changed=false;
  for(size_t i=physical.info[4];i<2*physical.info[4];++i)model_changed|=discrepancy(complex(changed_heads[i]),complex(heads[i]))>1e-12;
  if(!model_changed)throw std::runtime_error("model parameter change did not propagate");
  std::array<double,6> changed_oracle{};checked(sp_physical_float256_audit(physical.handle,changed_oracle.data(),6));if(changed_oracle[0]>1e-9)throw std::runtime_error("changed model Float256 oracle failed");
  physical.prepare(pool,all);
  Runs runs(physical);SharedMasters union_masters;
  const auto captured_original=runs.run_original(pool,all,union_masters);const auto captured_rust=runs.run_rust(pool,all,union_masters);
  double integrated_error=0.,madloop_error=0.;for(size_t row=0;row<count;++row){for(unsigned j=0;j<3;++j){integrated_error=std::max(integrated_error,near(complex(captured_rust[row*5+j]),complex(captured_original[row*5+j]),"shared integrated Laurent"));madloop_error=std::max(madloop_error,near(complex(captured_rust[row*5+j]),complex(pool[row].integrated[j]),"MadLoop integrated Laurent"));}near(complex(captured_rust[row*5]),complex(captured_rust[row*5+3])+complex(captured_rust[row*5+4]),"Rust finite split");near(complex(captured_original[row*5]),complex(captured_original[row*5+3])+complex(captured_original[row*5+4]),"original finite split");}
  std::array<uint64_t,7> before{},after{};checked(sp_master_stats(union_masters.handle(),before.data(),7));checked(sp_master_freeze(union_masters.handle()));
  size_t batch_checks=0;std::vector<size_t> widths;
  for(size_t width:std::vector<size_t>{0,1,3,8,31,32,128,1024}){if(width>maximum)continue;std::vector<size_t>indices(width);for(size_t i=0;i<width;++i)indices[i]=(2*i+1)%count;
    const auto original=runs.run_original(pool,indices,union_masters);const auto rust=runs.run_rust(pool,indices,union_masters);
    for(size_t row=0;row<width;++row)for(unsigned j=0;j<3;++j){near(complex(original[row*5+j]),complex(captured_original[indices[row]*5+j]),"original replay permutation");integrated_error=std::max(integrated_error,near(complex(rust[row*5+j]),complex(original[row*5+j]),"shared batch Laurent"));++batch_checks;}
    widths.push_back(width);
  }
  checked(sp_master_stats(union_masters.handle(),after.data(),7));if(before[0]!=after[0]||before[4]!=after[4]||after[6]!=0)throw std::runtime_error("frozen replay executed backend or missed a key");
  // Preparation alone covers all required widths even if integration is bounded.
  for(size_t width:std::vector<size_t>{0,1,8,32,128,1024,31}){std::vector<size_t>indices(width);for(size_t i=0;i<width;++i)indices[i]=i%count;physical.prepare(pool,indices);}
  SharedMasters live;checked(sp_master_set_live(live.handle()));std::array<uint64_t,3> live_before{},live_middle{},live_after{};checked(sp_master_execution_stats(live.handle(),live_before.data(),3));
  const auto live_original=runs.run_original(pool,all,live);checked(sp_master_execution_stats(live.handle(),live_middle.data(),3));const auto live_rust=runs.run_rust(pool,all,live);checked(sp_master_execution_stats(live.handle(),live_after.data(),3));
  for(size_t row=0;row<count;++row)for(unsigned j=0;j<3;++j){near(complex(live_original[row*5+j]),complex(captured_original[row*5+j]),"live original masters");near(complex(live_rust[row*5+j]),complex(captured_rust[row*5+j]),"live Rust masters");}
  std::array<uint64_t,3> live_repeat{};const auto repeated=runs.run_rust(pool,all,live);checked(sp_master_execution_stats(live.handle(),live_repeat.data(),3));
  for(size_t i=0;i<repeated.size();++i)near(complex(repeated[i]),complex(live_rust[i]),"live repeated point");
  if(live_repeat[0]<=live_after[0]||live_repeat[1]<=live_after[1])throw std::runtime_error("repeated points skipped live master expressions");
  if(live_middle[0]<=live_before[0]||live_middle[1]<=live_before[1]||live_after[0]<=live_middle[0]||live_after[1]<=live_middle[1]||live_after[2]!=1)throw std::runtime_error("complete paths did not execute master expressions");
  std::array<uint64_t,7> live_stats{};checked(sp_master_stats(live.handle(),live_stats.data(),7));if(live_stats[0]!=0||live_stats[5]!=0)throw std::runtime_error("live mode used master replay");
  std::array<uint64_t,3> prep{};checked(sp_physical_stats(physical.handle,prep.data(),3));
  std::cout<<std::setprecision(17)<<"{\"success\":true,\"coefficient_heads\":"<<physical.info[4]<<",\"raw_components_checked\":"<<raw_checks<<",\"raw_max_scaled_error\":"<<raw_error<<",\"coefficient_oracle_max_scaled_error\":"<<oracle[0]<<",\"jit_interpreter_max_scaled_error\":"<<head_error<<",\"integrated_max_scaled_error\":"<<integrated_error<<",\"madloop_integrated_max_scaled_error\":"<<madloop_error<<",\"batch_laurent_components_checked\":"<<batch_checks<<",\"integrated_widths\":[";
  for(size_t i=0;i<widths.size();++i){if(i)std::cout<<',';std::cout<<widths[i];}std::cout<<"],\"preparation_widths\":[0,1,8,32,128,1024,31],\"original\":";emit(captured_original);std::cout<<",\"rust\":";emit(captured_rust);
  std::cout<<",\"master_union_entries\":"<<after[0]<<",\"master_union_both_sources\":"<<after[1]<<",\"master_union_rust_only\":"<<after[2]<<",\"master_union_original_only\":"<<after[3]<<",\"live_original_backend_calls\":"<<live_middle[0]-live_before[0]<<",\"live_original_backend_rows\":"<<live_middle[1]-live_before[1]<<",\"live_rust_backend_calls\":"<<live_after[0]-live_middle[0]<<",\"live_rust_backend_rows\":"<<live_after[1]-live_middle[1]<<",\"live_rust_repeat_backend_calls\":"<<live_repeat[0]-live_after[0]<<",\"live_rust_repeat_backend_rows\":"<<live_repeat[1]-live_after[1]<<",\"changed_model_coefficient_oracle_max_scaled_error\":"<<changed_oracle[0]<<",\"preparation_failure_controls\":4,\"preparation_calls\":"<<prep[0]<<",\"preparation_rows\":"<<prep[1]<<",\"wavefunction_calls\":"<<prep[2]<<",\"transactional_preparation_checked\":true,\"same_preparation_both_paths\":true,\"rational_separation_parity_complete\":false,\"production_backend_comparison\":false,\"performance_acceptance\":false}\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
