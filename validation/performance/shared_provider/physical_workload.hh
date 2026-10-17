// Shared validation workload and retained buffers. No timing policy here.
#pragma once
#include "physical_bridge.h"
#include "masters.hh"
#include "kernel_selection.hh"
#ifdef SP_EXECUTION_DIAGNOSTICS
#include "execution.hh"
#endif
#include <algorithm>
#include <iomanip>
#include <limits>
#include <string>

using C=ninja::Complex;
[[maybe_unused]] static double discrepancy(C actual,C expected){
  if(!std::isfinite(actual.real())||!std::isfinite(actual.imag())||!std::isfinite(expected.real())||!std::isfinite(expected.imag()))throw std::runtime_error("nonfinite comparison");
  return std::abs(actual-expected)/std::max(1.,std::abs(expected));
}
[[maybe_unused]] static double near(C actual,C expected,const char *stage){
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
  void prepare(const std::vector<Point>&pool,const std::vector<size_t>&indices){
#ifdef SP_EXECUTION_DIAGNOSTICS
    execution_diagnostic::PreparationScope span(indices.size());
#endif
    load(pool,indices);checked(prepare_status());
  }
  std::vector<SpPair> coefficients(){std::vector<SpPair> result(scales.size()*info[4]);checked(sp_physical_coefficients(handle,result.data(),result.size()));return result;}
};
struct Session{
  SpSession *handle{};std::array<uint64_t,10> info{};std::array<uint64_t,4> conditional{};
  explicit Session(size_t n,size_t rank,unsigned selection=SP_KERNEL_SYMBOLICA){
    checked(sp_session_create_with_kernels(n,rank,3,selection,&handle));
    try{checked(sp_session_kernel_info(handle,info.data(),info.size()));validate_kernel_info(info,selection);checked(sp_session_rank_correction_info(handle,conditional.data(),conditional.size()));validate_rank_correction_info(conditional,selection,n,rank);}
    catch(...){sp_session_destroy(handle);throw;}
  }
  ~Session(){sp_session_destroy(handle);}Session(const Session&)=delete;Session&operator=(const Session&)=delete;
};
struct Runs{
  Physical *physical;SharedNumerator numerator;Session session;
  std::array<ninja::RealMomentum,5> shifts{};std::array<C,5> masses{};std::array<double,25> matrix{};
  ninja::Amplitude<ninja::ComplexMasses> original;
  std::vector<SpPair> v,m,s,sm,output;
  explicit Runs(Physical&p,unsigned selection=SP_KERNEL_SYMBOLICA):physical(&p),numerator(p.provider),session(p.info[0],p.info[1],selection),original(5,4,shifts.data(),masses.data()){}
  void use_prepared(Physical&p){if(p.info!=physical->info)throw std::runtime_error("prepared provider topology differs");physical=&p;numerator.use_provider(p.provider);}
  const std::vector<SpPair>&run_original(const std::vector<Point>&pool,const std::vector<size_t>&indices,SharedMasters&masters){
    physical->prepare(pool,indices);return reduce_original(masters);
  }
  const std::vector<SpPair>&reduce_original(ninja::IntegralLibrary&masters){
    output.resize(physical->scales.size()*5);
    for(size_t row=0;row<physical->scales.size();++row){const auto *data=physical->packed.data()+row*51;
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
    physical->prepare(pool,indices);return reduce_rust(&masters);
  }
  const std::vector<SpPair>&reduce_rust(SharedMasters*masters){
    const size_t rows=physical->scales.size();v.clear();m.clear();s.clear();sm.clear();
    for(size_t row=0;row<rows;++row){const auto *data=physical->packed.data()+row*51;v.insert(v.end(),data,data+20);m.insert(m.end(),data+20,data+25);sm.insert(sm.end(),data+25,data+50);s.push_back(data[50]);}
    output.resize(rows*5);
    if(masters)checked(sp_session_run(session.handle,physical->provider,masters->handle(),rows,v.data(),v.size(),m.data(),m.size(),s.data(),s.size(),sm.data(),sm.size(),output.data(),output.size()));
    else checked(sp_session_run_production(session.handle,physical->provider,rows,v.data(),v.size(),m.data(),m.size(),s.data(),s.size(),sm.data(),sm.size(),output.data(),output.size()));
    return output;
  }
};
[[maybe_unused]] static void emit(const std::vector<SpPair>&values){std::cout<<'[';for(size_t i=0;i<values.size();++i){if(i)std::cout<<',';std::cout<<'['<<values[i].re<<','<<values[i].im<<']';}std::cout<<']';}
