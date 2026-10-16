// Diagnostic-only spans around the actual original production backend.
// No wrapper or clocks exist in ordinary aggregate timing builds.
#pragma once
#include <ninja/avholo.hh>
#include <array>
#include <chrono>
#include <cstdint>
extern "C" int32_t sp_execution_begin();
extern "C" int32_t sp_execution_end(uint64_t *,size_t);

namespace execution_diagnostic {
inline thread_local bool preparation_enabled=false;
inline thread_local std::array<uint64_t,3> preparation{};
inline void begin_preparation(){preparation={};preparation_enabled=true;}
inline auto end_preparation(){preparation_enabled=false;return preparation;}
class PreparationScope {
  bool enabled_;size_t rows_;std::chrono::steady_clock::time_point start_{};
public:
  explicit PreparationScope(size_t rows):enabled_(preparation_enabled),rows_(rows){if(enabled_)start_=std::chrono::steady_clock::now();}
  ~PreparationScope(){if(enabled_){++preparation[0];preparation[1]+=rows_;preparation[2]+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start_).count();}}
};
}

class TimedOneLoop final:public ninja::IntegralLibrary {
  ninja::AvHOneLoop inner_;
  bool enabled_{false};
  std::array<std::array<uint64_t,3>,17> records_{};
  template<class F>void measure(size_t kind,F&&call){
    if(!enabled_){call();return;}
    const auto start=std::chrono::steady_clock::now();call();
    const auto elapsed=std::chrono::steady_clock::now()-start;
    ++records_[kind][0];records_[kind][1]+=kind==16?0:1;
    records_[kind][2]+=std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
  }
public:
  void begin(){records_={};enabled_=true;}
  const auto&end(){enabled_=false;return records_;}
  void init(ninja::Real scale)override{measure(16,[&]{inner_.init(scale);});}
#define SP_MASTER_SPAN(NAME,KIND,ARGS,CALL) void NAME ARGS override{measure(KIND,[&]{inner_.NAME CALL;});}
  SP_MASTER_SPAN(getTadpoleIntegralCM,0,(ninja::Complex out[3],const ninja::Complex&m),(out,m))
  SP_MASTER_SPAN(getTadpoleIntegralRM,0,(ninja::Complex out[3],ninja::Real m),(out,m))
  SP_MASTER_SPAN(getBubbleIntegralCM,1,(ninja::Complex out[3],ninja::Real s,const ninja::Complex&m1,const ninja::Complex&m2),(out,s,m1,m2))
  SP_MASTER_SPAN(getBubbleIntegralRM,1,(ninja::Complex out[3],ninja::Real s,ninja::Real m1,ninja::Real m2),(out,s,m1,m2))
  SP_MASTER_SPAN(getRank2BubbleIntegralCM,14,(ninja::Complex b11[3],ninja::Complex b1[3],ninja::Complex b0[3],ninja::Real s,const ninja::Complex&m1,const ninja::Complex&m2),(b11,b1,b0,s,m1,m2))
  SP_MASTER_SPAN(getRank2BubbleIntegralRM,14,(ninja::Complex b11[3],ninja::Complex b1[3],ninja::Complex b0[3],ninja::Real s,ninja::Real m1,ninja::Real m2),(b11,b1,b0,s,m1,m2))
  SP_MASTER_SPAN(getRank3BubbleIntegralCM,15,(ninja::Complex b111[3],ninja::Complex b11[3],ninja::Complex b1[3],ninja::Complex b0[3],ninja::Real s,const ninja::Complex&m1,const ninja::Complex&m2),(b111,b11,b1,b0,s,m1,m2))
  SP_MASTER_SPAN(getRank3BubbleIntegralRM,15,(ninja::Complex b111[3],ninja::Complex b11[3],ninja::Complex b1[3],ninja::Complex b0[3],ninja::Real s,ninja::Real m1,ninja::Real m2),(b111,b11,b1,b0,s,m1,m2))
  SP_MASTER_SPAN(getTriangleIntegralCM,5,(ninja::Complex out[3],ninja::Real s21,ninja::Real s32,ninja::Real s13,const ninja::Complex&m1,const ninja::Complex&m2,const ninja::Complex&m3),(out,s21,s32,s13,m1,m2,m3))
  SP_MASTER_SPAN(getTriangleIntegralRM,5,(ninja::Complex out[3],ninja::Real s21,ninja::Real s32,ninja::Real s13,ninja::Real m1,ninja::Real m2,ninja::Real m3),(out,s21,s32,s13,m1,m2,m3))
  SP_MASTER_SPAN(getBoxIntegralCM,6,(ninja::Complex out[3],ninja::Real s21,ninja::Real s32,ninja::Real s43,ninja::Real s14,ninja::Real s31,ninja::Real s42,const ninja::Complex&m1,const ninja::Complex&m2,const ninja::Complex&m3,const ninja::Complex&m4),(out,s21,s32,s43,s14,s31,s42,m1,m2,m3,m4))
  SP_MASTER_SPAN(getBoxIntegralRM,6,(ninja::Complex out[3],ninja::Real s21,ninja::Real s32,ninja::Real s43,ninja::Real s14,ninja::Real s31,ninja::Real s42,ninja::Real m1,ninja::Real m2,ninja::Real m3,ninja::Real m4),(out,s21,s32,s43,s14,s31,s42,m1,m2,m3,m4))
#undef SP_MASTER_SPAN
};
