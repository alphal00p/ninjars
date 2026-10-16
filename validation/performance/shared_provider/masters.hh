// Original Ninja's public integral-library extension point; validation only.
#pragma once
#include "numerator.hh"
#include <ninja/integral_library.hh>

class SharedMasters final:public ninja::IntegralLibrary {
  SpMasters *handle_{};
  SpPair scale_{1.,0.};
  void call(unsigned kind,ninja::Complex out[3],std::initializer_list<double> invariants,std::initializer_list<ninja::Complex> masses){
    if(invariants.size()>6||masses.size()>4)throw std::runtime_error("master wire shape exceeds original interface");
    std::array<SpPair,6> inv{};std::array<SpPair,4> mass{};
    unsigned index=0;for(double s:invariants)inv[index++]={s,0.};index=0;for(auto m:masses)mass[index++]=pair(m);
    std::array<SpPair,3> values{};
    checked(sp_master_call(handle_,kind,2,1,inv.data(),invariants.size(),mass.data(),masses.size(),&scale_,1,values.data(),values.size()));
    for(unsigned j=0;j<3;++j)out[j]=complex(values[j]);
  }
public:
  SharedMasters(){checked(sp_master_create(&handle_));}
  ~SharedMasters()override{sp_master_destroy(handle_);}
  SharedMasters(const SharedMasters&)=delete;SharedMasters&operator=(const SharedMasters&)=delete;
  SpMasters *handle(){return handle_;}
  void init(ninja::Real scale)override{scale_={scale,0.};}
  void getTadpoleIntegralCM(ninja::Complex out[3],const ninja::Complex&m)override{call(0,out,{}, {m});}
  void getTadpoleIntegralRM(ninja::Complex out[3],ninja::Real m)override{getTadpoleIntegralCM(out,m);}
  void getBubbleIntegralCM(ninja::Complex out[3],ninja::Real s,const ninja::Complex&m1,const ninja::Complex&m2)override{call(1,out,{s},{m1,m2});}
  void getBubbleIntegralRM(ninja::Complex out[3],ninja::Real s,ninja::Real m1,ninja::Real m2)override{getBubbleIntegralCM(out,s,m1,m2);}
  void getRank2BubbleIntegralCM(ninja::Complex b11[3],ninja::Complex b1[3],ninja::Complex b0[3],ninja::Real s,const ninja::Complex&m1,const ninja::Complex&m2)override{call(3,b11,{s},{m1,m2});call(2,b1,{s},{m1,m2});call(1,b0,{s},{m1,m2});}
  void getRank2BubbleIntegralRM(ninja::Complex b11[3],ninja::Complex b1[3],ninja::Complex b0[3],ninja::Real s,ninja::Real m1,ninja::Real m2)override{getRank2BubbleIntegralCM(b11,b1,b0,s,m1,m2);}
  // Override rank3 explicitly: never enter Ninja's lower-rank derivation.
  void getRank3BubbleIntegralCM(ninja::Complex b111[3],ninja::Complex b11[3],ninja::Complex b1[3],ninja::Complex b0[3],ninja::Real s,const ninja::Complex&m1,const ninja::Complex&m2)override{call(4,b111,{s},{m1,m2});getRank2BubbleIntegralCM(b11,b1,b0,s,m1,m2);}
  void getRank3BubbleIntegralRM(ninja::Complex b111[3],ninja::Complex b11[3],ninja::Complex b1[3],ninja::Complex b0[3],ninja::Real s,ninja::Real m1,ninja::Real m2)override{getRank3BubbleIntegralCM(b111,b11,b1,b0,s,m1,m2);}
  void getTriangleIntegralCM(ninja::Complex out[3],ninja::Real s21,ninja::Real s32,ninja::Real s13,const ninja::Complex&m1,const ninja::Complex&m2,const ninja::Complex&m3)override{call(5,out,{s21,s32,s13},{m1,m2,m3});}
  void getTriangleIntegralRM(ninja::Complex out[3],ninja::Real s21,ninja::Real s32,ninja::Real s13,ninja::Real m1,ninja::Real m2,ninja::Real m3)override{getTriangleIntegralCM(out,s21,s32,s13,m1,m2,m3);}
  void getBoxIntegralCM(ninja::Complex out[3],ninja::Real s21,ninja::Real s32,ninja::Real s43,ninja::Real s14,ninja::Real s31,ninja::Real s42,const ninja::Complex&m1,const ninja::Complex&m2,const ninja::Complex&m3,const ninja::Complex&m4)override{call(6,out,{s21,s32,s43,s14,s31,s42},{m1,m2,m3,m4});}
  void getBoxIntegralRM(ninja::Complex out[3],ninja::Real s21,ninja::Real s32,ninja::Real s43,ninja::Real s14,ninja::Real s31,ninja::Real s42,ninja::Real m1,ninja::Real m2,ninja::Real m3,ninja::Real m4)override{getBoxIntegralCM(out,s21,s32,s43,s14,s31,s42,m1,m2,m3,m4);}
};
