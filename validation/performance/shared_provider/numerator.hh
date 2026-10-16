// Validation-only original Ninja adapter for the shared Rust provider.
#pragma once
#include "bridge.h"
#include <ninja/ninja.hh>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

static void checked(int code) {
  if (code) { std::array<uint8_t, 1024> text{}; sp_error(text.data(),text.size());
    throw std::runtime_error(reinterpret_cast<const char *>(text.data())); }
}
static SpPair pair(const ninja::Complex &z) { return {z.real(),z.imag()}; }
static ninja::Complex complex(SpPair z) { return {z.re,z.im}; }
static void close(ninja::Complex actual,ninja::Complex expected) {
  if (!std::isfinite(actual.real()) || !std::isfinite(actual.imag()) ||
      !std::isfinite(expected.real()) || !std::isfinite(expected.imag()) ||
      std::abs(actual-expected)>1e-11*(1.+std::abs(expected)))
    throw std::runtime_error("shared provider response mismatch");
}
class SharedNumerator final:public ninja::Numerator {
  SpProvider *handle_{};
  std::array<uint64_t,3> info_{};
  uint64_t lane_{};
  std::vector<SpPair> input_, output_;
  std::vector<int32_t> powers_;
  void put(const ninja::ComplexMomentum &q) {for(unsigned j=0;j<4;++j)input_.push_back(pair(q[j]));}
  void call(unsigned kind,int cut,const ninja::PartitionInt *partition,unsigned columns) {
    if(cut<0||static_cast<uint64_t>(cut)>info_[0])throw std::runtime_error("invalid cut size");
    output_.resize(columns); const SpIdentity id{lane_,42};
    checked(sp_call(handle_,kind,powers_.data(),powers_.size(),&id,1,input_.data(),input_.size(),partition,cut,output_.data(),columns));
  }
public:
  explicit SharedNumerator(unsigned fixture){checked(sp_create(fixture,&handle_));checked(sp_info(handle_,info_.data(),info_.size()));input_.reserve(19);powers_.reserve(39);output_.reserve(13);}
  ~SharedNumerator() override {sp_destroy(handle_);}
  SharedNumerator(const SharedNumerator&)=delete;
  SharedNumerator&operator=(const SharedNumerator&)=delete;
  void bind(const std::vector<SpPair>&values,size_t rows){checked(sp_bind(handle_,values.data(),values.size(),rows));}
  void lane(uint64_t lane){lane_=lane;}
  SpProvider *handle(){return handle_;}
  ninja::Complex evaluate(const ninja::ComplexMomentum&q,const ninja::Complex&mu,int cut,const ninja::PartitionInt part[])override{
    input_.clear();powers_.clear();put(q);input_.push_back(pair(mu));call(0,cut,part,1);return complex(output_[0]);
  }
  void muExpansion(const ninja::ComplexMomentum v[],const ninja::PartitionInt part[],ninja::Complex out[])override{
    input_.clear();powers_.clear();put(v[0]);
    const bool higher=info_[1]>info_[0];
    if(higher)put(v[1]);else for(unsigned j=0;j<4;++j)input_.push_back({0.,0.});
    powers_.push_back(info_[1]);if(higher)powers_.push_back(info_[1]-1);
    call(1,4,part,higher?2:1);for(unsigned j=0;j<output_.size();++j)out[j]=complex(output_[j]);
  }
  void t3Expansion(const ninja::ComplexMomentum&a,const ninja::ComplexMomentum&e3,const ninja::ComplexMomentum&e4,const ninja::Complex&beta,int depth,int cut,const ninja::PartitionInt part[],ninja::Complex out[])override{
    if(depth<0||depth>4)throw std::runtime_error("invalid T3 depth");
    input_.clear();powers_.clear();put(a);put(e3);put(e4);input_.push_back(pair(beta));
    for(int d=0;d<=depth;++d)for(int mu=0;mu<=d/2;++mu){powers_.push_back(static_cast<int>(info_[1])-d);powers_.push_back(mu);}
    call(2,cut,part,powers_.size()/2);for(unsigned j=0;j<output_.size();++j)out[j]=complex(output_[j]);
  }
  void t2Expansion(const ninja::ComplexMomentum&a0,const ninja::ComplexMomentum&a1,const ninja::ComplexMomentum&e3,const ninja::ComplexMomentum&e4,const ninja::Complex beta[],int depth,int cut,const ninja::PartitionInt part[],ninja::Complex out[])override{
    if(depth<0||depth>3)throw std::runtime_error("invalid T2 depth");
    input_.clear();powers_.clear();put(a0);put(a1);put(e3);put(e4);for(unsigned j=0;j<3;++j)input_.push_back(pair(beta[j]));
    for(int d=0;d<=depth;++d)for(int x=0;x<=d;++x)for(int mu=0;mu<=(d-x)/2;++mu){powers_.push_back(static_cast<int>(info_[1])-d);powers_.push_back(x);powers_.push_back(mu);}
    call(3,cut,part,powers_.size()/3);for(unsigned j=0;j<output_.size();++j)out[j]=complex(output_[j]);
  }
};

