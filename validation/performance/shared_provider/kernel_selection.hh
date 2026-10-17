// Validation-only semantic selection and coverage checks, always outside timing.
#pragma once
#include "bridge.h"
#include <array>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

inline unsigned kernel_selection(const std::string &name) {
  if(name=="symbolica")return SP_KERNEL_SYMBOLICA;
  if(name=="native-denominators")return SP_KERNEL_DENOMINATORS;
  if(name=="native")return SP_KERNEL_NATIVE;
  if(name=="interpreted-rank")return SP_KERNEL_INTERPRETED_RANK;
  if(name=="native-rank")return SP_KERNEL_NATIVE_RANK;
  throw std::runtime_error("unknown reduction-kernel selection: "+name);
}
inline const char *kernel_name(unsigned selection) {
  switch(selection){case SP_KERNEL_SYMBOLICA:return "symbolica";case SP_KERNEL_DENOMINATORS:return "native-denominators";case SP_KERNEL_NATIVE:return "native";case SP_KERNEL_INTERPRETED_RANK:return "interpreted-rank";case SP_KERNEL_NATIVE_RANK:return "native-rank";default:throw std::runtime_error("invalid reduction-kernel ID");}
}
inline bool rank_selection(unsigned selection) {
  return selection==SP_KERNEL_INTERPRETED_RANK||selection==SP_KERNEL_NATIVE_RANK;
}
inline bool requires_native(unsigned selection) {
  return selection==SP_KERNEL_DENOMINATORS||selection==SP_KERNEL_NATIVE||selection==SP_KERNEL_NATIVE_RANK;
}
inline std::vector<unsigned> supported_kernel_selections(size_t n,size_t rank,bool compiled) {
  std::vector<unsigned> result{SP_KERNEL_SYMBOLICA};
  if(compiled){result.push_back(SP_KERNEL_DENOMINATORS);result.push_back(SP_KERNEL_NATIVE);}
  if(n==5&&rank==4){result.push_back(SP_KERNEL_INTERPRETED_RANK);if(compiled)result.push_back(SP_KERNEL_NATIVE_RANK);}
  return result;
}
inline void validate_kernel_info(const std::array<uint64_t,10>&info,unsigned selection) {
  kernel_name(selection); // Validate the semantic ID even if coverage is empty.
  if(info[0]!=selection||info[1]>1||(requires_native(selection)&&info[1]!=1))throw std::runtime_error("kernel selection/feature mismatch");
  const bool denominators=requires_native(selection),catalog=selection==SP_KERNEL_NATIVE||selection==SP_KERNEL_NATIVE_RANK;
  const std::array<uint64_t,8> expected{{denominators?1u:0u,denominators?1u:0u,catalog?1u:0u,catalog?1u:0u,catalog?1u:0u,0,0,0}};
  for(size_t i=0;i<expected.size();++i)if(info[i+2]!=expected[i])throw std::runtime_error("unexpected actual reduction-family coverage");
}
inline void validate_rank_correction_info(const std::array<uint64_t,4>&info,unsigned selection,size_t n,size_t rank) {
  if(info[0]!=n||info[1]!=rank||info[2]!=static_cast<uint64_t>(rank_selection(selection))||info[3]!=static_cast<uint64_t>(selection==SP_KERNEL_NATIVE_RANK))throw std::runtime_error("conditional correction/topology mismatch");
  if(rank_selection(selection)&&(n!=5||rank!=4))throw std::runtime_error("unsupported rank correction topology");
}
inline void emit_kernel_info(std::ostream&out,const std::array<uint64_t,10>&info,const std::array<uint64_t,4>&conditional) {
  const std::array<const char*,8> names{{"t2_denominators","t3_denominators","bubble_source_projection","triangle_bubble_correction","tadpole_source_projection","other_reduction","integration","accumulation"}};
  out<<"{\"selection\":\""<<kernel_name(info[0])<<"\",\"native_feature_enabled\":"<<(info[1]?"true":"false")<<",\"families\":{";
  for(size_t i=0;i<names.size();++i){if(i){out<<',';}out<<'"'<<names[i]<<"\":\""<<(info[i+2]?"native-rust":"symbolica-interpreter")<<'"';}
  out<<"},\"topology\":{\"propagators\":"<<conditional[0]<<",\"rank\":"<<conditional[1]<<'}';
  if(conditional[2])out<<",\"conditional_corrections\":{\"backend\":\""<<(conditional[3]?"native-rust":"symbolica-interpreter")<<"\",\"guard\":\"all-active-source-kind-structural-zeros\",\"fallback\":\"reported-generic-families\"}";
  out<<",\"selection_scope\":\"Setup-only partial catalog; family fields describe generic fallback and conditional corrections require the full source-kind zero guard. Numerator, geometry and masters are independent.\"}";
}
