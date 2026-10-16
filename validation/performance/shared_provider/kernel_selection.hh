// Validation-only semantic selection and coverage checks, always outside timing.
#pragma once
#include "bridge.h"
#include <array>
#include <ostream>
#include <stdexcept>
#include <string>

inline unsigned kernel_selection(const std::string &name) {
  if(name=="symbolica")return SP_KERNEL_SYMBOLICA;
  if(name=="native-denominators")return SP_KERNEL_DENOMINATORS;
  if(name=="native")return SP_KERNEL_NATIVE;
  throw std::runtime_error("unknown reduction-kernel selection: "+name);
}
inline const char *kernel_name(unsigned selection) {
  switch(selection){case SP_KERNEL_SYMBOLICA:return "symbolica";case SP_KERNEL_DENOMINATORS:return "native-denominators";case SP_KERNEL_NATIVE:return "native";default:throw std::runtime_error("invalid reduction-kernel ID");}
}
inline void validate_kernel_info(const std::array<uint64_t,9>&info,unsigned selection) {
  if(info[0]!=selection||info[1]>1||(selection&&info[1]!=1))throw std::runtime_error("kernel selection/feature mismatch");
  const std::array<uint64_t,7> expected{{selection?1u:0u,selection?1u:0u,selection==2?1u:0u,selection==2?1u:0u,0,0,0}};
  for(size_t i=0;i<expected.size();++i)if(info[i+2]!=expected[i])throw std::runtime_error("unexpected actual reduction-family coverage");
}
inline void emit_kernel_info(std::ostream&out,const std::array<uint64_t,9>&info) {
  const std::array<const char*,7> names{{"t2_denominators","t3_denominators","bubble_source_projection","triangle_bubble_correction","other_reduction","integration","accumulation"}};
  out<<"{\"selection\":\""<<kernel_name(info[0])<<"\",\"native_feature_enabled\":"<<(info[1]?"true":"false")<<",\"families\":{";
  for(size_t i=0;i<names.size();++i){if(i){out<<',';}out<<'"'<<names[i]<<"\":\""<<(info[i+2]?"native-rust":"symbolica-interpreter")<<'"';}
  out<<"},\"selection_scope\":\"Setup-only partial catalog; unused stages need not execute. Numerator, geometry and masters are independent.\"}";
}
