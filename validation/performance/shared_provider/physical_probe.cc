// Complete shared-provider correctness control. No timings are taken.
#include "physical_workload.hh"
int main(int argc,char**argv){try{
  const bool prune_swapped=take_pruned_swapped_flag(argc,argv);
  if(argc<3||argc>5)throw std::runtime_error("usage: physical-probe DOT maximum-integrated-batch [symbolica|native-denominators|native|interpreted-rank|native-rank|all] [native-block-rows] [--prune-swapped-arithmetic]");
  const bool all_kernels=argc>=4&&std::string(argv[3])=="all";
  const unsigned selection=argc>=4&&!all_kernels?kernel_selection(argv[3]):static_cast<unsigned>(SP_KERNEL_SYMBOLICA);
  const size_t block_rows=argc==5?native_block_rows(argv[4]):0;
  if(block_rows&&(all_kernels||!requires_native(selection)))throw std::runtime_error("native block rows require one explicit native selection");
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
  Runs runs(physical,selection,block_rows?SP_EXECUTION_FIXED_ROWS:SP_EXECUTION_WHOLE,block_rows,prune_swapped);SharedMasters union_masters;
  const auto captured_original=runs.run_original(pool,all,union_masters);const auto captured_rust=runs.run_rust(pool,all,union_masters);
  double integrated_error=0.,madloop_error=0.;for(size_t row=0;row<count;++row){for(unsigned j=0;j<3;++j){integrated_error=std::max(integrated_error,near(complex(captured_rust[row*5+j]),complex(captured_original[row*5+j]),"shared integrated Laurent"));madloop_error=std::max(madloop_error,near(complex(captured_rust[row*5+j]),complex(pool[row].integrated[j]),"MadLoop integrated Laurent"));}near(complex(captured_rust[row*5]),complex(captured_rust[row*5+3])+complex(captured_rust[row*5+4]),"Rust finite split");near(complex(captured_original[row*5]),complex(captured_original[row*5+3])+complex(captured_original[row*5+4]),"original finite split");}
  // Include both same-selection sessions in the frozen request union. This
  // setup is outside timing and only requested by the explicit pruning control.
  std::vector<SpPair> captured_full;
  if(prune_swapped){
    Runs full(physical,selection,block_rows?SP_EXECUTION_FIXED_ROWS:SP_EXECUTION_WHOLE,block_rows,false);
    captured_full=full.run_rust(pool,all,union_masters);
    if(captured_full.size()!=captured_rust.size())throw std::runtime_error("pruning capture dropped result heads");
    for(size_t i=0;i<captured_full.size();++i)near(complex(captured_rust[i]),complex(captured_full[i]),"pruning capture full five-head parity");
  }
  std::array<uint64_t,7> before{},after{};checked(sp_master_stats(union_masters.handle(),before.data(),7));checked(sp_master_freeze(union_masters.handle()));
  size_t batch_checks=0;std::vector<size_t> widths;
  for(size_t width:std::vector<size_t>{0,1,3,8,31,32,128,1024}){if(width>maximum)continue;std::vector<size_t>indices(width);for(size_t i=0;i<width;++i)indices[i]=(2*i+1)%count;
    const auto original=runs.run_original(pool,indices,union_masters);const auto rust=runs.run_rust(pool,indices,union_masters);
    for(size_t row=0;row<width;++row)for(unsigned j=0;j<3;++j){near(complex(original[row*5+j]),complex(captured_original[indices[row]*5+j]),"original replay permutation");integrated_error=std::max(integrated_error,near(complex(rust[row*5+j]),complex(original[row*5+j]),"shared batch Laurent"));++batch_checks;}
    widths.push_back(width);
  }
  checked(sp_master_stats(union_masters.handle(),after.data(),7));if(before[0]!=after[0]||before[4]!=after[4]||after[6]!=0)throw std::runtime_error("frozen replay executed backend or missed a key");
  // Optional one-artifact catalog parity: one fixed numerator and one frozen
  // union, new reducer sessions created before their runs. No provider switches.
  std::ostringstream catalog;
  catalog<<'[';
  bool first_catalog=true;
  if(all_kernels)for(unsigned selected:supported_kernel_selections(physical.info[0],physical.info[1],runs.session.info[1]!=0)){
    Runs candidate(physical,selected,SP_EXECUTION_WHOLE,0,prune_swapped);size_t checks=0;double error=0.;
    for(size_t width:widths){std::vector<size_t>indices(width);for(size_t row=0;row<width;++row)indices[row]=(2*row+1)%count;
      const auto values=candidate.run_rust(pool,indices,union_masters);
      if(values.size()!=width*5)throw std::runtime_error("catalog parity dropped output rows");
      for(size_t row=0;row<width;++row)for(unsigned column=0;column<5;++column){error=std::max(error,near(complex(values[row*5+column]),complex(captured_rust[indices[row]*5+column]),"catalog replay parity"));++checks;}
    }
    if(!first_catalog){catalog<<',';}first_catalog=false;catalog<<"{\"coverage\":";emit_kernel_info(catalog,candidate.session.info,candidate.session.conditional,candidate.session.execution,candidate.session.pruning);catalog<<",\"complex_components_checked\":"<<checks<<",\"max_scaled_error\":"<<std::setprecision(17)<<error<<'}';
  }
  catalog<<']';
  // Independent scheduling control, using the selected native catalog, the
  // same prepared numerator provider and the same frozen master-request union.
  std::ostringstream execution_parity;execution_parity<<'[';
  if(block_rows)for(unsigned policy:{SP_EXECUTION_WHOLE,SP_EXECUTION_FIXED_ROWS}){
    Runs candidate(physical,selection,policy,policy==SP_EXECUTION_FIXED_ROWS?block_rows:0,prune_swapped);size_t checks=0;double error=0.;
    for(size_t width:widths){std::vector<size_t>indices(width);for(size_t row=0;row<width;++row)indices[row]=(2*row+1)%count;
      const auto values=candidate.run_rust(pool,indices,union_masters);
      if(values.size()!=width*5)throw std::runtime_error("execution parity dropped output rows");
      for(size_t row=0;row<width;++row)for(unsigned column=0;column<5;++column){error=std::max(error,near(complex(values[row*5+column]),complex(captured_rust[indices[row]*5+column]),"native execution replay parity"));++checks;}
    }
    if(policy==SP_EXECUTION_FIXED_ROWS)execution_parity<<',';
    execution_parity<<"{\"coverage\":";emit_kernel_info(execution_parity,candidate.session.info,candidate.session.conditional,candidate.session.execution,candidate.session.pruning);execution_parity<<",\"complex_components_checked\":"<<checks<<",\"max_scaled_error\":"<<std::setprecision(17)<<error<<'}';
  }
  execution_parity<<']';
  // One preparation per width feeds both retained sessions. All five heads
  // use the same full-arithmetic reference; call/row counts are a separate
  // runtime control, not a claim of complete request-identity tracing.
  std::ostringstream pruning_parity;pruning_parity<<'[';
  if(prune_swapped){
    const unsigned policy=block_rows?SP_EXECUTION_FIXED_ROWS:SP_EXECUTION_WHOLE;
    Runs full(physical,selection,policy,block_rows,false),pruned(physical,selection,policy,block_rows,true);
    std::array<Runs*,2> candidates{{&full,&pruned}};
    std::array<size_t,2> checks{};std::array<double,2> errors{};
    std::array<std::array<uint64_t,8>,2> callbacks{};
    for(size_t width:widths){
      std::vector<size_t>indices(width);for(size_t row=0;row<width;++row)indices[row]=(2*row+1)%count;
      physical.prepare(pool,indices);
      std::array<std::array<uint64_t,8>,2> width_callbacks{};
      for(size_t option=0;option<2;++option){
        std::array<uint64_t,8> first{},last{};
        checked(sp_provider_stats(physical.provider,first.data(),first.size()));
        const auto &values=candidates[option]->reduce_rust(&union_masters);
        checked(sp_provider_stats(physical.provider,last.data(),last.size()));
        if(values.size()!=width*5)throw std::runtime_error("pruning parity dropped output rows or heads");
        for(size_t row=0;row<width;++row)for(unsigned column=0;column<5;++column){
          errors[option]=std::max(errors[option],near(complex(values[row*5+column]),complex(captured_full[indices[row]*5+column]),"pruning replay full five-head parity"));++checks[option];
        }
        for(size_t i=0;i<first.size();++i){
          if(last[i]<first[i])throw std::runtime_error("pruning callback counter decreased");
          width_callbacks[option][i]=last[i]-first[i];callbacks[option][i]+=width_callbacks[option][i];
        }
      }
      if(width_callbacks[0]!=width_callbacks[1])throw std::runtime_error("pruning changed full numerator call/row counts");
    }
    for(size_t option=0;option<2;++option){
      if(option)pruning_parity<<',';
      const auto &session=candidates[option]->session;
      pruning_parity<<"{\"coverage\":";emit_kernel_info(pruning_parity,session.info,session.conditional,session.execution,session.pruning);
      pruning_parity<<",\"complex_components_checked\":"<<checks[option]<<",\"max_scaled_error\":"<<std::setprecision(17)<<errors[option]<<",\"numerator_call_row_deltas\":[";
      for(size_t i=0;i<callbacks[option].size();++i){if(i)pruning_parity<<',';pruning_parity<<callbacks[option][i];}pruning_parity<<"]}";
    }
  }
  pruning_parity<<']';
  checked(sp_master_stats(union_masters.handle(),after.data(),7));if(before[0]!=after[0]||before[4]!=after[4]||after[6]!=0)throw std::runtime_error("catalog replay executed backend or missed a key");
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
  std::cout<<std::setprecision(17)<<"{\"success\":true,\"reduction_kernels\":";emit_kernel_info(std::cout,runs.session.info,runs.session.conditional,runs.session.execution,runs.session.pruning);std::cout<<",\"catalog_parity\":"<<catalog.str()<<",\"execution_parity\":"<<execution_parity.str()<<",\"pruning_parity\":"<<pruning_parity.str()<<",\"coefficient_heads\":"<<physical.info[4]<<",\"raw_components_checked\":"<<raw_checks<<",\"raw_max_scaled_error\":"<<raw_error<<",\"coefficient_oracle_max_scaled_error\":"<<oracle[0]<<",\"jit_interpreter_max_scaled_error\":"<<head_error<<",\"integrated_max_scaled_error\":"<<integrated_error<<",\"madloop_integrated_max_scaled_error\":"<<madloop_error<<",\"batch_laurent_components_checked\":"<<batch_checks<<",\"integrated_widths\":[";
  for(size_t i=0;i<widths.size();++i){if(i)std::cout<<',';std::cout<<widths[i];}std::cout<<"],\"preparation_widths\":[0,1,8,32,128,1024,31],\"original\":";emit(captured_original);std::cout<<",\"rust\":";emit(captured_rust);
  std::cout<<",\"master_union_entries\":"<<after[0]<<",\"master_union_both_sources\":"<<after[1]<<",\"master_union_rust_only\":"<<after[2]<<",\"master_union_original_only\":"<<after[3]<<",\"live_original_backend_calls\":"<<live_middle[0]-live_before[0]<<",\"live_original_backend_rows\":"<<live_middle[1]-live_before[1]<<",\"live_rust_backend_calls\":"<<live_after[0]-live_middle[0]<<",\"live_rust_backend_rows\":"<<live_after[1]-live_middle[1]<<",\"live_rust_repeat_backend_calls\":"<<live_repeat[0]-live_after[0]<<",\"live_rust_repeat_backend_rows\":"<<live_repeat[1]-live_after[1]<<",\"changed_model_coefficient_oracle_max_scaled_error\":"<<changed_oracle[0]<<",\"preparation_failure_controls\":4,\"preparation_calls\":"<<prep[0]<<",\"preparation_rows\":"<<prep[1]<<",\"wavefunction_calls\":"<<prep[2]<<",\"transactional_preparation_checked\":true,\"same_preparation_both_paths\":true,\"rational_separation_parity_complete\":false,\"production_backend_comparison\":false,\"performance_acceptance\":false}\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
