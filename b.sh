export TEST_TMPDIR=/tmp/bazel_cache
set -x 

sudo nvidia-smi -lgc 1200,1200; sudo nvidia-smi -lmc 1000,1000

#export XLA_FLAGS=--xla_disable_all_hlo_passes
#export XLA_FLAGS="--xla_disable_hlo_passes=layout-assignment"
export CUDA_VISIBLE_DEVICES=0

export TF_CPP_MIN_LOG_LEVEL=0
export TF_CPP_MAX_LOG_LEVEL=5
#export TF_CPP_VMODULE=heap_simulator=1
#export TF_CPP_VMODULE=gpu_compiler=2,buffer_assignment=2,heap_simulator=1
#export TF_CPP_VMODULE=ir_emitter=4,ir_emitter_unnested=2,gpu_compiler=2,elemental_ir_emitter=2,hlo_to_ir_bindings=2,gpu_compiler=2,compile_module_to_llvm_ir=4,ir_builder_mixin=4

#export TF_CPP_VMODULE=buffer_assignment=4,copy_insertion=4
#export TF_CPP_VMODULE=copy_insertion=4
export XLA_FLAGS="--xla_gpu_simplify_all_fp_conversions --xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_async_all_gather=true --xla_gpu_enable_async_reduce_scatter=true --xla_gpu_enable_triton_gemm=false"
#export XLA_FLAGS='--xla_disable_hlo_passes=layout-assignment --xla_gpu_simplify_all_fp_conversions --xla_dump_hlo_as_html --xla_dump_hlo_as_proto --xla_dump_hlo_as_text --xla_dump_to=xla_dump_add_bf16 --xla_dump_hlo_pass_re=.*'
export XLA_FLAGS=${XLA_FLAGS}' --xla_disable_all_hlo_passes'

#./bazel-5.3.0-linux-x86_64 run -c opt --config=cuda --action_env TF_CUDA_COMPUTE_CAPABILITIES=compute_80 --nocheck_visibility --copt="-Wno-error=switch" xla/service/copy_insertion_test  --test_filter=WhileCopyInsertionTest.DependentTupleElements
#./bazel-5.3.0-linux-x86_64 run -c opt --config=cuda --action_env TF_CUDA_COMPUTE_CAPABILITIES=compute_80 --nocheck_visibility --copt="-Wno-error=switch" xla/service/float_normalization_test 

#declare -a testArray=( \
#"module_0806.pjit_train_step.before_optimizations.txt" \
#"module_0806.pjit_train_step.0001.pre-spmd-partitioner.after_CallInliner.before_zero_sized_hlo_elimination.txt" \
#"module_0806.pjit_train_step.0000.pre-spmd-partitioner.after_pipeline-start.before_CallInliner.txt" \
#"module_0806.pjit_train_step.0009.spmd-simplify.after_algsimp.before_simplify-sorts.txt" \
#"module_0806.pjit_train_step.0008.spmd-simplify.after_pipeline-start.before_algsimp.txt" \
#"module_0806.pjit_train_step.0007.spmd-simplify.after_dce.before_pipeline-end.txt" \
#"module_0806.pjit_train_step.0006.spmd-simplify.after_constant_folding.before_simplify-conditional.txt" \
#"module_0806.pjit_train_step.0005.spmd-simplify.after_reshape-mover.before_constant_folding.txt" \
#"module_0806.pjit_train_step.0004.spmd-simplify.after_algsimp.before_simplify-sorts.txt" \
#"module_0806.pjit_train_step.0003.spmd-simplify.after_pipeline-start.before_algsimp.txt" \
#"module_0806.pjit_train_step.0002.spmd-partitioner.after_pipeline-start.before_spmd-simplify.txt" \
#"module_0806.pjit_train_step.0017.spmd-simplify.after_algsimp.before_simplify-sorts.txt" \
#"module_0806.pjit_train_step.0016.spmd-simplify.after_pipeline-start.before_algsimp.txt" \
#"module_0806.pjit_train_step.0015.spmd-simplify.after_reshape-mover.before_constant_folding.txt" \
#"module_0806.pjit_train_step.0014.spmd-simplify.after_algsimp.before_simplify-sorts.txt" \
#"module_0806.pjit_train_step.0013.spmd-simplify.after_pipeline-start.before_algsimp.txt" \
#"module_0806.pjit_train_step.0012.spmd-simplify.after_dce.before_pipeline-end.txt" \
#"module_0806.pjit_train_step.0011.spmd-simplify.after_constant_folding.before_simplify-conditional.txt" \
#"module_0806.pjit_train_step.0010.spmd-simplify.after_reshape-mover.before_constant_folding.txt" \
#"module_0806.pjit_train_step.0024.spmd-partitioner.after_spmd-simplify.before_hlo-constant-splitter.txt" \
#"module_0806.pjit_train_step.0023.spmd-simplify.after_pipeline-start.before_algsimp.txt" \
#"module_0806.pjit_train_step.0022.spmd-simplify.after_algsimp.before_simplify-sorts.txt" \
#"module_0806.pjit_train_step.0021.spmd-simplify.after_pipeline-start.before_algsimp.txt" \
#"module_0806.pjit_train_step.0020.spmd-simplify.after_algsimp.before_simplify-sorts.txt" \
#"module_0806.pjit_train_step.0019.spmd-simplify.after_pipeline-start.before_algsimp.txt" \
#"module_0806.pjit_train_step.0018.spmd-simplify.after_reshape-mover.before_constant_folding.txt" \
#"module_0806.pjit_train_step.0026.spmd-partitioner.after_sharding-propagation.before_spmd-partitioning.txt" \
#"module_0806.pjit_train_step.0025.spmd-partitioner.after_hlo-constant-splitter.before_sharding-propagation.txt" \
#"module_0806.pjit_train_step.0033.optimization.after_pipeline-start.before_topk-splitter.txt" \
#"module_0806.pjit_train_step.0032.spmd-partitioner.after_spmd-partitioning.before_collective-permute-motion.txt" \
#"module_0806.pjit_train_step.0031.spmd-cleanup.after_flatten-call-graph.before_pipeline-end.txt" \
#"module_0806.pjit_train_step.0030.spmd-cleanup.after_cse.before_flatten-call-graph.txt" \
#"module_0806.pjit_train_step.0029.spmd-cleanup.after_tuple-simplifier.before_dce.txt" \
#"module_0806.pjit_train_step.0028.spmd-cleanup.after_dce.before_tuple-simplifier.txt" \
#"module_0806.pjit_train_step.0027.spmd-cleanup.after_pipeline-start.before_dce.txt" \
#"module_0806.pjit_train_step.0036.optimization.after_dot_decomposer.before_stochastic_convert_decomposer.txt" \
#"module_0806.pjit_train_step.0035.optimization.after_CallInliner.before_dot_dimension_sorter.txt" \
#"module_0806.pjit_train_step.0034.optimization.after_rng-bit-generator-expander.before_comparison-expander.txt" \
#"module_0806.pjit_train_step.0038.optimization.after_dynamic_padder.before_simplification.txt" \
#"module_0806.pjit_train_step.0037.optimization.after_dynamic_dimension_simplifier.before_dynamic_padder.txt" \
#"module_0806.pjit_train_step.0039.simplification.after_pipeline-start.before_zero_sized_hlo_elimination.txt" \
#"module_0806.pjit_train_step.0041.simplification.after_bitcast_dtypes_expander.before_dot_dimension_sorter.txt" \
#"module_0806.pjit_train_step.0040.simplification.after_algsimp.before_bitcast_dtypes_expander.txt" \
#"module_0806.pjit_train_step.0045.simplification.after_cse.before_dce.txt" \
#"module_0806.pjit_train_step.0044.simplification.after_transpose-folding.before_cse.txt" \
#"module_0806.pjit_train_step.0043.simplification.after_constant_folding.before_simplify-conditional.txt" \
#"module_0806.pjit_train_step.0042.simplification.after_reshape-mover.before_constant_folding.txt" \
#"module_0806.pjit_train_step.0048.simplification.after_algsimp.before_bitcast_dtypes_expander.txt" \
#"module_0806.pjit_train_step.0047.simplification.after_pipeline-start.before_zero_sized_hlo_elimination.txt" \
#"module_0806.pjit_train_step.0046.simplification.after_dce.before_pipeline-end.txt" \
#"module_0806.pjit_train_step.0052.simplification.after_cse.before_dce.txt" \
#"module_0806.pjit_train_step.0051.simplification.after_constant_folding.before_simplify-conditional.txt" \
#"module_0806.pjit_train_step.0050.simplification.after_reshape-mover.before_constant_folding.txt" \
#"module_0806.pjit_train_step.0049.simplification.after_slice-sinker.before_reshape-mover.txt" \
#"module_0806.pjit_train_step.0055.simplification.after_algsimp.before_bitcast_dtypes_expander.txt" \
#"module_0806.pjit_train_step.0054.simplification.after_pipeline-start.before_zero_sized_hlo_elimination.txt" \
#"module_0806.pjit_train_step.0053.simplification.after_dce.before_pipeline-end.txt" \
#"module_0806.pjit_train_step.0059.simplification.after_pipeline-start.before_zero_sized_hlo_elimination.txt" \
#"module_0806.pjit_train_step.0058.simplification.after_cse.before_dce.txt" \
#"module_0806.pjit_train_step.0057.simplification.after_constant_folding.before_simplify-conditional.txt" \
#"module_0806.pjit_train_step.0056.simplification.after_reshape-mover.before_constant_folding.txt" \
#"module_0806.pjit_train_step.0060.simplification.after_algsimp.before_bitcast_dtypes_expander.txt" \
#"module_0806.pjit_train_step.0066.simplification.after_pipeline-start.before_zero_sized_hlo_elimination.txt" \
#"module_0806.pjit_train_step.0065.simplification.after_dce.before_pipeline-end.txt" \
#"module_0806.pjit_train_step.0064.simplification.after_cse.before_dce.txt" \
#"module_0806.pjit_train_step.0063.simplification.after_constant_folding.before_simplify-conditional.txt" \
#"module_0806.pjit_train_step.0062.simplification.after_reshape-mover.before_constant_folding.txt" \
#"module_0806.pjit_train_step.0061.simplification.after_slice-sinker.before_reshape-mover.txt" \
#"module_0806.pjit_train_step.0068.simplification.after_reshape-mover.before_constant_folding.txt" \
#"module_0806.pjit_train_step.0067.simplification.after_algsimp.before_bitcast_dtypes_expander.txt" \
#"module_0806.pjit_train_step.0070.simplification.after_algsimp.before_bitcast_dtypes_expander.txt" \
#"module_0806.pjit_train_step.0069.simplification.after_pipeline-start.before_zero_sized_hlo_elimination.txt" \
#"module_0806.pjit_train_step.0071.simplification.after_pipeline-start.before_zero_sized_hlo_elimination.txt" \
#"module_0806.pjit_train_step.0072.simplification.after_algsimp.before_bitcast_dtypes_expander.txt" \
#"module_0806.pjit_train_step.0073.simplification.after_pipeline-start.before_zero_sized_hlo_elimination.txt" \
#"module_0806.pjit_train_step.0077.simplification-2.after_pipeline-start.before_convert-mover.txt" \
#"module_0806.pjit_train_step.0076.simplification-2.after_convert-mover.before_algsimp.txt" \
#"module_0806.pjit_train_step.0075.simplification-2.after_pipeline-start.before_convert-mover.txt" \
#"module_0806.pjit_train_step.0074.optimization.after_simplification.before_simplification-2.txt" \
#"module_0806.pjit_train_step.0080.collective-optimizations.after_pipeline-start.before_all-reduce-folder.txt" \
#"module_0806.pjit_train_step.0079.optimization.after_computation-deduplicator.before_pipeline-end.txt" \
#"module_0806.pjit_train_step.0078.optimization.after_simplification-2.before_while-loop-trip-count-annotator.txt" \
#"module_0806.pjit_train_step.0082.conv_canonicalization.after_gpu-conv-rewriter.before_cudnn-fused-convolution-rewriter.txt" \
#"module_0806.pjit_train_step.0081.conv_canonicalization.after_pipeline-start.before_float-normalization-bf16.txt" \
#"module_0806.pjit_train_step.0083.conv_canonicalization.after_cudnn-fused-convolution-rewriter.before_gpu-conv-padding-legalization.txt" \
#"module_0806.pjit_train_step.0085.reshape_mover_after_conv_canonicalization.after_pipeline-start.before_reshape-mover.txt" \
#"module_0806.pjit_train_step.0084.conv_canonicalization.after_CallInliner.before_tuple-simplifier.txt" \
#"module_0806.pjit_train_step.0088.reshape_mover_after_conv_canonicalization.after_algsimp.before_pipeline-end.txt" \
#"module_0806.pjit_train_step.0087.reshape_mover_after_conv_canonicalization.after_pipeline-start.before_reshape-mover.txt" \
#"module_0806.pjit_train_step.0086.reshape_mover_after_conv_canonicalization.after_algsimp.before_pipeline-end.txt" \
#"module_0806.pjit_train_step.0093.simplify_after_conv_canonicalization.after_pipeline-start.before_convert-mover.txt" \
#"module_0806.pjit_train_step.0092.simplify_after_conv_canonicalization.after_convert-mover.before_algsimp.txt" \
#"module_0806.pjit_train_step.0091.simplify_after_conv_canonicalization.after_pipeline-start.before_convert-mover.txt" \
#"module_0806.pjit_train_step.0090.conv_canonicalization.after_reshape_mover_after_conv_canonicalization.before_simplify_after_conv_canonicalization.txt" \
#"module_0806.pjit_train_step.0089.reshape_mover_after_conv_canonicalization.after_pipeline-start.before_reshape-mover.txt" \
#"module_0806.pjit_train_step.0096.layout_assignment.after_flatten-call-graph.before_layout-assignment.txt" \
#"module_0806.pjit_train_step.0095.layout_assignment.after_pipeline-start.before_flatten-call-graph.txt" \
#"module_0806.pjit_train_step.0094.conv_canonicalization.after_simplify_after_conv_canonicalization.before_constant_folding.txt" \
#"module_0806.pjit_train_step.0099.nvptx_post-layout_assignment_part_1.after_cublas-pad-for-gemms.before_cublas-pad-for-gemms.txt" \
#"module_0806.pjit_train_step.0098.nvptx_post-layout_assignment_part_1.after_pipeline-start.before_cublas-pad-for-gemms.txt" \
#"module_0806.pjit_train_step.0097.layout_assignment.after_layout-assignment.before_pipeline-end.txt" \
#"module_0806.pjit_train_step.0103.hlo_normalization.after_reshape-decomposer.before_reduce-decomposer.txt" \
#"module_0806.pjit_train_step.0102.hlo_normalization.after_algsimp.before_transpose-folding.txt" \
#"module_0806.pjit_train_step.0101.hlo_normalization.after_dot_dimension_merger.before_algsimp.txt" \
#"module_0806.pjit_train_step.0100.hlo_normalization.after_pipeline-start.before_dot_dimension_merger.txt" \
#"module_0806.pjit_train_step.0105.hlo_normalization.after_cublas-gemm-rewriter.before_cublas-gemm-broadcast-folding-rewriter.txt" \
#"module_0806.pjit_train_step.0104.hlo_normalization.after_move_copy_to_users.before_triton-gemm-rewriter.txt" \
#"module_0806.pjit_train_step.0106.hlo_normalization.after_layout_normalization.before_algsimp.txt" \
#"module_0806.pjit_train_step.0107.hlo_normalization.after_algsimp.before_broadcast_canonicalizer.txt" \
#"module_0806.pjit_train_step.0110.hlo_normalization.after_reduction-dimension-grouper.before_reduction-splitter.txt" \
#"module_0806.pjit_train_step.0109.hlo_normalization.after_reduction-layout-normalizer.before_reduction-dimension-grouper.txt" \
#"module_0806.pjit_train_step.0108.hlo_normalization.after_reduction-degenerate-dim-remover.before_reduction-layout-normalizer.txt" \
#"module_0806.pjit_train_step.0113.post-layout_assignment.after_pipeline-start.before_collectives-schedule-linearizer.txt" \
#"module_0806.pjit_train_step.0112.hlo_normalization.after_gpu-tree-reduction-rewriter.before_pipeline-end.txt" \
#"module_0806.pjit_train_step.0111.hlo_normalization.after_reduction-splitter.before_gpu-tree-reduction-rewriter.txt" \
#"module_0806.pjit_train_step.0114.post-layout_assignment.after_collectives-schedule-linearizer.before_gpu-conv-algorithm-picker.txt" \
#"module_0806.pjit_train_step.0115.post-layout_assignment.after_gpu-conv-algorithm-picker.before_gemm-algorithm-picker.txt" \
#"module_0806.pjit_train_step.0116.post-layout_assignment.after_float-normalization-bf16.before_float-normalization-f8e5m2.txt" \
#"module_0806.pjit_train_step.0117.post-layout_assignment.after_simplify-fp-conversions.before_tuple-simplifier.txt" \
#"module_0806.pjit_train_step.0121.fusion.after_pipeline-start.before_variadic-op-splitter.txt" \
#"module_0806.pjit_train_step.0120.nvptx_post-layout_assignment_part_2.after_pipeline-start.before_triangular-solve-rewriter.txt" \
#"module_0806.pjit_train_step.0119.post-layout_assignment.after_cse.before_pipeline-end.txt" \
#"module_0806.pjit_train_step.0118.post-layout_assignment.after_algsimp.before_cse.txt" \
#"module_0806.pjit_train_step.0122.fusion.after_fusion.before_fusion.txt" \
#"module_0806.pjit_train_step.0123.fusion.after_fusion.before_fusion_merger.txt" \
#"module_0806.pjit_train_step.0124.fusion.after_fusion_merger.before_multi_output_fusion.txt" \
#"module_0806.pjit_train_step.0125.fusion.after_multi_output_fusion.before_cse.txt" \
#"module_0806.pjit_train_step.0128.fusion.after_pipeline-start.before_variadic-op-splitter.txt" \
#"module_0806.pjit_train_step.0127.fusion.after_dce.before_pipeline-end.txt" \
#"module_0806.pjit_train_step.0126.fusion.after_cse.before_dce.txt" \
#"module_0806.pjit_train_step.0129.fusion.after_fusion.before_fusion_merger.txt" \
#"module_0806.pjit_train_step.0131.fusion.after_cse.before_dce.txt" \
#"module_0806.pjit_train_step.0130.fusion.after_multi_output_fusion.before_cse.txt" \
#"module_0806.pjit_train_step.0132.fusion.after_pipeline-start.before_variadic-op-splitter.txt" \
#"module_0806.pjit_train_step.0134.horizontal_fusion.after_gpu_horizontal_loop_fusion.before_gpu_horizontal_input_fusion.txt" \
#"module_0806.pjit_train_step.0133.horizontal_fusion.after_pipeline-start.before_gpu_horizontal_loop_fusion.txt" \
#"module_0806.pjit_train_step.0136.horizontal_fusion.after_cse.before_dce.txt" \
#"module_0806.pjit_train_step.0135.horizontal_fusion.after_gpu_horizontal_input_fusion.before_cse.txt" \
#"module_0806.pjit_train_step.0139.horizontal_fusion.after_gpu_horizontal_input_fusion.before_cse.txt" \
#"module_0806.pjit_train_step.0138.horizontal_fusion.after_pipeline-start.before_gpu_horizontal_loop_fusion.txt" \
#"module_0806.pjit_train_step.0137.horizontal_fusion.after_dce.before_pipeline-end.txt" \
#"module_0806.pjit_train_step.0141.horizontal_fusion.after_pipeline-start.before_gpu_horizontal_loop_fusion.txt" \
#"module_0806.pjit_train_step.0140.horizontal_fusion.after_cse.before_dce.txt" \
#"module_0806.pjit_train_step.0142.post-fusion_optimization.after_pipeline-start.before_all-gather-combiner.txt" \
#"module_0806.pjit_train_step.0145.post-fusion_optimization.after_algsimp.before_computation-deduplicator.txt" \
#"module_0806.pjit_train_step.0144.post-fusion_optimization.after_gpu-async-collective-annotator.before_algsimp.txt" \
#"module_0806.pjit_train_step.0143.post-fusion_optimization.after_async-collective-creator.before_gpu-async-collective-annotator.txt" \
#"module_0806.pjit_train_step.0147.GPU-ir-emit-prepare.after_pipeline-start.before_dce.txt" \
#"module_0806.pjit_train_step.0146.post-fusion_optimization.after_computation-deduplicator.before_pipeline-end.txt" \
#"module_0806.pjit_train_step.0148.copy-insertion.after_adding_copies_to_resolve_interference.txt" \
#"module_0806.pjit_train_step.0149.copy-insertion.after_removing_unnecessary_copies.txt" \
#"module_0806.pjit_train_step.0152.horizontal-loop-fusion-for-copy.after_pipeline-start.before_gpu_horizontal_loop_fusion.txt" \
#"module_0806.pjit_train_step.0151.GPU-ir-emit-prepare.after_copy-insertion.before_horizontal-loop-fusion-for-copy.txt" \
#"module_0806.pjit_train_step.0150.copy-insertion.after_adding_special-case_copies.txt" \
#"module_0806.pjit_train_step.0155.GPU-ir-emit-prepare.after_sanitize-constant-names.before_pipeline-end.txt" \
#"module_0806.pjit_train_step.0154.GPU-ir-emit-prepare.after_horizontal-loop-fusion-for-copy.before_sanitize-constant-names.txt" \
#"module_0806.pjit_train_step.0153.horizontal-loop-fusion-for-copy.after_gpu_horizontal_loop_fusion.before_dce.txt" \
#"module_0806.pjit_train_step.0157.post-scheduling-passes.after_gpu-convert-async-collectives-to-sync.before_cse_barrier_expander.txt" \
#"module_0806.pjit_train_step.0156.post-scheduling-passes.after_pipeline-start.before_gpu-convert-async-collectives-to-sync.txt" \
#"module_0806.pjit_train_step.sm_8.0_gpu_after_optimizations.txt" \
#) 

declare -a testArray=( \
"module_0806.pjit_train_step.sm_8.0_gpu_after_optimizations.txt" \
) 


# for item in "${testArray[@]}"
# do
#  echo $item
#  ./bazel-5.3.0-linux-x86_64 build \
#    --verbose_failures \
#    -c opt \
#    --config=cuda \
#    --nocheck_visibility \
#    --copt="-Wno-error=switch" \
#    --copt="-DLLVM_ENABLE_DUMP=ON" \
#    --action_env TF_CUDA_COMPUTE_CAPABILITIES=compute_80 \
#    --compilation_mode=dbg \
#    //xla/tools:run_hlo_module 
# 
# 
#    bazel-bin/xla/tools/run_hlo_module \
#    --iterations=1 \
#    --input_format=hlo \
#    --reference_platform="" \
#    --random_init_input_literals \
#    --platform=gpu \
#    --input_module="/home/scratch.shawnw_inf/git/github/openxla/xla/xla_dump_vit/"$item \
#    |& tee xla_dump_buffer_assignment/$item 
# done

./bazel-5.3.0-linux-x86_64 build \
--verbose_failures \
-c opt \
--config=cuda \
--nocheck_visibility \
--copt="-Wno-error=switch" \
--copt="-DLLVM_ENABLE_DUMP=ON" \
--copt=-g0 \
--action_env TF_CUDA_COMPUTE_CAPABILITIES=compute_80 \
--compilation_mode=dbg \
//xla/tools:run_hlo_module 

gdb --args \
bazel-bin/xla/tools/run_hlo_module \
--iterations=1 \
--input_format=hlo \
--reference_platform="" \
--random_init_input_literals \
--platform=gpu \
--input_module="/home/scratch.shawnw_inf/git/github/openxla/xla/xla_dump_vit/module_0806.pjit_train_step.sm_8.0_gpu_after_optimizations.txt" 


#./bazel-5.3.0-linux-x86_64 build \
#--verbose_failures \
#-c opt \
#--config=cuda \
#--nocheck_visibility \
#--copt="-Wno-error=switch" \
#--copt="-DLLVM_ENABLE_DUMP=ON" \
#--action_env TF_CUDA_COMPUTE_CAPABILITIES=compute_80 \
#//xla/tools:run_hlo_module 
#
#bazel-bin/xla/tools/run_hlo_module \
#--iterations=1 \
#--input_format=hlo \
#--reference_platform="" \
#--random_init_input_literals \
#--platform=gpu \
#--input_module="/home/scratch.shawnw_inf/git/github/openxla/xla/xla_dump_vit/module_0806.pjit_train_step.sm_8.0_gpu_after_optimizations.txt" 
 

 
#./bazel-5.3.0-linux-x86_64 run \
#   --verbose_failures \
#   -c opt \
#   --config=cuda \
#   --nocheck_visibility \
#   --copt="-Wno-error=switch" \
#   --copt="-DLLVM_ENABLE_DUMP=ON" \
#   --action_env TF_CUDA_COMPUTE_CAPABILITIES=compute_80 \
#   //xla/tools:run_hlo_module \
#   -- \
#   --iterations=1 \
#   --input_format=hlo \
#   --reference_platform="" \
#   --random_init_input_literals \
#   --platform=gpu \
#   --input_module=/home/scratch.shawnw_inf/git/github/openxla/xla/xla_dump_vit/module_0806.pjit_train_step.sm_8.0_gpu_after_optimizations.txt


#   --input_module=/home/scratch.shawnw_inf/git/github/openxla/xla/tests/add
#OUTPUT="fusion.${precision}.$(hostname -s)"                        
#PROF="ncu -o ${OUTPUT} --set full -s 6 -c 2 --cache-control all"   
#PROF+=" --clock-control none -k 'regex:^fusion_1116$|^fusion_357$'"
#${PROF} python3 bench_toy_model.py ...

#NCU_CMD="ncu -k fusion -o ncu_bf16 --set full -s 6 -c 2 --cache-control all --clock-control none --target-processes all"
#NSYS_CMD="nsys profile -o nsys_bf16 -s none --force-overwrite true --stats=true"
#export XLA_FLAGS='--xla_disable_hlo_passes=layout-assignment --xla_gpu_simplify_all_fp_conversions --xla_dump_hlo_as_html --xla_dump_hlo_as_proto --xla_dump_hlo_as_text --xla_dump_to=xla_dump_1360 --xla_dump_hlo_pass_re=.*'
#
#${NCU_CMD} \
#bazel-bin/xla/tools/run_hlo_module \
#   --iterations=10 \
#   --input_format=hlo \
#   --reference_platform="" \
#   --random_init_input_literals \
#   --platform=gpu \
#   --input_module=/home/scratch.shawnw_inf/git/github/openxla/xla/tests/add
#
#NSYS_CMD="nsys profile -o nsys_fusion101 -s none --force-overwrite true --stats=true"
#
#NCU_CMD="ncu -o ncu_fusion101 --set full -s 6 -c 2 --cache-control all --clock-control none -k fusion --target-processes all"
#
#
#${NCU_CMD} \
#bazel-bin/xla/tools/run_hlo_module \
#   --iterations=10 \
#   --input_format=hlo \
#   --reference_platform="" \
#   --random_init_input_literals \
#   --platform=gpu \
#   --input_module=/home/scratch.shawnw_inf/git/github/openxla/xla/tests/fusion.101

#bazel test --nocheck_visibility xla/mlir/backends/gpu/transforms/tests/outline_cuda_graphs.mlir.test
