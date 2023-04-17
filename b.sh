export TEST_TMPDIR=/tmp/bazel_cache
set -x 

sudo nvidia-smi -lgc 1200,1200; sudo nvidia-smi -lmc 1000,1000

#export XLA_FLAGS=--xla_disable_all_hlo_passes
#export XLA_FLAGS="--xla_disable_hlo_passes=layout-assignment"

export TF_CPP_MIN_LOG_LEVEL=0
export TF_CPP_MAX_LOG_LEVEL=5
export TF_CPP_VMODULE=ir_emitter=4,ir_emitter_unnested=2,gpu_compiler=2,elemental_ir_emitter=2,hlo_to_ir_bindings=2,gpu_compiler=2,compile_module_to_llvm_ir=4,ir_builder_mixin=4

#export TF_CPP_VMODULE=buffer_assignment=4,copy_insertion=4
#export TF_CPP_VMODULE=copy_insertion=4
#export XLA_FLAGS="--xla_gpu_simplify_all_fp_conversions --xla_dump_hlo_as_text  --xla_dump_to=xla_dump "
export XLA_FLAGS='--xla_disable_hlo_passes=layout-assignment --xla_gpu_simplify_all_fp_conversions --xla_dump_hlo_as_html --xla_dump_hlo_as_proto --xla_dump_hlo_as_text --xla_dump_to=xla_dump_add_bf16 --xla_dump_hlo_pass_re=.*'

#./bazel-5.3.0-linux-x86_64 run -c opt --config=cuda --action_env TF_CUDA_COMPUTE_CAPABILITIES=compute_80 --nocheck_visibility --copt="-Wno-error=switch" xla/service/copy_insertion_test  --test_filter=WhileCopyInsertionTest.DependentTupleElements
#./bazel-5.3.0-linux-x86_64 run -c opt --config=cuda --action_env TF_CUDA_COMPUTE_CAPABILITIES=compute_80 --nocheck_visibility --copt="-Wno-error=switch" xla/service/float_normalization_test 

./bazel-5.3.0-linux-x86_64 run \
   --verbose_failures \
   -c opt \
   --config=cuda \
   --nocheck_visibility \
   --copt="-Wno-error=switch" \
   --copt="-DLLVM_ENABLE_DUMP=ON" \
   --action_env TF_CUDA_COMPUTE_CAPABILITIES=compute_80 \
   //xla/tools:run_hlo_module \
   -- \
   --input_format=hlo \
   --reference_platform="" \
   --random_init_input_literals \
   --platform=gpu \
   --input_module=/home/scratch.shawnw_inf/git/github/openxla/xla/tests/add

#OUTPUT="fusion.${precision}.$(hostname -s)"                        
#PROF="ncu -o ${OUTPUT} --set full -s 6 -c 2 --cache-control all"   
#PROF+=" --clock-control none -k 'regex:^fusion_1116$|^fusion_357$'"
#${PROF} python3 bench_toy_model.py ...

#NCU_CMD="ncu -k fusion -o ncu_bf16 --set full -s 6 -c 2 --cache-control all --clock-control none --target-processes all"
#NSYS_CMD="nsys profile -o nsys_bf16 -s none --force-overwrite true --stats=true"
#export XLA_FLAGS='--xla_disable_hlo_passes=layout-assignment --xla_gpu_simplify_all_fp_conversions --xla_dump_hlo_as_html --xla_dump_hlo_as_proto --xla_dump_hlo_as_text --xla_dump_to=xla_dump_1360 --xla_dump_hlo_pass_re=.*'
#
#${NCU_CMD} \
bazel-bin/xla/tools/run_hlo_module \
   --iterations=10 \
   --input_format=hlo \
   --reference_platform="" \
   --random_init_input_literals \
   --platform=gpu \
   --input_module=/home/scratch.shawnw_inf/git/github/openxla/xla/tests/add
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
