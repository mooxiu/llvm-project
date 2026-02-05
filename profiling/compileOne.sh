#!/bin/bash

projectRoot="/home/muyao/projects/llvm-project"
testcaseOutDir="${projectRoot}/profiling/out"

file=${1}
echo "Compile ${file}"
filename="$(echo "$(basename "${file}")" | sed "s/\..*//g")"
echo "Filename ${filename}"

# Compile with normal openmp workdistribute
echo "Compiling normal test cases"

${projectRoot}/build/bin/flang -O3 -fopenmp -fopenmp-version=60 \
  -fopenmp-targets=nvptx64-nvidia-cuda --offload-arch=sm_80 \
  -I${projectRoot}/build/runtimes/runtimes-bins/openmp/runtime/src \
  -L${projectRoot}/build/runtimes/runtimes-bins/openmp/runtime/src \
  -L${projectRoot}/build/lib \
  -lomptarget \
  ${file} -o "${testcaseOutDir}/${filename}-omp.out" 

# Compile with XLA
echo "Compiling XLA test cases"

${projectRoot}/build/bin/flang -O3 -fopenmp -fopenmp-version=60 \
  -fopenmp-targets=nvptx64-nvidia-cuda --offload-arch=sm_80 \
  -I${projectRoot}/build/runtimes/runtimes-bins/openmp/runtime/src \
  -L${projectRoot}/build/runtimes/runtimes-bins/openmp/runtime/src \
  -L${projectRoot}/build/lib \
  -lomptarget \
  ${file} -o "${testcaseOutDir}/${filename}-xla.out" -mmlir --jit-workdistribute 

