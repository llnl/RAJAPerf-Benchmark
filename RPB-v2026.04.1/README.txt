This directory contains performance data and scripts to generate and process
that data for the RAJA Performance Suite benchmarking exercise corresponding to
version v2026.04.1 of this repo. The scripts and data are contained in the
`scripts` and `data` subdirectories of this directory, respectively.

A detailed discussion of the kernels and results can be found here:
https://software.llnl.gov/benchmarks/13_rajaperf/rajaperf.html

However, the information there may not match what is described below due to
the independent nature of the LLNL Benchmarks project and this repo.

The results here include throughput studies for two GPU architectures:

  * AMD MI300A
  * NVIDIA H100

The benchmark uses two subsets of kernels in the RAJA Performance Suite called
Tier 1 and Tier 2. Each Tier contains ten kernels. The summarized metrics for
each kernel include:

  * Problem size at the saturation (iteration space size)
  * Compute rate (GFLOP/s) at the saturation point
  * Memory bandwidth rate (GB/s) at the saturation point

Code version, compilation, execution:
 
   * Performance data was generated using the RAJA Performance Suite v2025.12.1.
     To make sure you have the correct version of this benchmark repo and the
     corresponding version of the RAJAPerf code:

     $ cd RAJAPerf-Benchmark   (top-level of this repo)
     $ git pull
     $ git checkout v2026.04.1
     $ git submodule update --init --recursive

   * For MI300A architecture:

       * The code was built using version 9.0.1 of the Cray MPICH MPI library
         and the AMD clang compiler with ROCm version 6.4.3 targeting GPU
         compute architecture gfx942. Specifically,

           $ cd path/to/RAJAPerf
           $ ./scripts/lc-builds/toss4_cray-mpich_amdclang.sh 9.0.1 6.4.3 gfx942
           $ cd build_lc_toss4-cray-mpich-9.0.1-amdclang-6.4.3-gfx942
           $ make -j

       * The code was run on a single compute node containing 4 MI300A APUs
         in SPX mode and CPX mode using the script 'run_tier_mi300a.sh'
         contained in the scripts directory.

           * SPX node used 4 MPI ranks (1 per APU)
              
               $ cd path/to/RAJAPerf
               $ cd build_lc_toss4-cray-mpich-9.0.1-amdclang-6.4.3-gfx942
               $ ./run_tier_mi300a.sh spx tier1/tier2

           * CPX mode used 24 MPI ranks (6 per APU or 1 per XCD)

               $ cd path/to/RAJAPerf
               $ cd build_lc_toss4-cray-mpich-9.0.1-amdclang-6.4.3-gfx942
               $ ./run_tier_mi300a.sh cpx tier1/tier2

   * For H100 architecture:

       * The code was built using version 2.3.7 of the MVAPICH2 MPI library,
         version 12.9.1 of the nvcc compiler for CUDA targeting GPU compute
         architecture sm_90, and version 10.3.1 of the GNU compiler for
         compiling host code.

           $ cd path/to/RAJAPerf
           $ ./scripts/lc-builds/toss4_mvapich2_nvcc_gcc.sh 2.3.7 12.9.1 90 10.3.1
           $ cd build_lc_toss4-mvapich2-2.3.7-nvcc-12.9.1-90-gcc-10.3.1
           $ make -j

       * The code was run on a single compute node containing 4 H100 GPUs
         using the script 'run_tier_h100.sh' contained in the scripts
         directory, which runs the code using 4 MPI ranks (1 per GPU).

           $ cd path/to/RAJAPerf
           $ cd build_lc_toss4-mvapich2-2.3.7-nvcc-12.9.1-90-gcc-10.3.1
           $ ./run_tier_h100a.sh tier1/tier2 

After completing the runs described above, the raw run data was processed to generate
files containing summary tables and throughput plots. These files are contained in the
`data` sub-directory. For example, the files for the MI300A SPX mode, tier1 run were
generated with the command:

  $ python3 path/to/process_data.py \
    --root-dir path/to/build_lc_toss4-cray-mpich-9.0.1-amdclang-6.4.3-gfx942/RPBenchmark_MI300A_tier1-SPX \
    --output-dir path/to/build_lc_toss4-cray-mpich-9.0.1-amdclang-6.4.3-gfx942/RPBenchmark_MI300A_tier1-SPX/Output \
    --exclude-plot-variant Base_Seq



