This directory contains performance data and scripts to generate and process
that data for RAJA Perf FCR 2026 benchmarking activities.

The results include throughput studies for: 

  * AMD MI300A
  * NVIDIA H100

The summarized metrics for each kernel include:

  * Problem size at the saturation (iteration space size)
  * Compute rate (GFLOP/s) at the saturation point
  * Memory bandwidth rate (GB/s) at the saturation point


Details related to code compilation, execution and results can be found here:

https://software.llnl.gov/benchmarks/13_rajaperf/rajaperf.html

In particular:

   * Performance data was generated for two subsets of kernels in the RAJA
     Performance Suite, called Tier 1 and Tier 2. Each Tier contains ten
     kernels.
   * For MI300A architecture:
       * The code was built using version 9.0.1 of the Cray MPICH MPI library
         and the AMD clang compiler with ROCm version 6.4.3 targeting GPU
         compute architecture gfx942.
       * The code was run on a single compute node containing 4 MI300A APUs
         in SPX mode and CPX mode using the script 'run_tier_mi300a.py'
         contained in the scripts directory.
           * SPX node used 4 MPI ranks (1 per APU) 
           * CPX mode used 24 MPI ranks (6 per APU or 1 per XCD)
   * For H100 architecture:
       * The code was built using version 2.3.7 of the MVAPICH2 MPI library,
         version 12.9.1 of the nvcc compiler for CUDA targeting GPU compute
         architecture sm_90, and version 10.3.1 of the GNU compiler for
         compiling host code.
       * The code was run on a single compute node containing 4 H100 GPUs
         using the script 'run_tier_h100.py' contained in the scripts
         directory, which runs the code using 4 MPI ranks (1 per GPU).



