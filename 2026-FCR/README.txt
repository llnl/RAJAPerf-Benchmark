This directory contains performance data and scripts to generate and process
that data for RAJA Perf FCR 2026 benchmarking activities.

The results include throughput studies for: 

  * AMD MI300A
  * NVIDIA H100

Performance data is generated for two subsets of kernels, Tier 1 and Tier 2,
that are described here:

https://software.llnl.gov/benchmarks/13_rajaperf/rajaperf.html

The summarized metrics for each kernel include:

  * Problem size at the saturation (iteration space size)
  * Compute rate (GFLOP/s) at the saturation point
  * Memory bandwidth rate (GB/s) at the saturation point
