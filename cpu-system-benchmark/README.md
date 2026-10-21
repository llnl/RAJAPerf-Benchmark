# CPU system benchmark launchers

These scripts run the RAJA Performance Suite Tier 1 or Tier 2 throughput study
across all user-accessible CPU cores on one node. Each core receives one MPI
rank. The runs compare the `Base_Seq` and `RAJA_Seq` variants.

The scripts allocate a node before running the benchmark. Run them from a
RAJAPerf build directory containing an executable `./bin/raja-perf.exe` built
with MPI support.

## Run a study

Pass `tier1` or `tier2` to the frontend for the target system:

```bash
/path/to/cpu-system-benchmark/run_tier_ats3.sh tier1
/path/to/cpu-system-benchmark/run_tier_ats4.sh tier1
/path/to/cpu-system-benchmark/run_tier_cts1.sh tier1
/path/to/cpu-system-benchmark/run_tier_cts2.sh tier1
/path/to/cpu-system-benchmark/run_tier_ats5ea.sh tier1
```

Each frontend selects the scheduler and full-node rank count for its system.
The launchers rely on automatic CPU binding and do not set OpenMP environment
variables.

## Cache-scaled problem sizes

Each frontend calculates cache bytes per core for its system. The shared driver
uses that value to extend the problem-size sweep to approximately 32 times the
per-rank cache share. Kernels that consistently reach their large-problem
throughput plateau earlier (`EDGE3D`, `FEMSWEEP`, `INTSC_HEXRECT`, and
`INTSC_HEXHEX`) stop at approximately six times the cache share.
`HALO_PACKING_FUSED` also uses the shorter range because its work is not
proportional to problem size. See the frontend scripts for cache and topology
details.

## Outputs and interpretation

Output directories include the system and tier, for example:

```text
RPBenchmark_CPU_CTS2_SPR_tier1
```

RAJAPerf reports the problem size and performance of one rank. These results
therefore show per-rank performance while every user-accessible core is
simultaneously contending for node resources. They are not aggregate node
GFLOP/s or bandwidth values.

## Analyze the results

Run `process_data.py` separately for each system and tier:

```bash
python3 /path/to/RPB-v2026.10.0/scripts/process_data.py \
  --root-dir /path/to/build/RPBenchmark_CPU_CTS2_SPR_tier1 \
  --output-dir /path/to/build/RPBenchmark_CPU_CTS2_SPR_tier1/Output
```

## ATS-5-EA notes

Use the following compiler versions for ATS-5-EA builds:

- NVCC 13.4
- Clang 23
- GCC 15
- Clang 26 when building with NVHPC

ATS-5-EA allocations may need site- or experiment-specific `salloc` options.
Place these options after a `--` delimiter. The frontend passes each option
unchanged to `salloc`:

```bash
/path/to/cpu-system-benchmark/run_tier_ats5ea.sh tier1 -- \
  --partition=PARTITION --reservation=RESERVATION
```

Replace `PARTITION` and `RESERVATION` with the values assigned for the run.
These options affect the allocation only; they are not forwarded to `srun`.
