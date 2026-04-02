#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_tier_h100.sh <tier1|tier2>

Examples:
  ./run_tier_h100.sh tier1
  ./run_tier_h100.sh tier2
EOF
}

TIER="${1:-}"

############################################################################
#
# We are interested in finding the problem size at which each kernel
# saturates the compute resources on the NVIDIA H100 architecture. We do this
# by running a sequence of problem sizes for each kernel such that the 
# saturation point is evident on the associated throughput curve for each
# kernel.
#
# We run with 1 MPI rank per GPU on a node. We choose the smallest
# problem to use ~50,000 bytes of allocated memory and the largest problem
# to use ~150MB of allocated memory, which is about 3 times the L2-cache
# size on the H100. The L2-cache is 50 MB (50 * 1024 * 1024 = 52428800 bytes).
#
# IMPORTANT NOTE: Tier1 kernels, FEMSWEEP and MASS3DEA, are run over
#                 different problem size ranges than what's described above.
#                 These kernels do not have clear saturation points.
#
# IMPORTANT NOTES: Tier2 kernels, INDEXLIST_3LOOP and HALO_PACKING_FUSED,
#                  do not perform any floating point operations. So we
#                  recommend looking at bandwidth plots for those. Also,
#                  they are run over different problem size ranges than
#                  what's described above to better expose their bandwidth
#                  behavior.
#
############################################################################

BASE_OUTDIR="RPBenchmark_H100"
BASEMEM=50000
ALLOC_ARGS="-N1 --exclusive -t 45"
RUN_ARGS="-N1 -n4"

case "${TIER,,}" in
  tier1|tier2)
    ;;
  ""|-h|--help)
    usage
    exit 0
    ;;
  *)
    echo "Error: unknown kernel set '${TIER}'. Must be 'tier1' or 'tier2'."
    usage
    exit 2
    ;;
esac

OUTDIR="${BASE_OUTDIR}_${TIER}"

if [[ ! -x ./bin/raja-perf.exe ]]; then
  echo "Error: ./bin/raja-perf.exe not found or not executable."
  exit 1
fi

export OUTDIR BASEMEM RUN_ARGS TIER

salloc ${ALLOC_ARGS} bash -lc '
  set -euo pipefail

  case "${TIER,,}" in
    tier1)
       KERNELS=("DIFFUSION3DPA"
                "EDGE3D"
                "ENERGY"
                "INTSC_HEXRECT"
                "MASS3DPA_ATOMIC"
                "MASSVEC3DPA"
                "NODAL_ACCUMULATION_3D"
                "VOL3D")
       KERNELS_DIFFRANGE=("FEMSWEEP"
                          "MASS3DEA")
     ;;
    tier2)
       KERNELS=("CONVECTION3DPA"
                "DEL_DOT_VEC_2D"
                "INTSC_HEXHEX"
                "LTIMES"
                "MASS3DPA"
                "MATVEC_3D_STENCIL"
                "MULTI_REDUCE"
                "REDUCE_STRUCT")
       KERNELS_DIFFRANGE=("INDEXLIST_3LOOP"
                          "HALO_PACKING_FUSED")
      ;;
    *)
      echo "Error: unknown kernel set: ${TIER}"
      exit 2
      ;;
  esac

  FACTORS=(1 4 16 32 64 128 256 512 1024 1500 3000)
  FACTORS_DIFFRANGE=(32 64 128 256 512 1024 1500 3000 4000 5000 6000)

  for KERNEL_NAME in "${KERNELS[@]}"; do
    echo "Running kernel: ${KERNEL_NAME}"
    for factor in "${FACTORS[@]}"; do
      mem=$(( factor * BASEMEM ))
      echo "  Running with memory = ${mem}"

      srun ${RUN_ARGS} ./bin/raja-perf.exe \
        -k "${KERNEL_NAME}" \
        --npasses 1 \
        --npasses-combiners Average Minimum Maximum \
        --outdir "${OUTDIR}" \
        --outfile "${KERNEL_NAME}_factor_${factor}" \
        --memory-allocated "${mem}" \
        --warmup-perfrun-same \
        -ev RAJA_Seq Lambda
    done
  done

  if [[ -n "${KERNELS_DIFFRANGE:-}" ]]; then

    for KERNEL_NAME in "${KERNELS_DIFFRANGE[@]}"; do
      echo "Running kernel: ${KERNEL_NAME}"
      for factor in "${FACTORS_DIFFRANGE[@]}"; do
        mem=$(( factor * BASEMEM ))
        echo "  Running with memory = ${mem}"

        srun ${RUN_ARGS} ./bin/raja-perf.exe \
          -k "${KERNEL_NAME}" \
          --npasses 1 \
          --npasses-combiners Average Minimum Maximum \
          --outdir "${OUTDIR}" \
          --outfile "${KERNEL_NAME}_factor_${factor}" \
          --memory-allocated "${mem}" \
          --warmup-perfrun-same \
          -ev RAJA_Seq Lambda
      done
    done

  fi
'
