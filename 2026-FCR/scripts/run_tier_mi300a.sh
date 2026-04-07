#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_tier_mi300a.sh <spx|cpx> <tier1|tier2>

Examples:
  ./run_tier_mi300a.sh spx tier1
  ./run_tier_mi300a.sh spx tier2
  ./run_tier_mi300a.sh cpx tier1
  ./run_tier_mi300a.sh cpx tier2
EOF
}

MODE="${1:-}"
TIER="${2:-}"

############################################################################
#
# We are interested in finding the problem size at which each kernel
# saturates the compute resources on the AMD MI300A architecture. We do this
# by running a sequence of problem sizes for each kernel such that the 
# saturation point is evident on the associated throughput curve for each
# kernel.
#
# For SPX mode (run with 1 MPI rank per APU on a node), we choose the smallest
# problem to use ~100,000 bytes of allocated memory and the largest problem
# to use ~400MB of allocated memory, which is about 1.5 times MALL size on
# the MI300A. The MALL is 256 MB (256 * 1024 * 1024 = 268435456 bytes).
#
# For CPX mode (run with 6 MPI ranks per APU on a node), we choose the
# smallest problem to use ~50,000 bytes of allocated memory and the largest
# problem to use ~75MB of allocated memory, which is about less than 1/3
# the MALL size.
#
# IMPORTANT NOTES: Tier1 kernels FEMSWEEP and MASS3DEA are run over
#                  different problem size ranges than what's described above.
#                  These kernels do not have clear saturation points.
#
# IMPORTANT NOTES: Tier2 kernels INDEXLIST_3LOOP and HALO_PACKING_FUSED
#                  do not perform any floating point operations. So we
#                  recommend looking at bandwidth plots for those. Also,
#                  they are run over different problem size ranges than
#                  what's described above to better expose their bandwidth
#                  scaling behavior.
#
############################################################################

case "${MODE,,}" in
  spx)
    BASE_OUTDIR="RPBenchmark_MI300A"
    BASEMEM=100000
    ALLOC_ARGS="-xN1 -t 45"
    RUN_ARGS="-xN1 -n4"
    ;;
  cpx)
    BASE_OUTDIR="RPBenchmark_MI300A"
    BASEMEM=50000
    ALLOC_ARGS="-xN1 --amd-gpumode=CPX -t 45"
    RUN_ARGS="-xN1 -n24 -g 1"
    ;;
  ""|-h|--help)
    usage
    exit 0
    ;;
  *)
    echo "Error: unknown mode '${MODE}'. Must be 'spx' or 'cpx'."
    usage
    exit 2
    ;;
esac

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

OUTDIR="${BASE_OUTDIR}_${TIER}-${MODE^^}"

if [[ ! -x ./bin/raja-perf.exe ]]; then
  echo "Error: ./bin/raja-perf.exe not found or not executable."
  exit 1
fi

export OUTDIR BASEMEM RUN_ARGS MODE TIER

flux alloc ${ALLOC_ARGS} bash -lc '
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

  case "${MODE,,}" in
    spx)
        FACTORS=(1 4 16 32 64 128 256 512 1024 1500 2048 3000 4000)
        FACTORS_DIFFRANGE=(32 64 128 256 512 1024 1500 2048 3000 4000 5000 6000)
      ;;
    cpx)
        FACTORS=(1 4 16 32 64 128 256 512 1024 1500)
        FACTORS_DIFFRANGE=(32 64 128 256 512 1024 1500 2048 3000 4000)
      ;;
  esac

  for KERNEL_NAME in "${KERNELS[@]}"; do
    echo "Running kernel: ${KERNEL_NAME}"
    for factor in "${FACTORS[@]}"; do
      mem=$(( factor * BASEMEM ))
      echo "  Running with memory = ${mem}"

      flux run ${RUN_ARGS} ./bin/raja-perf.exe \
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

        flux run ${RUN_ARGS} ./bin/raja-perf.exe \
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
