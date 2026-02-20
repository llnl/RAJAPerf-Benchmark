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

case "${MODE,,}" in
  spx)
    BASE_OUTDIR="RPBenchmark"
    BASEMEM=134217728
    ALLOC_ARGS="-xN1 -t 45"
    RUN_ARGS="-xN1 -n4"
    ;;
  cpx)
    BASE_OUTDIR="RPBenchmark"
    BASEMEM=22369621
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

export OUTDIR BASEMEM RUN_ARGS TIER

flux alloc ${ALLOC_ARGS} bash -lc '
  set -euo pipefail

  FACTORS=(1 2 4 8 10 12 16 20)

  case "${TIER,,}" in
    tier1)
      KERNELS=("DIFFUSION3DPA"
               "EDGE3D"
               "ENERGY"
               "INTSC_HEXRECT"
               "MASS3DEA"
               "MASS3DPA_ATOMIC"
               "MASSVEC3DPA"
               "MATVEC_3D_STENCIL"
               "NODAL_ACCUMULATION_3D"
               "VOL3D")
      ;;
    tier2)
      KERNELS=("CONVECTION3DPA"
               "DEL_DOT_VEC_2D"
               "INTSC_HEXHEX"
               "LTIMES"
               "MASS3DPA"
               "MULTI_REDUCEC"
               "REDUCE_STRUCT"
               "INDEXLIST_3LOOP"
               "HALO_PACKING_FUSED")
      ;;
    *)
      echo "Error: unknown kernel set: ${TIER}"
      exit 2
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
        -ev Seq Lambda
    done
  done
'
