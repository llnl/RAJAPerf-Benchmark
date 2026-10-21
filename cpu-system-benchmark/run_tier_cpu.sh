#!/usr/bin/env bash
# The allocation body is intentionally single-quoted to match the existing
# launchers; ShellCheck otherwise parses its continuations in the outer shell.
# RUN_CMD and RUN_ARGS are selected internally and intentionally word-split.
# shellcheck disable=SC1004,SC2016,SC2086
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  Run this driver through a system frontend:

    run_tier_ats3.sh <tier1|tier2> [-- allocation-options...]
    run_tier_ats4.sh <tier1|tier2> [-- allocation-options...]
    run_tier_cts1.sh <tier1|tier2> [-- allocation-options...]
    run_tier_cts2.sh <tier1|tier2> [-- allocation-options...]
    run_tier_ats5ea.sh <tier1|tier2> [-- allocation-options...]

EOF
}

make_factors() {
  local minimum_factor="$1"
  local target_factor="$2"
  local factor
  local last_factor=0
  local -a factors=()
  local -a factor_ladder=(
    1 4 16 32 64 128 256 512 1024 1500 2048 3000 4000 5000 6000
    8000 12000 16000 24000 32000 48000 64000
  )

  for factor in "${factor_ladder[@]}"; do
    if (( factor >= minimum_factor && factor <= target_factor )); then
      factors+=("$factor")
      last_factor="$factor"
    fi
  done
  if (( last_factor != target_factor )); then
    factors+=("$target_factor")
  fi

  printf '%s ' "${factors[@]}"
}

############################################################################
#
# We are interested in finding the large-problem throughput plateau for each
# kernel on the selected architecture. We run enough large problem sizes to
# identify a stable tail on the associated throughput curve.
#
# We run with 1 MPI rank per CPU core on a node. We choose the smallest
# problem to use ~50,000 bytes of allocated memory. Kernels that consistently
# plateau early stop at about 6 times the per-core cache share; the remaining
# kernels continue to about 32 times that share.
#
# IMPORTANT NOTE: Tier1 kernels FEMSWEEP and MASS3DEA are run over
#                 different problem size ranges than what's described above.
#                 These kernels do not have clear saturation points.
#
# IMPORTANT NOTES: Tier2 kernels INDEXLIST_3LOOP, HALO_PACKING_FUSED,
#                  MULTI_REDUCE, and REDUCE_STRUCT are run over different
#                  problem size ranges than what's described above to better
#                  expose their throughput scaling behavior.
#
#                  The Tier2 kernels INDEXLIST_3LOOP and HALO_PACKING_FUSED
#                  do not perform any floating point operations. So we
#                  recommend looking at bandwidth plots for those.
#
############################################################################

: "${CPU_BENCH_SYSTEM:?CPU_BENCH_SYSTEM is not set; use a system frontend.}"
: "${CPU_BENCH_SCHEDULER:?CPU_BENCH_SCHEDULER is not set; use a system frontend.}"
: "${CPU_BENCH_RANKS:?CPU_BENCH_RANKS is not set.}"
: "${CPU_BENCH_CACHE_BYTES_PER_CORE:?CPU_BENCH_CACHE_BYTES_PER_CORE is not set.}"

BASE_OUTDIR="RPBenchmark_CPU_${CPU_BENCH_SYSTEM}"
BASEMEM=50000
SHORT_CACHE_MULTIPLIER=6
LONG_CACHE_MULTIPLIER=32
TIER="${1:-}"

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
shift

RANKS="${CPU_BENCH_RANKS}"

EXTRA_ALLOC_ARGS=()
if (( $# > 0 )); then
  if [[ "$1" != "--" ]]; then
    echo "Error: allocation options must follow a '--' delimiter."
    usage
    exit 2
  fi
  shift
  EXTRA_ALLOC_ARGS=("$@")
fi

CACHE_BYTES_PER_CORE="${CPU_BENCH_CACHE_BYTES_PER_CORE}"

short_target_factor=$((
  (SHORT_CACHE_MULTIPLIER * CACHE_BYTES_PER_CORE + BASEMEM - 1) / BASEMEM
))
long_target_factor=$((
  (LONG_CACHE_MULTIPLIER * CACHE_BYTES_PER_CORE + BASEMEM - 1) / BASEMEM
))
FACTORS_STRING="$(make_factors 1 "$long_target_factor")"
FACTORS_SHORT_STRING="$(make_factors 1 "$short_target_factor")"
FACTORS_DIFFRANGE_STRING="$(make_factors 32 "$long_target_factor")"
FACTORS_DIFFRANGE_SHORT_STRING="$(make_factors 32 "$short_target_factor")"

case "$CPU_BENCH_SCHEDULER" in
  flux)
    ALLOC_CMD=(flux alloc -xN1)
    RUN_CMD="flux run"
    RUN_ARGS="-N1 -n ${RANKS}"
    ;;
  slurm)
    ALLOC_CMD=(salloc -N1 --exclusive)
    RUN_CMD="srun"
    RUN_ARGS="-N1 -n ${RANKS}"
    ;;
  *)
    echo "Error: unsupported scheduler '${CPU_BENCH_SCHEDULER}'."
    exit 2
    ;;
esac

OUTDIR="${BASE_OUTDIR}_${TIER}"

if [[ ! -x ./bin/raja-perf.exe ]]; then
  echo "Error: ./bin/raja-perf.exe not found or not executable."
  exit 1
fi

export OUTDIR BASEMEM RUN_CMD RUN_ARGS TIER
export FACTORS_STRING FACTORS_SHORT_STRING
export FACTORS_DIFFRANGE_STRING FACTORS_DIFFRANGE_SHORT_STRING

"${ALLOC_CMD[@]}" "${EXTRA_ALLOC_ARGS[@]}" bash -lc '
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
                "MATVEC_3D_STENCIL")
       KERNELS_DIFFRANGE=("MULTI_REDUCE"
                          "REDUCE_STRUCT"
                          "INDEXLIST_3LOOP"
                          "HALO_PACKING_FUSED")
      ;;
    *)
      echo "Error: unknown kernel set: ${TIER}"
      exit 2
      ;;
  esac

  read -r -a FACTORS <<< "${FACTORS_STRING}"
  read -r -a FACTORS_SHORT <<< "${FACTORS_SHORT_STRING}"
  read -r -a FACTORS_DIFFRANGE <<< "${FACTORS_DIFFRANGE_STRING}"
  read -r -a FACTORS_DIFFRANGE_SHORT <<< "${FACTORS_DIFFRANGE_SHORT_STRING}"

  uses_short_max() {
    case "$1" in
      EDGE3D|FEMSWEEP|INTSC_HEXRECT|INTSC_HEXHEX|HALO_PACKING_FUSED)
        return 0
        ;;
      *)
        return 1
        ;;
    esac
  }

  for KERNEL_NAME in "${KERNELS[@]}"; do
    FACTORS_TO_RUN=("${FACTORS[@]}")
    if uses_short_max "${KERNEL_NAME}"; then
      FACTORS_TO_RUN=("${FACTORS_SHORT[@]}")
    fi

    echo "Running kernel: ${KERNEL_NAME}"
    for factor in "${FACTORS_TO_RUN[@]}"; do
      mem=$(( factor * BASEMEM ))
      echo "  Running with memory = ${mem}"

      ${RUN_CMD} ${RUN_ARGS} ./bin/raja-perf.exe \
        -k "${KERNEL_NAME}" \
        --npasses 1 \
        --npasses-combiners Average Minimum Maximum \
        --outdir "${OUTDIR}" \
        --outfile "${KERNEL_NAME}_factor_${factor}" \
        --memory-allocated "${mem}" \
        --warmup-perfrun-same \
        --variants Base_Seq RAJA_Seq
    done
  done

  if [[ -n "${KERNELS_DIFFRANGE:-}" ]]; then

    for KERNEL_NAME in "${KERNELS_DIFFRANGE[@]}"; do
      FACTORS_TO_RUN=("${FACTORS_DIFFRANGE[@]}")
      if uses_short_max "${KERNEL_NAME}"; then
        FACTORS_TO_RUN=("${FACTORS_DIFFRANGE_SHORT[@]}")
      fi

      echo "Running kernel: ${KERNEL_NAME}"
      for factor in "${FACTORS_TO_RUN[@]}"; do
        mem=$(( factor * BASEMEM ))
        echo "  Running with memory = ${mem}"

        ${RUN_CMD} ${RUN_ARGS} ./bin/raja-perf.exe \
          -k "${KERNEL_NAME}" \
          --npasses 1 \
          --npasses-combiners Average Minimum Maximum \
          --outdir "${OUTDIR}" \
          --outfile "${KERNEL_NAME}_factor_${factor}" \
          --memory-allocated "${mem}" \
          --warmup-perfrun-same \
          --variants Base_Seq RAJA_Seq
      done
    done

  fi
'
