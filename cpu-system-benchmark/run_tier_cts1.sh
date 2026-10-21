#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
readonly KIB=1024
readonly MIB=$((1024 * KIB))

readonly SOCKETS_PER_NODE=2
readonly CORES_PER_SOCKET=18
readonly RANKS_PER_NODE=$((SOCKETS_PER_NODE * CORES_PER_SOCKET))

readonly L2_BYTES_PER_CORE=$((256 * KIB))
readonly L3_BYTES_PER_SOCKET=$((45 * MIB))
readonly TOTAL_CACHE_BYTES=$((
  RANKS_PER_NODE * L2_BYTES_PER_CORE +
    SOCKETS_PER_NODE * L3_BYTES_PER_SOCKET
))
readonly CACHE_BYTES_PER_CORE=$((TOTAL_CACHE_BYTES / RANKS_PER_NODE))

export CPU_BENCH_SYSTEM="CTS1"
export CPU_BENCH_SCHEDULER="slurm"
export CPU_BENCH_RANKS="$RANKS_PER_NODE"
export CPU_BENCH_CACHE_BYTES_PER_CORE="$CACHE_BYTES_PER_CORE"

exec "${SCRIPT_DIR}/run_tier_cpu.sh" "$@"
