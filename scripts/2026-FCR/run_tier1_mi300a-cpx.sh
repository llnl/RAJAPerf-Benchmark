#!/usr/bin/env bash

###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA Performance Suite.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

## Run all 'Priority 1' benchmark kernels for GPU, non-lambda variants only
## on 24 MPI ranks and dump the results in the specified directory.

flux alloc -xN1 --amd-gpumode=CPX -t 45 bash -c '

OUTDIR=RPBenchmark_tier1-CPX

# Reference memory footprint for CPX mode (MI300A)
# 256MB MALL on MI300A --> 256 * 1024 * 1024 / 6 = 44739242 bytes
# Smallest problem size run is 1/2 that... 22369621 bytes
BASEMEM=22369621v

# Collection of problem size factors to run
FACTORS=(1 2 4 8 10 12 16 20)

# List of kernels to run
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
 
for KERNEL_NAME in "${KERNELS[@]}"; do
    echo "Running kernel: $KERNEL_NAME"
    for factor in "${FACTORS[@]}"; do
        mem=$((factor * BASEMEM))
        echo "  Running with memory = $mem"
        flux run -xN1 -n24 -g 1 ./bin/raja-perf.exe \
            -k "$KERNEL_NAME" \
            --npasses 1 \
            --npasses-combiners Average Minimum Maximum \
            --outdir ${OUTDIR} \
            --outfile "${KERNEL_NAME}_factor_${factor}" \
            --memory-allocated "$mem" \
            --warmup-perfrun-same \
            -ev Seq Lambda
    done
done
'

