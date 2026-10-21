#!/usr/bin/env python3

###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA Performance Suite.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

import os
import glob
import statistics
import argparse
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Try to import IPython display for notebook detection
try:
    from IPython.display import display
    from IPython import get_ipython
    IN_NOTEBOOK = get_ipython() is not None
except Exception:
    display = None
    IN_NOTEBOOK = False

# =========================
# Global constants
# =========================

RAW_FLOPS_COL = "Mean gFlops (gigaFLOP per sec.)"
SMOOTH_FLOPS_COL = "Smoothed Mean flops (gigaFLOP per sec.)"
PROBLEM_SIZE_COL = "Problem size"
VARIANT_TUNING_COL = "Variant_Tuning"
KERNEL_COL = "Kernel"
BANDWIDTH_COL = "Bandwidth (GiB per sec.)"
SMOOTH_BW_COL = "Smoothed Bandwidth (GiB per sec.)"
SAT_IDX_COL = "Saturation Index"
ALLOCATED_MEMORY_COL = "Bytes allocated per rep"

# =========================
# General utilities
# =========================

def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)

def sanitize_filename(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(text))

def find_csv_files(root_dir: str, patterns: List[str]) -> List[str]:
    all_files: List[str] = []
    for pattern in patterns:
        search_pattern = os.path.join(root_dir, pattern)
        files = glob.glob(search_pattern, recursive=True)
        all_files.extend(files)
    all_files = sorted(set(all_files))
    return all_files


# =========================
# Header detection / reading
# =========================

def _likely_header_score(line: str) -> int:
    tokens = [
        "Kernel", "Variant", "Problem size", "Problem Size", "Mean flops",
        "GFlop", "GFLOP", "GFLOPs", "GFLOPS"
    ]
    score = 0
    for t in tokens:
        if t in line:
            score += 1
    return score

def read_single_csv(path: str) -> Optional[pd.DataFrame]:
    """Read a CSV and heuristically detect the header row."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        print("Failed to read {}: {}".format(path, e))
        return None

    header_idx = None
    best_score = -1
    for i, line in enumerate(lines[:50]):
        if not line.strip():
            continue
        score = _likely_header_score(line)
        if score > best_score:
            best_score = score
            header_idx = i
            if score >= 3:
                break

    if header_idx is None:
        print("Could not find header in {}, skipping.".format(path))
        return None

    try:
        df = pd.read_csv(path, header=header_idx)
    except Exception as e:
        print("Failed to parse CSV {}: {}".format(path, e))
        return None

    df["__source_file__"] = path
    return df

# =========================
# Column normalization
# =========================

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize various benchmark CSV headers to a standard schema."""
    candidates = {
        "Kernel": ["Kernel", "Kernel name", "Benchmark", "Test"],
        "Variant": ["Variant", "Implementation", "Policy", "Config", "Backend", "Suite"],
        PROBLEM_SIZE_COL: [
            "Problem size", "Problem Size", "Size", "N", "DOF", "Elements",
            "ProblemSize", "Problem-size"
        ],
        RAW_FLOPS_COL: [
            "Mean flops (gigaFLOP per sec.)",
            "Mean flops (GFlop/s)",
            "Mean Flops (GFlop/s)",
            "GFLOP/s", "GFLOPs/s", "GFLOPS", "GFlops/s", "GFlop/s", "GF/s",
            "Mean GFLOP/s", "Mean GFLOPs/s"
        ],
        BANDWIDTH_COL: [
            "Bandwidth (GiB per sec.)",
            "Bandwidth (GiB/s)",
            "Bandwidth (GiB/s)",
            "Bandwidth (GiB/sec)",
            "Bandwidth (GiB/sec)",
            "Mean bandwidth (GiB/s)",
            "Mean Bandwidth (GiB/s)",
            "Mean Bandwidth (GiB per sec.)"
        ],
    }

    df = df.rename(columns={c: c.strip() for c in df.columns})
    new_col_map = {}

    for standard_name, names in candidates.items():
        for c in names:
            if c in df.columns:
                new_col_map[c] = standard_name
                break

    df = df.rename(columns=new_col_map)
    return df


def read_allocated_memory(path: str) -> Optional[pd.DataFrame]:
    """Read actual allocated bytes from a run's sibling kernel-details file."""
    suffix = "-kernel-run-data.csv"
    if not path.endswith(suffix):
        return None

    details_path = path[:-len(suffix)] + "-kernel-details.csv"
    if not os.path.isfile(details_path):
        return None

    details = read_single_csv(details_path)
    if details is None:
        return None
    details = normalize_columns(details)

    bytes_col = "BytesAllocated/rep"
    required = {KERNEL_COL, PROBLEM_SIZE_COL, bytes_col}
    if not required.issubset(details.columns):
        return None

    memory = details[[KERNEL_COL, PROBLEM_SIZE_COL, bytes_col]].copy()
    memory[KERNEL_COL] = memory[KERNEL_COL].astype(str).str.strip()
    memory[PROBLEM_SIZE_COL] = pd.to_numeric(memory[PROBLEM_SIZE_COL], errors="coerce")
    memory[bytes_col] = pd.to_numeric(memory[bytes_col], errors="coerce")
    memory = memory.dropna().drop_duplicates([KERNEL_COL, PROBLEM_SIZE_COL])
    return memory.rename(columns={bytes_col: ALLOCATED_MEMORY_COL})

# =========================
# Backend / tuning classification
# =========================

def classify_backend_from_variant(variant) -> str:
    s = "" if pd.isna(variant) else str(variant).strip()
    low = s.lower()
    if "hip" in low:
        return "HIP"
    if "cuda" in low:
        return "CUDA"
    if "openmp" in low or low.endswith("_omp") or " omp" in low or low.startswith("omp"):
        return "OpenMP"
    if "seq" in low or "serial" in low or "baseline" in low or "sequential" in low:
        return "Seq"
    return "Unknown"

def classify_tuning(row: pd.Series) -> str:
    if "Tuning" in row and pd.notna(row["Tuning"]):
        return str(row["Tuning"]).strip()
    src = row.get("__source_file__", "")
    if isinstance(src, str) and src:
        return os.path.basename(src)
    return "default"

# =========================
# Build combined table
# =========================

def build_combined_table(
    root_dir: str = ".",
    glob_patterns: Optional[List[str]] = None,
    kernel_whitelist: Optional[List[str]] = None,
    verbose: bool = True,
) -> Optional[pd.DataFrame]:
    if glob_patterns is None:
        glob_patterns = ["**/*factor*kernel-run-data.csv"]

    files = find_csv_files(root_dir, glob_patterns)
    if not files:
        if verbose:
            print("No files matching patterns {} found under '{}'".format(glob_patterns, root_dir))
        return None

    if verbose:
        print("Found CSV files:")
        for f in files:
            print("  ", f)

    dfs = []
    required_cols = {KERNEL_COL, "Variant", PROBLEM_SIZE_COL, RAW_FLOPS_COL}

    for path in files:
        df = read_single_csv(path)
        if df is None:
            continue
        df = normalize_columns(df)
        missing = required_cols - set(df.columns)
        if missing:
            if verbose:
                print("[SKIP] {} missing required columns after normalization: {}".format(path, sorted(missing)))
                print("       Columns present:", list(df.columns))
            continue
        df[KERNEL_COL] = df[KERNEL_COL].astype(str).str.strip()
        df[PROBLEM_SIZE_COL] = pd.to_numeric(df[PROBLEM_SIZE_COL], errors="coerce")
        memory = read_allocated_memory(path)
        if memory is not None:
            df = df.merge(memory, on=[KERNEL_COL, PROBLEM_SIZE_COL], how="left")
        else:
            df[ALLOCATED_MEMORY_COL] = np.nan
            if verbose:
                print(f"[WARN] Actual allocated memory unavailable for {path}")
        dfs.append(df)

    if not dfs:
        if verbose:
            print("No CSV files could be parsed with required columns.")
        return None

    combined_df = pd.concat(dfs, ignore_index=True)

    combined_df[KERNEL_COL] = combined_df[KERNEL_COL].astype(str).str.strip()
    combined_df["Variant"] = combined_df["Variant"].astype(str).str.strip()

    if kernel_whitelist:
        wl = [w.lower() for w in kernel_whitelist]
        kernel_series = combined_df[KERNEL_COL].fillna("").astype(str).str.lower()
        mask = kernel_series.apply(lambda k: any(w in k for w in wl))
        combined_df = combined_df[mask]
        if combined_df.empty:
            if verbose:
                print("After applying kernel_whitelist, no rows remain.")
            return None

    combined_df[PROBLEM_SIZE_COL] = pd.to_numeric(combined_df[PROBLEM_SIZE_COL], errors="coerce")
    combined_df[RAW_FLOPS_COL] = pd.to_numeric(combined_df[RAW_FLOPS_COL], errors="coerce")
    if BANDWIDTH_COL in combined_df.columns:
        combined_df[BANDWIDTH_COL] = pd.to_numeric(combined_df[BANDWIDTH_COL], errors="coerce")

    before_drop = len(combined_df)
    combined_df = combined_df.dropna(subset=[PROBLEM_SIZE_COL, RAW_FLOPS_COL])
    dropped = before_drop - len(combined_df)
    if verbose and dropped > 0:
        print("[CLEAN] Dropped {} rows with non-numeric {} or {}.".format(
            dropped, PROBLEM_SIZE_COL, RAW_FLOPS_COL))

    combined_df["Backend"] = combined_df["Variant"].apply(classify_backend_from_variant)
    combined_df["Tuning"] = combined_df.apply(classify_tuning, axis=1)

    combined_df[VARIANT_TUNING_COL] = (
        combined_df["Variant"].astype(str) + "-" + combined_df["Tuning"].astype(str)
    )

    return combined_df

# =========================
# Smoothing and saturation point utilities
# =========================

def moving_median_smooth(y, k: int = 3):
    n = len(y)
    if n == 0:
        return []
    m = (k - 1) // 2
    y_smooth = []
    for i in range(n):
        start = max(0, i - m)
        end = min(n - 1, i + m)
        window = y[start:end + 1]
        y_smooth.append(statistics.median(window))
    return y_smooth

def has_zero_range(y) -> bool:
    """Return whether all finite values in a non-empty curve are identical."""
    values = np.asarray(y, dtype=float)
    values = values[np.isfinite(values)]
    return len(values) > 0 and np.ptp(values) == 0

def select_saturation_curve(
    subdf: pd.DataFrame,
    preferred_raw_col: str = RAW_FLOPS_COL,
    preferred_smooth_col: str = SMOOTH_FLOPS_COL,
    alternate_raw_col: str = BANDWIDTH_COL,
    alternate_smooth_col: str = SMOOTH_BW_COL,
):
    """Select the preferred curve unless it is constant and the alternate varies."""
    preferred_raw = subdf[preferred_raw_col].astype(float).values

    if alternate_raw_col in subdf.columns and alternate_smooth_col in subdf.columns:
        alternate_raw = subdf[alternate_raw_col].astype(float).values
        if has_zero_range(preferred_raw) and not has_zero_range(alternate_raw):
            return subdf[alternate_smooth_col].astype(float).values

    return subdf[preferred_smooth_col].astype(float).values

def find_plateau(x, y_smooth, y_raw=None, eps: float = 0.1, w: int = 3):
    """Find a stable smoothed tail and summarize its corresponding raw values."""
    x_values = np.asarray(x, dtype=float)
    smooth_values = np.asarray(y_smooth, dtype=float)
    raw_values = smooth_values if y_raw is None else np.asarray(y_raw, dtype=float)
    finite = (
        np.isfinite(x_values)
        & np.isfinite(smooth_values)
        & np.isfinite(raw_values)
    )
    x_values = x_values[finite]
    smooth_values = smooth_values[finite]
    raw_values = raw_values[finite]

    if len(smooth_values) < w or np.all(raw_values == 0):
        return None

    def is_stable(values) -> bool:
        median = float(np.median(values))
        if not np.isfinite(median) or median == 0:
            return False
        return bool(np.all(np.abs(values - median) <= eps * abs(median)))

    plateau_start = len(smooth_values) - w
    if not is_stable(smooth_values[plateau_start:]):
        return None

    while plateau_start > 0 and is_stable(smooth_values[plateau_start - 1:]):
        plateau_start -= 1

    plateau_values = raw_values[plateau_start:]
    plateau_value = float(np.median(plateau_values))
    threshold = (1.0 - eps) * plateau_value

    threshold_idx = None
    for idx in range(len(smooth_values)):
        if np.all(smooth_values[idx:] >= threshold):
            threshold_idx = idx
            break

    if threshold_idx is None:
        return None

    return {
        "value": plateau_value,
        "minimum": float(np.min(plateau_values)),
        "maximum": float(np.max(plateau_values)),
        "points": len(plateau_values),
        "plateau_start_index": plateau_start,
        "threshold_index": threshold_idx,
        "threshold_size": float(x_values[threshold_idx]),
    }


def summarize_plateau_metric(
    subdf: pd.DataFrame,
    metric_col: str,
    eps: float = 0.1,
    w: int = 3,
):
    """Detect a smoothed plateau and summarize its raw performance values."""
    smooth_col = {
        RAW_FLOPS_COL: SMOOTH_FLOPS_COL,
        BANDWIDTH_COL: SMOOTH_BW_COL,
    }.get(metric_col)
    if metric_col not in subdf.columns or smooth_col not in subdf.columns:
        return None

    curve = subdf[[PROBLEM_SIZE_COL, metric_col, smooth_col]].copy()
    curve[PROBLEM_SIZE_COL] = pd.to_numeric(curve[PROBLEM_SIZE_COL], errors="coerce")
    curve[metric_col] = pd.to_numeric(curve[metric_col], errors="coerce")
    curve[smooth_col] = pd.to_numeric(curve[smooth_col], errors="coerce")
    curve = curve.dropna().groupby(PROBLEM_SIZE_COL, as_index=False)[
        [metric_col, smooth_col]
    ].mean()
    curve = curve.sort_values(PROBLEM_SIZE_COL)
    return find_plateau(
        curve[PROBLEM_SIZE_COL].values,
        curve[smooth_col].values,
        curve[metric_col].values,
        eps=eps,
        w=w,
    )


def find_sustained_threshold_x(
    subdf: pd.DataFrame,
    metric_col: str,
    x_col: str,
    plateau_value: float,
    eps: float = 0.1,
):
    """Find the smallest x after which performance stays above the threshold."""
    if metric_col not in subdf.columns or x_col not in subdf.columns:
        return np.nan

    curve = subdf[[x_col, metric_col]].copy()
    curve[x_col] = pd.to_numeric(curve[x_col], errors="coerce")
    curve[metric_col] = pd.to_numeric(curve[metric_col], errors="coerce")
    curve = curve.dropna().groupby(x_col, as_index=False)[metric_col].mean()
    curve = curve.sort_values(x_col)
    threshold = (1.0 - eps) * plateau_value
    values = curve[metric_col].to_numpy(dtype=float)
    for idx in range(len(curve)):
        if np.all(values[idx:] >= threshold):
            return float(curve[x_col].iloc[idx])
    return np.nan


def find_saturation_point(x, y_smooth, eps: float = 0.1, w: int = 3):
    """Return the earliest sustained plateau-threshold index, if one exists."""
    plateau = find_plateau(x, y_smooth, eps=eps, w=w)
    return None if plateau is None else plateau["threshold_index"]

# =========================
# Find the saturation point for single kernel
# =========================

def find_saturation_kernel(
    df: pd.DataFrame,
    kernel: str,
    k: int = 3,
    eps: float = 0.1,
    w: int = 3,
    save_dir: Optional[str] = None,
):
    variants = df[VARIANT_TUNING_COL].unique()

    # generate smoothed data
    if SMOOTH_FLOPS_COL not in df.columns:
        df[SMOOTH_FLOPS_COL] = np.nan

    if SMOOTH_BW_COL not in df.columns:
        df[SMOOTH_BW_COL] = np.nan

    for idx, variant in enumerate(variants):
        subdf = df[df[VARIANT_TUNING_COL] == variant].copy()
        subdf = subdf.sort_values(PROBLEM_SIZE_COL)
        x = subdf[PROBLEM_SIZE_COL].astype(float).values

        if len(x) == 0:
            continue

        if RAW_FLOPS_COL in subdf.columns:
            y = subdf[RAW_FLOPS_COL].astype(float).values

            y_smooth = moving_median_smooth(list(y), k=k)

            for xi, yi in zip(x, y_smooth):
                mask = (
                    (df[VARIANT_TUNING_COL] == variant)
                    & (df[PROBLEM_SIZE_COL] == xi)
                )
                df.loc[mask, SMOOTH_FLOPS_COL] = yi

        if BANDWIDTH_COL in subdf.columns:
            y_bw = subdf[BANDWIDTH_COL].astype(float).values

            y_bw_smooth = moving_median_smooth(list(y_bw), k=k)

            for xi, yi in zip(x, y_bw_smooth):
                mask = (
                    (df[VARIANT_TUNING_COL] == variant)
                    & (df[PROBLEM_SIZE_COL] == xi)
                )
                df.loc[mask, SMOOTH_BW_COL] = yi

    # FOM-style output as DataFrame
    fom_rows = []
    for variant in variants:
        subdf = df[df[VARIANT_TUNING_COL] == variant].copy()
        subdf = subdf.sort_values(PROBLEM_SIZE_COL)
        x = subdf[PROBLEM_SIZE_COL].astype(float).values
        flops = summarize_plateau_metric(subdf, RAW_FLOPS_COL, eps=eps, w=w)
        bandwidth = summarize_plateau_metric(subdf, BANDWIDTH_COL, eps=eps, w=w)
        primary = flops if flops is not None else bandwidth
        sat_size = primary["threshold_size"] if primary else ""
        sat_flops = flops["value"] if flops else ""
        sat_bw = bandwidth["value"] if bandwidth else ""
        sat_idx = None
        if primary is not None:
            matching_indices = np.flatnonzero(x == primary["threshold_size"])
            if len(matching_indices) > 0:
                sat_idx = int(matching_indices[0])

        df.loc[df[VARIANT_TUNING_COL] == variant, SAT_IDX_COL] = sat_idx

        if primary is None:
            print(
                "[INFO] No stable tail found for kernel '{}' "
                "variant '{}' (eps={}, w={})".format(kernel, variant, eps, w)
            )

        fom_rows.append({
            "Kernel": f"{kernel}-{variant}",
            "Sat Problem Size": sat_size if sat_size != "" else "N/A",
            "Sat GFLOP/s": sat_flops if sat_flops != "" else "N/A",
            "Sat B/W (GiB per sec.)": sat_bw if sat_bw != "" else "N/A",
        })

    fom_df = pd.DataFrame(fom_rows)
    if IN_NOTEBOOK and display is not None:
        display(fom_df)
    else:
        print(fom_df.to_string(index=False))

# =========================
# Plotting for a single kernel (FLOPs) with FOM output as DataFrame
# =========================

def plot_kernel(
    df: pd.DataFrame,
    kernel: str,
    k: int = 3,
    eps: float = 0.1,
    w: int = 3,
    save_dir: Optional[str] = None,
    exclude_plot_variants: Optional[List[str]] = None,
):
    if RAW_FLOPS_COL not in df.columns:
        print(f"[INFO] No '{RAW_FLOPS_COL}' column for kernel '{kernel}', skipping flops plot.")
        return

    if exclude_plot_variants is None:
        exclude_plot_variants = []

    exclude_plot_variants_set = {str(v).strip() for v in exclude_plot_variants}

    plot_df = df.copy()
    if exclude_plot_variants_set:
        plot_df = plot_df[~plot_df["Variant"].astype(str).str.strip().isin(exclude_plot_variants_set)]

    if plot_df.empty:
        print(f"[INFO] All variants excluded from FLOPs plot for kernel '{kernel}', skipping plot.")
        return

    variants = plot_df[VARIANT_TUNING_COL].unique()

    plt.figure(figsize=(18, 7))
    colors = plt.cm.tab10.colors

    ymin = 1e99
    ymax = -1e99

    for idx, variant in enumerate(variants):
        subdf = plot_df[plot_df[VARIANT_TUNING_COL] == variant].copy()
        subdf = subdf.sort_values(PROBLEM_SIZE_COL)
        x = subdf[PROBLEM_SIZE_COL].astype(float).values
        y = subdf[RAW_FLOPS_COL].astype(float).values
        y_smooth = subdf[SMOOTH_FLOPS_COL].astype(float).values

        if len(x) == 0:
            continue

        y_min = min(y)
        if (y_min < ymin):
            ymin = y_min
        y_smooth_min = min(y_smooth)
        if (y_smooth_min < ymin):
            ymin = y_smooth_min

        y_max = max(y)
        if (y_max > ymax):
            ymax = y_max
        y_smooth_max = max(y_smooth)
        if (y_smooth_max > ymax):
            ymax = y_smooth_max

        plt.plot(
            x, y, "-",
            label="{} (raw flops)".format(variant),
            color=colors[idx % len(colors)],
            marker="o",
            markersize=8,
            markerfacecolor=colors[idx % len(colors)],
            markeredgewidth=0
        )

        plt.plot(
            x, y_smooth, "--",
            label="{} (smoothed flops)".format(variant),
            linewidth=3,
            color=colors[idx % len(colors)],
            marker="+",
            markersize=10
        )

        plateau = summarize_plateau_metric(subdf, RAW_FLOPS_COL, eps=eps, w=w)
        if plateau is not None:
            sat_x = plateau["threshold_size"]
            sat_y = plateau["value"]
            plt.plot(
                [sat_x], [sat_y], "-",
                label="{} plateau threshold ({}, {})".format(variant, sat_x, sat_y),
                linewidth=0,
                color=colors[idx % len(colors)],
                marker="*",
                markersize=20
            )

    if ymin > 0:
        ymin = 0
    yrange = ymax-ymin
    yoverhang = yrange*0.1 if yrange > 0 else max(abs(ymax)*0.1, 1.0)

    plt.title("Kernel: {}".format(kernel), fontsize=22)
    plt.xlabel("Problem size (bytes)", fontsize=18)
    plt.ylabel("Mean flops (GFLOP per sec.)", fontsize=18)
    plt.grid(True, which="both", linestyle="--", linewidth=1)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim((ymin-yoverhang, ymax+yoverhang))
    plt.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=10,
        frameon=False,
        handlelength=1.4,
        labelspacing=0.7,
        borderaxespad=0.3)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    if save_dir is not None:
        ensure_dir(save_dir)
        fname = os.path.join(save_dir, "{}_flops.png".format(sanitize_filename(kernel)))
        plt.savefig(fname, dpi=200)
        print("Saved plot to {}".format(fname))
    if IN_NOTEBOOK:
        plt.show()
    else:
        plt.close()

# =========================
# Plotting for a single kernel (Bandwidth)
# =========================

def plot_kernel_bandwidth(
    df: pd.DataFrame,
    kernel: str,
    k: int = 3,
    eps: float = 0.1,
    w: int = 3,
    save_dir: Optional[str] = None,
    exclude_plot_variants: Optional[List[str]] = None,
):
    if BANDWIDTH_COL not in df.columns:
        print(f"[INFO] No '{BANDWIDTH_COL}' column for kernel '{kernel}', skipping bandwidth plot.")
        return

    if exclude_plot_variants is None:
        exclude_plot_variants = []

    exclude_plot_variants_set = {str(v).strip() for v in exclude_plot_variants}

    plot_df = df.copy()
    if exclude_plot_variants_set:
        plot_df = plot_df[~plot_df["Variant"].astype(str).str.strip().isin(exclude_plot_variants_set)]

    if plot_df.empty:
        print(f"[INFO] All variants excluded from bandwidth plot for kernel '{kernel}', skipping plot.")
        return

    variants = plot_df[VARIANT_TUNING_COL].unique()

    plt.figure(figsize=(18, 7))
    colors = plt.cm.tab10.colors

    ymin = 1e99
    ymax = -1e99

    for idx, variant in enumerate(variants):
        subdf = plot_df[plot_df[VARIANT_TUNING_COL] == variant].copy()
        subdf = subdf.sort_values(PROBLEM_SIZE_COL)
        x = subdf[PROBLEM_SIZE_COL].astype(float).values
        y_bw = subdf[BANDWIDTH_COL].astype(float).values
        y_bw_smooth = subdf[SMOOTH_BW_COL].astype(float).values

        if len(x) == 0:
            continue

        y_bw_min = min(y_bw)
        if (y_bw_min < ymin):
            ymin = y_bw_min
        y_bw_smooth_min = min(y_bw_smooth)
        if (y_bw_smooth_min < ymin):
            ymin = y_bw_smooth_min

        y_bw_max = max(y_bw)
        if (y_bw_max > ymax):
            ymax = y_bw_max
        y_bw_smooth_max = max(y_bw_smooth)
        if (y_bw_smooth_max > ymax):
            ymax = y_bw_smooth_max

        plt.plot(
            x, y_bw, "-",
            label=f"{variant} (raw B/W)",
            color=colors[idx % len(colors)],
            marker="o",
            markersize=8,
            markerfacecolor=colors[idx % len(colors)],
            markeredgewidth=0,
        )

        plt.plot(
            x, y_bw_smooth, "--",
            label=f"{variant} (smoothed B/W)",
            linewidth=3,
            color=colors[idx % len(colors)],
            marker="+",
            markersize=10
        )

        plateau = summarize_plateau_metric(subdf, BANDWIDTH_COL, eps=eps, w=w)
        if plateau is not None:
            sat_x = plateau["threshold_size"]
            sat_y = plateau["value"]
            plt.plot(
                [sat_x], [sat_y], "-",
                label="{} plateau threshold ({}, {})".format(variant, sat_x, sat_y),
                linewidth=0,
                color=colors[idx % len(colors)],
                marker="*",
                markersize=20
            )

    if ymin > 0:
        ymin = 0
    yrange = ymax-ymin
    yoverhang = yrange*0.1 if yrange > 0 else max(abs(ymax)*0.1, 1.0)

    plt.title(f"Kernel: {kernel} - Bandwidth", fontsize=22)
    plt.xlabel("Problem size", fontsize=18)
    plt.ylabel("Bandwidth (GiB per sec.)", fontsize=18)
    plt.grid(True, which="both", linestyle="--", linewidth=1)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim((ymin-yoverhang, ymax+yoverhang))
    plt.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=10,
        frameon=False,
        handlelength=1.4,
        labelspacing=0.7,
        borderaxespad=0.3)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    if save_dir is not None:
        ensure_dir(save_dir)
        fname = os.path.join(save_dir, "{}_bandwidth.png".format(sanitize_filename(kernel)))
        plt.savefig(fname, dpi=200)
        print(f"Saved bandwidth plot to {fname}")
    if IN_NOTEBOOK:
        plt.show()
    else:
        plt.close()

# =========================
# Smooth and plot all kernels (FLOPs and Bandwidth)
# =========================

def smooth_and_plot_all_kernels(
    df: pd.DataFrame,
    k: int = 3,
    eps: float = 0.1,
    w: int = 3,
    save_dir: Optional[str] = None,
    exclude_plot_variants: Optional[List[str]] = None,
) -> pd.DataFrame:
    df = df.dropna(subset=[KERNEL_COL, PROBLEM_SIZE_COL, RAW_FLOPS_COL, VARIANT_TUNING_COL]).copy()
    df[SMOOTH_FLOPS_COL] = np.nan

    if BANDWIDTH_COL in df.columns:
        df[SMOOTH_BW_COL] = np.nan

    kernels = df[KERNEL_COL].unique()
    print("Found {} kernels.".format(len(kernels)))

    for kernel in kernels:
        tmp = df[df[KERNEL_COL] == kernel].copy()

        find_saturation_kernel(tmp, kernel, k=k, eps=eps, w=w, save_dir=save_dir)
        if SMOOTH_FLOPS_COL in tmp.columns:
            df.loc[df[KERNEL_COL] == kernel, SMOOTH_FLOPS_COL] = tmp[SMOOTH_FLOPS_COL]
        if SMOOTH_BW_COL in tmp.columns:
            df.loc[df[KERNEL_COL] == kernel, SMOOTH_BW_COL] = tmp[SMOOTH_BW_COL]

        plot_kernel(
            tmp,
            kernel,
            k=k,
            eps=eps,
            w=w,
            save_dir=save_dir,
            exclude_plot_variants=exclude_plot_variants,
        )

        if BANDWIDTH_COL in df.columns:
            plot_kernel_bandwidth(
                tmp,
                kernel,
                k=k,
                eps=eps,
                w=w,
                save_dir=save_dir,
                exclude_plot_variants=exclude_plot_variants,
            )

    return df

# =========================
# Per-kernel tables (raw and smoothed)
# =========================

def save_kernel_tables(
    df: pd.DataFrame,
    outdir: str = "kernel_tables",
) -> None:
    ensure_dir(outdir)
    kernels = df[KERNEL_COL].unique()
    for kernel in kernels:
        df_kernel = df[df[KERNEL_COL] == kernel]
        raw_table = df_kernel.pivot_table(
            index=PROBLEM_SIZE_COL,
            columns=VARIANT_TUNING_COL,
            values=RAW_FLOPS_COL,
        )
        raw_table = raw_table.sort_index()
        raw_csv_path = os.path.join(outdir, "{}_flops_raw.csv".format(sanitize_filename(kernel)))
        raw_table.to_csv(raw_csv_path)
        smooth_table = df_kernel.pivot_table(
            index=PROBLEM_SIZE_COL,
            columns=VARIANT_TUNING_COL,
            values=SMOOTH_FLOPS_COL,
        )
        smooth_table = smooth_table.sort_index()
        smooth_csv_path = os.path.join(outdir, "{}_flops_smoothed.csv".format(sanitize_filename(kernel)))
        smooth_table.to_csv(smooth_csv_path)

        if BANDWIDTH_COL in df_kernel.columns and SMOOTH_BW_COL in df_kernel.columns:
            bw_raw_table = df_kernel.pivot_table(
                index=PROBLEM_SIZE_COL,
                columns=VARIANT_TUNING_COL,
                values=BANDWIDTH_COL,
            ).sort_index()
            bw_raw_csv_path = os.path.join(outdir, "{}_bandwidth_raw.csv".format(sanitize_filename(kernel)))
            bw_raw_table.to_csv(bw_raw_csv_path)

            bw_smooth_table = df_kernel.pivot_table(
                index=PROBLEM_SIZE_COL,
                columns=VARIANT_TUNING_COL,
                values=SMOOTH_BW_COL,
            ).sort_index()
            bw_smooth_csv_path = os.path.join(outdir, "{}_bandwidth_smoothed.csv".format(sanitize_filename(kernel)))
            bw_smooth_table.to_csv(bw_smooth_csv_path)

            print("Saved: {}, {}, {}, {}".format(
                raw_csv_path, smooth_csv_path, bw_raw_csv_path, bw_smooth_csv_path))
        else:
            print("Saved: {}, {}".format(raw_csv_path, smooth_csv_path))

# =========================
# Per-kernel saturation curve data (raw + smoothed in one file)
# =========================

def save_saturation_curve_data(
    df: pd.DataFrame,
    outdir: str = "saturation-curve-data",
) -> None:
    ensure_dir(outdir)
    kernels = df[KERNEL_COL].unique()
    for kernel in kernels:
        df_kernel = df[df[KERNEL_COL] == kernel].copy()
        raw_table = df_kernel.pivot_table(
            index=PROBLEM_SIZE_COL,
            columns=VARIANT_TUNING_COL,
            values=RAW_FLOPS_COL,
        )
        smooth_table = df_kernel.pivot_table(
            index=PROBLEM_SIZE_COL,
            columns=VARIANT_TUNING_COL,
            values=SMOOTH_FLOPS_COL,
        )
        raw_table = raw_table.sort_index()
        smooth_table = smooth_table.reindex(
            index=raw_table.index,
            columns=raw_table.columns,
        )
        combined = pd.DataFrame(index=raw_table.index)
        for vt in raw_table.columns:
            raw_col_name = f"{vt} (raw)"
            smooth_col_name = f"{vt} (smoothed)"
            combined[raw_col_name] = raw_table[vt]
            combined[smooth_col_name] = smooth_table[vt]

        if BANDWIDTH_COL in df_kernel.columns and SMOOTH_BW_COL in df_kernel.columns:
            bw_raw_table = df_kernel.pivot_table(
                index=PROBLEM_SIZE_COL,
                columns=VARIANT_TUNING_COL,
                values=BANDWIDTH_COL,
            ).sort_index()
            bw_smooth_table = df_kernel.pivot_table(
                index=PROBLEM_SIZE_COL,
                columns=VARIANT_TUNING_COL,
                values=SMOOTH_BW_COL,
            ).sort_index()
            bw_smooth_table = bw_smooth_table.reindex(
                index=bw_raw_table.index,
                columns=bw_raw_table.columns,
            )
            for vt in bw_raw_table.columns:
                bw_raw_col_name = f"{vt} (raw B/W)"
                bw_smooth_col_name = f"{vt} (smoothed B/W)"
                combined[bw_raw_col_name] = bw_raw_table[vt]
                combined[bw_smooth_col_name] = bw_smooth_table[vt]

        combined = combined.reset_index().rename(columns={PROBLEM_SIZE_COL: "Problem size"})
        out_path = os.path.join(outdir, f"{sanitize_filename(kernel)}.csv")
        combined.to_csv(out_path, index=False)
        print(f"Saved saturation curve data: {out_path}")

# =========================
# Per-kernel FOM tables and summary charts
# =========================

def save_fom_tables(
    df: pd.DataFrame,
    eps: float = 0.1,
    w: int = 3,
    outdir: str = "FOM"
) -> pd.DataFrame:
    ensure_dir(outdir)
    kernels = df[KERNEL_COL].unique()
    all_rows = []

    for kernel in kernels:
        df_kernel = df[df[KERNEL_COL] == kernel].copy()
        variant_tunings = df_kernel[VARIANT_TUNING_COL].unique()
        rows = []

        for vt in variant_tunings:
            subdf = df_kernel[df_kernel[VARIANT_TUNING_COL] == vt].copy()
            flops = summarize_plateau_metric(subdf, RAW_FLOPS_COL, eps=eps, w=w)
            bandwidth = summarize_plateau_metric(subdf, BANDWIDTH_COL, eps=eps, w=w)

            flops_memory = find_sustained_threshold_x(
                subdf, SMOOTH_FLOPS_COL, ALLOCATED_MEMORY_COL, flops["value"], eps=eps
            ) if flops else np.nan
            bandwidth_memory = find_sustained_threshold_x(
                subdf, SMOOTH_BW_COL, ALLOCATED_MEMORY_COL, bandwidth["value"], eps=eps
            ) if bandwidth else np.nan

            if flops is None and bandwidth is None:
                print(
                    "[INFO] No stable tail found for kernel '{}' variant '{}' "
                    "when building FOM table".format(kernel, vt)
                )

            variant = str(subdf["Variant"].iloc[0])
            tuning = str(subdf["Tuning"].iloc[0])
            primary_size = flops["threshold_size"] if flops else (
                bandwidth["threshold_size"] if bandwidth else np.nan
            )

            row = {
                "Kernel": f"{kernel}-{vt}",
                "Benchmark": kernel,
                "Variant": variant,
                "Tuning": tuning,
                "Sat Problem Size": primary_size,
                "Sat GFLOP/s": flops["value"] if flops else np.nan,
                "Sat B/W (GiB per sec.)": bandwidth["value"] if bandwidth else np.nan,
                "Sat FLOP/s Problem Size": flops["threshold_size"] if flops else np.nan,
                "Sat B/W Problem Size": bandwidth["threshold_size"] if bandwidth else np.nan,
                "Sat FLOP/s Allocated Memory (bytes)": flops_memory,
                "Sat B/W Allocated Memory (bytes)": bandwidth_memory,
                "FLOP/s Plateau Minimum": flops["minimum"] if flops else np.nan,
                "FLOP/s Plateau Maximum": flops["maximum"] if flops else np.nan,
                "FLOP/s Plateau Points": flops["points"] if flops else np.nan,
                "B/W Plateau Minimum": bandwidth["minimum"] if bandwidth else np.nan,
                "B/W Plateau Maximum": bandwidth["maximum"] if bandwidth else np.nan,
                "B/W Plateau Points": bandwidth["points"] if bandwidth else np.nan,
            }
            rows.append(row)

        out_path = os.path.join(outdir, f"{sanitize_filename(kernel)}.csv")
        pd.DataFrame(rows).to_csv(out_path, index=False, na_rep="N/A")
        print(f"Saved FOM table: {out_path}")
        all_rows.extend(rows)

    combined_path = os.path.join(outdir, "combined_fom.csv")
    combined_df = pd.DataFrame(all_rows)
    if not combined_df.empty:
        combined_df = combined_df.sort_values("Kernel")
    combined_df.to_csv(combined_path, index=False, na_rep="N/A")
    print(f"Saved combined FOM table: {combined_path}")
    summary_path = os.path.join(outdir, "summary.csv")
    combined_df.to_csv(summary_path, index=False, na_rep="N/A")
    print(f"Saved summary table: {summary_path}")
    return combined_df


def save_summary_chart(
    fom_df: pd.DataFrame,
    value_col: str,
    ylabel: str,
    title: str,
    out_path: str,
    log_scale: bool = False,
    minimum_col: Optional[str] = None,
    maximum_col: Optional[str] = None,
) -> None:
    """Plot grouped Base and RAJA variant/tuning bars for one FOM metric."""
    all_benchmarks = sorted(fom_df["Benchmark"].dropna().unique())
    all_variant_tunings = sorted(
        (fom_df["Variant"].astype(str) + "-" + fom_df["Tuning"].astype(str)).unique()
    )
    plot_df = fom_df[["Benchmark", "Variant", "Tuning", value_col]].copy()
    plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors="coerce")
    if not (plot_df[value_col].fillna(0) > 0).any():
        print(f"[INFO] No values available for summary chart: {title}")
        return

    plot_df["Variant/Tuning"] = (
        plot_df["Variant"].astype(str) + "-" + plot_df["Tuning"].astype(str)
    )
    present = plot_df.assign(Present=True).pivot_table(
        index="Benchmark",
        columns="Variant/Tuning",
        values="Present",
        aggfunc="max",
    ).reindex(index=all_benchmarks, columns=all_variant_tunings).eq(True)
    table = plot_df.pivot_table(
        index="Benchmark",
        columns="Variant/Tuning",
        values=value_col,
        aggfunc="first",
    ).reindex(index=all_benchmarks, columns=all_variant_tunings)
    missing = present & (table.isna() | (table <= 0))
    table = table.mask(missing, 0)

    figure_width = max(12, 1.25 * len(table.index))
    ax = table.plot(kind="bar", figsize=(figure_width, 8), width=0.82)
    if minimum_col and maximum_col:
        minimums = fom_df.pivot_table(
            index="Benchmark",
            columns=fom_df["Variant"].astype(str) + "-" + fom_df["Tuning"].astype(str),
            values=minimum_col,
            aggfunc="first",
        ).reindex(index=table.index, columns=table.columns)
        maximums = fom_df.pivot_table(
            index="Benchmark",
            columns=fom_df["Variant"].astype(str) + "-" + fom_df["Tuning"].astype(str),
            values=maximum_col,
            aggfunc="first",
        ).reindex(index=table.index, columns=table.columns)
        for column, container in zip(table.columns, ax.containers):
            for row_idx, bar in enumerate(container):
                value = table.iloc[row_idx][column]
                minimum = minimums.iloc[row_idx][column]
                maximum = maximums.iloc[row_idx][column]
                if pd.notna(value) and pd.notna(minimum) and pd.notna(maximum):
                    ax.errorbar(
                        bar.get_x() + bar.get_width() / 2,
                        value,
                        yerr=[[value - minimum], [maximum - value]],
                        fmt="none",
                        ecolor="black",
                        capsize=3,
                        linewidth=1,
                    )
    if log_scale:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("Kernel")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="white", edgecolor="black", hatch="//"))
    labels.append("N/A (shown as 0)")
    ax.legend(handles, labels, title="Variant/Tuning", fontsize=9)

    y_bottom, y_top = ax.get_ylim()
    na_y = y_bottom * 1.15 if log_scale else y_top * 0.01
    for column, container in zip(table.columns, ax.containers):
        for row_idx, bar in enumerate(container):
            if missing.iloc[row_idx][column]:
                ax.annotate(
                    "N/A",
                    (bar.get_x() + bar.get_width() / 2, na_y),
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=7,
                )
    ax.figure.tight_layout()
    ax.figure.savefig(out_path, dpi=200)
    plt.close(ax.figure)
    print(f"Saved summary chart: {out_path}")


def save_summary_charts(fom_df: pd.DataFrame, outdir: str) -> None:
    ensure_dir(outdir)
    chart_specs = [
        (
            "Sat GFLOP/s", "GFLOP/s", "Saturated FLOP/s", "summary_flops.png", False,
            "FLOP/s Plateau Minimum", "FLOP/s Plateau Maximum",
        ),
        (
            "Sat B/W (GiB per sec.)",
            "GiB/s",
            "Saturated Bandwidth",
            "summary_bandwidth.png",
            False,
            "B/W Plateau Minimum",
            "B/W Plateau Maximum",
        ),
        (
            "Sat FLOP/s Problem Size",
            "Problem size",
            "Smallest Size Sustaining Saturated FLOP/s",
            "summary_flops_problem_size.png",
            True,
            None,
            None,
        ),
        (
            "Sat B/W Problem Size",
            "Problem size",
            "Smallest Size Sustaining Saturated Bandwidth",
            "summary_bandwidth_problem_size.png",
            True,
            None,
            None,
        ),
        (
            "Sat FLOP/s Allocated Memory (bytes)",
            "Actual allocated memory (bytes)",
            "Smallest Actual Allocation Sustaining Saturated FLOP/s",
            "summary_flops_memory.png",
            True,
            None,
            None,
        ),
        (
            "Sat B/W Allocated Memory (bytes)",
            "Actual allocated memory (bytes)",
            "Smallest Actual Allocation Sustaining Saturated Bandwidth",
            "summary_bandwidth_memory.png",
            True,
            None,
            None,
        ),
    ]
    for (
        value_col, ylabel, title, filename, log_scale, minimum_col, maximum_col
    ) in chart_specs:
        save_summary_chart(
            fom_df,
            value_col=value_col,
            ylabel=ylabel,
            title=title,
            out_path=os.path.join(outdir, filename),
            log_scale=log_scale,
            minimum_col=minimum_col,
            maximum_col=maximum_col,
        )


# =========================
# Main pipeline: callable from notebook or CLI
# =========================

def run_pipeline(
    root_dir: str = ".",
    output_dir: str = "./output",
    glob_patterns: Optional[List[str]] = None,
    kernel_whitelist: Optional[List[str]] = None,
    smooth_window: int = 3,
    saturation_eps: float = 0.1,
    saturation_w: int = 3,
    exclude_plot_variants: Optional[List[str]] = None,
) -> Optional[pd.DataFrame]:
    if glob_patterns is None:
        glob_patterns = ["**/*factor*kernel-run-data.csv"]

    ensure_dir(output_dir)
    fig_dir = os.path.join(output_dir, "figures")
    ensure_dir(fig_dir)

    combined_df = build_combined_table(
        root_dir=root_dir,
        glob_patterns=glob_patterns,
        kernel_whitelist=kernel_whitelist,
        verbose=True,
    )
    if combined_df is None:
        print("No combined table created.")
        return None

    combined_csv_path = os.path.join(output_dir, "combined_table.csv")
    combined_df.to_csv(combined_csv_path, index=False)
    print("[SAVE] Combined table saved to {}".format(combined_csv_path))

    df_smoothed = smooth_and_plot_all_kernels(
        combined_df,
        k=smooth_window,
        eps=saturation_eps,
        w=saturation_w,
        save_dir=fig_dir,
        exclude_plot_variants=exclude_plot_variants,
    )

    output_variant_tuning_path = os.path.join(output_dir, "output_with_variant_tuning.csv")
    df_smoothed.to_csv(output_variant_tuning_path, index=False)
    print("[SAVE] Smoothed combined table saved to {}".format(output_variant_tuning_path))

    save_kernel_tables(df_smoothed, outdir=output_dir)
    saturation_dir = os.path.join(output_dir, "saturation-curve-data")
    save_saturation_curve_data(df_smoothed, outdir=saturation_dir)
    fom_dir = os.path.join(output_dir, "FOM")
    fom_df = save_fom_tables(df_smoothed, saturation_eps, saturation_w, outdir=fom_dir)
    save_summary_charts(fom_df, outdir=fig_dir)

    return df_smoothed

# =========================
# CLI entry point
# =========================

def main_cli():
    parser = argparse.ArgumentParser(
        description="Combine benchmark CSVs, smooth FLOPs/Bandwidth and generate plots/tables."
    )
    parser.add_argument("--root-dir", default=".", help="Root directory to search for CSV files (default: '.')")
    parser.add_argument("--output-dir", default="./output", help="Directory to write outputs (default: './output')")
    parser.add_argument(
        "--glob-pattern",
        action="append",
        default=None,
        help="Glob pattern for CSV files (can be repeated). Default: **/*factor*kernel-run-data.csv",
    )
    parser.add_argument(
        "--kernel-whitelist",
        nargs="*",
        default=None,
        help="List of substrings to filter Kernel names by (case insensitive, default: none).",
    )
    parser.add_argument("--smooth-window", type=int, default=3, help="Moving median window size (default: '3')")
    parser.add_argument("--saturation-eps", type=float, default=0.1, help="Epsilon for saturation threshold (default: '0.1')")
    parser.add_argument("--saturation-w", type=int, default=3, help="Consecutive points needed for saturation (default: '3')")
    parser.add_argument(
        "--exclude-plot-variant",
        action="append",
        default=None,
        help="Variant name to exclude from generated plots. Can be repeated, for example: "
             "--exclude-plot-variant Base_Seq --exclude-plot-variant CUDA",
    )

    args = parser.parse_args()

    run_pipeline(
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        glob_patterns=args.glob_pattern,
        kernel_whitelist=args.kernel_whitelist,
        smooth_window=args.smooth_window,
        saturation_eps=args.saturation_eps,
        saturation_w=args.saturation_w,
        exclude_plot_variants=args.exclude_plot_variant,
    )

# =========================
# Notebook helper (no argparse)
# =========================

def main_notebook(
    root_dir: str = ".",
    output_dir: str = "./output",
    glob_patterns: Optional[List[str]] = None,
    kernel_whitelist: Optional[List[str]] = None,
    smooth_window: int = 3,
    saturation_eps: float = 0.1,
    saturation_w: int = 3,
    exclude_plot_variants: Optional[List[str]] = None,
):
    return run_pipeline(
        root_dir=root_dir,
        output_dir=output_dir,
        glob_patterns=glob_patterns,
        kernel_whitelist=kernel_whitelist,
        smooth_window=smooth_window,
        saturation_eps=saturation_eps,
        saturation_w=saturation_w,
        exclude_plot_variants=exclude_plot_variants,
    )

if __name__ == "__main__":
    if not IN_NOTEBOOK:
        main_cli()
