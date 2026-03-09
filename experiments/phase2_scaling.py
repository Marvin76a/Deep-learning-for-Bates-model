"""Phase 2: Pricing Accuracy and Computational Complexity Scaling.

Purpose
-------
Demonstrate that the Triple-Net deep learning algorithm maintains accuracy
across all dimensions (d = 1 to 200), while MC computation cost grows
steeply with dimension.

Benchmarks:
  - COS Fourier transform (d=1, exact)
  - Large-scale Monte Carlo with antithetic variates (d >= 3)

Outputs
-------
  - Unified scaling table matching thesis Table format:
    Dimension | MC Benchmark (95% CI) | Our Model (Y0) | MC Time | Our Time (per 1k Epochs)
  - Computation time vs dimension chart
  - Relative error vs dimension chart
"""

import sys
import os
import time
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from src.config import BatesConfig
from src.cos_bates import cos_price_from_config
from src.mc_bates import mc_price_basket
from src import solver_triple


# -----------------------------------------------------------------------
# Experiment settings
# -----------------------------------------------------------------------
DIMS = [1, 3, 10, 50, 100, 200]
EPOCHS = 3000
N_STEPS = 100
BATCH_SIZE = 1024

MC_PATHS = {
    3:   10_000_000,
    10:  10_000_000,
    50:  10_000_000,
    100: 10_000_000,
    200: 10_000_000,
}
MC_BATCH = 50_000


def run_for_dim(d):
    """Run Triple-Net and benchmark for a single dimension."""
    print(f"\n{'#'*60}")
    print(f"# d = {d}")
    print(f"{'#'*60}")

    cfg = BatesConfig(d=d, N=N_STEPS, M=BATCH_SIZE, epochs=EPOCHS)

    row = {"d": d}

    if d == 1:
        t0 = time.time()
        cos_price = cos_price_from_config(cfg)
        cos_time = time.time() - t0
        row["ref_price"] = cos_price
        row["ref_ci"] = None
        row["mc_time"] = None
        row["label"] = "COS exact"
        print(f"COS exact price: {cos_price:.8f}  ({cos_time:.4f}s)")
    else:
        n_mc = MC_PATHS.get(d, 100_000)
        mc_result = mc_price_basket(cfg, n_paths=n_mc, batch_size=MC_BATCH)
        ci_95 = 1.96 * mc_result["std_err"]
        row["ref_price"] = mc_result["price"]
        row["ref_ci"] = ci_95
        row["mc_time"] = mc_result["elapsed_s"]
        n_mc_k = n_mc // 1_000_000 if n_mc >= 1_000_000 else n_mc // 1_000
        unit = "M" if n_mc >= 1_000_000 else "K"
        row["label"] = f"{n_mc_k}{unit} paths"
        print(f"MC reference: {mc_result['price']:.6f} ± {ci_95:.6f} "
              f"({mc_result['elapsed_s']:.1f}s)")

    # Triple-Net
    print(f"\nTraining Triple-Net  d={d} ...")
    _, losses, y0s, elapsed = solver_triple.train(cfg, verbose=True)
    y0 = y0s[-1]
    row["y0"] = y0
    row["dl_time_total"] = elapsed
    row["dl_time_per1k"] = elapsed * 1000 / EPOCHS

    return row


def print_scaling_table(rows):
    """Print the unified scaling table matching the thesis format."""
    w = 100
    print(f"\n{'='*w}")
    print(f"{'Dimension (d)':<20} {'MC Benchmark (95% CI)':>24} "
          f"{'Our Model (Y0)':>16} {'MC Time (s)':>12} {'Our Time (per 1k ep, s)':>24}")
    print(f"{'-'*w}")
    for r in rows:
        d_str = f"{r['d']} ({r['label']})"

        if r["ref_ci"] is not None:
            mc_str = f"{r['ref_price']:.4f} +/- {r['ref_ci']:.4f}"
        else:
            mc_str = f"{r['ref_price']:.4f}"

        mc_time_str = f"{r['mc_time']:.1f}" if r["mc_time"] is not None else "-"

        print(f"{d_str:<20} {mc_str:>24} "
              f"{r['y0']:>16.4f} {mc_time_str:>12} {r['dl_time_per1k']:>24.1f}")
    print(f"{'='*w}")


def plot_time_vs_dim(rows, save_dir="figs"):
    """Computation time vs dimension d."""
    os.makedirs(save_dir, exist_ok=True)

    mc_rows = [r for r in rows if r["mc_time"] is not None]
    mc_dims = [r["d"] for r in mc_rows]
    mc_times = [r["mc_time"] for r in mc_rows]

    all_dims = [r["d"] for r in rows]
    dl_times = [r["dl_time_per1k"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(mc_dims, mc_times, "s--", label="MC", linewidth=2)
    ax.plot(all_dims, dl_times, "o-", label="Triple-Net (ours, per 1k epochs)", linewidth=2)

    ax.set_xlabel("Dimension d", fontsize=13)
    ax.set_ylabel("Computation Time (s)", fontsize=13)
    ax.set_title("Computation Time vs Dimension", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.4)

    fig.tight_layout()
    path = os.path.join(save_dir, "phase2_time_vs_dim.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_relerr_vs_dim(rows, save_dir="figs"):
    """Relative error vs dimension d."""
    os.makedirs(save_dir, exist_ok=True)
    dims = [r["d"] for r in rows]
    relerrs = [abs(r["y0"] - r["ref_price"]) / abs(r["ref_price"])
               if r["ref_price"] != 0 else float("inf")
               for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dims, relerrs, "o-", label="Triple-Net (ours)", linewidth=2)

    ax.set_xlabel("Dimension d", fontsize=13)
    ax.set_ylabel("Relative Error", fontsize=13)
    ax.set_title("Relative Error vs Dimension", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.4)

    fig.tight_layout()
    path = os.path.join(save_dir, "phase2_relerr_vs_dim.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    # -------------------------------------------------------------------
    # Logging: tee stdout/stderr to a dated log file in this directory
    # -------------------------------------------------------------------
    script_dir = os.path.dirname(__file__)
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d")
    log_path = os.path.join(script_dir, f"{script_name}_{timestamp}.log")

    class _Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()

        def flush(self):
            for s in self.streams:
                s.flush()

    _log_file = open(log_path, "w", buffering=1, encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, _log_file)
    sys.stderr = _Tee(sys.stderr, _log_file)
    print(f"Logging to {log_path}")

    rows = []
    for d in DIMS:
        row = run_for_dim(d)
        rows.append(row)

    print_scaling_table(rows)
    plot_time_vs_dim(rows)
    plot_relerr_vs_dim(rows)
    print("\nPhase 2 complete.")
