"""Phase 2: High-Dimensional Scaling Test (d = 10, 50, 100, 200).

Purpose
-------
Demonstrate that the deep learning algorithm maintains linear time
complexity O(d * N) as dimensions grow, while MC exhibits exponential /
steep cost growth.

Benchmark: large-scale MC with antithetic variates.

Outputs
-------
  - Table of prices, relative errors, and computation times
  - **Computation time vs dimension** trend chart
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from src.config import BatesConfig
from src.mc_bates import mc_price_basket
from src import solver_triple, solver_dual, solver_single


# -----------------------------------------------------------------------
# Experiment settings
# -----------------------------------------------------------------------
DIMS = [10, 50, 100, 200]
EPOCHS = 3000
N_STEPS = 100
BATCH_SIZE = 1024

MC_PATHS = {
    10:  2_000_000,
    50:  1_000_000,
    100: 1_000_000,
    200: 500_000,
}
MC_BATCH = 50_000


def run_for_dim(d):
    """Run all methods for a single dimension."""
    print(f"\n{'#'*60}")
    print(f"# d = {d}")
    print(f"{'#'*60}")

    cfg = BatesConfig(d=d, N=N_STEPS, M=BATCH_SIZE, epochs=EPOCHS)

    # MC benchmark
    n_mc = MC_PATHS.get(d, 500_000)
    mc_result = mc_price_basket(cfg, n_paths=n_mc, batch_size=MC_BATCH)
    ref_price = mc_result["price"]
    print(f"MC reference: {ref_price:.6f} ± {mc_result['std_err']:.6f} "
          f"({mc_result['elapsed_s']:.1f}s)")

    # DL models
    row = {"d": d, "mc_price": ref_price, "mc_time": mc_result["elapsed_s"],
           "mc_stderr": mc_result["std_err"]}

    for tag, trainer in [
        ("triple", solver_triple.train),
        ("dual",   solver_dual.train),
        ("single", solver_single.train),
    ]:
        print(f"\nTraining {tag}-net  d={d} ...")
        _, losses, y0s, elapsed = trainer(cfg, verbose=True)
        y0 = y0s[-1]
        row[f"{tag}_price"] = y0
        row[f"{tag}_time"]  = elapsed
        row[f"{tag}_relerr"] = abs(y0 - ref_price) / abs(ref_price) if ref_price else float("inf")

    return row


def print_summary_table(rows):
    header = (f"{'d':>4} | {'MC Price':>10} {'MC Time':>8} | "
              f"{'Triple':>10} {'Time':>7} {'RelErr':>8} | "
              f"{'Dual':>10} {'Time':>7} {'RelErr':>8} | "
              f"{'Single':>10} {'Time':>7} {'RelErr':>8}")
    print(f"\n{'='*len(header)}")
    print(header)
    print(f"{'-'*len(header)}")
    for r in rows:
        line = (f"{r['d']:>4} | {r['mc_price']:>10.4f} {r['mc_time']:>7.1f}s | "
                f"{r['triple_price']:>10.4f} {r['triple_time']:>6.1f}s {r['triple_relerr']:>8.4f} | "
                f"{r['dual_price']:>10.4f} {r['dual_time']:>6.1f}s {r['dual_relerr']:>8.4f} | "
                f"{r['single_price']:>10.4f} {r['single_time']:>6.1f}s {r['single_relerr']:>8.4f}")
        print(line)
    print(f"{'='*len(header)}")


def plot_time_vs_dim(rows, save_dir="figs"):
    """Key figure: computation time vs dimension d."""
    os.makedirs(save_dir, exist_ok=True)
    dims = [r["d"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dims, [r["mc_time"]     for r in rows], "s--", label="MC", linewidth=2)
    ax.plot(dims, [r["triple_time"] for r in rows], "o-",  label="Triple-Net (ours)", linewidth=2)
    ax.plot(dims, [r["dual_time"]   for r in rows], "^-",  label="Dual-Net",   linewidth=2)
    ax.plot(dims, [r["single_time"] for r in rows], "v-",  label="Single-Net", linewidth=2)

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

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dims, [r["triple_relerr"] for r in rows], "o-",  label="Triple-Net (ours)", linewidth=2)
    ax.plot(dims, [r["dual_relerr"]   for r in rows], "^-",  label="Dual-Net",   linewidth=2)
    ax.plot(dims, [r["single_relerr"] for r in rows], "v-",  label="Single-Net", linewidth=2)

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
    rows = []
    for d in DIMS:
        row = run_for_dim(d)
        rows.append(row)

    print_summary_table(rows)
    plot_time_vs_dim(rows)
    plot_relerr_vs_dim(rows)
    print("\nPhase 2 complete.")
