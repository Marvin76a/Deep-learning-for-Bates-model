"""Robustness Study (d = 50).

Purpose
-------
Verify that the Triple-Net Deep BSDE solver is stable across:

  Part 1 — Random-seed test (anti cherry-picking):
      Run the full pipeline with 5 different seeds and report the
      mean and standard deviation of Y_0.

  Part 2 — Batch-size perturbation (convergence tolerance):
      Vary M in {512, 1024, 2048} and confirm Final Loss stays
      at the 1e-3 order of magnitude.

Outputs
-------
  - Summary tables for both parts (printed + logged)
  - figs/robustness_seed.png   — Y0 convergence curves (5 seeds)
  - figs/robustness_batch.png  — Loss convergence curves (3 batch sizes)
"""

import sys
import os
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from src.config import BatesConfig
from src import solver_triple

# -----------------------------------------------------------------------
# Experiment settings
# -----------------------------------------------------------------------
DIM = 50
EPOCHS = 3000
N_STEPS = 100
BATCH_SIZE = 1024

SEEDS = [42, 123, 2024, 7, 9999]
BATCH_SIZES = [512, 1024, 2048]


# ===================================================================
# Part 1 — Random-seed robustness
# ===================================================================
def run_seed_robustness():
    """Train Triple-Net with 5 seeds; report Y_0 mean & std."""
    print(f"\n{'#'*60}")
    print(f"# Part 1: Random-Seed Robustness — d = {DIM}")
    print(f"{'#'*60}")

    records = []
    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"Seed = {seed}")
        print(f"{'='*60}")
        cfg = BatesConfig(d=DIM, N=N_STEPS, M=BATCH_SIZE,
                          epochs=EPOCHS, seed=seed)
        _, losses, y0s, elapsed = solver_triple.train(cfg, verbose=True)
        records.append({
            "seed": seed,
            "y0": y0s[-1],
            "loss": losses[-1],
            "elapsed": elapsed,
            "y0s": y0s,
            "losses": losses,
        })

    y0_arr = np.array([r["y0"] for r in records])
    y0_mean = y0_arr.mean()
    y0_std = y0_arr.std()

    w = 60
    print(f"\n{'='*w}")
    print(f"{'Seed':>8} {'Final Y0':>12} {'Final Loss':>14} {'Time(s)':>10}")
    print(f"{'-'*w}")
    for r in records:
        print(f"{r['seed']:>8d} {r['y0']:>12.6f} {r['loss']:>14.4e} "
              f"{r['elapsed']:>10.1f}")
    print(f"{'-'*w}")
    print(f"{'Mean':>8s} {y0_mean:>12.6f}")
    print(f"{'Std':>8s} {y0_std:>12.6f}")
    print(f"{'='*w}")

    return records


# ===================================================================
# Part 2 — Batch-size perturbation
# ===================================================================
def run_batch_robustness():
    """Train Triple-Net with M in {512, 1024, 2048}; check loss stability."""
    print(f"\n{'#'*60}")
    print(f"# Part 2: Batch-Size Perturbation — d = {DIM}")
    print(f"{'#'*60}")

    records = []
    for bs in BATCH_SIZES:
        print(f"\n{'='*60}")
        print(f"Batch Size M = {bs}")
        print(f"{'='*60}")
        cfg = BatesConfig(d=DIM, N=N_STEPS, M=bs,
                          epochs=EPOCHS, seed=42)
        _, losses, y0s, elapsed = solver_triple.train(cfg, verbose=True)
        records.append({
            "batch_size": bs,
            "y0": y0s[-1],
            "loss": losses[-1],
            "elapsed": elapsed,
            "y0s": y0s,
            "losses": losses,
        })

    w = 60
    print(f"\n{'='*w}")
    print(f"{'Batch Size':>12} {'Final Y0':>12} {'Final Loss':>14} {'Time(s)':>10}")
    print(f"{'-'*w}")
    for r in records:
        print(f"{r['batch_size']:>12d} {r['y0']:>12.6f} {r['loss']:>14.4e} "
              f"{r['elapsed']:>10.1f}")
    print(f"{'='*w}")

    return records


# ===================================================================
# Plots
# ===================================================================
def plot_seed(records, save_dir="figs"):
    """Y0 convergence curves for each seed."""
    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for r in records:
        label = f"seed={r['seed']}"
        ax1.plot(r["losses"], label=label)
        ax2.plot(r["y0s"], label=label)

    ax1.set_yscale("log")
    ax1.set(title=f"Training Loss — Seed Robustness (d={DIM})",
            xlabel="Epoch", ylabel="MSE")
    ax1.legend(fontsize=9)
    ax1.grid(True, which="both", alpha=0.4)

    ax2.set(title=f"Y0 Convergence — Seed Robustness (d={DIM})",
            xlabel="Epoch", ylabel="Price")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.4)

    fig.tight_layout()
    path = os.path.join(save_dir, "robustness_seed.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_batch(records, save_dir="figs"):
    """Loss convergence curves for each batch size."""
    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for r in records:
        label = f"M={r['batch_size']}"
        ax1.plot(r["losses"], label=label)
        ax2.plot(r["y0s"], label=label)

    ax1.set_yscale("log")
    ax1.set(title=f"Training Loss — Batch-Size Perturbation (d={DIM})",
            xlabel="Epoch", ylabel="MSE")
    ax1.legend(fontsize=11)
    ax1.grid(True, which="both", alpha=0.4)

    ax2.set(title=f"Y0 Convergence — Batch-Size Perturbation (d={DIM})",
            xlabel="Epoch", ylabel="Price")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.4)

    fig.tight_layout()
    path = os.path.join(save_dir, "robustness_batch.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
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

    seed_records = run_seed_robustness()
    plot_seed(seed_records)

    batch_records = run_batch_robustness()
    plot_batch(batch_records)

    print("\nRobustness study complete.")
