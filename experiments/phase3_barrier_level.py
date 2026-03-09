"""Phase 3: Barrier Level Sensitivity & Price Discount (Experiment A).

Purpose
-------
Compare option prices under different barrier levels to verify that the
deep learning network correctly learns the price discount imposed by the
down-and-out boundary condition.

Setup
-----
  - Down-and-out basket call: knocked out if ANY asset S_i < B (Rebate=0).
  - Fixed: d=50, S0=1.0, K=1.0, lambda=1.0 (same global params as Phase 1).
  - Control variable B (barrier level):
      B=0.0  -> equivalent to a standard European option (baseline)
      B=0.6  -> hard to knock out, price slightly below European
      B=0.7  -> moderate
      B=0.8  -> easy to knock out
      B=0.9  -> very easy to knock out, price very low

Outputs
-------
  - Table: Barrier | Y0 (Price) | Survival Rate | Time (s)
  - Monotonically decreasing line chart: B vs Y0
"""

import sys
import os
import copy
import time
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.config import BatesConfig
from src import solver_triple
from src import solver_triple_barrier
from src.bates_sde import sample_noises
from src.solver_triple_barrier import generate_paths_with_barrier

# -----------------------------------------------------------------------
# Experiment settings
# -----------------------------------------------------------------------
BARRIERS = [0.0, 0.6, 0.7, 0.8, 0.9]
EPOCHS = 3000
N_STEPS = 100
BATCH_SIZE = 1024
DIM = 50


def evaluate_survival_rate(cfg, n_eval=10, eval_batch=8192):
    """Evaluate average survival rate over multiple large batches."""
    rates = []
    for _ in range(n_eval):
        eval_cfg = copy.copy(cfg)
        eval_cfg.M = eval_batch
        with torch.no_grad():
            dW_S, dW_v, _dW_v_tilde, dN, dN_tilde, J = sample_noises(eval_cfg, cfg.device)
            _, alive = generate_paths_with_barrier(
                eval_cfg, cfg.device, dW_S, dW_v, dN, J
            )
        rates.append(alive[-1].mean().item())
    return float(np.mean(rates))


def run_for_barrier(B):
    """Train and evaluate for a single barrier level."""
    print(f"\n{'#'*60}")
    print(f"# Barrier B = {B}")
    print(f"{'#'*60}")

    cfg = BatesConfig(
        d=DIM, N=N_STEPS, M=BATCH_SIZE, epochs=EPOCHS, barrier=B
    )

    if B == 0.0:
        _, losses, y0s, elapsed = solver_triple.train(cfg, verbose=True)
        surv_rate = 1.0
    else:
        _, losses, y0s, elapsed = solver_triple_barrier.train(cfg, verbose=True)
        surv_rate = evaluate_survival_rate(cfg)
        print(f"  Evaluated survival rate: {surv_rate:.4f}")

    return {
        "B": B,
        "y0": y0s[-1],
        "surv_rate": surv_rate,
        "elapsed": elapsed,
        "losses": losses,
        "y0s": y0s,
    }


def print_table(rows):
    """Print results table."""
    w = 72
    print(f"\n{'='*w}")
    print(f"{'Barrier (B)':<16} {'Price (Y0)':>14} "
          f"{'Survival Rate':>16} {'Time (s)':>12}")
    print(f"{'-'*w}")
    for r in rows:
        b_str = f"{r['B']:.1f}" if r["B"] > 0 else "0.0 (European)"
        print(f"{b_str:<16} {r['y0']:>14.6f} "
              f"{r['surv_rate']:>15.2%} {r['elapsed']:>12.1f}")
    print(f"{'='*w}")


def plot_barrier_vs_price(rows, save_dir="figs"):
    """Line chart: barrier level (x) vs option price Y0 (y)."""
    os.makedirs(save_dir, exist_ok=True)

    bs = [r["B"] for r in rows]
    y0s = [r["y0"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bs, y0s, "o-", linewidth=2, markersize=8, color="C0")

    for b, y in zip(bs, y0s):
        ax.annotate(f"{y:.4f}", (b, y), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=9)

    ax.set_xlabel("Barrier Level B", fontsize=13)
    ax.set_ylabel("Option Price $Y_0$", fontsize=13)
    ax.set_title("Down-and-Out Basket Call: Price vs Barrier Level (d=50)",
                 fontsize=14)
    ax.set_xticks(bs)
    ax.grid(True, alpha=0.4)

    fig.tight_layout()
    path = os.path.join(save_dir, "phase3_barrier_level.png")
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
    for B in BARRIERS:
        row = run_for_barrier(B)
        rows.append(row)

    print_table(rows)
    plot_barrier_vs_price(rows)
    print("\nPhase 3 (Barrier Level Sensitivity) complete.")
