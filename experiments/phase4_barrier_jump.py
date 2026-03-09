"""Phase 4: Impact of Jumps on Unexpected Knock-outs (Experiment B).

Purpose
-------
Verify that the algorithm (especially Brownian-bridge gap interpolation)
accurately captures "unexpected knock-outs" caused by discrete jumps and
measures their severe impact on option prices.

Setup
-----
  - Down-and-out basket call with a moderate barrier B=0.7.
  - Fixed: d=50, S0=1.0, K=1.0.
  - Control variable: jump intensity lambda
      lambda=0.0  -> pure Heston SV model, no jumps
      lambda=0.5  -> low-frequency jumps
      lambda=1.0  -> standard setting
      lambda=2.0  -> high-frequency extreme market

Outputs
-------
  - Table: lambda | Y0 (Price) | Knock-out Rate | Time (s)
  - Dual-panel chart: lambda vs Y0 and lambda vs knock-out rate
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
from src import solver_triple_barrier
from src.bates_sde import sample_noises
from src.solver_triple_barrier import generate_paths_with_barrier

# -----------------------------------------------------------------------
# Experiment settings
# -----------------------------------------------------------------------
LAMBDAS = [0.0, 0.5, 1.0, 2.0]
BARRIER = 0.7
EPOCHS = 3000
N_STEPS = 100
BATCH_SIZE = 1024
DIM = 50


def evaluate_knockout_rate(cfg, n_eval=10, eval_batch=8192):
    """Evaluate average knock-out rate over multiple large batches."""
    rates = []
    for _ in range(n_eval):
        eval_cfg = copy.copy(cfg)
        eval_cfg.M = eval_batch
        with torch.no_grad():
            dW_S, dW_v, _dW_v_tilde, dN, dN_tilde, J = sample_noises(eval_cfg, cfg.device)
            _, alive = generate_paths_with_barrier(
                eval_cfg, cfg.device, dW_S, dW_v, dN, J
            )
        rates.append(1.0 - alive[-1].mean().item())
    return float(np.mean(rates))


def run_for_lambda(lam):
    """Train and evaluate for a single jump intensity."""
    print(f"\n{'#'*60}")
    print(f"# lambda = {lam}")
    print(f"{'#'*60}")

    cfg = BatesConfig(
        d=DIM, N=N_STEPS, M=BATCH_SIZE, epochs=EPOCHS,
        barrier=BARRIER, lambda_=lam,
    )

    _, losses, y0s, elapsed = solver_triple_barrier.train(cfg, verbose=True)

    ko_rate = evaluate_knockout_rate(cfg)
    print(f"  Evaluated knock-out rate: {ko_rate:.4f}")

    return {
        "lam": lam,
        "y0": y0s[-1],
        "ko_rate": ko_rate,
        "elapsed": elapsed,
        "losses": losses,
        "y0s": y0s,
    }


def print_table(rows):
    """Print results table."""
    w = 72
    print(f"\n{'='*w}")
    print(f"{'Lambda':>10} {'Price (Y0)':>14} "
          f"{'Knock-out Rate':>18} {'Time (s)':>12}")
    print(f"{'-'*w}")
    for r in rows:
        print(f"{r['lam']:>10.1f} {r['y0']:>14.6f} "
              f"{r['ko_rate']:>17.2%} {r['elapsed']:>12.1f}")
    print(f"{'='*w}")


def plot_jump_impact(rows, save_dir="figs"):
    """Dual-panel chart: lambda vs price (left) and lambda vs KO rate (right)."""
    os.makedirs(save_dir, exist_ok=True)

    lams = [r["lam"] for r in rows]
    y0s = [r["y0"] for r in rows]
    ko_rates = [r["ko_rate"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(lams, y0s, "o-", linewidth=2, markersize=8, color="C0")
    for l, y in zip(lams, y0s):
        ax1.annotate(f"{y:.4f}", (l, y), textcoords="offset points",
                     xytext=(0, 12), ha="center", fontsize=9)
    ax1.set_xlabel("Jump Intensity $\\lambda$", fontsize=13)
    ax1.set_ylabel("Option Price $Y_0$", fontsize=13)
    ax1.set_title("Price vs Jump Intensity (B=0.7, d=50)", fontsize=14)
    ax1.set_xticks(lams)
    ax1.grid(True, alpha=0.4)

    ax2.plot(lams, [r * 100 for r in ko_rates], "s-", linewidth=2,
             markersize=8, color="C3")
    for l, k in zip(lams, ko_rates):
        ax2.annotate(f"{k:.1%}", (l, k * 100), textcoords="offset points",
                     xytext=(0, 12), ha="center", fontsize=9)
    ax2.set_xlabel("Jump Intensity $\\lambda$", fontsize=13)
    ax2.set_ylabel("Knock-out Rate (%)", fontsize=13)
    ax2.set_title("Knock-out Rate vs Jump Intensity (B=0.7, d=50)", fontsize=14)
    ax2.set_xticks(lams)
    ax2.grid(True, alpha=0.4)

    fig.tight_layout()
    path = os.path.join(save_dir, "phase4_barrier_jump.png")
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
    for lam in LAMBDAS:
        row = run_for_lambda(lam)
        rows.append(row)

    print_table(rows)
    plot_jump_impact(rows)
    print("\nPhase 4 (Jump Impact on Knock-outs) complete.")
