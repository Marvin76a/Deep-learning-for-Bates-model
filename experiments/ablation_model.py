"""Model Ablation Study (d = 50).

Purpose
-------
Compare three network architectures under the same SVJDM / Bates dynamics
to isolate the contribution of each architectural component:

  - Single-Net : Z(d+1) only          (Han et al. 2018 — no explicit jump)
  - Dual-Net   : Z(d+1) + U(1)        (diffusion + jump compensator)
  - Triple-Net : ZS(d) + Zv(1) + U(1) (ours — separated Delta / Vega / jump)

Benchmark: large-scale Monte Carlo with antithetic variates.

Outputs
-------
  - Ablation comparison table  (price, abs error, rel error, training time)
  - Convergence curves         (loss + Y0, three models on one figure)
"""

import sys
import os
import time
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from src.config import BatesConfig
from src.mc_bates import mc_price_basket
from src import solver_triple, solver_dual, solver_single


# -----------------------------------------------------------------------
# Experiment settings
# -----------------------------------------------------------------------
DIM = 50
EPOCHS = 3000
N_STEPS = 100
BATCH_SIZE = 1024
MC_PATHS = 1_000_000
MC_BATCH = 50_000

MODELS = [
    ("Triple-Net", solver_triple.train),
    ("Dual-Net",   solver_dual.train),
    ("Single-Net", solver_single.train),
]


def run_ablation():
    """Train all three architectures at d=50 and compare against MC."""
    print(f"\n{'#'*60}")
    print(f"# Model Ablation Study — d = {DIM}")
    print(f"{'#'*60}")

    cfg = BatesConfig(d=DIM, N=N_STEPS, M=BATCH_SIZE, epochs=EPOCHS)

    # MC benchmark
    mc_result = mc_price_basket(cfg, n_paths=MC_PATHS, batch_size=MC_BATCH)
    ref_price = mc_result["price"]
    print(f"\nMC reference price: {ref_price:.6f} ± {mc_result['std_err']:.6f} "
          f"({mc_result['elapsed_s']:.1f}s)")

    # Train each model
    results = []
    for name, trainer in MODELS:
        print(f"\n{'='*60}")
        print(f"Training {name}  d={DIM}")
        print(f"{'='*60}")
        _, losses, y0s, elapsed = trainer(cfg, verbose=True)
        results.append({
            "name": name,
            "y0": y0s[-1],
            "elapsed": elapsed,
            "losses": losses,
            "y0s": y0s,
        })

    # Summary table
    print(f"\n{'='*74}")
    print(f"{'Model':<16} {'Price':>12} {'AbsErr':>12} {'RelErr':>12} {'Time(s)':>10}")
    print(f"{'-'*74}")
    print(f"{'MC (ref)':<16} {ref_price:>12.6f} {'—':>12} {'—':>12} "
          f"{mc_result['elapsed_s']:>10.1f}")
    for r in results:
        ae = abs(r["y0"] - ref_price)
        re = ae / abs(ref_price) if ref_price != 0 else float("inf")
        print(f"{r['name']:<16} {r['y0']:>12.6f} {ae:>12.6f} {re:>12.6f} {r['elapsed']:>10.1f}")
    print(f"{'='*74}")

    return ref_price, results


def plot_ablation(results, save_dir="figs"):
    """Convergence curves: loss and Y0 for all three models."""
    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for r in results:
        ax1.plot(r["losses"], label=r["name"])
        ax2.plot(r["y0s"],    label=r["name"])

    ax1.set_yscale("log")
    ax1.set(title=f"Training Loss (d={DIM})", xlabel="Epoch", ylabel="MSE")
    ax1.legend(fontsize=11)
    ax1.grid(True, which="both", alpha=0.4)

    ax2.set(title=f"Y0 Convergence (d={DIM})", xlabel="Epoch", ylabel="Price")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.4)

    fig.tight_layout()
    path = os.path.join(save_dir, "ablation_model_d50.png")
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

    ref_price, results = run_ablation()
    plot_ablation(results)
    print("\nModel ablation study complete.")
