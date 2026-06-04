"""MLP vs XNet vs Hybrid comparison for the SVJDM basket solver.

Purpose
-------
Compare three policy-network choices under identical Bates / SVJDM dynamics:

  - Triple-Net MLP    : ZS, Zv, U all use MLP branches
  - Triple-Net XNet   : ZS, Zv, U all use XNet branches
  - Hybrid XNet/MLP   : ZS and Zv use XNet, U uses MLP

The hybrid architecture uses XNet for smoother diffusion exposures and keeps
an MLP for the jump compensator, whose discontinuous driver is less naturally
matched to the smooth XNet basis.

Outputs
-------
  - Summary table  (final Y0, final loss, parameter count, training time)
  - Convergence curves for loss and Y0
  - Saved to figs/compare_mlp_xnet_hybrid_d50.png
"""

import datetime
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt

from src.config import BatesConfig
from src import solver_hybrid, solver_triple, solver_xnet


# -----------------------------------------------------------------------
# Experiment settings
# -----------------------------------------------------------------------
DIM = 50
EPOCHS = 3000
N_STEPS = 100
BATCH_SIZE = 1024
XNET_BASIS = DIM

MODELS = [
    ("Triple-Net MLP", solver_triple.train),
    ("Triple-Net XNet", solver_xnet.train),
    ("Hybrid XNet/MLP", solver_hybrid.train),
]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_comparison():
    """Train all three architectures at d=50 with matched settings."""
    print(f"\n{'#'*60}")
    print(f"# MLP vs XNet vs Hybrid Comparison — d = {DIM}")
    print(f"{'#'*60}")

    cfg = BatesConfig(
        d=DIM,
        N=N_STEPS,
        M=BATCH_SIZE,
        epochs=EPOCHS,
        xnet_basis=XNET_BASIS,
    )

    results = []
    for name, trainer in MODELS:
        print(f"\n{'='*60}")
        print(f"Training {name}  d={DIM}")
        print(f"{'='*60}")
        model, losses, y0s, elapsed = trainer(cfg, verbose=True)
        results.append({
            "name": name,
            "y0": y0s[-1],
            "final_loss": losses[-1],
            "params": count_parameters(model),
            "elapsed": elapsed,
            "losses": losses,
            "y0s": y0s,
        })

    print(f"\n{'='*78}")
    print(f"{'Model':<18} {'Final Y0':>12} {'Final Loss':>14} {'Params':>12} {'Time(s)':>10}")
    print(f"{'-'*78}")
    for r in results:
        print(f"{r['name']:<18} {r['y0']:>12.6f} {r['final_loss']:>14.4e} "
              f"{r['params']:>12,d} {r['elapsed']:>10.1f}")
    print(f"{'='*78}")

    return results


def plot_comparison(results, save_dir="figs"):
    """Plot loss and Y0 convergence curves for all compared architectures."""
    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for r in results:
        ax1.plot(r["losses"], label=r["name"])
        ax2.plot(r["y0s"], label=r["name"])

    ax1.set_yscale("log")
    ax1.set(title=f"Training Loss (d={DIM})", xlabel="Epoch", ylabel="MSE")
    ax1.legend(fontsize=11)
    ax1.grid(True, which="both", alpha=0.4)

    ax2.set(title=f"Y0 Convergence (d={DIM})", xlabel="Epoch", ylabel="Price")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.4)

    fig.tight_layout()
    path = os.path.join(save_dir, f"compare_mlp_xnet_hybrid_d{DIM}.png")
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

    comparison_results = run_comparison()
    plot_comparison(comparison_results)
    print("\nMLP vs XNet vs Hybrid comparison complete.")
