"""Training-Strategy Ablation Study (d = 50).

Purpose
-------
Isolate the contributions of Nesterov momentum and learning-rate scheduling
to training convergence, using a 2x2 controlled-variable design:

  A) Adam  + Constant LR   — baseline
  B) NAdam + Constant LR   — Nesterov momentum accelerates early convergence
  C) Adam  + Scheduled LR  — LR decay suppresses late-stage MSE oscillation
  D) NAdam + Scheduled LR  — best of both: fastest convergence + smoothest tail

All four groups share the same network architecture (Triple-Net) and the
same random seed so that only the optimiser / schedule differs.

Outputs
-------
  - Summary table  (final Y0, final Loss, training time)
  - Single log-scale Loss plot with 4 curves
      colour  -> optimiser  (blue = NAdam, red = Adam)
      line    -> schedule   (solid = Scheduled, dashed = Constant)
  - Saved to figs/ablation_structure_d50.png
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from src.config import BatesConfig
from src.bates_sde import sample_noises, generate_paths
from src.solver_triple import TripleSolver
from src.utils import set_seed


# -----------------------------------------------------------------------
# Experiment settings
# -----------------------------------------------------------------------
DIM = 50
EPOCHS = 3000
N_STEPS = 100
BATCH_SIZE = 1024

GROUPS = [
    ("Adam + Constant LR",   optim.Adam,  False),
    ("NAdam + Constant LR",  optim.NAdam, False),
    ("Adam + Scheduled LR",  optim.Adam,  True),
    ("NAdam + Scheduled LR", optim.NAdam, True),
]


def train_variant(cfg, optimizer_cls, use_schedule, verbose=True):
    """Train Triple-Net with a specific optimiser / LR-schedule combination."""
    set_seed(cfg.seed)
    dev = cfg.device

    model = TripleSolver(cfg).to(dev)
    opt = optimizer_cls(model.parameters(), lr=cfg.lr)
    sched = (optim.lr_scheduler.MultiStepLR(opt, milestones=[1500, 2500], gamma=0.1)
             if use_schedule else None)

    losses, y0s = [], []
    t0 = time.time()

    for ep in range(cfg.epochs):
        dW_S, dW_v, dW_v_tilde, dN, dN_tilde, J = sample_noises(cfg, dev)
        with torch.no_grad():
            X = generate_paths(cfg, dev, dW_S, dW_v, dN, J)

        model.train()
        Y_pred, payoff = model(X, dW_S, dW_v_tilde, dN_tilde)
        loss = ((Y_pred - payoff) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()

        losses.append(loss.item())
        y0s.append(model.Y0.item())

        if verbose and (ep % 100 == 0 or ep == cfg.epochs - 1):
            print(f"  Epoch {ep:4d} | Loss {loss.item():.4e} "
                  f"| Y0 {model.Y0.item():.6f} | {time.time() - t0:.1f}s")

    elapsed = time.time() - t0
    if verbose:
        print(f"Training complete.  Y0 = {model.Y0.item():.6f}  ({elapsed:.1f}s)")
    return {"losses": losses, "y0s": y0s, "elapsed": elapsed}


# -----------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------

STYLE_MAP = {
    #                          colour   linestyle
    ("Adam",  False):         ("#d62728", "--"),    # red  dashed
    ("Adam",  True):          ("#d62728", "-"),     # red  solid
    ("NAdam", False):         ("#1f77b4", "--"),    # blue dashed
    ("NAdam", True):          ("#1f77b4", "-"),     # blue solid
}


def _style_key(name, use_schedule):
    opt_key = "NAdam" if "NAdam" in name else "Adam"
    return (opt_key, use_schedule)


def plot_loss_curves(results, save_dir="figs"):
    """4-curve log-scale Loss plot."""
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, use_sched, res in results:
        color, ls = STYLE_MAP[_style_key(name, use_sched)]
        ax.plot(res["losses"], label=name, color=color, linestyle=ls, linewidth=1.5)

    ax.set_yscale("log")
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Training Loss (MSE)", fontsize=13)
    ax.set_title(f"Training-Strategy Ablation (d={DIM})", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(save_dir, "ablation_structure_d50.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def run_ablation():
    print(f"\n{'#'*60}")
    print(f"# Training-Strategy Ablation — d = {DIM}")
    print(f"{'#'*60}")

    cfg = BatesConfig(d=DIM, N=N_STEPS, M=BATCH_SIZE, epochs=EPOCHS)

    results = []
    for name, opt_cls, use_sched in GROUPS:
        sched_tag = "Scheduled" if use_sched else "Constant"
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        res = train_variant(cfg, opt_cls, use_sched, verbose=True)
        results.append((name, use_sched, res))

    # Summary table
    w = 72
    print(f"\n{'='*w}")
    print(f"{'Group':<28} {'Final Y0':>10} {'Final Loss':>12} {'Time(s)':>10}")
    print(f"{'-'*w}")
    for name, _, res in results:
        print(f"{name:<28} {res['y0s'][-1]:>10.6f} "
              f"{res['losses'][-1]:>12.4e} {res['elapsed']:>10.1f}")
    print(f"{'='*w}")

    return results


if __name__ == "__main__":
    results = run_ablation()
    plot_loss_curves(results)
    print("\nTraining-strategy ablation complete.")
