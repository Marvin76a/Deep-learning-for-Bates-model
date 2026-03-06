"""Phase 1: Low-Dimensional Ground Truth Test (d = 1, 3).

Purpose
-------
Prove that the three-branch neural network is mathematically unbiased by
comparing against:
  - COS Fourier transform   (d=1, near-exact analytic benchmark)
  - Large-scale Monte Carlo  (d=3, 10M-path "relative truth")
  - Dual-network Deep BSDE   (ablation baseline)
  - Single-network Deep BSDE (Han et al. 2018 baseline)

Outputs
-------
  - Table of prices, absolute errors, and relative errors
  - Convergence curves for all DL models
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from src.config import BatesConfig
from src.cos_bates import cos_price_from_config
from src.mc_bates import mc_price_basket
from src import solver_triple, solver_dual, solver_single


# -----------------------------------------------------------------------
# Shared training parameters for low-dim experiments
# -----------------------------------------------------------------------
EPOCHS = 3000
N_STEPS = 100
BATCH_SIZE = 1024
MC_PATHS_REF = 10_000_000


def run_dl_models(cfg):
    """Train all three DL solvers and return (name, Y0, elapsed, losses)."""
    results = []
    for name, trainer in [
        ("Triple-Net", solver_triple.train),
        ("Dual-Net",   solver_dual.train),
        ("Single-Net", solver_single.train),
    ]:
        print(f"\n{'='*60}")
        print(f"Training {name}  d={cfg.d}")
        print(f"{'='*60}")
        _, losses, y0s, elapsed = trainer(cfg, verbose=True)
        results.append({
            "name": name,
            "y0": y0s[-1],
            "elapsed": elapsed,
            "losses": losses,
            "y0s": y0s,
        })
    return results


def run_phase1_d1():
    """d=1: Compare DL models against COS (exact benchmark)."""
    print("\n" + "#" * 60)
    print("# Phase 1 — d = 1 (COS exact benchmark)")
    print("#" * 60)

    cfg = BatesConfig(d=1, N=N_STEPS, M=BATCH_SIZE, epochs=EPOCHS)

    # COS benchmark
    t0 = time.time()
    cos_price = cos_price_from_config(cfg)
    cos_time = time.time() - t0
    print(f"\nCOS reference price: {cos_price:.8f}  ({cos_time:.4f}s)")

    # MC cross-check
    mc_result = mc_price_basket(cfg, n_paths=MC_PATHS_REF, batch_size=100_000)
    print(f"MC  cross-check:    {mc_result['price']:.8f} ± {mc_result['std_err']:.6f}")

    # DL models
    dl_results = run_dl_models(cfg)

    # Summary table
    print(f"\n{'='*70}")
    print(f"{'Method':<16} {'Price':>12} {'AbsErr':>12} {'RelErr':>12} {'Time(s)':>10}")
    print(f"{'-'*70}")
    print(f"{'COS (exact)':<16} {cos_price:>12.6f} {'—':>12} {'—':>12} {cos_time:>10.3f}")
    print(f"{'MC (10M)':<16} {mc_result['price']:>12.6f} "
          f"{abs(mc_result['price']-cos_price):>12.6f} "
          f"{abs(mc_result['price']-cos_price)/cos_price:>12.6f} "
          f"{mc_result['elapsed_s']:>10.1f}")
    for r in dl_results:
        ae = abs(r["y0"] - cos_price)
        re = ae / abs(cos_price) if cos_price != 0 else float("inf")
        print(f"{r['name']:<16} {r['y0']:>12.6f} {ae:>12.6f} {re:>12.6f} {r['elapsed']:>10.1f}")
    print(f"{'='*70}")

    return cos_price, dl_results


def run_phase1_d3():
    """d=3: Compare DL models against high-quality MC."""
    print("\n" + "#" * 60)
    print("# Phase 1 — d = 3 (MC benchmark, 10M paths)")
    print("#" * 60)

    cfg = BatesConfig(d=3, N=N_STEPS, M=BATCH_SIZE, epochs=EPOCHS)

    # MC benchmark
    mc_result = mc_price_basket(cfg, n_paths=MC_PATHS_REF, batch_size=100_000)
    ref_price = mc_result["price"]
    print(f"\nMC reference price: {ref_price:.8f} ± {mc_result['std_err']:.6f}")

    # DL models
    dl_results = run_dl_models(cfg)

    # Summary table
    print(f"\n{'='*70}")
    print(f"{'Method':<16} {'Price':>12} {'AbsErr':>12} {'RelErr':>12} {'Time(s)':>10}")
    print(f"{'-'*70}")
    print(f"{'MC (10M)':<16} {ref_price:>12.6f} {'—':>12} {'—':>12} "
          f"{mc_result['elapsed_s']:>10.1f}")
    for r in dl_results:
        ae = abs(r["y0"] - ref_price)
        re = ae / abs(ref_price) if ref_price != 0 else float("inf")
        print(f"{r['name']:<16} {r['y0']:>12.6f} {ae:>12.6f} {re:>12.6f} {r['elapsed']:>10.1f}")
    print(f"{'='*70}")

    return ref_price, dl_results


def plot_convergence(dl_results_list, dims, save_dir="figs"):
    """Plot Y0 convergence curves for each dimension."""
    os.makedirs(save_dir, exist_ok=True)
    for dim, results in zip(dims, dl_results_list):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        for r in results:
            ax1.plot(r["losses"], label=r["name"])
            ax2.plot(r["y0s"],    label=r["name"])
        ax1.set_yscale("log")
        ax1.set(title=f"Training Loss (d={dim})", xlabel="Epoch", ylabel="MSE")
        ax1.legend()
        ax1.grid(True, which="both", alpha=0.4)
        ax2.set(title=f"Y0 Convergence (d={dim})", xlabel="Epoch", ylabel="Price")
        ax2.legend()
        ax2.grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"phase1_d{dim}.png"), dpi=150)
        plt.close(fig)
        print(f"Saved: {save_dir}/phase1_d{dim}.png")


if __name__ == "__main__":
    _, dl1 = run_phase1_d1()
    _, dl3 = run_phase1_d3()
    plot_convergence([dl1, dl3], [1, 3])
    print("\nPhase 1 complete.")
