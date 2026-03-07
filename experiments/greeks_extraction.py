"""Greeks Extraction at t=0 for the d=50 Basket Option.

Purpose
-------
Demonstrate that the three-branch architecture simultaneously learns
the exact local hedging strategies (Delta and Vega) without additional
computational overhead.  The policy sub-networks directly output:
  - Delta exposure: Z_S(t=0)  (d-dim)
  - Vega  exposure: Z_v(t=0)  (scalar)

These are benchmarked against MC finite-difference estimates computed
with common random numbers (CRN) for variance reduction.

Mapping from Z to classical Greeks
-----------------------------------
  Z_S^i = (dV/dS_i) * S_i * sqrt(v)
  Z_v   = (dV/dv)   * sigma_v * sqrt(v)

Outputs
-------
  - Comparison table: MC (Finite Diff.) vs Our Model (Neural Output)
"""

import sys
import os
import time
import copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from src.config import BatesConfig
from src.bates_sde import sample_noises, generate_paths
from src import solver_triple
from src.utils import set_seed


# -----------------------------------------------------------------------
# Experiment settings
# -----------------------------------------------------------------------
DIM = 50
EPOCHS = 3000
N_STEPS = 100
BATCH_SIZE = 1024

MC_PATHS = 500_000
MC_BATCH = 25_000
EPS_S = 0.01          # S0 bump for Delta  (1% of S0=1.0)
EPS_V = 0.002         # v0 bump for Vega   (5% of v0=0.04)


# -----------------------------------------------------------------------
# Neural-network Greeks
# -----------------------------------------------------------------------

def extract_nn_greeks(model, cfg):
    """Extract Z_S(t=0) and Z_v(t=0) from the trained Triple-Net.

    Uses a batch of identical initial states for numerical stability
    with BatchNorm running statistics.
    """
    model.eval()
    dev = cfg.device
    n_eval = 256

    with torch.no_grad():
        x0 = torch.zeros(n_eval, cfg.d + 1, device=dev)
        x0[:, :cfg.d] = cfg.S0
        x0[:, cfg.d] = cfg.v0

        z_s = model.net_zs[0](x0).mean(dim=0).cpu().numpy()   # [d]
        z_v = model.net_zv[0](x0).mean().item()                # scalar

    return z_s, z_v


# -----------------------------------------------------------------------
# MC finite-difference Greeks (common random numbers)
# -----------------------------------------------------------------------

def _basket_payoff(X, cfg, dev):
    w = torch.tensor(cfg.weights, dtype=torch.float32, device=dev)
    return torch.clamp((w * X[-1, :, :cfg.d]).sum(1) - cfg.K, min=0.0)


def mc_greeks_fd(cfg, n_paths=MC_PATHS, batch_size=MC_BATCH,
                 eps_S=EPS_S, eps_v=EPS_V, seed=42, verbose=True):
    """Central-difference Greeks with common random numbers (CRN).

    For Delta: bump all S_i simultaneously by ±eps_S (exploiting symmetry).
      Sigma V_{S_i} = [V(S0+eps) - V(S0-eps)] / (2 eps)
      Sigma Z_S^i   = Sigma V_{S_i}  * S0 * sqrt(v0)

    For Vega:  bump v0 by ±eps_v.
      V_v  = [V(v0+eps) - V(v0-eps)] / (2 eps)
      Z_v  = V_v * sigma_v * sqrt(v0)
    """
    set_seed(seed)
    dev = cfg.device
    disc = np.exp(-cfg.r * cfg.T)

    delta_diffs = []
    vega_diffs = []

    remaining = n_paths
    batch_id = 0
    t0 = time.time()

    while remaining > 0:
        cur = min(batch_size, remaining)
        bcfg = copy.copy(cfg)
        bcfg.M = cur

        with torch.no_grad():
            dW_S, dW_v, dN, dN_tilde, J = sample_noises(bcfg, dev)

            # --- Delta: bump S0 for all assets ---
            bcfg_up = copy.copy(bcfg); bcfg_up.S0 = cfg.S0 + eps_S
            bcfg_dn = copy.copy(bcfg); bcfg_dn.S0 = cfg.S0 - eps_S
            X_up = generate_paths(bcfg_up, dev, dW_S, dW_v, dN, J)
            X_dn = generate_paths(bcfg_dn, dev, dW_S, dW_v, dN, J)
            p_up = _basket_payoff(X_up, cfg, dev)
            p_dn = _basket_payoff(X_dn, cfg, dev)
            delta_diffs.append(((p_up - p_dn) / (2 * eps_S)).cpu())

            # --- Vega: bump v0 ---
            bcfg_up = copy.copy(bcfg); bcfg_up.v0 = cfg.v0 + eps_v
            bcfg_dn = copy.copy(bcfg); bcfg_dn.v0 = cfg.v0 - eps_v
            X_up = generate_paths(bcfg_up, dev, dW_S, dW_v, dN, J)
            X_dn = generate_paths(bcfg_dn, dev, dW_S, dW_v, dN, J)
            p_up = _basket_payoff(X_up, cfg, dev)
            p_dn = _basket_payoff(X_dn, cfg, dev)
            vega_diffs.append(((p_up - p_dn) / (2 * eps_v)).cpu())

        remaining -= cur
        batch_id += 1
        if verbose and batch_id % 5 == 0:
            done = n_paths - remaining
            print(f"  MC Greeks batch {batch_id}: {done}/{n_paths}")

    elapsed = time.time() - t0

    delta_all = disc * torch.cat(delta_diffs).numpy()    # Σ ∂V/∂S_i
    vega_all = disc * torch.cat(vega_diffs).numpy()      # ∂V/∂v
    n = len(delta_all)
    sqrt_v0 = np.sqrt(cfg.v0)

    z_s_sum = float(np.mean(delta_all)) * cfg.S0 * sqrt_v0
    z_s_se = float(np.std(delta_all, ddof=1) / np.sqrt(n)) * cfg.S0 * sqrt_v0

    z_v = float(np.mean(vega_all)) * cfg.sigma_v * sqrt_v0
    z_v_se = float(np.std(vega_all, ddof=1) / np.sqrt(n)) * cfg.sigma_v * sqrt_v0

    if verbose:
        print(f"  MC Greeks computed in {elapsed:.1f}s ({n} paths)")

    return {"z_s_sum": z_s_sum, "z_s_se": z_s_se,
            "z_v": z_v, "z_v_se": z_v_se, "elapsed_s": elapsed}


# -----------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------

def run_greeks():
    print(f"\n{'#'*60}")
    print(f"# Greeks Extraction — d = {DIM}")
    print(f"{'#'*60}")

    cfg = BatesConfig(d=DIM, N=N_STEPS, M=BATCH_SIZE, epochs=EPOCHS)

    # --- Train Triple-Net ---
    print("\n--- Training Triple-Net ---")
    model, _, _, train_time = solver_triple.train(cfg, verbose=True)

    # --- Neural-network Greeks ---
    z_s, z_v = extract_nn_greeks(model, cfg)
    nn_delta_sum = float(z_s.sum())
    nn_vega = z_v

    # --- MC finite-difference Greeks ---
    print("\n--- MC Finite-Difference Greeks (CRN) ---")
    mc = mc_greeks_fd(cfg)

    # --- Summary table ---
    w = 78
    print(f"\n{'='*w}")
    print(f"  Initial Greeks Extraction at t=0  (d={DIM} Basket Option)")
    print(f"{'='*w}")
    print(f"{'Risk Parameter':<32} {'MC (Finite Diff.)':>22} {'Our Model (NN)':>20}")
    print(f"{'-'*w}")
    print(f"{'Delta Exposure (Σ Z_S)':<32} "
          f"{mc['z_s_sum']:>9.6f} +/- {mc['z_s_se']:.6f} "
          f"{nn_delta_sum:>20.6f}")
    print(f"{'Vega Exposure  (Z_v)':<32} "
          f"{mc['z_v']:>9.6f} +/- {mc['z_v_se']:.6f} "
          f"{nn_vega:>20.6f}")
    print(f"{'='*w}")

    return {"nn_delta": nn_delta_sum, "nn_vega": nn_vega, "mc": mc}


if __name__ == "__main__":
    run_greeks()
    print("\nGreeks extraction complete.")
