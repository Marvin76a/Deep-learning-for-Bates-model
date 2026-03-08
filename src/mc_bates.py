"""High-dimensional Monte Carlo pricing for Bates model basket call options.

Supports:
  - Antithetic variates (AV) for variance reduction
  - Very large path counts (batched to fit in memory)
  - Timing instrumentation

This serves as the 'ground truth' benchmark for Phase 2 (d >= 10) and as
a secondary benchmark for Phase 1 (d = 2, 3) where COS is unavailable.
"""

import time

import numpy as np
import torch

from .config import BatesConfig
from .bates_sde import sample_noises, generate_paths
from .utils import set_seed


def mc_price_basket(
    cfg: BatesConfig,
    n_paths: int = 1_000_000,
    batch_size: int = 50_000,
    antithetic: bool = True,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Monte Carlo estimate of the discounted European basket call price.

    Returns a dict with keys: price, std_err, ci_low, ci_high, elapsed_s.
    """
    set_seed(seed)
    dev = cfg.device

    if antithetic:
        n_paths = n_paths // 2 * 2  # ensure even

    payoffs = []
    t0 = time.time()

    remaining = n_paths
    batch_id = 0
    while remaining > 0:
        cur = min(batch_size, remaining)
        batch_cfg = _temp_cfg(cfg, cur if not antithetic else cur // 2)

        with torch.no_grad():
            dW_S, dW_v, _dW_v_tilde, dN, dN_tilde, J = sample_noises(batch_cfg, dev)

            if antithetic:
                # Original paths
                X_orig = generate_paths(batch_cfg, dev, dW_S, dW_v, dN, J)
                p_orig = _basket_payoff(X_orig, batch_cfg, dev)

                # Antithetic paths (negate Brownian increments)
                X_anti = generate_paths(batch_cfg, dev, -dW_S, -dW_v, dN, J)
                p_anti = _basket_payoff(X_anti, batch_cfg, dev)

                batch_payoff = 0.5 * (p_orig + p_anti)
            else:
                X = generate_paths(batch_cfg, dev, dW_S, dW_v, dN, J)
                batch_payoff = _basket_payoff(X, batch_cfg, dev)

        payoffs.append(batch_payoff.cpu())
        remaining -= cur
        batch_id += 1

        if verbose and batch_id % 5 == 0:
            done = n_paths - remaining
            print(f"  MC batch {batch_id}: {done}/{n_paths} paths")

    all_payoffs = torch.cat(payoffs)
    discounted = np.exp(-cfg.r * cfg.T) * all_payoffs.numpy()

    price = float(np.mean(discounted))
    std_err = float(np.std(discounted, ddof=1) / np.sqrt(len(discounted)))
    elapsed = time.time() - t0

    result = {
        "price": price,
        "std_err": std_err,
        "ci_low": price - 1.96 * std_err,
        "ci_high": price + 1.96 * std_err,
        "n_paths": len(discounted),
        "elapsed_s": elapsed,
    }
    if verbose:
        print(f"  MC price = {price:.6f} ± {std_err:.6f}  "
              f"({elapsed:.1f}s, {len(discounted)} paths)")
    return result


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _basket_payoff(X, cfg, dev):
    w = torch.tensor(cfg.weights, dtype=torch.float32, device=dev)
    return torch.clamp((w * X[-1, :, :cfg.d]).sum(1) - cfg.K, min=0.0)


def _temp_cfg(cfg, M):
    """Return a shallow copy with a different batch size."""
    import copy
    c = copy.copy(cfg)
    c.M = M
    return c


if __name__ == "__main__":
    cfg = BatesConfig(d=5, N=200)
    result = mc_price_basket(cfg, n_paths=500_000)
    print(result)
