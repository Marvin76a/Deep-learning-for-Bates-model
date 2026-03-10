"""Dual-network Deep BSDE solver for Bates model.

Uses two network branches:
  Z-network  -> (d+1)-dim  combined diffusion risk (assets + variance)
  U-network  -> d-dim      jump compensator

This is the "traditional" Deep BSDE layout (cf. pricing_svjdm.py) adapted
to use the shared Cholesky-orthogonalised SDE from ``bates_sde``.
"""

import time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from .config import BatesConfig
from .bates_sde import sample_noises, generate_paths
from .networks import SubNet
from .utils import set_seed, plot_training


# ---------------------------------------------------------------------------
# Internal helpers: reconstruct independent noise dW_iso from correlated dW
# ---------------------------------------------------------------------------

def _correlated_to_iso(dW_S, dW_v, L_full_inv, sqrt_dt):
    """Convert correlated increments back to independent standard normals.

    The dual-network Z lives in the *independent* noise space so that
    Z · dW_iso covers both asset and variance diffusion simultaneously.
    """
    dW = torch.cat([dW_S, dW_v], dim=-1)              # [N, M, d+1]
    L_inv = torch.tensor(L_full_inv, dtype=torch.float32, device=dW.device)
    dW_iso = (dW @ L_inv.T) / sqrt_dt * sqrt_dt       # keep units: already scaled
    return dW_iso                                      # [N, M, d+1]


class DualSolver(nn.Module):
    def __init__(self, cfg: BatesConfig):
        super().__init__()
        self.cfg = cfg
        x_dim = cfg.d + 1                             # state = (S, v)

        self.Y0 = nn.Parameter(torch.tensor(0.0))

        self.z_nets = nn.ModuleList([SubNet(x_dim, x_dim) for _ in range(cfg.N)])
        self.u_nets = nn.ModuleList([SubNet(x_dim, 1)     for _ in range(cfg.N)])

    def forward(self, X, dW_S, dW_v_tilde, dN_tilde):
        cfg = self.cfg
        dt = cfg.T / cfg.N
        M = X.shape[1]

        dW_full = torch.cat([dW_S, dW_v_tilde], dim=-1)  # [N, M, d+1]

        Y = self.Y0.expand(M)

        for n in range(cfg.N):
            X_n = X[n]                                 # [M, d+1]
            Z = self.z_nets[n](X_n)                    # [M, d+1]
            U = self.u_nets[n](X_n).squeeze(1)         # [M]

            Y = Y + (
                cfg.r * Y * dt
                + (Z * dW_full[n]).sum(1)
                + U * dN_tilde[n]
            )

        # Payoff
        w = torch.tensor(cfg.weights, dtype=torch.float32, device=X.device)
        payoff = torch.clamp((w * X[-1, :, :cfg.d]).sum(1) - cfg.K, min=0.0)

        return Y, payoff


def train(cfg: BatesConfig, verbose: bool = True):
    set_seed(cfg.seed)
    dev = cfg.device

    model = DualSolver(cfg).to(dev)
    opt = optim.AdamW(model.parameters(), lr=cfg.lr)
    sched = optim.lr_scheduler.MultiStepLR(opt, milestones=[1500, 2500], gamma=0.1)

    losses, y0s = [], []
    t0 = time.time()
    if verbose:
        print(f"=== Dual-Net Basket (SVJDM)  d={cfg.d}  N={cfg.N}  M={cfg.M} ===")

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
        sched.step()

        losses.append(loss.item())
        y0s.append(model.Y0.item())

        if verbose and (ep % 100 == 0 or ep == cfg.epochs - 1):
            print(f"  Epoch {ep:4d} | Loss {loss.item():.4e} "
                  f"| Y0 {model.Y0.item():.6f} | {time.time() - t0:.1f}s")

    elapsed = time.time() - t0
    if verbose:
        print(f"Training complete.  Y0 = {model.Y0.item():.6f}  ({elapsed:.1f}s)")
    return model, losses, y0s, elapsed


if __name__ == "__main__":
    cfg = BatesConfig()
    model, losses, y0s, _ = train(cfg)
    plot_training(losses, y0s, f"(Dual d={cfg.d})",
                  save_path=f"figs/dual_d{cfg.d}_N{cfg.N}.png")
