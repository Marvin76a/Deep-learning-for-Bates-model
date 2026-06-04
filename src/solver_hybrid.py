"""Hybrid three-branch Deep BSDE solver for European basket call under SVJDM.

The smoother diffusion exposures use XNet branches:
  Net_ZS -> d-dim Delta exposure
  Net_Zv -> 1-dim Vega exposure

The jump compensator keeps the MLP branch:
  Net_U  -> 1-dim jump compensator
"""

import time

import torch
import torch.nn as nn
import torch.optim as optim

from .bates_sde import generate_paths, sample_noises
from .config import BatesConfig
from .networks import SubNet, XNetSubNet
from .utils import plot_training, set_seed


class HybridSolver(nn.Module):
    def __init__(self, cfg: BatesConfig):
        super().__init__()
        self.cfg = cfg
        inp = cfg.d + 1
        basis = cfg.xnet_basis

        self.Y0 = nn.Parameter(torch.tensor(cfg.y0_init))

        self.net_zs = nn.ModuleList([XNetSubNet(inp, cfg.d, basis) for _ in range(cfg.N)])
        self.net_zv = nn.ModuleList([XNetSubNet(inp, 1, basis) for _ in range(cfg.N)])
        self.net_u = nn.ModuleList([SubNet(inp, 1) for _ in range(cfg.N)])

    def forward(self, X, dW_S, dW_v_tilde, dN_tilde):
        cfg = self.cfg
        dt = cfg.T / cfg.N
        M = X.shape[1]
        dev = X.device

        Y = self.Y0.expand(M)

        for n in range(cfg.N):
            S_n = X[n, :, :cfg.d]
            v_n = torch.clamp(X[n, :, cfg.d:cfg.d + 1], min=0.0)
            x_in = torch.cat([S_n, v_n], dim=1)

            Z_S = self.net_zs[n](x_in)
            Z_v = self.net_zv[n](x_in)
            U = self.net_u[n](x_in).squeeze(1)

            Y = Y + (
                cfg.r * Y * dt
                + (Z_S * dW_S[n]).sum(1)
                + (Z_v * dW_v_tilde[n]).sum(1)
                + U * dN_tilde[n]
            )

        w = torch.tensor(cfg.weights, dtype=torch.float32, device=dev)
        payoff = torch.clamp((w * X[-1, :, :cfg.d]).sum(1) - cfg.K, min=0.0)

        return Y, payoff


def train(cfg: BatesConfig, verbose: bool = True):
    set_seed(cfg.seed)
    dev = cfg.device

    model = HybridSolver(cfg).to(dev)
    opt = optim.AdamW(model.parameters(), lr=cfg.lr)
    sched = optim.lr_scheduler.MultiStepLR(opt, milestones=[1500, 2500], gamma=0.1)

    losses, y0s = [], []
    t0 = time.time()
    if verbose:
        print(
            f"=== Hybrid XNet/MLP Basket (SVJDM)  d={cfg.d}  N={cfg.N}  "
            f"M={cfg.M}  L={cfg.xnet_basis} ==="
        )

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
    plot_training(losses, y0s, f"(Hybrid d={cfg.d}, L={cfg.xnet_basis})",
                  save_path=f"figs/hybrid_d{cfg.d}_N{cfg.N}_L{cfg.xnet_basis}.png")
