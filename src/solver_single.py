"""Single-network Deep BSDE solver for Bates model (Han et al. 2018 style).

Only one Z-network approximates sigma^T * grad_u at each time step.
No separate U-network -- jump risk is entirely absorbed into the single
network's approximation error, following the original Deep BSDE paper
which did not explicitly handle jump compensators.
"""

import time

import torch
import torch.nn as nn
import torch.optim as optim

from .config import BatesConfig
from .bates_sde import sample_noises, generate_paths
from .networks import SimpleNet
from .utils import set_seed, plot_training


class SingleSolver(nn.Module):
    def __init__(self, cfg: BatesConfig):
        super().__init__()
        self.cfg = cfg
        x_dim = cfg.d + 1                             # state = (S, v)

        self.Y0 = nn.Parameter(torch.tensor(0.0))

        self.z_nets = nn.ModuleList([
            SimpleNet(x_dim, x_dim) for _ in range(cfg.N)
        ])

    def forward(self, X, dW_S, dW_v, dN_tilde):
        cfg = self.cfg
        dt = cfg.T / cfg.N
        M = X.shape[1]

        dW_full = torch.cat([dW_S, dW_v], dim=-1)     # [N, M, d+1]

        Y = self.Y0.expand(M)

        for n in range(cfg.N):
            X_n = X[n]                                 # [M, d+1]
            Z = self.z_nets[n](X_n)                    # [M, d+1]

            # f = -r * Y  =>  dY = rY dt + Z · dW
            # Jump risk NOT explicitly modelled -- absorbed into approximation
            Y = Y + (
                cfg.r * Y * dt
                + (Z * dW_full[n]).sum(1)
            )

        w = torch.tensor(cfg.weights, dtype=torch.float32, device=X.device)
        payoff = torch.clamp((w * X[-1, :, :cfg.d]).sum(1) - cfg.K, min=0.0)

        return Y, payoff


def train(cfg: BatesConfig, verbose: bool = True):
    set_seed(cfg.seed)
    dev = cfg.device

    model = SingleSolver(cfg).to(dev)
    opt = optim.NAdam(model.parameters(), lr=cfg.lr)
    sched = optim.lr_scheduler.MultiStepLR(opt, milestones=[1500, 2500], gamma=0.1)

    losses, y0s = [], []
    t0 = time.time()
    if verbose:
        print(f"=== Single-Net Basket (SVJDM)  d={cfg.d}  N={cfg.N}  M={cfg.M} ===")

    for ep in range(cfg.epochs):
        dW_S, dW_v, dN, dN_tilde, J = sample_noises(cfg, dev)
        with torch.no_grad():
            X = generate_paths(cfg, dev, dW_S, dW_v, dN, J)

        model.train()
        Y_pred, payoff = model(X, dW_S, dW_v, dN_tilde)
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
    plot_training(losses, y0s, f"(Single d={cfg.d})",
                  save_path=f"figs/single_d{cfg.d}_N{cfg.N}.png")
