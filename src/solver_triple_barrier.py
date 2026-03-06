"""Three-branch Deep BSDE solver for barrier options under SVJDM (Algorithm 2).

Identical architecture to the basket solver, augmented with per-path
stopping-time tracking and Brownian-bridge inter-step breach detection.
"""

import time

import torch
import torch.nn as nn
import torch.optim as optim

from .config import BatesConfig
from .bates_sde import sample_noises
from .networks import SubNet
from .utils import set_seed, plot_training


# -----------------------------------------------------------------------
# Path generation with barrier tracking
# -----------------------------------------------------------------------

def _brownian_bridge_breach(S_start, S_end, v_pos, dt, barrier, device):
    log_s0 = torch.log(torch.clamp(S_start / barrier, min=1e-8))
    log_s1 = torch.log(torch.clamp(S_end / barrier, min=1e-8))
    both_above = (log_s0 > 0) & (log_s1 > 0)

    exponent = torch.where(
        both_above,
        -2.0 * log_s0 * log_s1 / (v_pos * dt + 1e-10),
        torch.full_like(log_s0, -50.0),
    )
    prob = torch.exp(exponent)

    breach_per_asset = (torch.rand_like(prob) < prob) & both_above
    return breach_per_asset.any(dim=-1)


def generate_paths_with_barrier(cfg, device, dW_S, dW_v, dN, J):
    dt = cfg.T / cfg.N
    d = cfg.d
    mu = cfg.r - cfg.q - cfg.lambda_ * cfg.k_bar
    B = cfg.barrier

    X = torch.zeros(cfg.N + 1, cfg.M, d + 1, device=device)
    X[0, :, :d] = cfg.S0
    X[0, :, d] = cfg.v0

    alive = torch.ones(cfg.N + 1, cfg.M, device=device)

    for n in range(cfg.N):
        S = X[n, :, :d]
        v = X[n, :, d:d + 1]
        v_pos = torch.clamp(v, min=0.0)
        sqrt_v = torch.sqrt(v_pos)

        v_next = (v + cfg.kappa * (cfg.theta - v_pos) * dt
                  + cfg.sigma_v * sqrt_v * dW_v[n])
        X[n + 1, :, d:d + 1] = v_next

        S_cont = S + mu * S * dt + sqrt_v * S * dW_S[n]
        S_next = S_cont + S * (J[n] - 1) * dN[n].unsqueeze(-1)
        X[n + 1, :, :d] = S_next

        direct = S_next.min(dim=-1).values < B
        pre_jump = (S_cont.min(dim=-1).values < B) & (dN[n] > 0)
        bridge = _brownian_bridge_breach(S, S_cont, v_pos, dt, B, device)

        breach = direct | pre_jump | bridge
        alive[n + 1] = alive[n] * (~breach).float()

    return X, alive


# -----------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------

class BarrierSolver(nn.Module):
    def __init__(self, cfg: BatesConfig):
        super().__init__()
        self.cfg = cfg
        inp = cfg.d + 2

        self.Y0 = nn.Parameter(torch.tensor(0.0))

        self.net_zs = nn.ModuleList([SubNet(inp, cfg.d) for _ in range(cfg.N)])
        self.net_zv = nn.ModuleList([SubNet(inp, 1)     for _ in range(cfg.N)])
        self.net_u  = nn.ModuleList([SubNet(inp, 1)     for _ in range(cfg.N)])

    def forward(self, X, dW_S, dW_v, dN_tilde, alive):
        cfg = self.cfg
        dt = cfg.T / cfg.N
        M = X.shape[1]
        dev = X.device

        Y = self.Y0.expand(M)

        for n in range(cfg.N):
            mask = alive[n + 1]

            S_n = X[n, :, :cfg.d]
            v_n = torch.clamp(X[n, :, cfg.d:cfg.d + 1], min=0.0)
            t_vec = torch.full((M, 1), n * dt, device=dev)
            x_in = torch.cat([t_vec, S_n, v_n], dim=1)

            Z_S = self.net_zs[n](x_in) * mask.unsqueeze(1)
            Z_v = self.net_zv[n](x_in) * mask.unsqueeze(1)
            U   = self.net_u[n](x_in).squeeze(1) * mask

            dY = (cfg.r * Y * dt
                  + (Z_S * dW_S[n]).sum(1)
                  + (Z_v * dW_v[n]).sum(1)
                  + U * dN_tilde[n])

            Y = Y + dY * mask

        surv = alive[-1]
        w = torch.tensor(cfg.weights, dtype=torch.float32, device=dev)
        basket = torch.clamp((w * X[-1, :, :cfg.d]).sum(1) - cfg.K, min=0.0)
        rebate = getattr(cfg, "rebate", 0.0)
        g_target = surv * basket + (1.0 - surv) * rebate

        return Y, g_target


def train(cfg: BatesConfig, verbose: bool = True):
    set_seed(cfg.seed)
    dev = cfg.device

    model = BarrierSolver(cfg).to(dev)
    opt = optim.NAdam(model.parameters(), lr=cfg.lr)
    sched = optim.lr_scheduler.MultiStepLR(opt, milestones=[1500, 2500], gamma=0.1)

    losses, y0s = [], []
    t0 = time.time()
    if verbose:
        print(f"=== Barrier Option (SVJDM)  d={cfg.d}  N={cfg.N}  M={cfg.M} ===")
        print(f"    Barrier={cfg.barrier}  Rebate={getattr(cfg, 'rebate', 0.0)}")

    for ep in range(cfg.epochs):
        dW_S, dW_v, dN, dN_tilde, J = sample_noises(cfg, dev)
        with torch.no_grad():
            X, alive = generate_paths_with_barrier(cfg, dev, dW_S, dW_v, dN, J)

        model.train()
        Y_pred, g_target = model(X, dW_S, dW_v, dN_tilde, alive)
        loss = ((Y_pred - g_target) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        losses.append(loss.item())
        y0s.append(model.Y0.item())

        if verbose and (ep % 100 == 0 or ep == cfg.epochs - 1):
            surv_rate = alive[-1].mean().item()
            print(f"  Epoch {ep:4d} | Loss {loss.item():.4e} "
                  f"| Y0 {model.Y0.item():.6f} | Surv {surv_rate:.2%} "
                  f"| {time.time() - t0:.1f}s")

    elapsed = time.time() - t0
    if verbose:
        print(f"Training complete.  Y0 = {model.Y0.item():.6f}  ({elapsed:.1f}s)")
    return model, losses, y0s, elapsed
