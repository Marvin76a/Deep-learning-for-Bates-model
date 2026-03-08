import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions.poisson import Poisson
import matplotlib.pyplot as plt
import random
import time
import os


# ==========================================
# 1. Seed & Config
# ==========================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    """Bates (SVJDM) parameters for down-and-out barrier option."""

    def __init__(self):
        self.seed = 42
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Dimensions and discretization
        self.d = 50
        self.T = 1.0
        self.N = 100
        self.M = 1024
        self.r = 0.05
        self.q = 0.0

        # Heston stochastic volatility
        self.S0 = 1.0
        self.v0 = 0.04
        self.kappa = 2.0
        self.theta = 0.04
        self.sigma_v = 0.3

        # Correlation structure
        self.rho_assets = 0.3
        self.rho_sv = -0.5

        # Log-normal jumps (common Poisson)
        self.lambda_ = 1.0
        self.mu_J = -0.1
        self.sigma_J = 0.2

        # Basket payoff for surviving paths
        self.K = 1.0
        self.weights = None

        # Barrier parameters
        self.barrier = 0.5
        self.rebate = 0.0

        # Training
        self.epochs = 3000
        self.lr = 1e-3

        self._build()

    def _build(self):
        d = self.d
        self.weights = np.ones(d) / d
        self.k_bar = np.exp(self.mu_J + 0.5 * self.sigma_J ** 2) - 1

        R_S = np.full((d, d), self.rho_assets)
        np.fill_diagonal(R_S, 1.0)
        rho_vec = np.full((d, 1), self.rho_sv)
        R_full = np.block([[R_S, rho_vec],
                           [rho_vec.T, np.array([[1.0]])]])
        self.L_full = np.linalg.cholesky(R_full)


# ==========================================
# 2. Noise Sampling
# ==========================================

def sample_noises(cfg, device):
    """Sample all stochastic increments for one batch of M paths."""
    dt = cfg.T / cfg.N
    sqrt_dt = np.sqrt(dt)

    dZ = torch.randn(cfg.N, cfg.M, cfg.d + 1, device=device)
    L = torch.tensor(cfg.L_full, dtype=torch.float32, device=device)
    dW = dZ @ L.T * sqrt_dt
    dW_S = dW[:, :, :cfg.d]
    dW_v = dW[:, :, cfg.d:]

    # Orthogonal variance driver (block-diagonal decoupling)
    dW_v_tilde = dZ[:, :, cfg.d:] * sqrt_dt

    dN = Poisson(cfg.lambda_ * dt).sample((cfg.N, cfg.M)).to(device)
    dN_tilde = dN - cfg.lambda_ * dt

    ln_J = cfg.mu_J + cfg.sigma_J * torch.randn(cfg.N, cfg.M, cfg.d, device=device)
    J = torch.exp(ln_J)

    return dW_S, dW_v, dW_v_tilde, dN, dN_tilde, J


# ==========================================
# 3. Path Generation with Barrier Tracking
# ==========================================

def _brownian_bridge_breach(S_start, S_end, v_pos, dt, barrier, device):
    """Detect hidden continuous-path barrier crossings via Brownian bridge.

    For GBM with local variance v, the probability that the minimum of
    the log-price bridge between S_start and S_end dips below barrier B is:
        P = exp( -2 * log(S_start/B) * log(S_end/B) / (v * dt) )
    A uniform draw decides each asset; path breaches if ANY asset crosses.
    """
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
    return breach_per_asset.any(dim=-1)                  # [M]


def generate_paths_with_barrier(cfg, device, dW_S, dW_v, dN, J):
    """Euler-Maruyama paths + batch stopping-time tracking (Algorithm 2).

    Returns:
        X:     [N+1, M, d+1]  state paths
        alive: [N+1, M]       per-step active mask (1=alive, 0=knocked-out)
    """
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

        # CIR variance
        v_next = (v
                  + cfg.kappa * (cfg.theta - v_pos) * dt
                  + cfg.sigma_v * sqrt_v * dW_v[n])
        X[n + 1, :, d:d + 1] = v_next

        # Asset prices: split continuous and jump for inter-step detection
        S_cont = S + mu * S * dt + sqrt_v * S * dW_S[n]
        S_next = S_cont + S * (J[n] - 1) * dN[n].unsqueeze(-1)
        X[n + 1, :, :d] = S_next

        # ---- Stopping-time barrier breach detection ----
        # (a) Direct breach: any asset below B at t_{n+1}
        direct = S_next.min(dim=-1).values < B

        # (b) Pre-jump breach: continuous path fell below B while a jump occurred
        pre_jump = (S_cont.min(dim=-1).values < B) & (dN[n] > 0)

        # (c) Brownian bridge: hidden continuous crossing between t_n and t_{n+1}
        bridge = _brownian_bridge_breach(S, S_cont, v_pos, dt, B, device)

        breach = direct | pre_jump | bridge
        alive[n + 1] = alive[n] * (~breach).float()

    return X, alive


# ==========================================
# 4. Three-Branch Neural Network
# ==========================================

class SubNet(nn.Module):
    """Feed-forward sub-network with input BatchNorm."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        h = in_dim + 20
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(),
            nn.Linear(h, h),      nn.BatchNorm1d(h), nn.ReLU(),
            nn.Linear(h, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class BarrierSolver(nn.Module):
    """Deep BSDE solver for barrier option with stopping-time tracking (Algorithm 2).

    Identical three-branch architecture to BasketSolver, augmented with
    a per-path active mask that freezes the BSDE upon knock-out.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        inp = cfg.d + 2

        self.Y0 = nn.Parameter(torch.tensor(0.0))

        self.net_zs = nn.ModuleList([SubNet(inp, cfg.d) for _ in range(cfg.N)])
        self.net_zv = nn.ModuleList([SubNet(inp, 1)     for _ in range(cfg.N)])
        self.net_u  = nn.ModuleList([SubNet(inp, 1)     for _ in range(cfg.N)])

    def forward(self, X, dW_S, dW_v_tilde, dN_tilde, alive):
        """
        Args:
            X:           [N+1, M, d+1]  forward paths
            dW_S:        [N, M, d]
            dW_v_tilde:  [N, M, 1]      orthogonal variance driver
            dN_tilde:    [N, M]
            alive:       [N+1, M]       active mask from barrier tracking
        """
        cfg = self.cfg
        dt = cfg.T / cfg.N
        M = X.shape[1]
        dev = X.device

        Y = self.Y0.expand(M)

        for n in range(cfg.N):
            mask = alive[n + 1]                          # [M]  updated mask

            S_n = X[n, :, :cfg.d]
            v_n = torch.clamp(X[n, :, cfg.d:cfg.d + 1], min=0.0)
            t_vec = torch.full((M, 1), n * dt, device=dev)
            x_in = torch.cat([t_vec, S_n, v_n], dim=1)

            # Masked policy outputs: zero for knocked-out paths
            Z_S = self.net_zs[n](x_in) * mask.unsqueeze(1)
            Z_v = self.net_zv[n](x_in) * mask.unsqueeze(1)
            U   = self.net_u[n](x_in).squeeze(1) * mask

            dY = (cfg.r * Y * dt
                  + (Z_S * dW_S[n]).sum(1)
                  + (Z_v * dW_v_tilde[n]).sum(1)
                  + U * dN_tilde[n])

            Y = Y + dY * mask                           # freeze Y for knocked-out paths

        # Stopped payoff:  surviving -> basket call;  knocked-out -> rebate
        surv = alive[-1]
        w = torch.tensor(cfg.weights, dtype=torch.float32, device=dev)
        basket = torch.clamp((w * X[-1, :, :cfg.d]).sum(1) - cfg.K, min=0.0)
        g_target = surv * basket + (1.0 - surv) * cfg.rebate

        return Y, g_target


# ==========================================
# 5. Training Loop
# ==========================================

def train(cfg):
    set_seed(cfg.seed)
    dev = cfg.device

    model = BarrierSolver(cfg).to(dev)
    opt = optim.NAdam(model.parameters(), lr=cfg.lr)
    sched = optim.lr_scheduler.MultiStepLR(opt, milestones=[1500, 2500], gamma=0.1)

    losses, y0s = [], []
    t0 = time.time()
    print(f"=== Barrier Option (SVJDM)  d={cfg.d}  N={cfg.N}  M={cfg.M} ===")
    print(f"    Barrier={cfg.barrier}  Rebate={cfg.rebate}")

    for ep in range(cfg.epochs):
        dW_S, dW_v, dW_v_tilde, dN, dN_tilde, J = sample_noises(cfg, dev)
        with torch.no_grad():
            X, alive = generate_paths_with_barrier(cfg, dev, dW_S, dW_v, dN, J)

        model.train()
        Y_pred, g_target = model(X, dW_S, dW_v_tilde, dN_tilde, alive)
        loss = ((Y_pred - g_target) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        losses.append(loss.item())
        y0s.append(model.Y0.item())

        if ep % 100 == 0 or ep == cfg.epochs - 1:
            surv_rate = alive[-1].mean().item()
            print(f"  Epoch {ep:4d} | Loss {loss.item():.4e} "
                  f"| Y0 {model.Y0.item():.6f} | Surv {surv_rate:.2%} "
                  f"| {time.time() - t0:.1f}s")

    print(f"Training complete.  Y0 = {model.Y0.item():.6f}  ({time.time() - t0:.1f}s)")
    return model, losses, y0s


# ==========================================
# 6. Visualization
# ==========================================

def plot_results(losses, y0s, cfg):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(losses)
    ax1.set_yscale("log")
    ax1.set(title=f"Training Loss (Barrier d={cfg.d}, B={cfg.barrier})",
            xlabel="Epoch", ylabel="MSE")
    ax1.grid(True, which="both", alpha=0.4)

    ax2.plot(y0s, color="C1")
    ax2.set(title=f"Y0 Convergence \u2192 {y0s[-1]:.4f}",
            xlabel="Epoch", ylabel="Price")
    ax2.grid(True)

    os.makedirs("figs", exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"figs/barrier_d{cfg.d}_N{cfg.N}_B{cfg.barrier}.png")
    plt.close(fig)


# ==========================================
# 7. Entry Point
# ==========================================

if __name__ == "__main__":
    cfg = Config()
    model, losses, y0s = train(cfg)
    plot_results(losses, y0s, cfg)
