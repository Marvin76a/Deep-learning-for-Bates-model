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
    """High-dimensional Bates (SVJDM) parameters for European basket option."""

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

        # Basket option
        self.K = 1.0
        self.weights = None

        # XNet: the paper typically uses O(d) basis functions; increase this
        # value to trade parameters for stronger approximation capacity.
        self.xnet_basis = self.d

        # Training
        self.epochs = 3000
        self.lr = 1e-3

        self._build()

    def _build(self):
        d = self.d
        self.weights = np.ones(d) / d

        # k_bar = E[J-1] = exp(mu_J + sigma_J^2/2) - 1
        self.k_bar = np.exp(self.mu_J + 0.5 * self.sigma_J ** 2) - 1

        # Joint (d+1)x(d+1) correlation matrix  R_full = [[R_S, rho_Sv], [rho_Sv^T, 1]]
        R_S = np.full((d, d), self.rho_assets)
        np.fill_diagonal(R_S, 1.0)
        rho_vec = np.full((d, 1), self.rho_sv)
        R_full = np.block([[R_S, rho_vec],
                           [rho_vec.T, np.array([[1.0]])]])
        self.L_full = np.linalg.cholesky(R_full)


# ==========================================
# 2. Noise Sampling & Path Generation
# ==========================================

def sample_noises(cfg, device):
    """Sample all stochastic increments for one batch of M paths."""
    dt = cfg.T / cfg.N
    sqrt_dt = np.sqrt(dt)

    # Correlated Brownian increments via joint Cholesky:
    #   [dW_S; dW_v] = L_full @ dZ,  dZ ~ N(0, dt I)
    dZ = torch.randn(cfg.N, cfg.M, cfg.d + 1, device=device)
    L = torch.tensor(cfg.L_full, dtype=torch.float32, device=device)
    dW = dZ @ L.T * sqrt_dt                            # [N, M, d+1]
    dW_S = dW[:, :, :cfg.d]                             # [N, M, d]
    dW_v = dW[:, :, cfg.d:]                             # [N, M, 1]

    # Orthogonal variance driver (block-diagonal decoupling)
    dW_v_tilde = dZ[:, :, cfg.d:] * sqrt_dt             # [N, M, 1]

    # Common Poisson arrivals (single process for all assets)
    dN = Poisson(cfg.lambda_ * dt).sample((cfg.N, cfg.M)).to(device)
    dN_tilde = dN - cfg.lambda_ * dt

    # Heterogeneous log-normal jump sizes:  ln(J^(i)) ~ N(mu_J, sigma_J^2)
    ln_J = cfg.mu_J + cfg.sigma_J * torch.randn(cfg.N, cfg.M, cfg.d, device=device)
    J = torch.exp(ln_J)

    return dW_S, dW_v, dW_v_tilde, dN, dN_tilde, J


def generate_paths(cfg, device, dW_S, dW_v, dN, J):
    """Euler-Maruyama discretization of the high-dimensional Bates model."""
    dt = cfg.T / cfg.N
    d = cfg.d
    mu = cfg.r - cfg.q - cfg.lambda_ * cfg.k_bar

    X = torch.zeros(cfg.N + 1, cfg.M, d + 1, device=device)
    X[0, :, :d] = cfg.S0
    X[0, :, d] = cfg.v0

    for n in range(cfg.N):
        S = X[n, :, :d]                                # [M, d]
        v = X[n, :, d:d + 1]                           # [M, 1]
        v_pos = torch.clamp(v, min=0.0)                 # Full Truncation
        sqrt_v = torch.sqrt(v_pos)

        # CIR variance
        X[n + 1, :, d:d + 1] = (v
                                 + cfg.kappa * (cfg.theta - v_pos) * dt
                                 + cfg.sigma_v * sqrt_v * dW_v[n])

        # Asset prices:  dS/S = mu dt + sqrt(v) dW_S + (J-1) dN
        X[n + 1, :, :d] = (S
                            + mu * S * dt
                            + sqrt_v * S * dW_S[n]
                            + S * (J[n] - 1) * dN[n].unsqueeze(-1))
    return X


# ==========================================
# 3. Three-Branch XNet Policy Network
# ==========================================

class XNetSubNet(nn.Module):
    """Vector-valued XNet sub-network based on Cauchy activation functions."""

    def __init__(self, in_dim, out_dim, basis):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.basis = basis

        self.a = nn.Parameter(torch.empty(basis, in_dim))
        self.c = nn.Parameter(torch.empty(basis))
        self.raw_e = nn.Parameter(torch.empty(basis))
        self.alpha = nn.Parameter(torch.empty(out_dim, basis))
        self.beta = nn.Parameter(torch.empty(out_dim, basis))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.a)
        nn.init.uniform_(self.c, -0.1, 0.1)
        nn.init.uniform_(self.raw_e, 0.5, 1.0)
        nn.init.xavier_uniform_(self.alpha)
        nn.init.xavier_uniform_(self.beta)

    def forward(self, x):
        s = x @ self.a.T + self.c
        e = torch.nn.functional.softplus(self.raw_e) + 1e-6
        denom = s.square() + e.square()

        real_basis = s / denom
        imag_basis = e / denom
        return real_basis @ self.alpha.T + imag_basis @ self.beta.T


class BasketSolver(nn.Module):
    """Deep BSDE solver for European basket call under SVJDM (Algorithm 1).

    Three-branch separated XNet policy network:
      Net_ZS  -> d-dim  Delta exposure
      Net_Zv  -> 1-dim  Vega  exposure
      Net_U   -> 1-dim  Jump  compensator
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        inp = cfg.d + 1                                 # [S_n, v_n^+]
        basis = cfg.xnet_basis

        self.Y0 = nn.Parameter(torch.tensor(0.0))

        self.net_zs = nn.ModuleList([XNetSubNet(inp, cfg.d, basis) for _ in range(cfg.N)])
        self.net_zv = nn.ModuleList([XNetSubNet(inp, 1, basis)     for _ in range(cfg.N)])
        self.net_u = nn.ModuleList([XNetSubNet(inp, 1, basis)      for _ in range(cfg.N)])

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

            Z_S = self.net_zs[n](x_in)                  # [M, d]
            Z_v = self.net_zv[n](x_in)                  # [M, 1]
            U = self.net_u[n](x_in).squeeze(1)           # [M]

            # BSDE forward:  dY = rY dt + Z_S^T dW_S + Z_v dW_v_tilde + U dN_tilde
            Y = Y + (cfg.r * Y * dt
                     + (Z_S * dW_S[n]).sum(1)
                     + (Z_v * dW_v_tilde[n]).sum(1)
                     + U * dN_tilde[n])

        # Basket call payoff:  g(S_T) = (sum w_i S_i^T  - K)^+
        w = torch.tensor(cfg.weights, dtype=torch.float32, device=dev)
        payoff = torch.clamp((w * X[-1, :, :cfg.d]).sum(1) - cfg.K, min=0.0)

        return Y, payoff


# ==========================================
# 4. Training Loop
# ==========================================

def train(cfg):
    set_seed(cfg.seed)
    dev = cfg.device

    model = BasketSolver(cfg).to(dev)
    opt = optim.AdamW(model.parameters(), lr=cfg.lr)
    sched = optim.lr_scheduler.MultiStepLR(opt, milestones=[1500, 2500], gamma=0.1)

    losses, y0s = [], []
    t0 = time.time()
    print(f"=== Basket Option XNet (SVJDM)  d={cfg.d}  N={cfg.N}  M={cfg.M}  L={cfg.xnet_basis} ===")

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

        if ep % 100 == 0 or ep == cfg.epochs - 1:
            print(f"  Epoch {ep:4d} | Loss {loss.item():.4e} "
                  f"| Y0 {model.Y0.item():.6f} | {time.time() - t0:.1f}s")

    print(f"Training complete.  Y0 = {model.Y0.item():.6f}  ({time.time() - t0:.1f}s)")
    return model, losses, y0s


# ==========================================
# 5. Visualization
# ==========================================

def plot_results(losses, y0s, cfg):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(losses)
    ax1.set_yscale("log")
    ax1.set(title=f"Training Loss (XNet Basket d={cfg.d})",
            xlabel="Epoch", ylabel="MSE")
    ax1.grid(True, which="both", alpha=0.4)

    ax2.plot(y0s, color="C1")
    ax2.set(title=f"XNet Y0 Convergence -> {y0s[-1]:.4f}",
            xlabel="Epoch", ylabel="Price")
    ax2.grid(True)

    os.makedirs("figs", exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"figs/basket_xnet_d{cfg.d}_N{cfg.N}_L{cfg.xnet_basis}.png")
    plt.close(fig)


# ==========================================
# 6. Entry Point
# ==========================================

if __name__ == "__main__":
    cfg = Config()
    model, losses, y0s = train(cfg)
    plot_results(losses, y0s, cfg)
