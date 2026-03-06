"""Bates (SVJDM) forward SDE simulation shared by all solvers.

Provides noise sampling and Euler-Maruyama path generation using the
joint Cholesky orthogonalization described in the thesis (Section 3.1).
"""

import numpy as np
import torch
from torch.distributions.poisson import Poisson

from .config import BatesConfig


def sample_noises(cfg: BatesConfig, device: torch.device):
    """Sample all stochastic increments for one batch of *M* paths.

    Returns
    -------
    dW_S : Tensor [N, M, d]   – correlated asset Brownian increments
    dW_v : Tensor [N, M, 1]   – variance Brownian increments
    dN   : Tensor [N, M]      – Poisson jump counts
    dN_tilde : Tensor [N, M]  – compensated Poisson increments
    J    : Tensor [N, M, d]   – log-normal relative jump sizes
    """
    dt = cfg.T / cfg.N
    sqrt_dt = np.sqrt(dt)

    # Correlated increments via joint Cholesky: [dW_S; dW_v] = L_full @ dZ
    dZ = torch.randn(cfg.N, cfg.M, cfg.d + 1, device=device)
    L = torch.tensor(cfg.L_full, dtype=torch.float32, device=device)
    dW = dZ @ L.T * sqrt_dt                         # [N, M, d+1]
    dW_S = dW[:, :, :cfg.d]                          # [N, M, d]
    dW_v = dW[:, :, cfg.d:]                          # [N, M, 1]

    # Common Poisson arrivals (single process for all assets)
    dN = Poisson(cfg.lambda_ * dt).sample((cfg.N, cfg.M)).to(device)
    dN_tilde = dN - cfg.lambda_ * dt

    # Heterogeneous log-normal jump sizes: ln(J) ~ N(mu_J, sigma_J^2)
    ln_J = cfg.mu_J + cfg.sigma_J * torch.randn(cfg.N, cfg.M, cfg.d, device=device)
    J = torch.exp(ln_J)

    return dW_S, dW_v, dN, dN_tilde, J


def generate_paths(cfg: BatesConfig, device: torch.device, dW_S, dW_v, dN, J):
    """Euler-Maruyama discretisation of the high-dimensional Bates model.

    Returns
    -------
    X : Tensor [N+1, M, d+1]
        State paths.  ``X[:, :, :d]`` are asset prices,
        ``X[:, :, d]`` is the variance process.
    """
    dt = cfg.T / cfg.N
    d = cfg.d
    mu = cfg.r - cfg.q - cfg.lambda_ * cfg.k_bar

    X = torch.zeros(cfg.N + 1, cfg.M, d + 1, device=device)
    X[0, :, :d] = cfg.S0
    X[0, :, d] = cfg.v0

    for n in range(cfg.N):
        S = X[n, :, :d]                              # [M, d]
        v = X[n, :, d:d + 1]                         # [M, 1]
        v_pos = torch.clamp(v, min=0.0)               # Full Truncation
        sqrt_v = torch.sqrt(v_pos)

        # CIR variance
        X[n + 1, :, d:d + 1] = (
            v
            + cfg.kappa * (cfg.theta - v_pos) * dt
            + cfg.sigma_v * sqrt_v * dW_v[n]
        )

        # Asset prices: dS/S = mu dt + sqrt(v) dW_S + (J-1) dN
        X[n + 1, :, :d] = (
            S
            + mu * S * dt
            + sqrt_v * S * dW_S[n]
            + S * (J[n] - 1) * dN[n].unsqueeze(-1)
        )

    return X
