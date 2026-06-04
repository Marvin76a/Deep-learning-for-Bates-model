"""Shared neural-network building blocks used by all BSDE solvers."""

import torch
import torch.nn as nn


class SubNet(nn.Module):
    """Feed-forward sub-network (for three-branch solver)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        h = in_dim + 20
        self.net = nn.Sequential(
            nn.Linear(in_dim, h), nn.ReLU(),
            nn.Linear(h, h),      nn.ReLU(),
            nn.Linear(h, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class XNetSubNet(nn.Module):
    """Vector-valued XNet sub-network based on Cauchy activation functions."""

    def __init__(self, in_dim: int, out_dim: int, basis: int):
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


class SimpleNet(nn.Module):
    """Lightweight MLP without BatchNorm (for single / dual network solvers)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        h = in_dim + 10
        self.net = nn.Sequential(
            nn.Linear(in_dim, h), nn.ReLU(),
            nn.Linear(h, h),      nn.ReLU(),
            nn.Linear(h, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class BNNet(nn.Module):
    """MLP with input BatchNorm (for dual-network solver)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        h = in_dim + 20
        
        
        self.layers = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, out_dim),
        )

    def forward(self, x):
        return self.layers(self.bn0(x))
