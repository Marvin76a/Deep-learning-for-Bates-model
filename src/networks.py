"""Shared neural-network building blocks used by all BSDE solvers."""

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
