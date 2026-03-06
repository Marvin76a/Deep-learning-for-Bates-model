# SVJDM — Deep Learning Pricing for High-Dimensional Bates Model

A unified deep learning framework for pricing high-dimensional European basket options and barrier options under the Stochastic Volatility Jump-Diffusion Model (SVJDM / Bates 1996).

## Project Structure

```
svjdm/
├── src/                           # Core Python package
│   ├── config.py                  # BatesConfig – unified model parameters
│   ├── bates_sde.py               # Euler-Maruyama path generation (Cholesky)
│   ├── networks.py                # Neural network building blocks
│   ├── solver_triple.py           # Three-branch Deep BSDE (ZS + Zv + U)
│   ├── solver_dual.py             # Dual-network Deep BSDE (Z + U)
│   ├── solver_single.py           # Single-network Deep BSDE (Han et al. 2018)
│   ├── solver_triple_barrier.py   # Barrier option with stopping-time tracking
│   ├── cos_bates.py               # COS Fourier pricing (1-D Bates)
│   ├── mc_bates.py                # High-dim Monte Carlo with variance reduction
│   └── utils.py                   # Seed, timing, plotting utilities
├── experiments/
│   ├── phase1_accuracy.py         # Low-dim accuracy test (d=1, 3)
│   └── phase2_scaling.py          # High-dim scaling test (d=10–200)
├── pricing_algorithms/            # C++ reference (COS & MC for GBM/Heston)
├── thesis/                        # LaTeX source
├── reference/                     # Reference papers
└── figs/                          # Output figures
```

## Bates Model Dynamics

Under the risk-neutral measure, the *d*-dimensional asset prices and the common variance process follow:

```
dS_i / S_i = (r - q - λ·k̄) dt + √v dW_S^i + (J_i - 1) dN
dv = κ(θ - v) dt + σ_v √v dW_v
```

where `corr(dW_S^i, dW_v) = ρ_sv`, `corr(dW_S^i, dW_S^j) = ρ_assets`, and `ln(J) ~ N(μ_J, σ_J²)`.

## Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run experiments

```bash
# Phase 1: low-dim accuracy (d=1, 3)
python experiments/phase1_accuracy.py

# Phase 2: high-dim scaling (d=10, 50, 100, 200)
python experiments/phase2_scaling.py
```

### Train a single model

```python
from src.config import BatesConfig
from src.solver_triple import train

cfg = BatesConfig(d=50, epochs=3000)
model, losses, y0s, elapsed = train(cfg)
```

## Methods Compared

| Method | Networks | Jump handling | Reference |
|--------|----------|---------------|-----------|
| Triple-Net (ours) | ZS(d) + Zv(1) + U(1) | Explicit compensator | Thesis Algorithm 1 |
| Dual-Net | Z(d+1) + U(d) | Explicit compensator | Liu & Gu 2023 |
| Single-Net | Z(d+1) only | Not modelled | Han et al. 2018 |
| COS Fourier | — | Analytic CF | Fang & Oosterlee 2008 |
| Monte Carlo | — | Simulated jumps | Antithetic variates |

## Key Parameters (Default)

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Initial variance | v₀ | 0.04 |
| Mean-reversion speed | κ | 2.0 |
| Long-run variance | θ | 0.04 |
| Vol of vol | σ_v | 0.3 |
| Asset-variance correlation | ρ_sv | -0.5 |
| Inter-asset correlation | ρ_assets | 0.3 |
| Jump intensity | λ | 1.0 |
| Jump mean (log) | μ_J | -0.1 |
| Jump std (log) | σ_J | 0.2 |
| Risk-free rate | r | 0.05 |
| Strike | K | 1.0 |
| Maturity | T | 1.0 |
