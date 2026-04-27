# SVJDM — Deep Learning Pricing for High-Dimensional Bates Model

A structure-preserving, mesh-free deep learning framework for pricing high-dimensional European basket options and barrier options under the Stochastic Volatility Jump-Diffusion Model (SVJDM / Bates 1996). Scales to **200 dimensions** with $\mathcal{O}(d^2)$ complexity, achieving **< 0.1% relative pricing error** against Monte Carlo benchmarks while simultaneously extracting Delta and Vega hedging portfolios as a zero-cost byproduct.

> **Paper**: *Orthogonal Derivation and Deep Learning Pricing of High-Dimensional SVJDM* — with editor at *Quantitative Finance*.

## Highlights

| Metric | Result |
|--------|--------|
| Pricing accuracy (d=50) | 0.0210% relative error vs. 10⁷-path MC |
| Dimensionality | Tested up to d=200, stable convergence |
| Training time | ~215 s per 1k epochs, **constant** across d=3→200 |
| MC Greeks time (d=200) | 481 s (finite-diff bumping) vs. ~216 s (ours, price + Greeks) |
| Robustness | Std = 4.8×10⁻⁵ across 5 random seeds (d=50) |

## Bates Model Dynamics

Under the risk-neutral measure $\mathbb{Q}$, the $d$-dimensional asset prices and the common variance process follow:

$$\frac{dS_i}{S_i} = (r - q - \lambda\bar{k})\,dt + \sqrt{v}\,dW_S^i + (J_i - 1)\,dN$$

$$dv = \kappa(\theta - v)\,dt + \sigma_v \sqrt{v}\,dW_v$$

where $\text{corr}(dW_S^i, dW_v) = \rho_{sv}$, $\text{corr}(dW_S^i, dW_S^j) = \rho_{\text{assets}}$, and $\ln(J) \sim \mathcal{N}(\mu_J, \sigma_J^2)$.

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
│   ├── phase2_scaling.py          # High-dim scaling test (d=10–200)
│   ├── phase3_barrier_level.py    # Barrier level sensitivity (d=50)
│   ├── phase4_barrier_jump.py     # Jump impact on knock-outs (d=50)
│   ├── ablation_model.py          # Model ablation study (d=50)
│   ├── ablation_structure.py      # Optimizer/LR schedule ablation (d=50)
│   └── greeks_extraction.py       # Delta & Vega extraction (d=50)
├── pricing_algorithms/            # C++ reference (COS & MC for GBM/Heston)
└── figs/                          # Output figures
```

## Git Ignore Policy

The repository intentionally excludes environment artifacts, generated caches, and large research assets via `.gitignore`:

- Python/build artifacts: `__pycache__/`, `*.py[cod]`, `*.egg-info/`, `dist/`, `build/`, `.eggs/`
- Local environments: `.venv/`, `venv/`, `env/`
- IDE/OS noise: `.vscode/`, `.idea/`, `*.swp`, `*.swo`, `.DS_Store`, `Thumbs.db`
- Notebook/checkpoints: `.ipynb_checkpoints/`
- Model checkpoints: `*.pt`, `*.pth`
- Local tool config: `.cursor/`
- Large document assets (not tracked): `thesis/`, `reference/`


## Quick Start

### Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0 (CUDA recommended)

```bash
pip install -r requirements.txt
```

### Run experiments

```bash
# Phase 1: low-dim accuracy (d=1, 3) — Triple-Net vs COS / MC
python experiments/phase1_accuracy.py

# Phase 2: high-dim scaling (d=10, 50, 100, 200) — Triple-Net vs MC
python experiments/phase2_scaling.py

# Phase 3: barrier level sensitivity (d=50) — price vs barrier level
python experiments/phase3_barrier_level.py

# Phase 4: jump impact on knock-outs (d=50, B=0.7) — price & KO rate vs λ
python experiments/phase4_barrier_jump.py

# Model ablation (d=50) — Single-Net vs Dual-Net vs Triple-Net
python experiments/ablation_model.py

# Training-strategy ablation (d=50) — Adam/AdamW × Constant/Scheduled LR
python experiments/ablation_structure.py

# Greeks extraction (d=50) — Delta & Vega vs MC finite difference
python experiments/greeks_extraction.py
```

### Train a single model

```python
from src.config import BatesConfig
from src.solver_triple import train

cfg = BatesConfig(d=50, epochs=3000)
model, losses, y0s, elapsed = train(cfg)
```

## Key Results

### Pricing Accuracy & Computational Scaling (Basket Option)

| Dim (d) | MC Benchmark (95% CI) | Our Model | MC Greeks Time (s) | Our Time / 1k epochs (s) |
|---------|----------------------|-----------|-------------------|--------------------------|
| 1 (COS) | 0.1371 | 0.1371 | — | 213.7 |
| 3 | 0.1061 ± 0.0001 | 0.1061 | 7.8 | 213.5 |
| 10 | 0.0904 ± 0.0001 | 0.0904 | 22.4 | 214.0 |
| 50 | 0.0835 ± 0.0000 | 0.0836 | 105.2 | 215.7 |
| 100 | 0.0826 ± 0.0000 | 0.0827 | 208.2 | 215.7 |
| 200 | 0.0821 ± 0.0000 | 0.0822 | 480.7 | 216.2 |

MC Greeks extraction scales as $\mathcal{O}(d^3)$ (481 s at d=200); our framework stays flat at ~216 s.

### Architecture Ablation (d=50, Stress Scenario)

| Architecture | Rel. Error | Terminal MSE |
|-------------|-----------|-------------|
| Single-Net (Han et al. 2018) | 0.0787% | 7.2 × 10⁻³ |
| Dual-Net (Liu & Gu 2023) | 0.0576% | 4.3 × 10⁻³ |
| **Triple-Net (ours)** | **0.0335%** | **3.9 × 10⁻³** |

## Methods Compared

| Method | Networks | Jump Handling | Reference |
|--------|----------|---------------|-----------|
| Triple-Net (ours) | Z_S(d) + Z_v(1) + U(1) | Explicit compensator | Paper Algorithm 1 |
| Dual-Net | Z(d+1) + U(d) | Explicit compensator | Liu & Gu 2023 |
| Single-Net | Z(d+1) only | Not modelled | Han et al. 2018 |
| COS Fourier | — | Analytic CF | Fang & Oosterlee 2008 |
| Monte Carlo | — | Simulated jumps | Antithetic variates |

## Default Parameters

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Initial variance | v₀ | 0.04 |
| Mean-reversion speed | κ | 2.0 |
| Long-run variance | θ | 0.04 |
| Vol of vol | σ_v | 0.3 |
| Asset-variance correlation | ρ_sv | −0.5 |
| Inter-asset correlation | ρ_assets | 0.3 |
| Jump intensity | λ | 1.0 |
| Jump mean (log) | μ_J | −0.1 |
| Jump std (log) | σ_J | 0.2 |
| Risk-free rate | r | 0.05 |
| Strike | K | 1.0 |
| Maturity | T | 1.0 |

## Acknowledgment

The C++ engine for the COS method baseline used in Experiment 1 is built upon the work by [Luis Taneda](https://github.com/luisontaneda/option_pricing_pybind11).
