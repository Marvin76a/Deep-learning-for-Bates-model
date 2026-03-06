from __future__ import annotations

import random
import time
from contextlib import contextmanager
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@contextmanager
def timer(label: str = ""):
    """Context manager that prints elapsed time."""
    t0 = time.time()
    yield
    elapsed = time.time() - t0
    if label:
        print(f"[{label}] {elapsed:.2f}s")


def plot_training(losses, y0s, title: str = "", save_path: Optional[str] = None):
    """Plot loss curve and Y0 convergence side-by-side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(losses)
    ax1.set_yscale("log")
    ax1.set(title=f"Training Loss {title}", xlabel="Epoch", ylabel="MSE")
    ax1.grid(True, which="both", alpha=0.4)

    ax2.plot(y0s, color="C1")
    if y0s:
        ax2.set(title=f"Y0 → {y0s[-1]:.4f}", xlabel="Epoch", ylabel="Price")
    ax2.grid(True)

    fig.tight_layout()
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()
