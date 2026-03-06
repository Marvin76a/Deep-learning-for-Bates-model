"""COS Fourier method for 1-D European options under the Bates (SVJDM) model.

The Bates characteristic function factorises as:
    phi_Bates = phi_Heston * phi_MertonJumps

References
----------
* Fang & Oosterlee (2008) – COS method
* Bates (1996) – SVJDM characteristic function
* pricing_algorithms/COS_method/European/stoch_process.cpp – Heston CF
"""

import numpy as np


# -----------------------------------------------------------------------
# Characteristic functions
# -----------------------------------------------------------------------

def _cf_heston(u, T, r, v0, kappa, theta, sigma_v, rho):
    """Log-characteristic function of log(S_T/S_0) under the Heston model.

    Uses the formulation of Schoutens, Simons & Tistaert (2004) with the
    'little Heston trap' sign convention for numerical stability.
    """
    u = np.asarray(u, dtype=complex)
    iu = 1j * u

    d = np.sqrt(
        (rho * sigma_v * iu - kappa) ** 2
        + sigma_v ** 2 * (iu + u ** 2)
    )
    g = (kappa - rho * sigma_v * iu - d) / (kappa - rho * sigma_v * iu + d)

    C = (
        iu * r * T
        + (kappa * theta / sigma_v ** 2)
        * ((kappa - rho * sigma_v * iu - d) * T
           - 2.0 * np.log((1.0 - g * np.exp(-d * T)) / (1.0 - g)))
    )

    D = (
        (kappa - rho * sigma_v * iu - d) / sigma_v ** 2
        * (1.0 - np.exp(-d * T)) / (1.0 - g * np.exp(-d * T))
    )

    return np.exp(C + D * v0)


def _cf_merton_jumps(u, T, lambda_, mu_J, sigma_J):
    """Characteristic function contribution from Merton log-normal jumps.

    k_bar = E[J-1] is subtracted from the drift by the Bates compensator,
    so the full CF multiplier is:
        exp(lambda * T * (exp(i*u*mu_J - 0.5*sigma_J^2*u^2) - 1 - i*u*k_bar))
    """
    u = np.asarray(u, dtype=complex)
    k_bar = np.exp(mu_J + 0.5 * sigma_J ** 2) - 1.0

    jump_cf = np.exp(
        lambda_ * T * (
            np.exp(1j * u * mu_J - 0.5 * sigma_J ** 2 * u ** 2)
            - 1.0
            - 1j * u * k_bar
        )
    )
    return jump_cf


def cf_bates(u, T, r, v0, kappa, theta, sigma_v, rho,
             lambda_, mu_J, sigma_J):
    """Full characteristic function of log(S_T / S_0) under the Bates model."""
    return (
        _cf_heston(u, T, r, v0, kappa, theta, sigma_v, rho)
        * _cf_merton_jumps(u, T, lambda_, mu_J, sigma_J)
    )


# -----------------------------------------------------------------------
# COS payoff coefficients
# -----------------------------------------------------------------------

def _chi(a, b, c, d, k):
    """Helper integral for COS call/put coefficients."""
    k = np.asarray(k, dtype=float)
    kpi = k * np.pi / (b - a)
    denom = 1.0 + kpi ** 2
    val = (
        1.0 / denom
        * (np.cos(kpi * (d - a)) * np.exp(d)
           - np.cos(kpi * (c - a)) * np.exp(c)
           + kpi * np.sin(kpi * (d - a)) * np.exp(d)
           - kpi * np.sin(kpi * (c - a)) * np.exp(c))
    )
    return val


def _psi(a, b, c, d, k):
    """Helper integral for COS call/put coefficients."""
    k = np.asarray(k, dtype=float)
    kpi = k * np.pi / (b - a)
    # Avoid division by zero for k=0; np.where evaluates both branches so
    # we replace zeros before dividing.
    safe_kpi = np.where(np.abs(kpi) < 1e-14, 1.0, kpi)
    val = np.where(
        np.abs(kpi) < 1e-14,
        d - c,
        (np.sin(safe_kpi * (d - a)) - np.sin(safe_kpi * (c - a))) / safe_kpi,
    )
    return val


def _Hk_call(a, b, k):
    return 2.0 / (b - a) * (_chi(a, b, 0.0, b, k) - _psi(a, b, 0.0, b, k))


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

def cos_price_european_call(
    S0: float,
    K: float,
    T: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    lambda_: float,
    mu_J: float,
    sigma_J: float,
    N_cos: int = 2 ** 12,
    L: float = 12.0,
) -> float:
    """Price a European call option under the 1-D Bates model via COS.

    Parameters
    ----------
    N_cos : int
        Number of cosine expansion terms (higher = more accurate).
    L : float
        Truncation range in standard-deviation units.
    """
    x0 = np.log(S0 / K)

    # Truncation interval [a, b] via cumulants (simplified)
    a = -L * np.sqrt(T)
    b =  L * np.sqrt(T)

    k = np.arange(N_cos)
    u_k = k * np.pi / (b - a)

    # Evaluate characteristic function
    cf_vals = cf_bates(u_k, T, r, v0, kappa, theta, sigma_v, rho,
                       lambda_, mu_J, sigma_J)

    # COS coefficients for call payoff
    Hk = _Hk_call(a, b, k)
    Hk[0] *= 0.5  # halve the k=0 term

    summand = np.real(cf_vals * np.exp(1j * u_k * (x0 - a))) * Hk
    price = np.exp(-r * T) * K * np.sum(summand)

    return float(price)


def cos_price_from_config(cfg, N_cos: int = 2 ** 12, L: float = 12.0) -> float:
    """Convenience wrapper that reads parameters from a ``BatesConfig``."""
    return cos_price_european_call(
        S0=cfg.S0,
        K=cfg.K,
        T=cfg.T,
        r=cfg.r,
        v0=cfg.v0,
        kappa=cfg.kappa,
        theta=cfg.theta,
        sigma_v=cfg.sigma_v,
        rho=cfg.rho_sv,
        lambda_=cfg.lambda_,
        mu_J=cfg.mu_J,
        sigma_J=cfg.sigma_J,
        N_cos=N_cos,
        L=L,
    )


if __name__ == "__main__":
    from .config import BatesConfig

    cfg = BatesConfig(d=1)
    price = cos_price_from_config(cfg)
    print(f"COS price (1-D Bates European Call): {price:.8f}")
