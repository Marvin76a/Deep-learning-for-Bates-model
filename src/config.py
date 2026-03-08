import numpy as np
import torch


class BatesConfig:
    """Unified configuration for the high-dimensional Bates (SVJDM) model.

    Supports flexible dimension, payoff type, and training hyperparameters.
    Automatically builds the joint Cholesky factor for correlated Brownian
    increments when the configuration is constructed or ``d`` is changed.
    """

    def __init__(
        self,
        d: int = 50,
        N: int = 100,
        M: int = 1024,
        T: float = 1.0,
        r: float = 0.05,
        q: float = 0.0,
        S0: float = 1.0,
        v0: float = 0.04,
        kappa: float = 2.0,
        theta: float = 0.04,
        sigma_v: float = 0.3,
        rho_assets: float = 0.3,
        rho_sv: float = -0.5,
        lambda_: float = 1.0,
        mu_J: float = -0.1,
        sigma_J: float = 0.2,
        K: float = 1.0,
        payoff_type: str = "basket",
        barrier: float = 0.0,
        rebate: float = 0.0,
        epochs: int = 3000,
        lr: float = 1e-3,
        seed: int = 42,
    ):
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.d = d
        self.T = T
        self.N = N
        self.M = M
        self.r = r
        self.q = q

        self.S0 = S0
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma_v = sigma_v

        self.rho_assets = rho_assets
        self.rho_sv = rho_sv

        self.lambda_ = lambda_
        self.mu_J = mu_J
        self.sigma_J = sigma_J

        self.K = K
        self.payoff_type = payoff_type
        self.barrier = barrier
        self.rebate = rebate

        self.epochs = epochs
        self.lr = lr

        self.weights = None
        self.k_bar = None
        self.L_full = None

        self._build()

    def _build(self):
        d = self.d
        self.weights = np.ones(d) / d

        # k_bar = E[J - 1] = exp(mu_J + sigma_J^2 / 2) - 1
        self.k_bar = np.exp(self.mu_J + 0.5 * self.sigma_J ** 2) - 1

        # Joint (d+1) x (d+1) correlation matrix
        R_S = np.full((d, d), self.rho_assets)
        np.fill_diagonal(R_S, 1.0)
        rho_vec = np.full((d, 1), self.rho_sv)
        R_full = np.block([
            [R_S, rho_vec],
            [rho_vec.T, np.array([[1.0]])],
        ])
        self.L_full = np.linalg.cholesky(R_full)

    def with_d(self, d: int) -> "BatesConfig":
        """Return a copy with a different dimension."""
        import copy
        cfg = copy.copy(self)
        cfg.d = d
        cfg._build()
        return cfg
