"""Test matrices for the experiments."""
from __future__ import annotations

import numpy as np


def incoherent(n: int = 20_000, d: int = 50, noise: float = 1.0, seed: int = 0):
    """Gaussian design: every row is about equally important (leverage ~ d/n)."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, d))
    b = A @ rng.standard_normal(d) + noise * rng.standard_normal(n)
    return A, b


def coherent(n: int = 20_000, d: int = 50, noise: float = 1.0, spike: float = 100.0, seed: int = 0):
    """A few rows carry all the information about some directions.

    The first d/2 columns are dense Gaussian. Each of the last d/2 columns is
    nonzero (= ``spike``) in exactly one row, so that row has leverage score 1:
    it is the only row that says anything about its coefficient. A sketch that
    drops it fits that coefficient as 0 and pays (spike * coefficient)^2 in
    squared residual. Uniform sampling keeps a given row with probability only
    about m/n, so it almost always drops most of these rows.
    """
    rng = np.random.default_rng(seed)
    half = d // 2
    A = np.zeros((n, d))
    A[:, :half] = rng.standard_normal((n, half))
    spikes = rng.choice(n, size=d - half, replace=False)
    A[spikes, half + np.arange(d - half)] = spike
    b = A @ rng.standard_normal(d) + noise * rng.standard_normal(n)
    return A, b
