"""
Four ways to compress an n x d matrix A (n >> d) into m rows, m << n.

Each function returns the sketched pair (SA, Sb) directly instead of forming
the m x n matrix S, because for CountSketch and sampling S is never needed
explicitly. All four satisfy E[S^T S] = I, so ||S y||^2 is an unbiased
estimate of ||y||^2 for any fixed vector y.

    gaussian          S = G / sqrt(m), G with i.i.d. N(0, 1) entries.
                      Dense; applying it costs O(n d m).
    countsketch       each row i of A is added, with a random sign, to one
                      random row h(i) of the sketch. One pass, O(nnz(A)).
    uniform_sampling  keep m rows chosen uniformly (with replacement),
                      rescaled by sqrt(n / m).
    leverage_sampling keep m rows chosen with probability proportional to
                      their leverage score, rescaled by 1/sqrt(m p_i).

Gaussian and CountSketch are *oblivious*: they do not look at A. The two
sampling schemes do, and the difference between them is the point of the
coherent-matrix experiment: uniform sampling misses rare important rows.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def gaussian(A, b, m, rng):
    G = rng.standard_normal((m, A.shape[0])) / np.sqrt(m)
    return G @ A, G @ b


def countsketch(A, b, m, rng):
    n = A.shape[0]
    rows = rng.integers(0, m, size=n)                 # hash bucket h(i)
    signs = rng.choice([-1.0, 1.0], size=n)           # random sign s(i)
    S = sp.csr_matrix((signs, (rows, np.arange(n))), shape=(m, n))
    return S @ A, S @ b


def leverage_scores(A) -> np.ndarray:
    """l_i = ||U_i||^2 where A = QR (U = Q spans the column space). Sum = rank(A)."""
    Q, _ = np.linalg.qr(A, mode="reduced")
    return np.einsum("ij,ij->i", Q, Q)


def _sample(A, b, m, p, rng):
    idx = rng.choice(A.shape[0], size=m, replace=True, p=p)
    w = 1.0 / np.sqrt(m * p[idx])
    return A[idx] * w[:, None], b[idx] * w


def uniform_sampling(A, b, m, rng):
    n = A.shape[0]
    return _sample(A, b, m, np.full(n, 1.0 / n), rng)


def leverage_sampling(A, b, m, rng, scores=None):
    lev = leverage_scores(A) if scores is None else scores
    return _sample(A, b, m, lev / lev.sum(), rng)


SKETCHES = {
    "gaussian": gaussian,
    "countsketch": countsketch,
    "uniform": uniform_sampling,
    "leverage": leverage_sampling,
}
