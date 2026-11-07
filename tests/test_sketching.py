import numpy as np
import pytest

from sketching import SKETCHES, exact_lstsq, leverage_scores, relative_excess_residual, sketch_and_solve
from sketching.data import coherent, incoherent
from sketching.least_squares import gaussian_theory
from sketching.sketches import countsketch, gaussian


@pytest.mark.parametrize("name", list(SKETCHES))
def test_sketch_preserves_squared_norm_on_average(name):
    # E||S y||^2 = ||y||^2 for every sketch (they all satisfy E[S^T S] = I).
    rng = np.random.default_rng(0)
    A, b = incoherent(n=2000, d=5, seed=1)
    y = A[:, 0]
    vals = [np.sum(np.asarray(SKETCHES[name](A, y, 200, rng)[1]) ** 2) for _ in range(400)]
    assert abs(np.mean(vals) / (y @ y) - 1) < 0.05


def test_countsketch_matches_explicit_matrix():
    A, b = incoherent(n=50, d=3)
    SA, Sb = countsketch(A, b, 10, np.random.default_rng(3))
    # rebuild S from the same random draws
    rng = np.random.default_rng(3)
    rows = rng.integers(0, 10, size=50)
    signs = rng.choice([-1.0, 1.0], size=50)
    S = np.zeros((10, 50))
    S[rows, np.arange(50)] = signs
    np.testing.assert_allclose(SA, S @ A)
    np.testing.assert_allclose(Sb, S @ b)
    assert np.all(np.count_nonzero(S, axis=0) == 1)       # one nonzero per column


def test_leverage_scores_sum_to_d_and_find_spike_rows():
    A, _ = coherent(n=3000, d=10, seed=2)
    lev = leverage_scores(A)
    assert np.isclose(lev.sum(), 10)
    assert np.all((lev >= -1e-12) & (lev <= 1 + 1e-12))
    assert np.sum(lev > 0.999) == 5                        # the d/2 spike rows


def test_excess_residual_identity():
    A, b = incoherent(n=500, d=5)
    x_star = exact_lstsq(A, b)
    x = x_star + 0.1
    lhs = relative_excess_residual(A, b, x, x_star)
    r, r_star = A @ x - b, A @ x_star - b
    assert np.isclose(lhs, (r @ r - r_star @ r_star) / (r_star @ r_star))


def test_gaussian_sketch_matches_exact_expectation():
    A, b = incoherent(n=3000, d=10, seed=4)
    x_star = exact_lstsq(A, b)
    m = 60
    errs = [relative_excess_residual(A, b, sketch_and_solve(A, b, m, gaussian, np.random.default_rng(t)), x_star)
            for t in range(400)]
    assert abs(np.mean(errs) / gaussian_theory(m, 10) - 1) < 0.15


def test_uniform_sampling_fails_on_coherent_matrix_but_leverage_does_not():
    A, b = coherent(n=5000, d=10, seed=5)
    x_star = exact_lstsq(A, b)
    err = {k: np.median([relative_excess_residual(A, b, sketch_and_solve(A, b, 500, SKETCHES[k],
                                                                         np.random.default_rng(t)), x_star)
                         for t in range(5)]) for k in ("uniform", "leverage")}
    assert err["uniform"] > 1.0 and err["leverage"] < 0.1


def test_gaussian_subspace_embedding_bounds():
    A, b = incoherent(n=4000, d=20)
    U, _ = np.linalg.qr(A)
    s = np.linalg.svd(gaussian(U, b, 800, np.random.default_rng(0))[0], compute_uv=False)
    eps = np.sqrt(20 / 800)
    assert 1 - eps - 0.05 < s.min() and s.max() < 1 + eps + 0.05
