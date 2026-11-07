"""
Sketch-and-solve least squares.

Exact:   x* = argmin_x ||A x - b||,   residual r* = A x* - b.
Sketch:  x~ = argmin_x ||S A x - S b||, an m x d problem instead of n x d.

Quality is measured by the relative excess residual

    ( ||A x~ - b||^2 - ||A x* - b||^2 ) / ||A x* - b||^2  =  ||A (x~ - x*)||^2 / ||r*||^2

(the two forms are equal by Pythagoras, because r* is orthogonal to the
column space of A). A value of 0.01 means the sketched solution's squared
residual is 1% worse than optimal.

For a Gaussian sketch this has an exact expectation: d / (m - d - 1) (for
m > d + 1). The experiments check the code against that formula.
"""
from __future__ import annotations

import numpy as np


def exact_lstsq(A, b):
    return np.linalg.lstsq(A, b, rcond=None)[0]


def sketch_and_solve(A, b, m, sketch, rng):
    SA, Sb = sketch(A, b, m, rng)
    return np.linalg.lstsq(np.asarray(SA), np.asarray(Sb).ravel(), rcond=None)[0]


def relative_excess_residual(A, b, x, x_star) -> float:
    r_star = A @ x_star - b
    d = A @ (x - x_star)
    return float(d @ d / (r_star @ r_star))


def gaussian_theory(m: int, d: int) -> float:
    """E[relative excess residual] for a Gaussian sketch with m rows."""
    return d / (m - d - 1)
