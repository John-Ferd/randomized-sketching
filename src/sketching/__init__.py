"""Randomized sketching for least squares: Gaussian, CountSketch, uniform and
leverage-score row sampling, with sketch-and-solve and embedding diagnostics."""

from .least_squares import exact_lstsq, relative_excess_residual, sketch_and_solve
from .sketches import SKETCHES, countsketch, gaussian, leverage_sampling, leverage_scores, uniform_sampling

__all__ = [
    "SKETCHES",
    "countsketch",
    "exact_lstsq",
    "gaussian",
    "leverage_sampling",
    "leverage_scores",
    "relative_excess_residual",
    "sketch_and_solve",
    "uniform_sampling",
]
__version__ = "0.1.0"
