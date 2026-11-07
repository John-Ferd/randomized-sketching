"""Run the three experiments in the README and save results and plots.

1. Accuracy vs sketch size on an incoherent matrix (all four sketches),
   compared with the exact Gaussian formula d / (m - d - 1).
2. The same on a coherent matrix, where uniform sampling fails.
3. Subspace-embedding distortion of the Gaussian sketch vs the sqrt(d/m) rule.
4. Wall-clock time: exact solve vs CountSketch vs Gaussian sketch-and-solve.

Writes artifacts/results.json, accuracy.png and embedding.png.
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sketching import SKETCHES, exact_lstsq, leverage_scores, relative_excess_residual, sketch_and_solve
from sketching.data import coherent, incoherent
from sketching.least_squares import gaussian_theory
from sketching.sketches import countsketch, gaussian, leverage_sampling

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--trials", type=int, default=20)
ap.add_argument("--quick", action="store_true", help="small sizes for CI")
args = ap.parse_args()
out = Path(__file__).resolve().parents[1] / "artifacts"
if args.quick:                       # CI run: never overwrite the real results
    out = out / "quick"
out.mkdir(parents=True, exist_ok=True)

n, d = (4000, 20) if args.quick else (20_000, 50)
ms = [2 * d + 20, 4 * d, 10 * d, 20 * d, 40 * d, 100 * d]
trials = 3 if args.quick else args.trials
res = {"n": n, "d": d, "sketch_sizes": ms, "trials": trials}


def accuracy(A, b):
    x_star = exact_lstsq(A, b)
    lev = leverage_scores(A)
    table = {}
    for name, f in SKETCHES.items():
        if name == "leverage":
            f = lambda A_, b_, m_, r_: leverage_sampling(A_, b_, m_, r_, scores=lev)  # noqa: E731
        rows = []
        for m in ms:
            errs = [relative_excess_residual(A, b, sketch_and_solve(A, b, m, f, np.random.default_rng(t)), x_star)
                    for t in range(trials)]
            q = np.quantile(errs, [0.25, 0.5, 0.75])
            rows.append({"m": m, "median": float(q[1]), "q25": float(q[0]), "q75": float(q[2])})
        table[name] = rows
    return table, lev


acc_inc, lev_inc = accuracy(*incoherent(n, d))
acc_coh, lev_coh = accuracy(*coherent(n, d))
res["incoherent"] = acc_inc
res["coherent"] = acc_coh
res["gaussian_theory"] = {m: gaussian_theory(m, d) for m in ms}
res["max_leverage"] = {"incoherent": float(lev_inc.max()), "coherent": float(lev_coh.max())}

# 3. subspace-embedding distortion for the Gaussian sketch
A, b = incoherent(n, d)
U, _ = np.linalg.qr(A)
emb = []
for m in ms:
    s = [np.linalg.svd(gaussian(U, b, m, np.random.default_rng(t))[0], compute_uv=False) for t in range(trials)]
    emb.append({"m": m, "sigma_min": float(np.median([v.min() for v in s])),
                "sigma_max": float(np.median([v.max() for v in s])), "sqrt_d_over_m": float(np.sqrt(d / m))})
res["embedding"] = emb

# 4. timing on a taller matrix
N = 20_000 if args.quick else 200_000
A, b = incoherent(N, d, seed=1)
m = 20 * d


def best_of_3(fn):
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return round(min(times), 4)


timing = {
    "exact (numpy lstsq)": best_of_3(lambda: exact_lstsq(A, b)),
    "countsketch + solve": best_of_3(lambda: sketch_and_solve(A, b, m, countsketch, np.random.default_rng(0))),
    "gaussian + solve": best_of_3(lambda: sketch_and_solve(A, b, m, gaussian, np.random.default_rng(0))),
}
x_star = exact_lstsq(A, b)
timing["countsketch_excess_residual"] = relative_excess_residual(
    A, b, sketch_and_solve(A, b, m, countsketch, np.random.default_rng(0)), x_star)
res["timing"] = {"n": N, "d": d, "m": m, "seconds_best_of_3": timing}

(out / "results.json").write_text(json.dumps(res, indent=2))

# ---- print
for title, acc in (("incoherent", acc_inc), ("coherent", acc_coh)):
    print(f"\n{title} matrix: median relative excess residual over {trials} trials")
    print("m".rjust(6) + "".join(k.rjust(13) for k in acc) + "   Gaussian theory")
    for i, m_ in enumerate(ms):
        print(f"{m_:6d}" + "".join(f"{acc[k][i]['median']:13.4f}" for k in acc) + f"   {gaussian_theory(m_, d):.4f}")
print("\nGaussian embedding: singular values of S U")
for e in emb:
    print(f"m={e['m']:5d}  [{e['sigma_min']:.3f}, {e['sigma_max']:.3f}]   1 -/+ sqrt(d/m) = "
          f"[{1 - e['sqrt_d_over_m']:.3f}, {1 + e['sqrt_d_over_m']:.3f}]")
print("\nTiming:", json.dumps(res["timing"], indent=2))

# ---- plots
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
panels = (("Incoherent matrix", acc_inc), ("Coherent matrix (d/2 rows with leverage 1)", acc_coh))
for ax, (title, acc) in zip(axes, panels):
    for name, rows in acc.items():
        x = [r["m"] for r in rows]
        ax.plot(x, [r["median"] for r in rows], "o-", label=name)
        ax.fill_between(x, [r["q25"] for r in rows], [r["q75"] for r in rows], alpha=0.15)
    ax.plot(ms, [gaussian_theory(m_, d) for m_ in ms], "k--", lw=1, label="Gaussian theory d/(m-d-1)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("sketch size m (rows kept)")
    ax.set_title(title)
axes[0].set_ylabel("relative excess residual")
axes[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(out / "accuracy.png", dpi=130)
plt.close(fig)

fig, ax = plt.subplots(figsize=(5.5, 3.5))
x = [e["m"] for e in emb]
ax.fill_between(x, [1 - e["sqrt_d_over_m"] for e in emb], [1 + e["sqrt_d_over_m"] for e in emb],
                color="gray", alpha=0.2, label="1 ± sqrt(d/m)")
ax.plot(x, [e["sigma_min"] for e in emb], "o-", label="smallest singular value of SU")
ax.plot(x, [e["sigma_max"] for e in emb], "o-", label="largest singular value of SU")
ax.set_xscale("log")
ax.set_xlabel("sketch size m")
ax.legend(fontsize=8)
ax.set_title("Gaussian sketch as a subspace embedding")
fig.tight_layout()
fig.savefig(out / "embedding.png", dpi=130)
plt.close(fig)
