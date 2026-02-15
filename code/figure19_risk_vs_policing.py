# figure19_risk_vs_policing.py
# ------------------------------------------------------------
# STEP 2 (PNAS-ready coupling figure):
#   Main figure: clean binned curve + Wilson 95% CI (+ optional logistic fit)
#   SI figure: post-level scatter (optional)
#
# Outputs:
#   figures/Figure_1_DI_vs_Corrective.png
#   figures/Figure_S1_DI_vs_Corrective_Scatter.png
#   results/figure1_di_corrective_bins.csv
#   results/figureS1_post_level_scatter.csv
# ------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

from utils_openclaw import Paths, load_all, ensure_dirs


# ----------------------------
# Settings you may tune
# ----------------------------
N_POS_QUANTILE_BINS = 3      # bins among DI>0 posts (3 = terciles)
MIN_COMMENTS_PER_POST = 1    # keep unless you want to suppress single-reply posts
MAKE_SI_SCATTER = True
SCATTER_SAMPLE_FRAC = 1.0    # 1.0 = all posts in SI scatter

# Logistic-fit overlay on binned curve (uses post-level data, comment-weighted by n)
ADD_LOGISTIC_FIT = True


# ----------------------------
# Helpers
# ----------------------------
def wilson_ci(k: np.ndarray, n: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Wilson score interval for binomial proportion. Returns (lo, hi)."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    p = np.divide(k, n, out=np.zeros_like(k), where=(n > 0))

    z = norm.ppf(1 - alpha / 2)
    n_safe = np.clip(n, 1e-12, None)

    denom = 1 + (z**2) / n_safe
    center = (p + (z**2) / (2 * n_safe)) / denom
    half = (z * np.sqrt((p * (1 - p) + (z**2) / (4 * n_safe)) / n_safe)) / denom

    lo = np.clip(center - half, 0, 1)
    hi = np.clip(center + half, 0, 1)

    lo = np.where(n > 0, lo, 0.0)
    hi = np.where(n > 0, hi, 0.0)
    return lo, hi


def assign_bins_di0_plus_quantiles(df_posts: pd.DataFrame, n_pos_bins: int) -> pd.Series:
    """
    Bin assignment at POST level:
      - Bin 'DI=0' for DI == 0
      - For DI > 0, quantile bins Q1..Qk based on post DI distribution
    """
    di = pd.to_numeric(df_posts["di"], errors="coerce").fillna(0).astype(int)
    out = pd.Series(index=df_posts.index, dtype=object)
    out.loc[di == 0] = "DI=0"

    pos = df_posts.loc[di > 0, "di"].astype(float)
    if len(pos) == 0:
        return out.fillna("DI=0")

    qs = np.linspace(0, 1, n_pos_bins + 1)
    cuts = np.quantile(pos.to_numpy(), qs)
    cuts = np.unique(cuts)

    if len(cuts) <= 2:
        out.loc[di > 0] = "DI>0"
        return out

    labels = [f"Q{i+1}" for i in range(len(cuts) - 1)]
    b = pd.cut(pos, bins=cuts, include_lowest=True, right=True, labels=labels)
    if b.isna().any():
        out.loc[di > 0] = "DI>0"
        return out

    out.loc[di > 0] = b.astype(str).to_numpy()
    return out


def fit_logistic_post_level(di: np.ndarray, k: np.ndarray, n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple logistic fit using IRLS on binomial outcomes aggregated per post:
      response ~ Binomial(n_i, p_i), logit(p_i) = b0 + b1 * di_i

    Returns:
      grid_di, pred_p
    """
    # Filter valid
    di = np.asarray(di, dtype=float)
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)

    mask = (n > 0) & np.isfinite(di) & np.isfinite(k) & np.isfinite(n)
    di = di[mask]
    k = k[mask]
    n = n[mask]

    # If too small, return empty
    if len(di) < 5:
        return np.array([]), np.array([])

    # IRLS
    X = np.column_stack([np.ones_like(di), di])
    beta = np.zeros(2)

    for _ in range(50):
        eta = X @ beta
        p = 1 / (1 + np.exp(-eta))
        p = np.clip(p, 1e-6, 1 - 1e-6)

        # Working response and weights for binomial
        W = n * p * (1 - p)
        z = eta + (k - n * p) / (n * p * (1 - p))

        # Solve (X'WX) beta = X'Wz
        XtW = X.T * W
        A = XtW @ X
        b = XtW @ z

        try:
            beta_new = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            break

        if np.max(np.abs(beta_new - beta)) < 1e-6:
            beta = beta_new
            break
        beta = beta_new

    grid = np.linspace(float(np.min(di)), float(np.max(di)), 200)
    Xg = np.column_stack([np.ones_like(grid), grid])
    pg = 1 / (1 + np.exp(-(Xg @ beta)))
    return grid, pg


# ----------------------------
# Main
# ----------------------------
def main():
    project_root = Path(__file__).resolve().parent.parent
    datasets_dir = project_root / "Datasets"
    results_dir = project_root / "results"
    figures_dir = project_root / "figures"

    paths = Paths(
        agents_csv=str(datasets_dir / "agents.csv"),
        posts_csv=str(datasets_dir / "posts.csv"),
        comments_csv=str(datasets_dir / "comments.csv"),
        out_dir=str(results_dir),
        fig_dir=str(figures_dir),
    )
    ensure_dirs(paths)

    _, posts, comments = load_all(paths)

    # Standardize IDs
    posts = posts.copy()
    comments = comments.copy()
    posts["id"] = posts["id"].astype(str)
    comments["post_id"] = comments["post_id"].astype(str)

    posts["di"] = pd.to_numeric(posts["di"], errors="coerce").fillna(0).astype(int)
    comments["response_type"] = comments["response_type"].fillna("neutral").astype(str)
    comments["is_corrective"] = (comments["response_type"] == "corrective").astype(int)

    # Aggregate comment outcomes to post level
    post_agg = (
        comments.groupby("post_id")["is_corrective"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"sum": "k_corrective", "count": "n_comments"})
    )
    post_agg["k_corrective"] = pd.to_numeric(post_agg["k_corrective"], errors="coerce").fillna(0).astype(int)
    post_agg["n_comments"] = pd.to_numeric(post_agg["n_comments"], errors="coerce").fillna(0).astype(int)

    df = post_agg.merge(posts[["id", "di"]].rename(columns={"id": "post_id"}), on="post_id", how="left")
    df["di"] = pd.to_numeric(df["di"], errors="coerce").fillna(0).astype(int)

    # Filter
    df = df[df["n_comments"] >= int(MIN_COMMENTS_PER_POST)].copy()

    # Assign post-level bins
    df["di_bin"] = assign_bins_di0_plus_quantiles(df[["di"]], N_POS_QUANTILE_BINS)

    # Bin aggregation (comment-weighted by construction: sums over posts in bin)
    bins = (
        df.groupby("di_bin")
        .agg(
            di_min=("di", "min"),
            di_max=("di", "max"),
            di_med=("di", "median"),
            n_posts=("post_id", "count"),
            n_comments=("n_comments", "sum"),
            k_corrective=("k_corrective", "sum"),
        )
        .reset_index()
    )

    # Order
    def _key(s: str):
        if s == "DI=0":
            return (0, 0)
        if s.startswith("Q"):
            try:
                return (1, int(s[1:]))
            except Exception:
                return (1, 999)
        if s == "DI>0":
            return (2, 0)
        return (3, 0)

    bins = bins.sort_values(by="di_bin", key=lambda c: c.map(_key)).reset_index(drop=True)

    bins["p_corrective"] = np.divide(
        bins["k_corrective"].to_numpy(dtype=float),
        bins["n_comments"].to_numpy(dtype=float),
        out=np.zeros(len(bins), dtype=float),
        where=(bins["n_comments"].to_numpy(dtype=float) > 0),
    )
    lo, hi = wilson_ci(bins["k_corrective"].to_numpy(), bins["n_comments"].to_numpy())
    bins["ci_low"] = lo
    bins["ci_high"] = hi

    # Save main table
    out_bins = results_dir / "figure1_di_corrective_bins.csv"
    bins.to_csv(out_bins, index=False)

    # ----------------------------
    # MAIN FIGURE (PNAS style)
    # ----------------------------
    fig = plt.figure(figsize=(7.6, 4.8))
    ax = plt.gca()

    # x positions are categorical bins
    x = np.arange(len(bins))
    y = bins["p_corrective"].to_numpy(dtype=float)
    yerr = np.vstack([
        np.clip(y - bins["ci_low"].to_numpy(dtype=float), 0, None),
        np.clip(bins["ci_high"].to_numpy(dtype=float) - y, 0, None),
    ])

    ax.errorbar(x, y, yerr=yerr, fmt="o-", capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(bins["di_bin"].tolist())
    ax.set_xlabel("Directive Intensity (DI) bins (posts)")
    ax.set_ylabel("P(Corrective signaling | reply)  (Wilson 95% CI)")
    ax.set_title("Corrective signaling scales with directive intensity (DI)")
    ax.set_ylim(0, max(0.25, float(np.nanmax(bins["ci_high"].to_numpy()) * 1.15)))

    # annotate with n_comments and DI range
    for i, row in bins.iterrows():
        label = f"n={int(row['n_comments'])}\nDI∈[{int(row['di_min'])},{int(row['di_max'])}]"
        ax.text(i, float(row["p_corrective"]) + 0.012, label, ha="center", va="bottom", fontsize=9)

    # Optional logistic fit overlay from post-level data (comment-weighted)
    if ADD_LOGISTIC_FIT:
        grid, pred = fit_logistic_post_level(df["di"].to_numpy(), df["k_corrective"].to_numpy(), df["n_comments"].to_numpy())
        if len(grid) > 0:
            # map grid DI to x-axis positions by placing it on a secondary axis scale:
            # simplest: overlay fit as a smooth curve in DI-space on a twinx with DI scale,
            # but that complicates. Instead, draw a faint fit in DI-space as annotation.
            # We'll plot fit on DI scale along the bottom using a second x-axis.
            ax2 = ax.twiny()
            ax2.set_xlim(float(np.min(df["di"])), float(np.max(df["di"])))
            ax2.plot(grid, pred, alpha=0.55)
            ax2.set_xlabel("DI (for logistic fit overlay)")
            ax2.tick_params(axis="x", labelsize=8)

    plt.tight_layout()
    fig1_path = figures_dir / "Figure_1_DI_vs_Corrective.png"
    plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close()

    # ----------------------------
    # SI SCATTER (optional)
    # ----------------------------
    if MAKE_SI_SCATTER:
        scat = df.copy()
        scat["post_corrective_rate"] = np.divide(
            scat["k_corrective"].to_numpy(dtype=float),
            scat["n_comments"].to_numpy(dtype=float),
            out=np.zeros(len(scat), dtype=float),
            where=(scat["n_comments"].to_numpy(dtype=float) > 0),
        )

        if 0 < SCATTER_SAMPLE_FRAC < 1.0 and len(scat) > 0:
            scat = scat.sample(frac=SCATTER_SAMPLE_FRAC, random_state=7)

        scat_out = results_dir / "figureS1_post_level_scatter.csv"
        scat.to_csv(scat_out, index=False)

        plt.figure(figsize=(7.6, 4.8))
        plt.scatter(scat["di"], scat["post_corrective_rate"], s=14, alpha=0.25)
        plt.xlabel("Post Directive Intensity (DI)")
        plt.ylabel("Post-level corrective rate among replies")
        plt.title("Post-level DI vs corrective signaling (scatter; SI)")
        plt.ylim(-0.02, 1.02)
        plt.tight_layout()
        figS1_path = figures_dir / "Figure_S1_DI_vs_Corrective_Scatter.png"
        plt.savefig(figS1_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        scat_out = None
        figS1_path = None

    print("[OK] Wrote:")
    print(f" - {out_bins}")
    if MAKE_SI_SCATTER:
        print(f" - {scat_out}")
    print("[OK] Figures:")
    print(f" - {fig1_path}")
    if MAKE_SI_SCATTER:
        print(f" - {figS1_path}")


if __name__ == "__main__":
    main()
