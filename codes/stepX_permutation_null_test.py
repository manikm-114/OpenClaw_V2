                                      
from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils_openclaw import ACTION_PATTERNS, SENSITIVE_PATTERNS


                              
                            
                              
POSTS_FILE = "posts.csv"
COMMENTS_FILE = "comments_labeled.csv"

                   
POST_ID_COL = "id"
POST_TITLE_COL = "title"
POST_CONTENT_COL = "content"

                              
CMT_POST_ID_COL = "post_id"
CMT_RESPONSE_TYPE_COL = "response_type"

                  
CORRECTIVE_LABEL = "corrective"

                                  
DI_CAP = 10

              
N_PERM = 1000                                 
RNG_SEED = 0

                                
                                              
N_QUANTILES_DI_POS = 6                                              

        
DERIVED_POSTS_WITH_DI = "posts_with_di.csv"
OUT_SLOPES_CSV = "perm_null_slopes.csv"
OUT_SUMMARY_TXT = "perm_null_summary.txt"
OUT_FIG = "Figure_perm_null.png"


                              
         
                              
def compute_di(text: str, action_pats, sensitive_pats, cap: int = 10) -> int:
    t = (text or "").lower()
    a = sum(1 for pat in action_pats if re.search(pat, t))
    s = sum(1 for pat in sensitive_pats if re.search(pat, t))
    return int(min(a + s, cap))


def standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x, ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return x * 0.0
    return (x - mu) / sd


def fit_logit_1var_coef(x: np.ndarray, y: np.ndarray, max_iter: int = 60, tol: float = 1e-8) -> float:
    """
    Logistic regression with intercept + 1 predictor using Newton-Raphson (IRLS).
    Returns beta1 (slope on x). No external deps.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    n = len(y)
    if n == 0:
        return np.nan

    X1 = x
    b0, b1 = 0.0, 0.0

    for _ in range(max_iter):
        eta = b0 + b1 * X1
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        w = p * (1.0 - p)
        w = np.clip(w, 1e-9, None)

                  
        g0 = np.sum(y - p)
        g1 = np.sum((y - p) * X1)

                                     
        h00 = -np.sum(w)
        h01 = -np.sum(w * X1)
        h11 = -np.sum(w * X1 * X1)

        det = h00 * h11 - h01 * h01
        if abs(det) < 1e-12:
            return np.nan

                         
        d0 = ( g0 * h11 - g1 * h01) / det
        d1 = (-g0 * h01 + g1 * h00) / det

        b0_new = b0 - d0
        b1_new = b1 - d1

        if max(abs(b0_new - b0), abs(b1_new - b1)) < tol:
            b0, b1 = b0_new, b1_new
            break

        b0, b1 = b0_new, b1_new

    return float(b1)


def make_di_bins(di_post: pd.Series, n_q: int) -> pd.Series:

    di = pd.to_numeric(di_post, errors="coerce").fillna(0).astype(int)
    out = pd.Series(np.zeros(len(di), dtype=int), index=di.index)

    pos = di[di > 0]
    if len(pos) == 0:
        return out.astype(int)

                                               
    try:
        qbins = pd.qcut(pos, q=n_q, labels=False, duplicates="drop")
        out.loc[pos.index] = (qbins.astype(int) + 1).values
    except Exception:
        cbins = pd.cut(pos.astype(float), bins=n_q, labels=False, include_lowest=True)
        out.loc[pos.index] = (cbins.astype(int) + 1).values

    return out.astype(int)


def binned_slope(di_bins: np.ndarray, y: np.ndarray) -> float:
    """
    Compute binned corrective probability per DI-bin, then regress p_bin on bin_index (OLS).
    Returns slope.
    """
    di_bins = np.asarray(di_bins, dtype=int)
    y = np.asarray(y, dtype=float)

    bins = np.unique(di_bins)
    xs = []
    ps = []
    for b in bins:
        m = di_bins == b
        if m.sum() == 0:
            continue
        xs.append(float(b))
        ps.append(float(np.mean(y[m])))

    if len(xs) < 2:
        return np.nan

    x = np.asarray(xs, dtype=float)
    p = np.asarray(ps, dtype=float)

    x_c = x - x.mean()
    denom = np.sum(x_c * x_c)
    if denom == 0:
        return np.nan
    return float(np.sum(x_c * (p - p.mean())) / denom)


def perm_p_value(null_vals: np.ndarray, true_val: float, side: str = "two-sided") -> float:
    null_vals = np.asarray(null_vals, dtype=float)
    null_vals = null_vals[np.isfinite(null_vals)]
    if len(null_vals) == 0 or not np.isfinite(true_val):
        return np.nan

    if side == "greater":
        return float((np.sum(null_vals >= true_val) + 1) / (len(null_vals) + 1))
    if side == "less":
        return float((np.sum(null_vals <= true_val) + 1) / (len(null_vals) + 1))

    center = np.median(null_vals)
    dist_true = abs(true_val - center)
    dist_null = np.abs(null_vals - center)
    return float((np.sum(dist_null >= dist_true) + 1) / (len(null_vals) + 1))


                              
      
                              
def main():
    project_root = Path(__file__).resolve().parent.parent
    datasets_dir = project_root / "Datasets"
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    posts_path = datasets_dir / POSTS_FILE
    cmts_path = datasets_dir / COMMENTS_FILE

    if not posts_path.exists():
        raise FileNotFoundError(f"Missing: {posts_path}")
    if not cmts_path.exists():
        raise FileNotFoundError(f"Missing: {cmts_path}")

    posts = pd.read_csv(posts_path)
    cmts = pd.read_csv(cmts_path)

                               
    need_posts = {POST_ID_COL, POST_TITLE_COL, POST_CONTENT_COL}
    need_cmts = {CMT_POST_ID_COL, CMT_RESPONSE_TYPE_COL}

    miss_posts = need_posts - set(posts.columns)
    miss_cmts = need_cmts - set(cmts.columns)

    if miss_posts:
        raise ValueError(f"posts.csv missing columns: {sorted(miss_posts)}")
    if miss_cmts:
        raise ValueError(f"comments_labeled.csv missing columns: {sorted(miss_cmts)}")

    posts = posts.copy()
    cmts = cmts.copy()

    posts[POST_ID_COL] = posts[POST_ID_COL].astype(str)
    cmts[CMT_POST_ID_COL] = cmts[CMT_POST_ID_COL].astype(str)
    cmts[CMT_RESPONSE_TYPE_COL] = cmts[CMT_RESPONSE_TYPE_COL].fillna("neutral").astype(str).str.lower()

                                                                                       
    posts["text_for_di"] = (
        posts[POST_TITLE_COL].fillna("").astype(str) + " " + posts[POST_CONTENT_COL].fillna("").astype(str)
    ).str.strip()

    posts["di_post"] = posts["text_for_di"].apply(
        lambda t: compute_di(t, ACTION_PATTERNS, SENSITIVE_PATTERNS, DI_CAP)
    ).astype(int)

                                     
    out_posts_with_di = results_dir / DERIVED_POSTS_WITH_DI
    posts_out = posts.drop(columns=["text_for_di"], errors="ignore")
    posts_out.to_csv(out_posts_with_di, index=False)

                                         
    post_di_map = posts.set_index(POST_ID_COL)["di_post"]
    cmts["di_post"] = cmts[CMT_POST_ID_COL].map(post_di_map)
    cmts = cmts.dropna(subset=["di_post"]).copy()
    cmts["di_post"] = cmts["di_post"].astype(int)

                            
    y_true = (cmts[CMT_RESPONSE_TYPE_COL] == CORRECTIVE_LABEL).astype(int).to_numpy(dtype=int)

                                     
    x_std = standardize(cmts["di_post"].to_numpy(dtype=float))

                              
    di_bins = make_di_bins(cmts["di_post"], N_QUANTILES_DI_POS).to_numpy(dtype=int)

                 
    true_logit = fit_logit_1var_coef(x_std, y_true)
    true_binned = binned_slope(di_bins, y_true)

    rng = np.random.default_rng(RNG_SEED)
    null_logit = np.empty(N_PERM, dtype=float)
    null_binned = np.empty(N_PERM, dtype=float)

    for i in range(N_PERM):
        y_perm = rng.permutation(y_true)
        null_logit[i] = fit_logit_1var_coef(x_std, y_perm)
        null_binned[i] = binned_slope(di_bins, y_perm)

        if (i + 1) % 100 == 0:
            print(f"[perm] {i+1}/{N_PERM}")

                          
    p_logit = perm_p_value(null_logit, true_logit, side="two-sided")
    p_binned = perm_p_value(null_binned, true_binned, side="two-sided")

                      
    out_slopes = results_dir / OUT_SLOPES_CSV
    pd.DataFrame({"null_logit_beta": null_logit, "null_binned_slope": null_binned}).to_csv(out_slopes, index=False)

                    
    fig_path = results_dir / OUT_FIG
    fig = plt.figure(figsize=(10, 4.2))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.hist(null_logit[np.isfinite(null_logit)], bins=40)
    ax1.axvline(true_logit, linewidth=2, color="red")
    ax1.set_title("Null slopes: logistic beta")
    ax1.set_xlabel("beta (DI_std → corrective)")
    ax1.set_ylabel("count")
    ax1.text(
        0.02, 0.98,
        f"true={true_logit:.4g}\nperm p={p_logit:.3g}",
        transform=ax1.transAxes, va="top", ha="left"
    )

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.hist(null_binned[np.isfinite(null_binned)], bins=40)
    ax2.axvline(true_binned, linewidth=2, color="red")
    ax2.set_title("Null slopes: binned probability slope")
    ax2.set_xlabel("slope (p_corrective vs DI-bin index)")
    ax2.set_ylabel("count")
    ax2.text(
        0.02, 0.98,
        f"true={true_binned:.4g}\nperm p={p_binned:.3g}",
        transform=ax2.transAxes, va="top", ha="left"
    )

    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

                   
    out_txt = results_dir / OUT_SUMMARY_TXT
    out_txt.write_text(
        "\n".join(
            [
                "Permutation null test: shuffle corrective labels across comments",
                "=============================================================",
                "",
                f"Comments used: {len(cmts)}",
                f"Permutations: {N_PERM}",
                f"DI definition: regex lexicon (ACTION_PATTERNS + SENSITIVE_PATTERNS), cap={DI_CAP}, post text = title+content",
                f"DI bins: DI=0 + {N_QUANTILES_DI_POS} quantile bins over DI>0",
                "",
                "[TRUE]",
                f"  logistic beta (DI_std -> corrective): {true_logit:.6g}",
                f"  binned slope (p_corrective vs DI-bin index): {true_binned:.6g}",
                "",
                "[NULL]",
                f"  logistic beta: mean={np.nanmean(null_logit):.6g}, sd={np.nanstd(null_logit):.6g}",
                f"  binned slope: mean={np.nanmean(null_binned):.6g}, sd={np.nanstd(null_binned):.6g}",
                "",
                "[PERMUTATION P-VALUES (two-sided)]",
                f"  logistic beta p_perm: {p_logit:.6g}",
                f"  binned slope p_perm: {p_binned:.6g}",
                "",
                "Files written:",
                f"  - {out_posts_with_di}",
                f"  - {out_slopes}",
                f"  - {fig_path}",
                f"  - {out_txt}",
            ]
        ),
        encoding="utf-8",
    )

    print(" -", out_posts_with_di)
    print(" -", out_slopes)
    print(" -", fig_path)
    print(" -", out_txt)


if __name__ == "__main__":
    main()
