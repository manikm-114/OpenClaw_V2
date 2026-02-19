from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.stats import norm

from utils_openclaw import (
    Paths, ensure_dirs, load_all,
    ACTION_PATTERNS, SENSITIVE_PATTERNS
)

                              
           
                              
def wilson_ci(k, n, alpha=0.05):
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

                              
                
                              
def compute_di_from_patterns(text: str, action_pats, sensitive_pats, cap=10) -> int:
    t = (text or "").lower()
    a = sum(1 for pat in action_pats if re.search(pat, t))
    s = sum(1 for pat in sensitive_pats if re.search(pat, t))
    return int(min(a + s, cap))

                                                        
DROP_SUBSTRINGS = [
    r"\bpython\b",
    r"\bbash\b",
    r"\bpowershell\b",
    r"\bcurl\b",
    r"\bdownload\b",
    r"\binstall\b",
    r"\brun\b",
    r"\bexecute\b",
    r"\bcommand\b",
    r"\bcopy\b",
    r"\bpaste\b",
]

def make_variant_action_patterns():
    keep = []
    for p in ACTION_PATTERNS:
                                                                         
        if any(p == d for d in DROP_SUBSTRINGS):
            continue
        keep.append(p)
    return keep

def assign_bins_di0_plus_quantiles(di: pd.Series, n_pos_bins: int = 3) -> pd.Series:
    di = pd.to_numeric(di, errors="coerce").fillna(0).astype(int)
    out = pd.Series(index=di.index, dtype=object)
    out.loc[di == 0] = "DI=0"
    pos = di[di > 0].astype(float)
    if len(pos) == 0:
        return out.fillna("DI=0")
    qs = np.linspace(0, 1, n_pos_bins + 1)
    cuts = np.unique(np.quantile(pos.to_numpy(), qs))
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

def summarize_bins(posts: pd.DataFrame, comments: pd.DataFrame, di_col: str, n_pos_bins=3) -> pd.DataFrame:
                                                                                   
                            
    df = comments.merge(posts[["id", di_col]].rename(columns={"id": "post_id"}), on="post_id", how="left")
    df = df.dropna(subset=[di_col]).copy()
    df[di_col] = pd.to_numeric(df[di_col], errors="coerce").fillna(0).astype(int)
    df["di_bin"] = assign_bins_di0_plus_quantiles(df[di_col], n_pos_bins=n_pos_bins)

    out = (
        df.groupby("di_bin")
        .agg(
            n_comments=("is_corrective", "size"),
            k_corrective=("is_corrective", "sum"),
            di_min=(di_col, "min"),
            di_max=(di_col, "max"),
            di_med=(di_col, "median"),
        )
        .reset_index()
    )
    out["p_corrective"] = out["k_corrective"] / out["n_comments"]
    lo, hi = wilson_ci(out["k_corrective"].to_numpy(), out["n_comments"].to_numpy())
    out["ci_low"] = lo
    out["ci_high"] = hi

           
    def _key(s):
        if s == "DI=0": return (0, 0)
        if s.startswith("Q"):
            try: return (1, int(s[1:]))
            except: return (1, 999)
        if s == "DI>0": return (2, 0)
        return (3, 0)
    out = out.sort_values("di_bin", key=lambda c: c.map(_key)).reset_index(drop=True)
    return out

def main():
    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "results"
    figures_dir = project_root / "figures"
    datasets_dir = project_root / "Datasets"

    paths = Paths(
        agents_csv=str(datasets_dir / "agents.csv"),
        posts_csv=str(datasets_dir / "posts.csv"),
        comments_csv=str(datasets_dir / "comments.csv"),
        out_dir=str(results_dir),
        fig_dir=str(figures_dir),
    )
    ensure_dirs(paths)

    _, posts, comments = load_all(paths)

    posts = posts.copy()
    comments = comments.copy()
    posts["id"] = posts["id"].astype(str)
    comments["post_id"] = comments["post_id"].astype(str)

                                 
    comments["is_corrective"] = (comments["response_type"].fillna("neutral").astype(str) == "corrective").astype(int)

                                                           
    if "text" not in posts.columns:
                                                         
        title = posts.get("title", "").fillna("").astype(str)
        content = posts.get("content", "").fillna("").astype(str)
        posts["text"] = (title + "\n\n" + content).str.strip()

                                             
    variant_action = make_variant_action_patterns()
    posts["di_variant"] = posts["text"].fillna("").astype(str).apply(
        lambda t: compute_di_from_patterns(t, variant_action, SENSITIVE_PATTERNS, cap=10)
    ).astype(int)

    main_bins = summarize_bins(posts, comments, di_col="di", n_pos_bins=3)
    var_bins = summarize_bins(posts, comments, di_col="di_variant", n_pos_bins=3)

    out_main = Path(paths.out_results) / "di_robustness_bins_main.csv"
    out_var = Path(paths.out_results) / "di_robustness_bins_variant.csv"
    main_bins.to_csv(out_main, index=False)
    var_bins.to_csv(out_var, index=False)

                                    
    txt = []
    txt.append("DI lexicon robustness summary\n")
    txt.append("============================\n\n")
    txt.append("Robustness design:\n")
    txt.append("  Variant DI removes broad technical tokens from ACTION_PATTERNS\n")
    txt.append(f"  Removed patterns (exact matches): {', '.join(DROP_SUBSTRINGS)}\n\n")
    txt.append("Key check:\n")
    txt.append("  We recomputed the DI–corrective coupling bins using (i) main DI and (ii) variant DI.\n")
    txt.append("  We report the binned corrective probabilities with Wilson 95% CIs.\n\n")
    txt.append(f"Outputs:\n")
    txt.append(f"  - {out_main}\n")
    txt.append(f"  - {out_var}\n\n")
    txt.append("Main DI bins:\n")
    txt.append(main_bins.to_string(index=False) + "\n\n")
    txt.append("Variant DI bins:\n")
    txt.append(var_bins.to_string(index=False) + "\n\n")
    txt.append("Interpretation guidance (to write in SI):\n")
    txt.append("  If the probabilities increase monotonically across bins in both main and variant DI,\n")
    txt.append("  the coupling result is robust to reasonable lexicon pruning.\n")

    out_txt = Path(paths.out_results) / "di_validation_summary.txt"
    out_txt.write_text("".join(txt), encoding="utf-8")

    print(f" - {out_txt}")
    print(f" - {out_main}")
    print(f" - {out_var}")

if __name__ == "__main__":
    main()
