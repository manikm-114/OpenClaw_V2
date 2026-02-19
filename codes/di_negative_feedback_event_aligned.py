from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd

from utils_openclaw import ACTION_PATTERNS, SENSITIVE_PATTERNS


                              
                                             
                              
def compute_di_from_patterns(text: str, action_pats, sensitive_pats, cap: int = 10) -> int:
    t = (text or "").lower()
    a = sum(1 for pat in action_pats if re.search(pat, t))
    s = sum(1 for pat in sensitive_pats if re.search(pat, t))
    return int(min(a + s, cap))


def safe_parse_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True)


def summarize_block(sub: pd.DataFrame, high_thresh: int) -> tuple[float, float, float]:
    """Return (mean_di, max_di, high_rate) for sub; NaNs if empty."""
    if sub is None or len(sub) == 0:
        return (np.nan, np.nan, np.nan)
    mean_di = float(sub["di_comment"].mean())
    max_di = float(sub["di_comment"].max())
    high_rate = float((sub["di_comment"] >= high_thresh).mean())
    return (mean_di, max_di, high_rate)


def summarize_group(df: pd.DataFrame, delta_mean_col: str, delta_high_col: str, prefix: str) -> dict:
    out = {}
    out[f"{prefix}n"] = int(len(df))
    out[f"{prefix}mean_delta_mean_di"] = float(np.nanmean(df[delta_mean_col])) if len(df) else np.nan
    out[f"{prefix}median_delta_mean_di"] = float(np.nanmedian(df[delta_mean_col])) if len(df) else np.nan
    out[f"{prefix}mean_delta_high_di_rate"] = float(np.nanmean(df[delta_high_col])) if len(df) else np.nan
    out[f"{prefix}median_delta_high_di_rate"] = float(np.nanmedian(df[delta_high_col])) if len(df) else np.nan
    return out


def main():
    project_root = Path(__file__).resolve().parent.parent
    datasets_dir = project_root / "Datasets"
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

                                                                  
                                           
                                                                  
    COMMENTS_FILE = "comments_labeled.csv"                                    
    T_HOURS = 12                                                  
    USE_N_WINDOW = True
    N = 5                                                                  
    HIGH_DI_THRESH = 3                                                          
                                                                  

    comments_path = datasets_dir / COMMENTS_FILE
    if not comments_path.exists():
        raise FileNotFoundError(f"Missing: {comments_path}")

    comments = pd.read_csv(comments_path)

                  
    required_cols = {"post_id", "content", "created_at"}
    missing = required_cols - set(comments.columns)
    if missing:
        raise ValueError(f"{COMMENTS_FILE} is missing required columns: {sorted(missing)}")

    if "response_type" not in comments.columns:
        raise ValueError(
            f"{COMMENTS_FILE} does NOT contain 'response_type'.\n"
            "You must run your reply-typing / labeling pipeline first so comments get labeled as:\n"
            "  affirmation / corrective / adversarial / neutral\n"
            "Then re-run this script."
        )

                 
    comments = comments.copy()
    comments["post_id"] = comments["post_id"].astype(str)
    comments["created_at_dt"] = safe_parse_datetime(comments["created_at"])
    comments = comments.dropna(subset=["created_at_dt"]).copy()

    comments["response_type"] = comments["response_type"].fillna("neutral").astype(str).str.lower()
    comments["is_corrective"] = (comments["response_type"] == "corrective").astype(int)

                        
    comments["di_comment"] = comments["content"].fillna("").astype(str).apply(
        lambda t: compute_di_from_patterns(t, ACTION_PATTERNS, SENSITIVE_PATTERNS, cap=10)
    ).astype(int)

                           
    window = pd.Timedelta(hours=T_HOURS)

    rows = []
    for post_id, df in comments.groupby("post_id", sort=False):
        df = df.sort_values("created_at_dt")

                                       
        corr_locs = df.index[df["is_corrective"] == 1]
        if len(corr_locs) == 0:
            continue
        t0 = df.loc[corr_locs[0], "created_at_dt"]

                                      
                                   
                                      
        before_mask = (df["created_at_dt"] < t0) & (df["created_at_dt"] >= (t0 - window))
        after_mask  = (df["created_at_dt"] > t0) & (df["created_at_dt"] <= (t0 + window))
        before = df.loc[before_mask]
        after  = df.loc[after_mask]

        n_before = int(len(before))
        n_after  = int(len(after))

        mean_b, max_b, high_b = summarize_block(before, HIGH_DI_THRESH)
        mean_a, max_a, high_a = summarize_block(after, HIGH_DI_THRESH)

        delta_mean = (mean_a - mean_b) if (not np.isnan(mean_a) and not np.isnan(mean_b)) else np.nan
        delta_high = (high_a - high_b) if (not np.isnan(high_a) and not np.isnan(high_b)) else np.nan

                                      
                                    
                                      
        if USE_N_WINDOW:
            before_n = df.loc[df["created_at_dt"] < t0].tail(N)
            after_n  = df.loc[df["created_at_dt"] > t0].head(N)

            nwin_n_before = int(len(before_n))
            nwin_n_after  = int(len(after_n))

            mean_bn, max_bn, high_bn = summarize_block(before_n, HIGH_DI_THRESH)
            mean_an, max_an, high_an = summarize_block(after_n, HIGH_DI_THRESH)

            nwin_delta_mean = (mean_an - mean_bn) if (not np.isnan(mean_an) and not np.isnan(mean_bn)) else np.nan
            nwin_delta_high = (high_an - high_bn) if (not np.isnan(high_an) and not np.isnan(high_bn)) else np.nan
        else:
            nwin_n_before = 0
            nwin_n_after = 0
            mean_bn = max_bn = high_bn = np.nan
            mean_an = max_an = high_an = np.nan
            nwin_delta_mean = np.nan
            nwin_delta_high = np.nan

        rows.append({
            "post_id": post_id,
            "t0_first_corrective_utc": str(t0),
            "window_hours": int(T_HOURS),
            "high_di_thresh": int(HIGH_DI_THRESH),

                               
            "n_before": n_before,
            "n_after": n_after,
            "mean_di_before": mean_b,
            "mean_di_after": mean_a,
            "max_di_before": max_b,
            "max_di_after": max_a,
            "high_di_rate_before": high_b,
            "high_di_rate_after": high_a,
            "delta_mean_di": delta_mean,
            "delta_high_di_rate": delta_high,

                            
            "nwin_enabled": int(bool(USE_N_WINDOW)),
            "nwin_N": int(N) if USE_N_WINDOW else 0,
            "nwin_n_before": nwin_n_before,
            "nwin_n_after": nwin_n_after,
            "nwin_mean_di_before": mean_bn,
            "nwin_mean_di_after": mean_an,
            "nwin_max_di_before": max_bn,
            "nwin_max_di_after": max_an,
            "nwin_high_di_rate_before": high_bn,
            "nwin_high_di_rate_after": high_an,
            "nwin_delta_mean_di": nwin_delta_mean,
            "nwin_delta_high_di_rate": nwin_delta_high,
        })

    threads = pd.DataFrame(rows)

    out_threads = results_dir / "negative_feedback_event_aligned_threads.csv"
    threads.to_csv(out_threads, index=False)

                                                                  
                 
                                                                  
                                                                               
    usable = threads[(threads["n_before"] > 0) & (threads["n_after"] > 0)].copy()

                                                                                
                                                                     
    regulatable = usable[(usable["max_di_before"] > 0)].copy()

    agg = {}
    agg["n_threads_with_corrective"] = int(len(threads))
    agg["timewindow_T_hours"] = int(T_HOURS)
    agg["high_di_thresh"] = int(HIGH_DI_THRESH)

                        
    agg.update(summarize_group(usable, "delta_mean_di", "delta_high_di_rate", "usable_"))
    agg.update(summarize_group(regulatable, "delta_mean_di", "delta_high_di_rate", "reg_"))

                                                                
    agg["nwin_enabled"] = int(bool(USE_N_WINDOW))
    agg["nwin_N"] = int(N) if USE_N_WINDOW else 0
    if USE_N_WINDOW:
        usable_n = threads[(threads["nwin_n_before"] > 0) & (threads["nwin_n_after"] > 0)].copy()
        regulatable_n = usable_n[(usable_n["nwin_max_di_before"] > 0)].copy()
        agg.update(summarize_group(usable_n, "nwin_delta_mean_di", "nwin_delta_high_di_rate", "nwin_usable_"))
        agg.update(summarize_group(regulatable_n, "nwin_delta_mean_di", "nwin_delta_high_di_rate", "nwin_reg_"))
    else:
                      
        agg["nwin_usable_n"] = 0
        agg["nwin_reg_n"] = 0

    out_agg = results_dir / "negative_feedback_event_aligned_aggregate.csv"
    pd.DataFrame([agg]).to_csv(out_agg, index=False)

                                                                  
                            
                                                                  
    lines = []
    lines.append("Event-aligned negative feedback summary (within-thread)\n")
    lines.append("====================================================\n\n")
    lines.append(f"Input file: {comments_path}\n")
    lines.append(f"Time window: ±{T_HOURS} hours around first corrective reply (t0)\n")
    lines.append(f"N-window: {'ON' if USE_N_WINDOW else 'OFF'}")
    if USE_N_WINDOW:
        lines.append(f" (last N={N} before vs first N={N} after)\n")
    else:
        lines.append("\n")
    lines.append("DI metric: DI_comment computed on comment content using the same DI lexicon as posts\n")
    lines.append(f"High-DI threshold: DI_comment >= {HIGH_DI_THRESH}\n\n")

    lines.append(f"Threads with at least one corrective reply: {agg['n_threads_with_corrective']}\n\n")

    lines.append("TIME-WINDOW (±T hours) results\n")
    lines.append("-----------------------------\n")
    lines.append(f"Usable threads (>=1 before and after): {agg.get('usable_n', np.nan)}\n")
    lines.append("Descriptive paired changes (after - before):\n")
    lines.append(f"  mean(delta mean DI_comment):   {agg.get('usable_mean_delta_mean_di', np.nan)}\n")
    lines.append(f"  median(delta mean DI_comment): {agg.get('usable_median_delta_mean_di', np.nan)}\n")
    lines.append(f"  mean(delta high-DI rate):      {agg.get('usable_mean_delta_high_di_rate', np.nan)}\n")
    lines.append(f"  median(delta high-DI rate):    {agg.get('usable_median_delta_high_di_rate', np.nan)}\n\n")

    lines.append("Regulatable threads (usable AND max DI_comment before > 0): "
                 f"{agg.get('reg_n', np.nan)}\n")
    lines.append("Descriptive paired changes (after - before):\n")
    lines.append(f"  mean(delta mean DI_comment):   {agg.get('reg_mean_delta_mean_di', np.nan)}\n")
    lines.append(f"  median(delta mean DI_comment): {agg.get('reg_median_delta_mean_di', np.nan)}\n")
    lines.append(f"  mean(delta high-DI rate):      {agg.get('reg_mean_delta_high_di_rate', np.nan)}\n")
    lines.append(f"  median(delta high-DI rate):    {agg.get('reg_median_delta_high_di_rate', np.nan)}\n\n")

    lines.append("N-WINDOW (fixed-N comments) results\n")
    lines.append("----------------------------------\n")
    if USE_N_WINDOW:
        lines.append(f"Usable N-window threads (>=1 before and after): {agg.get('nwin_usable_n', np.nan)}\n")
        lines.append("Descriptive paired changes (after - before):\n")
        lines.append(f"  mean(delta mean DI_comment):   {agg.get('nwin_usable_mean_delta_mean_di', np.nan)}\n")
        lines.append(f"  median(delta mean DI_comment): {agg.get('nwin_usable_median_delta_mean_di', np.nan)}\n")
        lines.append(f"  mean(delta high-DI rate):      {agg.get('nwin_usable_mean_delta_high_di_rate', np.nan)}\n")
        lines.append(f"  median(delta high-DI rate):    {agg.get('nwin_usable_median_delta_high_di_rate', np.nan)}\n\n")

        lines.append("Regulatable N-window threads (usable AND max DI_comment before > 0): "
                     f"{agg.get('nwin_reg_n', np.nan)}\n")
        lines.append("Descriptive paired changes (after - before):\n")
        lines.append(f"  mean(delta mean DI_comment):   {agg.get('nwin_reg_mean_delta_mean_di', np.nan)}\n")
        lines.append(f"  median(delta mean DI_comment): {agg.get('nwin_reg_median_delta_mean_di', np.nan)}\n")
        lines.append(f"  mean(delta high-DI rate):      {agg.get('nwin_reg_mean_delta_high_di_rate', np.nan)}\n")
        lines.append(f"  median(delta high-DI rate):    {agg.get('nwin_reg_median_delta_high_di_rate', np.nan)}\n\n")
    else:
        lines.append("N-window disabled.\n\n")

    lines.append("Files written:\n")
    lines.append(f"  - {out_threads}\n")
    lines.append(f"  - {out_agg}\n")

    out_txt = results_dir / "negative_feedback_event_aligned_summary.txt"
    out_txt.write_text("".join(lines), encoding="utf-8")

    print(f" - {out_txt}")
    print(f" - {out_threads}")
    print(f" - {out_agg}")


if __name__ == "__main__":
    main()
