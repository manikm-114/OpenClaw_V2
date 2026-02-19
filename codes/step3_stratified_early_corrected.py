from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd

from utils_openclaw import ACTION_PATTERNS, SENSITIVE_PATTERNS

                              
        
                              
POSTS_FILE = "posts.csv"
COMMENTS_FILE = "comments_labeled.csv"

           
POST_ID_COL = "id"
POST_TIME_COL = "created_at"
POST_TITLE_COL = "title"
POST_CONTENT_COL = "content"
POST_AUTHOR_COL = "agent_id"

                      
CMT_ID_COL = "id"
CMT_POST_ID_COL = "post_id"
CMT_TIME_COL = "created_at"
CMT_TEXT_COL = "content"
CMT_RESPONSE_TYPE_COL = "response_type"
CMT_PARENT_ID_COL = "parent_id"
CMT_DI_COL = "di_comment"                               

CORRECTIVE_LABEL = "corrective"

                              
E_LIST = [5]                                            
EARLY_HOURS_LIST = [6]                                          
EARLY_VOLUME_BINS = 3                                             

             
DI_CAP = 10
HIGH_DI = 3                                                           
DI_POST_BIN_MODE = "binary"                                                           
DI_POST_Q = 3                                  

               
                                                                
                   
                          
                                                    
REPORT_DI_POS_ONLY = True

RNG_SEED = 0


                              
         
                              
def safe_parse_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True)


def compute_di(text: str, action_pats, sensitive_pats, cap: int = 10) -> int:
    t = (text or "").lower()
    a = sum(1 for pat in action_pats if re.search(pat, t))
    s = sum(1 for pat in sensitive_pats if re.search(pat, t))
    return int(min(a + s, cap))


def qcut_bins(x: pd.Series, q: int) -> pd.Series:
    """Robust quantile bins with ties."""
    x = pd.to_numeric(x, errors="coerce").fillna(0).astype(float)
    if x.nunique(dropna=True) < q:
        return pd.cut(x, bins=q, labels=False, include_lowest=True)
    try:
        return pd.qcut(x, q=q, labels=False, duplicates="drop")
    except Exception:
        return pd.cut(x, bins=q, labels=False, include_lowest=True)


def di_post_bins(di_post: pd.Series, mode: str, q: int) -> pd.Series:
    di = pd.to_numeric(di_post, errors="coerce").fillna(0).astype(int)
    if mode == "binary":
        return np.where(di == 0, "DI=0", "DI>0")
    if mode == "quantile":
                                                       
        out = np.array(["DI=0"] * len(di), dtype=object)
        mask = di > 0
        if mask.sum() == 0:
            return out
        bins = qcut_bins(di[mask], q=q)
                                     
        out[mask] = ["DI>0_Q" + str(int(b) + 1) for b in bins.astype("Int64")]
        return out
    raise ValueError(f"Unknown DI_POST_BIN_MODE: {mode}")


def stratified_effects(df: pd.DataFrame, group_cols: list[str], early_col: str, y_cols: list[str]) -> pd.DataFrame:
    """
    For each stratum, compare early vs not-early:
      - risk diff for binary y
      - mean diff for continuous y
    Output columns include group size and per-group means.
    """
    rows = []
    for keys, h in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        keys_dict = {c: k for c, k in zip(group_cols, keys)}

        n1 = int((h[early_col] == 1).sum())
        n0 = int((h[early_col] == 0).sum())

        row = dict(keys_dict)
        row.update({"n_early": n1, "n_not_early": n0, "n_total": int(len(h))})

        for y in y_cols:
            if n1 == 0 or n0 == 0:
                row[f"{y}_mean_early"] = np.nan
                row[f"{y}_mean_not_early"] = np.nan
                row[f"{y}_diff"] = np.nan
            else:
                m1 = float(h.loc[h[early_col] == 1, y].mean())
                m0 = float(h.loc[h[early_col] == 0, y].mean())
                row[f"{y}_mean_early"] = m1
                row[f"{y}_mean_not_early"] = m0
                row[f"{y}_diff"] = m1 - m0

        rows.append(row)

    return pd.DataFrame(rows)


def weighted_average_diff(strata_df: pd.DataFrame, diff_col: str, n1_col: str, n0_col: str) -> float:
    """
    Weighted average over strata where both groups exist.
    Uses harmonic-like weight: 2*n1*n0/(n1+n0) (good for diff comparisons).
    """
    ok = strata_df.dropna(subset=[diff_col]).copy()
    if len(ok) == 0:
        return float("nan")
    w = 2 * ok[n1_col].astype(float) * ok[n0_col].astype(float) / (ok[n1_col].astype(float) + ok[n0_col].astype(float))
    if float(w.sum()) == 0:
        return float("nan")
    return float(np.average(ok[diff_col].astype(float), weights=w))


                              
      
                              
def main():
    project_root = Path(__file__).resolve().parent.parent
    datasets_dir = project_root / "Datasets"
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    posts_path = datasets_dir / POSTS_FILE
    cmts_path = datasets_dir / COMMENTS_FILE

    posts = pd.read_csv(posts_path)
    cmts = pd.read_csv(cmts_path)

                                    
    posts = posts.copy()
    cmts = cmts.copy()

    posts[POST_ID_COL] = posts[POST_ID_COL].astype(str)
    posts[POST_AUTHOR_COL] = posts[POST_AUTHOR_COL].astype(str)
    posts["post_dt"] = safe_parse_datetime(posts[POST_TIME_COL])
    posts = posts.dropna(subset=["post_dt"]).copy()

    cmts[CMT_ID_COL] = cmts[CMT_ID_COL].astype(str)
    cmts[CMT_POST_ID_COL] = cmts[CMT_POST_ID_COL].astype(str)
    cmts["cmt_dt"] = safe_parse_datetime(cmts[CMT_TIME_COL])
    cmts = cmts.dropna(subset=["cmt_dt"]).copy()
    cmts[CMT_RESPONSE_TYPE_COL] = cmts[CMT_RESPONSE_TYPE_COL].fillna("neutral").astype(str).str.lower()

                                        
    if CMT_DI_COL not in cmts.columns:
        cmts[CMT_DI_COL] = cmts[CMT_TEXT_COL].fillna("").astype(str).apply(
            lambda t: compute_di(t, ACTION_PATTERNS, SENSITIVE_PATTERNS, DI_CAP)
        ).astype(int)
    else:
        cmts[CMT_DI_COL] = pd.to_numeric(cmts[CMT_DI_COL], errors="coerce").fillna(0).astype(int)

                                       
    posts["text_for_di"] = (
        posts[POST_TITLE_COL].fillna("").astype(str) + " " + posts[POST_CONTENT_COL].fillna("").astype(str)
    ).str.strip()
    posts["di_post"] = posts["text_for_di"].apply(
        lambda t: compute_di(t, ACTION_PATTERNS, SENSITIVE_PATTERNS, DI_CAP)
    ).astype(int)

                                      
    post_time_map = posts.set_index(POST_ID_COL)["post_dt"]
    cmts["post_dt"] = cmts[CMT_POST_ID_COL].map(post_time_map)
    cmts = cmts.dropna(subset=["post_dt"]).copy()

                                  
    cmts = cmts.sort_values([CMT_POST_ID_COL, "cmt_dt", CMT_ID_COL]).reset_index(drop=True)

                                                   
    all_runs = []

    for E in E_LIST:
        for H in EARLY_HOURS_LIST:
            thread_rows = []
            for post_id, g in cmts.groupby(CMT_POST_ID_COL, sort=False):
                g = g.sort_values(["cmt_dt", CMT_ID_COL])
                post_time = g["post_dt"].iloc[0]

                early_by_count = g.head(E)
                early_corrected_E = int((early_by_count[CMT_RESPONSE_TYPE_COL] == CORRECTIVE_LABEL).any())

                early_end = post_time + pd.Timedelta(hours=H)
                g_early_time = g[g["cmt_dt"] <= early_end]
                early_corrected_H = int((g_early_time[CMT_RESPONSE_TYPE_COL] == CORRECTIVE_LABEL).any())
                early_count_T = int(len(g_early_time))

                                                                            
                downstream = g.iloc[E:] if len(g) > E else g.iloc[0:0]

                                                                            
                esc_high = int((downstream[CMT_DI_COL] >= HIGH_DI).any()) if len(downstream) else 0
                max_di_down = int(downstream[CMT_DI_COL].max()) if len(downstream) else 0
                mean_di_down = float(downstream[CMT_DI_COL].mean()) if len(downstream) else 0.0

                                                                                                           
                max_di_early = int(early_by_count[CMT_DI_COL].max()) if len(early_by_count) else 0
                mean_di_early = float(early_by_count[CMT_DI_COL].mean()) if len(early_by_count) else 0.0

                thread_rows.append(
                    {
                        "post_id": post_id,
                        "E": int(E),
                        "H_hours": int(H),
                        "n_comments": int(len(g)),
                        "early_count_T": int(early_count_T),
                        "early_corrected_E": int(early_corrected_E),
                        "early_corrected_H": int(early_corrected_H),
                        "max_di_early": int(max_di_early),
                        "mean_di_early": float(mean_di_early),
                        "esc_high_after_E": int(esc_high),
                        "max_di_after_E": int(max_di_down),
                        "mean_di_after_E": float(mean_di_down),
                        "n_downstream_after_E": int(len(downstream)),
                    }
                )

            threads = pd.DataFrame(thread_rows)

                          
            threads = threads.merge(posts[[POST_ID_COL, "di_post"]], left_on="post_id", right_on=POST_ID_COL, how="left")
            threads = threads.drop(columns=[POST_ID_COL])
            threads = threads.dropna(subset=["di_post"]).copy()
            threads["di_post"] = threads["di_post"].astype(int)

                                 
            threads["di_bin"] = di_post_bins(threads["di_post"], DI_POST_BIN_MODE, DI_POST_Q)
            threads["vol_bin"] = qcut_bins(threads["early_count_T"], EARLY_VOLUME_BINS).astype("Int64")

                                                                            
                                                      
            threads["early_di_bin"] = np.where(threads["max_di_early"] == 0, "earlyDI=0", "earlyDI>0")

                                                                 
            y_cols = ["esc_high_after_E", "max_di_after_E", "mean_di_after_E"]
            group_cols = ["di_bin", "vol_bin", "early_di_bin"]

            strata_E = stratified_effects(threads, group_cols, "early_corrected_E", y_cols)
            strata_H = stratified_effects(threads, group_cols, "early_corrected_H", y_cols)

                                         
            agg = {
                "E": E,
                "H_hours": H,
                "threads_n": int(len(threads)),
                "agg_RD_E_esc_high": weighted_average_diff(strata_E, "esc_high_after_E_diff", "n_early", "n_not_early"),
                "agg_Diff_E_maxDI": weighted_average_diff(strata_E, "max_di_after_E_diff", "n_early", "n_not_early"),
                "agg_Diff_E_meanDI": weighted_average_diff(strata_E, "mean_di_after_E_diff", "n_early", "n_not_early"),
                "agg_RD_H_esc_high": weighted_average_diff(strata_H, "esc_high_after_E_diff", "n_early", "n_not_early"),
                "agg_Diff_H_maxDI": weighted_average_diff(strata_H, "max_di_after_E_diff", "n_early", "n_not_early"),
                "agg_Diff_H_meanDI": weighted_average_diff(strata_H, "mean_di_after_E_diff", "n_early", "n_not_early"),
            }

                                                  
            threads_pos = threads[threads["di_post"] > 0].copy()
            strata_E_pos = stratified_effects(threads_pos, group_cols, "early_corrected_E", y_cols) if len(threads_pos) else pd.DataFrame()
            strata_H_pos = stratified_effects(threads_pos, group_cols, "early_corrected_H", y_cols) if len(threads_pos) else pd.DataFrame()

            agg_pos = {
                "E": E,
                "H_hours": H,
                "threads_DIpos_n": int(len(threads_pos)),
                "agg_RD_E_esc_high_DIpos": weighted_average_diff(strata_E_pos, "esc_high_after_E_diff", "n_early", "n_not_early") if len(strata_E_pos) else float("nan"),
                "agg_Diff_E_maxDI_DIpos": weighted_average_diff(strata_E_pos, "max_di_after_E_diff", "n_early", "n_not_early") if len(strata_E_pos) else float("nan"),
                "agg_Diff_E_meanDI_DIpos": weighted_average_diff(strata_E_pos, "mean_di_after_E_diff", "n_early", "n_not_early") if len(strata_E_pos) else float("nan"),
                "agg_RD_H_esc_high_DIpos": weighted_average_diff(strata_H_pos, "esc_high_after_E_diff", "n_early", "n_not_early") if len(strata_H_pos) else float("nan"),
                "agg_Diff_H_maxDI_DIpos": weighted_average_diff(strata_H_pos, "max_di_after_E_diff", "n_early", "n_not_early") if len(strata_H_pos) else float("nan"),
                "agg_Diff_H_meanDI_DIpos": weighted_average_diff(strata_H_pos, "mean_di_after_E_diff", "n_early", "n_not_early") if len(strata_H_pos) else float("nan"),
            }

                                   
            tag = f"E{E}_H{H}"
            out_threads = results_dir / f"step3_threads_{tag}.csv"
            out_strata_E = results_dir / f"step3_strata_earlyByCount_{tag}.csv"
            out_strata_H = results_dir / f"step3_strata_earlyByTime_{tag}.csv"

            threads.to_csv(out_threads, index=False)
            strata_E.to_csv(out_strata_E, index=False)
            strata_H.to_csv(out_strata_H, index=False)

            all_runs.append(
                {
                    **agg,
                    **agg_pos,
                    "out_threads": str(out_threads),
                    "out_strata_E": str(out_strata_E),
                    "out_strata_H": str(out_strata_H),
                }
            )

    summary = pd.DataFrame(all_runs)

    out_summary_csv = results_dir / "step3_stratified_summary_v2.csv"
    out_summary_txt = results_dir / "step3_stratified_summary_v2.txt"
    summary.to_csv(out_summary_csv, index=False)

                                 
    lines = []
    lines += [
        "Step 3 (v2): Coarse stratified comparison (early-corrected vs not early-corrected)",
        "===============================================================================",
        "",
        f"Binary escalation outcome uses HIGH_DI={HIGH_DI} on comment di_comment.",
        f"Strata = (di_post bin mode: {DI_POST_BIN_MODE}) x (early volume tercile) x (early max DI bin 0 vs >0).",
        "",
        "We report weighted average differences over strata where both groups exist.",
        "For interpretability/power, DI>0-only summaries are also reported.",
        "",
        "Runs:",
        summary.to_string(index=False),
        "",
        "Files written:",
        f"  - {out_summary_csv}",
        f"  - {out_summary_txt}",
        "",
        "Per-run thread/strata CSVs are listed in the summary table (out_threads/out_strata_*).",
    ]
    out_summary_txt.write_text("\n".join(lines), encoding="utf-8")

    print(" -", out_summary_txt)
    print(" -", out_summary_csv)


if __name__ == "__main__":
    main()
