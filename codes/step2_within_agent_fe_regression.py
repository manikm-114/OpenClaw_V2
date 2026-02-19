                                           
from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd

from utils_openclaw import ACTION_PATTERNS, SENSITIVE_PATTERNS

                                                              
                            
                                                              
POSTS_FILE = "posts.csv"
COMMENTS_FILE = "comments_labeled.csv"                                                   

                   
POST_ID_COL = "id"
POST_AUTHOR_COL = "agent_id"
POST_TIME_COL = "created_at"
POST_TITLE_COL = "title"
POST_CONTENT_COL = "content"

                              
CMT_ID_COL = "id"
CMT_POST_ID_COL = "post_id"
CMT_AUTHOR_COL = "agent_id"
CMT_PARENT_ID_COL = "parent_id"
CMT_TIME_COL = "created_at"
CMT_TEXT_COL = "content"
CMT_RESPONSE_TYPE_COL = "response_type"

                      
CORRECTIVE_LABEL = "corrective"

                                
DI_CAP = 10

                                                              
M_LIST = [5, 10, 20]

                                                                                            
REQUIRE_PRE_AND_POST = True

                                                                  
RNG_SEED = 0

                                                              
         
                                                              
def safe_parse_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True)

def compute_di(text: str, action_pats, sensitive_pats, cap: int = 10) -> int:
    t = (text or "").lower()
    a = sum(1 for pat in action_pats if re.search(pat, t))
    s = sum(1 for pat in sensitive_pats if re.search(pat, t))
    return int(min(a + s, cap))

def ensure_str_series(x: pd.Series) -> pd.Series:
                                               
    out = x.copy()
    mask = out.notna()
    out.loc[mask] = out.loc[mask].astype(str)
    return out

def cluster_se_single_regressor(x: np.ndarray, y: np.ndarray, beta: float, clusters: np.ndarray) -> float:
    """
    Cluster-robust SE for single-regressor OLS (no intercept), using Liang-Zeger style:
    Var(beta) = (1/Sxx^2) * sum_g (sum_{i in g} x_i * e_i)^2
    with small-sample correction.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    clusters = np.asarray(clusters)

    e = y - beta * x
    Sxx = float(np.sum(x * x))
    if Sxx <= 0:
        return np.nan

                
    df = pd.DataFrame({"cl": clusters, "xe": x * e})
    gsum = df.groupby("cl", dropna=True)["xe"].sum().to_numpy(dtype=float)

    meat = float(np.sum(gsum * gsum))

                             
    G = len(gsum)
    N = len(x)
    if G <= 1 or N <= 2:
        return np.nan
                   
    corr = (G / (G - 1)) * ((N - 1) / (N - 1 - 1))                      
    varb = corr * meat / (Sxx * Sxx)
    return float(np.sqrt(max(varb, 0.0)))

def within_demean_by_group(df: pd.DataFrame, group_col: str, cols: list[str]) -> pd.DataFrame:
    """
    Add demeaned columns: col_dm = col - mean(col | group)
    """
    out = df.copy()
    gmeans = out.groupby(group_col, dropna=True)[cols].transform("mean")
    for c in cols:
        out[c + "_dm"] = out[c] - gmeans[c]
    return out

                                                              
                                                           
                                                              
def load_and_compute_di(posts: pd.DataFrame, cmts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    posts = posts.copy()
    cmts = cmts.copy()

                     
    posts[POST_ID_COL] = ensure_str_series(posts[POST_ID_COL])
    posts[POST_AUTHOR_COL] = ensure_str_series(posts[POST_AUTHOR_COL])
    posts["created_at_dt"] = safe_parse_datetime(posts[POST_TIME_COL])
    posts = posts.dropna(subset=["created_at_dt", POST_ID_COL, POST_AUTHOR_COL]).copy()
    posts[POST_ID_COL] = posts[POST_ID_COL].astype(str)
    posts[POST_AUTHOR_COL] = posts[POST_AUTHOR_COL].astype(str)

    cmts[CMT_ID_COL] = ensure_str_series(cmts[CMT_ID_COL])
    cmts[CMT_POST_ID_COL] = ensure_str_series(cmts[CMT_POST_ID_COL])
    cmts[CMT_AUTHOR_COL] = ensure_str_series(cmts[CMT_AUTHOR_COL])
    cmts[CMT_PARENT_ID_COL] = ensure_str_series(cmts[CMT_PARENT_ID_COL])
    cmts["created_at_dt"] = safe_parse_datetime(cmts[CMT_TIME_COL])
    cmts = cmts.dropna(subset=["created_at_dt", CMT_ID_COL, CMT_POST_ID_COL, CMT_AUTHOR_COL]).copy()
    cmts[CMT_ID_COL] = cmts[CMT_ID_COL].astype(str)
    cmts[CMT_POST_ID_COL] = cmts[CMT_POST_ID_COL].astype(str)
    cmts[CMT_AUTHOR_COL] = cmts[CMT_AUTHOR_COL].astype(str)
    cmts[CMT_PARENT_ID_COL] = cmts[CMT_PARENT_ID_COL].fillna("").astype(str)

    cmts[CMT_RESPONSE_TYPE_COL] = cmts[CMT_RESPONSE_TYPE_COL].fillna("neutral").astype(str).str.lower()

                                    
    posts["text_for_di"] = (
        posts[POST_TITLE_COL].fillna("").astype(str) + " " + posts[POST_CONTENT_COL].fillna("").astype(str)
    ).str.strip()
    posts["di_post"] = posts["text_for_di"].apply(
        lambda t: compute_di(t, ACTION_PATTERNS, SENSITIVE_PATTERNS, DI_CAP)
    ).astype(int)

                               
    cmts["di_comment"] = cmts[CMT_TEXT_COL].fillna("").astype(str).apply(
        lambda t: compute_di(t, ACTION_PATTERNS, SENSITIVE_PATTERNS, DI_CAP)
    ).astype(int)

    return posts, cmts

def build_contributions(posts: pd.DataFrame, cmts: pd.DataFrame) -> pd.DataFrame:
    """
    One unified contribution stream: each row is a post or comment authored by an agent.
    Columns: agent_id, item_type, item_id, created_at_dt, di
    """
    contrib_posts = posts[[POST_AUTHOR_COL, POST_ID_COL, "created_at_dt", "di_post"]].copy()
    contrib_posts = contrib_posts.rename(
        columns={POST_AUTHOR_COL: "agent_id", POST_ID_COL: "item_id", "di_post": "di"}
    )
    contrib_posts["item_type"] = "post"

    contrib_cmts = cmts[[CMT_AUTHOR_COL, CMT_ID_COL, "created_at_dt", "di_comment"]].copy()
    contrib_cmts = contrib_cmts.rename(
        columns={CMT_AUTHOR_COL: "agent_id", CMT_ID_COL: "item_id", "di_comment": "di"}
    )
    contrib_cmts["item_type"] = "comment"

    contrib = pd.concat([contrib_posts, contrib_cmts], ignore_index=True)

                                              
    contrib["agent_id"] = ensure_str_series(contrib["agent_id"])
    contrib = contrib.dropna(subset=["agent_id", "created_at_dt", "di"]).copy()
    contrib["agent_id"] = contrib["agent_id"].astype(str)
    contrib = contrib[contrib["agent_id"].str.strip() != ""].copy()

    contrib["item_id"] = ensure_str_series(contrib["item_id"])
    contrib = contrib.dropna(subset=["item_id"]).copy()
    contrib["item_id"] = contrib["item_id"].astype(str)

    contrib = contrib.sort_values(["agent_id", "created_at_dt", "item_type", "item_id"]).reset_index(drop=True)
    contrib["seq"] = contrib.groupby("agent_id").cumcount().astype("Int64")                 
    return contrib

def build_correction_events(posts: pd.DataFrame, cmts: pd.DataFrame) -> pd.DataFrame:
    """
    Build correction events where a corrective comment targets:
      - parent comment if parent_id is set/non-empty, else
      - the post (root)
    Output columns:
      target_type in {"post","comment"}, target_id, target_author_id, t0_dt, post_id, corrective_comment_id
    """
    corr = cmts[cmts[CMT_RESPONSE_TYPE_COL] == CORRECTIVE_LABEL].copy()
    if len(corr) == 0:
        raise ValueError(f"No corrective comments found where {CMT_RESPONSE_TYPE_COL} == '{CORRECTIVE_LABEL}'")

                                            
    post_author = posts.set_index(POST_ID_COL)[POST_AUTHOR_COL].to_dict()
    cmt_author = cmts.set_index(CMT_ID_COL)[CMT_AUTHOR_COL].to_dict()

    events = []
    for _, r in corr.iterrows():
        parent_id = str(r[CMT_PARENT_ID_COL] or "").strip()
        post_id = str(r[CMT_POST_ID_COL])
        t0 = r["created_at_dt"]
        corr_id = str(r[CMT_ID_COL])

        if parent_id != "":
            target_type = "comment"
            target_id = parent_id
            target_author = cmt_author.get(parent_id, None)
        else:
            target_type = "post"
            target_id = post_id
            target_author = post_author.get(post_id, None)

        if target_author is None or pd.isna(target_author):
            continue

        events.append(
            {
                "target_type": target_type,
                "target_id": str(target_id),
                "target_author_id": str(target_author),
                "post_id": str(post_id),
                "t0_dt": t0,
                "corrective_comment_id": corr_id,
            }
        )

    ev = pd.DataFrame(events)
    if len(ev) == 0:
        raise ValueError("No usable correction events (could not resolve target authors).")

                                                                                                            
    ev = (
        ev.sort_values(["target_type", "target_id", "t0_dt"])
        .drop_duplicates(subset=["target_type", "target_id"], keep="first")
        .reset_index(drop=True)
    )
    return ev

def add_treatment_windows(contrib: pd.DataFrame, events: pd.DataFrame, M: int) -> pd.DataFrame:
    """
    For each event for agent A at time t0:
      treat the next M contributions by that agent AFTER t0 (strictly > t0).
    A contribution is treated if it falls into ANY event window for that agent.
    """
    df = contrib.copy()
    df["treated"] = 0

                               
    by_agent = {aid: sub.index.to_numpy() for aid, sub in df.groupby("agent_id", dropna=True)}

    for _, ev in events.iterrows():
        aid = ev["target_author_id"]
        t0 = ev["t0_dt"]

        idx = by_agent.get(aid, None)
        if idx is None or len(idx) == 0:
            continue

        sub = df.loc[idx, ["created_at_dt"]].copy()
                           
        after_mask = sub["created_at_dt"] > t0
        after_idx = sub.index[after_mask]
        if len(after_idx) == 0:
            continue

        treat_idx = after_idx[:M]
        df.loc[treat_idx, "treated"] = 1

    return df

                                                              
                              
                                                              
def run_fe_regression(df: pd.DataFrame, outcome_col: str) -> dict:
    """
    Within-agent FE regression:
      outcome ~ beta * treated + agent FE
    Implemented by demeaning within agent and running single-regressor OLS (no intercept).
    Cluster-robust SE by agent on demeaned model.
    """
    work = df[["agent_id", "treated", outcome_col]].copy()
    work = work.dropna(subset=["agent_id", "treated", outcome_col]).copy()
    work["agent_id"] = work["agent_id"].astype(str)

                                                        
            
    work = within_demean_by_group(work, "agent_id", ["treated", outcome_col])

    x = work["treated_dm"].to_numpy(dtype=float)
    y = work[outcome_col + "_dm"].to_numpy(dtype=float)
    cl = work["agent_id"].to_numpy()

                               
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]; cl = cl[m]

    Sxx = float(np.sum(x * x))
    if Sxx <= 0:
        return {
            "n": int(len(x)),
            "n_agents": int(pd.Series(cl).nunique()),
            "beta": np.nan,
            "se_cluster": np.nan,
            "t": np.nan,
            "note": "No within-agent variation in treated after demeaning (Sxx=0).",
        }

    beta = float(np.sum(x * y) / Sxx)
    se = cluster_se_single_regressor(x, y, beta, cl)
    tval = float(beta / se) if (se is not None and np.isfinite(se) and se > 0) else np.nan

    return {
        "n": int(len(x)),
        "n_agents": int(pd.Series(cl).nunique()),
        "beta": beta,
        "se_cluster": se,
        "t": tval,
        "note": "",
    }

                                                              
      
                                                              
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

                                                                                
    need_posts = {POST_ID_COL, POST_AUTHOR_COL, POST_TIME_COL, POST_TITLE_COL, POST_CONTENT_COL}
    need_cmts = {CMT_ID_COL, CMT_POST_ID_COL, CMT_AUTHOR_COL, CMT_TIME_COL, CMT_TEXT_COL, CMT_RESPONSE_TYPE_COL, CMT_PARENT_ID_COL}
    miss_posts = need_posts - set(posts.columns)
    miss_cmts = need_cmts - set(cmts.columns)
    if miss_posts:
        raise ValueError(f"posts.csv missing columns: {sorted(miss_posts)}")
    if miss_cmts:
        raise ValueError(f"comments_labeled.csv missing columns: {sorted(miss_cmts)}")

    posts, cmts = load_and_compute_di(posts, cmts)
    contrib = build_contributions(posts, cmts)
    events = build_correction_events(posts, cmts)

                                                                                                  
    if REQUIRE_PRE_AND_POST:
        keep = []
                                                 
        times_by_agent = {aid: sub["created_at_dt"].to_numpy() for aid, sub in contrib.groupby("agent_id", dropna=True)}
        for _, ev in events.iterrows():
            aid = ev["target_author_id"]
            t0 = ev["t0_dt"]
            arr = times_by_agent.get(aid, None)
            if arr is None or len(arr) == 0:
                continue
            if np.any(arr < t0) and np.any(arr > t0):
                keep.append(True)
            else:
                keep.append(False)
        if len(keep) == len(events):
            events = events.loc[keep].reset_index(drop=True)

                      
    events_out = results_dir / "step2_fe_events_used.csv"
    events.to_csv(events_out, index=False)

              
    base = contrib.copy()
    base["y_di"] = base["di"].astype(float)
    base["y_pos"] = (base["di"] > 0).astype(float)

    summary_lines = []
    summary_lines += [
        "Step 2 FE: Within-agent fixed-effects regression after receiving correction",
        "===========================================================================",
        "",
        f"Definition: event = corrective comment targets an agent-authored post or comment at time t0",
        "Targeting: if parent_id exists -> targets that parent comment; else targets the post",
        "Treatment: next M contributions by that agent AFTER t0 are marked treated=1",
        "Model: y_it = beta * treated_it + agent FE + error_it",
        "Estimation: within-agent demeaning + OLS (single regressor), cluster-robust SE by agent",
        "",
        f"Events used: {len(events)} (saved: {events_out})",
        f"Total contributions: {len(base)}",
        "",
    ]

    for M in M_LIST:
        dfM = add_treatment_windows(base, events, M=M)

                                               
        treated_rate = float(dfM["treated"].mean()) if len(dfM) else 0.0
        summary_lines += [f"[M={M}] treated share = {treated_rate:.6g}"]

        res_di = run_fe_regression(dfM, "y_di")
        res_pos = run_fe_regression(dfM, "y_pos")

        summary_lines += [
            f"  Outcome y=DI:",
            f"    n={res_di['n']}, agents={res_di['n_agents']}, beta={res_di['beta']:.6g}, se(cluster)={res_di['se_cluster']:.6g}, t={res_di['t']:.6g}",
        ]
        if res_di["note"]:
            summary_lines += [f"    note: {res_di['note']}"]

        summary_lines += [
            f"  Outcome y=1[DI>0]:",
            f"    n={res_pos['n']}, agents={res_pos['n_agents']}, beta={res_pos['beta']:.6g}, se(cluster)={res_pos['se_cluster']:.6g}, t={res_pos['t']:.6g}",
        ]
        if res_pos["note"]:
            summary_lines += [f"    note: {res_pos['note']}"]

                                                                        
        out_panel = results_dir / f"step2_fe_panel_M{M}.csv"
        dfM[["agent_id", "item_type", "item_id", "created_at_dt", "di", "treated"]].to_csv(out_panel, index=False)
        summary_lines += [f"  panel saved: {out_panel}", ""]

    out_txt = results_dir / "step2_fe_summary.txt"
    out_txt.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f" - {out_txt}")
    print(f" - {events_out}")

if __name__ == "__main__":
    main()
