from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


                              
                               
                              
@dataclass
class Paths:
    """
    Resolves paths relative to Codes/utils_openclaw.py so scripts work
    regardless of current working directory.

    Data CSVs:   <root>/Datasets/*.csv
    Outputs:     <root>/results, <root>/figures
    """
    agents_csv: str | Path | None = None
    posts_csv: str | Path | None = None
    comments_csv: str | Path | None = None

    out_dir: str | Path | None = None
    fig_dir: str | Path | None = None

             
    codes_dir: Path = Path(__file__).resolve().parent                     
    root_dir: Path = Path(__file__).resolve().parent.parent                  
    data_dir: Path = Path(__file__).resolve().parent.parent / "Datasets"

    def __post_init__(self):
                      
        self.agents_csv = Path(self.agents_csv) if self.agents_csv is not None else (self.data_dir / "agents.csv")
        self.posts_csv = Path(self.posts_csv) if self.posts_csv is not None else (self.data_dir / "posts.csv")
        self.comments_csv = Path(self.comments_csv) if self.comments_csv is not None else (self.data_dir / "comments.csv")

                         
        self.out_dir = Path(self.out_dir) if self.out_dir is not None else (self.root_dir / "results")
        self.fig_dir = Path(self.fig_dir) if self.fig_dir is not None else (self.root_dir / "figures")

    @property
    def out_results(self) -> str:
        return str(self.out_dir)

    @property
    def out_figures(self) -> str:
        return str(self.fig_dir)

                           
    @property
    def results_dir(self) -> str:
        return str(self.out_dir)

    @property
    def figures_dir(self) -> str:
        return str(self.fig_dir)


def ensure_dirs(out_results: str | Path | Paths, out_figures: str | Path | None = None) -> None:
    """
    Backward-compatible:
      - ensure_dirs(paths)
      - ensure_dirs("results", "figures")
    """
    if isinstance(out_results, Paths):
        out_dir = Path(out_results.out_dir)
        fig_dir = Path(out_results.fig_dir)
    else:
        out_dir = Path(out_results)
        fig_dir = Path(out_figures) if out_figures is not None else Path("figures")

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)


                              
                       
                              
def safe_log1p(x: pd.Series | np.ndarray) -> np.ndarray:
    arr = pd.to_numeric(x, errors="coerce").fillna(0).to_numpy()
    arr[arr < 0] = 0
    return np.log1p(arr)


def robust_zscore(x: pd.Series | np.ndarray) -> np.ndarray:
    raw = pd.Series(x)
    s = pd.to_numeric(raw, errors="coerce").dropna()
    if len(s) == 0:
        return np.zeros(len(raw))
    med = float(s.median())
    mad = float(np.median(np.abs(s - med)))
    if mad == 0 or np.isnan(mad):
        return np.zeros(len(raw))
    z = 0.6745 * (pd.to_numeric(raw, errors="coerce").fillna(med) - med) / mad
    return z.to_numpy()


def quantile_bins(
    s: pd.Series,
    labels=("low", "mid", "high"),
    q=(0.2, 0.8),
    default_label="mid",
) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    lo = x.quantile(q[0])
    hi = x.quantile(q[1])

    out = pd.Series([default_label] * len(x), index=x.index, dtype="object")
    out[x <= lo] = labels[0]
    out[(x > lo) & (x <= hi)] = labels[1]
    out[x > hi] = labels[2]
    out[x.isna()] = default_label
    return out


                              
              
                              
def _parse_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


                              
                          
                              
                         
                                                             
                                                         
 
                                                                                      
ACTION_PATTERNS = [
    r"\bdo this\b",
    r"\bfollow (these|this) steps\b",
    r"\bhow to\b",
    r"\bguide\b",
    r"\btutorial\b",
    r"\bwalkthrough\b",
    r"\bstep[- ]by[- ]step\b",
    r"\byou should\b",
    r"\byou must\b",
    r"\btry\b",
    r"\brun\b",
    r"\bexecute\b",
    r"\binstall\b",
    r"\bdownload\b",
    r"\bconfigure\b",
    r"\bset up\b",
    r"\bcopy\b",
    r"\bpaste\b",
    r"\bcommand\b",
    r"\bcurl\b",
    r"\bpowershell\b",
    r"\bbash\b",
    r"\bpython\b",
]

SENSITIVE_PATTERNS = [
    r"\bapi key\b",
    r"\btoken\b",
    r"\bpassword\b",
    r"\bcredential\b",
    r"\bsecret\b",
    r"\bauth\b",
    r"\bexploit\b",
    r"\bhack\b",
    r"\bbypass\b",
    r"\bmalware\b",
    r"\bphishing\b",
    r"\bsteal\b",
    r"\bdump\b",
    r"\bleak\b",
    r"\bpostgres\b",
]


def compute_di(text: str) -> int:
    """Compute Directive Intensity (DI) for a post's text."""
    t = (text or "").lower()
    a = sum(1 for pat in ACTION_PATTERNS if re.search(pat, t))
    s = sum(1 for pat in SENSITIVE_PATTERNS if re.search(pat, t))
    return int(min(a + s, 10))


                                                    
def compute_airs(text: str) -> int:
    return compute_di(text)


def add_post_text(posts: pd.DataFrame) -> pd.DataFrame:
    posts = posts.copy()
    title = posts.get("title", pd.Series([""] * len(posts))).fillna("").astype(str)
    content = posts.get("content", pd.Series([""] * len(posts))).fillna("").astype(str)
    posts["text"] = (title + "\n\n" + content).str.strip()
    return posts


def add_di(posts: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - di: Directive Intensity (0..10)
      - is_directive: indicator di>0

    Also adds backward-compatible aliases:
      - airs == di
      - is_action == is_directive
    """
    posts = posts.copy()
    if "text" not in posts.columns:
        posts = add_post_text(posts)

    posts["di"] = posts["text"].apply(compute_di).astype(int)
    posts["is_directive"] = (posts["di"] > 0).astype(int)

                                 
    posts["airs"] = posts["di"]
    posts["is_action"] = posts["is_directive"]
    return posts


                          
def add_airs(posts: pd.DataFrame) -> pd.DataFrame:
    return add_di(posts)


                              
                                  
                              
                                                
                                                    
 
                                          
                                     
RESPONSE_TYPE_DISPLAY: Dict[str, str] = {
    "affirmation": "Affirmation",
    "corrective": "Corrective Signaling",
    "adversarial": "Adversarial Response",
    "neutral": "Neutral Interaction",
}

RESPONSE_TYPE_LEGACY: Dict[str, str] = {
    "affirmation": "endorse",
    "corrective": "enforce",
    "adversarial": "toxic",
    "neutral": "other",
}

ENDORSE_PAT = r"(?:\bthank(s| you)?\b|\bgood\b|\bgreat\b|\bawesome\b|\bexactly\b|\bagree\b|\byes\b|\bcorrect\b|\blgtm\b|\bupvote\b)"
ENFORCE_PAT = r"(?:\bdon't\b|\bdo not\b|\bshouldn't\b|\bunsafe\b|\billegal\b|\bagainst\b|\brules?\b|\bnot allowed\b|\bpolicy\b|\bavoid\b|\bstop\b|\bno\b|\bforbidden\b|\bban\b)"
TOXIC_PAT = r"(?:\bfuck\b|\bshit\b|\bidiot\b|\bstupid\b|\bretard\b|\bslur\b|\bkill\b|\bdie\b)"


def classify_comment(text: str) -> str:
    """Return canonical response_type label."""
    t = (text or "").lower()
    if re.search(TOXIC_PAT, t):
        return "adversarial"
    if re.search(ENFORCE_PAT, t):
        return "corrective"
    if re.search(ENDORSE_PAT, t):
        return "affirmation"
    return "neutral"


def add_response_types(comments: pd.DataFrame) -> pd.DataFrame:
    comments = comments.copy()
    comments["response_type"] = comments["content"].fillna("").astype(str).apply(classify_comment)

                                                           
    comments["response_type_legacy"] = comments["response_type"].map(RESPONSE_TYPE_LEGACY).fillna("other")
    return comments


                              
         
                              
def load_agents(path: str | Path) -> pd.DataFrame:
    agents = pd.read_csv(path)
    agents["first_seen_at_dt"] = _parse_dt(agents["first_seen_at"]) if "first_seen_at" in agents.columns else pd.NaT
    agents["last_seen_at_dt"] = _parse_dt(agents["last_seen_at"]) if "last_seen_at" in agents.columns else pd.NaT
    agents["created_at_dt"] = _parse_dt(agents["created_at"]) if "created_at" in agents.columns else pd.NaT
    return agents


def load_posts(path: str | Path) -> pd.DataFrame:
    posts = pd.read_csv(path)
    posts["created_at_dt"] = _parse_dt(posts["created_at"]) if "created_at" in posts.columns else pd.NaT
    posts["fetched_at_dt"] = _parse_dt(posts["fetched_at"]) if "fetched_at" in posts.columns else pd.NaT
    posts = add_post_text(posts)
    posts = add_di(posts)
    return posts


def load_comments(path: str | Path) -> pd.DataFrame:
    comments = pd.read_csv(path)
    comments["created_at_dt"] = _parse_dt(comments["created_at"]) if "created_at" in comments.columns else pd.NaT
    comments["fetched_at_dt"] = _parse_dt(comments["fetched_at"]) if "fetched_at" in comments.columns else pd.NaT
    comments = add_response_types(comments)
    return comments


def load_all(paths: Paths) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    agents = load_agents(paths.agents_csv)
    posts = load_posts(paths.posts_csv)
    comments = load_comments(paths.comments_csv)
    return agents, posts, comments


                              
                                      
                              
def di_bucket(di_value, q1=None, q2=None) -> str:
    """3-level bucket for DI (low/med/high)."""
    try:
        a = float(di_value)
    except Exception:
        return "low"

    if q1 is not None and q2 is not None:
        try:
            q1f, q2f = float(q1), float(q2)
            if a <= q1f:
                return "low"
            elif a <= q2f:
                return "med"
            return "high"
        except Exception:
            pass

    if a <= 0:
        return "low"
    elif a <= 1:
        return "med"
    return "high"


                          
def airs_bucket(airs_value, q1=None, q2=None) -> str:
    return di_bucket(airs_value, q1=q1, q2=q2)


if __name__ == "__main__":
    p = Paths()
    a, po, c = load_all(p)
