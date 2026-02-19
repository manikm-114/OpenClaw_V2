from __future__ import annotations
from pathlib import Path
import re
import pandas as pd

import utils_openclaw as U


def compute_di_comment(text: str, cap: int = 10) -> int:
    """DI on comment text using the same lexicon as posts."""
    t = (text or "").lower()
    a = sum(1 for pat in U.ACTION_PATTERNS if re.search(pat, t))
    s = sum(1 for pat in U.SENSITIVE_PATTERNS if re.search(pat, t))
    return int(min(a + s, cap))


def main():
                                                            
    project_root = Path(__file__).resolve().parent.parent
    datasets_dir = project_root / "Datasets"

    in_comments = datasets_dir / "comments.csv"
    if not in_comments.exists():
        raise FileNotFoundError(f"Missing: {in_comments}")

                       
    df = pd.read_csv(in_comments)

                                                   
    required = {"id", "post_id", "agent_id", "parent_id", "content", "created_at"}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"comments.csv missing required columns: {missing}")

                                                   
    df = U.add_response_types(df)

                    
    df["di_comment"] = df["content"].fillna("").astype(str).apply(compute_di_comment).astype(int)

    out_path = datasets_dir / "comments_labeled.csv"
    df.to_csv(out_path, index=False)

    print("\nresponse_type counts:")
    print(df["response_type"].value_counts(dropna=False).to_string())
    print("\ndi_comment summary:")
    print(df["di_comment"].describe().to_string())


if __name__ == "__main__":
    main()
