from __future__ import annotations
from pathlib import Path
import pandas as pd

from utils_openclaw import Paths, ensure_dirs, ACTION_PATTERNS, SENSITIVE_PATTERNS

def main():
    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "results"
    figures_dir = project_root / "figures"

    paths = Paths(out_dir=str(results_dir), fig_dir=str(figures_dir))
    ensure_dirs(paths)

                         
    rows = []
    for i, p in enumerate(ACTION_PATTERNS, 1):
        rows.append({"category": "action_oriented", "idx": i, "pattern": p})
    for i, p in enumerate(SENSITIVE_PATTERNS, 1):
        rows.append({"category": "sensitive_execution_related", "idx": i, "pattern": p})

    df = pd.DataFrame(rows)
    out_csv = Path(paths.out_results) / "di_lexicon_patterns.csv"
    df.to_csv(out_csv, index=False)

                  
    txt = []
    txt.append("Directive Intensity (DI) lexicon summary\n")
    txt.append("=======================================\n\n")
    txt.append("Definition:\n")
    txt.append("  DI is the capped count (cap=10) of matched regex patterns across two categories:\n")
    txt.append("    (1) action-oriented / instructional patterns\n")
    txt.append("    (2) sensitive / execution-related patterns\n\n")
    txt.append(f"Counts:\n")
    txt.append(f"  Action-oriented patterns: {len(ACTION_PATTERNS)}\n")
    txt.append(f"  Sensitive/execution-related patterns: {len(SENSITIVE_PATTERNS)}\n")
    txt.append(f"  Total patterns: {len(ACTION_PATTERNS) + len(SENSITIVE_PATTERNS)}\n\n")
    txt.append("Implementation:\n")
    txt.append("  Source of truth: Codes/utils_openclaw.py\n")
    txt.append("  Matching: case-insensitive search over post text (title + content)\n")
    txt.append("  Scoring: DI = min( matches_action + matches_sensitive, 10 )\n\n")
    txt.append("Pattern list:\n")
    txt.append(f"  CSV exported to: {out_csv}\n")

    out_txt = Path(paths.out_results) / "di_lexicon_summary.txt"
    out_txt.write_text("".join(txt), encoding="utf-8")

    print(f" - {out_txt}")
    print(f" - {out_csv}")

if __name__ == "__main__":
    main()
