
from pathlib import Path
import numpy as np
import pandas as pd


def bootstrap_ci(x: np.ndarray, stat_fn, n_boot: int = 20000, alpha: float = 0.05, seed: int = 0):
    """
    Basic percentile bootstrap CI for a statistic.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_stats = stat_fn(x[idx])

    lo = float(np.quantile(boot_stats, alpha / 2))
    hi = float(np.quantile(boot_stats, 1 - alpha / 2))
    est = float(stat_fn(x.reshape(1, -1))[0])
    return est, lo, hi


def main():
    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "results"

    in_csv = results_dir / "negative_feedback_event_aligned_threads.csv"
    if not in_csv.exists():
        raise FileNotFoundError(f"Missing: {in_csv}")

    df = pd.read_csv(in_csv)

                    
    required = {"n_before", "n_after", "max_di_before", "delta_mean_di"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

                                                            
    usable = df[(df["n_before"] > 0) & (df["n_after"] > 0)].copy()

                                                               
    reg = usable[usable["max_di_before"] > 0].copy()

                       
    x_time = reg["delta_mean_di"].to_numpy(dtype=float)
    x_time = x_time[np.isfinite(x_time)]

                               
    has_nwin = "nwin_delta_mean_di" in reg.columns
    x_nwin = None
    if has_nwin:
        x_nwin = reg["nwin_delta_mean_di"].to_numpy(dtype=float)
        x_nwin = x_nwin[np.isfinite(x_nwin)]

                        
    N_BOOT = 20000
    ALPHA = 0.05

                                
    stat_mean = lambda a: np.mean(a, axis=1)
    stat_median = lambda a: np.median(a, axis=1)

    rows = []

    def add_block(name: str, x: np.ndarray):
        if x is None or len(x) == 0:
            rows.append({
                "window": name,
                "n": 0,
                "mean": np.nan, "mean_ci_lo": np.nan, "mean_ci_hi": np.nan,
                "median": np.nan, "median_ci_lo": np.nan, "median_ci_hi": np.nan,
            })
            return

        mean_est, mean_lo, mean_hi = bootstrap_ci(x, stat_mean, n_boot=N_BOOT, alpha=ALPHA, seed=0)
        med_est, med_lo, med_hi = bootstrap_ci(x, stat_median, n_boot=N_BOOT, alpha=ALPHA, seed=1)

        rows.append({
            "window": name,
            "n": int(len(x)),
            "mean": float(np.mean(x)),
            "mean_ci_lo": mean_lo,
            "mean_ci_hi": mean_hi,
            "median": float(np.median(x)),
            "median_ci_lo": med_lo,
            "median_ci_hi": med_hi,
        })

    add_block("time_window", x_time)
    if has_nwin:
        add_block("n_window", x_nwin)

    out_df = pd.DataFrame(rows)

    out_csv = results_dir / "event_aligned_bootstrap_summary.csv"
    out_txt = results_dir / "event_aligned_bootstrap_summary.txt"

    out_df.to_csv(out_csv, index=False)

                         
    lines = []
    lines.append("Event-aligned bootstrap summary (regulatable threads)\n")
    lines.append("====================================================\n\n")
    lines.append(f"Input: {in_csv}\n")
    lines.append("Filter: usable (n_before>0 & n_after>0), regulatable (max_di_before>0)\n")
    lines.append(f"Bootstrap: percentile CI, n_boot={N_BOOT}, alpha={ALPHA}\n\n")

    for _, r in out_df.iterrows():
        lines.append(f"[{r['window']}]\n")
        lines.append(f"  n = {int(r['n'])}\n")
        lines.append(f"  mean   = {r['mean']:.6g}   (95% CI {r['mean_ci_lo']:.6g}, {r['mean_ci_hi']:.6g})\n")
        lines.append(f"  median = {r['median']:.6g} (95% CI {r['median_ci_lo']:.6g}, {r['median_ci_hi']:.6g})\n\n")

    out_txt.write_text("".join(lines), encoding="utf-8")

    print(f" - {out_csv}")
    print(f" - {out_txt}")
    print("\n" + "".join(lines))


if __name__ == "__main__":
    main()
