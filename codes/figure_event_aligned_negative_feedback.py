
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

                                                    
                            
                                                    
mpl.rcParams.update({
    "font.size": 5,
    "axes.titlesize": 5,
    "axes.labelsize": 5,
    "xtick.labelsize": 5,
    "ytick.labelsize": 5,
    "legend.fontsize": 5,
    "mathtext.fontset": "cm",
})

def main():
    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "results"
    figures_dir = project_root / "Figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    in_csv = results_dir / "negative_feedback_event_aligned_threads.csv"
    if not in_csv.exists():
        raise FileNotFoundError(f"Missing: {in_csv}")

    df = pd.read_csv(in_csv)

    required = {"n_before", "n_after", "max_di_before", "delta_mean_di"}
    if not required.issubset(df.columns):
        raise ValueError("CSV missing required columns")

                    
    usable = df[(df["n_before"] > 0) & (df["n_after"] > 0)]
    reg = usable[usable["max_di_before"] > 0].dropna(subset=["delta_mean_di"])

    deltas = reg["delta_mean_di"].to_numpy(float)

                                            
    rng = np.random.default_rng(0)
    y = rng.normal(0.0, 0.03, size=len(deltas))

    fig = plt.figure(figsize=(3.2, 2.2))
    ax = plt.gca()

    ax.scatter(deltas, y, s=14, alpha=0.8, linewidths=0)

                                            
    ax.axvline(0.0, linewidth=1, color="0.5")

                                       
    median_delta = float(np.median(deltas))
    ax.axvline(median_delta, linewidth=2, color="red")

    ax.set_xlabel(r"$\Delta \overline{DI}_{comment}$ (after $t_0$ minus before $t_0$)")
    ax.set_ylabel("Regulatable threads (vertical jitter only)")
    ax.set_yticks([])
    ax.set_ylim(-0.12, 0.12)

    fig.tight_layout()
    fig.savefig(figures_dir / "Figure_2_EventAligned_DeltaDI.png", dpi=300)
    fig.savefig(figures_dir / "Figure_2_EventAligned_DeltaDI.pdf")
    plt.close(fig)

if __name__ == "__main__":
    main()
