# figure0_conceptual_overview.py
# ------------------------------------------------------------
# Step 5: Conceptual overview schematic for the paper
# Writes:
#   figures/Figure_0_Conceptual_Overview.png
#   figures/Figure_0_Conceptual_Overview.pdf
# ------------------------------------------------------------

from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def add_box(ax, xy, w, h, title, body, fontsize=10):
    x, y = xy
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2,
        facecolor="white",
        edgecolor="black"
    )
    ax.add_patch(box)
    ax.text(x + 0.02*w, y + h - 0.28*h, title, fontsize=fontsize+1, weight="bold", va="top")
    ax.text(x + 0.02*w, y + h - 0.40*h, body, fontsize=fontsize, va="top")
    return box


def add_arrow(ax, start, end):
    arr = FancyArrowPatch(
        start, end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.2,
        color="black"
    )
    ax.add_patch(arr)


def main():
    project_root = Path(__file__).resolve().parent.parent
    fig_dir = project_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6.8))
    ax = plt.gca()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Layout
    b1 = add_box(
        ax, (0.05, 0.62), 0.28, 0.28,
        "Agent-only society (Moltbook/OpenClaw)",
        "Posts + comment threads\nNo human moderation\nNo centralized controller"
    )

    b2 = add_box(
        ax, (0.38, 0.70), 0.26, 0.20,
        "Measure: Directive Intensity (DI)",
        "Transparent regex lexicon\n23 action + 15 sensitive patterns\nDI = capped match count (0–10)"
    )

    b3 = add_box(
        ax, (0.38, 0.45), 0.26, 0.20,
        "Reply typing (rule-based)",
        "Affirmation\nCorrective signaling\nAdversarial response\nNeutral interaction"
    )

    b4 = add_box(
        ax, (0.70, 0.62), 0.25, 0.28,
        "Core finding",
        "Corrective signaling probability\nincreases with DI\n(monotonic coupling)"
    )

    b5 = add_box(
        ax, (0.70, 0.25), 0.25, 0.28,
        "Robustness / inference",
        "Accounts for clustering:\ncomments nested in posts\n(post random intercept / clustered SE)\nResult direction persists"
    )

    # Arrows
    add_arrow(ax, (0.33, 0.76), (0.38, 0.80))  # society -> DI
    add_arrow(ax, (0.33, 0.70), (0.38, 0.55))  # society -> typing
    add_arrow(ax, (0.64, 0.80), (0.70, 0.76))  # DI -> finding
    add_arrow(ax, (0.64, 0.55), (0.70, 0.70))  # typing -> finding
    add_arrow(ax, (0.82, 0.62), (0.82, 0.53))  # finding -> robustness

    # Central takeaway banner
    banner = FancyBboxPatch(
        (0.05, 0.08), 0.90, 0.12,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2,
        facecolor="white",
        edgecolor="black"
    )
    ax.add_patch(banner)
    ax.text(
        0.50, 0.14,
        "Takeaway: In a purely synthetic, agent-only society, corrective signaling can emerge endogenously\n"
        "and scales with directive intensity even without centralized moderation.",
        ha="center", va="center", fontsize=12, weight="bold"
    )

    out_png = fig_dir / "Figure_0_Conceptual_Overview.png"
    out_pdf = fig_dir / "Figure_0_Conceptual_Overview.pdf"
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()

    print("[OK] Wrote:")
    print(f" - {out_png}")
    print(f" - {out_pdf}")


if __name__ == "__main__":
    main()
