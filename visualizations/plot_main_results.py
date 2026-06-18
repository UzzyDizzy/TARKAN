"""Grouped bar chart of TARKAN joint P/R/F1 per dataset (Table 1)."""
from _viz import TABLES, plt, save


def main():
    import pandas as pd

    csv = TABLES / "main_results.csv"
    if not csv.exists():
        print(f"missing {csv}; run experiments/run_main.py first")
        return
    df = pd.read_csv(csv)
    fig, ax = plt.subplots(figsize=(7, 4))
    metrics = ["joint_P", "joint_R", "joint_F1"]
    x = range(len(df))
    w = 0.25
    for i, m in enumerate(metrics):
        ax.bar([xi + i * w for xi in x], df[m], width=w, label=m.replace("joint_", ""))
    ax.set_xticks([xi + w for xi in x])
    ax.set_xticklabels(df["dataset"])
    ax.set_ylabel("score (%)")
    ax.set_title("TARKAN joint MABSA (Table 1)")
    ax.legend()
    save(fig, "main_results.png")


if __name__ == "__main__":
    main()
