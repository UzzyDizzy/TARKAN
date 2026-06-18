"""Bar charts for KG retrieval/filtering stats (Table 8) and visual-relevance F1 (Table 9)."""
from _viz import TABLES, plt, save


def main():
    import pandas as pd

    kg = TABLES / "kg_diagnostics.csv"
    if kg.exists():
        df = pd.read_csv(kg)
        cols = ["avg_retrieved_per_aspect", "avg_retained_per_aspect", "SenticNet_contribution", "ConceptNet_contribution"]
        fig, ax = plt.subplots(figsize=(9, 4))
        x = range(len(cols))
        for j, (_, row) in enumerate(df.iterrows()):
            ax.bar([xi + j * 0.4 for xi in x], [row[c] for c in cols], width=0.4, label=row["dataset"])
        ax.set_xticks([xi + 0.2 for xi in x])
        ax.set_xticklabels(cols, rotation=20, ha="right")
        ax.set_title("KG retrieval & filtering (Table 8)")
        ax.legend()
        save(fig, "kg_stats.png")
    else:
        print(f"missing {kg}; run analysis/kg_diagnostics.py first")

    rel = TABLES / "visual_relevance_conditions.csv"
    if rel.exists():
        df = pd.read_csv(rel)
        fig, ax = plt.subplots(figsize=(9, 4))
        for ds, sub in df.groupby("dataset"):
            ax.bar(sub["condition"], sub["TARKAN_F1"], label=ds, alpha=0.7)
        ax.set_ylabel("F1 (%)")
        ax.set_title("Performance by visual-relevance condition (Table 9)")
        ax.set_xticklabels(df["condition"].unique(), rotation=20, ha="right")
        ax.legend()
        save(fig, "visual_relevance.png")


if __name__ == "__main__":
    main()
