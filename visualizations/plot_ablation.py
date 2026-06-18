"""Bar charts for the component ablation (Table 6) and fusion ablation (Table 10)."""
from _viz import TABLES, plt, save


def _plot(csv_name, label_col, value_cols, title, out):
    import pandas as pd

    csv = TABLES / csv_name
    if not csv.exists():
        print(f"missing {csv}; run the corresponding ablation runner first")
        return
    df = pd.read_csv(csv)
    cols = [c for c in value_cols if c in df.columns]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = range(len(df))
    w = 0.8 / max(1, len(cols))
    for i, c in enumerate(cols):
        ax.bar([xi + i * w for xi in x], df[c], width=w, label=c)
    ax.set_xticks([xi + 0.4 for xi in x])
    ax.set_xticklabels(df[label_col], rotation=30, ha="right")
    ax.set_ylabel("F1 (%)")
    ax.set_title(title)
    ax.legend()
    save(fig, out)


def main():
    _plot("ablation_components.csv", "variant", ["twitter2015_F1", "twitter2017_F1"],
          "Component ablation (Table 6)", "ablation_components.png")
    _plot("ablation_fusion.csv", "fusion", ["twitter2015_F1", "twitter2017_F1"],
          "Fusion-strategy ablation (Table 10)", "ablation_fusion.png")


if __name__ == "__main__":
    main()
