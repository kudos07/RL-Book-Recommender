# plot_eval.py
import os
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.dirname(__file__)
CSV_PATH = os.path.join(ROOT, "eval_results.csv")
OUT_DIR = os.path.join(ROOT, "plots")
os.makedirs(OUT_DIR, exist_ok=True)

# Load
df = pd.read_csv(CSV_PATH)

# Filter only CTR@k_episode and NDCG@k
metrics_to_plot = ["CTR@k_episode", "NDCG@k"]

for metric in metrics_to_plot:
    sub = df[df["metric"] == metric]

    # Pivot for plotting
    pivot_df = sub.pivot(index="k", columns="model", values="value").sort_index()

    ax = pivot_df.plot(
        kind="bar",
        figsize=(6, 4),
        width=0.8,
        edgecolor="black",
    )

    ax.set_title(f"{metric} Comparison", fontsize=14, pad=12)
    ax.set_xlabel("k", fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_xticklabels(pivot_df.index, rotation=0)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    plt.legend(title="Model")
    plt.tight_layout()

    out_file = os.path.join(OUT_DIR, f"{metric.replace('@','_at_')}.png")
    plt.savefig(out_file, dpi=300)
    print(f"Saved plot: {out_file}")

plt.show()
