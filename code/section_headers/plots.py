import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import config

GROUP_ORDER = ["cross_reference", "field", "background", "summary", "drawings", "detailed", "examples"]
SECTION_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#008300"]
GROUP_COLOR = dict(zip(GROUP_ORDER, SECTION_COLORS))
MUTED = "#c3c2b7"
INK = "#0b0b0b"
SECONDARY = "#52514e"
GRID = "#d6d5d0"

GROUP_NAMES = {
    "cross_reference": "Ссылки", "field": "Область", "background": "Уровень\nтехники",
    "summary": "Сущность", "drawings": "Чертежи", "detailed": "Подробное\nописание", "examples": "Примеры",
    "industrial": "Применимость", "other": "Прочие", "before_first_header": "До первого\nзаголовка",
}


def clean_axes(ax, keep=("bottom",)):
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(side in keep)
    for side in keep:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=SECONDARY, length=0)


def plot_presence(results, figures):
    df = pd.read_csv(results / "sections.csv").set_index("group").reindex(GROUP_ORDER).dropna(subset=["share_patents"])
    df = df.sort_values("share_patents")
    fig, ax = plt.subplots(figsize=(4.5, 2.4))
    y = range(len(df))
    ax.barh(y, df["share_patents"] * 100, height=0.6, color=[GROUP_COLOR[g] for g in df.index])
    for i, v in enumerate(df["share_patents"] * 100):
        ax.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=8, color=INK)
    ax.set_yticks(list(y), [GROUP_NAMES[g].replace("\n", " ") for g in df.index], fontsize=8, color=INK)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Доля патентов, где найден раздел, %", fontsize=9, color=SECONDARY)
    clean_axes(ax)
    fig.tight_layout()
    fig.savefig(figures / "sections_presence.pdf")
    plt.close(fig)


def plot_lengths(results, figures):
    df = pd.read_csv(results / "sections.csv").set_index("group").reindex(GROUP_ORDER).dropna(subset=["sentences_median"])
    df = df.sort_values("sentences_median")
    fig, ax = plt.subplots(figsize=(5, 2.8))
    for i, (g, r) in enumerate(df.iterrows()):
        color = GROUP_COLOR[g]
        ax.plot([r["sentences_q10"], r["sentences_q90"]], [i, i], color=color, linewidth=6, alpha=0.3,
                solid_capstyle="round")
        ax.plot(r["sentences_median"], i, "o", markersize=7, color=color, markeredgecolor="white", markeredgewidth=1.5)
    ax.set_xscale("log")
    ticks = [2, 5, 10, 20, 50, 100, 200, 500]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlim(1.5, 600)
    ax.set_ylim(-0.6, len(df) - 0.4)
    ax.set_yticks(range(len(df)), [GROUP_NAMES[g].replace("\n", " ") for g in df.index])
    clean_axes(ax, keep=())
    ax.tick_params(axis="y", labelsize=8, labelcolor=INK, length=0)
    ax.tick_params(axis="x", labelsize=8, labelcolor=SECONDARY, length=0, pad=4)
    ax.set_xlabel("Длина раздела в предложениях", fontsize=9, color=INK, labelpad=6)
    ax.grid(axis="x", color="#ebeae6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Длина разделов описания патента", fontsize=10, color=INK, pad=8)
    fig.tight_layout()
    fig.savefig(figures / "sections_lengths.pdf")
    plt.close(fig)


def plot_clusters(results, figures):
    d = results / "clusters"
    if not (d / "header_clusters.csv").exists():
        return
    df = pd.read_csv(d / "header_clusters.csv")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sc = ax.scatter(df["x"], df["y"], c=df["cluster"], cmap="tab20",
                    s=10 + 60 * df["n_patents_strict"] / df["n_patents_strict"].max())
    for _, r in df.nlargest(40, "n_patents_strict").iterrows():
        ax.annotate(r["header"].title(), (r["x"], r["y"]), fontsize=6, color=INK)
    ax.legend(*sc.legend_elements(), title="Кластер", fontsize=6, loc="best")
    ax.set_xlabel("Компонента 1", fontsize=9, color=SECONDARY)
    ax.set_ylabel("Компонента 2", fontsize=9, color=SECONDARY)
    clean_axes(ax, keep=("bottom", "left"))
    fig.tight_layout()
    fig.savefig(figures / "header_clusters.pdf")
    plt.close(fig)

    if (d / "k_scan.csv").exists():
        scan = pd.read_csv(d / "k_scan.csv")
        fig, ax = plt.subplots(figsize=(4.5, 3))
        ax.plot(scan["k"], scan["silhouette"], marker="o", markersize=4, color=SECTION_COLORS[0], label="Силуэт")
        ax.plot(scan["k"], scan["ari_vs_rules"], marker="s", markersize=4, color=SECTION_COLORS[1],
                label="ARI с группами по ключевым словам")
        ax.set_xlabel("Число кластеров $k$", fontsize=9, color=SECONDARY)
        ax.legend(fontsize=7, frameon=False)
        clean_axes(ax, keep=("bottom", "left"))
        fig.tight_layout()
        fig.savefig(figures / "k_scan.pdf")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(config.RESULTS))
    parser.add_argument("--figures", default=str(config.FIGURES))
    args = parser.parse_args()
    results, figures = Path(args.results), Path(args.figures)
    figures.mkdir(parents=True, exist_ok=True)

    plot_presence(results, figures)
    plot_lengths(results, figures)
    plot_clusters(results, figures)
    print(f"рисунки: {figures}")


if __name__ == "__main__":
    main()
