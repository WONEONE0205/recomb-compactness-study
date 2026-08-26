from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 修改为截图中12个结果文件夹所在的总文件夹
RESULTS_ROOT = Path(
    r"D:\python project\Gerrymandering\results\all_results"
)

OUTPUT_FOLDER = RESULTS_ROOT / "summary_figures"
OUTPUT_FOLDER.mkdir(exist_ok=True)

BETA_ORDER = [0.0, 0.05, 0.1, 0.5]
SEED_ORDER = [42, 123, 1234]


def load_all_chains(results_root):
    """Read the twelve chains from extracted result folders."""

    chains = {}

    summary_files = list(
        results_root.rglob("summary_*.json")
    )

    print(f"Results folder exists: {results_root.exists()}")
    print(f"Number of summary files found: {len(summary_files)}")

    for summary_path in summary_files:
        run_folder = summary_path.parent

        with open(
            summary_path,
            "r",
            encoding="utf-8"
        ) as file:
            summary = json.load(file)

        beta = float(summary["beta"])
        seed = int(summary["random_seed"])
        initial_cut_edges = int(
            summary["initial_cut_edges"]
        )

        transition_files = sorted(
            run_folder.glob("transitions_*.csv.gz")
        )

        print(
            f"Reading beta={beta}, seed={seed}: "
            f"{len(transition_files)} transition files"
        )

        if len(transition_files) != 10:
            raise ValueError(
                f"beta={beta}, seed={seed}: expected 10 "
                f"transition files, but found "
                f"{len(transition_files)}."
            )

        frames = [
            pd.read_csv(path)
            for path in transition_files
        ]

        transitions = pd.concat(
            frames,
            ignore_index=True
        )

        if len(transitions) != 10000:
            raise ValueError(
                f"beta={beta}, seed={seed}: expected "
                f"10,000 transitions, but found "
                f"{len(transitions)}."
            )

        key = (beta, seed)

        if key in chains:
            raise ValueError(
                f"Duplicate result found for "
                f"beta={beta}, seed={seed}."
            )

        chains[key] = {
            "data": transitions,
            "initial_cut_edges": initial_cut_edges
        }

        print(
            f"Loaded beta={beta}, seed={seed}"
        )

    return chains


chains = load_all_chains(RESULTS_ROOT)


# 检查12条链是否全部读取成功
expected = {
    (beta, seed)
    for beta in BETA_ORDER
    for seed in SEED_ORDER
}

missing = expected - set(chains)

if missing:
    raise ValueError(
        f"The following chains are missing: "
        f"{sorted(missing)}"
    )

print("All 12 chains loaded successfully.")


# 三个随机种子的颜色
seed_colours = {
    42: "#3366A0",
    123: "#D55E00",
    1234: "#009E73"
}


# 创建四面板trajectory汇总图
fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 8),
    sharex=True,
    sharey=True
)

for axis, beta in zip(axes.flat, BETA_ORDER):

    for seed in SEED_ORDER:
        chain = chains[(beta, seed)]
        data = chain["data"]

        # 将初始状态加入transition 0
        x = np.concatenate(
            (
                [0],
                data["transition"].to_numpy()
            )
        )

        y = np.concatenate(
            (
                [chain["initial_cut_edges"]],
                data["cut_edges"].to_numpy()
            )
        )

        axis.plot(
            x,
            y,
            color=seed_colours[seed],
            linewidth=0.55,
            alpha=0.85,
            label=f"Seed {seed}"
        )

    axis.set_title(
        rf"$\beta={beta:g}$",
        fontsize=13
    )

    axis.grid(alpha=0.2)


# 坐标轴标题
axes[0, 0].set_ylabel("Cut-edge count")
axes[1, 0].set_ylabel("Cut-edge count")

axes[1, 0].set_xlabel("Transition")
axes[1, 1].set_xlabel("Transition")


# 共用图例
handles, labels = (
    axes[0, 0].get_legend_handles_labels()
)

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.98),
    ncol=3,
    frameon=False
)

fig.tight_layout(
    rect=[0, 0, 1, 0.91]
)


# 保存图片
png_path = (
    OUTPUT_FOLDER /
    "cut_edge_trajectories.png"
)

pdf_path = (
    OUTPUT_FOLDER /
    "cut_edge_trajectories.pdf"
)

fig.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    pdf_path,
    bbox_inches="tight"
)

plt.show()

print(f"PNG saved to: {png_path}")
print(f"PDF saved to: {pdf_path}")
