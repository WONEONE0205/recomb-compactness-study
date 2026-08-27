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

summary_rows = []

for beta in BETA_ORDER:
    # Combine the 10,000 states from each of the three seeds
    values = pd.concat(
        [
            chains[(beta, seed)]["data"]["cut_edges"]
            for seed in SEED_ORDER
        ],
        ignore_index=True,
    )

    # Check that exactly 30,000 states are included
    if len(values) != 30000:
        raise ValueError(
            f"beta={beta}: expected 30,000 cut-edge counts, "
            f"but found {len(values)}."
        )

    summary_rows.append(
        {
            "Beta": beta,
            "Mean": values.mean(),
            "Median": values.median(),
            "SD": values.std(),
            "5th percentile": values.quantile(0.05),
            "95th percentile": values.quantile(0.95),
        }
    )

summary_table = pd.DataFrame(summary_rows)

# Format the results in the same way as Table 4.1
summary_table["Mean"] = summary_table["Mean"].round(2)
summary_table["Median"] = summary_table["Median"].round(0).astype(int)
summary_table["SD"] = summary_table["SD"].round(2)
summary_table["5th percentile"] = (
    summary_table["5th percentile"].round(0).astype(int)
)
summary_table["95th percentile"] = (
    summary_table["95th percentile"].round(0).astype(int)
)

print("\n===== Table 4.1: Cut-edge Summary Statistics =====")
print(summary_table.to_string(index=False))

# Save the table as a CSV file
summary_path = OUTPUT_FOLDER / "table_4_1_cut_edge_summary.csv"
summary_table.to_csv(summary_path, index=False)

print(f"\nSummary table saved to: {summary_path}")


beta = 0.5
interval_rows = []

for interval_number in range(10):
    start = interval_number * 1000 + 1
    end = (interval_number + 1) * 1000

    # Combine the same interval from all three seeds
    interval_values = pd.concat(
        [
            chains[(beta, seed)]["data"].loc[
                chains[(beta, seed)]["data"]["transition"].between(
                    start,
                    end,
                ),
                "cut_edges",
            ]
            for seed in SEED_ORDER
        ],
        ignore_index=True,
    )

    # Each interval should contain 1,000 × 3 = 3,000 values
    if len(interval_values) != 3000:
        raise ValueError(
            f"Transitions {start}-{end}: expected 3,000 values, "
            f"but found {len(interval_values)}."
        )

    interval_rows.append(
        {
            "Interval": interval_number + 1,
            "Transitions": f"{start}-{end}",
            "Number of values": len(interval_values),
            "Mean cut edges": round(interval_values.mean(), 1),
        }
    )

interval_table = pd.DataFrame(interval_rows)

print("\n===== Beta 0.5: Interval Means =====")
print(interval_table.to_string(index=False))

interval_path = OUTPUT_FOLDER / "beta_0p5_interval_means.csv"
interval_table.to_csv(interval_path, index=False)

print(f"\nInterval means saved to: {interval_path}")

rate_rows = []

for beta in BETA_ORDER:
    combined_data = pd.concat(
        [
            chains[(beta, seed)]["data"]
            for seed in SEED_ORDER
        ],
        ignore_index=True,
    )

    if len(combined_data) != 30000:
        raise ValueError(
            f"beta={beta}: expected 30,000 transitions, "
            f"but found {len(combined_data)}."
        )

    # Convert True/False columns safely
    accepted = (
        combined_data["accepted"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("true")
    )

    assignment_repeated = (
        combined_data["assignment_repeated"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("true")
    )

    boundary_repeated = (
        combined_data["boundary_repeated"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("true")
    )

    rate_rows.append(
        {
            "Beta": beta,
            "Accepted count": accepted.sum(),
            "Acceptance rate": round(accepted.mean() * 100, 2),
            "Assignment repeated count": assignment_repeated.sum(),
            "Assignment repetition": round(
                assignment_repeated.mean() * 100,
                2,
            ),
            "Boundary repeated count": boundary_repeated.sum(),
            "Boundary repetition": round(
                boundary_repeated.mean() * 100,
                2,
            ),
        }
    )

rate_table = pd.DataFrame(rate_rows)

print("\n===== Table 4.2: Acceptance and Repetition Rates =====")
print(rate_table.to_string(index=False))

rate_path = OUTPUT_FOLDER / "table_4_2_acceptance_repetition.csv"
rate_table.to_csv(rate_path, index=False)

difference_rows = []

for beta in BETA_ORDER:
    combined_data = pd.concat(
        [
            chains[(beta, seed)]["data"]
            for seed in SEED_ORDER
        ],
        ignore_index=True,
    )

    cut_difference = pd.to_numeric(
        combined_data["cut_difference"],
        errors="raise",
    )

    accepted = (
        combined_data["accepted"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("true")
    )

    positive_mask = cut_difference > 0
    nonpositive_mask = cut_difference <= 0

    positive_total = positive_mask.sum()
    positive_accepted = accepted[positive_mask].sum()

    nonpositive_total = nonpositive_mask.sum()
    nonpositive_accepted = accepted[nonpositive_mask].sum()

    difference_rows.append(
        {
            "Beta": beta,
            "Delta C > 0 proposals": positive_total,
            "Delta C > 0 accepted": positive_accepted,
            "Delta C > 0 acceptance rate": round(
                positive_accepted / positive_total * 100,
                2,
            ),
            "Delta C <= 0 proposals": nonpositive_total,
            "Delta C <= 0 accepted": nonpositive_accepted,
            "Delta C <= 0 acceptance rate": round(
                nonpositive_accepted / nonpositive_total * 100,
                2,
            ),
        }
    )

difference_table = pd.DataFrame(difference_rows)

print("\n===== Acceptance Rates by Cut-Edge Difference =====")
print(difference_table.to_string(index=False))

difference_path = (
    OUTPUT_FOLDER / "acceptance_by_cut_edge_difference.csv"
)
difference_table.to_csv(difference_path, index=False)

print(f"\nDifference table saved to: {difference_path}")

print(f"\nTable 4.2 saved to: {rate_path}")

individual_rows = []

for beta in BETA_ORDER:
    for seed in SEED_ORDER:
        data = chains[(beta, seed)]["data"]

        if len(data) != 10000:
            raise ValueError(
                f"beta={beta}, seed={seed}: expected 10,000 "
                f"transitions, but found {len(data)}."
            )

        accepted = (
            data["accepted"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("true")
        )

        assignment_repeated = (
            data["assignment_repeated"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("true")
        )

        individual_rows.append(
            {
                "Beta": beta,
                "Seed": seed,
                "Mean cut edges": round(
                    data["cut_edges"].mean(),
                    2,
                ),
                "Accepted count": accepted.sum(),
                "Acceptance rate": round(
                    accepted.mean() * 100,
                    2,
                ),
                "Assignment repeated count": (
                    assignment_repeated.sum()
                ),
                "Assignment repetition": round(
                    assignment_repeated.mean() * 100,
                    2,
                ),
            }
        )

individual_table = pd.DataFrame(individual_rows)

print("\n===== Table 4.3: Individual Chain Results =====")
print(individual_table.to_string(index=False))

individual_path = OUTPUT_FOLDER / "table_4_3_individual_chains.csv"
individual_table.to_csv(individual_path, index=False)

print(f"\nTable 4.3 saved to: {individual_path}")

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