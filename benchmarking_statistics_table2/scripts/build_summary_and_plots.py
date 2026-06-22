from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
PAIRWISE_ROOT = REPO_ROOT / "outputs" / "pairwise"
SUMMARY_ROOT = REPO_ROOT / "outputs" / "summary"
FIGURE_ROOT = REPO_ROOT / "outputs" / "figures"

COMPARISON_SPECS = [
    ("A vs E", "table2_groupA_vs_groupE_stats_testset.csv", "E"),
    ("A vs F", "table2_groupA_vs_groupF_stats_testset.csv", "F"),
    ("A vs C", "table2_groupA_vs_groupC_stats_testset.csv", "C"),
    ("A vs D", "table2_groupA_vs_groupD_stats_testset.csv", "D"),
    ("A vs G", "table2_groupA_vs_groupG_stats_testset.csv", "G"),
]

COLOR_MAP = {
    "A vs E": "#9467bd",
    "A vs F": "#1f77b4",
    "A vs C": "#2ca02c",
    "A vs D": "#d62728",
    "A vs G": "#ff7f0e",
}

ENDPOINT_ORDER = ["CL", "VDss"]
COMPARISON_ORDER = ["A vs E", "A vs F", "A vs C", "A vs D", "A vs G"]


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    text = df.to_csv(index=False, float_format="%.6g").strip()
    lines = [line.split(",") for line in text.splitlines()]
    widths = [max(len(row[i]) for row in lines) for i in range(len(lines[0]))]

    def fmt(row: list[str]) -> str:
        return "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |"

    header = fmt(lines[0])
    sep = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [fmt(row) for row in lines[1:]]
    return "\n".join([header, sep, *body])


def load_metric_rows(metric: str) -> tuple[pd.DataFrame, str, str, str]:
    frames = []
    for label, filename, suffix in COMPARISON_SPECS:
        path = PAIRWISE_ROOT / filename
        df = pd.read_csv(path)
        df = df[df["Endpoint"].isin(ENDPOINT_ORDER)].copy()

        if metric == "mae":
            delta_col = f"delta_MAE_A_minus_{suffix}"
            lo_col = "delta_MAE_CI95_lo"
            hi_col = "delta_MAE_CI95_hi"
            xlabel = "Delta MAE (A - comparator)"
            title = "Bootstrap 95% CI for Delta MAE on CL and VDss"
            stem = "delta_mae"
        elif metric == "fe2":
            delta_col = f"delta_FE<2_A_minus_{suffix}"
            lo_col = "delta_FE<2_CI95_lo"
            hi_col = "delta_FE<2_CI95_hi"
            xlabel = "Delta FE<2 (A - comparator)"
            title = "Bootstrap 95% CI for Delta FE<2 on CL and VDss"
            stem = "delta_fe2"
        elif metric == "r2":
            delta_col = f"delta_R2_A_minus_{suffix}"
            lo_col = "delta_R2_CI95_lo"
            hi_col = "delta_R2_CI95_hi"
            xlabel = "Delta R2 (A - comparator)"
            title = "Bootstrap 95% CI for Delta R2 on CL and VDss"
            stem = "delta_r2"
        elif metric == "mse":
            delta_col = f"delta_MSE_A_minus_{suffix}"
            lo_col = "delta_MSE_CI95_lo"
            hi_col = "delta_MSE_CI95_hi"
            xlabel = "Delta MSE (A - comparator)"
            title = "Bootstrap 95% CI for Delta MSE on CL and VDss"
            stem = "delta_mse"
        else:
            raise RuntimeError(metric)

        df["Comparison"] = label
        df["Comparator"] = suffix
        df["delta"] = df[delta_col]
        df["ci_lo"] = df[lo_col]
        df["ci_hi"] = df[hi_col]
        frames.append(df)

    return pd.concat(frames, ignore_index=True), xlabel, title, stem


def plot_metric(metric: str) -> None:
    df, xlabel, title, stem = load_metric_rows(metric)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.0), sharex=True, sharey=False)

    for ax, endpoint in zip(axes, ENDPOINT_ORDER):
        sub = df[df["Endpoint"] == endpoint].copy()
        sub["Comparison"] = pd.Categorical(sub["Comparison"], categories=COMPARISON_ORDER, ordered=True)
        sub = sub.sort_values("Comparison")

        y = list(range(len(sub), 0, -1))
        for yy, (_, row) in zip(y, sub.iterrows()):
            ax.hlines(yy, row["ci_lo"], row["ci_hi"], color=COLOR_MAP[row["Comparison"]], linewidth=2.2)
            ax.plot(row["delta"], yy, "o", color=COLOR_MAP[row["Comparison"]], markersize=6.5)

        ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_title(endpoint, fontsize=12)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["Comparison"], fontsize=10)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.grid(axis="x", linestyle=":", alpha=0.4)

    axes[0].set_ylabel("Comparison", fontsize=10)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    fig.savefig(FIGURE_ROOT / f"figure_bootstrap_ci_cl_vdss_{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_ROOT / f"figure_bootstrap_ci_cl_vdss_{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_combined_metrics() -> None:
    metric_specs = [
        ("mae", "A", "Delta MAE", "Lower error for A"),
        ("mse", "B", "Delta MSE", "Lower error for A"),
        ("r2", "C", "Delta R2", "Higher R2 for A"),
        ("fe2", "D", "Delta FE<2", "Higher FE<2 for A"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2), sharey=True)

    for ax, (metric, panel, panel_title, direction_label) in zip(axes.flat, metric_specs):
        df, _, _, _ = load_metric_rows(metric)
        df = df[df["Endpoint"].isin(ENDPOINT_ORDER)].copy()
        df["Comparison"] = pd.Categorical(df["Comparison"], categories=COMPARISON_ORDER, ordered=True)
        df["Endpoint"] = pd.Categorical(df["Endpoint"], categories=ENDPOINT_ORDER, ordered=True)
        df = df.sort_values(["Comparison", "Endpoint"])

        y_positions = {}
        ytick_positions = []
        ytick_labels = []
        base_positions = list(range(len(COMPARISON_ORDER), 0, -1))
        endpoint_offsets = {"CL": 0.16, "VDss": -0.16}
        endpoint_markers = {"CL": "o", "VDss": "s"}

        for base_y, comparison in zip(base_positions, COMPARISON_ORDER):
            ytick_positions.append(base_y)
            ytick_labels.append(comparison)
            for endpoint in ENDPOINT_ORDER:
                y_positions[(comparison, endpoint)] = base_y + endpoint_offsets[endpoint]

        for _, row in df.iterrows():
            yy = y_positions[(row["Comparison"], row["Endpoint"])]
            ax.hlines(yy, row["ci_lo"], row["ci_hi"], color=COLOR_MAP[row["Comparison"]], linewidth=2.0)
            ax.plot(
                row["delta"],
                yy,
                marker=endpoint_markers[row["Endpoint"]],
                color=COLOR_MAP[row["Comparison"]],
                markersize=6.5,
                markeredgecolor="white",
                markeredgewidth=0.4,
            )

        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_title(f"{panel}. {panel_title}", loc="left", fontsize=12.5, fontweight="bold")
        ax.set_xlabel(f"{panel_title} (A - comparator)", fontsize=10.5)
        ax.set_yticks(ytick_positions)
        ax.set_yticklabels(ytick_labels, fontsize=10)
        ax.tick_params(axis="x", labelsize=9.5)
        ax.grid(axis="x", linestyle=":", alpha=0.45)
        ax.text(0.99, 0.04, direction_label, transform=ax.transAxes, ha="right", va="bottom", fontsize=9)

    axes[0, 0].set_ylabel("Comparison", fontsize=10.5)
    axes[1, 0].set_ylabel("Comparison", fontsize=10.5)

    handles = [
        plt.Line2D([0], [0], marker="o", color="black", linestyle="None", markersize=6.5, label="CL"),
        plt.Line2D([0], [0], marker="s", color="black", linestyle="None", markersize=6.5, label="VDss"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.54, 1.015))
    fig.suptitle("Bootstrap 95% CI for matched Table 2 performance differences", fontsize=14, y=1.045)
    fig.tight_layout(rect=[0, 0, 1, 0.985], h_pad=2.0, w_pad=2.4)

    fig.savefig(FIGURE_ROOT / "figure_bootstrap_ci_cl_vdss_combined_metrics.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURE_ROOT / "figure_bootstrap_ci_cl_vdss_combined_metrics.pdf", bbox_inches="tight")
    plt.close(fig)


def build_combined_tables() -> None:
    copied_rows = []
    tidy_rows = []
    for label, filename, suffix in COMPARISON_SPECS:
        src = PAIRWISE_ROOT / filename
        df = pd.read_csv(src)
        df.insert(0, "Comparison", label)
        df.insert(1, "Comparator", suffix)
        copied_rows.append(df)

        for stem in ["stats", "pairs"]:
            candidate = PAIRWISE_ROOT / filename.replace("_stats_", f"_{stem}_")
            if candidate.exists():
                shutil.copy2(candidate, SUMMARY_ROOT / candidate.name)

        md_src = PAIRWISE_ROOT / filename.replace(".csv", ".md")
        if md_src.exists():
            shutil.copy2(md_src, SUMMARY_ROOT / md_src.name)

        comp_metric_cols = {
            "MAE": f"{suffix}_MAE",
            "RMSE": f"{suffix}_RMSE",
            "GMFE": f"{suffix}_GMFE",
            "FE<2": f"{suffix}_FE<2",
            "R2": f"{suffix}_R2",
            "delta_MAE": f"delta_MAE_A_minus_{suffix}",
            "delta_RMSE": f"delta_RMSE_A_minus_{suffix}",
            "delta_MSE": f"delta_MSE_A_minus_{suffix}",
            "delta_FE<2": f"delta_FE<2_A_minus_{suffix}",
            "delta_R2": f"delta_R2_A_minus_{suffix}",
        }

        for _, row in df.iterrows():
            tidy_rows.append(
                {
                    "Comparison": label,
                    "Comparator": suffix,
                    "Endpoint": row["Endpoint"],
                    "N_paired": row["N_paired"],
                    "A_MAE": row["A_MAE"],
                    "Comparator_MAE": row.get(comp_metric_cols["MAE"]),
                    "delta_MAE": row.get(comp_metric_cols["delta_MAE"]),
                    "delta_MAE_CI95_lo": row["delta_MAE_CI95_lo"],
                    "delta_MAE_CI95_hi": row["delta_MAE_CI95_hi"],
                    "wilcoxon_p_abs_err": row["wilcoxon_p_abs_err"],
                    "A_RMSE": row["A_RMSE"],
                    "Comparator_RMSE": row.get(comp_metric_cols["RMSE"]),
                    "delta_RMSE": row.get(comp_metric_cols["delta_RMSE"]),
                    "delta_MSE": row.get(comp_metric_cols["delta_MSE"]),
                    "delta_MSE_CI95_lo": row["delta_MSE_CI95_lo"],
                    "delta_MSE_CI95_hi": row["delta_MSE_CI95_hi"],
                    "A_GMFE": row["A_GMFE"],
                    "Comparator_GMFE": row.get(comp_metric_cols["GMFE"]),
                    "A_FE<2": row["A_FE<2"],
                    "Comparator_FE<2": row.get(comp_metric_cols["FE<2"]),
                    "delta_FE<2": row.get(comp_metric_cols["delta_FE<2"]),
                    "delta_FE<2_CI95_lo": row["delta_FE<2_CI95_lo"],
                    "delta_FE<2_CI95_hi": row["delta_FE<2_CI95_hi"],
                    "mcnemar_p_FE<2": row["mcnemar_p_FE<2"],
                    "A_R2": row["A_R2"],
                    "Comparator_R2": row.get(comp_metric_cols["R2"]),
                    "delta_R2": row.get(comp_metric_cols["delta_R2"]),
                    "delta_R2_CI95_lo": row["delta_R2_CI95_lo"],
                    "delta_R2_CI95_hi": row["delta_R2_CI95_hi"],
                }
            )

    wide = pd.concat(copied_rows, ignore_index=True)
    wide.to_csv(SUMMARY_ROOT / "table2_groupA_vs_controls_stats_testset_wide.csv", index=False)

    combined = pd.DataFrame(tidy_rows)
    combined.to_csv(SUMMARY_ROOT / "table2_groupA_vs_controls_stats_testset.csv", index=False)
    combined[combined["Endpoint"].isin(ENDPOINT_ORDER)].to_csv(
        SUMMARY_ROOT / "table2_groupA_vs_controls_stats_cl_vdss.csv", index=False
    )

    md_lines = [
        "# Combined Compound-Level Statistics (Test Set)\n",
        "Comparisons included: GroupA vs GroupE/F/C/D/G.\n",
        dataframe_to_markdown(combined),
    ]
    (SUMMARY_ROOT / "table2_groupA_vs_controls_stats_testset.md").write_text("\n".join(md_lines))


def main() -> None:
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    build_combined_tables()
    for metric in ["mae", "fe2", "r2", "mse"]:
        plot_metric(metric)
    plot_combined_metrics()


if __name__ == "__main__":
    main()
