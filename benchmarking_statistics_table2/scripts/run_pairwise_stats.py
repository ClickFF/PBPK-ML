import math
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError:  # pragma: no cover - used in lightweight repository environments.
    stats = None


REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = REPO_ROOT / "inputs"
PAIRWISE_OUT = REPO_ROOT / "outputs" / "pairwise"

COMPARISONS = [
    ("E", "DL-only control", ["Fu", "CL", "VDss"]),
    ("F", "RDKit-only control", ["Fu", "CL", "VDss"]),
    ("C", "descriptor/control comparison C", ["CL", "VDss"]),
    ("D", "descriptor/control comparison D", ["CL", "VDss"]),
    ("G", "embeddings-only control", ["Fu", "CL", "VDss"]),
]

ENDPOINT_TO_FILE = {
    "Fu": "res_Fu.csv",
    "CL": "res_CL.csv",
    "VDss": "res_VDss.csv",
}

ENDPOINT_TO_DATASET_TOKEN = {
    "Fu": "lgFu_test",
    "CL": "lgCL_test",
    "VDss": "lgVD_test",
}


def fold_error_log10(y_true_log10: np.ndarray, y_pred_log10: np.ndarray) -> np.ndarray:
    delta = y_pred_log10 - y_true_log10
    return np.maximum(10.0**delta, 10.0 ** (-delta))


def gmfe_from_fe(fe: np.ndarray) -> float:
    return float(np.exp(np.mean(np.log(fe))))


def metric_block(y_true_log10: np.ndarray, y_pred_log10: np.ndarray) -> Dict[str, float]:
    err = y_pred_log10 - y_true_log10
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    fe = fold_error_log10(y_true_log10, y_pred_log10)
    sse = float(np.sum((y_true_log10 - y_pred_log10) ** 2))
    sst = float(np.sum((y_true_log10 - float(np.mean(y_true_log10))) ** 2))
    r2 = float(1.0 - (sse / sst)) if sst != 0.0 else float("nan")
    return {
        "MAE": mae,
        "RMSE": rmse,
        "GMFE": gmfe_from_fe(fe),
        "FE<2": float(np.mean(fe <= 2.0)),
        "R2": r2,
    }


def paired_bootstrap_delta(
    a_vals: np.ndarray, b_vals: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(a_vals)
    idx = rng.integers(0, n, size=(n_boot, n))
    deltas = np.mean(a_vals[idx], axis=1) - np.mean(b_vals[idx], axis=1)
    delta = float(np.mean(a_vals) - np.mean(b_vals))
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return delta, float(lo), float(hi)


def bootstrap_delta_r2(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    idx = rng.integers(0, n, size=(n_boot, n))

    def r2(yt: np.ndarray, yp: np.ndarray) -> float:
        sse = np.sum((yt - yp) ** 2)
        sst = np.sum((yt - np.mean(yt)) ** 2)
        return float(1.0 - (sse / sst)) if sst != 0.0 else float("nan")

    base_delta = r2(y_true, y_pred_a) - r2(y_true, y_pred_b)
    deltas = []
    for j in range(n_boot):
        ii = idx[j]
        deltas.append(r2(y_true[ii], y_pred_a[ii]) - r2(y_true[ii], y_pred_b[ii]))
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return float(base_delta), float(lo), float(hi)


def wilcoxon_p(a_vals: np.ndarray, b_vals: np.ndarray) -> float:
    diff = a_vals - b_vals
    if np.allclose(diff, 0):
        return 1.0
    if stats is not None:
        return float(stats.wilcoxon(diff, zero_method="pratt", alternative="two-sided", mode="auto").pvalue)

    # Fallback: two-sided normal approximation to the signed-rank statistic.
    # Zero differences are excluded in this lightweight implementation.
    nonzero = diff[~np.isclose(diff, 0)]
    n = len(nonzero)
    if n == 0:
        return 1.0
    abs_diff = np.abs(nonzero)
    order = np.argsort(abs_diff)
    ranks = np.empty(n, dtype=float)
    sorted_abs = abs_diff[order]
    start = 0
    while start < n:
        end = start + 1
        while end < n and np.isclose(sorted_abs[end], sorted_abs[start]):
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    w_plus = float(np.sum(ranks[nonzero > 0]))
    mean_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0
    z = (abs(w_plus - mean_w) - 0.5) / math.sqrt(var_w)
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def mcnemar_p(a_success: np.ndarray, b_success: np.ndarray) -> float:
    a = a_success.astype(bool)
    b = b_success.astype(bool)
    b01 = int(np.sum((~a) & b))
    b10 = int(np.sum(a & (~b)))
    n = b01 + b10
    if n == 0:
        return 1.0
    if stats is not None:
        return float(stats.binomtest(min(b01, b10), n, 0.5, alternative="two-sided").pvalue)

    # Fallback: exact two-sided binomial test for p=0.5.
    k = min(b01, b10)
    p = 2.0 * sum(math.comb(n, i) * (0.5**n) for i in range(k + 1))
    return float(min(1.0, p))


def load_endpoint(group: str, endpoint: str) -> pd.DataFrame:
    path = INPUT_ROOT / f"Group{group}" / ENDPOINT_TO_FILE[endpoint]
    df = pd.read_csv(path)
    token = ENDPOINT_TO_DATASET_TOKEN[endpoint]
    if "Dataset" in df.columns:
        df = df[df["Dataset"].astype(str).str.contains(token, case=False, na=False)].copy()
    return df[["PUBCHEM_CID", "Actual", "Predicted", "Dataset"]].copy()


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


def build_stats_for_comparison(comparator: str, endpoints: Iterable[str]) -> None:
    rows = []
    pair_frames = []

    for endpoint in endpoints:
        df_a = load_endpoint("A", endpoint)
        df_b = load_endpoint(comparator, endpoint)
        merged = df_a.merge(df_b, on="PUBCHEM_CID", suffixes=("_A", f"_{comparator}"))

        y_true_a = merged["Actual_A"].to_numpy(dtype=float)
        y_pred_a = merged["Predicted_A"].to_numpy(dtype=float)
        y_true_b = merged[f"Actual_{comparator}"].to_numpy(dtype=float)
        y_pred_b = merged[f"Predicted_{comparator}"].to_numpy(dtype=float)

        true_diff = float(np.mean(np.abs(y_true_a - y_true_b)))
        m_a = metric_block(y_true_a, y_pred_a)
        m_b = metric_block(y_true_b, y_pred_b)

        abs_err_a = np.abs(y_pred_a - y_true_a)
        abs_err_b = np.abs(y_pred_b - y_true_b)
        delta_mae, lo_mae, hi_mae = paired_bootstrap_delta(abs_err_a, abs_err_b)

        sq_err_a = (y_pred_a - y_true_a) ** 2
        sq_err_b = (y_pred_b - y_true_b) ** 2
        delta_mse, lo_mse, hi_mse = paired_bootstrap_delta(sq_err_a, sq_err_b)

        fe_a = fold_error_log10(y_true_a, y_pred_a)
        fe_b = fold_error_log10(y_true_b, y_pred_b)
        delta_fe2, lo_fe2, hi_fe2 = paired_bootstrap_delta(
            (fe_a <= 2.0).astype(float), (fe_b <= 2.0).astype(float)
        )
        delta_r2, lo_r2, hi_r2 = bootstrap_delta_r2(y_true_a, y_pred_a, y_pred_b)

        rows.append(
            {
                "Endpoint": endpoint,
                "N_paired": int(len(merged)),
                "mean_abs_true_diff_log10": true_diff,
                "A_MAE": m_a["MAE"],
                f"{comparator}_MAE": m_b["MAE"],
                f"delta_MAE_A_minus_{comparator}": delta_mae,
                "delta_MAE_CI95_lo": lo_mae,
                "delta_MAE_CI95_hi": hi_mae,
                "wilcoxon_p_abs_err": wilcoxon_p(abs_err_a, abs_err_b),
                "A_RMSE": m_a["RMSE"],
                f"{comparator}_RMSE": m_b["RMSE"],
                f"delta_RMSE_A_minus_{comparator}": float(m_a["RMSE"] - m_b["RMSE"]),
                f"delta_MSE_A_minus_{comparator}": float(delta_mse),
                "delta_MSE_CI95_lo": float(lo_mse),
                "delta_MSE_CI95_hi": float(hi_mse),
                "A_GMFE": m_a["GMFE"],
                f"{comparator}_GMFE": m_b["GMFE"],
                "A_FE<2": m_a["FE<2"],
                f"{comparator}_FE<2": m_b["FE<2"],
                f"delta_FE<2_A_minus_{comparator}": delta_fe2,
                "delta_FE<2_CI95_lo": lo_fe2,
                "delta_FE<2_CI95_hi": hi_fe2,
                "mcnemar_p_FE<2": mcnemar_p(fe_a <= 2.0, fe_b <= 2.0),
                "A_R2": m_a["R2"],
                f"{comparator}_R2": m_b["R2"],
                f"delta_R2_A_minus_{comparator}": delta_r2,
                "delta_R2_CI95_lo": lo_r2,
                "delta_R2_CI95_hi": hi_r2,
            }
        )

        pair_frames.append(
            merged[
                [
                    "PUBCHEM_CID",
                    "Actual_A",
                    "Predicted_A",
                    f"Actual_{comparator}",
                    f"Predicted_{comparator}",
                ]
            ].assign(
                Endpoint=endpoint,
                abs_err_A=abs_err_a,
                **{f"abs_err_{comparator}": abs_err_b, "fe_A": fe_a, f"fe_{comparator}": fe_b},
            )
        )

    stats_df = pd.DataFrame(rows)
    stats_path = PAIRWISE_OUT / f"table2_groupA_vs_group{comparator}_stats_testset.csv"
    pairs_path = PAIRWISE_OUT / f"table2_groupA_vs_group{comparator}_pairs_testset.csv"
    md_path = PAIRWISE_OUT / f"table2_groupA_vs_group{comparator}_stats_testset.md"

    stats_df.to_csv(stats_path, index=False)
    pd.concat(pair_frames, ignore_index=True).to_csv(pairs_path, index=False)

    md_lines = [
        "# Table 2 Compound-Level Statistics (Test Set)\n",
        f"Comparison: GroupA vs Group{comparator}, paired by PUBCHEM_CID.\n",
        "For each endpoint, N is the number of paired compounds in that endpoint's test split.\n",
        dataframe_to_markdown(stats_df),
    ]
    md_path.write_text("\n".join(md_lines))


def main() -> None:
    PAIRWISE_OUT.mkdir(parents=True, exist_ok=True)
    for comparator, _, endpoints in COMPARISONS:
        build_stats_for_comparison(comparator, endpoints)


if __name__ == "__main__":
    main()
