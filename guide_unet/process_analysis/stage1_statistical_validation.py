from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_READOUT_CSV = "results/stage1_case_readout_alignment_n512/table_stage1_case_readout.csv"
DEFAULT_CLASS_REGION_CSV = (
    "stage1_explanation_suite_20260517/results/e9_class_region_response_formal_n512/"
    "table_class_region_response_case_level.csv"
)
DEFAULT_PERTURBATION_CSV = (
    "stage1_explanation_suite_20260517/results/e4_faithfulness_perturbation_formal_n512_allnodes_main/"
    "table_perturbation_random_adjusted.csv"
)
DEFAULT_JOINED_CSV = (
    "stage1_explanation_suite_20260517/results/e9_class_region_response_formal_n512/"
    "table_class_region_perturbation_joined_case_level.csv"
)
DEFAULT_OUTPUT_ROOT = "stage1_explanation_suite_20260517/results/e10_statistical_validation_formal_n512"

NODE_ORDER = ("down1", "down2", "down3", "down4", "up1", "up2", "up3", "up4")
MODEL_LABEL = {"baseline": "有跳接 U-Net", "noskip_unet": "无跳接 U-Net"}
SOURCE_LABEL = {"cam": "CAM", "feature_l2": "原始特征响应"}
REGION_LABEL = {
    "background": "背景",
    "edema": "水肿",
    "necrosis": "坏死/核心",
    "enhancing": "增强肿瘤",
    "tumor_total": "肿瘤总响应",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E10 statistical validation for Stage-1 explanation claims.")
    parser.add_argument("--readout-csv", type=str, default=DEFAULT_READOUT_CSV)
    parser.add_argument("--class-region-csv", type=str, default=DEFAULT_CLASS_REGION_CSV)
    parser.add_argument("--perturbation-csv", type=str, default=DEFAULT_PERTURBATION_CSV)
    parser.add_argument("--joined-csv", type=str, default=DEFAULT_JOINED_CSV)
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def configure_chinese_font() -> None:
    font_candidates = [
        r"C:\Windows\Fonts\NotoSansSC-VF.ttf",
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    for font_path in font_candidates:
        if os.path.isfile(font_path):
            font_manager.fontManager.addfont(font_path)
            plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=font_path).get_name()]
            break
    plt.rcParams["axes.unicode_minus"] = False


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def finite(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def bootstrap_ci(values: Sequence[float], n_boot: int, seed: int) -> Tuple[float, float]:
    arr = finite(values)
    if arr.size < 2 or int(n_boot) <= 0:
        return np.nan, np.nan
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(n_boot), dtype=np.float64)
    for idx in range(int(n_boot)):
        sample_idx = rng.integers(0, arr.size, size=arr.size)
        means[idx] = float(arr[sample_idx].mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def cohen_d(values: Sequence[float]) -> float:
    arr = finite(values)
    if arr.size < 2:
        return np.nan
    sd = float(arr.std(ddof=1))
    if sd <= 1e-12:
        return np.nan
    return float(arr.mean() / sd)


def paired_values(df_a: pd.DataFrame, df_b: pd.DataFrame, value_col: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    a = df_a[["case_name", value_col]].dropna().rename(columns={value_col: "a"})
    b = df_b[["case_name", value_col]].dropna().rename(columns={value_col: "b"})
    merged = a.merge(b, on="case_name", how="inner")
    return merged["a"].to_numpy(dtype=np.float64), merged["b"].to_numpy(dtype=np.float64), merged["case_name"].astype(str).tolist()


def p_wilcoxon(diff: np.ndarray, alternative: str) -> Tuple[float, float]:
    diff = finite(diff)
    if diff.size < 2 or np.allclose(diff, 0.0):
        return np.nan, np.nan
    try:
        res = stats.wilcoxon(diff, alternative=alternative, zero_method="wilcox", method="auto")
        return float(res.statistic), float(res.pvalue)
    except ValueError:
        return np.nan, np.nan


def p_ttest(diff: np.ndarray, alternative: str) -> Tuple[float, float]:
    diff = finite(diff)
    if diff.size < 2:
        return np.nan, np.nan
    res = stats.ttest_1samp(diff, popmean=0.0, alternative=alternative)
    return float(res.statistic), float(res.pvalue)


def add_test_row(
    rows: List[Dict],
    claim_id: str,
    claim: str,
    metric: str,
    comparison: str,
    diff: Sequence[float],
    a: Sequence[float] | None,
    b: Sequence[float] | None,
    alternative: str,
    bootstrap: int,
    seed: int,
    test_family: str,
) -> None:
    diff_arr = finite(diff)
    if diff_arr.size == 0:
        return
    wilcoxon_stat, wilcoxon_p = p_wilcoxon(diff_arr, alternative=alternative)
    t_stat, t_p = p_ttest(diff_arr, alternative=alternative)
    ci_low, ci_high = bootstrap_ci(diff_arr, n_boot=bootstrap, seed=seed)
    payload = {
        "claim_id": claim_id,
        "claim": claim,
        "metric": metric,
        "comparison": comparison,
        "test_family": test_family,
        "n": int(diff_arr.size),
        "mean_a": float(np.nanmean(a)) if a is not None else np.nan,
        "mean_b": float(np.nanmean(b)) if b is not None else np.nan,
        "mean_diff": float(diff_arr.mean()),
        "median_diff": float(np.median(diff_arr)),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "cohen_dz": cohen_d(diff_arr),
        "wilcoxon_stat": wilcoxon_stat,
        "wilcoxon_p": wilcoxon_p,
        "ttest_stat": t_stat,
        "ttest_p": t_p,
        "alternative": alternative,
    }
    rows.append(payload)


def bh_fdr(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=np.float64)
    out = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    if not np.any(valid):
        return out
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    adjusted = np.empty(m, dtype=np.float64)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        val = ranked[i] * m / float(i + 1)
        prev = min(prev, val)
        adjusted[i] = prev
    restored = np.empty(m, dtype=np.float64)
    restored[order] = np.clip(adjusted, 0.0, 1.0)
    out[valid] = restored
    return out


def classify_p(p: float) -> str:
    if not np.isfinite(p):
        return "未检验"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def prepare_readout_tests(readout: pd.DataFrame, rows: List[Dict], bootstrap: int, seed: int) -> None:
    # Endpoint crossing: baseline up4 minus no-skip up4.
    base_up4 = readout[(readout["model"] == "baseline") & (readout["node"] == "up4")]
    noskip_up4 = readout[(readout["model"] == "noskip_unet") & (readout["node"] == "up4")]
    a, b, _ = paired_values(base_up4, noskip_up4, "task_dice_gt_case")
    add_test_row(
        rows,
        "C1_endpoint_crossing",
        "有跳接 U-Net 在 up4 的节点语义一致性是否超过无跳接 U-Net",
        "task_dice_gt_case",
        "baseline_up4 - noskip_up4",
        a - b,
        a,
        b,
        "greater",
        bootstrap,
        seed + 1,
        "readout",
    )

    # Late decoder delta comparison.
    delta_rows = []
    for model_name, group in readout.groupby("model", sort=False):
        pivot = group.pivot_table(index="case_name", columns="node", values="task_dice_gt_case", aggfunc="mean")
        for transition in [("up1", "up2"), ("up2", "up3"), ("up3", "up4")]:
            src, dst = transition
            delta = pivot[dst] - pivot[src]
            tmp = pd.DataFrame(
                {
                    "case_name": delta.index.astype(str),
                    "model": model_name,
                    "transition": "{}->{}".format(src, dst),
                    "delta": delta.to_numpy(dtype=np.float64),
                }
            )
            delta_rows.append(tmp)
    delta_df = pd.concat(delta_rows, ignore_index=True)
    for model_name in ("baseline", "noskip_unet"):
        sub = delta_df[(delta_df["model"] == model_name) & (delta_df["transition"] == "up3->up4")]
        add_test_row(
            rows,
            "C2_late_delta_positive_{}".format(model_name),
            "{} 的 up3->up4 节点语义一致性增量是否大于 0".format(MODEL_LABEL.get(model_name, model_name)),
            "task_dice_gt_delta",
            "{}_up3_to_up4_delta > 0".format(model_name),
            sub["delta"].to_numpy(dtype=np.float64),
            sub["delta"].to_numpy(dtype=np.float64),
            None,
            "greater",
            bootstrap,
            seed + 2,
            "readout",
        )
    base_delta = delta_df[(delta_df["model"] == "baseline") & (delta_df["transition"] == "up3->up4")]
    noskip_delta = delta_df[(delta_df["model"] == "noskip_unet") & (delta_df["transition"] == "up3->up4")]
    a, b, _ = paired_values(base_delta, noskip_delta, "delta")
    add_test_row(
        rows,
        "C3_late_acceleration_gap",
        "有跳接 U-Net 的 up3->up4 增量是否大于无跳接 U-Net",
        "task_dice_gt_delta",
        "baseline_up3_to_up4_delta - noskip_up3_to_up4_delta",
        a - b,
        a,
        b,
        "greater",
        bootstrap,
        seed + 3,
        "readout",
    )


def class_region_wide(class_region: pd.DataFrame) -> pd.DataFrame:
    value_cols = [
        "region_area_fraction",
        "response_mass_fraction",
        "top10_area_fraction",
        "top10_enrichment",
    ]
    wide = class_region.pivot_table(
        index=["model", "case_name", "node", "response_source"],
        columns="region",
        values=value_cols,
        aggfunc="mean",
    )
    wide.columns = ["{}_{}".format(metric, region) for metric, region in wide.columns]
    wide = wide.reset_index()
    for metric in value_cols:
        cols = ["{}_{}".format(metric, r) for r in ("edema", "necrosis", "enhancing")]
        if all(c in wide.columns for c in cols):
            wide["{}_tumor_total".format(metric)] = wide[cols].sum(axis=1)
    return wide


def prepare_class_region_tests(class_region: pd.DataFrame, rows: List[Dict], bootstrap: int, seed: int) -> None:
    wide = class_region_wide(class_region)
    cam = wide[wide["response_source"] == "cam"].copy()
    # Up4 tumor mass baseline vs no-skip.
    base = cam[(cam["model"] == "baseline") & (cam["node"] == "up4")]
    noskip = cam[(cam["model"] == "noskip_unet") & (cam["node"] == "up4")]
    a, b, _ = paired_values(base, noskip, "response_mass_fraction_tumor_total")
    add_test_row(
        rows,
        "C4_up4_tumor_mass_crossing",
        "up4 时有跳接 U-Net 的 CAM 肿瘤总响应质量是否高于无跳接 U-Net",
        "cam_response_mass_fraction_tumor_total",
        "baseline_up4_tumor_mass - noskip_up4_tumor_mass",
        a - b,
        a,
        b,
        "greater",
        bootstrap,
        seed + 4,
        "class_region",
    )
    # Up3->up4 region transfer.
    deltas = []
    delta_metrics = [
        "response_mass_fraction_background",
        "response_mass_fraction_edema",
        "response_mass_fraction_enhancing",
        "response_mass_fraction_tumor_total",
    ]
    for (model_name, source), group in wide.groupby(["model", "response_source"], sort=False):
        pivot = group.pivot_table(index="case_name", columns="node", values=delta_metrics, aggfunc="mean")
        for metric in delta_metrics:
            if (metric, "up3") not in pivot.columns or (metric, "up4") not in pivot.columns:
                continue
            delta = pivot[(metric, "up4")] - pivot[(metric, "up3")]
            deltas.append(
                pd.DataFrame(
                    {
                        "case_name": delta.index.astype(str),
                        "model": model_name,
                        "response_source": source,
                        "metric": metric,
                        "delta": delta.to_numpy(dtype=np.float64),
                    }
                )
            )
    delta_df = pd.concat(deltas, ignore_index=True)
    for metric, alternative in [
        ("response_mass_fraction_background", "less"),
        ("response_mass_fraction_edema", "greater"),
        ("response_mass_fraction_enhancing", "greater"),
        ("response_mass_fraction_tumor_total", "greater"),
    ]:
        sub = delta_df[
            (delta_df["model"] == "baseline") & (delta_df["response_source"] == "cam") & (delta_df["metric"] == metric)
        ]
        add_test_row(
            rows,
            "C5_baseline_late_transfer_{}".format(metric.replace("response_mass_fraction_", "")),
            "有跳接 U-Net 的 CAM 在 up3->up4 是否发生关键类别转移",
            "cam_{}_delta".format(metric),
            "baseline_cam_up4_minus_up3_{}".format(metric),
            sub["delta"].to_numpy(dtype=np.float64),
            sub["delta"].to_numpy(dtype=np.float64),
            None,
            alternative,
            bootstrap,
            seed + 5,
            "class_region",
        )
    # Edema response mass vs area and enrichment vs 1 at up4.
    for model_name in ("baseline", "noskip_unet"):
        sub = cam[(cam["model"] == model_name) & (cam["node"] == "up4")]
        diff = sub["response_mass_fraction_edema"].to_numpy(dtype=np.float64) - sub[
            "region_area_fraction_edema"
        ].to_numpy(dtype=np.float64)
        add_test_row(
            rows,
            "C6_edema_mass_above_area_{}".format(model_name),
            "{} 的 up4 CAM 水肿响应质量是否高于水肿面积占比".format(MODEL_LABEL.get(model_name, model_name)),
            "cam_up4_edema_mass_minus_area",
            "{}_edema_mass_fraction - edema_area_fraction".format(model_name),
            diff,
            sub["response_mass_fraction_edema"].to_numpy(dtype=np.float64),
            sub["region_area_fraction_edema"].to_numpy(dtype=np.float64),
            "greater",
            bootstrap,
            seed + 6,
            "class_region",
        )
        enrich_diff = sub["top10_enrichment_edema"].to_numpy(dtype=np.float64) - 1.0
        add_test_row(
            rows,
            "C7_edema_enrichment_above_one_{}".format(model_name),
            "{} 的 up4 CAM 前 10% 高响应区域是否富集水肿".format(MODEL_LABEL.get(model_name, model_name)),
            "cam_up4_edema_top10_enrichment_minus_1",
            "{}_edema_enrichment - 1".format(model_name),
            enrich_diff,
            sub["top10_enrichment_edema"].to_numpy(dtype=np.float64),
            np.ones(len(sub), dtype=np.float64),
            "greater",
            bootstrap,
            seed + 7,
            "class_region",
        )


def prepare_perturbation_tests(perturbation: pd.DataFrame, rows: List[Dict], bootstrap: int, seed: int) -> None:
    df = perturbation[
        (np.isclose(perturbation["top_fraction"], 0.10))
        & (np.isclose(perturbation["gamma"], 1.0))
        & (perturbation["mask_type"] == "topk")
    ].copy()
    metric = "random_adjusted_final_fg_prob_drop"
    for model_name in ("baseline", "noskip_unet"):
        for source in ("cam", "feature_l2"):
            sub = df[(df["model"] == model_name) & (df["response_source"] == source) & (df["node"] == "up4")]
            add_test_row(
                rows,
                "C8_up4_perturbation_{}_{}".format(model_name, source),
                "{} 的 up4 {} 高响应遮挡是否导致额外输出下降".format(
                    MODEL_LABEL.get(model_name, model_name), SOURCE_LABEL.get(source, source)
                ),
                "{}_up4_random_adjusted_final_fg_prob_drop".format(source),
                "{}_{}_up4_drop > 0".format(model_name, source),
                sub[metric].to_numpy(dtype=np.float64),
                sub[metric].to_numpy(dtype=np.float64),
                None,
                "greater",
                bootstrap,
                seed + 8,
                "perturbation",
            )
    # Timing pattern: baseline up4 exceeds up3, no-skip up2 exceeds up4.
    for model_name, source, node_a, node_b, claim_id in [
        ("baseline", "cam", "up4", "up3", "C9_baseline_up4_drop_exceeds_up3"),
        ("noskip_unet", "cam", "up2", "up4", "C10_noskip_up2_drop_exceeds_up4"),
    ]:
        a_df = df[(df["model"] == model_name) & (df["response_source"] == source) & (df["node"] == node_a)]
        b_df = df[(df["model"] == model_name) & (df["response_source"] == source) & (df["node"] == node_b)]
        a, b, _ = paired_values(a_df, b_df, metric)
        add_test_row(
            rows,
            claim_id,
            "{} 的 CAM 遮挡敏感性时序是否符合曲线解释".format(MODEL_LABEL.get(model_name, model_name)),
            "cam_random_adjusted_final_fg_prob_drop",
            "{}_{}_drop - {}_{}_drop".format(model_name, node_a, model_name, node_b),
            a - b,
            a,
            b,
            "greater",
            bootstrap,
            seed + 9,
            "perturbation",
        )


def bootstrap_corr_ci(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 6 or int(n_boot) <= 0:
        return np.nan, np.nan
    rng = np.random.default_rng(int(seed))
    vals = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, x.size, size=x.size)
        xx, yy = x[idx], y[idx]
        if float(np.std(xx)) <= 1e-12 or float(np.std(yy)) <= 1e-12:
            continue
        vals.append(float(stats.pearsonr(xx, yy).statistic))
    if not vals:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def prepare_correlation_tests(joined: pd.DataFrame, bootstrap: int, seed: int) -> pd.DataFrame:
    rows = []
    outcome = "random_adjusted_final_fg_prob_drop"
    predictors = {
        "background": "top10_area_fraction_background",
        "edema": "top10_area_fraction_edema",
        "necrosis": "top10_area_fraction_necrosis",
        "enhancing": "top10_area_fraction_enhancing",
    }
    df = joined[(joined["node"] == "up4") & (np.isclose(joined["top_fraction"], 0.10)) & (np.isclose(joined["gamma"], 1.0))]
    for (model_name, source), group in df.groupby(["model", "response_source"], sort=False):
        for region, predictor in predictors.items():
            x = group[predictor].to_numpy(dtype=np.float64)
            y = group[outcome].to_numpy(dtype=np.float64)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 3 or float(np.std(x[mask])) <= 1e-12 or float(np.std(y[mask])) <= 1e-12:
                r, p = np.nan, np.nan
            else:
                res = stats.pearsonr(x[mask], y[mask])
                r, p = float(res.statistic), float(res.pvalue)
            ci_low, ci_high = bootstrap_corr_ci(x, y, bootstrap, seed + 20)
            rows.append(
                {
                    "model": model_name,
                    "response_source": source,
                    "node": "up4",
                    "region": region,
                    "predictor": predictor,
                    "outcome": outcome,
                    "n": int(mask.sum()),
                    "pearson_r": r,
                    "pearson_ci95_low": ci_low,
                    "pearson_ci95_high": ci_high,
                    "pearson_p": p,
                }
            )
    out = pd.DataFrame(rows)
    out["pearson_p_fdr"] = bh_fdr(out["pearson_p"].to_numpy(dtype=np.float64))
    out["significance"] = [classify_p(p) for p in out["pearson_p_fdr"]]
    return out


def plot_effects(tests: pd.DataFrame, out_path: str) -> None:
    configure_chinese_font()
    plot_df = tests[
        tests["claim_id"].isin(
            [
                "C1_endpoint_crossing",
                "C3_late_acceleration_gap",
                "C4_up4_tumor_mass_crossing",
                "C5_baseline_late_transfer_background",
                "C5_baseline_late_transfer_edema",
                "C8_up4_perturbation_baseline_cam",
                "C9_baseline_up4_drop_exceeds_up3",
                "C10_noskip_up2_drop_exceeds_up4",
            ]
        )
    ].copy()
    labels = {
        "C1_endpoint_crossing": "终点语义读数反超",
        "C3_late_acceleration_gap": "后期语义增量差",
        "C4_up4_tumor_mass_crossing": "up4 肿瘤响应总量差",
        "C5_baseline_late_transfer_background": "有跳接背景响应下降",
        "C5_baseline_late_transfer_edema": "有跳接水肿响应上升",
        "C8_up4_perturbation_baseline_cam": "有跳接 up4 遮挡影响",
        "C9_baseline_up4_drop_exceeds_up3": "有跳接遮挡影响末端增强",
        "C10_noskip_up2_drop_exceeds_up4": "无跳接遮挡影响早期更强",
    }
    plot_df["label"] = plot_df["claim_id"].map(labels)
    plot_df = plot_df.dropna(subset=["label", "mean_diff"]).iloc[::-1]
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(9.0, 0.48 * len(plot_df) + 1.2), dpi=180)
    y = np.arange(len(plot_df))
    x = plot_df["mean_diff"].to_numpy(dtype=np.float64)
    lo = plot_df["ci95_low"].to_numpy(dtype=np.float64)
    hi = plot_df["ci95_high"].to_numpy(dtype=np.float64)
    ax.errorbar(x, y, xerr=[x - lo, hi - x], fmt="o", color="#1f77b4", ecolor="#666666", capsize=3)
    ax.axvline(0.0, color="#777777", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"].tolist(), fontsize=9)
    ax.set_xlabel("均值差或相对 0 的均值效应（95% bootstrap CI）")
    ax.set_title("Stage 1 解释链关键统计效应")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_root)
    readout = pd.read_csv(args.readout_csv)
    class_region = pd.read_csv(args.class_region_csv)
    perturbation = pd.read_csv(args.perturbation_csv)
    joined = pd.read_csv(args.joined_csv)

    rows: List[Dict] = []
    prepare_readout_tests(readout, rows, bootstrap=int(args.bootstrap), seed=int(args.seed))
    prepare_class_region_tests(class_region, rows, bootstrap=int(args.bootstrap), seed=int(args.seed))
    prepare_perturbation_tests(perturbation, rows, bootstrap=int(args.bootstrap), seed=int(args.seed))
    tests = pd.DataFrame(rows)
    tests["wilcoxon_p_fdr"] = bh_fdr(tests["wilcoxon_p"].to_numpy(dtype=np.float64))
    tests["ttest_p_fdr"] = bh_fdr(tests["ttest_p"].to_numpy(dtype=np.float64))
    tests["primary_p"] = tests["wilcoxon_p"]
    tests["primary_p_fdr"] = tests["wilcoxon_p_fdr"]
    tests["significance"] = [classify_p(p) for p in tests["primary_p_fdr"]]
    tests = tests.sort_values(["test_family", "claim_id"]).reset_index(drop=True)

    corr = prepare_correlation_tests(joined, bootstrap=int(args.bootstrap), seed=int(args.seed))

    tests.to_csv(os.path.join(args.output_root, "table_e10_key_statistical_tests.csv"), index=False)
    corr.to_csv(os.path.join(args.output_root, "table_e10_location_drop_correlation_tests.csv"), index=False)
    plot_effects(tests, os.path.join(args.output_root, "plot_e10_key_effects_with_ci.png"))

    summary = {
        "purpose": "E10 statistical validation for Stage-1 explanation evidence chain",
        "n_tests": int(len(tests)),
        "n_correlation_tests": int(len(corr)),
        "primary_test": "paired/one-sample Wilcoxon with BH-FDR; t-test also reported",
        "bootstrap": int(args.bootstrap),
        "inputs": {
            "readout_csv": args.readout_csv,
            "class_region_csv": args.class_region_csv,
            "perturbation_csv": args.perturbation_csv,
            "joined_csv": args.joined_csv,
        },
        "outputs": [
            "table_e10_key_statistical_tests.csv",
            "table_e10_location_drop_correlation_tests.csv",
            "plot_e10_key_effects_with_ci.png",
        ],
    }
    with open(os.path.join(args.output_root, "summary_statistical_validation.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
