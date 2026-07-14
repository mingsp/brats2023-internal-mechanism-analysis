from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NODE_ORDER = ("down1", "down2", "down3", "down4", "up1", "up2", "up3", "up4")
TRANSITIONS = tuple(zip(NODE_ORDER[:-1], NODE_ORDER[1:]))
KEY_METRICS = (
    "cam_final_prob_pearson",
    "cam_gt_mass_fraction",
    "cam_gt_fg_bg_contrast",
    "cam_entropy_norm",
    "feature_final_prob_pearson",
    "feature_gt_mass_fraction",
    "feature_gt_fg_bg_contrast",
    "feature_entropy_norm",
)
CASE_DESCRIPTOR_METRICS = (
    "final_vs_gt_mean_dice",
    "gt_fg_pixels",
    "pred_fg_pixels",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Close the Stage-1 phenomenon explanation loop with sensitivity, bootstrap, and transition checks."
    )
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--stage1_csv", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top_cases", type=int, default=3)
    return parser.parse_args()


def patient_id(case_name: str) -> str:
    text = str(case_name)
    return text.rsplit("_", 1)[0] if "_" in text else text


def require_file(path: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return path


def read_stage1(stage1_csv: str) -> pd.DataFrame:
    df = pd.read_csv(require_file(stage1_csv))
    rows: List[Dict] = []
    for _, row in df.iterrows():
        node = str(row["node"])
        if node not in NODE_ORDER:
            continue
        node_order = int(row.get("node_order", NODE_ORDER.index(node)))
        for model, suffix in (("baseline", "baseline"), ("noskip_unet", "noskip")):
            rows.append(
                {
                    "model": model,
                    "node": node,
                    "node_order": node_order,
                    "metric": "task_dice_gt",
                    "value": float(row[f"task_dice_gt_{suffix}"]),
                }
            )
    out = pd.DataFrame(rows).sort_values(["model", "node_order"]).reset_index(drop=True)
    dyn_rows: List[Dict] = []
    for model, group in out.groupby("model", sort=False):
        values = group.sort_values("node_order")["value"].astype(float).to_numpy()
        nodes = group.sort_values("node_order")["node"].astype(str).tolist()
        velocities = np.diff(values)
        accelerations = np.diff(velocities)
        for idx, node in enumerate(nodes):
            dyn_rows.append(
                {
                    "model": model,
                    "node": node,
                    "node_order": idx,
                    "metric": "task_dice_gt",
                    "value": values[idx],
                    "velocity_to_next": velocities[idx] if idx < len(velocities) else np.nan,
                    "acceleration_to_next": accelerations[idx] if idx < len(accelerations) else np.nan,
                }
            )
    return pd.DataFrame(dyn_rows)


def load_result_dir(result_dir: str, label: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    node = pd.read_csv(require_file(os.path.join(result_dir, "table_phase1_2_gradcam_node_summary.csv")))
    case = pd.read_csv(require_file(os.path.join(result_dir, "table_phase1_2_gradcam_per_case.csv")))
    dyn = pd.read_csv(require_file(os.path.join(result_dir, "table_phase0_3_semantic_cam_dynamics.csv")))
    for frame in (node, case, dyn):
        frame["target_source"] = label
    case["patient_id"] = case["case_name"].map(patient_id)
    return node, case, dyn


def bootstrap_mean_ci(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> Tuple[float, float, float, int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan, 0
    if values.size == 1 or n_boot <= 0:
        val = float(values.mean())
        return val, val, val, int(values.size)
    idx = rng.integers(0, values.size, size=(int(n_boot), values.size))
    boot = values[idx].mean(axis=1)
    return float(values.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), int(values.size)


def patient_metric_table(case_df: pd.DataFrame) -> pd.DataFrame:
    descriptor_cols = [col for col in CASE_DESCRIPTOR_METRICS if col in case_df.columns]
    cols = [
        "target_source",
        "target_mode",
        "model",
        "patient_id",
        "case_name",
        "node",
        "node_order",
        *KEY_METRICS,
        *descriptor_cols,
    ]
    return case_df[cols].groupby(
        ["target_source", "target_mode", "model", "patient_id", "case_name", "node", "node_order"], as_index=False
    ).mean()


def patient_bootstrap_tables(patient_df: pd.DataFrame, n_boot: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    node_rows: List[Dict] = []
    transition_rows: List[Dict] = []
    for keys, group in patient_df.groupby(["target_source", "target_mode", "model", "node", "node_order"], sort=False):
        target_source, target_mode, model, node, node_order = keys
        patient_group = group.groupby("patient_id", as_index=False)[list(KEY_METRICS)].mean()
        for metric in KEY_METRICS:
            mean, lo, hi, n = bootstrap_mean_ci(patient_group[metric].to_numpy(), n_boot=n_boot, rng=rng)
            node_rows.append(
                {
                    "target_source": target_source,
                    "target_mode": target_mode,
                    "model": model,
                    "node": node,
                    "node_order": int(node_order),
                    "metric": metric,
                    "mean": mean,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "n_patients": n,
                }
            )

    patient_node = patient_df.groupby(["target_source", "target_mode", "model", "patient_id", "node", "node_order"], as_index=False)[
        list(KEY_METRICS)
    ].mean()
    for (target_source, target_mode, model, patient), group in patient_node.groupby(
        ["target_source", "target_mode", "model", "patient_id"], sort=False
    ):
        pivot = group.set_index("node")
        for src, dst in TRANSITIONS:
            if src not in pivot.index or dst not in pivot.index:
                continue
            for metric in KEY_METRICS:
                transition_rows.append(
                    {
                        "target_source": target_source,
                        "target_mode": target_mode,
                        "model": model,
                        "patient_id": patient,
                        "transition": f"{src}->{dst}",
                        "source_node": src,
                        "target_node": dst,
                        "source_order": NODE_ORDER.index(src),
                        "metric": metric,
                        "delta": float(pivot.loc[dst, metric] - pivot.loc[src, metric]),
                    }
                )
    transition_df = pd.DataFrame(transition_rows)
    transition_ci_rows: List[Dict] = []
    for keys, group in transition_df.groupby(["target_source", "target_mode", "model", "transition", "source_order", "metric"], sort=False):
        target_source, target_mode, model, transition, source_order, metric = keys
        mean, lo, hi, n = bootstrap_mean_ci(group["delta"].to_numpy(), n_boot=n_boot, rng=rng)
        transition_ci_rows.append(
            {
                "target_source": target_source,
                "target_mode": target_mode,
                "model": model,
                "transition": transition,
                "source_order": int(source_order),
                "metric": metric,
                "delta_mean": mean,
                "delta_ci95_low": lo,
                "delta_ci95_high": hi,
                "n_patients": n,
            }
        )
    return pd.DataFrame(node_rows), pd.DataFrame(transition_ci_rows), transition_df


def corr_pair(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    sx = pd.Series(x, dtype=float)
    sy = pd.Series(y, dtype=float)
    mask = sx.notna() & sy.notna()
    if int(mask.sum()) < 3:
        return np.nan, np.nan
    return float(sx[mask].corr(sy[mask], method="pearson")), float(sx[mask].corr(sy[mask], method="spearman"))


def transition_alignment(stage1_df: pd.DataFrame, node_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    stage_transitions: Dict[str, Dict[str, float]] = {}
    for model, group in stage1_df[stage1_df["metric"] == "task_dice_gt"].groupby("model"):
        pivot = group.set_index("node")
        stage_transitions[model] = {
            f"{src}->{dst}": float(pivot.loc[dst, "value"] - pivot.loc[src, "value"])
            for src, dst in TRANSITIONS
            if src in pivot.index and dst in pivot.index
        }
    for (target_source, target_mode, model), group in node_df.groupby(["target_source", "target_mode", "model"]):
        pivot = group.set_index("node")
        for metric in KEY_METRICS:
            task_delta: List[float] = []
            metric_delta: List[float] = []
            transitions: List[str] = []
            for src, dst in TRANSITIONS:
                transition = f"{src}->{dst}"
                if transition not in stage_transitions.get(model, {}) or src not in pivot.index or dst not in pivot.index:
                    continue
                task_delta.append(stage_transitions[model][transition])
                metric_delta.append(float(pivot.loc[dst, metric] - pivot.loc[src, metric]))
                transitions.append(transition)
            pearson, spearman = corr_pair(task_delta, metric_delta)
            rows.append(
                {
                    "target_source": target_source,
                    "target_mode": target_mode,
                    "model": model,
                    "metric": metric,
                    "n_transitions": len(transitions),
                    "pearson_task_delta_vs_metric_delta": pearson,
                    "spearman_task_delta_vs_metric_delta": spearman,
                    "scope": "transition_level_node_mean_not_case_level",
                    "transitions": ",".join(transitions),
                }
            )
    return pd.DataFrame(rows)


def value_at(df: pd.DataFrame, target_source: str, model: str, node: str, metric: str) -> float:
    sub = df[(df["target_source"] == target_source) & (df["model"] == model) & (df["node"] == node)]
    return float(sub.iloc[0][metric]) if not sub.empty else np.nan


def mean_nodes(df: pd.DataFrame, target_source: str, model: str, nodes: Sequence[str], metric: str) -> float:
    vals = [value_at(df, target_source, model, node, metric) for node in nodes]
    vals = [v for v in vals if math.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


def claim_support_checks(node_df: pd.DataFrame, stage1_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    for target_source in sorted(node_df["target_source"].unique()):
        for metric in ("feature_final_prob_pearson", "feature_gt_mass_fraction", "feature_gt_fg_bg_contrast"):
            base = mean_nodes(node_df, target_source, "baseline", ("down1", "down2", "down3"), metric)
            noskip = mean_nodes(node_df, target_source, "noskip_unet", ("down1", "down2", "down3"), metric)
            rows.append(
                {
                    "claim": "early_gap_noskip_feature_response_higher",
                    "target_source": target_source,
                    "metric": metric,
                    "baseline_value": base,
                    "noskip_value": noskip,
                    "effect": noskip - base,
                    "passed": bool(noskip > base),
                    "evidence_scope": "node_mean_down1_down3",
                }
            )
        for metric in ("feature_final_prob_pearson", "feature_gt_mass_fraction", "feature_gt_fg_bg_contrast"):
            down3 = value_at(node_df, target_source, "baseline", "down3", metric)
            valley = np.nanmean(
                [
                    value_at(node_df, target_source, "baseline", "down4", metric),
                    value_at(node_df, target_source, "baseline", "up1", metric),
                ]
            )
            rows.append(
                {
                    "claim": "baseline_down4_up1_valley",
                    "target_source": target_source,
                    "metric": metric,
                    "baseline_value": valley,
                    "reference_value": down3,
                    "effect": valley - down3,
                    "passed": bool(valley < down3),
                    "evidence_scope": "node_mean_down3_vs_down4_up1",
                }
            )
        for metric, expected_sign in (
            ("cam_final_prob_pearson", 1.0),
            ("cam_gt_mass_fraction", 1.0),
            ("cam_gt_fg_bg_contrast", 1.0),
            ("cam_entropy_norm", -1.0),
        ):
            base_delta = value_at(node_df, target_source, "baseline", "up4", metric) - value_at(
                node_df, target_source, "baseline", "up3", metric
            )
            noskip_delta = value_at(node_df, target_source, "noskip_unet", "up4", metric) - value_at(
                node_df, target_source, "noskip_unet", "up3", metric
            )
            rows.append(
                {
                    "claim": "baseline_late_jump_stronger_than_noskip",
                    "target_source": target_source,
                    "metric": metric,
                    "baseline_delta_up3_up4": base_delta,
                    "noskip_delta_up3_up4": noskip_delta,
                    "effect": expected_sign * (base_delta - noskip_delta),
                    "passed": bool(expected_sign * base_delta > 0 and expected_sign * (base_delta - noskip_delta) > 0),
                    "evidence_scope": "node_mean_up3_to_up4",
                }
            )
        for metric in ("cam_final_prob_pearson", "cam_gt_mass_fraction", "cam_gt_fg_bg_contrast"):
            base = value_at(node_df, target_source, "baseline", "up4", metric)
            noskip = value_at(node_df, target_source, "noskip_unet", "up4", metric)
            rows.append(
                {
                    "claim": "baseline_final_up4_alignment_higher",
                    "target_source": target_source,
                    "metric": metric,
                    "baseline_value": base,
                    "noskip_value": noskip,
                    "effect": base - noskip,
                    "passed": bool(base > noskip),
                    "evidence_scope": "node_mean_up4",
                }
            )
    stage = stage1_df[stage1_df["metric"] == "task_dice_gt"]
    for model, group in stage.groupby("model"):
        pivot = group.set_index("node")
        up_deltas = [
            float(pivot.loc["up2", "value"] - pivot.loc["up1", "value"]),
            float(pivot.loc["up3", "value"] - pivot.loc["up2", "value"]),
            float(pivot.loc["up4", "value"] - pivot.loc["up3", "value"]),
        ]
        rows.append(
            {
                "claim": "stage1_upstage_acceleration_pattern",
                "target_source": "stage1",
                "metric": "task_dice_gt",
                "model": model,
                "delta_up1_up2": up_deltas[0],
                "delta_up2_up3": up_deltas[1],
                "delta_up3_up4": up_deltas[2],
                "passed": bool(up_deltas[0] < up_deltas[1] < up_deltas[2])
                if model == "baseline"
                else bool(up_deltas[0] > up_deltas[1] > up_deltas[2]),
                "evidence_scope": "stage1_node_mean",
            }
        )
    return pd.DataFrame(rows)


def case_scores(patient_df: pd.DataFrame, target_source: str = "pred_fg") -> pd.DataFrame:
    sub = patient_df[patient_df["target_source"] == target_source]
    rows: List[Dict] = []
    for (patient, case_name), group in sub.groupby(["patient_id", "case_name"]):
        pivot = group.set_index(["model", "node"])
        needed = [("baseline", node) for node in NODE_ORDER] + [("noskip_unet", node) for node in NODE_ORDER]
        if any(item not in pivot.index for item in needed):
            continue
        early_base = np.mean([pivot.loc[("baseline", n), "feature_final_prob_pearson"] for n in ("down1", "down2", "down3")])
        early_noskip = np.mean([pivot.loc[("noskip_unet", n), "feature_final_prob_pearson"] for n in ("down1", "down2", "down3")])
        early_gap = float(early_noskip - early_base)
        valley = float(
            pivot.loc[("baseline", "down3"), "feature_final_prob_pearson"]
            - np.mean(
                [
                    pivot.loc[("baseline", "down4"), "feature_final_prob_pearson"],
                    pivot.loc[("baseline", "up1"), "feature_final_prob_pearson"],
                ]
            )
        )
        base_late = float(
            (pivot.loc[("baseline", "up4"), "cam_final_prob_pearson"] - pivot.loc[("baseline", "up3"), "cam_final_prob_pearson"])
            + (pivot.loc[("baseline", "up4"), "cam_gt_mass_fraction"] - pivot.loc[("baseline", "up3"), "cam_gt_mass_fraction"])
            + (pivot.loc[("baseline", "up4"), "cam_gt_fg_bg_contrast"] - pivot.loc[("baseline", "up3"), "cam_gt_fg_bg_contrast"])
            - (pivot.loc[("baseline", "up4"), "cam_entropy_norm"] - pivot.loc[("baseline", "up3"), "cam_entropy_norm"])
        )
        noskip_late = float(
            (pivot.loc[("noskip_unet", "up4"), "cam_final_prob_pearson"] - pivot.loc[("noskip_unet", "up3"), "cam_final_prob_pearson"])
            + (pivot.loc[("noskip_unet", "up4"), "cam_gt_mass_fraction"] - pivot.loc[("noskip_unet", "up3"), "cam_gt_mass_fraction"])
            + (pivot.loc[("noskip_unet", "up4"), "cam_gt_fg_bg_contrast"] - pivot.loc[("noskip_unet", "up3"), "cam_gt_fg_bg_contrast"])
            - (pivot.loc[("noskip_unet", "up4"), "cam_entropy_norm"] - pivot.loc[("noskip_unet", "up3"), "cam_entropy_norm"])
        )
        late_advantage = base_late - noskip_late
        gt_fg_pixels = float(pivot.loc[("baseline", "up4"), "gt_fg_pixels"])
        baseline_up4_prob_r = float(pivot.loc[("baseline", "up4"), "cam_final_prob_pearson"])
        baseline_up4_gt_mass = float(pivot.loc[("baseline", "up4"), "cam_gt_mass_fraction"])
        rows.append(
            {
                "patient_id": patient,
                "case_name": case_name,
                "target_source": target_source,
                "early_gap_score": early_gap,
                "baseline_valley_score": valley,
                "baseline_late_jump_score": base_late,
                "late_advantage_over_noskip": late_advantage,
                "all_components_support": bool(early_gap > 0 and valley > 0 and base_late > 0 and late_advantage > 0),
                "gt_fg_pixels": gt_fg_pixels,
                "baseline_up4_cam_final_prob_pearson": baseline_up4_prob_r,
                "baseline_up4_cam_gt_mass_fraction": baseline_up4_gt_mass,
                "visual_evidence_usable": bool(gt_fg_pixels >= 1000 and baseline_up4_prob_r >= 0.40),
                "total_evidence_score": early_gap + valley + base_late + late_advantage,
            }
        )
    return pd.DataFrame(rows).sort_values("total_evidence_score", ascending=False)


def plot_sensitivity(node_df: pd.DataFrame, save_path: str) -> None:
    metrics = [
        ("feature_final_prob_pearson", "Feature-final-prob Pearson"),
        ("feature_gt_mass_fraction", "Feature GT mass"),
        ("cam_final_prob_pearson", "CAM-final-prob Pearson"),
        ("cam_gt_mass_fraction", "CAM GT mass"),
        ("cam_gt_fg_bg_contrast", "CAM GT contrast"),
        ("cam_entropy_norm", "CAM entropy"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), dpi=160)
    axes = axes.reshape(-1)
    styles = {
        ("pred_fg", "baseline"): ("#1f77b4", "-"),
        ("pred_fg", "noskip_unet"): ("#ff7f0e", "-"),
        ("gt_fg", "baseline"): ("#1f77b4", "--"),
        ("gt_fg", "noskip_unet"): ("#ff7f0e", "--"),
    }
    for ax, (metric, title) in zip(axes, metrics):
        for (target_source, model), group in node_df.groupby(["target_source", "model"]):
            group = group.sort_values("node_order")
            color, linestyle = styles.get((target_source, model), ("#333333", "-"))
            ax.plot(group["node"], group[metric].astype(float), marker="o", linewidth=2, color=color, linestyle=linestyle, label=f"{model} {target_source}")
        ax.set_title(title)
        ax.set_xlabel("network node")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_transition_ci(transition_ci: pd.DataFrame, save_path: str) -> None:
    metrics = ["cam_final_prob_pearson", "cam_gt_mass_fraction", "cam_gt_fg_bg_contrast", "cam_entropy_norm"]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 3.2 * len(metrics)), dpi=160)
    colors = {"baseline": "#1f77b4", "noskip_unet": "#ff7f0e"}
    for ax, metric in zip(axes, metrics):
        sub = transition_ci[(transition_ci["target_source"] == "pred_fg") & (transition_ci["metric"] == metric)]
        for model, group in sub.groupby("model"):
            group = group.sort_values("source_order")
            x = np.arange(len(group))
            y = group["delta_mean"].astype(float).to_numpy()
            lo = group["delta_ci95_low"].astype(float).to_numpy()
            hi = group["delta_ci95_high"].astype(float).to_numpy()
            ax.plot(group["transition"], y, marker="o", linewidth=2, label=model, color=colors.get(model))
            ax.fill_between(x, lo, hi, alpha=0.15, color=colors.get(model))
        ax.axhline(0.0, color="#777777", linestyle="--", linewidth=1)
        ax.set_title(f"Patient-level transition delta with bootstrap CI: {metric}")
        ax.grid(alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def dataframe_to_markdown(df: pd.DataFrame, cols: Sequence[str]) -> str:
    if df.empty:
        return ""
    view = df.loc[:, list(cols)].copy()
    lines = [
        "| " + " | ".join(view.columns.astype(str)) + " |",
        "| " + " | ".join(["---"] * len(view.columns)) + " |",
    ]
    for _, row in view.iterrows():
        values = []
        for col in view.columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_markdown(args: argparse.Namespace, claim_df: pd.DataFrame, selected: pd.DataFrame, out_path: str) -> None:
    pass_rate = float(claim_df["passed"].mean()) if "passed" in claim_df and not claim_df.empty else np.nan
    lines = [
        "# Stage 1 解释闭环验证结果",
        "",
        "本文档记录 Stage 1 现象原因解释的闭环验证输出。当前目标不是扩展新方向，而是检查已有解释是否被可视化和量化证据支撑。",
        "",
        "## 1. 输入结果",
        "",
        f"- pred_fg 结果目录：`{args.pred_dir}`",
        f"- gt_fg 结果目录：`{args.gt_dir}`",
        f"- Stage 1 曲线表：`{args.stage1_csv}`",
        "",
        "## 2. 关键输出",
        "",
        "- `table_target_mode_sensitivity_node_summary.csv`：pred_fg 与 gt_fg 的节点指标对照。",
        "- `table_patient_node_bootstrap_ci.csv`：病例级节点指标 bootstrap 置信区间。",
        "- `table_patient_transition_bootstrap_ci.csv`：病例级相邻节点增量 bootstrap 置信区间。",
        "- `table_transition_alignment_task_vs_cam_feature.csv`：节点均值层面的 task_dice_gt 增量与 CAM/feature 增量一致性。",
        "- `table_claim_support_checks.csv`：逐条机制解释是否通过量化检查。",
        "- `table_typical_counterexample_cases.csv`：典型病例和反例候选。",
        "- `plot_target_mode_sensitivity_curves.png`：pred_fg/gt_fg 敏感性曲线。",
        "- `plot_patient_transition_bootstrap_ci.png`：病例级转移增量置信区间图。",
        "",
        "## 3. 当前检查结论",
        "",
        f"- 量化检查通过率：`{pass_rate:.3f}`。",
        "- 本脚本输出的是 target mode 敏感性、bootstrap 和节点均值层面的闭环检查。",
        "- 病例级 Stage 1 readout 与同病例 task/CAM/feature 增量相关性已由 `export_stage1_case_readout_alignment.py` 补充完成，结果目录为 `results/stage1_case_readout_alignment_v1/`。",
        "- 写作时必须区分：病例级支持较强的早期差距/中部低谷/末端反超，以及只能写成群体均值曲线形态的后期加速/减速。",
        "",
        "## 4. 典型病例和反例候选",
        "",
    ]
    if not selected.empty:
        cols = [
            "case_role",
            "case_name",
            "total_evidence_score",
            "early_gap_score",
            "baseline_valley_score",
            "baseline_late_jump_score",
            "late_advantage_over_noskip",
        ]
        lines.append(dataframe_to_markdown(selected, cols))
    else:
        lines.append("未能生成病例筛选表。")
    lines.extend(
        [
            "",
            "## 5. 写作约束",
            "",
            "后续主文只能保留通过 `table_claim_support_checks.csv` 支撑的结论；未通过或只有单一证据支持的内容只能写成限制或待验证问题。",
        ]
    )
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    pred_node, pred_case, pred_dyn = load_result_dir(args.pred_dir, "pred_fg")
    gt_node, gt_case, gt_dyn = load_result_dir(args.gt_dir, "gt_fg")
    node_df = pd.concat([pred_node, gt_node], ignore_index=True)
    case_df = pd.concat([pred_case, gt_case], ignore_index=True)
    dyn_df = pd.concat([pred_dyn, gt_dyn], ignore_index=True)
    stage1_df = read_stage1(args.stage1_csv)

    patient_df = patient_metric_table(case_df)
    node_ci, transition_ci, patient_transition = patient_bootstrap_tables(patient_df, n_boot=args.bootstrap, seed=args.seed)
    alignment_df = transition_alignment(stage1_df=stage1_df, node_df=node_df)
    claim_df = claim_support_checks(node_df=node_df, stage1_df=stage1_df)

    scores = case_scores(patient_df, target_source="pred_fg")
    top_n = max(1, int(args.top_cases))
    supportive_scores = scores[scores["all_components_support"]].copy()
    visually_supportive_scores = supportive_scores[supportive_scores["visual_evidence_usable"]].copy()
    weak_scores = scores[~scores["all_components_support"]].copy()
    visually_weak_scores = weak_scores[weak_scores["gt_fg_pixels"].astype(float) >= 1000].copy()
    typical_pool = visually_supportive_scores if not visually_supportive_scores.empty else supportive_scores
    counter_pool = visually_weak_scores if not visually_weak_scores.empty else weak_scores
    typical = typical_pool.head(top_n).copy() if not typical_pool.empty else scores.head(top_n).copy()
    counter = counter_pool.sort_values("total_evidence_score", ascending=True).head(top_n).copy() if not counter_pool.empty else scores.tail(top_n).copy()
    typical["case_role"] = "typical_high_support"
    counter["case_role"] = "counterexample_low_support"
    selected = pd.concat([typical, counter], ignore_index=True) if not scores.empty else pd.DataFrame()

    node_df.to_csv(os.path.join(args.save_dir, "table_target_mode_sensitivity_node_summary.csv"), index=False)
    dyn_df.to_csv(os.path.join(args.save_dir, "table_target_mode_sensitivity_dynamics.csv"), index=False)
    node_ci.to_csv(os.path.join(args.save_dir, "table_patient_node_bootstrap_ci.csv"), index=False)
    transition_ci.to_csv(os.path.join(args.save_dir, "table_patient_transition_bootstrap_ci.csv"), index=False)
    patient_transition.to_csv(os.path.join(args.save_dir, "table_patient_transition_deltas.csv"), index=False)
    alignment_df.to_csv(os.path.join(args.save_dir, "table_transition_alignment_task_vs_cam_feature.csv"), index=False)
    claim_df.to_csv(os.path.join(args.save_dir, "table_claim_support_checks.csv"), index=False)
    selected.to_csv(os.path.join(args.save_dir, "table_typical_counterexample_cases.csv"), index=False)
    stage1_df.to_csv(os.path.join(args.save_dir, "table_stage1_task_dice_dynamics.csv"), index=False)

    plot_sensitivity(node_df=node_df, save_path=os.path.join(args.save_dir, "plot_target_mode_sensitivity_curves.png"))
    plot_transition_ci(transition_ci=transition_ci, save_path=os.path.join(args.save_dir, "plot_patient_transition_bootstrap_ci.png"))

    payload = {
        "pred_dir": args.pred_dir,
        "gt_dir": args.gt_dir,
        "stage1_csv": args.stage1_csv,
        "bootstrap": int(args.bootstrap),
        "claim_pass_rate": float(claim_df["passed"].mean()) if not claim_df.empty else None,
        "selected_cases": selected.to_dict(orient="records") if not selected.empty else [],
        "important_limitation": "This closure script is not the per-case Stage-1 readout exporter. Per-case task-vs-CAM/feature alignment is handled by export_stage1_case_readout_alignment.py and should be interpreted together with these closure checks.",
    }
    with open(os.path.join(args.save_dir, "summary_stage1_explanation_closure.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    write_markdown(args=args, claim_df=claim_df, selected=selected, out_path=os.path.join(args.save_dir, "58_stage1_explanation_closure_results_cn.md"))
    print("Saved Stage-1 explanation closure outputs to {}".format(args.save_dir))


if __name__ == "__main__":
    main()
