from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


GUIDE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if GUIDE_ROOT not in sys.path:
    sys.path.insert(0, GUIDE_ROOT)

from process_analysis.stage1_explanation_utils import (  # noqa: E402
    FeatureStore,
    NETWORK_ORDER,
    TRANSITIONS,
    add_node_order,
    bootstrap_corr_ci,
    bootstrap_mean_ci,
    build_eval_loader,
    corr_value,
    ensure_dir,
    feature_response_from_activation,
    load_model,
    map_metric_row,
    mask_jaccard,
    parse_csv_list,
    parse_model_specs,
    resolve_device,
    resolve_node_names,
    safe_corr,
    set_seed,
    top_fraction_mask,
    transition_delta_table,
    tumor_prob_map,
    unpack_logits,
    write_json,
    write_manifest,
    write_run_config,
)


AGGREGATIONS = ("l2", "mean_abs", "max_abs", "topk_channel", "positive_only")
ALIGNMENT_METRICS = (
    "final_prob_pearson",
    "gt_mass_fraction",
    "gt_fg_bg_contrast",
    "entropy_norm",
    "top10_gt_fraction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E5 raw feature response aggregation robustness for Stage-1 explanation.")
    parser.add_argument("--model-spec", nargs="+", required=True, help="Format: model_name=/path/to/checkpoint")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-root", type=str, default="stage1_explanation_suite_20260517/results/e5_feature_response_robustness")
    parser.add_argument("--case-set-csv", type=str, default="")
    parser.add_argument("--case-names", type=str, default="")
    parser.add_argument("--stage1-readout-csv", type=str, default="results/stage1_case_readout_alignment_n512/table_stage1_case_readout.csv")
    parser.add_argument("--nodes", type=str, default="all")
    parser.add_argument("--aggregations", type=str, default=",".join(AGGREGATIONS))
    parser.add_argument("--num-cases", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--topk-channel-fraction", type=float, default=0.10)
    parser.add_argument("--foreground-only", action="store_true")
    parser.add_argument("--patient-balanced", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _metric_cols(prefix: str = "") -> List[str]:
    return ["{}{}".format(prefix, name) for name in ALIGNMENT_METRICS]


def collect_model_rows(spec, loader, device: torch.device, node_names: Sequence[str], aggregations: Sequence[str], topk_channel_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = load_model(model_name=spec.name, ckpt_path=spec.ckpt_path, device=device)
    store = FeatureStore(model)
    rows: List[Dict] = []
    stability_rows: List[Dict] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="E5 feature robustness {}".format(spec.name)):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            case_names = list(batch["case_name"])
            store.clear()
            logits = unpack_logits(model(images))
            node_map = store.node_map()
            pred_np = torch.argmax(logits, dim=1).detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            final_prob_np = tumor_prob_map(logits).detach().cpu().numpy()

            response_maps: Dict[str, Dict[str, torch.Tensor]] = {}
            for node_name in node_names:
                response_maps[node_name] = {}
                for aggregation in aggregations:
                    response = feature_response_from_activation(
                        node_map[node_name],
                        aggregation=aggregation,
                        topk_channel_fraction=float(topk_channel_fraction),
                    )
                    if tuple(response.shape[-2:]) != tuple(logits.shape[-2:]):
                        response = torch.nn.functional.interpolate(
                            response.unsqueeze(1),
                            size=tuple(logits.shape[-2:]),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(1)
                    response_maps[node_name][aggregation] = response.detach().cpu()

            for local_idx, case_name in enumerate(case_names):
                for node_order, node_name in enumerate(node_names):
                    l2_map = response_maps[node_name].get("l2")
                    l2_np = l2_map[local_idx].numpy() if l2_map is not None else None
                    l2_top = top_fraction_mask(l2_np, fraction=0.10, largest=True) if l2_np is not None else None
                    for aggregation in aggregations:
                        response = response_maps[node_name][aggregation][local_idx].numpy()
                        row = {
                            "model": spec.name,
                            "case_name": str(case_name),
                            "patient_id": str(case_name).rsplit("_", 1)[0],
                            "node": node_name,
                            "node_order": int(node_order),
                            "aggregation": aggregation,
                            "seed": np.nan,
                            "n_slices": 1,
                        }
                        row.update(map_metric_row(response, final_prob_np[local_idx], labels_np[local_idx], pred_np[local_idx], prefix=""))
                        rows.append(row)

                        if l2_np is not None and aggregation != "l2":
                            top = top_fraction_mask(response, fraction=0.10, largest=True)
                            stability_rows.append(
                                {
                                    "model": spec.name,
                                    "case_name": str(case_name),
                                    "node": node_name,
                                    "node_order": int(node_order),
                                    "aggregation": aggregation,
                                    "spearman_vs_l2": float(pd.Series(response.reshape(-1)).corr(pd.Series(l2_np.reshape(-1)), method="spearman")),
                                    "pearson_vs_l2": float(safe_corr(response, l2_np)),
                                    "top10_jaccard_vs_l2": float(mask_jaccard(top, l2_top)),
                                }
                            )

    store.remove()
    return pd.DataFrame(rows), pd.DataFrame(stability_rows)


def summarize(per_case: pd.DataFrame, bootstrap: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_cols = _metric_cols()
    group_cols = ["model", "aggregation", "node", "node_order"]
    rows = []
    ci_rows = []
    for key, group in per_case.groupby(group_cols, sort=False):
        payload = dict(zip(group_cols, key))
        payload["n_cases"] = int(group["case_name"].nunique())
        payload["n_slices"] = int(len(group))
        payload["seed"] = int(seed)
        for metric in metric_cols:
            payload[metric] = float(group[metric].mean())
            lo, hi = bootstrap_mean_ci(group[metric].to_numpy(), n_boot=int(bootstrap), seed=int(seed))
            ci_rows.append({**dict(zip(group_cols, key)), "metric": metric, "mean": payload[metric], "ci_low": lo, "ci_high": hi, "n_cases": payload["n_cases"], "n_slices": payload["n_slices"]})
        rows.append(payload)
    return pd.DataFrame(rows).sort_values(["model", "aggregation", "node_order"]), pd.DataFrame(ci_rows)


def summarize_stability(stability: pd.DataFrame, bootstrap: int, seed: int) -> pd.DataFrame:
    if stability.empty:
        return pd.DataFrame()
    rows = []
    for key, group in stability.groupby(["model", "aggregation", "node", "node_order"], sort=False):
        payload = dict(zip(["model", "aggregation", "node", "node_order"], key))
        payload["n_cases"] = int(group["case_name"].nunique())
        for metric in ("spearman_vs_l2", "pearson_vs_l2", "top10_jaccard_vs_l2"):
            payload[metric] = float(group[metric].mean())
            lo, hi = bootstrap_mean_ci(group[metric].to_numpy(), n_boot=int(bootstrap), seed=int(seed))
            payload["{}_ci_low".format(metric)] = lo
            payload["{}_ci_high".format(metric)] = hi
        rows.append(payload)
    return pd.DataFrame(rows).sort_values(["model", "aggregation", "node_order"])


def build_delta_correlations(per_case: pd.DataFrame, stage1_readout_csv: str, bootstrap: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not stage1_readout_csv or not os.path.isfile(stage1_readout_csv):
        return pd.DataFrame(), pd.DataFrame()
    readout = pd.read_csv(stage1_readout_csv)
    needed = {"model", "case_name", "node", "task_dice_gt_case"}
    if not needed.issubset(readout.columns):
        return pd.DataFrame(), pd.DataFrame()
    readout = readout[["model", "case_name", "node", "task_dice_gt_case"]].copy()
    merged = pd.merge(per_case, readout, on=["model", "case_name", "node"], how="inner")
    value_cols = ["task_dice_gt_case", *_metric_cols()]
    delta = transition_delta_table(
        merged,
        value_cols=value_cols,
        group_cols=["model", "case_name", "patient_id", "aggregation"],
    )
    rows = []
    for (model, aggregation), group in delta.groupby(["model", "aggregation"], sort=False):
        for metric in _metric_cols():
            col = "{}_delta".format(metric)
            pearson = corr_value(group["task_dice_gt_case_delta"], group[col], method="pearson")
            spearman = corr_value(group["task_dice_gt_case_delta"], group[col], method="spearman")
            lo, hi = bootstrap_corr_ci(group["task_dice_gt_case_delta"], group[col], n_boot=int(bootstrap), seed=int(seed), method="pearson")
            rows.append(
                {
                    "model": model,
                    "aggregation": aggregation,
                    "metric": metric,
                    "n_case_transition_pairs": int(len(group)),
                    "pearson_task_delta_vs_metric_delta": pearson,
                    "pearson_ci95_low": lo,
                    "pearson_ci95_high": hi,
                    "spearman_task_delta_vs_metric_delta": spearman,
                }
            )
    return delta, pd.DataFrame(rows)


def plot_curves(summary: pd.DataFrame, out_path: str) -> None:
    plot_metrics = ("gt_mass_fraction", "final_prob_pearson", "gt_fg_bg_contrast", "entropy_norm")
    models = list(summary["model"].drop_duplicates())
    fig, axes = plt.subplots(len(plot_metrics), len(models), figsize=(6 * max(1, len(models)), 3.4 * len(plot_metrics)), dpi=160, squeeze=False)
    for col_idx, model in enumerate(models):
        sub_model = summary[summary["model"] == model]
        for row_idx, metric in enumerate(plot_metrics):
            ax = axes[row_idx, col_idx]
            for aggregation, group in sub_model.groupby("aggregation", sort=False):
                group = group.sort_values("node_order")
                ax.plot(group["node"], group[metric], marker="o", linewidth=1.7, label=aggregation)
            ax.set_title("{} | {}".format(model, metric))
            ax.grid(alpha=0.25)
            ax.tick_params(axis="x", rotation=35)
            if row_idx == 0:
                ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_delta_corr(corr: pd.DataFrame, out_path: str) -> None:
    if corr.empty:
        return
    pivot = corr.pivot_table(index=["model", "aggregation"], columns="metric", values="pearson_task_delta_vs_metric_delta")
    fig, ax = plt.subplots(figsize=(11, max(4, 0.36 * len(pivot))), dpi=160)
    im = ax.imshow(pivot.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(["{} | {}".format(a, b) for a, b in pivot.index], fontsize=8)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(list(pivot.columns), rotation=45, ha="right", fontsize=8)
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Delta task_dice_gt vs response metric delta")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.num_cases = min(int(args.num_cases), 4)
        args.batch_size = min(int(args.batch_size), 2)
        args.bootstrap = min(int(args.bootstrap), 50)
    ensure_dir(args.output_root)
    set_seed(int(args.seed))
    specs = parse_model_specs(args.model_spec)
    node_names = resolve_node_names(args.nodes)
    aggregations = parse_csv_list(args.aggregations)
    invalid = sorted(set(aggregations) - set(AGGREGATIONS))
    if invalid:
        raise ValueError("Unsupported aggregations: {}".format(invalid))
    device = resolve_device(args.device)
    _, indices, case_names, loader = build_eval_loader(
        data_dir=args.data_dir,
        split=args.split,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        num_cases=int(args.num_cases),
        seed=int(args.seed),
        foreground_only=bool(args.foreground_only),
        case_set_csv=args.case_set_csv,
        case_names=parse_csv_list(args.case_names),
        patient_balanced=bool(args.patient_balanced),
    )

    frames: List[pd.DataFrame] = []
    stability_frames: List[pd.DataFrame] = []
    for spec in specs:
        frame, stability_frame = collect_model_rows(
            spec=spec,
            loader=loader,
            device=device,
            node_names=node_names,
            aggregations=aggregations,
            topk_channel_fraction=float(args.topk_channel_fraction),
        )
        frames.append(frame)
        stability_frames.append(stability_frame)
    per_case = pd.concat(frames, ignore_index=True)
    per_case["seed"] = int(args.seed)
    stability = pd.concat(stability_frames, ignore_index=True) if stability_frames else pd.DataFrame()
    summary, ci = summarize(per_case, bootstrap=int(args.bootstrap), seed=int(args.seed))
    stability_summary = summarize_stability(stability, bootstrap=int(args.bootstrap), seed=int(args.seed))
    delta, corr = build_delta_correlations(per_case, args.stage1_readout_csv, bootstrap=int(args.bootstrap), seed=int(args.seed))

    per_case.to_csv(os.path.join(args.output_root, "table_response_aggregation_case_level.csv"), index=False)
    summary.to_csv(os.path.join(args.output_root, "table_response_aggregation_summary.csv"), index=False)
    ci.to_csv(os.path.join(args.output_root, "table_response_aggregation_bootstrap_ci.csv"), index=False)
    stability.to_csv(os.path.join(args.output_root, "table_response_aggregation_rank_stability_case_level.csv"), index=False)
    stability_summary.to_csv(os.path.join(args.output_root, "table_response_aggregation_rank_stability.csv"), index=False)
    delta.to_csv(os.path.join(args.output_root, "table_response_aggregation_transition_delta.csv"), index=False)
    corr.to_csv(os.path.join(args.output_root, "table_response_aggregation_delta_correlation.csv"), index=False)
    plot_curves(summary, os.path.join(args.output_root, "plot_aggregation_robustness_curves.png"))
    plot_delta_corr(corr, os.path.join(args.output_root, "plot_aggregation_delta_correlation.png"))

    write_run_config(args, args.output_root, extra={"device": str(device), "selected_indices": [int(v) for v in indices]})
    write_manifest(
        args.output_root,
        [
            {
                "output": name,
                "path": os.path.join(args.output_root, name),
            }
            for name in (
                "table_response_aggregation_case_level.csv",
                "table_response_aggregation_summary.csv",
                "table_response_aggregation_bootstrap_ci.csv",
                "table_response_aggregation_rank_stability.csv",
                "table_response_aggregation_delta_correlation.csv",
                "plot_aggregation_robustness_curves.png",
                "plot_aggregation_delta_correlation.png",
            )
        ],
    )
    write_json(
        {
            "purpose": "E5 raw feature response aggregation robustness",
            "num_cases": int(len(case_names)),
            "models": [spec.name for spec in specs],
            "nodes": list(node_names),
            "aggregations": list(aggregations),
            "guardrail": "This experiment tests node-response readout robustness; it does not introduce model-improvement or skip-causality claims.",
        },
        os.path.join(args.output_root, "summary_feature_response_robustness.json"),
    )
    print("Saved E5 feature response robustness outputs to {}".format(args.output_root))
    print("Device:", device)
    print("Cases:", len(case_names))


if __name__ == "__main__":
    main()
