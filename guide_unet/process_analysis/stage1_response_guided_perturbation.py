from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


GUIDE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if GUIDE_ROOT not in sys.path:
    sys.path.insert(0, GUIDE_ROOT)

from process_analysis.export_stage1_case_readout_alignment import (  # noqa: E402
    NodeCAMProjector,
    train_projectors,
)
from process_analysis.stage1_explanation_utils import (  # noqa: E402
    FeatureStore,
    GradCAMStore,
    SpatialMaskPerturbationHook,
    bootstrap_mean_ci,
    build_eval_loader,
    compute_cam_variant_native,
    compute_prediction_metrics,
    ensure_dir,
    feature_response_from_activation,
    load_model,
    mean_region_dice,
    parse_csv_list,
    parse_model_specs,
    resolve_device,
    resolve_node_names,
    segmentation_target_score,
    set_seed,
    target_mask_from_logits,
    top_fraction_mask,
    tumor_prob_map,
    unpack_logits,
    write_json,
    write_manifest,
    write_run_config,
)
from analysis_common import select_indices, build_loader  # noqa: E402
from datasets.dataset_brats import BraTSDataset  # noqa: E402


RESPONSE_SOURCES = ("cam", "feature_l2")
MASK_TYPES = ("topk", "random", "low_response")

MODEL_LABEL = {
    "baseline": "有跳接 U-Net",
    "noskip_unet": "无跳接 U-Net",
}
RESPONSE_LABEL = {
    "cam": "CAM",
    "feature_l2": "原始特征响应",
}
MASK_LABEL = {
    "topk": "高响应遮挡",
    "random": "随机遮挡",
    "low_response": "低响应遮挡",
}


def configure_chinese_font() -> None:
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Noto Sans SC",
        "Source Han Sans CN",
    ]
    available = {font.name for font in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E4 response-guided faithfulness perturbation for Stage-1 explanation.")
    parser.add_argument("--model-spec", nargs="+", required=True, help="Format: model_name=/path/to/checkpoint")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--eval-split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--fit-split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--output-root", type=str, default="stage1_explanation_suite_20260517/results/e4_faithfulness_perturbation")
    parser.add_argument("--case-set-csv", type=str, default="")
    parser.add_argument("--case-names", type=str, default="")
    parser.add_argument("--nodes", type=str, default="down3,down4,up1,up3,up4")
    parser.add_argument("--response-sources", type=str, default="cam,feature_l2")
    parser.add_argument("--top-fractions", type=str, default="0.05,0.10,0.20")
    parser.add_argument("--gammas", type=str, default="0.5,1.0")
    parser.add_argument("--mask-types", type=str, default="topk,random,low_response")
    parser.add_argument("--target-mode", type=str, default="pred_fg", choices=["pred_fg", "gt_fg"])
    parser.add_argument("--num-cases", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--random-repeats", type=int, default=5)
    parser.add_argument("--foreground-only", action="store_true")
    parser.add_argument("--patient-balanced", action="store_true")
    parser.add_argument("--fit-readout-projectors", action="store_true")
    parser.add_argument("--num-fit-samples", type=int, default=128)
    parser.add_argument("--fit-epochs", type=int, default=2)
    parser.add_argument("--fit-lr", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def parse_float_list(text: str) -> List[float]:
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def native_response_maps(store: GradCAMStore, node_name: str, sources: Sequence[str]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    if "cam" in sources:
        out["cam"] = compute_cam_variant_native(store=store, node_name=node_name, variant="gradcam")
    if "feature_l2" in sources:
        out["feature_l2"] = feature_response_from_activation(store.activations[node_name], aggregation="l2")
    return out


def batch_masks_from_response(response: torch.Tensor, fraction: float, mask_type: str, rng: np.random.Generator) -> torch.Tensor:
    response_np = response.detach().cpu().numpy()
    masks = []
    for idx in range(response_np.shape[0]):
        arr = response_np[idx]
        if mask_type == "topk":
            mask = top_fraction_mask(arr, fraction=fraction, largest=True)
        elif mask_type == "low_response":
            mask = top_fraction_mask(arr, fraction=fraction, largest=False)
        elif mask_type == "random":
            flat_size = int(arr.size)
            k = max(1, int(round(float(fraction) * flat_size)))
            chosen = rng.choice(flat_size, size=k, replace=False)
            flat = np.zeros(flat_size, dtype=bool)
            flat[chosen] = True
            mask = flat.reshape(arr.shape)
        else:
            raise ValueError(mask_type)
        masks.append(torch.from_numpy(mask.astype(np.float32)))
    return torch.stack(masks, dim=0)


def build_fit_loader(data_dir: str, split: str, num_fit_samples: int, foreground_only: bool, seed: int, batch_size: int, num_workers: int) -> DataLoader:
    dataset = BraTSDataset(base_dir=data_dir, split=split)
    indices = select_indices(
        dataset=dataset,
        num_samples=int(num_fit_samples),
        foreground_only=bool(foreground_only),
        seed=int(seed),
        patient_balanced=False,
    )
    return build_loader(dataset=dataset, indices=indices, batch_size=int(batch_size), shuffle=True, num_workers=int(num_workers))


def fit_projectors_for_model(model, node_names: Sequence[str], args: argparse.Namespace, device: torch.device) -> Optional[Dict[str, NodeCAMProjector]]:
    if not bool(args.fit_readout_projectors):
        return None
    fit_loader = build_fit_loader(
        data_dir=args.data_dir,
        split=args.fit_split,
        num_fit_samples=int(args.num_fit_samples),
        foreground_only=bool(args.foreground_only),
        seed=int(args.seed),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
    )
    collector = FeatureStore(model)
    warmup = next(iter(fit_loader))
    with torch.no_grad():
        collector.clear()
        _ = model(warmup["image"].to(device))
        node_map = collector.node_map()
        final_feat = node_map["up4"]
    projectors = {
        node_name: NodeCAMProjector(
            in_channels=int(node_map[node_name].shape[1]),
            target_channels=int(final_feat.shape[1]),
            target_size=tuple(int(v) for v in final_feat.shape[2:]),
        ).to(device)
        for node_name in node_names
    }
    train_projectors(
        model=model,
        collector=collector,
        projectors=projectors,
        fit_loader=fit_loader,
        device=device,
        fit_epochs=int(args.fit_epochs),
        fit_lr=float(args.fit_lr),
    )
    collector.remove()
    for projector in projectors.values():
        projector.eval()
    return projectors


def node_readout_dice(projector: NodeCAMProjector, model, activation: torch.Tensor, labels: torch.Tensor) -> List[float]:
    with torch.no_grad():
        feat = projector(activation.detach())
        logits = model.outc(feat)
        pred = torch.argmax(logits, dim=1).detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        return [float(mean_region_dice(pred[idx], labels_np[idx])) for idx in range(pred.shape[0])]


def analyze_model(spec, loader, args: argparse.Namespace, device: torch.device, node_names: Sequence[str], sources: Sequence[str], mask_types: Sequence[str], top_fractions: Sequence[float], gammas: Sequence[float]) -> pd.DataFrame:
    model = load_model(model_name=spec.name, ckpt_path=spec.ckpt_path, device=device)
    projectors = fit_projectors_for_model(model, node_names=node_names, args=args, device=device)
    store = GradCAMStore(model=model, node_names=node_names)
    rows: List[Dict] = []
    rng = np.random.default_rng(int(args.seed))

    for batch in tqdm(loader, desc="E4 perturb {}".format(spec.name)):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        case_names = list(batch["case_name"])

        model.zero_grad(set_to_none=True)
        store.clear()
        logits = unpack_logits(model(images))
        score = segmentation_target_score(logits=logits, labels=labels, mode=args.target_mode)
        score.backward()
        base_pred = torch.argmax(logits.detach(), dim=1)
        base_metrics = compute_prediction_metrics(logits, labels)
        base_final_prob = tumor_prob_map(logits).detach().cpu().numpy()
        _ = target_mask_from_logits(logits=logits, labels=labels, mode=args.target_mode)

        for node_name in node_names:
            response_maps = native_response_maps(store=store, node_name=node_name, sources=sources)
            original_activation = store.activations[node_name].detach()
            base_node_readout = None
            if projectors is not None and node_name in projectors:
                base_node_readout = node_readout_dice(projectors[node_name], model, original_activation, labels)

            for source in sources:
                response = response_maps[source]
                for fraction in top_fractions:
                    for mask_type in mask_types:
                        repeat_count = int(args.random_repeats) if mask_type == "random" else 1
                        for repeat_idx in range(repeat_count):
                            mask = batch_masks_from_response(response, fraction=float(fraction), mask_type=mask_type, rng=rng)
                            for gamma in gammas:
                                hook = SpatialMaskPerturbationHook(mask=mask, gamma=float(gamma))
                                handle = getattr(model, node_name).register_forward_hook(hook)
                                with torch.no_grad():
                                    perturbed_logits = unpack_logits(model(images))
                                handle.remove()
                                pert_metrics = compute_prediction_metrics(perturbed_logits, labels, reference_pred=base_pred)
                                pert_node_readout = None
                                if projectors is not None and hook.last_output is not None and node_name in projectors:
                                    pert_node_readout = node_readout_dice(projectors[node_name], model, hook.last_output, labels)
                                for local_idx, case_name in enumerate(case_names):
                                    base = base_metrics[local_idx]
                                    pert = pert_metrics[local_idx]
                                    node_drop = np.nan
                                    if base_node_readout is not None and pert_node_readout is not None:
                                        node_drop = float(base_node_readout[local_idx] - pert_node_readout[local_idx])
                                    rows.append(
                                        {
                                            "model": spec.name,
                                            "case_name": str(case_name),
                                            "patient_id": str(case_name).rsplit("_", 1)[0],
                                            "target_mode": str(args.target_mode),
                                            "node": node_name,
                                            "node_order": int(node_names.index(node_name)),
                                            "response_source": source,
                                            "mask_type": mask_type,
                                            "top_fraction": float(fraction),
                                            "gamma": float(gamma),
                                            "random_repeat": int(repeat_idx) if mask_type == "random" else -1,
                                            "mask_area_fraction": float(mask[local_idx].float().mean().item()),
                                            "baseline_final_dice": float(base["mean_dice"]),
                                            "perturbed_final_dice": float(pert["mean_dice"]),
                                            "final_dice_drop": float(base["mean_dice"] - pert["mean_dice"]),
                                            "baseline_gt_fg_prob": float(base["gt_fg_prob"]),
                                            "perturbed_gt_fg_prob": float(pert["gt_fg_prob"]),
                                            "gt_fg_prob_drop": float(base["gt_fg_prob"] - pert["gt_fg_prob"]),
                                            "baseline_final_fg_prob": float(base["pred_fg_prob"]),
                                            "perturbed_final_fg_prob": float(pert["pred_fg_prob"]),
                                            "final_fg_prob_drop": float(base["pred_fg_prob"] - pert["pred_fg_prob"]),
                                            "baseline_whole_tumor_prob_mean": float(base["whole_tumor_prob_mean"]),
                                            "perturbed_whole_tumor_prob_mean": float(pert["whole_tumor_prob_mean"]),
                                            "whole_tumor_prob_mean_drop": float(base["whole_tumor_prob_mean"] - pert["whole_tumor_prob_mean"]),
                                            "node_readout_drop": node_drop,
                                            "gt_fg_pixels": int(base["gt_fg_pixels"]),
                                            "pred_fg_pixels": int(base["pred_fg_pixels"]),
                                            "seed": int(args.seed),
                                        }
                                    )

    store.remove()
    return pd.DataFrame(rows)


def summarize(case_df: pd.DataFrame, bootstrap: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_cols = ["model", "target_mode", "node", "node_order", "response_source", "mask_type", "top_fraction", "gamma"]
    metrics = ("final_fg_prob_drop", "final_dice_drop", "gt_fg_prob_drop", "whole_tumor_prob_mean_drop", "node_readout_drop")
    rows = []
    ci_rows = []
    for key, group in case_df.groupby(group_cols, sort=False):
        payload = dict(zip(group_cols, key))
        payload["n_cases"] = int(group["case_name"].nunique())
        payload["n_slices"] = int(len(group))
        payload["random_repeats"] = int(max(group["random_repeat"].max(), 0) + 1) if str(payload["mask_type"]) == "random" else 1
        payload["seed"] = int(seed)
        for metric in metrics:
            payload[metric] = float(group[metric].mean()) if metric in group.columns else np.nan
            lo, hi = bootstrap_mean_ci(group[metric].to_numpy(), n_boot=int(bootstrap), seed=int(seed)) if metric in group.columns else (np.nan, np.nan)
            ci_rows.append({**dict(zip(group_cols, key)), "metric": metric, "mean": payload[metric], "ci_low": lo, "ci_high": hi, "n_cases": payload["n_cases"], "n_slices": payload["n_slices"]})
        rows.append(payload)
    return pd.DataFrame(rows).sort_values(["model", "node_order", "response_source", "top_fraction", "gamma", "mask_type"]), pd.DataFrame(ci_rows)


def adjusted_drop_table(case_df: pd.DataFrame) -> pd.DataFrame:
    metrics = ("final_fg_prob_drop", "final_dice_drop", "gt_fg_prob_drop", "whole_tumor_prob_mean_drop", "node_readout_drop")
    keys = ["model", "case_name", "target_mode", "node", "node_order", "response_source", "top_fraction", "gamma"]
    top = case_df[case_df["mask_type"] == "topk"].copy()
    random_mean = (
        case_df[case_df["mask_type"] == "random"]
        .groupby(keys, as_index=False)[list(metrics)]
        .mean()
        .rename(columns={metric: "random_{}".format(metric) for metric in metrics})
    )
    low = case_df[case_df["mask_type"] == "low_response"].copy().rename(columns={metric: "low_{}".format(metric) for metric in metrics})
    out = top.merge(random_mean, on=keys, how="left")
    out = out.merge(low[keys + ["low_{}".format(metric) for metric in metrics]], on=keys, how="left")
    for metric in metrics:
        out["random_adjusted_{}".format(metric)] = out[metric] - out["random_{}".format(metric)]
        out["low_adjusted_{}".format(metric)] = out[metric] - out["low_{}".format(metric)]
    return out


def summarize_adjusted(adjusted: pd.DataFrame, bootstrap: int, seed: int) -> pd.DataFrame:
    if adjusted.empty:
        return pd.DataFrame()
    metrics = [col for col in adjusted.columns if col.startswith("random_adjusted_") or col.startswith("low_adjusted_")]
    group_cols = ["model", "target_mode", "node", "node_order", "response_source", "top_fraction", "gamma"]
    rows = []
    for key, group in adjusted.groupby(group_cols, sort=False):
        payload = dict(zip(group_cols, key))
        payload["n_cases"] = int(group["case_name"].nunique())
        for metric in metrics:
            payload[metric] = float(group[metric].mean())
            lo, hi = bootstrap_mean_ci(group[metric].to_numpy(), n_boot=int(bootstrap), seed=int(seed))
            payload["{}_ci_low".format(metric)] = lo
            payload["{}_ci_high".format(metric)] = hi
        rows.append(payload)
    return pd.DataFrame(rows).sort_values(["model", "node_order", "response_source", "top_fraction", "gamma"])


def plot_drop(summary: pd.DataFrame, out_path: str) -> None:
    if summary.empty:
        return
    configure_chinese_font()
    sub = summary[(summary["top_fraction"] == summary["top_fraction"].min()) | (summary["top_fraction"] == 0.10)]
    metric = "final_fg_prob_drop"
    models = list(sub["model"].drop_duplicates())
    fig, axes = plt.subplots(len(models), 1, figsize=(11, 4.2 * max(1, len(models))), dpi=160, squeeze=False)
    for row_idx, model in enumerate(models):
        ax = axes[row_idx, 0]
        msub = sub[(sub["model"] == model) & (sub["gamma"] == sub["gamma"].max())]
        for (source, mask_type), group in msub.groupby(["response_source", "mask_type"], sort=False):
            group = group.sort_values("node_order")
            label = "{} | {}".format(RESPONSE_LABEL.get(source, source), MASK_LABEL.get(mask_type, mask_type))
            ax.plot(group["node"], group[metric], marker="o", linewidth=1.7, label=label)
        ax.set_title("{} | 响应引导遮挡后的输出下降".format(MODEL_LABEL.get(model, model)))
        ax.set_ylabel("最终前景概率下降")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[-1, 0].set_xlabel("Stage 1 节点")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_adjusted(adjusted_summary: pd.DataFrame, out_path: str) -> None:
    if adjusted_summary.empty:
        return
    configure_chinese_font()
    metric = "random_adjusted_final_fg_prob_drop"
    models = list(adjusted_summary["model"].drop_duplicates())
    fig, axes = plt.subplots(len(models), 1, figsize=(11, 4.2 * max(1, len(models))), dpi=160, squeeze=False)
    for row_idx, model in enumerate(models):
        ax = axes[row_idx, 0]
        msub = adjusted_summary[(adjusted_summary["model"] == model) & (adjusted_summary["gamma"] == adjusted_summary["gamma"].max())]
        for source, group in msub.groupby("response_source", sort=False):
            group = group.sort_values("node_order")
            ax.plot(group["node"], group[metric], marker="o", linewidth=1.8, label=RESPONSE_LABEL.get(source, source))
        ax.axhline(0.0, color="#777777", linestyle="--", linewidth=0.8)
        ax.set_title("{} | 高响应遮挡相对随机遮挡的额外输出下降".format(MODEL_LABEL.get(model, model)))
        ax.set_ylabel("校正后前景概率下降")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[-1, 0].set_xlabel("Stage 1 节点")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.num_cases = min(int(args.num_cases), 2)
        args.batch_size = 1
        args.bootstrap = min(int(args.bootstrap), 50)
        args.random_repeats = min(int(args.random_repeats), 2)
        args.top_fractions = "0.10"
        args.gammas = "0.5"
        args.num_fit_samples = min(int(args.num_fit_samples), 8)
        args.fit_epochs = min(int(args.fit_epochs), 1)
    ensure_dir(args.output_root)
    set_seed(int(args.seed))
    specs = parse_model_specs(args.model_spec)
    node_names = resolve_node_names(args.nodes)
    sources = parse_csv_list(args.response_sources)
    mask_types = parse_csv_list(args.mask_types)
    invalid_sources = sorted(set(sources) - set(RESPONSE_SOURCES))
    invalid_masks = sorted(set(mask_types) - set(MASK_TYPES))
    if invalid_sources:
        raise ValueError("Unsupported response sources: {}".format(invalid_sources))
    if invalid_masks:
        raise ValueError("Unsupported mask types: {}".format(invalid_masks))
    top_fractions = parse_float_list(args.top_fractions)
    gammas = parse_float_list(args.gammas)
    device = resolve_device(args.device)
    _, indices, case_names, loader = build_eval_loader(
        data_dir=args.data_dir,
        split=args.eval_split,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        num_cases=int(args.num_cases),
        seed=int(args.seed),
        foreground_only=bool(args.foreground_only),
        case_set_csv=args.case_set_csv,
        case_names=parse_csv_list(args.case_names),
        patient_balanced=bool(args.patient_balanced),
    )

    frames = []
    for spec in specs:
        frames.append(analyze_model(spec, loader, args, device, node_names, sources, mask_types, top_fractions, gammas))
    case_df = pd.concat(frames, ignore_index=True)
    summary, ci = summarize(case_df, bootstrap=int(args.bootstrap), seed=int(args.seed))
    adjusted = adjusted_drop_table(case_df)
    adjusted_summary = summarize_adjusted(adjusted, bootstrap=int(args.bootstrap), seed=int(args.seed))

    case_df.to_csv(os.path.join(args.output_root, "table_perturbation_case_level.csv"), index=False)
    summary.to_csv(os.path.join(args.output_root, "table_perturbation_summary_by_node.csv"), index=False)
    ci.to_csv(os.path.join(args.output_root, "table_perturbation_bootstrap_ci.csv"), index=False)
    adjusted.to_csv(os.path.join(args.output_root, "table_perturbation_random_adjusted.csv"), index=False)
    adjusted_summary.to_csv(os.path.join(args.output_root, "table_perturbation_adjusted_summary.csv"), index=False)
    plot_drop(summary, os.path.join(args.output_root, "plot_perturbation_drop_by_node.png"))
    plot_adjusted(adjusted_summary, os.path.join(args.output_root, "plot_perturbation_late_transition_focus.png"))

    write_run_config(args, args.output_root, extra={"device": str(device), "selected_indices": [int(v) for v in indices]})
    write_manifest(
        args.output_root,
        [
            {"output": name, "path": os.path.join(args.output_root, name)}
            for name in (
                "table_perturbation_case_level.csv",
                "table_perturbation_summary_by_node.csv",
                "table_perturbation_bootstrap_ci.csv",
                "table_perturbation_random_adjusted.csv",
                "table_perturbation_adjusted_summary.csv",
                "plot_perturbation_drop_by_node.png",
                "plot_perturbation_late_transition_focus.png",
            )
        ],
    )
    write_json(
        {
            "purpose": "E4 response-guided faithfulness perturbation",
            "num_cases": int(len(case_names)),
            "models": [spec.name for spec in specs],
            "nodes": list(node_names),
            "response_sources": list(sources),
            "mask_types": list(mask_types),
            "top_fractions": list(top_fractions),
            "gammas": list(gammas),
            "fit_readout_projectors": bool(args.fit_readout_projectors),
            "guardrail": "Perturbation supports task relevance of response regions; it is not a complete proof of skip causality.",
        },
        os.path.join(args.output_root, "summary_response_guided_perturbation.json"),
    )
    print("Saved E4 response-guided perturbation outputs to {}".format(args.output_root))
    print("Device:", device)
    print("Cases:", len(case_names))


if __name__ == "__main__":
    main()
