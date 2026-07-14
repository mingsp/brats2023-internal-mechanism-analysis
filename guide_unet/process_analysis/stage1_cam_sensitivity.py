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
    GradCAMStore,
    bootstrap_mean_ci,
    build_eval_loader,
    compute_cam_variant_native,
    ensure_dir,
    load_model,
    map_metric_row,
    mask_jaccard,
    parse_csv_list,
    parse_model_specs,
    resolve_device,
    resolve_node_names,
    safe_corr,
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


CAM_VARIANTS = ("gradcam", "hirescam")
TARGET_MODES = ("pred_fg", "gt_fg")
METRICS = (
    "final_prob_pearson",
    "gt_mass_fraction",
    "gt_fg_bg_contrast",
    "entropy_norm",
    "top10_gt_fraction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E6 CAM target/variant sensitivity for Stage-1 explanation.")
    parser.add_argument("--model-spec", nargs="+", required=True, help="Format: model_name=/path/to/checkpoint")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-root", type=str, default="stage1_explanation_suite_20260517/results/e6_cam_sensitivity")
    parser.add_argument("--case-set-csv", type=str, default="")
    parser.add_argument("--case-names", type=str, default="")
    parser.add_argument("--nodes", type=str, default="down4,up1,up3,up4")
    parser.add_argument("--target-modes", type=str, default="pred_fg,gt_fg")
    parser.add_argument("--cam-variants", type=str, default="gradcam,hirescam")
    parser.add_argument("--num-cases", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--foreground-only", action="store_true")
    parser.add_argument("--patient-balanced", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def collect_model_rows(spec, loader, device: torch.device, node_names: Sequence[str], target_modes: Sequence[str], variants: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = load_model(model_name=spec.name, ckpt_path=spec.ckpt_path, device=device)
    store = GradCAMStore(model=model, node_names=node_names)
    rows: List[Dict] = []
    sensitivity_rows: List[Dict] = []
    reference_maps: Dict[tuple, np.ndarray] = {}

    for batch in tqdm(loader, desc="E6 CAM sensitivity {}".format(spec.name)):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        case_names = list(batch["case_name"])
        for target_mode in target_modes:
            model.zero_grad(set_to_none=True)
            store.clear()
            logits = unpack_logits(model(images))
            score = segmentation_target_score(logits=logits, labels=labels, mode=target_mode)
            score.backward()
            pred_np = torch.argmax(logits.detach(), dim=1).cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            final_prob_np = tumor_prob_map(logits).detach().cpu().numpy()
            _ = target_mask_from_logits(logits=logits, labels=labels, mode=target_mode)

            for node_name in node_names:
                for variant in variants:
                    native = compute_cam_variant_native(store=store, node_name=node_name, variant=variant)
                    if tuple(native.shape[-2:]) != tuple(logits.shape[-2:]):
                        cam = torch.nn.functional.interpolate(
                            native.unsqueeze(1),
                            size=tuple(logits.shape[-2:]),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(1)
                    else:
                        cam = native
                    cam_np = cam.detach().cpu().numpy()
                    for local_idx, case_name in enumerate(case_names):
                        response = cam_np[local_idx]
                        key = (spec.name, str(case_name), node_name, target_mode, variant)
                        reference_maps[key] = response
                        row = {
                            "model": spec.name,
                            "case_name": str(case_name),
                            "patient_id": str(case_name).rsplit("_", 1)[0],
                            "target_mode": target_mode,
                            "cam_variant": variant,
                            "node": node_name,
                            "node_order": int(node_names.index(node_name)),
                            "seed": np.nan,
                            "n_slices": 1,
                        }
                        row.update(map_metric_row(response, final_prob_np[local_idx], labels_np[local_idx], pred_np[local_idx], prefix=""))
                        rows.append(row)

    store.remove()

    ref_target = "pred_fg" if "pred_fg" in target_modes else target_modes[0]
    ref_variant = "gradcam" if "gradcam" in variants else variants[0]
    for key, response in reference_maps.items():
        model_name, case_name, node_name, target_mode, variant = key
        if target_mode == ref_target and variant == ref_variant:
            continue
        ref_key = (model_name, case_name, node_name, ref_target, ref_variant)
        if ref_key not in reference_maps:
            continue
        ref = reference_maps[ref_key]
        sensitivity_rows.append(
            {
                "model": model_name,
                "case_name": case_name,
                "node": node_name,
                "target_mode": target_mode,
                "cam_variant": variant,
                "reference_target_mode": ref_target,
                "reference_cam_variant": ref_variant,
                "pearson_vs_reference": float(safe_corr(response, ref)),
                "spearman_vs_reference": float(pd.Series(response.reshape(-1)).corr(pd.Series(ref.reshape(-1)), method="spearman")),
                "top10_jaccard_vs_reference": float(mask_jaccard(top_fraction_mask(response, 0.10), top_fraction_mask(ref, 0.10))),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(sensitivity_rows)


def summarize(per_case: pd.DataFrame, bootstrap: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_cols = ["model", "target_mode", "cam_variant", "node", "node_order"]
    rows = []
    ci_rows = []
    for key, group in per_case.groupby(group_cols, sort=False):
        payload = dict(zip(group_cols, key))
        payload["n_cases"] = int(group["case_name"].nunique())
        payload["n_slices"] = int(len(group))
        payload["seed"] = int(seed)
        for metric in METRICS:
            payload[metric] = float(group[metric].mean())
            lo, hi = bootstrap_mean_ci(group[metric].to_numpy(), n_boot=int(bootstrap), seed=int(seed))
            ci_rows.append({**dict(zip(group_cols, key)), "metric": metric, "mean": payload[metric], "ci_low": lo, "ci_high": hi, "n_cases": payload["n_cases"], "n_slices": payload["n_slices"]})
        rows.append(payload)
    return pd.DataFrame(rows).sort_values(["model", "target_mode", "cam_variant", "node_order"]), pd.DataFrame(ci_rows)


def summarize_sensitivity(sens: pd.DataFrame, bootstrap: int, seed: int) -> pd.DataFrame:
    if sens.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["model", "target_mode", "cam_variant", "node"]
    for key, group in sens.groupby(group_cols, sort=False):
        payload = dict(zip(group_cols, key))
        payload["n_cases"] = int(group["case_name"].nunique())
        for metric in ("pearson_vs_reference", "spearman_vs_reference", "top10_jaccard_vs_reference"):
            payload[metric] = float(group[metric].mean())
            lo, hi = bootstrap_mean_ci(group[metric].to_numpy(), n_boot=int(bootstrap), seed=int(seed))
            payload["{}_ci_low".format(metric)] = lo
            payload["{}_ci_high".format(metric)] = hi
        rows.append(payload)
    return pd.DataFrame(rows)


def plot_sensitivity(summary: pd.DataFrame, out_path: str) -> None:
    if summary.empty:
        return
    combos = ["{}|{}".format(row["target_mode"], row["cam_variant"]) for _, row in summary[["target_mode", "cam_variant"]].drop_duplicates().iterrows()]
    models = list(summary["model"].drop_duplicates())
    fig, axes = plt.subplots(len(METRICS), len(models), figsize=(6.2 * max(1, len(models)), 3.3 * len(METRICS)), dpi=160, squeeze=False)
    for col_idx, model in enumerate(models):
        sub_model = summary[summary["model"] == model]
        for row_idx, metric in enumerate(METRICS):
            ax = axes[row_idx, col_idx]
            for (target_mode, variant), group in sub_model.groupby(["target_mode", "cam_variant"], sort=False):
                group = group.sort_values("node_order")
                ax.plot(group["node"], group[metric], marker="o", linewidth=1.7, label="{}|{}".format(target_mode, variant))
            ax.set_title("{} | {}".format(model, metric))
            ax.grid(alpha=0.25)
            if row_idx == 0:
                ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.num_cases = min(int(args.num_cases), 2)
        args.batch_size = 1
        args.bootstrap = min(int(args.bootstrap), 50)
    ensure_dir(args.output_root)
    set_seed(int(args.seed))
    specs = parse_model_specs(args.model_spec)
    node_names = resolve_node_names(args.nodes)
    target_modes = parse_csv_list(args.target_modes)
    variants = parse_csv_list(args.cam_variants)
    invalid_targets = sorted(set(target_modes) - set(TARGET_MODES))
    invalid_variants = sorted(set(variants) - set(CAM_VARIANTS) - {"gradcam_abs"})
    if invalid_targets:
        raise ValueError("Unsupported target modes: {}".format(invalid_targets))
    if invalid_variants:
        raise ValueError("Unsupported CAM variants: {}".format(invalid_variants))
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

    frames = []
    sens_frames = []
    for spec in specs:
        frame, sens = collect_model_rows(spec, loader, device=device, node_names=node_names, target_modes=target_modes, variants=variants)
        frames.append(frame)
        sens_frames.append(sens)
    per_case = pd.concat(frames, ignore_index=True)
    per_case["seed"] = int(args.seed)
    sensitivity = pd.concat(sens_frames, ignore_index=True) if sens_frames else pd.DataFrame()
    summary, ci = summarize(per_case, bootstrap=int(args.bootstrap), seed=int(args.seed))
    sens_summary = summarize_sensitivity(sensitivity, bootstrap=int(args.bootstrap), seed=int(args.seed))

    per_case.to_csv(os.path.join(args.output_root, "table_cam_variant_case_level.csv"), index=False)
    summary.to_csv(os.path.join(args.output_root, "table_cam_variant_summary.csv"), index=False)
    ci.to_csv(os.path.join(args.output_root, "table_cam_sensitivity_bootstrap_ci.csv"), index=False)
    sensitivity.to_csv(os.path.join(args.output_root, "table_cam_variant_sensitivity_case_level.csv"), index=False)
    sens_summary.to_csv(os.path.join(args.output_root, "table_cam_variant_sensitivity_summary.csv"), index=False)
    plot_sensitivity(summary, os.path.join(args.output_root, "plot_cam_variant_sensitivity.png"))

    write_run_config(args, args.output_root, extra={"device": str(device), "selected_indices": [int(v) for v in indices]})
    write_manifest(
        args.output_root,
        [
            {"output": name, "path": os.path.join(args.output_root, name)}
            for name in (
                "table_cam_variant_case_level.csv",
                "table_cam_variant_summary.csv",
                "table_cam_sensitivity_bootstrap_ci.csv",
                "table_cam_variant_sensitivity_summary.csv",
                "plot_cam_variant_sensitivity.png",
            )
        ],
    )
    write_json(
        {
            "purpose": "E6 CAM target/variant sensitivity",
            "num_cases": int(len(case_names)),
            "models": [spec.name for spec in specs],
            "nodes": list(node_names),
            "target_modes": list(target_modes),
            "cam_variants": list(variants),
            "guardrail": "This is a sensitivity check, not a CAM-method competition or a new explanation algorithm claim.",
        },
        os.path.join(args.output_root, "summary_cam_sensitivity.json"),
    )
    print("Saved E6 CAM sensitivity outputs to {}".format(args.output_root))
    print("Device:", device)
    print("Cases:", len(case_names))


if __name__ == "__main__":
    main()
