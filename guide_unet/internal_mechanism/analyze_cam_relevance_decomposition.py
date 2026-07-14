from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


GUIDE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if GUIDE_ROOT not in sys.path:
    sys.path.insert(0, GUIDE_ROOT)

from analysis_common import (
    ANALYSIS_NODE_NAMES,
    MODEL_CHOICES,
    FeatureStore,
    default_dir,
    iter_node_configs,
    load_model,
    mean_region_dice,
    node_group_hint,
    prepare_fit_eval_loaders,
    safe_corr,
    safe_cos,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight CAM/relevance decomposition analysis.")
    parser.add_argument("--model", type=str, required=True, choices=MODEL_CHOICES)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default=default_dir("data"))
    parser.add_argument("--fit_split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--eval_split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--save_dir",
        type=str,
        default=os.path.join(default_dir("results"), "internal_mechanism", "cam_relevance_decomposition_v1"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_fit_samples", type=int, default=256)
    parser.add_argument("--num_eval_samples", type=int, default=512)
    parser.add_argument("--same_split_fit_ratio", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--fit_epochs", type=int, default=4)
    parser.add_argument("--fit_lr", type=float, default=5e-4)
    parser.add_argument("--fusion_epochs", type=int, default=80)
    parser.add_argument("--fusion_lr", type=float, default=5e-2)
    parser.add_argument("--foreground_only", action="store_true")
    parser.add_argument("--patient_balanced_sampling", action="store_true")
    return parser.parse_args()


class NodeCAMProjector(nn.Module):
    def __init__(self, in_channels: int, target_channels: int, target_size):
        super().__init__()
        self.target_size = tuple(int(v) for v in target_size)
        self.proj = nn.Conv2d(int(in_channels), int(target_channels), kernel_size=1, bias=False)
        self.refine = nn.Sequential(
            nn.Conv2d(int(target_channels), int(target_channels), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(int(target_channels)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(target_channels), int(target_channels), kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        if x.shape[2:] != self.target_size:
            x = F.interpolate(x, size=self.target_size, mode="bilinear", align_corners=False)
        return x + self.refine(x)


class ClasswiseCAMFusion(nn.Module):
    def __init__(self, node_names: List[str], num_classes: int = 4):
        super().__init__()
        self.node_names = list(node_names)
        self.num_classes = int(num_classes)
        self.raw_weights = nn.Parameter(torch.zeros(self.num_classes, len(self.node_names)))

    def normalized_weights(self) -> torch.Tensor:
        return torch.softmax(self.raw_weights, dim=1)

    def forward(self, cam_stack: torch.Tensor) -> torch.Tensor:
        # cam_stack: [B, N, C, H, W]
        weights = self.normalized_weights()  # [C, N]
        weights = weights.transpose(0, 1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [1, N, C, 1, 1]
        return (cam_stack * weights).sum(dim=1)


def _compute_logits_loss(logits_hat: torch.Tensor, target_logits: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(logits_hat, target_logits)
    cos = 1.0 - F.cosine_similarity(logits_hat.flatten(1), target_logits.flatten(1), dim=1).mean()
    return mse + 0.2 * cos


def _structure_edge_map(image_chw: np.ndarray) -> np.ndarray:
    image_2d = image_chw.mean(axis=0).astype(np.float32)
    gy, gx = np.gradient(image_2d)
    edge = np.sqrt(gx**2 + gy**2)
    denom = float(np.max(edge))
    if denom > 0:
        edge = edge / denom
    return edge


def _tumor_response(logits_chw: np.ndarray) -> np.ndarray:
    logits = torch.from_numpy(logits_chw).unsqueeze(0)
    probs = torch.softmax(logits, dim=1)[0].numpy()
    return probs[1:].sum(axis=0)


def _train_projectors(
    model,
    collector: FeatureStore,
    projectors: Dict[str, nn.Module],
    fit_loader,
    device: torch.device,
    fit_epochs: int,
    fit_lr: float,
):
    params = []
    for proj in projectors.values():
        params.extend(list(proj.parameters()))
    optimizer = torch.optim.Adam(params, lr=float(fit_lr))
    history = []
    for _ in range(int(fit_epochs)):
        epoch_losses = []
        for batch in fit_loader:
            images = batch["image"].to(device)
            collector.clear()
            with torch.no_grad():
                _ = model(images)
                node_map = collector.node_map()
                final_feat = node_map["up4"].detach()
                final_logits = model.outc(final_feat).detach()

            optimizer.zero_grad()
            loss = final_logits.new_zeros(())
            for node_name, projector in projectors.items():
                feat_hat = projector(node_map[node_name].detach())
                logits_hat = model.outc(feat_hat)
                loss = loss + _compute_logits_loss(logits_hat, final_logits)
            loss = loss / float(len(projectors))
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        history.append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)
    return history


def _collect_eval_outputs(
    model,
    collector: FeatureStore,
    projectors: Dict[str, nn.Module],
    eval_loader,
    device: torch.device,
):
    rows = []
    fusion_buffers = []
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="CAM relevance eval", leave=False):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            case_names = list(batch["case_name"])
            collector.clear()
            logits_final = model(images)[0]
            node_map = collector.node_map()
            pred_final = torch.argmax(logits_final, dim=1)

            node_logits = {}
            for node_name, projector in projectors.items():
                feat_hat = projector(node_map[node_name].detach())
                logits_hat = model.outc(feat_hat)
                node_logits[node_name] = logits_hat

            cam_stack = torch.stack([node_logits[node_name] for node_name in ANALYSIS_NODE_NAMES], dim=1)
            fusion_buffers.append(
                {
                    "cam_stack": cam_stack.detach().cpu(),
                    "final_logits": logits_final.detach().cpu(),
                    "labels": labels.detach().cpu(),
                }
            )

            for local_idx, case_name in enumerate(case_names):
                label_np = labels[local_idx].detach().cpu().numpy()
                final_logits_np = logits_final[local_idx].detach().cpu().numpy()
                final_tumor = _tumor_response(final_logits_np)
                structure_anchor = _structure_edge_map(batch["image"][local_idx].numpy())
                pred_final_np = pred_final[local_idx].detach().cpu().numpy()

                for node_name in ANALYSIS_NODE_NAMES:
                    logits_hat_np = node_logits[node_name][local_idx].detach().cpu().numpy()
                    pred_hat_np = torch.argmax(node_logits[node_name][local_idx], dim=0).detach().cpu().numpy()
                    node_tumor = _tumor_response(logits_hat_np)

                    rows.append(
                        {
                            "case_name": case_name,
                            "node": node_name,
                            "group_hint": node_group_hint(node_name),
                            "cam_final_corr": safe_corr(node_tumor, final_tumor),
                            "cam_final_cosine": safe_cos(node_tumor, final_tumor),
                            "cam_structure_corr": safe_corr(node_tumor, structure_anchor),
                            "cam_structure_cosine": safe_cos(node_tumor, structure_anchor),
                            "cam_vs_final_mask_mean_dice": float(mean_region_dice(pred_hat_np, pred_final_np)),
                            "cam_vs_gt_mean_dice": float(mean_region_dice(pred_hat_np, label_np)),
                            "final_vs_gt_mean_dice": float(mean_region_dice(pred_final_np, label_np)),
                        }
                    )
    return rows, fusion_buffers


def _train_fusion_model(buffers, num_classes: int, fusion_epochs: int, fusion_lr: float, device: torch.device):
    fusion = ClasswiseCAMFusion(node_names=list(ANALYSIS_NODE_NAMES), num_classes=int(num_classes)).to(device)
    optimizer = torch.optim.Adam(fusion.parameters(), lr=float(fusion_lr))
    history = []
    for _ in range(int(fusion_epochs)):
        epoch_losses = []
        for item in buffers:
            cam_stack = item["cam_stack"].to(device)
            final_logits = item["final_logits"].to(device)
            optimizer.zero_grad()
            fused_logits = fusion(cam_stack)
            loss = _compute_logits_loss(fused_logits, final_logits)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        history.append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)
    return fusion, history


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_bundle = prepare_fit_eval_loaders(
        data_dir=args.data_dir,
        fit_split=args.fit_split,
        eval_split=args.eval_split,
        num_fit_samples=int(args.num_fit_samples),
        num_eval_samples=int(args.num_eval_samples),
        foreground_only=bool(args.foreground_only),
        seed=int(args.seed),
        fit_ratio=float(args.same_split_fit_ratio),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        patient_balanced=bool(args.patient_balanced_sampling),
    )

    model = load_model(model_name=args.model, ckpt_path=args.ckpt_path, device=device)
    collector = FeatureStore(model)
    warmup_batch = next(iter(split_bundle["fit_loader"]))
    with torch.no_grad():
        collector.clear()
        _ = model(warmup_batch["image"].to(device))
        node_map = collector.node_map()
        final_feat = node_map["up4"]
    target_channels = int(final_feat.shape[1])
    target_size = tuple(int(v) for v in final_feat.shape[2:])

    projectors = {
        node_name: NodeCAMProjector(
            in_channels=int(node_map[node_name].shape[1]),
            target_channels=target_channels,
            target_size=target_size,
        ).to(device)
        for node_name in ANALYSIS_NODE_NAMES
    }

    projector_history = _train_projectors(
        model=model,
        collector=collector,
        projectors=projectors,
        fit_loader=split_bundle["fit_loader"],
        device=device,
        fit_epochs=int(args.fit_epochs),
        fit_lr=float(args.fit_lr),
    )

    rows, fusion_buffers = _collect_eval_outputs(
        model=model,
        collector=collector,
        projectors=projectors,
        eval_loader=split_bundle["eval_loader"],
        device=device,
    )
    collector.remove()

    fusion_model, fusion_history = _train_fusion_model(
        buffers=fusion_buffers,
        num_classes=4,
        fusion_epochs=int(args.fusion_epochs),
        fusion_lr=float(args.fusion_lr),
        device=device,
    )

    result_df = pd.DataFrame(rows)
    result_df.to_csv(os.path.join(args.save_dir, f"{args.model}_cam_relevance_per_case.csv"), index=False)

    node_summary = {}
    for node_name, group_df in result_df.groupby("node"):
        node_summary[node_name] = {
            "group_hint": str(group_df["group_hint"].iloc[0]),
            "cam_final_corr": float(group_df["cam_final_corr"].mean()),
            "cam_final_cosine": float(group_df["cam_final_cosine"].mean()),
            "cam_structure_corr": float(group_df["cam_structure_corr"].mean()),
            "cam_structure_cosine": float(group_df["cam_structure_cosine"].mean()),
            "cam_vs_final_mask_mean_dice": float(group_df["cam_vs_final_mask_mean_dice"].mean()),
            "cam_vs_gt_mean_dice": float(group_df["cam_vs_gt_mean_dice"].mean()),
            "final_vs_gt_mean_dice": float(group_df["final_vs_gt_mean_dice"].mean()),
        }

    weight_rows = []
    weights = fusion_model.normalized_weights().detach().cpu().numpy()  # [C, N]
    class_names = ["bg", "NCR", "Edema", "ET"]
    for cls_idx, class_name in enumerate(class_names):
        for node_idx, node_name in enumerate(ANALYSIS_NODE_NAMES):
            weight_rows.append(
                {
                    "class_name": class_name,
                    "class_idx": cls_idx,
                    "node": node_name,
                    "group_hint": node_group_hint(node_name),
                    "relevance_weight": float(weights[cls_idx, node_idx]),
                }
            )
    weight_df = pd.DataFrame(weight_rows)
    weight_df.to_csv(os.path.join(args.save_dir, f"{args.model}_cam_relevance_weights.csv"), index=False)

    plot_curve = pd.DataFrame(
        [
            {
                "node": node_name,
                "group_hint": payload["group_hint"],
                "structure_corr": payload["cam_structure_corr"],
                "task_corr": payload["cam_final_corr"],
                "task_dice_gt": payload["cam_vs_gt_mean_dice"],
            }
            for node_name, payload in node_summary.items()
        ]
    )
    plot_curve.to_csv(os.path.join(args.save_dir, f"{args.model}_cam_structure_task_curve.csv"), index=False)

    summary = {
        "model": args.model,
        "fit_split": args.fit_split,
        "eval_split": args.eval_split,
        "num_fit_samples": int(len(split_bundle["fit_indices"])),
        "num_eval_samples": int(len(split_bundle["eval_indices"])),
        "projector_fit_history": projector_history,
        "fusion_fit_history": fusion_history,
        "node_mean": node_summary,
    }
    with open(os.path.join(args.save_dir, f"{args.model}_cam_relevance_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Saved CAM relevance outputs to {}".format(args.save_dir))


if __name__ == "__main__":
    main()
