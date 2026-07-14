"""Current U-Net training entry for internal mechanism analysis.

What this file does:
- Trains baseline U-Net or no-skip U-Net on 2D BraTS slices.
- Records per-epoch training/validation metrics to CSV.
- Saves latest and best-val-loss checkpoints with explicit names.

Key modules:
- Data: `datasets.dataset_brats.BraTSDataset`
- Models: `Architecture/*`
- Loss: CE + Dice (0.2 / 0.8)

Key outputs:
- Checkpoints: `<save_dir>/<run_name>/<model>/seed_<seed>/`
- Logs: `<log_dir>/<run_name>/<model>/seed_<seed>/`
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


GUIDE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE_ROOT = os.path.dirname(GUIDE_ROOT)
sys.path.insert(0, GUIDE_ROOT)

from Architecture.unet_baseline import UNetBaseline
from Architecture.unet_noskip import UNetNoSkip
from datasets.dataset_brats import BraTSDataset


class DiceLoss:
    def __init__(self, n_classes: int):
        self.n_classes = int(n_classes)

    def _one_hot_encoder(self, input_tensor: torch.Tensor) -> torch.Tensor:
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        return torch.cat(tensor_list, dim=1).float()

    @staticmethod
    def _dice_loss(score: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        return 1 - (2 * intersect + smooth) / (z_sum + y_sum + smooth)

    def forward(self, inputs: torch.Tensor, target: torch.Tensor, softmax: bool = False) -> torch.Tensor:
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target_oh = self._one_hot_encoder(target)
        loss = 0.0
        for i in range(self.n_classes):
            loss = loss + self._dice_loss(inputs[:, i], target_oh[:, i])
        return loss / float(self.n_classes)

    def __call__(self, inputs: torch.Tensor, target: torch.Tensor, softmax: bool = False) -> torch.Tensor:
        return self.forward(inputs, target, softmax=softmax)


def compute_dice_per_class(pred: torch.Tensor, target: torch.Tensor, n_classes: int):
    dice_scores = {}
    class_names = {1: "NCR", 2: "Edema", 3: "ET"}
    for c in range(1, int(n_classes)):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        dice = (2 * intersection + 1e-5) / (union + 1e-5)
        dice_scores[class_names.get(c, f"Class_{c}")] = float(dice.item())
    dice_scores["Mean"] = float(np.mean(list(dice_scores.values()))) if dice_scores else 0.0
    return dice_scores


class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 0.0, mode: str = "max"):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = str(mode)
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        score = float(score)
        if self.best_score is None:
            self.best_score = score
            return False

        improved = False
        if self.mode == "max":
            improved = score > float(self.best_score) + self.min_delta
        else:
            improved = score < float(self.best_score) - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            return True

        return False


MODEL_CHOICES = (
    "baseline",
    "noskip_unet",
)


def unpack_model_outputs(model_outputs):
    if not isinstance(model_outputs, (tuple, list)):
        raise TypeError("Model forward must return tuple/list with logits as first element.")
    if len(model_outputs) not in (4, 5):
        raise ValueError(
            f"Unexpected number of model outputs: {len(model_outputs)}. Expected 4 or 5."
        )
    logits = model_outputs[0]
    extras = {}
    if len(model_outputs) == 5 and isinstance(model_outputs[4], dict):
        extras = model_outputs[4]
    return logits, extras


def compute_seg_loss(logits, labels, ce_loss, dice_loss):
    loss_ce = ce_loss(logits, labels.long())
    loss_d = dice_loss(logits, labels, softmax=True)
    loss = 0.2 * loss_ce + 0.8 * loss_d
    return loss, loss_ce, loss_d


def compute_aux_losses(extras, logits, labels, ce_loss, dice_loss):
    aux_logits = extras.get("aux_logits", {}) if isinstance(extras, dict) else {}
    if not aux_logits:
        zero = logits.new_zeros(())
        return zero, zero

    aux_gt_weights = {"up1": 0.20, "up2": 0.12}
    aux_int_weights = {"up1": 0.05, "up2": 0.025}

    final_probs = torch.softmax(logits.detach(), dim=1)
    total_aux_gt = logits.new_zeros(())
    total_aux_int = logits.new_zeros(())
    target_hw = labels.shape[-2:]

    for name, aux_logit in aux_logits.items():
        aux_up = aux_logit
        if aux_up.shape[-2:] != target_hw:
            aux_up = nn.functional.interpolate(aux_up, size=target_hw, mode="bilinear", align_corners=False)

        aux_seg, _, _ = compute_seg_loss(aux_up, labels, ce_loss, dice_loss)
        total_aux_gt = total_aux_gt + float(aux_gt_weights.get(name, 0.0)) * aux_seg

        aux_probs = torch.softmax(aux_up, dim=1)
        consistency = nn.functional.mse_loss(aux_probs, final_probs)
        total_aux_int = total_aux_int + float(aux_int_weights.get(name, 0.0)) * consistency

    return total_aux_gt, total_aux_int


def load_partial_state_dict(model, ckpt_path, device):
    payload = torch.load(ckpt_path, map_location=device)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    elif isinstance(payload, dict):
        state_dict = payload
    else:
        raise TypeError("Unsupported init checkpoint payload type: {}".format(type(payload)))

    model_state = model.state_dict()
    compatible = OrderedDict()
    skipped = []
    for key, value in state_dict.items():
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape):
            compatible[key] = value
        else:
            skipped.append(key)

    missing = [key for key in model_state.keys() if key not in compatible]
    model.load_state_dict(compatible, strict=False)
    print(
        "Loaded partial init from {} | matched: {} | skipped from ckpt: {} | model missing: {}".format(
            ckpt_path,
            len(compatible),
            len(skipped),
            len(missing),
        )
    )
    if skipped:
        print("  Skipped keys (shape/name mismatch): {}".format(", ".join(skipped[:12])))
    if missing:
        print("  Model keys left randomly initialized: {}".format(", ".join(missing[:12])))


def train_one_epoch(model, dataloader, optimizer, ce_loss, dice_loss, device):
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_dice = 0.0
    total_aux_gt = 0.0
    total_aux_int = 0.0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits, extras = unpack_model_outputs(model(images))

        loss, loss_ce, loss_d = compute_seg_loss(logits, labels, ce_loss, dice_loss)
        loss_aux_gt, loss_aux_int = compute_aux_losses(extras, logits, labels, ce_loss, dice_loss)
        loss = loss + loss_aux_gt + loss_aux_int

        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_ce += float(loss_ce.item())
        total_dice += float(loss_d.item())
        total_aux_gt += float(loss_aux_gt.item())
        total_aux_int += float(loss_aux_int.item())

        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "ce": f"{loss_ce.item():.4f}",
                "dice": f"{loss_d.item():.4f}",
                "aux_gt": f"{loss_aux_gt.item():.4f}",
                "aux_int": f"{loss_aux_int.item():.4f}",
            }
        )

    n = max(1, int(len(dataloader)))
    return total_loss / n, total_ce / n, total_dice / n, total_aux_gt / n, total_aux_int / n


def validate(model, dataloader, device, n_classes, ce_loss, dice_loss):
    model.eval()
    all_dice = []
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits, _ = unpack_model_outputs(model(images))
            loss, _, _ = compute_seg_loss(logits, labels, ce_loss, dice_loss)

            total_loss += float(loss.item())
            total_batches += 1

            if torch.sum(labels) == 0:
                continue

            pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            for i in range(int(pred.shape[0])):
                dice_scores = compute_dice_per_class(pred[i], labels[i], n_classes)
                all_dice.append(dice_scores)

    if all_dice:
        dice_df = pd.DataFrame(all_dice)
        dice_ncr = float(dice_df["NCR"].mean())
        dice_edema = float(dice_df["Edema"].mean())
        dice_et = float(dice_df["ET"].mean())
        dice_mean = float(dice_df["Mean"].mean())
    else:
        dice_ncr = 0.0
        dice_edema = 0.0
        dice_et = 0.0
        dice_mean = 0.0

    val_loss = total_loss / float(total_batches) if total_batches > 0 else float("inf")
    return {
        "Val_Loss": float(val_loss),
        "Dice_NCR": dice_ncr,
        "Dice_Edema": dice_edema,
        "Dice_ET": dice_et,
        "Dice_Mean": dice_mean,
    }


def _default_dir(name: str) -> str:
    return os.path.join(WORKSPACE_ROOT, name)


def main():
    parser = argparse.ArgumentParser(description="BraTS segmentation training (guide-unet)")
    parser.add_argument("--model", type=str, required=True, choices=MODEL_CHOICES)

    parser.add_argument("--data_dir", type=str, default=_default_dir("data"))
    parser.add_argument("--val_data_dir", type=str, default="")
    parser.add_argument("--test_data_dir", type=str, default="")
    parser.add_argument("--manifest_path", type=str, default="")

    parser.add_argument("--save_dir", type=str, default=_default_dir("checkpoints"))
    parser.add_argument("--log_dir", type=str, default=_default_dir("logs"))
    parser.add_argument(
        "--run_name",
        type=str,
        default="manual",
        help="Run group name used in output directory layout",
    )

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--patience", type=int, default=15)

    parser.add_argument("--resume", type=str, default="")
    parser.add_argument(
        "--init_from",
        type=str,
        default="",
        help="Optional checkpoint to partially initialize matching weights before training",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    train_data_dir = args.data_dir
    val_data_dir = args.val_data_dir if args.val_data_dir else args.data_dir
    if args.test_data_dir and os.path.abspath(args.test_data_dir) == os.path.abspath(val_data_dir):
        raise ValueError("`--test_data_dir` must not equal validation data directory used for model selection.")

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(args.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    run_name = args.run_name.strip() or "manual"
    save_dir = os.path.join(args.save_dir, run_name, args.model, "seed_{}".format(int(args.seed)))
    log_dir = os.path.join(args.log_dir, run_name, args.model, "seed_{}".format(int(args.seed)))
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    latest_ckpt_path = os.path.join(
        save_dir,
        "{}_seed{}_latest.pth".format(args.model, int(args.seed)),
    )
    best_ckpt_path = os.path.join(
        save_dir,
        "{}_seed{}_best_val_loss.pth".format(args.model, int(args.seed)),
    )

    if args.model == "baseline":
        model = UNetBaseline(n_channels=4, n_classes=int(args.num_classes))
    elif args.model == "noskip_unet":
        model = UNetNoSkip(n_channels=4, n_classes=int(args.num_classes))
    else:
        raise ValueError("Unsupported model: {}".format(args.model))

    model = model.to(device)
    total_params = sum(int(p.numel()) for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    if args.init_from:
        if not os.path.exists(args.init_from):
            raise FileNotFoundError("`--init_from` not found: {}".format(args.init_from))
        load_partial_state_dict(model, args.init_from, device)

    print("Loading datasets...")
    train_dataset = BraTSDataset(base_dir=train_data_dir, split="train")
    val_dataset = BraTSDataset(base_dir=val_data_dir, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(args.num_workers),
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    if args.test_data_dir:
        print(f"Test data dir recorded only (never loaded in training): {args.test_data_dir}")

    optimizer = optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(args.max_epochs), eta_min=1e-6)

    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(int(args.num_classes))
    early_stopping = EarlyStopping(patience=int(args.patience), mode="min")

    start_epoch = 0
    best_dice = 0.0
    best_val_loss = float("inf")
    train_history = []

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_dice = float(checkpoint.get("best_dice", 0.0))
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        train_history = list(checkpoint.get("history", []))
        print(
            f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.4f}, best dice: {best_dice:.4f}"
        )

    csv_path = os.path.join(
        log_dir,
        "{}_seed{}_training_log.csv".format(args.model, int(args.seed)),
    )
    metadata_path = os.path.join(
        log_dir,
        "{}_seed{}_training_metadata.json".format(args.model, int(args.seed)),
    )
    split_provenance = {"data_dir": train_data_dir, "val_data_dir": val_data_dir}
    if args.manifest_path:
        split_provenance["manifest_path"] = args.manifest_path
    if args.test_data_dir:
        split_provenance["test_data_dir_record_only"] = args.test_data_dir

    training_metadata = {
        "seed": int(args.seed),
        "model_config": {"model_key": args.model},
        "split_provenance": split_provenance,
        "hyperparameters": {
            "model": args.model,
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.lr),
            "max_epochs": int(args.max_epochs),
            "num_classes": int(args.num_classes),
            "patience": int(args.patience),
            "num_workers": int(args.num_workers),
        },
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(training_metadata, f, indent=2, sort_keys=True)

    print("Starting training...")
    print(f"Model: {args.model}")
    print(f"Run name: {run_name}")
    print(f"Max epochs: {int(args.max_epochs)}")
    print(f"Early stopping patience: {int(args.patience)}")
    print("-" * 60)

    for epoch in range(int(start_epoch), int(args.max_epochs)):
        print(f"\nEpoch {epoch + 1}/{int(args.max_epochs)}")
        train_loss, train_ce, train_dice, train_aux_gt, train_aux_int = train_one_epoch(
            model, train_loader, optimizer, ce_loss, dice_loss, device
        )
        val_metrics = validate(model, val_loader, device, int(args.num_classes), ce_loss, dice_loss)
        scheduler.step()
        current_lr = float(optimizer.param_groups[0]["lr"])

        history_entry = {
            "Epoch": int(epoch + 1),
            "Train_Loss": float(train_loss),
            "Train_CE": float(train_ce),
            "Train_Dice_Loss": float(train_dice),
            "Train_Aux_GT": float(train_aux_gt),
            "Train_Aux_Internal": float(train_aux_int),
            "LR": float(current_lr),
        }
        history_entry.update(val_metrics)
        train_history.append(history_entry)

        pd.DataFrame(train_history).to_csv(csv_path, index=False)

        print(
            f"  Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['Val_Loss']:.4f} | "
            f"Val Dice: {val_metrics['Dice_Mean']:.4f} | LR: {current_lr:.6f}"
        )
        print(f"  Aux - GT: {train_aux_gt:.4f} | Internal: {train_aux_int:.4f}")
        print(
            f"  Dice - NCR: {val_metrics['Dice_NCR']:.4f}, Edema: {val_metrics['Dice_Edema']:.4f}, "
            f"ET: {val_metrics['Dice_ET']:.4f}"
        )

        checkpoint = {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_dice": float(best_dice),
            "best_val_loss": float(best_val_loss),
            "history": train_history,
        }
        torch.save(checkpoint, latest_ckpt_path)

        if float(val_metrics["Val_Loss"]) < float(best_val_loss):
            best_val_loss = float(val_metrics["Val_Loss"])
            best_dice = float(val_metrics["Dice_Mean"])
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"  >>> New best model! Val Loss: {best_val_loss:.4f} | Dice: {best_dice:.4f}")

        if early_stopping(float(val_metrics["Val_Loss"])):
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            print(f"Best validation Loss: {float(early_stopping.best_score):.4f}")
            break

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best validation Loss: {best_val_loss:.4f}")
    print(f"Dice at best validation Loss checkpoint: {best_dice:.4f}")
    print(f"Best checkpoint: {best_ckpt_path}")
    print(f"Latest checkpoint: {latest_ckpt_path}")
    print(f"Checkpoint directory: {save_dir}")
    print(f"Training log saved to: {csv_path}")
    print(f"Training metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
