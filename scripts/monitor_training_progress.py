from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import yaml


MODEL_CONFIGS = {
    "unet_baseline": Path("configs/models/unet_baseline.yaml"),
    "unet_noskip": Path("configs/models/unet_noskip.yaml"),
    "transunet_r50_vit_b16": Path("configs/models/transunet_r50_vit_b16.yaml"),
}

# These conservative values are used only until a model completes one epoch on
# the profiled RTX 4090 batches. They are then replaced automatically.
PROVISIONAL_EPOCH_SECONDS = {
    "unet_baseline": 4.0 * 60.0,
    "unet_noskip": 4.0 * 60.0,
    "transunet_r50_vit_b16": 10.0 * 60.0,
}


def _parse_job(value: str) -> tuple[str, int]:
    model, separator, raw_seed = value.rpartition(":")
    if not separator or model not in MODEL_CONFIGS:
        raise ValueError(f"Invalid training job: {value}")
    return model, int(raw_seed)


def _load_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _training_config(model: str) -> tuple[int, int]:
    payload = yaml.safe_load(MODEL_CONFIGS[model].read_text(encoding="utf-8"))
    training = payload["training"]
    return int(training["epochs"]), int(training["patience"])


def _runner_pid(pid_path: Path) -> int | None:
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
        return pid
    except (OSError, ValueError):
        return None


def _proc_values(pid: int) -> tuple[int, float] | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        parent_pid = int(fields[3])
        start_ticks = int(fields[21])
        boot_time = next(
            int(line.split()[1])
            for line in Path("/proc/stat").read_text(encoding="utf-8").splitlines()
            if line.startswith("btime ")
        )
        started_at = boot_time + start_ticks / os.sysconf("SC_CLK_TCK")
        return parent_pid, started_at
    except (OSError, StopIteration, ValueError, IndexError):
        return None


def _proc_command(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return []
    return [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]


def _current_job(runner_pid: int | None) -> dict[str, Any] | None:
    if runner_pid is None:
        return None
    for proc_path in Path("/proc").glob("[0-9]*"):
        pid = int(proc_path.name)
        values = _proc_values(pid)
        if values is None or values[0] != runner_pid:
            continue
        command = _proc_command(pid)
        if not any(value.endswith("scripts/train_segmentation.py") for value in command):
            continue
        try:
            config = Path(command[command.index("--config") + 1]).stem
            seed = int(command[command.index("--seed") + 1])
        except (ValueError, IndexError):
            continue
        model = next(
            (name for name, path in MODEL_CONFIGS.items() if path.stem == config),
            config,
        )
        return {
            "pid": pid,
            "model": model,
            "seed": seed,
            "started_at": values[1],
        }
    return None


def _history(output_root: Path, model: str, seed: int) -> tuple[list[dict], float | None]:
    path = output_root / model / f"seed_{seed}" / "history.json"
    payload = _load_json(path, {"epochs": []})
    epochs = payload.get("epochs", []) if isinstance(payload, dict) else []
    try:
        modified_at = path.stat().st_mtime
    except OSError:
        modified_at = None
    return epochs, modified_at


def _summary_status(output_root: Path, model: str, seed: int) -> str | None:
    path = output_root / model / f"seed_{seed}" / "training_summary.json"
    payload = _load_json(path, {})
    status = payload.get("status") if isinstance(payload, dict) else None
    return str(status) if status is not None else None


def _no_improvement_streak(history: list[dict]) -> int:
    streak = 0
    for epoch in reversed(history):
        if bool(epoch.get("improved")):
            break
        streak += 1
    return streak


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    days, seconds = divmod(seconds, 86_400)
    hours, seconds = divmod(seconds, 3_600)
    minutes, _ = divmod(seconds, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours or days:
        parts.append(f"{hours}h")
    parts.append(f"{minutes}m")
    return " ".join(parts)


def _format_time(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")


def _gpu_status() -> str:
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        utilization, used, total, temperature = [part.strip() for part in output.split(",")]
        return f"{utilization}% | {used}/{total} MiB | {temperature} C"
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return "unavailable"


def _rate_for_model(
    model: str,
    calibrated: dict[str, dict[str, Any]],
) -> tuple[float, str]:
    entry = calibrated.get(model, {})
    seconds = entry.get("epoch_seconds")
    if isinstance(seconds, (int, float)) and math.isfinite(seconds) and seconds > 0:
        return float(seconds), f"基于 {entry.get('completed_epochs', 0)} 轮实测"
    if model == "unet_noskip" and "unet_baseline" in calibrated:
        baseline = calibrated["unet_baseline"].get("epoch_seconds")
        if isinstance(baseline, (int, float)) and baseline > 0:
            return float(baseline), "按 U-Net baseline 实测速率暂估"
    return PROVISIONAL_EPOCH_SECONDS[model], "暂估"


def _update_calibration(
    state: dict[str, Any],
    current: dict[str, Any] | None,
    output_root: Path,
    now: float,
) -> None:
    if current is None:
        return
    history, modified_at = _history(output_root, current["model"], current["seed"])
    if not history or modified_at is None:
        return
    elapsed_to_last_epoch = modified_at - float(current["started_at"])
    if elapsed_to_last_epoch <= 0:
        return
    epoch_seconds = elapsed_to_last_epoch / len(history)
    if not math.isfinite(epoch_seconds) or epoch_seconds <= 0:
        return
    state.setdefault("model_calibration", {})[current["model"]] = {
        "epoch_seconds": epoch_seconds,
        "completed_epochs": len(history),
        "updated_at": now,
    }


def _render(args: argparse.Namespace) -> str:
    now = time.time()
    jobs = [_parse_job(value) for value in args.jobs]
    runner_pid = _runner_pid(args.pid_file)
    current = _current_job(runner_pid)
    state = _load_json(args.state_file, {"model_calibration": {}})
    _update_calibration(state, current, args.output_root, now)
    _atomic_json(args.state_file, state)
    calibrated = state.get("model_calibration", {})

    rows = []
    conservative_remaining = 0.0
    dynamic_remaining = 0.0
    current_remaining = None
    completed_count = 0
    provisional_models = set()

    for index, (model, seed) in enumerate(jobs, start=1):
        total_epochs, patience = _training_config(model)
        history, modified_at = _history(args.output_root, model, seed)
        completed_epochs = len(history)
        status = _summary_status(args.output_root, model, seed)
        is_current = bool(
            current
            and current["model"] == model
            and int(current["seed"]) == seed
        )
        if status == "PASS":
            label = "完成"
            completed_count += 1
            rows.append((label, index, model, seed, completed_epochs, total_epochs, status))
            continue
        if status == "FAIL":
            label = "失败"
            rows.append((label, index, model, seed, completed_epochs, total_epochs, status))
            continue

        label = "运行" if is_current else "等待"
        rate, source = _rate_for_model(model, calibrated)
        if "暂估" in source:
            provisional_models.add(model)
        conservative_epochs = max(0, total_epochs - completed_epochs)
        streak = _no_improvement_streak(history)
        dynamic_target = total_epochs
        if streak >= args.early_stop_projection_after:
            dynamic_target = min(total_epochs, completed_epochs + max(0, patience - streak))
        dynamic_epochs = max(0, dynamic_target - completed_epochs)
        partial_epoch = 0.0
        if is_current:
            reference = modified_at if modified_at is not None else float(current["started_at"])
            partial_epoch = min(rate, max(0.0, now - reference))
        conservative_seconds = max(0.0, conservative_epochs * rate - partial_epoch)
        dynamic_seconds = max(0.0, dynamic_epochs * rate - partial_epoch)
        conservative_remaining += conservative_seconds
        dynamic_remaining += dynamic_seconds
        if is_current:
            current_remaining = dynamic_seconds
        rows.append((label, index, model, seed, completed_epochs, total_epochs, status))

    state_label = "运行中" if runner_pid is not None else "已停止"
    waiting_count = sum(1 for row in rows if row[0] == "等待")
    running_count = sum(1 for row in rows if row[0] == "运行")
    lines = [
        "=" * 78,
        "PPTT 正式训练矩阵监控",
        f"更新时间：{_format_time(now)} | 调度器：{state_label}"
        + (f"（PID {runner_pid}）" if runner_pid is not None else ""),
        f"总进度：完成 {completed_count}/{len(jobs)} | 运行 {running_count} | "
        f"等待 {waiting_count} | GPU：{_gpu_status()}",
        "-" * 78,
    ]
    if current is not None:
        history, _ = _history(args.output_root, current["model"], current["seed"])
        total_epochs, _ = _training_config(current["model"])
        elapsed = now - float(current["started_at"])
        rate, source = _rate_for_model(current["model"], calibrated)
        lines.extend(
            [
                f"正在运行：{current['model']} | seed={current['seed']} | PID={current['pid']}",
                f"训练轮次：已完成 {len(history)}/{total_epochs}"
                + (" | 正在执行首轮" if not history else ""),
                f"已用时间：{_format_duration(elapsed)} | 单轮估计：{_format_duration(rate)}（{source}）",
            ]
        )
        if history:
            latest = history[-1]
            validation = latest.get("validation", {})
            lines.append(
                "最新指标：train_loss={:.4f} | val_loss={:.4f} | val_Dice={:.4f}".format(
                    float(latest.get("train", {}).get("loss", float("nan"))),
                    float(validation.get("loss", float("nan"))),
                    float(validation.get("mean_tumor_dice", float("nan"))),
                )
            )
        if current_remaining is not None:
            lines.append(
                f"当前任务预计完成：{_format_time(now + current_remaining)} "
                f"（剩余 {_format_duration(current_remaining)}）"
            )
    else:
        lines.append("正在运行：未检测到 train_segmentation.py 进程")

    lines.extend(["-" * 78, "任务队列："])
    for label, index, model, seed, completed, total, status in rows:
        suffix = f" | status={status}" if status else ""
        lines.append(
            f"[{label}] {index}/{len(jobs)} {model:<25} seed={seed:<4} "
            f"epochs={completed}/{total}{suffix}"
        )

    confidence = "已校准" if not provisional_models else "暂估"
    lines.extend(
        [
            "-" * 78,
            f"整体预计完成（{confidence}）：{_format_time(now + dynamic_remaining)} "
            f"（剩余 {_format_duration(dynamic_remaining)}）",
            f"满 100 轮保守上界：{_format_time(now + conservative_remaining)} "
            f"（剩余 {_format_duration(conservative_remaining)}）",
        ]
    )
    if provisional_models:
        lines.append(
            "等待实测校准：" + ", ".join(sorted(provisional_models))
        )
    lines.append(
        "完成首轮后自动改用实测速率；触发早停时，预计完成时间会相应提前。"
    )
    lines.append("=" * 78)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Display current task, remaining queue, and dynamic ETA."
    )
    parser.add_argument("--jobs", nargs="+", required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/segmentation_training"),
    )
    parser.add_argument(
        "--pid-file",
        type=Path,
        default=Path("logs/segmentation_training/launcher.pid"),
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=Path("logs/segmentation_training/progress_monitor_state.json"),
    )
    parser.add_argument("--interval-seconds", type=int, default=0)
    parser.add_argument("--early-stop-projection-after", type=int, default=5)
    args = parser.parse_args()
    if args.interval_seconds < 0:
        parser.error("--interval-seconds must be nonnegative")

    try:
        while True:
            if args.interval_seconds:
                print("\033[2J\033[H", end="")
            print(_render(args), flush=True)
            if not args.interval_seconds:
                break
            time.sleep(args.interval_seconds)
    except KeyboardInterrupt:
        print("\n监控已退出，后台训练不受影响。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
