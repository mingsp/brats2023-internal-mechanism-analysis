from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from pptt.training.engine import (
    TrainingProgress,
    load_training_checkpoint,
    save_training_checkpoint,
    seed_everything,
    train_one_epoch,
)


class _TinyDataset(Dataset):
    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(11)
        self.images = torch.randn(6, 4, 8, 8, generator=generator)
        self.labels = torch.randint(0, 4, (6, 8, 8), generator=generator)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        return {"image": self.images[index], "label": self.labels[index]}


class _TinySegmentationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(4, 8, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.Conv2d(8, 4, 1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layers(image)


def _components():
    model = _TinySegmentationModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=4,
        eta_min=1e-6,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    return model, optimizer, scheduler, scaler


def test_resume_reproduces_the_next_epoch_exactly(tmp_path: Path):
    seed_everything(123)
    loader = DataLoader(_TinyDataset(), batch_size=2, shuffle=False)
    model, optimizer, scheduler, scaler = _components()
    first = train_one_epoch(
        model,
        loader,
        optimizer,
        scaler,
        device=torch.device("cpu"),
    )
    scheduler.step()
    checkpoint = tmp_path / "latest.pt"
    progress = TrainingProgress(completed_epoch=0, global_step=first.optimizer_steps)
    save_training_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        progress=progress,
        metadata={"purpose": "resume-test"},
    )

    continuous = train_one_epoch(
        model,
        loader,
        optimizer,
        scaler,
        device=torch.device("cpu"),
    )
    continuous_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    seed_everything(999)
    restored_model, restored_optimizer, restored_scheduler, restored_scaler = _components()
    restored_progress, metadata = load_training_checkpoint(
        checkpoint,
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        scaler=restored_scaler,
    )
    resumed = train_one_epoch(
        restored_model,
        loader,
        restored_optimizer,
        restored_scaler,
        device=torch.device("cpu"),
    )

    assert restored_progress == progress
    assert metadata == {"purpose": "resume-test"}
    assert resumed.loss == continuous.loss
    for key, expected in continuous_state.items():
        torch.testing.assert_close(restored_model.state_dict()[key], expected, rtol=0, atol=0)
