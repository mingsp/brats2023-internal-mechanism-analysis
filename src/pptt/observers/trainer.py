from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch

from pptt.observers.cache import ObserverCache
from pptt.observers.linear import LinearObserver
from pptt.observers.objective import weighted_distillation_objective


@dataclass(frozen=True)
class ObserverTrainingConfig:
    l2: float = 1e-3
    temperature: float = 1.0
    max_iter: int = 100
    tolerance_grad: float = 1e-7
    tolerance_change: float = 1e-12
    history_size: int = 50

    def validate(self) -> None:
        if self.l2 <= 0 or self.temperature <= 0:
            raise ValueError("l2 and temperature must be strictly positive")
        if self.max_iter <= 0 or self.history_size <= 0:
            raise ValueError("max_iter and history_size must be positive")
        if self.tolerance_grad <= 0 or self.tolerance_change <= 0:
            raise ValueError("optimizer tolerances must be positive")


@dataclass(frozen=True)
class OptimizationReport:
    seed: int
    initial_loss: float
    final_loss: float
    final_data_loss: float
    final_regularization: float
    final_gradient_norm: float
    relative_loss_change: float
    optimizer_iterations: int
    closure_evaluations: int

    def to_dict(self) -> dict:
        return asdict(self)


def _gradient_norm(observer: LinearObserver) -> float:
    squared = 0.0
    for parameter in observer.parameters():
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().square().sum().item())
    return squared**0.5


def train_observer(
    cache: ObserverCache,
    *,
    config: ObserverTrainingConfig,
    seed: int,
    device: str | torch.device = "cpu",
) -> tuple[LinearObserver, OptimizationReport]:
    config.validate()
    torch.manual_seed(int(seed))
    observer = LinearObserver(cache.in_channels, cache.num_classes).to(device)
    with torch.no_grad():
        torch.nn.init.normal_(observer.projection.weight, mean=0.0, std=0.01)
        torch.nn.init.zeros_(observer.projection.bias)
    features = torch.from_numpy(cache.features.astype(np.float32)).to(device)
    targets = torch.from_numpy(cache.target_probabilities).to(device)
    weights = torch.from_numpy(cache.pixel_weights).to(device)

    def objective() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return weighted_distillation_objective(
            observer,
            features,
            targets,
            weights,
            temperature=config.temperature,
            l2=config.l2,
        )

    with torch.no_grad():
        initial_loss = float(objective()[0].item())
    optimizer = torch.optim.LBFGS(
        observer.parameters(),
        lr=1.0,
        max_iter=config.max_iter,
        max_eval=max(config.max_iter * 2, 20),
        tolerance_grad=config.tolerance_grad,
        tolerance_change=config.tolerance_change,
        history_size=config.history_size,
        line_search_fn="strong_wolfe",
    )
    closure_evaluations = 0

    def closure() -> torch.Tensor:
        nonlocal closure_evaluations
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = objective()
        loss.backward()
        closure_evaluations += 1
        return loss

    optimizer.step(closure)
    optimizer.zero_grad(set_to_none=True)
    final_loss_tensor, data_loss, regularization = objective()
    final_loss_tensor.backward()
    final_gradient_norm = _gradient_norm(observer)
    final_loss = float(final_loss_tensor.detach().item())
    optimizer_state = optimizer.state[next(iter(observer.parameters()))]
    iterations = int(optimizer_state.get("n_iter", 0))
    relative_change = (initial_loss - final_loss) / max(abs(initial_loss), 1e-12)
    observer.eval()
    report = OptimizationReport(
        seed=int(seed),
        initial_loss=initial_loss,
        final_loss=final_loss,
        final_data_loss=float(data_loss.detach().item()),
        final_regularization=float(regularization.detach().item()),
        final_gradient_norm=final_gradient_norm,
        relative_loss_change=float(relative_change),
        optimizer_iterations=iterations,
        closure_evaluations=closure_evaluations,
    )
    return observer, report
