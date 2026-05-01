from collections.abc import Iterable
from typing import Literal

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import trange

PredictionType = Literal["x", "v"]
LossType = Literal["x", "v"]
TimeSchedule = Literal["uniform", "shift_high_noise", "shift_low_noise"]


def cycle(loader: DataLoader) -> Iterable[torch.Tensor]:
    while True:
        for batch in loader:
            yield batch


def sample_training_times(
    batch_size: int,
    *,
    device: torch.device | str,
    t_eps: float,
    schedule: TimeSchedule = "uniform",
    shift: float = 1.0,
) -> torch.Tensor:
    u = t_eps + (1 - 2 * t_eps) * torch.rand(batch_size, 1, device=device)
    if schedule == "uniform":
        return u

    logits = torch.logit(u)
    if schedule == "shift_high_noise":
        shifted = torch.sigmoid(logits + shift)
    elif schedule == "shift_low_noise":
        shifted = torch.sigmoid(logits - shift)
    else:
        raise ValueError(f"Unknown time schedule={schedule!r}")

    return shifted.clamp(t_eps, 1 - t_eps)


def predictions_from_output(
    output: torch.Tensor,
    z_t: torch.Tensor,
    t: torch.Tensor,
    prediction_type: PredictionType,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    t_safe = t.clamp(min=eps)

    if prediction_type == "x":
        pred_x = output
        pred_v = (z_t - pred_x) / t_safe
    elif prediction_type == "v":
        pred_v = output
        pred_x = z_t - t * pred_v
    else:
        raise ValueError(f"Unknown prediction_type={prediction_type!r}")

    return pred_x, pred_v


def train_flow_matching(
    model: nn.Module,
    dataloader: DataLoader,
    *,
    prediction_type: PredictionType = "v",
    loss_type: LossType = "v",
    steps: int = 25_000,
    lr: float = 1e-3,
    device: torch.device | str = "cpu",
    log_every: int = 1_000,
    t_eps: float = 1e-4,
    show_progress: bool = True,
    time_schedule: TimeSchedule = "uniform",
    time_shift: float = 1.0,
) -> list[float]:
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    batches = cycle(dataloader)
    losses: list[float] = []

    progress = trange(
        steps,
        desc=f"{prediction_type}-pred/{loss_type}-loss",
        leave=False,
        disable=not show_progress,
    )
    for step in progress:
        batch = next(batches)
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        x = batch.to(device=device, dtype=torch.float32)
        t = sample_training_times(
            x.shape[0],
            device=device,
            t_eps=t_eps,
            schedule=time_schedule,
            shift=time_shift,
        )
        eps = torch.randn_like(x)
        z_t = (1 - t) * x + t * eps
        target_v = eps - x

        output = model(z_t, t)
        pred_x, pred_v = predictions_from_output(output, z_t, t, prediction_type, eps=t_eps)
        if loss_type == "x":
            loss = mse(pred_x, x)
        elif loss_type == "v":
            loss = mse(pred_v, target_v)
        else:
            raise ValueError(f"Unknown loss_type={loss_type!r}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 0 or (step + 1) % log_every == 0 or step + 1 == steps:
            losses.append(float(loss.detach().cpu()))
            progress.set_postfix(loss=f"{losses[-1]:.4f}")

    return losses


def train_v_prediction(
    model: nn.Module,
    dataloader: DataLoader,
    *,
    steps: int = 25_000,
    lr: float = 1e-3,
    device: torch.device | str = "cpu",
    log_every: int = 1_000,
) -> list[float]:
    return train_flow_matching(
        model,
        dataloader,
        prediction_type="v",
        loss_type="v",
        steps=steps,
        lr=lr,
        device=device,
        log_every=log_every,
    )


def train_mean_flow(
    model: nn.Module,
    dataloader: DataLoader,
    *,
    steps: int = 50_000,
    lr: float = 1e-3,
    device: torch.device | str = "cpu",
    log_every: int = 1_000,
    t_eps: float = 1e-4,
    mean_ratio: float = 0.5,
    show_progress: bool = True,
) -> list[float]:
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    batches = cycle(dataloader)
    losses: list[float] = []

    progress = trange(
        steps,
        desc="meanflow",
        leave=False,
        disable=not show_progress,
    )
    for step in progress:
        batch = next(batches)
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        x = batch.to(device=device, dtype=torch.float32)
        batch_size = x.shape[0]
        t = sample_training_times(batch_size, device=device, t_eps=t_eps)
        eps = torch.randn_like(x)
        v = eps - x
        z_t = x + t * v

        mean_mask = torch.rand(batch_size, 1, device=device) < mean_ratio
        r = torch.rand(batch_size, 1, device=device) * t
        h = torch.where(mean_mask, t - r, torch.zeros_like(t))

        ones = torch.ones_like(t)

        def mean_velocity(z_in: torch.Tensor, t_in: torch.Tensor, h_in: torch.Tensor) -> torch.Tensor:
            return model(z_in, t_in, h_in)

        u, du_dt = torch.func.jvp(mean_velocity, (z_t, t, h), (v, ones, ones))
        target = (v - h * du_dt).detach()
        loss = mse(u, target)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 0 or (step + 1) % log_every == 0 or step + 1 == steps:
            losses.append(float(loss.detach().cpu()))
            progress.set_postfix(loss=f"{losses[-1]:.4f}")

    return losses


@torch.no_grad()
def sample_mean_flow(
    model: nn.Module,
    *,
    num_samples: int,
    data_dim: int,
    steps: int = 1,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    model.to(device)
    model.eval()
    z = torch.randn(num_samples, data_dim, device=device)
    dt = 1.0 / steps

    for i in range(steps):
        t_value = 1.0 - i * dt
        h_value = min(dt, t_value)
        t = torch.full((num_samples, 1), t_value, device=device)
        h = torch.full((num_samples, 1), h_value, device=device)
        u = model(z, t, h)
        z = z - h * u

    return z.cpu()


@torch.no_grad()
def sample_euler(
    model: nn.Module,
    *,
    prediction_type: PredictionType = "v",
    num_samples: int,
    data_dim: int,
    steps: int = 50,
    device: torch.device | str = "cpu",
    t_eps: float = 1e-4,
) -> torch.Tensor:
    model.to(device)
    model.eval()
    z = torch.randn(num_samples, data_dim, device=device)
    dt = -1.0 / steps

    for i in range(steps):
        t_value = 1.0 - i / steps
        t = torch.full((num_samples, 1), t_value, device=device)
        output = model(z, t)
        _, v = predictions_from_output(output, z, t, prediction_type, eps=t_eps)
        z = z + v * dt

    return z.cpu()
