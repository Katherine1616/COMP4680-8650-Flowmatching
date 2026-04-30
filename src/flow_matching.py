from collections.abc import Iterable
from typing import Literal

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import trange

PredictionType = Literal["x", "v"]
LossType = Literal["x", "v"]


def cycle(loader: DataLoader) -> Iterable[torch.Tensor]:
    while True:
        for batch in loader:
            yield batch


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
        x = next(batches).to(device=device, dtype=torch.float32)
        t = t_eps + (1 - 2 * t_eps) * torch.rand(x.shape[0], 1, device=device)
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
