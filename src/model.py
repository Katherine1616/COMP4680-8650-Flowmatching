import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("Embedding dimension must be even.")
        half_dim = dim // 2
        frequencies = torch.exp(
            -torch.arange(half_dim, dtype=torch.float32) * math.log(10000) / (half_dim - 1)
        )
        self.register_buffer("frequencies", frequencies)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        angles = t * self.frequencies[None, :]
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class FlowMLP(nn.Module):
    def __init__(
        self,
        data_dim: int,
        time_embed_dim: int = 128,
        hidden_dim: int = 256,
        hidden_layers: int = 5,
    ):
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(time_embed_dim)

        layers: list[nn.Module] = []
        in_dim = data_dim + time_embed_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, data_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_embed = self.time_embedding(t.to(dtype=z.dtype))
        return self.net(torch.cat([z, t_embed], dim=-1))


class MeanFlowMLP(nn.Module):
    def __init__(
        self,
        data_dim: int,
        time_embed_dim: int = 128,
        hidden_dim: int = 256,
        hidden_layers: int = 5,
    ):
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(time_embed_dim)
        self.horizon_embedding = SinusoidalTimeEmbedding(time_embed_dim)

        layers: list[nn.Module] = []
        in_dim = data_dim + 2 * time_embed_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, data_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        t_embed = self.time_embedding(t.to(dtype=z.dtype))
        h_embed = self.horizon_embedding(h.to(dtype=z.dtype))
        return self.net(torch.cat([z, t_embed, h_embed], dim=-1))
