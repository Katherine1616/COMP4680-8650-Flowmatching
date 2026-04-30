from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))

import matplotlib.pyplot as plt

from src.dataloader import AVAILABLE_DATASETS, ToyDiffusionDataset, get_dataloader
from src.flow_matching import sample_euler, train_v_prediction
from src.model import FlowMLP


OUTPUT_DIR = ROOT / "outputs" / "part1"


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def scatter(ax: plt.Axes, points: np.ndarray, title: str) -> None:
    ax.scatter(points[:, 0], points[:, 1], s=4, alpha=0.55, linewidths=0)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])


def save_data_visualizations(num_points: int = 4096) -> None:
    for name in AVAILABLE_DATASETS:
        ds2 = ToyDiffusionDataset(name=name, dim=2)
        ds32 = ToyDiffusionDataset(name=name, dim=32)

        data2 = ds2.data[:num_points].numpy()
        data32_back = ds32.to_2d(ds32.data[:num_points].numpy())

        fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
        scatter(axes[0], data2, f"{name}: original D=2")
        scatter(axes[1], data32_back, f"{name}: D=32 back-projected")
        fig.savefig(OUTPUT_DIR / f"data_{name}_d2_vs_d32.png", dpi=200)
        plt.close(fig)

        for suffix, points, title in [
            ("d2", data2, f"{name}: original D=2"),
            ("d32_back_projected", data32_back, f"{name}: D=32 back-projected"),
        ]:
            fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
            scatter(ax, points, title)
            fig.savefig(OUTPUT_DIR / f"data_{name}_{suffix}.png", dpi=200)
            plt.close(fig)


def train_and_plot(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    config = {
        "prediction": "v",
        "loss": "v",
        "model": "5 hidden layer MLP, 256 hidden units, 128 sinusoidal time embedding",
        "optimizer": "Adam",
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "training_steps": args.train_steps,
        "sampling": "Euler ODE",
        "sampling_steps": args.sample_steps,
        "num_samples": args.num_samples,
        "device": str(device),
    }

    for name in AVAILABLE_DATASETS:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        dataset = ToyDiffusionDataset(name=name, dim=2)
        dataloader = get_dataloader(
            name=name,
            dim=2,
            batch_size=args.batch_size,
            shuffle=True,
        )
        model = FlowMLP(data_dim=2)
        print(f"Training {name} at D=2 on {device}...")
        losses = train_v_prediction(
            model,
            dataloader,
            steps=args.train_steps,
            lr=args.lr,
            device=device,
            log_every=args.log_every,
        )

        samples = sample_euler(
            model,
            num_samples=args.num_samples,
            data_dim=2,
            steps=args.sample_steps,
            device=device,
        ).numpy()
        ground_truth = dataset.data[: args.num_samples].numpy()

        fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
        scatter(axes[0], ground_truth, f"{name}: ground truth")
        scatter(axes[1], samples, f"{name}: generated")
        fig.savefig(OUTPUT_DIR / f"generated_{name}_v_pred_v_loss_d2.png", dpi=200)
        plt.close(fig)

        torch.save(model.state_dict(), OUTPUT_DIR / f"model_{name}_v_pred_v_loss_d2.pt")
        config[f"{name}_losses"] = losses

    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Part 1 flow matching experiments.")
    parser.add_argument("--train-steps", type=int, default=25_000)
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--skip-training", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_data_visualizations(num_points=args.num_samples)

    config: dict[str, object] = {}
    if not args.skip_training:
        device = choose_device()
        config = train_and_plot(args, device)

    with open(OUTPUT_DIR / "part1_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Saved Part 1 outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
