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

from src.dataloader import AVAILABLE_DATASETS, ToyDiffusionDataset
from src.flow_matching import sample_euler
from src.model import FlowMLP


OUTPUT_BASE_DIR = ROOT / "outputs" / "part4"
DEFAULT_STEP_COUNTS = (1, 2, 5, 10, 20, 50, 100, 200)


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


def load_part2_model(dataset: str, *, dim: int, prediction: str, loss: str, device: torch.device) -> FlowMLP:
    path = ROOT / "outputs" / "part2" / f"model_{dataset}_d{dim}_{prediction}_pred_{loss}_loss.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing Part 2 checkpoint: {path}")
    model = FlowMLP(data_dim=dim)
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model


def save_dataset_grid(
    *,
    dataset: ToyDiffusionDataset,
    samples_by_step: dict[int, np.ndarray],
    out_path: Path,
    num_samples: int,
) -> None:
    step_counts = list(samples_by_step)
    cols = 3
    rows = 3
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12), constrained_layout=True)
    axes = axes.ravel()

    gt = dataset.to_2d(dataset.data[:num_samples].numpy())
    scatter(axes[0], gt, "ground truth")
    for ax, steps in zip(axes[1:], step_counts):
        scatter(ax, dataset.to_2d(samples_by_step[steps]), f"{steps} Euler step(s)")

    for ax in axes[1 + len(step_counts):]:
        ax.axis("off")

    fig.suptitle(f"Sampling efficiency: {dataset.name} D={dataset.dim}", fontsize=18)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def parse_step_counts(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Part 4.1 sampling efficiency experiments.")
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--prediction", default="x", choices=("x", "v"))
    parser.add_argument("--loss", default="v", choices=("x", "v"))
    parser.add_argument("--step-counts", default=",".join(str(x) for x in DEFAULT_STEP_COUNTS))
    parser.add_argument("--num-samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=4100)
    parser.add_argument("--t-eps", type=float, default=1e-2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = OUTPUT_BASE_DIR / f"sampling_efficiency_{args.prediction}_pred_{args.loss}_loss"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device()
    datasets = AVAILABLE_DATASETS if args.datasets == "all" else tuple(args.datasets.split(","))
    step_counts = parse_step_counts(args.step_counts)

    config = {
        "source": "Part 2 best model",
        "dim": args.dim,
        "prediction": args.prediction,
        "loss": args.loss,
        "step_counts": step_counts,
        "num_samples": args.num_samples,
        "seed": args.seed,
        "device": str(device),
        "figures": {},
    }

    for dataset_index, dataset_name in enumerate(datasets):
        dataset = ToyDiffusionDataset(dataset_name, dim=args.dim)
        model = load_part2_model(
            dataset_name,
            dim=args.dim,
            prediction=args.prediction,
            loss=args.loss,
            device=device,
        )

        samples_by_step: dict[int, np.ndarray] = {}
        for steps in step_counts:
            torch.manual_seed(args.seed + dataset_index)
            np.random.seed(args.seed + dataset_index)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed + dataset_index)
            samples = sample_euler(
                model,
                prediction_type=args.prediction,
                num_samples=args.num_samples,
                data_dim=args.dim,
                steps=steps,
                device=device,
                t_eps=args.t_eps,
            ).numpy()
            samples_by_step[steps] = samples

            single_path = output_dir / f"{dataset_name}_d{args.dim}_{steps}_steps.png"
            fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
            scatter(axes[0], dataset.to_2d(dataset.data[: args.num_samples].numpy()), "ground truth")
            scatter(axes[1], dataset.to_2d(samples), f"{steps} Euler step(s)")
            fig.savefig(single_path, dpi=200)
            plt.close(fig)

        grid_path = output_dir / f"{dataset_name}_d{args.dim}_steps_grid.png"
        save_dataset_grid(
            dataset=dataset,
            samples_by_step=samples_by_step,
            out_path=grid_path,
            num_samples=args.num_samples,
        )
        config["figures"][dataset_name] = {
            "grid": str(grid_path.relative_to(ROOT)),
            "individual": [
                str((output_dir / f"{dataset_name}_d{args.dim}_{steps}_steps.png").relative_to(ROOT))
                for steps in step_counts
            ],
        }

    with open(output_dir / "sampling_efficiency_results.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Saved Part 4.1 outputs to {output_dir}")


if __name__ == "__main__":
    main()
