from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))

import matplotlib.pyplot as plt

from src.dataloader import AVAILABLE_DATASETS, ToyDiffusionDataset, get_dataloader
from src.flow_matching import sample_euler, sample_mean_flow, train_mean_flow
from src.model import FlowMLP, MeanFlowMLP


OUTPUT_BASE_DIR = ROOT / "outputs" / "part4" / "meanflow"
DEFAULT_MEANFLOW_STEPS = (1, 2, 5)
DEFAULT_FM_STEPS = (1, 2, 5, 10, 20, 50)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def scatter(ax: plt.Axes, points: np.ndarray, title: str) -> None:
    ax.scatter(points[:, 0], points[:, 1], s=4, alpha=0.55, linewidths=0)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_part2_model(dataset: str, *, dim: int, prediction: str, loss: str, device: torch.device) -> FlowMLP:
    path = ROOT / "outputs" / "part2" / f"model_{dataset}_d{dim}_{prediction}_pred_{loss}_loss.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing Part 2 checkpoint: {path}")
    model = FlowMLP(data_dim=dim)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model


def save_pair_figure(
    *,
    dataset: ToyDiffusionDataset,
    samples: np.ndarray,
    step_count: int,
    out_path: Path,
    num_samples: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    gt = dataset.to_2d(dataset.data[:num_samples].numpy())
    scatter(axes[0], gt, "ground truth")
    scatter(axes[1], dataset.to_2d(samples), f"MeanFlow {step_count} step(s)")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_meanflow_grid(
    *,
    dataset: ToyDiffusionDataset,
    samples_by_step: dict[int, np.ndarray],
    out_path: Path,
    num_samples: int,
) -> None:
    cols = 1 + len(samples_by_step)
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4), constrained_layout=True)
    axes = np.atleast_1d(axes)
    gt = dataset.to_2d(dataset.data[:num_samples].numpy())
    scatter(axes[0], gt, "ground truth")
    for ax, steps in zip(axes[1:], samples_by_step):
        scatter(ax, dataset.to_2d(samples_by_step[steps]), f"MeanFlow {steps} step(s)")
    fig.suptitle(f"MeanFlow: {dataset.name} D={dataset.dim}", fontsize=16)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_comparison_grid(
    *,
    dataset: ToyDiffusionDataset,
    meanflow_samples: dict[int, np.ndarray],
    fm_samples: dict[int, np.ndarray],
    out_path: Path,
    num_samples: int,
) -> None:
    rows = 3
    cols = max(1 + len(meanflow_samples), 1 + len(fm_samples))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 12), constrained_layout=True)
    gt = dataset.to_2d(dataset.data[:num_samples].numpy())

    for ax in axes.ravel():
        ax.axis("off")

    scatter(axes[0, 0], gt, "ground truth")
    axes[0, 0].axis("on")
    for col, steps in enumerate(meanflow_samples, start=1):
        scatter(
            axes[0, col],
            dataset.to_2d(meanflow_samples[steps]),
            f"MeanFlow {steps} step(s)",
        )
        axes[0, col].axis("on")

    scatter(axes[1, 0], gt, "ground truth")
    axes[1, 0].axis("on")
    for col, steps in enumerate(fm_samples, start=1):
        scatter(
            axes[1, col],
            dataset.to_2d(fm_samples[steps]),
            f"FM {steps} step(s)",
        )
        axes[1, col].axis("on")

    scatter(axes[2, 0], gt, "ground truth")
    axes[2, 0].axis("on")
    for col, steps in enumerate((10, 20, 50), start=1):
        if steps not in fm_samples:
            continue
        scatter(
            axes[2, col],
            dataset.to_2d(fm_samples[steps]),
            f"FM {steps} step(s)",
        )
        axes[2, col].axis("on")

    fig.suptitle(f"MeanFlow vs standard flow matching: {dataset.name} D={dataset.dim}", fontsize=16)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Part 4.2 MeanFlow experiments.")
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--train-steps", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--hidden-layers", type=int, default=5)
    parser.add_argument("--mean-ratio", type=float, default=0.5)
    parser.add_argument("--t-eps", type=float, default=1e-2)
    parser.add_argument("--num-samples", type=int, default=4096)
    parser.add_argument("--meanflow-steps", default=",".join(str(x) for x in DEFAULT_MEANFLOW_STEPS))
    parser.add_argument("--fm-steps", default=",".join(str(x) for x in DEFAULT_FM_STEPS))
    parser.add_argument("--fm-prediction", default="x", choices=("x", "v"))
    parser.add_argument("--fm-loss", default="v", choices=("x", "v"))
    parser.add_argument("--seed", type=int, default=4200)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = OUTPUT_BASE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device()
    datasets = AVAILABLE_DATASETS if args.datasets == "all" else tuple(args.datasets.split(","))
    meanflow_steps = parse_ints(args.meanflow_steps)
    fm_steps = parse_ints(args.fm_steps)

    result = {
        "paper": "MeanFlow, arXiv:2505.13447",
        "dim": args.dim,
        "train_steps": args.train_steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "hidden_layers": args.hidden_layers,
        "mean_ratio": args.mean_ratio,
        "t_eps": args.t_eps,
        "num_samples": args.num_samples,
        "meanflow_steps": meanflow_steps,
        "fm_baseline": {
            "prediction": args.fm_prediction,
            "loss": args.fm_loss,
            "steps": fm_steps,
        },
        "device": str(device),
        "datasets": {},
    }

    for dataset_index, dataset_name in enumerate(datasets):
        seed = args.seed + dataset_index
        seed_all(seed)
        dataset = ToyDiffusionDataset(dataset_name, dim=args.dim)
        dataloader = get_dataloader(dataset_name, dim=args.dim, batch_size=args.batch_size)
        model = MeanFlowMLP(
            data_dim=args.dim,
            hidden_dim=args.hidden_dim,
            hidden_layers=args.hidden_layers,
        )
        ckpt_path = output_dir / f"meanflow_{dataset_name}_d{args.dim}.pt"

        losses: list[float] = []
        train_seconds = 0.0
        if args.skip_existing and ckpt_path.exists():
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.to(device)
        else:
            start_time = time.perf_counter()
            losses = train_mean_flow(
                model,
                dataloader,
                steps=args.train_steps,
                lr=args.lr,
                device=device,
                log_every=max(1, args.train_steps // 25),
                t_eps=args.t_eps,
                mean_ratio=args.mean_ratio,
                show_progress=not args.no_progress,
            )
            train_seconds = time.perf_counter() - start_time
            torch.save(model.state_dict(), ckpt_path)

        meanflow_samples: dict[int, np.ndarray] = {}
        figure_paths: list[str] = []
        for steps in meanflow_steps:
            seed_all(seed + steps)
            samples = sample_mean_flow(
                model,
                num_samples=args.num_samples,
                data_dim=args.dim,
                steps=steps,
                device=device,
            ).numpy()
            meanflow_samples[steps] = samples
            fig_path = output_dir / f"meanflow_{dataset_name}_d{args.dim}_{steps}_steps.png"
            save_pair_figure(
                dataset=dataset,
                samples=samples,
                step_count=steps,
                out_path=fig_path,
                num_samples=args.num_samples,
            )
            figure_paths.append(str(fig_path.relative_to(ROOT)))

        meanflow_grid_path = output_dir / f"meanflow_{dataset_name}_d{args.dim}_steps_grid.png"
        save_meanflow_grid(
            dataset=dataset,
            samples_by_step=meanflow_samples,
            out_path=meanflow_grid_path,
            num_samples=args.num_samples,
        )

        fm_samples: dict[int, np.ndarray] = {}
        try:
            fm_model = load_part2_model(
                dataset_name,
                dim=args.dim,
                prediction=args.fm_prediction,
                loss=args.fm_loss,
                device=device,
            )
            for steps in fm_steps:
                seed_all(seed + 1000 + steps)
                fm_samples[steps] = sample_euler(
                    fm_model,
                    prediction_type=args.fm_prediction,
                    num_samples=args.num_samples,
                    data_dim=args.dim,
                    steps=steps,
                    device=device,
                    t_eps=args.t_eps,
                ).numpy()
        except FileNotFoundError as exc:
            print(exc)

        comparison_path = output_dir / f"comparison_{dataset_name}_d{args.dim}.png"
        if fm_samples:
            save_comparison_grid(
                dataset=dataset,
                meanflow_samples=meanflow_samples,
                fm_samples=fm_samples,
                out_path=comparison_path,
                num_samples=args.num_samples,
            )

        result["datasets"][dataset_name] = {
            "checkpoint": str(ckpt_path.relative_to(ROOT)),
            "train_seconds": train_seconds,
            "losses": losses,
            "meanflow_grid": str(meanflow_grid_path.relative_to(ROOT)),
            "meanflow_figures": figure_paths,
            "comparison_grid": str(comparison_path.relative_to(ROOT)) if fm_samples else None,
        }

    with open(output_dir / "meanflow_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Saved Part 4.2 MeanFlow outputs to {output_dir}")


if __name__ == "__main__":
    main()
