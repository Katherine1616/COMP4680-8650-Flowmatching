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

from src.dataloader import AVAILABLE_DATASETS, AVAILABLE_DIMS, ToyDiffusionDataset, get_dataloader
from src.flow_matching import sample_euler, train_flow_matching
from src.model import FlowMLP


OUTPUT_DIR = ROOT / "outputs" / "part2"
PREDICTION_TYPES = ("x", "v")
LOSS_TYPES = ("x", "v")


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_csv(value: str, allowed: tuple) -> list:
    if value == "all":
        return list(allowed)
    items = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in items if item not in allowed]
    if unknown:
        raise ValueError(f"Unknown values {unknown}; choose from {allowed} or 'all'.")
    return items


def parse_int_csv(value: str, allowed: tuple[int, ...]) -> list[int]:
    if value == "all":
        return list(allowed)
    items = [int(item.strip()) for item in value.split(",") if item.strip()]
    unknown = [item for item in items if item not in allowed]
    if unknown:
        raise ValueError(f"Unknown dimensions {unknown}; choose from {allowed} or 'all'.")
    return items


def scatter(ax: plt.Axes, points: np.ndarray, title: str) -> None:
    ax.scatter(points[:, 0], points[:, 1], s=4, alpha=0.55, linewidths=0)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])


def experiment_name(dataset: str, dim: int, prediction: str, loss: str) -> str:
    return f"{dataset}_d{dim}_{prediction}_pred_{loss}_loss"


def save_comparison_plot(
    *,
    dataset: ToyDiffusionDataset,
    generated: np.ndarray,
    out_path: Path,
    title_suffix: str,
    num_samples: int,
) -> None:
    ground_truth = dataset.data[:num_samples].numpy()
    ground_truth_2d = dataset.to_2d(ground_truth)
    generated_2d = dataset.to_2d(generated)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    scatter(axes[0], ground_truth_2d, f"{dataset.name} D={dataset.dim}: ground truth")
    scatter(axes[1], generated_2d, f"{dataset.name} D={dataset.dim}: {title_suffix}")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run_experiment(
    *,
    dataset_name: str,
    dim: int,
    prediction_type: str,
    loss_type: str,
    args: argparse.Namespace,
    device: torch.device,
    index: int,
) -> dict[str, object]:
    name = experiment_name(dataset_name, dim, prediction_type, loss_type)
    fig_path = OUTPUT_DIR / f"generated_{name}.png"
    model_path = OUTPUT_DIR / f"model_{name}.pt"

    if args.skip_existing and fig_path.exists() and model_path.exists():
        print(f"Skipping existing {name}")
        return {
            "name": name,
            "dataset": dataset_name,
            "dim": dim,
            "prediction_type": prediction_type,
            "loss_type": loss_type,
            "skipped": True,
            "figure": str(fig_path.relative_to(ROOT)),
            "checkpoint": str(model_path.relative_to(ROOT)),
        }

    seed = args.seed + index
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dataset = ToyDiffusionDataset(name=dataset_name, dim=dim)
    dataloader = get_dataloader(
        name=dataset_name,
        dim=dim,
        batch_size=args.batch_size,
        shuffle=True,
    )
    model = FlowMLP(data_dim=dim)

    print(f"Training {name} on {device}...")
    start = time.time()
    losses = train_flow_matching(
        model,
        dataloader,
        prediction_type=prediction_type,
        loss_type=loss_type,
        steps=args.train_steps,
        lr=args.lr,
        device=device,
        log_every=args.log_every,
        t_eps=args.t_eps,
        show_progress=not args.no_progress,
    )
    elapsed = time.time() - start

    samples = sample_euler(
        model,
        prediction_type=prediction_type,
        num_samples=args.num_samples,
        data_dim=dim,
        steps=args.sample_steps,
        device=device,
        t_eps=args.t_eps,
    ).numpy()

    save_comparison_plot(
        dataset=dataset,
        generated=samples,
        out_path=fig_path,
        title_suffix=f"{prediction_type}-pred/{loss_type}-loss",
        num_samples=args.num_samples,
    )
    torch.save(model.state_dict(), model_path)

    return {
        "name": name,
        "dataset": dataset_name,
        "dim": dim,
        "prediction_type": prediction_type,
        "loss_type": loss_type,
        "seed": seed,
        "losses": losses,
        "elapsed_seconds": elapsed,
        "figure": str(fig_path.relative_to(ROOT)),
        "checkpoint": str(model_path.relative_to(ROOT)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Part 2 parameterization experiments.")
    parser.add_argument("--datasets", default="all", help="Comma-separated datasets or 'all'.")
    parser.add_argument("--dims", default="all", help="Comma-separated dimensions or 'all'.")
    parser.add_argument("--predictions", default="all", help="Comma-separated prediction types or 'all'.")
    parser.add_argument("--losses", default="all", help="Comma-separated loss types or 'all'.")
    parser.add_argument("--train-steps", type=int, default=25_000)
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--t-eps", type=float, default=1e-2)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-experiments", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = parse_csv(args.datasets, AVAILABLE_DATASETS)
    dims = parse_int_csv(args.dims, AVAILABLE_DIMS)
    predictions = parse_csv(args.predictions, PREDICTION_TYPES)
    losses = parse_csv(args.losses, LOSS_TYPES)
    device = choose_device()

    config = {
        "model": "5 hidden layer MLP, 256 hidden units, 128 sinusoidal time embedding",
        "optimizer": "Adam",
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "training_steps": args.train_steps,
        "sampling": "Euler ODE from t=1 to t=0",
        "sampling_steps": args.sample_steps,
        "num_samples": args.num_samples,
        "t_eps": args.t_eps,
        "device": str(device),
        "experiments": [],
    }

    results_path = OUTPUT_DIR / "part2_results.json"
    jsonl_path = OUTPUT_DIR / "part2_results.jsonl"
    jsonl_path.write_text("", encoding="utf-8")

    experiment_index = 0
    for dim in dims:
        for dataset_name in datasets:
            for prediction_type in predictions:
                for loss_type in losses:
                    if args.max_experiments is not None and experiment_index >= args.max_experiments:
                        break
                    result = run_experiment(
                        dataset_name=dataset_name,
                        dim=dim,
                        prediction_type=prediction_type,
                        loss_type=loss_type,
                        args=args,
                        device=device,
                        index=experiment_index,
                    )
                    config["experiments"].append(result)
                    with open(jsonl_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(result) + "\n")
                    experiment_index += 1
                if args.max_experiments is not None and experiment_index >= args.max_experiments:
                    break
            if args.max_experiments is not None and experiment_index >= args.max_experiments:
                break
        if args.max_experiments is not None and experiment_index >= args.max_experiments:
            break

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Saved Part 2 outputs to {OUTPUT_DIR}")
    print(f"Ran {experiment_index} experiments.")


if __name__ == "__main__":
    main()
