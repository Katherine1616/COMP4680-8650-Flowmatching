from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))

import matplotlib.pyplot as plt

from src.dataloader import ToyDiffusionDataset, get_dataloader
from src.flow_matching import sample_euler, train_flow_matching
from src.model import FlowMLP


OUTPUT_DIR = ROOT / "outputs" / "part3"


@dataclass(frozen=True)
class Experiment:
    name: str
    description: str
    mode: str
    train_steps: int
    hidden_dim: int = 256
    hidden_layers: int = 5
    batch_size: int = 1024
    lr: float = 1e-3
    time_schedule: str = "uniform"
    time_shift: float = 1.0


DEFAULT_EXPERIMENTS = (
    Experiment(
        name="ambient_v_default",
        description="Default full D=32 v-prediction baseline from Part 2; expected to fail.",
        mode="ambient",
        train_steps=25_000,
    ),
    Experiment(
        name="ambient_v_long",
        description="Same model as default, trained 4x longer to test whether compute alone helps.",
        mode="ambient",
        train_steps=100_000,
    ),
    Experiment(
        name="ambient_v_wide",
        description="A wider MLP inspired by RAE's width-vs-dimension argument.",
        mode="ambient",
        train_steps=50_000,
        hidden_dim=1024,
    ),
    Experiment(
        name="ambient_v_shift_high_noise",
        description="Default model with a dimension-style timestep shift toward higher noise levels.",
        mode="ambient",
        train_steps=50_000,
        time_schedule="shift_high_noise",
        time_shift=2.0,
    ),
    Experiment(
        name="ambient_v_wide_shift_high_noise",
        description="Wider model plus high-noise timestep shift, combining two RAE-inspired fixes.",
        mode="ambient",
        train_steps=50_000,
        hidden_dim=1024,
        time_schedule="shift_high_noise",
        time_shift=2.0,
    ),
    Experiment(
        name="projected_2d_v",
        description="Train v-prediction only in the intrinsic 2D subspace, then embed samples back to D=32.",
        mode="projected_2d",
        train_steps=25_000,
    ),
)


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


def save_plot(dataset: ToyDiffusionDataset, generated_32d: np.ndarray, path: Path, title: str) -> None:
    ground_truth = dataset.to_2d(dataset.data[: len(generated_32d)].numpy())
    generated = dataset.to_2d(generated_32d)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    scatter(axes[0], ground_truth, "swiss_roll D=32: ground truth")
    scatter(axes[1], generated, title)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def make_projected_loader(dataset: ToyDiffusionDataset, batch_size: int) -> DataLoader:
    projected = dataset.to_2d(dataset.data.numpy()).astype("float32")
    tensor = torch.from_numpy(projected)
    return DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=True, drop_last=False)


def train_projected_2d(
    experiment: Experiment,
    dataset: ToyDiffusionDataset,
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[FlowMLP, np.ndarray, list[float], float]:
    loader = make_projected_loader(dataset, experiment.batch_size)
    model = FlowMLP(data_dim=2, hidden_dim=experiment.hidden_dim, hidden_layers=experiment.hidden_layers)
    start = time.time()
    losses = train_flow_matching(
        model,
        loader,
        prediction_type="v",
        loss_type="v",
        steps=experiment.train_steps,
        lr=experiment.lr,
        device=device,
        log_every=args.log_every,
        t_eps=args.t_eps,
        show_progress=not args.no_progress,
        time_schedule=experiment.time_schedule,
        time_shift=experiment.time_shift,
    )
    elapsed = time.time() - start

    samples_2d = sample_euler(
        model,
        prediction_type="v",
        num_samples=args.num_samples,
        data_dim=2,
        steps=args.sample_steps,
        device=device,
        t_eps=args.t_eps,
    ).numpy()
    samples_32d = samples_2d @ dataset.P
    return model, samples_32d, losses, elapsed


def train_ambient(
    experiment: Experiment,
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[FlowMLP, np.ndarray, list[float], float]:
    loader = get_dataloader("swiss_roll", dim=32, batch_size=experiment.batch_size, shuffle=True)
    model = FlowMLP(data_dim=32, hidden_dim=experiment.hidden_dim, hidden_layers=experiment.hidden_layers)
    start = time.time()
    losses = train_flow_matching(
        model,
        loader,
        prediction_type="v",
        loss_type="v",
        steps=experiment.train_steps,
        lr=experiment.lr,
        device=device,
        log_every=args.log_every,
        t_eps=args.t_eps,
        show_progress=not args.no_progress,
        time_schedule=experiment.time_schedule,
        time_shift=experiment.time_shift,
    )
    elapsed = time.time() - start

    samples = sample_euler(
        model,
        prediction_type="v",
        num_samples=args.num_samples,
        data_dim=32,
        steps=args.sample_steps,
        device=device,
        t_eps=args.t_eps,
    ).numpy()
    return model, samples, losses, elapsed


def get_experiments(names: str) -> list[Experiment]:
    if names == "all":
        return list(DEFAULT_EXPERIMENTS)
    requested = {name.strip() for name in names.split(",") if name.strip()}
    experiments = [exp for exp in DEFAULT_EXPERIMENTS if exp.name in requested]
    missing = requested - {exp.name for exp in experiments}
    if missing:
        available = ", ".join(exp.name for exp in DEFAULT_EXPERIMENTS)
        raise ValueError(f"Unknown experiments {sorted(missing)}. Available: {available}")
    return experiments


def run_experiment(
    experiment: Experiment,
    *,
    args: argparse.Namespace,
    dataset: ToyDiffusionDataset,
    device: torch.device,
    index: int,
) -> dict[str, object]:
    fig_path = OUTPUT_DIR / f"{experiment.name}.png"
    model_path = OUTPUT_DIR / f"{experiment.name}.pt"
    if args.skip_existing and fig_path.exists() and model_path.exists():
        print(f"Skipping existing {experiment.name}")
        return {
            **asdict(experiment),
            "skipped": True,
            "figure": str(fig_path.relative_to(ROOT)),
            "checkpoint": str(model_path.relative_to(ROOT)),
        }

    seed = args.seed + index
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"Training {experiment.name} on {device}: {experiment.description}")
    if experiment.mode == "ambient":
        model, samples, losses, elapsed = train_ambient(experiment, args=args, device=device)
    elif experiment.mode == "projected_2d":
        model, samples, losses, elapsed = train_projected_2d(
            experiment, dataset, args=args, device=device
        )
    else:
        raise ValueError(f"Unknown experiment mode={experiment.mode!r}")

    save_plot(dataset, samples, fig_path, f"{experiment.name}")
    torch.save(model.state_dict(), model_path)

    return {
        **asdict(experiment),
        "seed": seed,
        "losses": losses,
        "elapsed_seconds": elapsed,
        "figure": str(fig_path.relative_to(ROOT)),
        "checkpoint": str(model_path.relative_to(ROOT)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Part 3 v-prediction rescue experiments.")
    parser.add_argument("--experiments", default="all")
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--num-samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--t-eps", type=float, default=1e-2)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = ToyDiffusionDataset("swiss_roll", dim=32)
    device = choose_device()
    experiments = get_experiments(args.experiments)

    config = {
        "dataset": "swiss_roll",
        "dim": 32,
        "prediction_type": "v",
        "loss_type": "v",
        "sample_steps": args.sample_steps,
        "num_samples": args.num_samples,
        "t_eps": args.t_eps,
        "device": str(device),
        "experiments": [],
    }
    jsonl_path = OUTPUT_DIR / "part3_results.jsonl"
    jsonl_path.write_text("", encoding="utf-8")

    for index, experiment in enumerate(experiments):
        result = run_experiment(
            experiment,
            args=args,
            dataset=dataset,
            device=device,
            index=index,
        )
        config["experiments"].append(result)
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")

    with open(OUTPUT_DIR / "part3_results.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Saved Part 3 outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
