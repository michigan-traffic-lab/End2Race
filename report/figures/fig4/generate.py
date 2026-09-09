import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import NullLocator
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from imitation.model import End2Race
from imitation.train import BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, SequenceDataset


SPEED_LOSS_WEIGHT = 0.05
GRADIENT_CLIP_NORM = 1.0


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate the training-curves figure")
    parser.add_argument("--train_bc", action="store_true")
    parser.add_argument("--dataset_dir", type=Path, default=Path("dataset/success"))
    parser.add_argument(
        "--bc_metrics",
        type=Path,
        default=Path("checkpoint/bc_metrics.csv"),
    )
    parser.add_argument(
        "--ppo_metrics",
        type=Path,
        default=Path("checkpoint/metrics.jsonl"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("report/figures/fig4"),
    )
    return parser.parse_args()


def train_bc(dataset_dir, metrics_path):
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SequenceDataset(dataset_dir)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    if len(loader) != 1:
        raise ValueError(
            f"Expected the BC dataset to fit in one batch of {BATCH_SIZE}, found {len(loader)} batches"
        )
    lidar, speed, actions = next(iter(loader))
    lidar = lidar.to(device, non_blocking=True)
    speed = speed.to(device, non_blocking=True)
    actions = actions.to(device, non_blocking=True)
    model = End2Race().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    records = []
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        predicted, _ = model(lidar, speed)
        steering_loss = F.mse_loss(predicted[..., 0], actions[..., 0])
        weighted_speed_loss = SPEED_LOSS_WEIGHT * F.mse_loss(
            predicted[..., 1], actions[..., 1]
        )
        total_loss = steering_loss + weighted_speed_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=GRADIENT_CLIP_NORM
        )
        optimizer.step()

        record = {
            "epoch": epoch,
            "total_loss": total_loss.item(),
            "steering_loss": steering_loss.item(),
            "weighted_speed_loss": weighted_speed_loss.item(),
        }
        records.append(record)
        if epoch == 1 or epoch % 25 == 0:
            print(
                f"Epoch {epoch}/{NUM_EPOCHS}: "
                f"total={record['total_loss']:.6f}, "
                f"steering={record['steering_loss']:.6f}, "
                f"weighted_speed={record['weighted_speed_loss']:.6f}"
            )

    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


def read_bc_metrics(path):
    with path.open(encoding="utf-8") as stream:
        return [
            {name: int(value) if name == "epoch" else float(value) for name, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def read_ppo_metrics(path):
    records = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record["epoch"] > NUM_EPOCHS:
                break
            records.append(record)
    if len(records) != NUM_EPOCHS:
        raise ValueError(f"Expected {NUM_EPOCHS} PPO epochs in {path}, found {len(records)}")
    return records


def plot_figures(bc_records, ppo_records, output_dir):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Nimbus Roman"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "font.size": 6,
            "axes.labelsize": 6,
            "axes.titlesize": 6,
            "legend.fontsize": 4.8,
            "xtick.labelsize": 5.5,
            "ytick.labelsize": 5.5,
        }
    )
    colors = {
        "blue": "#0072B2",
        "orange": "#D55E00",
        "green": "#009E73",
    }
    bc_figure, bc_axis = plt.subplots(figsize=(1.75, 1.35))
    ppo_figure, ppo_axis = plt.subplots(figsize=(1.75, 1.35))

    bc_epochs = [record["epoch"] for record in bc_records]
    bc_axis.plot(
        bc_epochs,
        [record["total_loss"] for record in bc_records],
        color=colors["blue"],
        linewidth=1.2,
        label="Total",
    )
    bc_axis.plot(
        bc_epochs,
        [record["weighted_speed_loss"] for record in bc_records],
        color=colors["orange"],
        linewidth=1.1,
        label="Speed (weighted)",
    )
    bc_axis.plot(
        bc_epochs,
        [record["steering_loss"] for record in bc_records],
        color=colors["green"],
        linewidth=1.1,
        label="Steering",
    )
    bc_axis.set_yscale("log")
    bc_axis.yaxis.set_minor_locator(NullLocator())
    bc_axis.set_ylabel("Loss")
    bc_axis.set_xlim(0, 520)
    bc_axis.legend(
        handles=[
            Patch(facecolor=colors["blue"], label="Total"),
            Patch(facecolor=colors["orange"], label="Speed (weighted)"),
            Patch(facecolor=colors["green"], label="Steering"),
        ],
        frameon=True,
        framealpha=1.0,
        edgecolor="black",
        fancybox=False,
        loc="upper right",
        bbox_to_anchor=(0.96, 0.95),
        borderpad=0.4,
        labelspacing=0.35,
        handlelength=1.4,
        handleheight=0.6,
        handletextpad=0.5,
    )
    bc_axis.text(0.5, -0.26, "(a) BC", transform=bc_axis.transAxes, ha="center")

    ppo_epochs = [record["epoch"] for record in ppo_records]
    ppo_axis.plot(
        ppo_epochs,
        [100.0 * record["screening_safety_rate"] for record in ppo_records],
        color=colors["blue"],
        linewidth=1.1,
        label="Safety",
    )
    ppo_axis.plot(
        ppo_epochs,
        [100.0 * record["screening_overtake_rate"] for record in ppo_records],
        color=colors["orange"],
        linewidth=1.1,
        label="Overtake",
    )
    ppo_axis.set_ylabel("Rate (%)")
    ppo_axis.set_xlim(0, 520)
    ppo_axis.set_ylim(0, 100)
    ppo_axis.set_yticks([20, 40, 60, 80, 100])
    ppo_axis.legend(
        handles=[
            Patch(facecolor=colors["blue"], label="Safety"),
            Patch(facecolor=colors["orange"], label="Overtake"),
        ],
        frameon=True,
        framealpha=1.0,
        edgecolor="black",
        fancybox=False,
        loc="lower right",
        bbox_to_anchor=(1.0, 0.06),
        handlelength=1.4,
    )
    ppo_axis.text(0.5, -0.26, "(b) PPO", transform=ppo_axis.transAxes, ha="center")
    for axis in (bc_axis, ppo_axis):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_xticks([0, 100, 200, 300, 400, 500])
        axis.tick_params(axis="both", which="major", length=2.0)

    for figure, name in (
        (bc_figure, "bc-training.pdf"),
        (ppo_figure, "ppo-training.pdf"),
    ):
        figure.subplots_adjust(left=0.22, right=0.99, top=0.98, bottom=0.25)
        figure.savefig(output_dir / name)
        plt.close(figure)


def main():
    args = parse_arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bc_metrics_path = args.bc_metrics
    if args.train_bc:
        bc_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        train_bc(args.dataset_dir, bc_metrics_path)
    if not bc_metrics_path.is_file():
        raise FileNotFoundError(f"BC metrics not found: {bc_metrics_path}; pass --train_bc")

    plot_figures(
        read_bc_metrics(bc_metrics_path),
        read_ppo_metrics(args.ppo_metrics),
        args.output_dir,
    )


if __name__ == "__main__":
    main()
