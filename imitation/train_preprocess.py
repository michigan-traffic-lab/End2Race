import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from expert.utils import require_end2race_runtime
from imitation.model_preprocess import End2Race

NUM_EPOCHS = 500
CHECKPOINT_INTERVAL = 500
BATCH_SIZE = 1024
LEARNING_RATE = 1e-4


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train the End2Race LiDAR preprocessing ablation")
    parser.add_argument("--dataset_dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--preprocessing", choices=End2Race.PREPROCESSING_OPTIONS, required=True)

    parser.add_argument("--speed_loss_weight", type=float, default=0.05)
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)

    args = parser.parse_args()
    expected_output_directory = f"ckp_ablation_{args.preprocessing}"
    if args.output_dir is None:
        args.output_dir = Path("checkpoint") / expected_output_directory

    known_output_directories = {
        f"ckp_ablation_{mode}" for mode in End2Race.PREPROCESSING_OPTIONS
    }
    conflicting_directories = (
        known_output_directories.intersection(args.output_dir.parts)
        - {expected_output_directory}
    )
    if conflicting_directories:
        parser.error(
            f"--preprocessing {args.preprocessing} cannot save under "
            f"{sorted(conflicting_directories)[0]}"
        )
    return args


class SequenceDataset(Dataset):
    def __init__(self, data_path: str | Path):
        self.lidar_columns = [
            f"lidar_{index}" for index in range(End2Race.NUM_LIDAR_FEATURES)
        ]
        self.speed_column = "current_speed"
        self.action_columns = ["steer", "desired_speed"]
        self.expected_columns = ["time", self.speed_column] + self.action_columns + self.lidar_columns
        self.sequence_length = self._determine_sequence_length(data_path)
        self.sequences = []
        self._load_episodes(data_path)
        if not self.sequences:
            raise ValueError(f"No usable training sequences found in {data_path}")
        print(f"Loaded {len(self.sequences)} sequences")

    def _load_episodes(self, data_path: str | Path):
        csv_files = sorted(Path(data_path).glob("*.csv"))
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            if list(df.columns) != self.expected_columns:
                raise ValueError(
                    f"{csv_file} does not match the collected CSV header"
                )

            if len(df) < self.sequence_length:
                continue

            lidar_data = df[self.lidar_columns].values.astype(np.float32)
            speed_data = df[[self.speed_column]].values.astype(np.float32)
            action_data = df[self.action_columns].values.astype(np.float32)

            self._create_sequences(lidar_data, speed_data, action_data)

    def _determine_sequence_length(self, data_path: str | Path) -> int:
        csv_files = sorted(Path(data_path).glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No training CSV files found in {data_path}")
        sequence_length = len(pd.read_csv(csv_files[0]))
        print(f"Sequence length: {sequence_length}")
        return sequence_length

    def _create_sequences(
        self,
        lidar_data: np.ndarray,
        speed_data: np.ndarray,
        action_data: np.ndarray,
    ):
        previous_speed = np.concatenate((speed_data[:1], speed_data[:-1]))
        for end_idx in range(self.sequence_length - 1, len(lidar_data)):
            start_idx = end_idx - self.sequence_length + 1
            self.sequences.append(
                {
                    "lidar": lidar_data[start_idx : end_idx + 1],
                    "speed": previous_speed[start_idx : end_idx + 1],
                    "action": action_data[start_idx : end_idx + 1],
                }
            )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence = self.sequences[index]
        return (
            torch.from_numpy(sequence["lidar"]),
            torch.from_numpy(sequence["speed"]),
            torch.from_numpy(sequence["action"]),
        )


def train_epoch(
    model,
    train_loader,
    optimizer,
    speed_loss_weight,
    gradient_clip_norm,
):
    device = next(model.parameters()).device
    model.train()
    total_loss = 0.0
    total_steering_loss = 0.0
    total_weighted_speed_loss = 0.0
    for lidar_seq, speed_seq, target_actions in train_loader:
        lidar_seq = lidar_seq.to(device, non_blocking=True)
        speed_seq = speed_seq.to(device, non_blocking=True)
        target_actions = target_actions.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        predicted_actions, _ = model(lidar_seq, speed_seq)
        predicted_actions_flat = predicted_actions.reshape(-1, predicted_actions.shape[-1])
        target_actions_flat = target_actions.reshape(-1, target_actions.shape[-1])
        steering_loss = F.mse_loss(predicted_actions_flat[:, 0], target_actions_flat[:, 0])
        weighted_speed_loss = speed_loss_weight * F.mse_loss(
            predicted_actions_flat[:, 1], target_actions_flat[:, 1]
        )
        loss = steering_loss + weighted_speed_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
        optimizer.step()
        total_loss += loss.item()
        total_steering_loss += steering_loss.item()
        total_weighted_speed_loss += weighted_speed_loss.item()
    batches = len(train_loader)
    return (
        total_loss / batches,
        total_steering_loss / batches,
        total_weighted_speed_loss / batches,
    )


def main():
    args = parse_arguments()
    require_end2race_runtime()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(
        f"Training settings: epochs={NUM_EPOCHS}, checkpoint_interval={CHECKPOINT_INTERVAL}, "
        f"batch_size={BATCH_SIZE}, learning_rate={LEARNING_RATE}, arguments={vars(args)}"
    )

    dataset = SequenceDataset(args.dataset_dir / "success")

    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )

    model = End2Race(args.preprocessing).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Train batches: {len(train_loader)}")

    records = []
    for epoch in range(1, NUM_EPOCHS + 1):
        total_loss, steering_loss, weighted_speed_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            args.speed_loss_weight,
            args.gradient_clip_norm,
        )
        records.append(
            {
                "epoch": epoch,
                "total_loss": total_loss,
                "steering_loss": steering_loss,
                "weighted_speed_loss": weighted_speed_loss,
            }
        )
        if epoch % CHECKPOINT_INTERVAL:
            continue
        checkpoint_path = args.output_dir / f"epoch_{epoch:05d}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(
            f"Epoch {epoch}/{NUM_EPOCHS}, loss: {total_loss:.5f}, "
            f"saved {checkpoint_path}"
        )

    metrics_path = args.output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved {metrics_path}")


if __name__ == "__main__":
    main()
