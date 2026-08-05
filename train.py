import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import load_project_config
from model import End2Race
from utils import require_end2race_runtime


class SequenceDataset(Dataset):
    def __init__(self, data_path: str | Path):
        self.lidar_columns = [
            f"lidar_{index}" for index in range(End2Race.NUM_LIDAR_FEATURES)
        ]
        self.speed_column = "current_speed"
        self.action_columns = ["steer", "desired_speed"]
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
            available_lidar_columns = [
                column for column in df.columns if column.startswith("lidar_")
            ]
            if available_lidar_columns != self.lidar_columns:
                raise ValueError(
                    f"{csv_file} must contain exactly "
                    f"{End2Race.NUM_LIDAR_FEATURES} ordered LiDAR columns"
                )

            missing_actions = set(self.action_columns) - set(df.columns)
            if missing_actions:
                raise ValueError(
                    f"{csv_file} is missing action columns: "
                    f"{sorted(missing_actions)}"
                )
            if self.speed_column not in df.columns:
                raise ValueError(
                    f"{csv_file} is missing speed column: {self.speed_column}"
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
        df = pd.read_csv(csv_files[0])
        sequence_length = len(df)
        if sequence_length < 1:
            raise ValueError(
                f"Training episodes in {data_path} must contain at least one row"
            )
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

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence = self.sequences[index]
        return (
            torch.from_numpy(sequence["lidar"]),
            torch.from_numpy(sequence["speed"]),
            torch.from_numpy(sequence["action"]),
        )


def train(model, train_loader, criterion, optimizer, config):
    device = next(model.parameters()).device

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0.0

        with tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs}"
        ) as pbar:
            for lidar_seq, speed_seq, target_actions in pbar:
                lidar_seq = lidar_seq.to(device, non_blocking=True)
                speed_seq = speed_seq.to(device, non_blocking=True)
                target_actions = target_actions.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                predicted_actions, _ = model(lidar_seq, speed_seq)

                predicted_actions_flat = predicted_actions.reshape(
                    -1, predicted_actions.shape[-1]
                )
                target_actions_flat = target_actions.reshape(
                    -1, target_actions.shape[-1]
                )

                steer_loss = criterion(
                    predicted_actions_flat[:, 0], target_actions_flat[:, 0]
                )
                speed_loss = criterion(
                    predicted_actions_flat[:, 1], target_actions_flat[:, 1]
                )
                loss = steer_loss + speed_loss * config.speed_loss_weight

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=config.gradient_clip_norm
                )
                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{config.num_epochs}, Loss: {avg_loss:.5f}")

        checkpoint_path = Path(f"checkpoint_{epoch + 1:05d}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")


def main():
    require_end2race_runtime()
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python train.py <dataset_dir>")

    config = load_project_config().training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_path = Path(sys.argv[1]) / "success"
    dataset = SequenceDataset(data_path)

    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )

    model = End2Race().to(device)

    print(f"Train batches: {len(train_loader)}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    train(model, train_loader, criterion, optimizer, config)

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
