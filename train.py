import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from model import End2Race
from utils import require_end2race_runtime


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train End2Race speed-conditioned model")
    parser.add_argument("--dataset_dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output_dir", type=Path, default=Path("checkpoint"))
    parser.add_argument("--save_interval", type=int, default=500)

    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--num_epochs", type=int, default=5000)
    parser.add_argument("--speed_loss_weight", type=float, default=0.05)
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)

    return parser.parse_args()


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
    for lidar_seq, speed_seq, target_actions in train_loader:
        lidar_seq = lidar_seq.to(device, non_blocking=True)
        speed_seq = speed_seq.to(device, non_blocking=True)
        target_actions = target_actions.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        predicted_actions, _ = model(lidar_seq, speed_seq)
        predicted_actions_flat = predicted_actions.reshape(-1, predicted_actions.shape[-1])
        target_actions_flat = target_actions.reshape(-1, target_actions.shape[-1])
        steer_loss = F.mse_loss(predicted_actions_flat[:, 0], target_actions_flat[:, 0])
        speed_loss = F.mse_loss(predicted_actions_flat[:, 1], target_actions_flat[:, 1])
        loss = steer_loss + speed_loss * speed_loss_weight
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def main():
    args = parse_arguments()
    require_end2race_runtime()
    if args.save_interval < 1:
        raise SystemExit("--save_interval must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training arguments: {vars(args)}")

    dataset = SequenceDataset(args.dataset_dir / "success")

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )

    model = End2Race().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Train batches: {len(train_loader)}")

    for epoch in range(1, args.num_epochs + 1):
        loss = train_epoch(
            model,
            train_loader,
            optimizer,
            args.speed_loss_weight,
            args.gradient_clip_norm,
        )
        if epoch % args.save_interval and epoch != args.num_epochs:
            continue
        checkpoint_path = args.output_dir / f"epoch_{epoch:05d}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(
            f"Epoch {epoch}/{args.num_epochs}, loss: {loss:.5f}, "
            f"saved {checkpoint_path}"
        )


if __name__ == "__main__":
    main()
