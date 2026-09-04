import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from expert.utils import require_end2race_runtime
from transformer.model import End2RaceTransformer

NUM_EPOCHS = 36
LEARNING_RATE = 1e-4
SPEED_LOSS_WEIGHT = 0.05
GRADIENT_CLIP_NORM = 1.0
NUM_WORKERS = 4
DATASET_DIR = Path("dataset")
CHECKPOINT_PATH = Path("checkpoint/transformer.pt")
METRICS_PATH = Path("checkpoint/transformer_metrics.csv")


class DemonstrationDataset(Dataset):
    def __init__(self, data_path: str | Path):
        self.lidar_columns = [
            f"lidar_{index}" for index in range(End2RaceTransformer.NUM_LIDAR_FEATURES)
        ]
        self.speed_column = "current_speed"
        self.action_columns = ["steer", "desired_speed"]
        self.expected_columns = (
            ["time", self.speed_column] + self.action_columns + self.lidar_columns
        )
        self.episodes = []
        self._load_episodes(data_path)
        if not self.episodes:
            raise ValueError(f"No usable training episodes found in {data_path}")
        print(f"Loaded {len(self.episodes)} episodes")

    def _load_episodes(self, data_path: str | Path):
        csv_files = sorted(Path(data_path).glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No training CSV files found in {data_path}")
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            if list(df.columns) != self.expected_columns:
                raise ValueError(f"{csv_file} does not match the collected CSV header")

            if len(df) < End2RaceTransformer.CONTEXT_LENGTH:
                continue

            lidar_data = df[self.lidar_columns].values.astype(np.float32)
            speed_data = df[[self.speed_column]].values.astype(np.float32)
            action_data = df[self.action_columns].values.astype(np.float32)
            self.episodes.append({
                "lidar": lidar_data,
                "speed": np.concatenate((speed_data[:1], speed_data[:-1])),
                "action": action_data,
            })

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        episode = self.episodes[index]
        return (
            torch.from_numpy(episode["lidar"]),
            torch.from_numpy(episode["speed"]),
            torch.from_numpy(episode["action"]),
        )


def _collate_episode(batch):
    if len(batch) != 1:
        raise ValueError("The Transformer loader requires one episode per batch")
    lidar_episode, speed_episode, target_actions = batch[0]
    episode_length = len(lidar_episode)
    context_length = End2RaceTransformer.CONTEXT_LENGTH
    lidar_batch = lidar_episode.new_zeros(
        episode_length,
        context_length,
        End2RaceTransformer.NUM_LIDAR_FEATURES,
    )
    speed_batch = speed_episode.new_zeros(episode_length, context_length, 1)
    lengths = torch.arange(1, episode_length + 1).clamp(max=context_length)
    for end_idx in range(1, episode_length + 1):
        start_idx = max(0, end_idx - context_length)
        length = end_idx - start_idx
        lidar_batch[end_idx - 1, :length] = lidar_episode[start_idx:end_idx]
        speed_batch[end_idx - 1, :length] = speed_episode[start_idx:end_idx]
    positions = torch.arange(context_length)[None]
    padding_mask = positions >= lengths[:, None]
    return (
        lidar_batch,
        speed_batch,
        target_actions,
        lengths,
        padding_mask,
    )


def train_epoch(model, train_loader, optimizer):
    device = next(model.parameters()).device
    model.train()
    total_loss = 0.0
    total_steering_loss = 0.0
    total_weighted_speed_loss = 0.0
    for lidar_seq, speed_seq, target_actions, lengths, padding_mask in train_loader:
        lidar_seq = lidar_seq.to(device, non_blocking=True)
        speed_seq = speed_seq.to(device, non_blocking=True)
        target_actions = target_actions.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        padding_mask = padding_mask.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        predicted_actions, _ = model(
            lidar_seq,
            speed_seq,
            padding_mask=padding_mask,
        )
        batch_indices = torch.arange(len(lengths), device=device)
        final_actions = predicted_actions[batch_indices, lengths - 1]
        squared_error = (final_actions - target_actions) ** 2
        steering_loss = squared_error[:, 0].mean()
        weighted_speed_loss = SPEED_LOSS_WEIGHT * squared_error[:, 1].mean()
        loss = steering_loss + weighted_speed_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_NORM)
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
    require_end2race_runtime()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(
        f"Training settings: epochs={NUM_EPOCHS}, batch=one episode, "
        f"learning_rate={LEARNING_RATE}, speed_loss_weight={SPEED_LOSS_WEIGHT}, "
        f"gradient_clip_norm={GRADIENT_CLIP_NORM}"
    )

    dataset = DemonstrationDataset(DATASET_DIR / "success")
    train_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        pin_memory=device.type == "cuda",
        num_workers=NUM_WORKERS,
        persistent_workers=NUM_WORKERS > 0,
        collate_fn=_collate_episode,
    )

    model = End2RaceTransformer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Train batches: {len(train_loader)}")

    records = []
    for epoch in range(1, NUM_EPOCHS + 1):
        total_loss, steering_loss, weighted_speed_loss = train_epoch(
            model,
            train_loader,
            optimizer,
        )
        records.append(
            {
                "epoch": epoch,
                "total_loss": total_loss,
                "steering_loss": steering_loss,
                "weighted_speed_loss": weighted_speed_loss,
            }
        )
        print(f"Epoch {epoch}/{NUM_EPOCHS}, loss: {total_loss:.5f}")

    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"Saved {CHECKPOINT_PATH}")

    with METRICS_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved {METRICS_PATH}")


if __name__ == "__main__":
    main()
