import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from config import load_project_config
from eval_single import (
    EVALUATION_NOISE,
    EVALUATION_SEED,
    LAP_COUNT,
    MINIMUM_LAP_TIME,
    START_INDEX,
    EvaluationSettings,
    evaluate_laps,
)
from model import End2Race
from utils import require_end2race_runtime

CHECKPOINT_DIRECTORY = Path("checkpoint")
TRAINING_STATE_PATH = CHECKPOINT_DIRECTORY / "training_state.pt"
DATASET_DIRECTORY = Path("dataset")
EPOCH_INTERVAL = 500
MAX_EPOCHS = 5000
MAX_MODELS = 10
SINGLE_AGENT_MAPS = (
    "Austin",
    "Hockenheim",
    "MoscowRaceway",
    "Nuerburgring",
)


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


def train_epoch(model, train_loader, criterion, optimizer, config):
    device = next(model.parameters()).device
    model.train()
    total_loss = 0.0
    for lidar_seq, speed_seq, target_actions in train_loader:
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
    return total_loss / len(train_loader)


def evaluate_checkpoint(model, device, vehicle, checkpoint_path):
    for map_name in SINGLE_AGENT_MAPS:
        print(f"Single-agent evaluation: {map_name}")
        settings = EvaluationSettings(
            map_name=map_name,
            checkpoint_path=checkpoint_path,
            noise=EVALUATION_NOISE,
            seed=EVALUATION_SEED,
            render=False,
            lap_num=LAP_COUNT,
            start_idx=START_INDEX,
            minimum_lap_time=MINIMUM_LAP_TIME,
        )
        if not evaluate_laps(model, device, vehicle, settings):
            return False
    print("Austin 720-scenario multi-agent evaluation")
    return (
        subprocess.run(["bash", "eval_multi.sh", str(checkpoint_path)]).returncode
        == 0
    )


def save_training_state(model_number, epoch, model, optimizer):
    torch.save(
        {
            "model_number": model_number,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        TRAINING_STATE_PATH,
    )


def remove_checkpoints():
    for checkpoint_path in CHECKPOINT_DIRECTORY.glob("checkpoint_*.pt"):
        checkpoint_path.unlink()
    TRAINING_STATE_PATH.unlink(missing_ok=True)


def main():
    require_end2race_runtime()
    project = load_project_config()
    config = project.training
    if config.num_epochs != MAX_EPOCHS:
        raise ValueError(f"training.num_epochs must be {MAX_EPOCHS}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_path = DATASET_DIRECTORY / "success"
    dataset = SequenceDataset(data_path)

    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )

    criterion = nn.MSELoss()
    CHECKPOINT_DIRECTORY.mkdir(exist_ok=True)
    print(f"Train batches: {len(train_loader)}")
    if TRAINING_STATE_PATH.is_file():
        state = torch.load(TRAINING_STATE_PATH, map_location=device)
        model_number = state["model_number"]
        completed_epochs = state["epoch"]
        model = End2Race().to(device)
        model.load_state_dict(state["model_state_dict"])
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        checkpoint_path = (
            CHECKPOINT_DIRECTORY / f"checkpoint_{completed_epochs:05d}.pt"
        )
        print(
            f"Resuming model {model_number}/{MAX_MODELS} at "
            f"epoch {completed_epochs}"
        )
        if evaluate_checkpoint(model, device, project.vehicle, checkpoint_path):
            print(f"Model {model_number} passed all evaluations")
            return
    else:
        model_number = 1
        completed_epochs = 0
        model = None
        optimizer = None

    while model_number <= MAX_MODELS:
        print(f"\nStarting model {model_number}/{MAX_MODELS}")
        if model is None:
            model = End2Race().to(device)
            optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        for epoch in range(completed_epochs + 1, MAX_EPOCHS + 1):
            loss = train_epoch(model, train_loader, criterion, optimizer, config)
            if epoch % EPOCH_INTERVAL:
                continue
            checkpoint_path = (
                CHECKPOINT_DIRECTORY / f"checkpoint_{epoch:05d}.pt"
            )
            torch.save(model.state_dict(), checkpoint_path)
            save_training_state(model_number, epoch, model, optimizer)
            print(f"Epoch {epoch}/{MAX_EPOCHS}, loss: {loss:.5f}")
            if evaluate_checkpoint(
                model, device, project.vehicle, checkpoint_path
            ):
                print(f"Model {model_number} passed all evaluations")
                return
        print(f"Model {model_number} did not pass; removing checkpoints")
        remove_checkpoints()
        model_number += 1
        completed_epochs = 0
        model = None
        optimizer = None
    raise RuntimeError(f"No usable model found after {MAX_MODELS} attempts")


if __name__ == "__main__":
    main()
