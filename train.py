import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

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
from utils import load_racetrack_config, require_end2race_runtime

DATASET_DIRECTORY = Path("dataset")
EPOCH_INTERVAL = 500
MAX_MODELS = 10
SINGLE_AGENT_MAPS = (
    "Austin",
    "Hockenheim",
    "MoscowRaceway",
    "Nuerburgring",
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train End2Race speed-conditioned model"
    )
    parser.add_argument(
        "--model_path", type=Path, default=Path("checkpoint/checkpoint.pt")
    )
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
        self.expected_columns = (
            ["time", self.speed_column]
            + self.action_columns
            + self.lidar_columns
        )
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

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence = self.sequences[index]
        return (
            torch.from_numpy(sequence["lidar"]),
            torch.from_numpy(sequence["speed"]),
            torch.from_numpy(sequence["action"]),
        )


def train_epoch(
    model,
    train_loader,
    criterion,
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
        loss = steer_loss + speed_loss * speed_loss_weight
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=gradient_clip_norm
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


def save_training_state(path, model_number, epoch, model, optimizer):
    torch.save(
        {
            "model_number": model_number,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def main():
    args = parse_arguments()
    require_end2race_runtime()
    vehicle = load_racetrack_config().vehicle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training arguments: {vars(args)}")

    dataset = SequenceDataset(DATASET_DIRECTORY / "success")

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )

    criterion = nn.MSELoss()
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    training_state_path = args.model_path.with_name("training_state.pt")
    print(f"Train batches: {len(train_loader)}")
    if training_state_path.is_file():
        state = torch.load(training_state_path, map_location=device)
        model_number = state["model_number"]
        completed_epochs = state["epoch"]
        if completed_epochs > args.num_epochs:
            raise ValueError(
                f"Training state epoch {completed_epochs} exceeds "
                f"--num_epochs {args.num_epochs}"
            )
        model = End2Race().to(device)
        model.load_state_dict(state["model_state_dict"])
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = args.learning_rate
        print(
            f"Resuming model {model_number}/{MAX_MODELS} at "
            f"epoch {completed_epochs}"
        )
        if evaluate_checkpoint(model, device, vehicle, args.model_path):
            print(f"Model {model_number} passed all evaluations")
            return
    else:
        model_number = 1
        completed_epochs = 0
        model = None

    while model_number <= MAX_MODELS:
        print(f"\nStarting model {model_number}/{MAX_MODELS}")
        if model is None:
            model = End2Race().to(device)
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        for epoch in range(completed_epochs + 1, args.num_epochs + 1):
            loss = train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                args.speed_loss_weight,
                args.gradient_clip_norm,
            )
            if epoch % EPOCH_INTERVAL and epoch != args.num_epochs:
                continue
            torch.save(model.state_dict(), args.model_path)
            save_training_state(
                training_state_path, model_number, epoch, model, optimizer
            )
            print(f"Epoch {epoch}/{args.num_epochs}, loss: {loss:.5f}")
            if evaluate_checkpoint(model, device, vehicle, args.model_path):
                print(f"Model {model_number} passed all evaluations")
                return
        print(f"Model {model_number} did not pass; removing saved model")
        args.model_path.unlink(missing_ok=True)
        training_state_path.unlink(missing_ok=True)
        model_number += 1
        completed_epochs = 0
        model = None
    raise RuntimeError(f"No usable model found after {MAX_MODELS} attempts")


if __name__ == "__main__":
    main()
