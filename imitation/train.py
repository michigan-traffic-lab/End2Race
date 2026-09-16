from itertools import chain
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Dataset

from imitation.model import End2Race

class SequenceDataset(Dataset):
    def __init__(self, data_path: str | Path):
        paths = iter(sorted(Path(data_path).glob("*.csv")))
        first_episode = pd.read_csv(next(paths))
        self.sequence_length = len(first_episode)
        self.sequences = []
        lidar_columns = [f"lidar_{index}" for index in range(End2Race.NUM_LIDAR_FEATURES)]
        for episode in chain((first_episode,), map(pd.read_csv, paths)):
            lidar = episode[lidar_columns].to_numpy(dtype=np.float32)
            speed = episode[['current_speed']].to_numpy(dtype=np.float32)
            actions = episode[['steer', 'desired_speed']].to_numpy(dtype=np.float32)
            previous_speed = np.concatenate((speed[:1], speed[:-1]))
            for start in range(len(episode) - self.sequence_length + 1):
                end = start + self.sequence_length
                self.sequences.append((lidar[start:end], previous_speed[start:end], actions[start:end]))
        print(f"Loaded {len(self.sequences)} sequences of length {self.sequence_length}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return tuple(torch.from_numpy(array) for array in self.sequences[index])


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
    root = Path(__file__).resolve().parents[1]
    with (root / "config.yaml").open() as stream:
        config = yaml.safe_load(stream)
    settings = config['imitation']
    device = torch.device(config['runtime']['device'])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA.")
    if device.index is not None:
        torch.cuda.set_device(device)
    print(f"Using device: {device}")
    print(
        f"Training settings: epochs={settings['num_epochs']}, "
        f"learning_rate={settings['learning_rate']}, arguments={settings}"
    )

    dataset = SequenceDataset(root / settings['dataset_dir'] / "success")

    train_loader = DataLoader(
        dataset,
        batch_size=len(dataset),
        num_workers=config['runtime']['workers'],
        pin_memory=True,
    )

    model = End2Race().to(device)
    optimizer = optim.Adam(model.parameters(), lr=settings['learning_rate'])
    output_dir = root / settings['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Train batches: {len(train_loader)}")

    for epoch in range(1, settings['num_epochs'] + 1):
        loss = train_epoch(
            model,
            train_loader,
            optimizer,
            settings['speed_loss_weight'],
            settings['gradient_clip_norm'],
        )
        print(f"Epoch {epoch}/{settings['num_epochs']}, loss: {loss:.5f}")

    checkpoint_path = output_dir / f"epoch_{settings['num_epochs']:05d}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved {checkpoint_path}")


if __name__ == "__main__":
    main()
