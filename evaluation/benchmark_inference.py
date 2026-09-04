import argparse
import csv
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from f1tenth_sim.utils import simulation_config
from imitation.model import End2Race
from imitation.model_mlp import End2RaceMLP
from imitation.model_preprocess import End2Race as End2RacePreprocess
from imitation.model_transformer import End2RaceTransformer

WARMUP_STEPS = 300
TIMED_STEPS = 500
REPEATS = 7
EGO_SPEED = 5.0
MEBIBYTE = 1024**2

MODELS = {
    "bc_gru": (End2Race, Path("checkpoint/bc.pt")),
    "mlp": (End2RaceMLP, Path("checkpoint/ablation_mlp")),
    "transformer": (End2RaceTransformer, Path("checkpoint/ablation_transformer")),
    "linear": (lambda: End2RacePreprocess("linear"), Path("checkpoint/ablation_linear")),
    "no_norm": (lambda: End2RacePreprocess("no_normalization"), Path("checkpoint/ablation_nonorm")),
}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Benchmark End2Race inference at the deployment batch size")
    parser.add_argument("--output_dir", type=Path, default=Path("eval_results/inference_benchmark"))
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))

    parser.add_argument("--devices", nargs="+", choices=["cpu", "cuda"], default=["cpu"])
    parser.add_argument("--threads", type=int, default=1)

    return parser.parse_args()


def checkpoints(path: Path) -> list[Path]:
    return [path] if path.is_file() else sorted(path.glob("run_*/epoch_*.pt"))


def library_workspace_mib(device: torch.device) -> float:
    # The first matrix multiplication of a process allocates a cuBLAS workspace of a
    # few mebibytes. It is charged to whichever model runs first, so it is claimed here
    # instead, and every model measured afterwards reports only its own memory.
    if device.type != "cuda":
        return 0.0
    baseline = torch.cuda.memory_allocated(device)
    probe = torch.zeros(64, 64, device=device)
    torch.matmul(probe, probe).sum().item()
    torch.cuda.synchronize(device)
    workspace = torch.cuda.memory_allocated(device) - baseline - probe.numel() * probe.element_size()
    return workspace / MEBIBYTE


def load_model(factory, checkpoint_path: Path, device: torch.device):
    model = factory().to(device)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    return model.eval()


def make_step(model, device: torch.device):
    # One control step of the deployed policy: a single downsampled scan and the speed
    # measured at the previous step produce a single action. The batch dimension is the
    # number of vehicles being controlled, so it is always one.
    scan = (np.random.default_rng(0).random(model.NUM_LIDAR_FEATURES) * 10.0).astype(np.float32)
    stateful = isinstance(model(torch.zeros(1, 1, model.NUM_LIDAR_FEATURES, device=device),
                                torch.zeros(1, 1, 1, device=device)), tuple)
    state = None

    def step():
        nonlocal state
        lidar = torch.as_tensor(scan, dtype=torch.float32, device=device)[None, None]
        speed = torch.tensor([[[EGO_SPEED]]], dtype=torch.float32, device=device)
        if stateful:
            actions, state = model(lidar, speed, state)
        else:
            actions = model(lidar, speed)
        # Reading the action out is part of a control step, and on the GPU it is also
        # what forces the asynchronous queue to drain, so it belongs inside the timing.
        return actions[0, -1, 0].item(), actions[0, -1, 1].item()

    return step


def measure(model, device: torch.device) -> dict:
    weights = sum(t.numel() * t.element_size() for t in model.parameters())
    buffers = sum(t.numel() * t.element_size() for t in model.buffers())
    cuda = device.type == "cuda"
    baseline = torch.cuda.memory_allocated(device) - weights - buffers if cuda else 0

    with torch.no_grad():
        step = make_step(model, device)
        # A stateful policy reaches its steady state only once its context is full, and
        # that is also its slowest state, so the timed window starts after the warm-up.
        for _ in range(WARMUP_STEPS):
            step()
        if cuda:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        resident = torch.cuda.memory_allocated(device) - baseline if cuda else weights + buffers
        peak = torch.cuda.max_memory_allocated(device) - baseline if cuda else None

        latencies = []
        for _ in range(REPEATS):
            started = time.perf_counter()
            for _ in range(TIMED_STEPS):
                step()
            latencies.append((time.perf_counter() - started) / TIMED_STEPS * 1000.0)
        if cuda:
            torch.cuda.synchronize(device)

    return {
        "params": sum(t.numel() for t in model.parameters()),
        "weights_mib": (weights + buffers) / MEBIBYTE,
        "resident_mib": resident / MEBIBYTE,
        "peak_mib": None if peak is None else peak / MEBIBYTE,
        "latency_ms": statistics.median(latencies),
        "latency_min_ms": min(latencies),
        "latency_max_ms": max(latencies),
    }


def main():
    args = parse_arguments()
    torch.set_num_threads(args.threads)
    control_period_ms = simulation_config().control_timestep * 1000.0
    print(f"Control period: {control_period_ms:.1f} ms, batch size 1, {args.threads} CPU thread(s)")

    records = []
    for device_name in args.devices:
        device = torch.device(device_name)
        workspace = library_workspace_mib(device)
        if workspace:
            print(f"cuBLAS workspace on {device_name}: {workspace:.3f} MiB, excluded from every model below")

        for name in args.models:
            factory, path = MODELS[name]
            for checkpoint_path in checkpoints(path):
                model = load_model(factory, checkpoint_path, device)
                record = {"model": name, "device": device_name, "checkpoint": str(checkpoint_path)}
                record.update(measure(model, device))
                record["control_period_percent"] = 100.0 * record["latency_ms"] / control_period_ms
                records.append(record)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "inference.csv"
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    print(f"\n{'model':<14}{'device':>7}{'params':>10}{'weights':>10}{'latency':>11}"
          f"{'of period':>11}{'resident':>11}{'peak':>9}")
    for device_name in args.devices:
        for name in args.models:
            group = [r for r in records if r["model"] == name and r["device"] == device_name]
            if not group:
                continue
            latency = statistics.median([r["latency_ms"] for r in group])
            peak = group[0]["peak_mib"]
            print(f"{name:<14}{device_name:>7}{group[0]['params']:>10,}"
                  f"{group[0]['weights_mib']:>9.2f}M{latency:>8.3f} ms"
                  f"{100.0 * latency / control_period_ms:>10.1f}%"
                  f"{group[0]['resident_mib']:>10.2f}M"
                  f"{'' if peak is None else f'{peak:>8.2f}M'}")
    print(f"\nSaved {output_path}")


if __name__ == "__main__":
    main()
