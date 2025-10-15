"""
Utility to pretrain the attack-adaptive attention module using captured rounds.

This script expects a dataset directory created by running a Plato experiment
with ``algorithm.dataset_capture_dir`` configured. Each file in that directory
(`round_*.pt`) should contain:

    {
        "projection": (channels, num_clients) tensor,
        "reference_weights": (num_clients,) tensor,
        "attention_weights": (num_clients,) tensor,
        "num_samples": (num_clients,) tensor,
        "client_ids": (num_clients,) tensor,
    }

The script fits the attention network to the reference weights (falling back to
the attention weights if reference weights are unavailable) and stores the
resulting checkpoint for production use.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F

# Allow running as a script without installing the examples package.
SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_ROOT = SCRIPT_DIR.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from attack_adaptive_server_strategy import _AttentionLoop
from torch.utils.data import DataLoader, Dataset, random_split


def _normalise_weights(weights: torch.Tensor) -> torch.Tensor:
    """Normalise weights to sum to one along the final dimension."""
    weights = weights.clone()
    weights = F.normalize(weights, p=1, dim=-1)
    if torch.isnan(weights).any():
        weights = torch.full_like(weights, 1.0 / weights.shape[-1])
    return weights


class RoundDataset(Dataset):
    """Dataset of captured attack-adaptive rounds."""

    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir)
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory '{dataset_dir}' not found.")

        self.files = sorted(self.dataset_dir.glob("round_*.pt"))
        if not self.files:
            raise ValueError(
                f"No captured rounds found in '{dataset_dir}'. "
                "Verify that dataset_capture_dir was populated."
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        payload = torch.load(self.files[index], map_location="cpu")
        projection = payload["projection"].float()
        reference = payload.get("reference_weights")
        attention = payload.get("attention_weights")

        if reference is not None and reference.numel() > 0:
            target = reference.float()
        else:
            target = attention.float()

        return projection, target


def _split_dataset(
    dataset: Dataset,
    val_ratio: float,
) -> Tuple[Dataset, Dataset]:
    if val_ratio <= 0 or len(dataset) == 1:
        return dataset, dataset

    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    return random_split(dataset, [train_size, val_size])


def _train_epoch(
    model: _AttentionLoop,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for projection, target in loader:
        projection = projection.to(device)
        target = _normalise_weights(target.to(device))

        optimizer.zero_grad()
        beta = projection.median(dim=-1, keepdim=True).values
        pred = model.get_weights(beta, projection).squeeze(1)
        loss = F.l1_loss(pred, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * projection.size(0)

    return total_loss / len(loader.dataset)


def _evaluate(
    model: _AttentionLoop,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for projection, target in loader:
            projection = projection.to(device)
            target = _normalise_weights(target.to(device))
            beta = projection.median(dim=-1, keepdim=True).values
            pred = model.get_weights(beta, projection).squeeze(1)
            loss = F.l1_loss(pred, target)
            total_loss += loss.item() * projection.size(0)
    return total_loss / len(loader.dataset)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain attack-adaptive attention.")
    parser.add_argument(
        "--dataset-dir",
        required=True,
        type=Path,
        help="Directory produced by dataset_capture_dir.",
    )
    parser.add_argument(
        "--save-path",
        required=True,
        type=Path,
        help="Destination path for the trained attention model.",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=0.005)
    parser.add_argument("--scale", type=float, default=10.0)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--attention-loops", type=int, default=5)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device (default: cuda if available else cpu).",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> None:
    args = parse_args(argv)
    device = torch.device(args.device)

    dataset = RoundDataset(args.dataset_dir)
    train_dataset, val_dataset = _split_dataset(dataset, args.val_ratio)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    sample_projection, _ = dataset[0]
    channels, num_clients = sample_projection.shape

    model = _AttentionLoop(
        in_channels=channels,
        out_channels=args.hidden_size,
        iterations=args.attention_loops,
        epsilon=args.epsilon,
        scale=args.scale,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, device)
        val_loss = _evaluate(model, val_loader, device)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                "model_state": model.state_dict(),
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        print(
            f"[Epoch {epoch:03d}] train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
        )

    if best_state is None:
        raise RuntimeError("Training failed to produce a valid model state.")

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state["model_state"], args.save_path)
    print(
        f"Saved attention model to '{args.save_path}' "
        f"(epoch {best_state['epoch']}, val_loss={best_state['val_loss']:.6f})."
    )


if __name__ == "__main__":
    main(sys.argv[1:])
