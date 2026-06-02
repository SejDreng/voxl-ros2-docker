#!/usr/bin/env python3
"""Convert the XOR checkpoint state_dict into a TorchScript .pt file.

Usage:
    python3 scripts/save_xor_as_pt.py
    python3 scripts/save_xor_as_pt.py --input models/xor_model.pth --output models/xor_model.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "models" / "xor_model.pth"
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "xor_model.pt"


class XORModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(2, 8)
        self.linear15 = nn.Linear(8, 2)
        self.linear2 = nn.Linear(2, 1)
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear15(x))
        return self.linear2(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save the XOR checkpoint as a TorchScript .pt file.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to the input XOR checkpoint.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Path for the saved .pt file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input checkpoint not found: {input_path}")

    state_dict = torch.load(input_path, map_location="cpu", weights_only=False)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected a state_dict in {input_path}, got {type(state_dict).__name__}")

    model = XORModel()
    model.load_state_dict(state_dict)
    model.eval()

    scripted = torch.jit.script(model)
    scripted.save(str(output_path))

    print(f"Saved TorchScript model to: {output_path}")


if __name__ == "__main__":
    main()