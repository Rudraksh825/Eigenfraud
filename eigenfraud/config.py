"""Configuration management via YAML + optional CLI overrides."""
from __future__ import annotations

import copy
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    root: str = "~/data"
    genimage_generator: str = "Stable Diffusion V1.4"
    ff_compression: str = "c23"
    ff_methods: list[str] = field(default_factory=lambda: [
        "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"
    ])
    image_size: int = 256
    batch_size: int = 64
    num_workers: int = 4


@dataclass
class AugConfig:
    blur_prob: float = 0.5
    jpeg_prob: float = 0.5
    jpeg_quality_range: list[int] = field(default_factory=lambda: [30, 95])
    blur_sigma_range: list[float] = field(default_factory=lambda: [0.0, 3.0])


@dataclass
class ModelConfig:
    type: str = "resnet2d"   # "resnet2d" | "cnn1d"
    pretrained: bool = False
    input_dim: int = 128     # for cnn1d: image_size // 2


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-4
    checkpoint_dir: str = "outputs/checkpoints"
    log_wandb: bool = False


@dataclass
class AttackConfig:
    type: str = "pgd_freq"       # "pgd_spatial" | "pgd_freq" | "pgd_freq_band"
    eps: float = 8 / 255
    alpha: float = 2 / 255
    num_iter: int = 20
    eps_freq: float = 0.05
    alpha_freq: float = 0.005
    spatial_eps: float | None = None  # optional spatial projection after IFFT
    band_low: float = 0.0
    band_high: float = 1.0
    n_samples: int = 1000
    checkpoint: str = ""


@dataclass
class RobustnessConfig:
    jpeg_qualities: list[int] = field(default_factory=lambda: [100, 90, 75, 50, 30])
    resize_factors: list[float] = field(default_factory=lambda: [1.0, 0.9, 0.8, 0.7, 0.5])
    blur_sigmas: list[float] = field(default_factory=lambda: [0.0, 1.0, 2.0, 3.0])
    checkpoint: str = ""
    output_csv: str = "outputs/results/robustness.csv"


@dataclass
class Config:
    seed: int = 42
    device: str = "cuda"
    data: DataConfig = field(default_factory=DataConfig)
    aug: AugConfig = field(default_factory=AugConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)
    run_name: str = "run"

    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path, overrides: list[str] | None = None) -> "Config":
        with open(path) as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        if overrides:
            raw = _apply_overrides(raw, overrides)
        return _dict_to_config(cls, raw)

    @classmethod
    def from_cli(cls) -> "Config":
        """Load config from first CLI arg, apply remaining args as overrides."""
        args = sys.argv[1:]
        if not args:
            return cls()
        path = args[0]
        overrides = args[1:]
        return cls.from_yaml(path, overrides)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_overrides(raw: dict, overrides: list[str]) -> dict:
    """Apply key=value overrides using dot notation, e.g. 'data.batch_size=32'."""
    raw = copy.deepcopy(raw)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be key=value, got: {override!r}")
        key, _, val_str = override.partition("=")
        keys = key.split(".")
        d = raw
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        # attempt type coercion via yaml
        d[keys[-1]] = yaml.safe_load(val_str)
    return raw


_SUB_CONFIG_FIELDS = {
    "data": DataConfig,
    "aug": AugConfig,
    "model": ModelConfig,
    "train": TrainConfig,
    "attack": AttackConfig,
    "robustness": RobustnessConfig,
}


def _dict_to_config(cls, d: dict) -> Any:
    kwargs: dict[str, Any] = {}
    for f_name, f_type in _SUB_CONFIG_FIELDS.items():
        if f_name in d:
            sub = d.pop(f_name)
            kwargs[f_name] = f_type(**sub) if isinstance(sub, dict) else sub
    kwargs.update(d)
    return cls(**kwargs)
