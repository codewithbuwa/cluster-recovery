from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scr.config import TrainConfig, WorldConfig


@dataclass(frozen=True)
class ExperimentBConfig:
    world: WorldConfig = WorldConfig()
    train: TrainConfig = TrainConfig()
    seeds: tuple[int, ...] = tuple(range(50))
    pi_a_values: tuple[float, ...] = (0.5, 0.7, 0.85, 0.95, 0.99)
    output_dir: Path = Path("outputs/exp_b/")
