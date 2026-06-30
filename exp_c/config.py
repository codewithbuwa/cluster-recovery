from dataclasses import dataclass
from pathlib import Path

from scr.config import TrainConfig, WorldConfig


@dataclass(frozen=True)
class ExperimentCConfig:
    world: WorldConfig = WorldConfig(pi_a=0.9, tau_a=0.25, tau_b=0.75, eps_a=0.05, eps_b=0.05)
    train: TrainConfig = TrainConfig()
    seeds: tuple[int, ...] = tuple(range(10))
    c3_seeds: tuple[int, ...] = tuple(range(10))
    beta_values: tuple[float, ...] = (0.3, 1.0, 3.0, 10.0)
    output_dir: Path = Path("outputs/exp_c/")
