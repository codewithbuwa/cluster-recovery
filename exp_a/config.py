from dataclasses import dataclass
from pathlib import Path

from scr.config import TrainConfig, WorldConfig


@dataclass(frozen=True)
class ExperimentAConfig:
    world: WorldConfig = WorldConfig(pi_a=0.9, tau_a=0.25, tau_b=0.75, eps_a=0.05, eps_b=0.05)
    train: TrainConfig = TrainConfig()
    seeds: tuple[int, ...] = (0, 1, 2, 3)
    output_dir: Path = Path("outputs/exp_a/")
