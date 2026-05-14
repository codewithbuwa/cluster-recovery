from dataclasses import dataclass
from pathlib import Path

from scr.config import TrainConfig, WorldConfig


@dataclass(frozen=True)
class ExperimentDConfig:
    world: WorldConfig = WorldConfig(
        pi_a=0.85,
        tau_a=0.25,
        tau_b=0.75,
        eps_a=0.05,
        eps_b=0.05,
        n_annotators=120,
        samples_per_annotator=40,
    )
    train: TrainConfig = TrainConfig(steps=250, batch_size=128, learning_rate=0.2)
    seeds: tuple[int, ...] = (0, 1, 2)
    k_values: tuple[int, ...] = (1, 2, 3, 5, 10)
    output_dir: Path = Path("outputs/exp_d/")
