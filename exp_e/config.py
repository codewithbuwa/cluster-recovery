from dataclasses import dataclass
from pathlib import Path

from scr.config import TrainConfig, WorldConfig


@dataclass(frozen=True)
class ExperimentEConfig:
    world: WorldConfig = WorldConfig(pi_a=0.9, tau_a=0.25, tau_b=0.75, eps_a=0.05, eps_b=0.05)
    train: TrainConfig = TrainConfig(steps=500, eval_every=25, learning_rate=0.15, pair_noise=0.05)
    budget_seeds: tuple[int, ...] = (0, 1)
    alpha_seeds: tuple[int, ...] = range(50)
    pia_seeds: tuple[int, ...] = (0, 1)
    reference_seeds: tuple[int, ...] = (0, 1)
    effort: int = 256
    fixed_n_unary: int = 128
    fixed_n_pair: int = 64
    pair_fraction_values: tuple[float, ...] = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)
    alpha_values: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
    pi_a_values: tuple[float, ...] = (0.5, 0.7, 0.85, 0.95, 0.99)
    output_dir: Path = Path("outputs/exp_e/")
