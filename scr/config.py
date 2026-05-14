from dataclasses import dataclass


@dataclass(frozen=True)
class WorldConfig:
    n_prompts: int = 8
    n_responses: int = 16
    n_clusters: int = 2
    pi_a: float = 0.9
    tau_a: float = 0.25
    tau_b: float = 0.75
    eps_a: float = 0.05
    eps_b: float = 0.05
    n_annotators: int = 120
    samples_per_annotator: int = 40


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 400
    batch_size: int = 128
    learning_rate: float = 0.15
    beta: float = 1.0
    alpha: float = 0.0
    lambda_desirable: float = 1.0
    lambda_undesirable: float = 1.0
    ema_rate: float = 0.1
    eval_every: int = 20
