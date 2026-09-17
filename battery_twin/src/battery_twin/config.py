from dataclasses import dataclass


@dataclass(frozen=True)
class TwinConfig:
    """Configuration shared by data, training, and adaptation code."""

    eol_soh: float = 70.0
    nominal_capacity_ah: float = 2.0
    history_cycles: int = 12
    max_points_per_cycle: int = 256
    hidden_size: int = 128
    num_heads: int = 4
    num_transformer_layers: int = 2
    dropout: float = 0.10
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    physics_weight: float = 1.0
    consistency_weight: float = 1.0
    rul_scale: float = 250.0
    batch_size: int = 32
    epochs: int = 40
    seed: int = 7
    replay_size: int = 512
    drift_warmup: int = 8
    drift_quantile: float = 0.90
    drift_cooldown: int = 12
    adaptation_steps: int = 1
    adaptation_learning_rate: float = 1e-4
