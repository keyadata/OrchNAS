from dataclasses import dataclass
from typing import List


@dataclass
class OrchNASConfig:
    num_clients: int = 5
    rounds: int = 10
    clients_per_round: int = 3
    local_epochs: int = 1
    batch_size: int = 32
    lr: float = 1e-3

    input_channels: int = 3
    num_classes: int = 10

    # Search space
    depth_choices: List[int] = None
    width_choices: List[int] = None
    kernel_choices: List[int] = None

    # Energy-aware settings
    lambda_energy: float = 0.1
    energy_budget: float = 50.0
    memory_budget: float = 200.0
    compute_budget: float = 100.0

    def __post_init__(self):
        if self.depth_choices is None:
            self.depth_choices = [2, 3, 4]
        if self.width_choices is None:
            self.width_choices = [16, 32, 64]
        if self.kernel_choices is None:
            self.kernel_choices = [3, 5]
