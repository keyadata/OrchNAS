from typing import Dict
from energy.profiler import EnergyProfiler


def satisfies_constraints(arch: Dict, constraints: Dict) -> bool:
    energy = EnergyProfiler.estimate_energy(arch, {"alpha": constraints.get("alpha", 1.0)})
    memory = EnergyProfiler.estimate_memory(arch)
    compute = EnergyProfiler.estimate_compute(arch)

    return (
        energy <= constraints["energy_budget"] and
        memory <= constraints["memory_budget"] and
        compute <= constraints["compute_budget"]
    )


def energy_aware_score(acc: float, arch: Dict, constraints: Dict, lambda_energy: float) -> float:
    energy = EnergyProfiler.estimate_energy(arch, {"alpha": constraints.get("alpha", 1.0)})
    return acc - lambda_energy * energy
