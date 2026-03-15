from typing import Dict


def extract_subnet_config(backbone_arch: Dict, client_constraints: Dict) -> Dict:
    """
    Progressive greedy pruning placeholder.
    Produces a client-specific subnet from shared backbone architecture.
    """
    depth = backbone_arch["depth"]
    width = backbone_arch["width"]
    kernels = list(backbone_arch["kernels"])

    if client_constraints["energy_budget"] < 30:
        depth = max(1, depth - 1)
        width = max(16, width // 2)
        kernels = kernels[:depth]
    elif client_constraints["memory_budget"] < 100:
        width = max(16, width // 2)
        kernels = kernels[:depth]
    else:
        kernels = kernels[:depth]

    return {
        "depth": depth,
        "width": width,
        "kernels": kernels,
    }
