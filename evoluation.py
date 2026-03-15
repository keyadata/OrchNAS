import random
from typing import Dict, List


def mutate_arch(arch: Dict, depth_choices: List[int], width_choices: List[int], kernel_choices: List[int]) -> Dict:
    new_arch = {
        "depth": arch["depth"],
        "width": arch["width"],
        "kernels": list(arch["kernels"]),
    }

    mutation_type = random.choice(["depth", "width", "kernel"])

    if mutation_type == "depth":
        new_depth = random.choice(depth_choices)
        new_arch["depth"] = new_depth
        if len(new_arch["kernels"]) < new_depth:
            new_arch["kernels"].extend([random.choice(kernel_choices) for _ in range(new_depth - len(new_arch["kernels"]))])
        else:
            new_arch["kernels"] = new_arch["kernels"][:new_depth]

    elif mutation_type == "width":
        new_arch["width"] = random.choice(width_choices)

    else:
        if len(new_arch["kernels"]) > 0:
            idx = random.randint(0, len(new_arch["kernels"]) - 1)
            new_arch["kernels"][idx] = random.choice(kernel_choices)

    return new_arch
