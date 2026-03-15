from typing import Dict, List
import torch
import torch.nn as nn
from models.ops import conv_block


class SuperNet(nn.Module):
    """
    One-shot supernet that can instantiate different backbone paths.
    """

    def __init__(self, input_channels: int, num_classes: int, max_depth: int, max_width: int, kernel_choices: List[int]):
        super().__init__()
        self.max_depth = max_depth
        self.max_width = max_width
        self.kernel_choices = kernel_choices

        self.stem = conv_block(input_channels, max_width, 3)

        self.layers = nn.ModuleList()
        for _ in range(max_depth):
            op_dict = nn.ModuleDict({
                str(k): conv_block(max_width, max_width, k) for k in kernel_choices
            })
            self.layers.append(op_dict)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(max_width, num_classes)

    def forward(self, x: torch.Tensor, arch: Dict) -> torch.Tensor:
        """
        arch example:
        {
            "depth": 3,
            "width": 32,
            "kernels": [3, 5, 3]
        }
        """
        depth = arch["depth"]
        kernels = arch["kernels"]

        x = self.stem(x)

        for i in range(depth):
            k = str(kernels[i])
            x = self.layers[i][k](x)

        # simple channel truncation to emulate width choice
        width = arch["width"]
        x = x[:, :width, :, :]

        pooled = self.pool(x).view(x.size(0), -1)

        if pooled.size(1) < self.classifier.in_features:
            pad = self.classifier.in_features - pooled.size(1)
            pooled = torch.cat([pooled, torch.zeros(pooled.size(0), pad, device=pooled.device)], dim=1)

        return self.classifier(pooled)
