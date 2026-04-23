import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional

class PrunableLinear(nn.Module):
    """
    A custom linear layer that learns to prune its own weights using learnable gates.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(PrunableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Standard weight and bias parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        # Gate scores initialized to 1.0 (fully open initially)
        self.gate_scores = nn.Parameter(torch.ones(out_features, in_features))
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initializes weights using Kaiming Uniform and bias using standard uniform distribution.
        """
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applying learnable gates to weights.
        """
        # Calculate gates using sigmoid (constrains values between 0 and 1)
        gates = torch.sigmoid(self.gate_scores)
        
        # Apply gates to weights (element-wise multiplication)
        pruned_weight = self.weight * gates
        
        # Perform linear operation with pruned weights
        return F.linear(x, pruned_weight, self.bias)

class AuraPruneNet(nn.Module):
    """
    4-layer Feed-Forward Network for CIFAR-10 with PrunableLinear layers.
    Includes Batch Normalization and ReLU activations for stability.
    """
    def __init__(self, input_dim: int = 3072, hidden_dims: List[int] = [1024, 512, 256], num_classes: int = 10) -> None:
        super(AuraPruneNet, self).__init__()
        
        self.flatten = nn.Flatten()
        
        # Layer 1
        self.fc1 = PrunableLinear(input_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        
        # Layer 2
        self.fc2 = PrunableLinear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        
        # Layer 3
        self.fc3 = PrunableLinear(hidden_dims[1], hidden_dims[2])
        self.bn3 = nn.BatchNorm1d(hidden_dims[2])
        
        # Layer 4 (Output)
        self.fc4 = PrunableLinear(hidden_dims[2], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes CIFAR-10 images through the prunable network.
        """
        x = self.flatten(x)
        
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        
        x = self.fc4(x)
        return x
