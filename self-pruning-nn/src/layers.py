import math
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore

class PrunableLinear(nn.Module):
    """
    A Linear layer that supports self-pruning via learnable gate scores.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(PrunableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Learnable parameters
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        self.gate_scores = nn.Parameter(torch.empty((out_features, in_features)))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initialize weight and bias similarly to nn.Linear (kaiming uniform).
        Initialize gate_scores to near zero (e.g. small normal distribution).
        """
        # Standard kaiming uniform initialization for weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        # Standard bias initialization as in nn.Linear
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
            
        # Initialize gate scores near zero
        # small gaussian noise near 0, meaning initial gates ~ 0.5
        nn.init.normal_(self.gate_scores, mean=0.0, std=0.01)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applying the gating mechanism to weights.
        """
        # Calculate gates via sigmoid -> values between 0 and 1
        gates = torch.sigmoid(self.gate_scores)
        
        # Element-wise multiplication to prune weights
        pruned_weights = self.weight * gates
        
        # Apply standard linear transformation using the pruned weights
        return F.linear(input, pruned_weights, self.bias)
