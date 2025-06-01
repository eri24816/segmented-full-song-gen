import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, dim: int, out_dim: int, num_hidden_layers: int, residual: bool = False):
        super().__init__()
        hidden = []
        self.input_layer = nn.Linear(in_dim, dim)
        for _ in range(num_hidden_layers):
            hidden.append(nn.GELU())
            hidden.append(nn.Linear(dim, dim))
        hidden.append(nn.GELU())
        self.hidden_layers = nn.Sequential(*hidden)
        self.output_layer = nn.Linear(dim, out_dim)
        self.residual = residual
        if residual:
            assert out_dim == dim

    def forward(self, x: torch.Tensor):
        x = skip = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        if self.residual:
            x = x + skip
        return x
