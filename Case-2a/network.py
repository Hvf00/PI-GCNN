import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


# Weight Initialization

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# Gated Network

class Atanh(nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh = nn.Tanh()
        self.alpha = nn.Parameter(torch.tensor(0.9))

    def forward(self, x):
        return self.tanh(self.alpha * x)


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, scale=1.0):
        super().__init__()
        self.B = nn.Parameter(
            scale * torch.randn(in_features, out_features),
            requires_grad=False
        )

    def forward(self, x):
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([x, torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Gate1(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.atanh = Atanh()

    def forward(self, x):
        return self.atanh(self.linear(x))


class Gate2(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.atanh = Atanh()

    def forward(self, x):
        return self.atanh(self.linear(x))


class GNetwork(nn.Module):
    def __init__(self, num_input=2, fourier_features=10, layers=[64, 64, 64], num_output=3, scale=1.0):
        super().__init__()
        self.fourier = FourierFeatures(num_input, fourier_features, scale)
        input_size = num_input + 2 * fourier_features
        self.input_layer = nn.Linear(input_size, layers[0])
        self.gate1 = Gate1(input_size, layers[0])
        self.gate2 = Gate2(input_size, layers[0])
        self.z_layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.z_layers.append(nn.Linear(layers[i], layers[i+1]))
        self.output_layer = nn.Linear(layers[-1], num_output)

    def forward(self, x):
        x = self.fourier(x)
        U = self.gate1(x)
        V = self.gate2(x)
        h = torch.tanh(self.input_layer(x))
        for z_layer in self.z_layers:
            Z = torch.tanh(z_layer(h))
            h = (1 - Z) * U + Z * V
        return self.output_layer(h)


# Conventional Network

class CNetwork(nn.Module):
    def __init__(self, num_input=2, layers=[64, 64, 64], num_output=3):
        super().__init__()
        self.input_layer = nn.Linear(num_input, layers[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.hidden_layers.append(nn.Linear(layers[i], layers[i+1]))
        self.output_layer = nn.Linear(layers[-1], num_output)

    def forward(self, x):
        x = torch.tanh(self.input_layer(x))
        for layer in self.hidden_layers:
            x = torch.tanh(layer(x))
        return self.output_layer(x)


# Fourier Feature Network

class FourierFeaturess(nn.Module):
    def __init__(self, in_features, out_features, scale=0.1):
        super().__init__()
        self.B = nn.Parameter(
            scale * torch.randn(in_features, out_features),
            requires_grad=False
        )

    def forward(self, x):
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([x, torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class FFNetwork(nn.Module):
    def __init__(self, num_input=2, layers=[64, 64, 64, 64], num_output=3):
        super().__init__()
        self.fourier = FourierFeaturess(num_input, 20)
        input_size = num_input + 2 * 20
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, layers[0]))
        for i in range(1, len(layers)):
            self.layers.append(nn.Linear(layers[i-1], layers[i]))
        self.output_layer = nn.Linear(layers[-1], num_output)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.fourier(x)
        for layer in self.layers:
            x = self.activation(layer(x))
        return self.output_layer(x)


# Graph Convolutional Network

class GraphConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.neighbor_linear = nn.Linear(in_dim, out_dim)
        self.self_linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        x = x.float()
        adj = adj.float()
        neighbor_agg = adj @ x
        return self.neighbor_linear(neighbor_agg) + self.self_linear(x)

class GNNetwork(nn.Module):
    def __init__(self, num_input, layers, num_output):
        super().__init__()
        self.layers = nn.ModuleList()
        in_dim = num_input
        for out_dim in layers:
            self.layers.append(GraphConvLayer(in_dim, out_dim))
            in_dim = out_dim
        self.output_layer = nn.Linear(in_dim, num_output)
        self.register_buffer('adj', None)

    def set_adj(self, adj_norm):
        self.adj = adj_norm.float()

    def forward(self, x):
        x = x.float()
        adj = self.adj.float()
    
        for layer in self.layers:
            x = torch.tanh(layer(x, adj))
        return self.output_layer(x)

#Gated Graph Convolutional Network

class GatedGraphLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.U_layer = GraphConvLayer(in_dim, out_dim)
        self.V_layer = GraphConvLayer(in_dim, out_dim)
        self.Z_layer = GraphConvLayer(in_dim, out_dim)
        self.atanh = Atanh()

    def forward(self, x, adj):
        U = self.atanh(self.U_layer(x, adj))
        V = self.atanh(self.V_layer(x, adj))
        Z = torch.tanh(self.Z_layer(x, adj))
        return (1 - Z) * U + Z * V

class FGatedGNN(nn.Module):
    def __init__(self, num_input, fourier_features, layers, num_output, scale=1.0):
        super().__init__()

        # Fourier embedding
        self.fourier = FourierFeatures(num_input, fourier_features, scale)
        input_size = num_input + 2 * fourier_features

        # Gated graph layers
        self.layers = nn.ModuleList()
        in_dim = input_size
        for out_dim in layers:
            self.layers.append(GatedGraphLayer(in_dim, out_dim))
            in_dim = out_dim

        # Output layer
        self.output_layer = nn.Linear(in_dim, num_output)

        # adjacency (set externally)
        self.register_buffer("adj", None)

    def set_adj(self, adj_norm):
        self.adj = adj_norm.float()

    def forward(self, x):
        x = x.float()
        adj = self.adj.float()

        # Fourier features
        x = self.fourier(x)

        # Gated Graph Layers
        for layer in self.layers:
            x = layer(x, adj)

        # Output
        return self.output_layer(x)