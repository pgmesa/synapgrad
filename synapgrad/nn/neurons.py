import math
import numpy as np
from .. import Tensor, nn


weight_initializers = ['glorot', 'glorot_norm', 'he']


class Neuron(nn.Module):
    
    def __init__(self, inputs:int, weight_init_method='he') -> None:
        self.inputs = inputs
        # Randomly initialize weights and bias
        weight_values = np.expand_dims(init_weights(inputs, 1, weight_init_method), 0)
        self.weights = Tensor(weight_values, requires_grad=True)
        self.bias = Tensor([0], requires_grad=True)
    
    def forward(self, x:Tensor) -> Tensor:
        assert x[0].matches_shape(self.weights[0]), f"Expected input size '{self.weights[0].shape}' but received '{x[0].shape}'"

        out = (x @ self.weights.transpose(0,-1)) + self.bias
    
        return out

    def parameters(self):
        return [self.weights, self.bias]

    def __repr__(self) -> str:
        return f"Neuron(weights={self.weights}, bias={self.bias})"


def init_weights(inputs, outputs, method) -> np.ndarray:
    if method not in weight_initializers:
        raise ValueError(f"'{method}' is not a valid weight initializer")
    
    if method == 'glorot': return glorot(inputs)
    elif method == 'glorot_norm': return glorot_norm(inputs, outputs)
    elif method == 'he': return he(inputs)
    

def he(nodes) -> np.ndarray:
    std = math.sqrt(2.0 / nodes)
    # generate random numbers
    numbers = np.random.randn(nodes)
    # scale to the desired range
    scaled = numbers * std
    
    return scaled
    
        
def glorot(nodes) -> np.ndarray:
    lower = -(1.0 / math.sqrt(nodes))
    upper = (1.0 / math.sqrt(nodes))
    numbers = np.random.rand(nodes)
    scaled = lower + numbers * (upper - lower)
    
    return scaled


def glorot_norm(nodes_in, nodes_out) -> np.ndarray:
    lower = -(math.sqrt(6.0) / math.sqrt(nodes_in + nodes_out))
    upper = (math.sqrt(6.0) / math.sqrt(nodes_in + nodes_out))
    # generate random numbers
    numbers = np.random.rand(nodes_in)
    # scale to the desired range
    scaled = lower + numbers * (upper - lower)
    
    return scaled