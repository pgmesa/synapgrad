
import math
import numpy as np
from deeplib.engine import Tensor
from deeplib.modules import Module


weight_initializers = ['glorot', 'glorot_norm', 'he']


class Neuron(Module):
    
    def __init__(self, inputs:int, weight_init_method='he') -> None:
        self.inputs = inputs
        # Randomly initialize weights and bias
        self.weights = Tensor(init_weights(inputs, 1, weight_init_method), requires_grad=True)
        self.weights.retain_grad()
        self.bias = Tensor(0, requires_grad=True)
        self.bias.retain_grad()
        assert len(self.weights) == inputs, f"{len(self.weights)} {inputs}"
    
    def forward(self, x:Tensor) -> Tensor:
        assert x[0].matches_shape(self.weights), f"Expected input size '{self.weights.shape}' but received '{x[0].shape}'"

        out = []
        for inp in x:
            o = (inp @ self.weights) + self.bias
            out.append(o.unsqueeze(0))
            
        out = Tensor.concat(out, dim=0).unsqueeze(1)
    
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