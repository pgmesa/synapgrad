
import math
import numpy as np
from deeplib.engine import Tensor, Module


weight_initializers = ['glorot', 'glorot_norm', 'he']


class Neuron(Module):
    
    def __init__(self, inputs:int, outputs:int, weight_init_method='he') -> None:
        self.inputs = inputs
        self.outputs = outputs
        # Randomly initialize weights and bias
        self.weights = Tensor(init_weights(inputs, outputs, weight_init_method))
        assert len(self.weights) == inputs, f"{len(self.weights)} {inputs}"
        self.bias = Tensor(0)
    
    def __call__(self, x:Tensor) -> float:
        assert x.shape[1] == len(self.weights), f"Expected input size '{len(self.weights)}' but received '{x.shape[1]}'"
        self.x = x
        self.activation = (x @ self.weights) + self.bias
        return self.activation


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