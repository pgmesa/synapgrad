import numpy as np
from synapgrad.tensor import Tensor
from synapgrad import nn
from synapgrad.nn.initializations import init_weights


class Neuron(nn.Module):
    
    def __init__(self, inputs:int, weight_init_method='he_normal') -> None:
        self.inputs = inputs
        
        # Randomly initialize weights and bias
        self.weight_init_method = weight_init_method
        weight_values = init_weights((1, inputs), weight_init_method).astype(np.float32)
        self.weight = Tensor(weight_values, requires_grad=True)
        self.bias = Tensor([0], requires_grad=True)
    
    def forward(self, x:Tensor) -> Tensor:
        assert x[0].matches_shape(self.weight[0]), f"Expected input size '{self.weight[0].shape}' but received '{x[0].shape}'"

        out = (x @ self.weight.transpose(0,-1)) + self.bias
    
        return out

    def parameters(self):
        return [self.weight, self.bias]

    def __repr__(self) -> str:
        return f"Neuron(weights={self.weight}, bias={self.bias})"