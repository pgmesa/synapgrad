import numpy as np
from .. import Tensor, nn
from .initializations import init_weights



class Neuron(nn.Module):
    
    def __init__(self, inputs:int, weight_init_method='he_normal') -> None:
        self.inputs = inputs
        
        # Randomly initialize weights and bias
        self.weight_init_method = weight_init_method
        weight_values = init_weights((1, inputs), weight_init_method).astype(np.float32)
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